import os
import cv2
import argparse
import math
import numpy as np
import time
from collections import deque
from absl import logging
from multiprocessing import Pool, cpu_count

# Set environment variable for minimal QT platform
os.environ["QT_QPA_PLATFORM"] = "minimal"

# Constants
BALL_TO_PERSON_RATIO = 0.08  # Minimum ball size relative to person height
DEFAULT_BALL_SIZE = (20, 20)   # Default ball size when no previous detection
HISTORY_LENGTH = 3             # For ball tracking velocity estimation
DEFAULT_MAX_VELOCITY_FACTOR = 1.0
DEFAULT_RECOVERY_FRAMES = 3

# Global variables for worker processes.
global_model = None
global_pose_detector = None
global_manual_rotation = 0  # Manual rotation angle (in degrees, clockwise)
global_criteria = 1         # Criteria for release detection
global_show_overlays = False  # Whether to show overlays
global_max_velocity = DEFAULT_MAX_VELOCITY_FACTOR
global_recovery_frames = DEFAULT_RECOVERY_FRAMES

# ---------------------------
# Helper Functions
# ---------------------------
def setup_video_io(input_path, manual_rotation=0):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Error: Could not open video.")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if manual_rotation in [90, 270]:
        width, height = height, width
    return cap, manual_rotation, width, height, fps

def process_frame(frame, rotation):
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame

def calculate_person_height(pose_landmarks, frame_shape):
    if not pose_landmarks:
        return None
    landmarks = pose_landmarks.landmark
    min_y = min(lm.y for lm in landmarks)
    max_y = max(lm.y for lm in landmarks)
    return (max_y - min_y) * frame_shape[0]

def calculate_torso_data(pose_landmarks, frame_shape):
    if not pose_landmarks:
        return None, None, None, None, None, None, None
    try:
        landmarks = pose_landmarks.landmark
        left_shoulder = (landmarks[11].x * frame_shape[1], landmarks[11].y * frame_shape[0])
        right_shoulder = (landmarks[12].x * frame_shape[1], landmarks[12].y * frame_shape[0])
        left_hip = (landmarks[23].x * frame_shape[1], landmarks[23].y * frame_shape[0])
        right_hip = (landmarks[24].x * frame_shape[1], landmarks[24].y * frame_shape[0])
        shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2,
                           (left_shoulder[1] + right_shoulder[1]) / 2)
        hip_center = ((left_hip[0] + right_hip[0]) / 2,
                      (left_hip[1] + right_hip[1]) / 2)
        torso_height = math.sqrt((shoulder_center[0] - hip_center[0])**2 +
                                 (shoulder_center[1] - hip_center[1])**2)
        return torso_height, shoulder_center, hip_center, left_shoulder, right_shoulder, left_hip, right_hip
    except Exception:
        return None, None, None, None, None, None, None

def get_right_wrist_point(pose_landmarks, frame_shape):
    if not pose_landmarks:
        return None
    right_wrist = pose_landmarks.landmark[16]
    return (int(right_wrist.x * frame_shape[1]), int(right_wrist.y * frame_shape[0]))

def get_right_elbow_point(pose_landmarks, frame_shape):
    if not pose_landmarks:
        return None
    right_elbow = pose_landmarks.landmark[14]
    return (int(right_elbow.x * frame_shape[1]), int(right_elbow.y * frame_shape[0]))

def calculate_distance_fraction(point1, point2, torso_height):
    if None in (point1, point2, torso_height) or torso_height <= 0:
        return None
    distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    return distance / torso_height

def calculate_torso_arm_angle(shoulder, elbow, wrist, hip_center):
    if None in (shoulder, elbow, wrist, hip_center):
        return None
    torso_vector = (hip_center[0] - shoulder[0], hip_center[1] - shoulder[1])
    arm_vector = (elbow[0] - shoulder[0], elbow[1] - shoulder[1])
    dot_product = torso_vector[0] * arm_vector[0] + torso_vector[1] * arm_vector[1]
    torso_magnitude = math.sqrt(torso_vector[0]**2 + torso_vector[1]**2)
    arm_magnitude = math.sqrt(arm_vector[0]**2 + arm_vector[1]**2)
    if torso_magnitude == 0 or arm_magnitude == 0:
        return None
    cosine_angle = dot_product / (torso_magnitude * arm_magnitude)
    cosine_angle = max(-1.0, min(1.0, cosine_angle))
    return math.degrees(math.acos(cosine_angle))

def draw_ball(frame, bbox, center, is_prediction=False):
    color = (0, 165, 255) if is_prediction else (0, 255, 0)
    label = "Predicted Ball" if is_prediction else "Ball"
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.circle(frame, center, radius=5, color=color, thickness=-1)

def draw_distance_indicator(frame, distance_fraction, width):
    text = f"Dist: {distance_fraction:.2f} torso" if distance_fraction is not None else "Dist: N/A"
    cv2.putText(frame, text, (width - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

def annotate_release(frame):
    cv2.putText(frame, "RELEASE", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

def annotate_non_release(frame, reasons, ball_tracker=None, torso_height=None):
    y_offset = 30
    for reason in reasons:
        cv2.putText(frame, reason, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y_offset += 25
    if ball_tracker and ball_tracker.velocity_history:
        avg_velocity = ball_tracker.get_average_velocity()
        max_allowed = ball_tracker.max_velocity_factor * (torso_height or 100)
        velocity_info = f"Avg Vel: {avg_velocity:.1f} (max: {max_allowed:.1f})"
        cv2.putText(frame, velocity_info, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 25
        if ball_tracker.is_outlier:
            outlier_info = f"OUTLIER: {ball_tracker.consecutive_stable_frames}/{ball_tracker.recovery_threshold} stable frames"
            cv2.putText(frame, outlier_info, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def check_release_candidate(window, fps, criteria=1):
    candidate = window[3]
    reasons = []
    if candidate["shoulder"] is None or candidate["hip"] is None:
        reasons.append("Missing shoulder/hip data")
    else:
        threshold_y = candidate["hip"][1] - (2 / 3) * (candidate["hip"][1] - candidate["shoulder"][1])
        if candidate["ball_center"] is None:
            reasons.append("Missing ball center")
        elif candidate["ball_center"][1] > threshold_y:
            reasons.append("Ball center below threshold")
        if candidate["wrist"] is None:
            reasons.append("Missing wrist")
        elif candidate["wrist"][1] > threshold_y:
            reasons.append("Wrist below threshold")
    ball_vel_failures = 0
    if candidate["torso_height"] is None or candidate["torso_height"] <= 0:
        reasons.append("Invalid torso height")
    else:
        allowed_ball_delta = (3 * candidate["torso_height"]) / fps
        for i in range(1, 7):
            prev = window[i - 1]["ball_rel_y"]
            curr = window[i]["ball_rel_y"]
            if prev is None or curr is None:
                ball_vel_failures += 1
            else:
                delta = curr - prev
                if delta > allowed_ball_delta:
                    ball_vel_failures += 1
        if ball_vel_failures > 2:
            reasons.append(f"Ball velocity failure count = {ball_vel_failures} (>2)")
    if criteria == 1:
        if candidate["ball_hand_distance"] is None:
            reasons.append("Candidate missing ball-hand distance")
        elif candidate["ball_hand_distance"] < 0.5:
            reasons.append("Candidate ball-hand distance < 0.5")
        ball_hand_failures = 0
        for i in range(0, 3):
            if window[i]["ball_hand_distance"] is None or window[i]["ball_hand_distance"] >= 0.5:
                ball_hand_failures += 1
        for i in range(4, 7):
            if window[i]["ball_hand_distance"] is None or window[i]["ball_hand_distance"] <= 0.5:
                ball_hand_failures += 1
        if ball_hand_failures > 2:
            reasons.append(f"Ball-hand distance adjacent failure count = {ball_hand_failures} (>2)")
        elbow_failures = 0
        if candidate["torso_height"] is None or candidate["torso_height"] <= 0:
            reasons.append("Invalid torso height for elbow check")
        else:
            allowed_elbow_delta = (3 * candidate["torso_height"]) / fps
            for i in range(1, 4):
                prev = window[i - 1]["elbow_rel_y"]
                curr = window[i]["elbow_rel_y"]
                if prev is None or curr is None:
                    elbow_failures += 1
                else:
                    delta = curr - prev
                    if delta > allowed_elbow_delta:
                        elbow_failures += 1
            if elbow_failures > 2:
                reasons.append(f"Elbow motion failure count = {elbow_failures} (>2)")
    elif criteria == 2:
        angle = candidate['torso_arm_angle']
        if angle is None:
            reasons.append("Unable to calculate torso-arm angle")
        elif angle < 90:
            reasons.append(f"Torso-arm angle ({angle:.1f}) < 90 degrees")
        for i in range(0, 3):
            prev_angle = window[i]['torso_arm_angle']
            if prev_angle is not None and prev_angle >= 90:
                reasons.append(f"Frame {i} already has angle >= 90 degrees")
                break
    elif criteria == 3:
        if candidate['ball_center'] is None or candidate['shoulder'] is None or candidate['torso_height'] is None:
            reasons.append("Missing data for ball height check")
        else:
            ball_shoulder_height = candidate['shoulder'][1] - candidate['ball_center'][1]
            required_height = 2/3 * candidate['torso_height']
            if ball_shoulder_height < required_height:
                reasons.append(f"Ball not high enough above shoulder ({ball_shoulder_height:.1f} < {required_height:.1f})")
            for i in range(0, 3):
                if (window[i]['ball_center'] is not None and 
                    window[i]['shoulder'] is not None and 
                    window[i]['torso_height'] is not None):
                    prev_height = window[i]['shoulder'][1] - window[i]['ball_center'][1]
                    if prev_height >= required_height:
                        reasons.append(f"Frame {i} already has sufficient ball height")
                        break
        elbow_failures = 0
        if candidate["torso_height"] is None or candidate["torso_height"] <= 0:
            reasons.append("Invalid torso height for elbow check")
        else:
            allowed_elbow_delta = (3 * candidate["torso_height"]) / fps
            for i in range(1, 4):
                prev = window[i - 1]["elbow_rel_y"]
                curr = window[i]["elbow_rel_y"]
                if prev is None or curr is None:
                    elbow_failures += 1
                else:
                    delta = curr - prev
                    if delta > allowed_elbow_delta:
                        elbow_failures += 1
            if elbow_failures > 2:
                reasons.append(f"Elbow motion failure count = {elbow_failures} (>2)")
    valid = len(reasons) == 0
    return valid, reasons

# ---------------------------
# Ball Tracker Class
# ---------------------------
class BallTracker:
    def __init__(self, history_length=HISTORY_LENGTH, conf_threshold=0.1):
        self.history_length = history_length
        self.conf_threshold = conf_threshold
        self.position_history = []
        self.last_ball_bbox = None
        self.predicted_center = None
        self.velocity_history = []
        self.consecutive_stable_frames = 0
        self.is_outlier = False
        self.max_velocity_factor = 1.0
        self.recovery_threshold = 3
        self.velocity_window = None

    def update_with_detection(self, bbox, torso_height=None, fps=None):
        center = self.calculate_center(bbox)
        if fps is not None and self.velocity_window is None:
            self.velocity_window = max(2, int(fps/15))
        if self.position_history:
            velocity = self.calculate_velocity(self.position_history[-1], center)
            self.velocity_history.append(velocity)
            if len(self.velocity_history) > self.history_length:
                self.velocity_history.pop(0)
            is_current_outlier = self.is_velocity_outlier(velocity, torso_height)
            if is_current_outlier:
                self.is_outlier = True
                self.consecutive_stable_frames = 0
                return self.predicted_center
            elif self.is_outlier:
                last_valid_center = self.position_history[-1] if self.position_history else None
                if last_valid_center and self.is_position_stable(last_valid_center, center):
                    self.consecutive_stable_frames += 1
                else:
                    self.consecutive_stable_frames = 0
                if self.consecutive_stable_frames >= self.recovery_threshold:
                    self.is_outlier = False
                else:
                    return self.predicted_center
        self.last_ball_bbox = bbox
        self.predicted_center = center
        self.position_history.append(center)
        if len(self.position_history) > self.history_length:
            self.position_history.pop(0)
        return center

    def predict_position(self):
        if len(self.position_history) < 2:
            return None, None
        num_positions = len(self.position_history)
        dx = (self.position_history[-1][0] - self.position_history[0][0]) / (num_positions - 1)
        dy = (self.position_history[-1][1] - self.position_history[0][1]) / (num_positions - 1)
        if self.predicted_center is None:
            self.predicted_center = self.position_history[-1]
        else:
            self.predicted_center = (int(self.predicted_center[0] + dx), int(self.predicted_center[1] + dy))
        if self.last_ball_bbox is not None:
            width_box = self.last_ball_bbox[2] - self.last_ball_bbox[0]
            height_box = self.last_ball_bbox[3] - self.last_ball_bbox[1]
        else:
            width_box, height_box = DEFAULT_BALL_SIZE
        predicted_bbox = [
            self.predicted_center[0] - width_box // 2,
            self.predicted_center[1] - height_box // 2,
            self.predicted_center[0] + width_box // 2,
            self.predicted_center[1] + height_box // 2,
        ]
        self.last_ball_bbox = predicted_bbox
        return self.predicted_center, predicted_bbox

    @staticmethod
    def calculate_center(bbox):
        return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
    
    @staticmethod
    def calculate_velocity(pos1, pos2):
        return math.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)
    
    def is_velocity_outlier(self, velocity, torso_height):
        if self.max_velocity_factor <= 0:
            return False
        if torso_height is None or torso_height <= 0:
            return velocity > 100
        return velocity > self.max_velocity_factor * torso_height
    
    def is_position_stable(self, pos1, pos2, threshold_factor=0.1):
        distance = self.calculate_velocity(pos1, pos2)
        if self.last_ball_bbox is not None:
            avg_ball_size = ((self.last_ball_bbox[2] - self.last_ball_bbox[0]) +
                             (self.last_ball_bbox[3] - self.last_ball_bbox[1])) / 2
            threshold = avg_ball_size * threshold_factor
        else:
            threshold = 5
        return distance <= threshold
    
    def get_average_velocity(self, window_size=None):
        if not self.velocity_history:
            return 0
        if window_size is None:
            window_size = self.velocity_window if self.velocity_window else len(self.velocity_history)
        window_size = min(window_size, len(self.velocity_history))
        recent_velocities = self.velocity_history[-window_size:]
        return sum(recent_velocities) / len(recent_velocities)
    
    def draw_velocity_graph(self, frame, torso_height=None):
        if not self.velocity_history:
            return
        h, w = frame.shape[:2]
        graph_height = 100
        graph_width = 200
        graph_x = w - graph_width - 20
        graph_y = h - graph_height - 20
        cv2.rectangle(frame, (graph_x, graph_y), 
                     (graph_x + graph_width, graph_y + graph_height), 
                     (0, 0, 0), -1)
        cv2.rectangle(frame, (graph_x, graph_y), 
                     (graph_x + graph_width, graph_y + graph_height), 
                     (255, 255, 255), 1)
        max_threshold = self.max_velocity_factor * (torso_height or 100)
        max_velocity = max(max(self.velocity_history), max_threshold) * 1.2
        threshold_y = graph_y + graph_height - int((max_threshold / max_velocity) * graph_height)
        cv2.line(frame, (graph_x, threshold_y), (graph_x + graph_width, threshold_y), 
                (0, 0, 255), 1, cv2.LINE_AA)
        num_points = len(self.velocity_history)
        point_width = graph_width / (num_points - 1) if num_points > 1 else graph_width
        points = []
        for i, velocity in enumerate(self.velocity_history):
            x = graph_x + int(i * point_width)
            y = graph_y + graph_height - int((velocity / max_velocity) * graph_height)
            points.append((x, y))
        for i in range(1, len(points)):
            color = (0, 0, 255) if self.velocity_history[i] > max_threshold else (0, 255, 0)
            cv2.line(frame, points[i-1], points[i], color, 2, cv2.LINE_AA)
        for i, point in enumerate(points):
            color = (0, 0, 255) if self.velocity_history[i] > max_threshold else (0, 255, 0)
            cv2.circle(frame, point, 3, color, -1)
        cv2.putText(frame, "Velocity", (graph_x, graph_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Max: {max_threshold:.1f}", (graph_x, graph_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(frame, f"Cur: {self.velocity_history[-1]:.1f}", 
                   (graph_x, graph_y + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                   (0, 0, 255) if self.velocity_history[-1] > max_threshold else (0, 255, 0), 1)

# ---------------------------
# Video Processing Function
# ---------------------------
def process_video(video_path, yolo_model, manual_rotation, criteria=1, show_overlays=False):
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    cap, rotation, width, height, fps = setup_video_io(video_path, manual_rotation)
    ball_tracker = BallTracker(history_length=HISTORY_LENGTH)
    ball_tracker.max_velocity_factor = global_max_velocity
    ball_tracker.recovery_threshold = global_recovery_frames
    ball_tracker.velocity_window = max(2, int(fps/15))
    
    # Use the global MediaPipe pose detector initialized in init_worker
    pose_detector = global_pose_detector

    frame_buffer = deque()
    frames = []
    release_index = None
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(frame, rotation)
        clean_frame = frame.copy()

        # Convert to RGB for MediaPipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose_detector.process(frame_rgb)

        person_height_pixels = calculate_person_height(pose_results.pose_landmarks, frame.shape)
        torso_data = calculate_torso_data(pose_results.pose_landmarks, frame.shape)
        torso_height = torso_data[0] if torso_data else None

        if show_overlays and pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
            )

        # Run YOLO detection for ball
        detected_ball = False
        yolo_results = yolo_model(frame, conf=0.3, verbose=False)[0]
        if yolo_results.boxes is not None:
            for box in yolo_results.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                class_name = yolo_model.names[cls]
                if class_name == "sports ball":
                    ball_width = xyxy[2] - xyxy[0]
                    ball_height = xyxy[3] - xyxy[1]
                    ball_diameter = max(ball_width, ball_height)
                    if person_height_pixels is not None:
                        if ball_diameter < BALL_TO_PERSON_RATIO * person_height_pixels:
                            continue
                    detected_ball = True
                    detected_center = ball_tracker.update_with_detection(xyxy, torso_height, fps)
                    if show_overlays:
                        draw_ball(frame, xyxy, detected_center)
                        cv2.putText(frame, f"{conf:.2f}", (xyxy[0], xyxy[1] - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        if ball_tracker.is_outlier:
                            cv2.putText(frame, f"OUTLIER ({ball_tracker.consecutive_stable_frames}/{ball_tracker.recovery_threshold})",
                                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    break
        if not detected_ball:
            predicted_center, predicted_bbox = ball_tracker.predict_position()
            if predicted_center is not None and show_overlays:
                draw_ball(frame, predicted_bbox, predicted_center, is_prediction=True)

        wrist_point = get_right_wrist_point(pose_results.pose_landmarks, frame.shape)
        elbow_point = get_right_elbow_point(pose_results.pose_landmarks, frame.shape)
        torso_data = calculate_torso_data(pose_results.pose_landmarks, frame.shape)
        torso_height = torso_data[0] if torso_data else None
        shoulder_center = torso_data[1] if torso_data else None
        hip_center = torso_data[2] if torso_data else None
        right_shoulder = torso_data[4] if torso_data else None
        ball_center = ball_tracker.predicted_center

        ball_hand_distance = None
        if ball_center and wrist_point and torso_height:
            ball_hand_distance = calculate_distance_fraction(ball_center, wrist_point, torso_height)
            if show_overlays:
                draw_distance_indicator(frame, ball_hand_distance, width)

        ball_rel_y = ball_center[1] - shoulder_center[1] if (ball_center and shoulder_center) else None
        elbow_rel_y = elbow_point[1] - shoulder_center[1] if (elbow_point and shoulder_center) else None

        torso_arm_angle = None
        if right_shoulder and elbow_point and wrist_point and hip_center:
            torso_arm_angle = calculate_torso_arm_angle(right_shoulder, elbow_point, wrist_point, hip_center)

        current_frame = frame if show_overlays else clean_frame
        frames.append(current_frame.copy())

        frame_data = {
            "frame_index": frame_index,
            "frame": current_frame,
            "ball_center": ball_center,
            "wrist": wrist_point,
            "elbow": elbow_point,
            "shoulder": shoulder_center,
            "hip": hip_center,
            "torso_height": torso_height,
            "ball_hand_distance": ball_hand_distance,
            "ball_rel_y": ball_rel_y,
            "elbow_rel_y": elbow_rel_y,
            "torso_arm_angle": torso_arm_angle,
            "release_detected": False,
        }
        frame_buffer.append(frame_data)
        frame_index += 1

        if len(frame_buffer) == 7:
            window = list(frame_buffer)
            valid, reasons = check_release_candidate(window, fps, criteria)
            if valid and release_index is None:
                window[3]["release_detected"] = True
                release_index = window[3]["frame_index"]
                if show_overlays:
                    annotate_release(window[3]["frame"])
                    frames[window[3]["frame_index"]] = window[3]["frame"].copy()
            elif show_overlays:
                annotate_non_release(window[3]["frame"], reasons, ball_tracker, torso_height)
                frames[window[3]["frame_index"]] = window[3]["frame"].copy()
            frame_buffer.popleft()

    cap.release()
    return {
        "frames": frames,
        "release_index": release_index,
        "fps": fps,
        "width": width,
        "height": height,
        "ball_tracker": ball_tracker
    }

def process_video_wrapper(video_path):
    return process_video(video_path, global_model, global_manual_rotation, global_criteria, global_show_overlays)

# ---------------------------
# Worker Initializer
# ---------------------------
def init_worker(manual_rotation, criteria=1, show_overlays=False):
    global global_model, global_pose_detector, global_manual_rotation, global_criteria, global_show_overlays
    global_manual_rotation = manual_rotation
    global_criteria = criteria
    global_show_overlays = show_overlays

    from ultralytics import YOLO
    global_model = YOLO("yolo11x.pt")
    
    import mediapipe as mp
    global_pose_detector = mp.solutions.pose.Pose(
         static_image_mode=False,
         model_complexity=1,
         enable_segmentation=False,
         min_detection_confidence=0.5,
         min_tracking_confidence=0.5
    )
    logging.set_verbosity(logging.ERROR)

# ---------------------------
# Main Composite Video Creation
# ---------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Composite synchronized basketball shots using MediaPipe for pose detection"
    )
    parser.add_argument("input_dir", type=str, help="Folder containing input videos")
    parser.add_argument("output", type=str, help="Output composite video file")
    parser.add_argument("--history", type=int, default=HISTORY_LENGTH, help="History length for ball tracking")
    parser.add_argument("--num-processes", type=int, default=max(1, cpu_count() // 2),
                        help="Number of processes to use")
    parser.add_argument("--rotation", type=int, default=0, choices=[0, 90, 180, 270],
                        help="Manual rotation angle (in degrees, clockwise)")
    parser.add_argument("--criteria", type=int, default=1, choices=[1, 2, 3],
                        help="Set of criteria for release detection: 1=original, 2=angle-based, 3=ball-height")
    parser.add_argument("--show-overlays", action="store_true", help="Show overlays on each video")
    parser.add_argument("--max-velocity", type=float, default=DEFAULT_MAX_VELOCITY_FACTOR,
                        help="Maximum allowed ball velocity as a factor of torso height per frame. Set to 0 to disable.")
    parser.add_argument("--recovery-frames", type=int, default=DEFAULT_RECOVERY_FRAMES,
                        help="Number of consecutive stable frames needed to recover from an outlier detection.")
    args = parser.parse_args()

    total_start_time = time.time()
    
    logging.set_verbosity(logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Timing: video file discovery
    file_discovery_start = time.time()
    video_extensions = (".mp4", ".avi", ".mov", ".mkv")
    video_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
                   if f.lower().endswith(video_extensions)]
    if not video_files:
        print("No video files found in the specified folder.")
        return
    file_discovery_end = time.time()
    print(f"File discovery time: {file_discovery_end - file_discovery_start:.2f} seconds. Found {len(video_files)} videos.")

    print(f"Starting analysis stage for release detection on {len(video_files)} videos...")
    pool_size = args.num_processes
    print(f"Using {pool_size} processes for video processing.")

    global global_max_velocity, global_recovery_frames
    global_max_velocity = args.max_velocity
    global_recovery_frames = args.recovery_frames

    # Timing: analysis stage (multiprocessing)
    analysis_start_time = time.time()
    import multiprocessing
    with Pool(pool_size, initializer=init_worker, initargs=(args.rotation, args.criteria, args.show_overlays)) as pool:
        results = pool.map(process_video_wrapper, video_files)
    analysis_end_time = time.time()
    analysis_time = analysis_end_time - analysis_start_time
    print(f"Analysis stage time: {analysis_time:.2f} seconds (average {analysis_time/len(video_files):.2f} seconds per video)")

    # Timing: filtering stage
    filtering_start_time = time.time()
    video_data = []
    for path, data in zip(video_files, results):
        if data["release_index"] is None:
            print(f"Warning: No release detected in {path}. Skipping this video.")
        else:
            video_data.append(data)

    if not video_data:
        print("No videos with valid release detection. Exiting.")
        return
    filtering_end_time = time.time()
    print(f"Filtering stage time: {filtering_end_time - filtering_start_time:.2f} seconds")

    # Timing: calculation stage
    calculation_start_time = time.time()
    pre_frames_list = [data["release_index"] for data in video_data]
    post_frames_list = [len(data["frames"]) - data["release_index"] - 1 for data in video_data]
    global_pre = max(pre_frames_list)
    global_post = max(max(post_frames_list), 0)
    total_composite_frames = global_pre + global_post + 1
    print(f"Composite timeline: {global_pre} frames before and {global_post} frames after release, total {total_composite_frames} frames.")

    num_videos = len(video_data)
    composite_width = 3840   # Fixed 4K width
    composite_height = 2160  # Fixed 4K height

    # Arrange videos in two rows (top and bottom)
    row_height = composite_height // 2
    top_row_count = (num_videos + 1) // 2
    bottom_row_count = num_videos - top_row_count
    top_row_videos = video_data[:top_row_count]
    bottom_row_videos = video_data[top_row_count:]

    # Precompute resized width for each video
    for data in video_data:
        if len(data["frames"]) == 0:
            print(f"Warning: Video with release_index {data['release_index']} has no frames. Skipping.")
            continue
        frame0 = data["frames"][0]
        orig_h, orig_w = frame0.shape[:2]
        scale = row_height / orig_h
        data["resized_width"] = int(orig_w * scale)
    calculation_end_time = time.time()
    print(f"Calculation stage time: {calculation_end_time - calculation_start_time:.2f} seconds")

    # Timing: composite video creation
    composite_start_time = time.time()
    out_fps = video_data[0]["fps"]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(args.output, fourcc, out_fps, (composite_width, composite_height))

    for t in range(-global_pre, global_post + 1):
        top_row_images = []
        for data in top_row_videos:
            target_idx = data["release_index"] + t
            if 0 <= target_idx < len(data["frames"]):
                frame = data["frames"][target_idx]
                resized = cv2.resize(frame, (data["resized_width"], row_height))
            else:
                resized = np.zeros((row_height, data["resized_width"], 3), dtype=np.uint8)
            top_row_images.append(resized)
        if top_row_images:
            top_row_concat = np.concatenate(top_row_images, axis=1)
        else:
            top_row_concat = np.zeros((row_height, composite_width, 3), dtype=np.uint8)
        if top_row_concat.shape[1] < composite_width:
            pad_width = composite_width - top_row_concat.shape[1]
            top_row_concat = np.pad(top_row_concat, ((0, 0), (0, pad_width), (0, 0)), mode="constant")
        else:
            top_row_concat = top_row_concat[:, :composite_width]

        bottom_row_images = []
        for data in bottom_row_videos:
            target_idx = data["release_index"] + t
            if 0 <= target_idx < len(data["frames"]):
                frame = data["frames"][target_idx]
                resized = cv2.resize(frame, (data["resized_width"], row_height))
            else:
                resized = np.zeros((row_height, data["resized_width"], 3), dtype=np.uint8)
            bottom_row_images.append(resized)
        if bottom_row_images:
            bottom_row_concat = np.concatenate(bottom_row_images, axis=1)
        else:
            bottom_row_concat = np.zeros((row_height, composite_width, 3), dtype=np.uint8)
        if bottom_row_concat.shape[1] < composite_width:
            pad_width = composite_width - bottom_row_concat.shape[1]
            bottom_row_concat = np.pad(bottom_row_concat, ((0, 0), (0, pad_width), (0, 0)), mode="constant")
        else:
            bottom_row_concat = bottom_row_concat[:, :composite_width]

        composite_frame = np.vstack([top_row_concat, bottom_row_concat])
        out_writer.write(composite_frame)

    out_writer.release()
    composite_end_time = time.time()
    composite_time = composite_end_time - composite_start_time
    print(f"Composite video creation time: {composite_time:.2f} seconds ({composite_time/total_composite_frames:.4f} seconds per frame)")
    
    total_end_time = time.time()
    print(f"Total execution time: {total_end_time - total_start_time:.2f} seconds")
    print(f"Composite video saved as {args.output}")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()