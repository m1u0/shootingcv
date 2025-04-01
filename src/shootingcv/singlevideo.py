import cv2
from ultralytics import YOLO
import mediapipe as mp
import argparse
import subprocess
import os
import math
from absl import logging
import numpy as np
from collections import deque

os.environ["QT_QPA_PLATFORM"] = "minimal"

# Constants
BALL_TO_PERSON_RATIO = 0.08  # Minimum ball size relative to person height
DEFAULT_BALL_SIZE = (20, 20)  # Default ball size when no previous detection

class BallTracker:
    """Class to manage ball tracking state and prediction logic."""
    
    def __init__(self, history_length=5, conf_threshold=0.05):
        self.history_length = history_length
        self.conf_threshold = conf_threshold
        self.position_history = []
        self.last_ball_bbox = None
        self.predicted_center = None
        
    def update_with_detection(self, bbox):
        """Update tracker with a new detection."""
        self.last_ball_bbox = bbox
        center = self.calculate_center(bbox)
        self.predicted_center = center
        
        # Update position history
        self.position_history.append(center)
        if len(self.position_history) > self.history_length:
            self.position_history.pop(0)
        
        return center
        
    def predict_position(self):
        """Predict ball position based on velocity."""
        if len(self.position_history) < 2:
            return None, None
            
        # Calculate average velocity vector
        num_positions = len(self.position_history)
        dx = (self.position_history[-1][0] - self.position_history[0][0]) / (num_positions - 1)
        dy = (self.position_history[-1][1] - self.position_history[0][1]) / (num_positions - 1)
        
        if self.predicted_center is None:
            self.predicted_center = self.position_history[-1]
        else:
            # Update predicted center
            self.predicted_center = (int(self.predicted_center[0] + dx), int(self.predicted_center[1] + dy))
        
        # Use the size of the last known bounding box
        if self.last_ball_bbox is not None:
            width_box = self.last_ball_bbox[2] - self.last_ball_bbox[0]
            height_box = self.last_ball_bbox[3] - self.last_ball_bbox[1]
        else:
            width_box, height_box = DEFAULT_BALL_SIZE
        
        predicted_bbox = [
            self.predicted_center[0] - width_box // 2,
            self.predicted_center[1] - height_box // 2,
            self.predicted_center[0] + width_box // 2,
            self.predicted_center[1] + height_box // 2
        ]
        self.last_ball_bbox = predicted_bbox
        
        return self.predicted_center, predicted_bbox
    
    @staticmethod
    def calculate_center(bbox):
        """Calculate the center point of a bounding box."""
        return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

def get_video_rotation(video_path):
    """Get rotation metadata using ffprobe."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
             'stream_tags=rotate', '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        rotation_str = result.stdout.strip()
        return int(rotation_str) if rotation_str else 0
    except Exception as e:
        print("Warning: Could not determine rotation, defaulting to 0.")
        return 0

def setup_video_io(input_path, output_path):
    """Setup video capture and writer objects."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Error: Could not open video.")
        
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Determine input video rotation
    rotation = get_video_rotation(input_path)
    if rotation in [90, 270]:
        width, height = height, width
    
    # Define the codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    return cap, out, rotation, width, height, fps

def process_frame(frame, rotation):
    """Apply rotation to frame if needed."""
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame

def calculate_person_height(pose_landmarks, frame_shape):
    """Calculate the person's height from pose landmarks."""
    if not pose_landmarks:
        return None
        
    landmarks = pose_landmarks.landmark
    min_y = min(lm.y for lm in landmarks)
    max_y = max(lm.y for lm in landmarks)
    return (max_y - min_y) * frame_shape[0]

def calculate_torso_data(pose_landmarks, frame_shape):
    """Calculate torso height and related points from pose landmarks."""
    if not pose_landmarks:
        return None, None, None, None
    try:
        landmarks = pose_landmarks.landmark
        
        # Extract key points: using MediaPipe indices
        left_shoulder = (landmarks[11].x * frame_shape[1], landmarks[11].y * frame_shape[0])
        right_shoulder = (landmarks[12].x * frame_shape[1], landmarks[12].y * frame_shape[0])
        left_hip = (landmarks[23].x * frame_shape[1], landmarks[23].y * frame_shape[0])
        right_hip = (landmarks[24].x * frame_shape[1], landmarks[24].y * frame_shape[0])
        
        # Calculate centers
        shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2,
                           (left_shoulder[1] + right_shoulder[1]) / 2)
        hip_center = ((left_hip[0] + right_hip[0]) / 2,
                       (left_hip[1] + right_hip[1]) / 2)
        
        # Calculate torso height (Euclidean distance between shoulder and hip centers)
        torso_height = math.sqrt((shoulder_center[0] - hip_center[0])**2 +
                                 (shoulder_center[1] - hip_center[1])**2)
        
        return torso_height, shoulder_center, hip_center, left_shoulder, right_shoulder, left_hip, right_hip
    except Exception:
        return None, None, None, None

def get_right_wrist_point(pose_landmarks, frame_shape):
    """Get the position of the right wrist."""
    if not pose_landmarks:
        return None
    right_wrist = pose_landmarks.landmark[16]
    return (int(right_wrist.x * frame_shape[1]), int(right_wrist.y * frame_shape[0]))

def get_right_elbow_point(pose_landmarks, frame_shape):
    """Get the position of the right elbow."""
    if not pose_landmarks:
        return None
    right_elbow = pose_landmarks.landmark[14]
    return (int(right_elbow.x * frame_shape[1]), int(right_elbow.y * frame_shape[0]))

def calculate_distance_fraction(point1, point2, torso_height):
    """Calculate distance between points as a fraction of torso height."""
    if None in (point1, point2, torso_height) or torso_height <= 0:
        return None
    distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    return distance / torso_height

def calculate_torso_arm_angle(shoulder, elbow, wrist, hip_center):
    """Calculate the angle between torso and arm."""
    if None in (shoulder, elbow, wrist, hip_center):
        return None
    
    # Calculate torso vector (from shoulder to hip)
    torso_vector = (hip_center[0] - shoulder[0], hip_center[1] - shoulder[1])
    
    # Calculate arm vector (from shoulder to elbow instead of wrist)
    # This better represents the upper arm position which is more relevant for shooting mechanics
    arm_vector = (elbow[0] - shoulder[0], elbow[1] - shoulder[1])
    
    # Calculate dot product
    dot_product = torso_vector[0] * arm_vector[0] + torso_vector[1] * arm_vector[1]
    
    # Calculate magnitudes
    torso_magnitude = math.sqrt(torso_vector[0]**2 + torso_vector[1]**2)
    arm_magnitude = math.sqrt(arm_vector[0]**2 + arm_vector[1]**2)
    
    # Calculate cosine of the angle
    if torso_magnitude == 0 or arm_magnitude == 0:
        return None
    
    cosine_angle = dot_product / (torso_magnitude * arm_magnitude)
    
    # Handle floating point errors
    cosine_angle = max(-1.0, min(1.0, cosine_angle))
    
    # Calculate angle in degrees
    angle = math.degrees(math.acos(cosine_angle))
    return angle

def draw_ball(frame, bbox, center, is_prediction=False):
    """Draw ball bounding box and center point."""
    color = (0, 165, 255) if is_prediction else (0, 255, 0)
    label = "Predicted Ball" if is_prediction else "Ball"
    
    # Draw bounding box
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    
    # Draw label
    cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw center dot
    cv2.circle(frame, center, radius=5, color=color, thickness=-1)

def draw_distance_indicator(frame, distance_fraction, width):
    """Draw distance indicator on the frame."""
    text = f"Dist: {distance_fraction:.2f} torso" if distance_fraction is not None else "Dist: N/A"
    cv2.putText(frame, text, (width - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

def annotate_release(frame):
    """Annotate frame with 'RELEASE' in bright red in the top left."""
    cv2.putText(frame, "RELEASE", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

def annotate_non_release(frame, reasons):
    """Annotate frame with the list of constraints preventing release detection."""
    y_offset = 30
    for reason in reasons:
        cv2.putText(frame, reason, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y_offset += 25

def check_release_candidate(window, fps, criteria=1):
    """
    Check whether the candidate frame (index 3) in the 7-frame window qualifies as the release frame.
    
    Args:
        window: 7-frame window with candidate frame at index 3
        fps: Frames per second of the video
        criteria: Which set of criteria to use (1, 2, or 3)
        
    Returns:
        (True, []) if all conditions pass, else (False, [list of failing conditions]).
    """
    candidate = window[3]
    reasons = []
    
    # Check ball position requirement (common across all criteria)
    if candidate['shoulder'] is None or candidate['hip'] is None:
        reasons.append("Missing shoulder/hip data")
    else:
        threshold_y = candidate['hip'][1] - (2/3) * (candidate['hip'][1] - candidate['shoulder'][1])
        if candidate['ball_center'] is None:
            reasons.append("Missing ball center")
        elif candidate['ball_center'][1] > threshold_y:
            reasons.append("Ball center below threshold")
        if candidate['wrist'] is None:
            reasons.append("Missing wrist")
        elif candidate['wrist'][1] > threshold_y:
            reasons.append("Wrist below threshold")
    
    # Check ball vertical velocity (common across all criteria)
    ball_vel_failures = 0
    if candidate['torso_height'] is None or candidate['torso_height'] <= 0:
        reasons.append("Invalid torso height")
    else:
        allowed_ball_delta = (3 * candidate['torso_height']) / fps
        # Check pairs: indices 0->1, 1->2, 2->3, 3->4, 4->5, 5->6.
        for i in range(1, 7):
            prev = window[i-1]['ball_rel_y']
            curr = window[i]['ball_rel_y']
            if prev is None or curr is None:
                ball_vel_failures += 1
            else:
                delta = curr - prev
                # Upward motion (delta negative) is fine.
                # If moving downward (delta positive), allow if delta <= allowed_ball_delta.
                if delta > allowed_ball_delta:
                    ball_vel_failures += 1
        if ball_vel_failures > 2:
            reasons.append(f"Ball velocity failure count = {ball_vel_failures} (>2)")
    
    # Criteria Set 1: Original criteria
    if criteria == 1:
        # Check ball-hand distance
        if candidate['ball_hand_distance'] is None:
            reasons.append("Candidate missing ball-hand distance")
        elif candidate['ball_hand_distance'] < 0.5:
            reasons.append("Candidate ball-hand distance < 0.5")
        
        # For adjacent frames: frames 0-2 must be below 0.5 and frames 4-6 above 0.5.
        ball_hand_failures = 0
        for i in range(0, 3):
            if window[i]['ball_hand_distance'] is None or window[i]['ball_hand_distance'] >= 0.5:
                ball_hand_failures += 1
        for i in range(4, 7):
            if window[i]['ball_hand_distance'] is None or window[i]['ball_hand_distance'] <= 0.5:
                ball_hand_failures += 1
        if ball_hand_failures > 2:
            reasons.append(f"Ball-hand distance adjacent failure count = {ball_hand_failures} (>2)")
        
        # Check elbow motion (for frames before candidate)
        elbow_failures = 0
        if candidate['torso_height'] is None or candidate['torso_height'] <= 0:
            reasons.append("Invalid torso height for elbow check")
        else:
            allowed_elbow_delta = (3 * candidate['torso_height']) / fps
            for i in range(1, 4):
                prev = window[i-1]['elbow_rel_y']
                curr = window[i]['elbow_rel_y']
                if prev is None or curr is None:
                    elbow_failures += 1
                else:
                    delta = curr - prev
                    # Upward motion (delta negative) is acceptable.
                    # If moving downward, allow if delta <= allowed_elbow_delta.
                    if delta > allowed_elbow_delta:
                        elbow_failures += 1
            if elbow_failures > 2:
                reasons.append(f"Elbow motion failure count = {elbow_failures} (>2)")
    
    # Criteria Set 2: Torso and arm angle >= 90 degrees
    elif criteria == 2:
        # Calculate torso-arm angle
        angle = candidate['torso_arm_angle']
        if angle is None:
            reasons.append("Unable to calculate torso-arm angle")
        # Using original 90 degree threshold
        elif angle < 90:
            reasons.append(f"Torso-arm angle ({angle:.1f}) < 90 degrees")
        
        # Check previous frames to ensure this is the first frame >= 90 degrees
        for i in range(0, 3):
            prev_angle = window[i]['torso_arm_angle']
            if prev_angle is not None and prev_angle >= 90:
                reasons.append(f"Frame {i} already has angle >= 90 degrees")
                break
    
    # Criteria Set 3: Ball height 2/3 torsos above shoulder
    elif criteria == 3:
        # Check if ball is 2/3 torsos above shoulder
        if candidate['ball_center'] is None or candidate['shoulder'] is None or candidate['torso_height'] is None:
            reasons.append("Missing data for ball height check")
        else:
            ball_shoulder_height = candidate['shoulder'][1] - candidate['ball_center'][1]
            required_height = 2/3 * candidate['torso_height']
            
            if ball_shoulder_height < required_height:
                reasons.append(f"Ball not high enough above shoulder ({ball_shoulder_height:.1f} < {required_height:.1f})")
            
            # Check previous frames to ensure this is the first frame meeting the height requirement
            for i in range(0, 3):
                if (window[i]['ball_center'] is not None and 
                    window[i]['shoulder'] is not None and 
                    window[i]['torso_height'] is not None):
                    prev_height = window[i]['shoulder'][1] - window[i]['ball_center'][1]
                    if prev_height >= required_height:
                        reasons.append(f"Frame {i} already has sufficient ball height")
                        break
        
        # Check elbow motion (same as criteria 1)
        elbow_failures = 0
        if candidate['torso_height'] is None or candidate['torso_height'] <= 0:
            reasons.append("Invalid torso height for elbow check")
        else:
            allowed_elbow_delta = (3 * candidate['torso_height']) / fps
            for i in range(1, 4):
                prev = window[i-1]['elbow_rel_y']
                curr = window[i]['elbow_rel_y']
                if prev is None or curr is None:
                    elbow_failures += 1
                else:
                    delta = curr - prev
                    # Upward motion (delta negative) is acceptable.
                    # If moving downward, allow if delta <= allowed_elbow_delta.
                    if delta > allowed_elbow_delta:
                        elbow_failures += 1
            if elbow_failures > 2:
                reasons.append(f"Elbow motion failure count = {elbow_failures} (>2)")
    
    valid = (len(reasons) == 0)
    return valid, reasons

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, required=True, help='Path to output video')
    parser.add_argument('--history', type=int, default=5, help='Number of frames for velocity estimation')
    parser.add_argument('--criteria', type=int, default=1, choices=[1, 2, 3], 
                        help='Set of criteria to use for release detection: 1=original, 2=angle-based, 3=ball-height')
    args = parser.parse_args()
    
    # Initialize models
    model = YOLO("yolo11x.pt")
    pose_detector = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    mp_pose_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    
    # Setup video I/O
    cap, out, rotation, width, height, fps = setup_video_io(args.input, args.output)
    
    # Initialize ball tracker
    ball_tracker = BallTracker(history_length=args.history)
    
    # Create a buffer (rolling window) to hold frame data for release detection (size 7)
    frame_buffer = deque()
    
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply rotation if necessary
        frame = process_frame(frame, rotation)
        
        # Convert frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect body pose
        pose_results = pose_detector.process(frame_rgb)
        
        # Calculate person height (for ball size filtering)
        person_height_pixels = calculate_person_height(pose_results.pose_landmarks, frame.shape)
        
        # Draw pose landmarks if available
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_pose_styles.get_default_pose_landmarks_style()
            )
        
        # Detect ball with YOLO
        detected_ball = False
        results = model(frame, conf=ball_tracker.conf_threshold, verbose=False)[0]
        
        if results.boxes is not None:
            for box in results.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)  # [x1, y1, x2, y2]
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                class_name = model.names[cls]
                
                if class_name == 'sports ball':
                    # Calculate ball diameter
                    ball_width = xyxy[2] - xyxy[0]
                    ball_height = xyxy[3] - xyxy[1]
                    ball_diameter = max(ball_width, ball_height)
                    
                    # Check ball size relative to person
                    if person_height_pixels is not None:
                        if ball_diameter < BALL_TO_PERSON_RATIO * person_height_pixels:
                            continue
                    
                    # Accept the detection
                    detected_ball = True
                    
                    # Update tracker and get center
                    detected_center = ball_tracker.update_with_detection(xyxy)
                    
                    # Draw ball on current frame
                    draw_ball(frame, xyxy, detected_center)
                    
                    # Add confidence to the label
                    cv2.putText(frame, f"{conf:.2f}", (xyxy[0], xyxy[1] - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Only use the first valid ball detection
                    break
        
        # Predict ball position if not detected
        if not detected_ball:
            predicted_center, predicted_bbox = ball_tracker.predict_position()
            if predicted_center is not None:
                draw_ball(frame, predicted_bbox, predicted_center, is_prediction=True)
        
        # Get right wrist and right elbow positions
        wrist_point = get_right_wrist_point(pose_results.pose_landmarks, frame.shape)
        elbow_point = get_right_elbow_point(pose_results.pose_landmarks, frame.shape)
        
        # Calculate torso data
        torso_data = calculate_torso_data(pose_results.pose_landmarks, frame.shape)
        torso_height = torso_data[0] if torso_data else None
        shoulder_center = torso_data[1] if torso_data else None
        hip_center = torso_data[2] if torso_data else None
        right_shoulder = torso_data[4] if torso_data else None
        
        # Use ball_tracker.predicted_center as the ball center
        ball_center = ball_tracker.predicted_center
        
        # Calculate normalized ball-hand distance
        ball_hand_distance = None
        if ball_center and wrist_point and torso_height:
            ball_hand_distance = calculate_distance_fraction(ball_center, wrist_point, torso_height)
        
        # Calculate ball relative vertical position (relative to shoulder)
        ball_rel_y = None
        if ball_center and shoulder_center:
            ball_rel_y = ball_center[1] - shoulder_center[1]
        
        # Calculate elbow relative vertical position (relative to shoulder)
        elbow_rel_y = None
        if elbow_point and shoulder_center:
            elbow_rel_y = elbow_point[1] - shoulder_center[1]
        
        # Calculate torso-arm angle (for criteria set 2)
        torso_arm_angle = None
        if right_shoulder and elbow_point and wrist_point and hip_center:
            torso_arm_angle = calculate_torso_arm_angle(right_shoulder, elbow_point, wrist_point, hip_center)
        
        # Prepare a data dictionary for the current frame for release detection
        frame_data = {
            'frame_index': frame_index,
            'frame': frame,
            'ball_center': ball_center,
            'wrist': wrist_point,
            'elbow': elbow_point,
            'shoulder': shoulder_center,
            'hip': hip_center,
            'torso_height': torso_height,
            'ball_hand_distance': ball_hand_distance,
            'ball_rel_y': ball_rel_y,
            'elbow_rel_y': elbow_rel_y,
            'torso_arm_angle': torso_arm_angle,  # Added for criteria set 2
            'release_detected': False  # flag to mark release candidate
        }
        
        frame_buffer.append(frame_data)
        frame_index += 1
        
        # When we have a full window of 7 frames, check for release detection in the middle candidate.
        if len(frame_buffer) == 7:
            window = list(frame_buffer)
            valid, reasons = check_release_candidate(window, fps, args.criteria)
            if valid:
                window[3]['release_detected'] = True
                frame_buffer[3]['release_detected'] = True
            else:
                # Annotate the candidate frame with the list of failing restraints.
                annotate_non_release(window[3]['frame'], reasons)
            
            # Output the oldest frame (which now has full context for any release detection)
            oldest = frame_buffer.popleft()
            if oldest['release_detected']:
                annotate_release(oldest['frame'])
            out.write(oldest['frame'])
        
    # Flush remaining frames in the buffer
    while frame_buffer:
        oldest = frame_buffer.popleft()
        if oldest['release_detected']:
            annotate_release(oldest['frame'])
        out.write(oldest['frame'])

    # Cleanup
    cap.release()
    out.release()

if __name__ == "__main__":
    main()
