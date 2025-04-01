import os
import sys
import argparse
import subprocess
from scenedetect import VideoManager, SceneManager, open_video
from scenedetect.detectors import ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg

def check_ffmpeg():
    """Check if ffmpeg is available in the system path."""
    try:
        result = subprocess.run(['ffmpeg', '-version'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def split_video_into_scenes(input_file, output_folder, threshold=30):
    """
    Split a video file into separate clips based on scene detection.
    
    Args:
        input_file (str): Path to input video file
        output_folder (str): Path to output folder for video clips
        threshold (int): Threshold for content detection (lower = more sensitive)
    """
    # Validate input file exists
    if not os.path.isfile(input_file):
        print(f"Error: Input file '{input_file}' does not exist")
        return False
        
    print(f"Processing video: {input_file}")
    print(f"Output folder: {output_folder}")
    
    # Check ffmpeg is installed
    if not check_ffmpeg():
        print("Error: FFmpeg is not installed or not in your PATH.")
        print("Please install FFmpeg: https://ffmpeg.org/download.html")
        return False
    
    # Create output directory if it doesn't exist
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Created output directory: {output_folder}")
        
        # Test write permission
        test_file = os.path.join(output_folder, '.test_write_permission')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
    except (IOError, PermissionError) as e:
        print(f"Error: Cannot write to output folder: {str(e)}")
        return False
    
    # Create video & scene managers
    try:
        video_manager = VideoManager([input_file])
        scene_manager = SceneManager()
        
        # Add ContentDetector (with threshold)
        scene_manager.add_detector(ContentDetector(threshold=threshold))
        
        # Improve performance by downscaling factor
        video_manager.set_downscale_factor()
        
        # Start video manager
        video_manager.start()
    except Exception as e:
        print(f"Error initializing video: {str(e)}")
        return False
    
    # Perform scene detection
    print("Detecting scenes...")
    try:
        scene_manager.detect_scenes(frame_source=video_manager)
        
        # Get list of detected scenes
        scene_list = scene_manager.get_scene_list()
        
        if not scene_list:
            print("No scenes detected! Try a lower threshold value.")
            return False
        
        print(f"Detected {len(scene_list)} scenes")
    except Exception as e:
        print(f"Error during scene detection: {str(e)}")
        return False
    
    # Generate output file names
    filename = os.path.basename(input_file)
    name, ext = os.path.splitext(filename)
    
    # Split the video into scenes
    print("Splitting video...")
    try:
        # Use custom command for better control and error reporting
        for i, (start_time, end_time) in enumerate(scene_list):
            output_file = os.path.join(output_folder, f"{name}_scene_{i+1:03d}{ext}")
            
            start_seconds = start_time.get_seconds()
            duration = end_time.get_seconds() - start_seconds
            
            cmd = [
                'ffmpeg', '-y',
                '-i', input_file,
                '-ss', str(start_seconds),
                '-t', str(duration),
                '-c', 'copy',  # Use copy to avoid re-encoding (much faster)
                output_file
            ]
            
            print(f"Creating scene {i+1}/{len(scene_list)}: {output_file}")
            result = subprocess.run(cmd,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   text=True)
            
            if result.returncode != 0:
                print(f"Error creating scene {i+1}:")
                print(result.stderr)
                continue
        
        print(f"Video splitting complete! {len(scene_list)} scenes created.")
        print(f"Output clips saved to {output_folder}")
        return True
            
    except Exception as e:
        print(f"Error during video splitting: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split a video into scenes')
    parser.add_argument('input_file', help='Path to input video file')
    parser.add_argument('output_folder', help='Path to output folder')
    parser.add_argument('--threshold', type=int, default=25,
                        help='Content detection threshold (default: 25, lower = more sensitive)')
    
    args = parser.parse_args()
    
    success = split_video_into_scenes(args.input_file, args.output_folder, args.threshold)
    if not success:
        sys.exit(1)
