# test_video.py
import sys
import os
sys.path.append('.')

from video_loader import VideoProcessor

processor = VideoProcessor()
test_file = "C:\\Users\\chatt\\OneDrive\\Desktop\\multimodal_rag\\data\\test.mp4"  # Make sure this file exists

if os.path.exists(test_file):
    result = processor.process_video_file(test_file)
    print(f"Success: {result}")
else:
    print(f"Test file {test_file} not found")