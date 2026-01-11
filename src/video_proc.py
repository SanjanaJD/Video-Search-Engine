import cv2
import os
from PIL import Image
from .config import FRAME_INTERVAL

def extract_frames(video_path):
    """
    Extracts frames from a video file every FRAME_INTERVAL seconds.
    Returns a list of dictionaries: [{'image': PIL Image, 'timestamp': float, 'video_name': str}]
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        # Fallback or error if video is unreadable
        fps = 24.0 
    
    video_name = os.path.basename(video_path)
    
    frames_data = []
    current_sec = 0
    
    while cap.isOpened():
        # Calculate frame ID for the current second
        frame_id = int(current_sec * fps)
        
        # Get total frame count to avoid seeking past end
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if frame_id >= total_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR (OpenCV) to RGB (PIL)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        frames_data.append({
            "image": pil_image,
            "timestamp": current_sec,
            "video_name": video_name
        })
        
        current_sec += FRAME_INTERVAL

    cap.release()
    return frames_data