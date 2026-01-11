import os
from src.config import VIDEO_DIR
from src.video_proc import extract_frames
from src.search_engine import MultimodalSearchEngine

def main():
    # 1. Initialize Engine
    engine = MultimodalSearchEngine()
    
    # 2. Process all videos in the folder
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.mov', '.avi'))]
    
    if not video_files:
        print("No videos found in data/videos/. Please add some files!")
        return

    for video_file in video_files:
        print(f"Processing {video_file}...")
        video_path = os.path.join(VIDEO_DIR, video_file)
        
        # Extract
        frames = extract_frames(video_path)
        print(f"Extracted {len(frames)} frames.")
        
        # Embed & Store
        engine.index_video_frames(frames)
        
    print("Ingestion Complete! You can now run app.py")

if __name__ == "__main__":
    main()