import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_DIR = os.path.join(BASE_DIR, 'data', 'videos')
DB_PATH = os.path.join(BASE_DIR, 'data', 'lancedb')
TABLE_NAME = "video_search_index"

# Model Settings
MODEL_ID = "openai/clip-vit-base-patch32"
FRAME_INTERVAL = 2  # Extract 1 frame every X seconds (adjust for granularity)