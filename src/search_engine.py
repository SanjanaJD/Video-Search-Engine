import lancedb
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from .config import DB_PATH, TABLE_NAME, MODEL_ID

class MultimodalSearchEngine:
    def __init__(self):
        print("Loading CLIP Model... (this may take a moment)")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(MODEL_ID).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(MODEL_ID)
        
        # Connect to LanceDB
        self.db = lancedb.connect(DB_PATH)
        self.table = None
        self._init_table()

    def _init_table(self):
        """Initialize or load the LanceDB table."""
        if TABLE_NAME in self.db.table_names():
            self.table = self.db.open_table(TABLE_NAME)
        else:
            # Define schema implicitly by passing the first record later
            pass

    def embed_text(self, text):
        """Convert text query to vector."""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
        return features.cpu().numpy().flatten()

    def embed_images(self, images):
        """Convert list of PIL Images to vectors."""
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        return features.cpu().numpy()

    def index_video_frames(self, frames_data):
        """Embeds frames and saves them to the database."""
        images = [f['image'] for f in frames_data]
        vectors = self.embed_images(images)
        
        data_to_store = []
        for i, vector in enumerate(vectors):
            data_to_store.append({
                "vector": vector,
                "video_name": frames_data[i]['video_name'],
                "timestamp": frames_data[i]['timestamp'],
                "frame_id": i  # Simple unique ID helper
            })
            
        if self.table is None:
            self.table = self.db.create_table(TABLE_NAME, data=data_to_store)
        else:
            self.table.add(data_to_store)
        print(f"Indexed {len(data_to_store)} frames.")

    def search(self, query_text, limit=5):
        """Search the database for frames matching the text."""
        query_vector = self.embed_text(query_text)
        
        if self.table is None:
            return []
            
        results = self.table.search(query_vector).limit(limit).to_list()
        return results