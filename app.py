import streamlit as st
import os
from src.search_engine import MultimodalSearchEngine
from src.config import VIDEO_DIR

st.set_page_config(layout="wide", page_title="Semantic Video Search")

@st.cache_resource
def load_engine():
    return MultimodalSearchEngine()

engine = load_engine()

st.title("🔍 Semantic Video Search Engine")
st.caption("Search inside videos using natural language (e.g., 'a dog running', 'chef cooking')")

# Search Bar
query = st.text_input("Describe a scene:", placeholder="Type here...")

if query:
    with st.spinner("Searching visual embeddings..."):
        results = engine.search(query, limit=3)
    
    if not results:
        st.warning("No matches found.")
    else:
        st.subheader("Top Matches")
        cols = st.columns(3)
        
        for idx, res in enumerate(results):
            score = 1 - res['_distance']  # LanceDB returns distance, we want similarity
            video_name = res['video_name']
            timestamp = res['timestamp']
            
            with cols[idx]:
                st.markdown(f"**{video_name}**")
                st.caption(f"Timestamp: {timestamp}s | Relevance: {score:.2f}")
                
                # Video Player
                video_path = os.path.join(VIDEO_DIR, video_name)
                video_file = open(video_path, 'rb')
                video_bytes = video_file.read()
                
                # Streamlit video player supports start_time in seconds
                st.video(video_bytes, start_time=int(timestamp))