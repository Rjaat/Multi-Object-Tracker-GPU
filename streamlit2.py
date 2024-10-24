import streamlit as st
import cv2
import tempfile
import os
from pathlib import Path
import time
import numpy as np
from datetime import datetime
import logging
from typing import Optional, List, Tuple
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Object Tracking App",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS with enhanced dark theme
def load_custom_css():
    st.markdown("""
    <style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    .main {
        padding: 1.5rem;
        font-family: 'Space Grotesk', sans-serif;
        background: #0F172A;
        color: #E2E8F0;
    }
    
    /* Hide Streamlit Default Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Typography */
    h1, h2, h3 {
        color: #F8FAFC;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    /* Premium Container */
    .premium-container {
        background: linear-gradient(145deg, #1E293B, #0F172A);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    /* Video Container */
    .video-container {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Custom Button */
    .stButton > button {
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%);
        color: white;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        border: 1px solid rgba(99, 102, 241, 0.2);
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #4338CA 0%, #4F46E5 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
    }
    
    /* Recording Indicator */
    .recording-indicator {
        color: #EF4444;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: rgba(239, 68, 68, 0.1);
        border-radius: 6px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'processed_video' not in st.session_state:
        st.session_state.processed_video = None
    if 'input_video' not in st.session_state:
        st.session_state.input_video = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'frames_buffer' not in st.session_state:
        st.session_state.frames_buffer = []
    if 'camera_video' not in st.session_state:
        st.session_state.camera_video = None

def create_temp_file(suffix: str) -> str:
    """Create a temporary file with the given suffix."""
    temp_dir = Path(tempfile.gettempdir())
    return str(temp_dir / f"temp_{int(time.time())}{suffix}")

def process_uploaded_video(video_file) -> Optional[str]:
    """Process an uploaded video file."""
    try:
        temp_path = create_temp_file('.mp4')
        with open(temp_path, 'wb') as f:
            f.write(video_file.getvalue())
        return temp_path
    except Exception as e:
        logger.error(f"Error processing uploaded video: {e}")
        return None

def start_camera_recording():
    """Initialize camera recording."""
    st.session_state.recording = True
    st.session_state.frames_buffer = []
    st.session_state.camera_video = cv2.VideoCapture(0)

def stop_camera_recording() -> Optional[str]:
    """Stop camera recording and save the video."""
    if st.session_state.camera_video:
        st.session_state.recording = False
        st.session_state.camera_video.release()
        
        if st.session_state.frames_buffer:
            output_path = create_temp_file('.mp4')
            height, width = st.session_state.frames_buffer[0].shape[:2]
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
            
            for frame in st.session_state.frames_buffer:
                out.write(frame)
            out.release()
            
            return output_path
    return None

def main():
    # Initialize the app
    load_custom_css()
    init_session_state()
    
    # App header
    st.markdown("""
    <div class="premium-container">
        <h1>🎯 Object Tracking App</h1>
        <p>Upload a video or use your camera to track objects in real-time.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Upload Video", "Use Camera"],
        horizontal=True
    )
    
    if input_method == "Upload Video":
        video_file = st.file_uploader(
            "Upload a video file",
            type=['mp4', 'avi', 'mov'],
            help="Supported formats: MP4, AVI, MOV"
        )
        
        if video_file:
            st.session_state.input_video = process_uploaded_video(video_file)
            if st.session_state.input_video:
                st.success("Video uploaded successfully!")
    
    else:  # Camera input
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        if not st.session_state.recording:
            if st.button("Start Recording"):
                start_camera_recording()
        else:
            st.markdown('<div class="recording-indicator">● Recording...</div>', unsafe_allow_html=True)
            if st.button("Stop Recording"):
                video_path = stop_camera_recording()
                if video_path:
                    st.session_state.input_video = video_path
                    st.success("Recording saved successfully!")
        
        # Display camera feed
        if st.session_state.recording and st.session_state.camera_video:
            stframe = st.empty()
            while st.session_state.recording:
                ret, frame = st.session_state.camera_video.read()
                if ret:
                    st.session_state.frames_buffer.append(frame)
                    stframe.image(frame, channels="BGR")
                else:
                    break
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Processing section
    if st.session_state.input_video:
        st.markdown("""
        <div class="premium-container">
            <h2>Video Processing</h2>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Start Processing"):
            with st.spinner("Processing video..."):
                # Here you would call your OptimizedOpticalFlowTracker
                # For now, we'll just simulate processing
                time.sleep(2)
                st.session_state.processed_video = st.session_state.input_video
                st.session_state.processing_complete = True
        
        if st.session_state.processing_complete:
            st.success("Processing complete!")
            st.video(st.session_state.processed_video)
    
    # Instructions
    with st.expander("📖 Instructions"):
        st.markdown("""
        ### How to use this app:
        1. Choose your input method (upload video or use camera)
        2. If uploading, select your video file
        3. If using camera, click 'Start Recording' and record your scene
        4. Click 'Start Processing' to begin object tracking
        5. View the results and download if needed
        
        ### Supported Features:
        - Multiple object tracking
        - Real-time processing
        - Video upload and camera recording
        - Processing progress visualization
        """)

if __name__ == "__main__":
    main()