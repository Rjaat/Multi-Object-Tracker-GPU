import streamlit as st
import tempfile
from pathlib import Path
import time
from typing import Optional
import logging
import os
import torch
from tracker.optical_flow import OptimizedOpticalFlowTracker
from tracker.video_processor import VideoProcessor
from tracker.camera_stream import CameraStream

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Object Tracking App",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def load_custom_css():
    st.markdown("""
    <style>
    /* Previous Global Styles remain the same */
    
    /* Enhanced Tab Design */
    .stTabs {
        background: transparent !important;
        padding: 0 !important;
        border: none !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: transparent;
        padding: 0;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(145deg, #1E293B, #0F172A);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem !important;
        white-space: pre;
        font-family: 'Space Grotesk', sans-serif;
        position: relative;
        overflow: hidden;
        min-width: 200px;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #4F46E5, #6366F1);
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
    }
    
    .stTabs [data-baseweb="tab"]:hover::before {
        transform: scaleX(1);
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #1E293B, #2D3B55);
        border-color: #4F46E5;
        box-shadow: 0 8px 20px rgba(79, 70, 229, 0.2);
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"]::before {
        transform: scaleX(1);
    }
    
    /* Tab Content */
    .tab-content {
        text-align: center;
        padding: 1rem;
    }
    
    .tab-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
        display: block;
    }
    
    .tab-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
        color: #F8FAFC;
    }
    
    .tab-description {
        font-size: 0.9rem;
        color: #94A3B8;
        margin-bottom: 0;
    }
    
    /* Content Containers */
    .content-container {
        background: linear-gradient(145deg, #1E293B, #0F172A);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .content-container:hover {
        border-color: #4F46E5;
        box-shadow: 0 8px 20px rgba(79, 70, 229, 0.15);
    }
    
    .content-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, #4F46E5, #6366F1);
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }
    
    .content-container:hover::before {
        transform: scaleX(1);
    }
    
    /* Enhanced Live Indicator */
    .live-indicator {
        background: linear-gradient(135deg, #EF4444, #DC2626);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
        animation: pulse 2s infinite;
    }
    
    /* Enhanced Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%);
        color: white;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2);
        backdrop-filter: blur(8px);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #4338CA 0%, #4F46E5 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(79, 70, 229, 0.3);
    }
    
    /* File Uploader */
    .uploadedFile {
        background: #1E293B;
        border: 2px dashed #4F46E5;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .uploadedFile:hover {
        border-color: #6366F1;
        background: #262f45;
    }
    
    /* Camera Selection */
    .stSelectbox {
        background: #1E293B;
        border-radius: 12px;
        padding: 0.5rem;
    }
    
    .stSelectbox > div {
        background: transparent;
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #4F46E5, #6366F1);
        border-radius: 9999px;
    }
    </style>
    """, unsafe_allow_html=True)


def init_session_state():
    if 'processed_video' not in st.session_state:
        st.session_state.processed_video = None
    if 'input_video' not in st.session_state:
        st.session_state.input_video = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'streaming' not in st.session_state:
        st.session_state.streaming = False
    if 'camera_stream' not in st.session_state:
        st.session_state.camera_stream = None


def handle_video_upload():
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload a video file",
        type=['mp4', 'avi', 'mov'],
        help="Supported formats: MP4, AVI, MOV"
    )
    
    if uploaded_file:
        # Save uploaded file
        temp_input = str(Path(tempfile.gettempdir()) / f"input_{int(time.time())}.mp4")
        with open(temp_input, 'wb') as f:
            f.write(uploaded_file.getvalue())
        st.session_state.input_video = temp_input
        
        # Display uploaded video
        st.markdown('<div class="video-player">', unsafe_allow_html=True)
        st.video(uploaded_file)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Processing controls
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("Process Video", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner("Processing video..."):
                    temp_output = str(Path(tempfile.gettempdir()) / f"output_{int(time.time())}.mp4")
                    tracker = OptimizedOpticalFlowTracker()
                    
                    # Process video
                    success = VideoProcessor.process_video(
                        st.session_state.input_video,
                        temp_output,
                        tracker,
                        lambda p: progress_bar.progress(p / 100)
                    )
                    
                    if success:
                        st.session_state.processed_video = temp_output
                        st.session_state.processing_complete = True
                        status_text.markdown('<div class="status-indicator status-success">Processing complete!</div>', unsafe_allow_html=True)
                        
                        # Show download button
                        with open(st.session_state.processed_video, 'rb') as f:
                            st.download_button(
                                label="Download Processed Video",
                                data=f.read(),
                                file_name="tracked_video.mp4",
                                mime="video/mp4",
                                use_container_width=True
                            )
                    else:
                        st.error("Failed to process video")
    
    st.markdown('</div>', unsafe_allow_html=True)

def process_uploaded_video(input_path: str, progress_bar) -> Optional[str]:
    """Process uploaded video using the tracker module."""
    try:
        output_path = str(Path(tempfile.gettempdir()) / f"processed_{int(time.time())}.mp4")
        tracker = OptimizedOpticalFlowTracker()
        
        def update_progress(progress: float):
            progress_bar.progress(progress / 100)
            st.session_state.progress = progress
        
        success = VideoProcessor.process_video(
            input_path,
            output_path,
            tracker,
            update_progress
        )
        
        return output_path if success else None
        
    except Exception as e:
        logger.error(f"Error in video processing: {e}")
        return None


def handle_live_stream():
    st.markdown('<div class="stream-container">', unsafe_allow_html=True)
    
    # Camera selection
    camera_options = {0: "Default Camera", 1: "External Camera 1", 2: "External Camera 2"}
    camera_id = st.selectbox(
        "Select Camera",
        options=list(camera_options.keys()),
        format_func=lambda x: camera_options[x]
    )
    
    # Stream controls
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if not st.session_state.streaming:
            if st.button("Start Streaming", use_container_width=True):
                st.session_state.camera_stream = CameraStream(camera_id)
                if st.session_state.camera_stream.start():
                    st.session_state.streaming = True
                else:
                    st.error("Failed to start camera stream")
    
    with col2:
        if st.session_state.streaming:
            if st.button("Stop Streaming", use_container_width=True):
                if st.session_state.camera_stream:
                    st.session_state.camera_stream.stop()
                    st.session_state.streaming = False
                    st.session_state.camera_stream = None
    
    with col3:
        if st.session_state.streaming:
            st.markdown('<div class="live-indicator">● LIVE</div>', unsafe_allow_html=True)
    
    # # Stream display using a placeholder
    # if st.session_state.streaming and st.session_state.camera_stream:
    #     stream_placeholder = st.empty()
    #     update_frequency = 0.033  # Target ~30 FPS updates
        
    #     while st.session_state.streaming:
    #         frame = st.session_state.camera_stream.get_frame()
    #         if frame is not None:
    #             # Convert frame for Streamlit display
    #             try:
    #                 stream_placeholder.image(frame, channels="BGR", use_column_width=True)
    #             except:
    #                 continue
            
    #         # Control update rate
    #         time.sleep(update_frequency)
    
    # st.markdown('</div>', unsafe_allow_html=True)
    # Stream display
    stream_placeholder = st.empty()
    
    if st.session_state.streaming and st.session_state.camera_stream:
        while st.session_state.streaming:
            frame = st.session_state.camera_stream.get_frame()
            if frame is not None:
                stream_placeholder.image(frame, channels="BGR", use_column_width=True)
            time.sleep(0.033)  # ~30 FPS
    
    st.markdown('</div>', unsafe_allow_html=True)



def main():
    load_custom_css()
    init_session_state()
    
    # App header
    st.markdown("""
    <div class="app-header">
        <h1>🎯 Advanced Object Tracking</h1>
        <p>Real-time object tracking with YOLO and Optical Flow</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content
    tab1, tab2 = st.tabs(["📹 Live Streaming", "📤 Video Upload"])
    
    with tab1:
        handle_live_stream()
    
    with tab2:
        handle_video_upload()

    # Floating device info
    st.markdown("""
    <div style="position: fixed; bottom: 2rem; right: 2rem; 
                background: linear-gradient(135deg, #1E293B, #0F172A);
                border: 1px solid #334155; border-radius: 12px; padding: 1rem;
                box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3); z-index: 1000;">
        <div style="color: #94A3B8; font-size: 0.9rem;">
            Running on: {}
        </div>
    </div>
    """.format("GPU 🚀" if torch.cuda.is_available() else "CPU 💻"), unsafe_allow_html=True)
    
    # Model information
    with st.expander("ℹ️ Model Information"):
        st.markdown("""
        ### YOLO Object Detection
        - Using YOLOv8 model for object detection
        - Optimized for real-time processing
        - Running on: {}
        
        ### Features
        - Real-time object tracking
        - Live camera streaming
        - Video file processing
        - Track history visualization
        - FPS monitoring
        """.format("GPU" if torch.cuda.is_available() else "CPU"))

if __name__ == "__main__":
    main()