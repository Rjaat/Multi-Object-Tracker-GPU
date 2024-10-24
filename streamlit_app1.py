import streamlit as st
import cv2
import tempfile
import os
from pathlib import Path
from GPU_MultiObjectTracking import OptimizedOpticalFlowTracker
import time
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Object Tracking App",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced Dark Theme Custom CSS
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
    h1, h2, h3, h4, h5, h6 {
        color: #F8FAFC;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    p {
        color: #CBD5E1;
        line-height: 1.6;
    }
    
    /* Section Divider */
    .section-divider {
        border: none;
        border-top: 1px solid #1E293B;
        margin: 2rem 0;
        opacity: 0.8;
    }
    
    /* Premium Container Styles */
    .premium-container {
        background: linear-gradient(145deg, #1E293B, #0F172A);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    
    .premium-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.3), transparent);
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%);
        color: white;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        border: none;
        font-weight: 500;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        border: 1px solid rgba(99, 102, 241, 0.2);
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #4338CA 0%, #4F46E5 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    /* Video Container Styles */
    .video-container {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        position: relative;
    }
    
    .video-container::after {
        content: '';
        position: absolute;
        top: -1px;
        left: -1px;
        right: -1px;
        bottom: -1px;
        border-radius: 12px;
        border: 1px solid transparent;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), transparent);
        pointer-events: none;
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
        border: 1px solid rgba(239, 68, 68, 0.2);
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    /* Progress Bar Styling */
    .stProgress > div > div {
        background: linear-gradient(90deg, #4F46E5, #6366F1);
    }
    
    /* File Uploader Styling */
    .uploadedFile {
        border: 1px dashed #334155;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        background: #1E293B;
        transition: all 0.3s ease;
    }
    
    .uploadedFile:hover {
        border-color: #4F46E5;
        background: #262f45;
    }
    
    /* Instructions Card */
    .instructions-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 2rem;
        position: relative;
    }
    
    .instructions-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), transparent);
        pointer-events: none;
        border-radius: 12px;
    }
    
    .instructions-card h3 {
        color: #F8FAFC;
        margin-bottom: 1rem;
    }
    
    .instructions-card ol {
        margin-left: 1.5rem;
        color: #CBD5E1;
    }
    
    .instructions-card li {
        margin-bottom: 0.5rem;
    }
    
    /* Status Messages */
    .status-message {
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: 500;
    }
    
    .status-success {
        background: rgba(34, 197, 94, 0.1);
        color: #4ADE80;
        border: 1px solid rgba(34, 197, 94, 0.2);
    }
    
    .status-processing {
        background: rgba(99, 102, 241, 0.1);
        color: #818CF8;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    /* Section Headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #334155;
    }
    
    .section-header h3 {
        margin: 0;
    }
    
    /* Premium Accents */
    .premium-accent {
        position: relative;
        padding-left: 1rem;
    }
    
    .premium-accent::before {
        content: '';
        position: absolute;
        left: 0;
        top: 50%;
        transform: translateY(-50%);
        width: 3px;
        height: 70%;
        background: linear-gradient(to bottom, #4F46E5, #6366F1);
        border-radius: 2px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state variables (same as before)
if 'processed_video' not in st.session_state:
    st.session_state.processed_video = None
if 'input_video' not in st.session_state:
    st.session_state.input_video = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'camera_video' not in st.session_state:
    st.session_state.camera_video = None
if 'frames_buffer' not in st.session_state:
    st.session_state.frames_buffer = []

def ensure_directory_exists(path):
    """Ensure the directory exists for saving files"""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def save_video_from_frames(frames, output_path, fps=20.0):
    """Save video from frames buffer"""
    if not frames:
        return None
    
    try:
        ensure_directory_exists(output_path)
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        logger.info(f"Video saved successfully to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error saving video: {str(e)}")
        return None

def process_camera_stream():
    """Real-time camera streaming and processing function"""
    try:
        # Initialize video capture
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open camera. Please check your camera connection.")
            return

        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initialize the tracker
        tracker = OptimizedOpticalFlowTracker()
        
        # Create container and placeholders once, outside the loop
        stream_container = st.container()
        with stream_container:
            st.markdown('<div class="video-container">', unsafe_allow_html=True)
            frame_placeholder = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)
            status_placeholder = st.empty()
        
        stop_streaming = st.button("Stop Streaming")
        
        # FPS calculation variables
        fps_start_time = time.time()
        fps_counter = 0
        current_fps = 0
        
        # Streaming loop
        while not stop_streaming:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from camera")
                break
            
            # Calculate FPS
            fps_counter += 1
            elapsed_time = time.time() - fps_start_time
            if elapsed_time > 1:
                current_fps = fps_counter / elapsed_time
                fps_counter = 0
                fps_start_time = time.time()
            
            # Process frame using the tracker
            processed_frame = tracker.process_frame(frame, current_fps)
            
            # Convert to RGB for display
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Update frame in the existing placeholder
            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
            
            # Update status with FPS
            status_text = "🎥 Live Processing..."
            if current_fps > 0:
                status_text += f" FPS: {current_fps:.2f}"
            status_placeholder.markdown(
                f'<p class="recording-indicator">{status_text}</p>',
                unsafe_allow_html=True
            )
            
            # Add a small delay to prevent overwhelming the UI
            time.sleep(0.01)
        
        # Cleanup
        cap.release()
        frame_placeholder.empty()
        status_placeholder.empty()
        
    except Exception as e:
        logger.error(f"Error during streaming: {str(e)}")
        st.error(f"An error occurred during streaming: {str(e)}")
        if 'cap' in locals() and cap.isOpened():
            cap.release()

def process_video(input_path):
    """Process the video using the OptimizedOpticalFlowTracker"""
    if not os.path.exists(input_path):
        st.error("Input video file not found")
        return None
        
    output_path = os.path.join(tempfile.gettempdir(), "processed_output.mp4")
    tracker = OptimizedOpticalFlowTracker()
    
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            st.error("Could not open input video file")
            return None
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (640, 640))
        
        frame_count = 0
        start_time = time.time()
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            progress = frame_count / total_frames if total_frames > 0 else 0
            progress_bar.progress(progress)
            
            elapsed_time = time.time() - start_time
            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            processed_frame = tracker.process_frame(frame, current_fps)
            out.write(processed_frame)
            
            frame_count += 1
            status_text.text(f"Processing frame {frame_count}... FPS: {current_fps:.2f}")
        
        cap.release()
        out.release()
        progress_bar.progress(1.0)
        status_text.text("Processing complete!")
        return output_path
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        st.error(f"An error occurred during processing: {str(e)}")
        return None

# Main App Layout with Premium Dark Theme
st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='font-size: 2.5rem; font-weight: 700; color: #F8FAFC; text-shadow: 0 2px 4px rgba(0,0,0,0.2);'>
            🎯 Smart Object Tracking
        </h1>
        <p style='font-size: 1.2rem; color: #CBD5E1; margin-top: 0.5rem;'>
            Professional AI-powered object tracking solution
        </p>
    </div>
""", unsafe_allow_html=True)

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class='section-header premium-accent'>
            <h3>📤 Upload Video</h3>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        try:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            st.session_state.input_video = tfile.name
            
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            st.markdown("""
                <div class='section-header premium-accent'>
                    <h4>📺 Original Video</h4>
                </div>
            """, unsafe_allow_html=True)
            
            with st.container():
                st.markdown('<div class="video-container">', unsafe_allow_html=True)
                st.video(uploaded_file)
                st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error uploading file: {str(e)}")

with col2:
    st.markdown("""
        <div class='section-header premium-accent'>
            <h3>📹 Live Camera Processing</h3>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("▶️ Start Live Processing", key="start_streaming"):
        process_camera_stream()

# Process button
if st.session_state.input_video and os.path.exists(st.session_state.input_video):
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    if st.button("🚀 Process Video"):
        with st.spinner(""):
            st.markdown("""
                <div class='status-message status-processing'>
                    <span>🔄 Processing video...</span>
                </div>
            """, unsafe_allow_html=True)
            output_path = process_video(st.session_state.input_video)
            if output_path and os.path.exists(output_path):
                st.session_state.processed_video = output_path
                st.session_state.processing_complete = True

# Display processed video
if st.session_state.processing_complete and st.session_state.processed_video:
#     st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
#     st.markdown("""
#         <div class='section-header premium-accent'>
#             <h3>🎥 Processed Result</h3>
#         </div>
#     """, unsafe_allow_html=True)
    
#     with st.container():
#         st.markdown('<div class="video-container">', unsafe_allow_html=True)
#         st.video(st.session_state.processed_video)
#         st.markdown('</div>', unsafe_allow_html=True)
    
    try:
        with open(st.session_state.processed_video, 'rb') as f:
            st.markdown("<div style='text-align: center; margin-top: 1rem;'>", unsafe_allow_html=True)
            st.download_button(
                label="⬇️ Download Processed Video",
                data=f.read(),
                file_name="processed_video.mp4",
                mime="video/mp4"
            )
            st.markdown("</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error preparing download: {str(e)}")

# Footer with Instructions
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown("""
    <div class='instructions-card'>
        <div class='section-header premium-accent'>
            <h3>📝 Quick Guide</h3>
        </div>
        <ol>
            <li>Select your input method:
                <ul>
                    <li>Upload a pre-recorded video, or</li>
                    <li>Use live camera feed</li>
                </ul>
            </li>
            <li>For camera input: Use the Start/Stop controls</li>
            <li>For uploaded videos: Click Process Video</li>
            <li>Monitor the processing progress</li>
            <li>Download your processed result</li>
        </ol>
    </div>
""", unsafe_allow_html=True)

# Cleanup function remains the same
def cleanup():
    try:
        if st.session_state.input_video and os.path.exists(st.session_state.input_video):
            os.unlink(st.session_state.input_video)
        if st.session_state.processed_video and os.path.exists(st.session_state.processed_video):
            os.unlink(st.session_state.processed_video)
        if st.session_state.camera_video and os.path.exists(st.session_state.camera_video):
            os.unlink(st.session_state.camera_video)
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

# Register cleanup function
import atexit
atexit.register(cleanup)
