# tracker/camera_stream.py
import cv2
import time
import numpy as np
from typing import Optional
import logging
from tracker.optical_flow import OptimizedOpticalFlowTracker
import threading
from queue import Queue
import streamlit as st

logger = logging.getLogger(__name__)

class CameraStream:
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        self.tracker = None
        self.is_running = False
        self.frame_queue = Queue(maxsize=2)  # Limit queue size to reduce latency
        self.thread = None
        
    def start(self) -> bool:
        """Initialize camera and tracker."""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                logger.error("Failed to open camera")
                return False
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size
            
            # Initialize tracker
            self.tracker = OptimizedOpticalFlowTracker()
            self.is_running = True
            
            # Start capture thread
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting camera stream: {e}")
            return False
    
    def stop(self):
        """Stop the camera stream."""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
    
    def _capture_loop(self):
        """Continuous capture loop running in separate thread."""
        last_frame_time = 0
        min_frame_delay = 1.0 / 30.0  # Cap at 30 FPS to prevent overload
        #min_frame_delay = 1.0 / 24.0
        
        while self.is_running:
            current_time = time.time()
            elapsed = current_time - last_frame_time
            
            if elapsed >= min_frame_delay:
                ret, frame = self.cap.read()
                if ret:
                    # Process frame
                    processed_frame = self.tracker.detect_and_track(frame)
                    
                    # Update queue
                    if not self.frame_queue.full():
                        self.frame_queue.put(processed_frame)
                    
                    last_frame_time = current_time
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get latest processed frame."""
        if not self.is_running:
            return None
        
        try:
            frame = self.frame_queue.get_nowait()
            return frame
        except:
            return None