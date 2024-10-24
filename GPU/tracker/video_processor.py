# tracker/video_processor.py
import cv2
import os
from typing import Optional, Callable
import logging
import torch
from tracker.optical_flow import OptimizedOpticalFlowTracker  # Fixed import

logger = logging.getLogger(__name__)

class VideoProcessor:
    @staticmethod
    def process_video(
        input_path: str,
        output_path: str,
        tracker: OptimizedOpticalFlowTracker,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> bool:
        """
        Process a video file using the tracker.
        
        Args:
            input_path: Path to input video
            output_path: Path to save processed video
            tracker: Instance of OptimizedOpticalFlowTracker
            progress_callback: Optional callback function for progress updates
        
        Returns:
            bool: True if processing successful, False otherwise
        """
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                logger.error(f"Could not open video source: {input_path}")
                return False
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            tracker.total_frames = total_frames
            
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                processed_frame = tracker.detect_and_track(frame)
                out.write(processed_frame)
                
                frame_count += 1
                if progress_callback:
                    progress = min(100, int(frame_count / total_frames * 100))
                    progress_callback(progress)
            
            cap.release()
            out.release()
            return True
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return False
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()