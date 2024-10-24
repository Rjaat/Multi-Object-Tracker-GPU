# tracker/optical_flow.py
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import defaultdict
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedOpticalFlowTracker:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.6):
        # Check for GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize YOLO model
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.conf_threshold = conf_threshold
        
        # Parameters for optical flow
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            minEigThreshold=0.001
        )
        
        # Track history
        self.track_history = defaultdict(lambda: [])
        self.flow_points = defaultdict(lambda: [])
        self.prev_gray = None
        
        # FPS calculation
        self.prev_time = 0
        self.fps = 0
        self.fps_history = []
        self.fps_avg_frame_count = 30
        
        # Progress tracking
        self.total_frames = 0
        self.processed_frames = 0
    
    def update_fps(self):
        current_time = time.time()
        if self.prev_time:
            self.fps = 1 / (current_time - self.prev_time)
            self.fps_history.append(self.fps)
            if len(self.fps_history) > self.fps_avg_frame_count:
                self.fps_history.pop(0)
        self.prev_time = current_time
    
    def get_avg_fps(self):
        return sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0

    def calculate_optical_flow(self, gray, old_points):
        try:
            if not old_points or len(old_points) < 1:
                return []

            old_points = np.array(old_points, dtype=np.float32)
            if old_points.size == 0:
                return []

            old_points = old_points.reshape(-1, 1, 2)

            new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray,
                gray,
                old_points,
                None,
                **self.lk_params
            )

            if new_points is not None:
                good_new = []
                for i, (new, stat) in enumerate(zip(new_points, status)):
                    if stat[0]:
                        good_new.append(tuple(new.ravel()))
                return good_new
            return []

        except Exception as e:
            logger.error(f"Optical flow calculation error: {str(e)}")
            return []

    def detect_and_track(self, frame):
        self.processed_frames += 1
        self.update_fps()
        
        progress = (self.processed_frames / self.total_frames * 100) if self.total_frames > 0 else 0
        logger.info(f"Processing: {progress:.1f}% complete | FPS: {self.fps:.1f}")
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = self.model.track(frame, persist=True, conf=self.conf_threshold)
        
        if len(results) == 0:
            return self.add_fps_display(frame)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                center_point = (float(x), float(y))
                
                track = self.track_history[track_id]
                track.append(center_point)
                if len(track) > 30:
                    track.pop(0)
                
                if self.prev_gray is not None and track_id in self.flow_points:
                    current_points = self.flow_points[track_id]
                    if current_points:
                        good_new = self.calculate_optical_flow(gray, current_points)
                        self.flow_points[track_id] = good_new
                
                self.flow_points[track_id].append(center_point)
                if len(self.flow_points[track_id]) > 10:
                    self.flow_points[track_id].pop(0)
        
        self.prev_gray = gray.copy()
        return self.visualize_tracking(frame, results[0])
    
    def add_fps_display(self, frame):
        avg_fps = self.get_avg_fps()
        device_info = "GPU" if torch.cuda.is_available() else "CPU"
        
        cv2.rectangle(frame, (10, 10), (250, 90), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Avg FPS: {avg_fps:.1f}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Device: {device_info}", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def visualize_tracking(self, frame, result):
        annotated_frame = result.plot()
        
        if result.boxes.id is not None:
            boxes = result.boxes.xywh.cpu()
            track_ids = result.boxes.id.int().cpu().tolist()
            
            for box, track_id in zip(boxes, track_ids):
                track = self.track_history[track_id]
                if len(track) > 1:
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], False, (0, 255, 0), 2)
                
                flow_track = self.flow_points[track_id]
                if len(flow_track) > 1:
                    for i in range(len(flow_track) - 1):
                        pt1 = tuple(map(int, flow_track[i]))
                        pt2 = tuple(map(int, flow_track[i + 1]))
                        cv2.line(annotated_frame, pt1, pt2, (255, 0, 0), 1)
        
        return self.add_fps_display(annotated_frame)