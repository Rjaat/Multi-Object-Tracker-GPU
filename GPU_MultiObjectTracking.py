import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time
import os
import torch

class OptimizedOpticalFlowTracker:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.6):
        # Check for GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
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
        
        # Track history for visualization
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
        """
        Calculate optical flow with error handling and point validation.
        """
        try:
            if not old_points or len(old_points) < 1:
                return []

            # Convert points to correct format
            old_points = np.array(old_points, dtype=np.float32)
            if old_points.size == 0:
                return []

            # Reshape points to required format (N, 1, 2)
            old_points = old_points.reshape(-1, 1, 2)

            # Calculate optical flow
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray,
                gray,
                old_points,
                None,
                **self.lk_params
            )

            # Filter only valid points
            if new_points is not None:
                good_new = []
                for i, (new, stat) in enumerate(zip(new_points, status)):
                    if stat[0]:
                        good_new.append(tuple(new.ravel()))
                return good_new
            return []

        except Exception as e:
            print(f"\nOptical flow calculation error: {str(e)}")
            return []

    def detect_and_track(self, frame):
        self.processed_frames += 1
        self.update_fps()
        
        # Calculate and display progress
        progress = (self.processed_frames / self.total_frames * 100) if self.total_frames > 0 else 0
        print(f"\rProcessing: {progress:.1f}% complete | FPS: {self.fps:.1f} | Device: {self.device}", end="")
        
        # Convert frame to grayscale for optical flow
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Run YOLO tracking
        results = self.model.track(frame, persist=True, conf=self.conf_threshold)
        
        if len(results) == 0:
            return self.add_fps_display(frame)
        
        # Get boxes and track IDs
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            # Process each detected object
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                center_point = (float(x), float(y))
                
                # Update track history
                track = self.track_history[track_id]
                track.append(center_point)
                if len(track) > 30:
                    track.pop(0)
                
                # Calculate optical flow if we have previous frame and points
                if self.prev_gray is not None and track_id in self.flow_points:
                    current_points = self.flow_points[track_id]
                    if current_points:  # Only calculate if we have points
                        good_new = self.calculate_optical_flow(gray, current_points)
                        self.flow_points[track_id] = good_new
                
                # Always add the current center point
                self.flow_points[track_id].append(center_point)
                if len(self.flow_points[track_id]) > 10:
                    self.flow_points[track_id].pop(0)
        
        # Update previous frame
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
                # Draw track history
                track = self.track_history[track_id]
                if len(track) > 1:
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], False, (0, 255, 0), 2)
                
                # Draw optical flow vectors
                flow_track = self.flow_points[track_id]
                if len(flow_track) > 1:
                    for i in range(len(flow_track) - 1):
                        pt1 = tuple(map(int, flow_track[i]))
                        pt2 = tuple(map(int, flow_track[i + 1]))
                        cv2.line(annotated_frame, pt1, pt2, (255, 0, 0), 1)
        
        return self.add_fps_display(annotated_frame)

def process_video(source_path, output_path, model_path="yolov8n.pt"):
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Initialize video capture
    cap = cv2.VideoCapture(int(source_path) if source_path.isdigit() else source_path)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize tracker
    tracker = OptimizedOpticalFlowTracker(model_path=model_path)
    tracker.total_frames = total_frames
    
    print(f"Processing video: {source_path}")
    print(f"Output will be saved to: {output_path}")
    print(f"Video properties: {width}x{height} @ {fps}fps")
    
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # Process frame
            processed_frame = tracker.detect_and_track(frame)
            
            # Write frame to output video
            out.write(processed_frame)
            
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    finally:
        # Cleanup
        cap.release()
        out.release()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\nProcessing complete!")
        print(f"Output saved to: {output_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='video source (0 for webcam or video path)')
    parser.add_argument('--output', type=str, default='output.mp4', help='output video path')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLO model path')
    args = parser.parse_args()
    
    process_video(args.source, args.output, args.model)

if __name__ == "__main__":
    main()
