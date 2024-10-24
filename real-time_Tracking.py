import cv2
from ultralytics import YOLO
import time
import os
import sys
import numpy as np

# Fix for Qt platform plugin error
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["OPENCV_VIDEOIO_PRIORITY_BACKEND"] = "2"

class ObjectTracker:
    def __init__(self):
        self.tracks = {}  # Store tracking history
        self.max_track_length = 30  # Maximum number of points to keep in track
        self.prev_gray = None
        self.prev_points = None
        self.track_colors = {}  # Store colors for each track

    def get_track_color(self, track_id):
        if track_id not in self.track_colors:
            # Generate random color for new track
            self.track_colors[track_id] = tuple(map(int, np.random.randint(0, 255, 3)))
        return self.track_colors[track_id]

    def update_tracks(self, frame, boxes):
        # Convert current frame to grayscale for optical flow
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Get current points from boxes
        current_points = []
        current_ids = []
        
        for box in boxes:
            if box.id is None:
                continue
            track_id = int(box.id[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Use center point of box
            center_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            current_points.append(center_point)
            current_ids.append(track_id)
            
            # Initialize new tracks
            if track_id not in self.tracks:
                self.tracks[track_id] = []
            
            # Add current point to track
            self.tracks[track_id].append(center_point)
            
            # Limit track length
            if len(self.tracks[track_id]) > self.max_track_length:
                self.tracks[track_id] = self.tracks[track_id][-self.max_track_length:]

        # Update previous frame
        self.prev_gray = gray
        return current_points, current_ids

    def draw_tracks(self, frame):
        # Draw tracks for each object
        for track_id, points in self.tracks.items():
            if len(points) < 2:
                continue
                
            color = self.get_track_color(track_id)
            
            # Draw track line
            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i + 1], color, 2)
            
            # Draw direction arrow
            if len(points) >= 2:
                last_point = points[-1]
                prev_point = points[-2]
                # Calculate direction vector
                direction = (last_point[0] - prev_point[0], last_point[1] - prev_point[1])
                # Draw arrow
                if abs(direction[0]) > 5 or abs(direction[1]) > 5:  # Only draw if movement is significant
                    cv2.arrowedLine(frame, prev_point, last_point, color, 2, tipLength=0.5)

def initialize_camera():
    """Initialize camera with debug information"""
    print("Attempting to open camera...")
    
    for camera_idx in [0, 2, 1]:
        print(f"Trying camera index {camera_idx}")
        cap = cv2.VideoCapture(camera_idx)
        
        if cap.isOpened():
            print(f"Successfully opened camera {camera_idx}")
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            ret, frame = cap.read()
            if ret:
                print("Successfully read test frame")
                return cap
            else:
                print(f"Could not read frame from camera {camera_idx}")
                cap.release()
        else:
            print(f"Could not open camera {camera_idx}")
    
    raise RuntimeError("No working camera found")

def main():
    try:
        print("Testing camera feed...")
        cap = initialize_camera()
        
        # Initialize object tracker
        tracker = ObjectTracker()
        
        # Test video display
        start_time = time.time()
        while time.time() - start_time < 3:
            ret, frame = cap.read()
            if ret:
                cv2.imshow("Camera Test", frame)
                cv2.waitKey(1)
            else:
                print("Error reading frame during test")
                return
        
        print("Camera test successful!")
        print("Loading YOLO model...")
        model = YOLO("yolo11n.pt")
        print("YOLO model loaded successfully!")
        
        frame_count = 0
        fps_start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Calculate FPS
            frame_count += 1
            if time.time() - fps_start_time >= 1.0:
                fps = frame_count / (time.time() - fps_start_time)
                print(f"FPS: {fps:.2f}")
                frame_count = 0
                fps_start_time = time.time()
            
            try:
                # Run YOLO tracking
                results = model.track(
                    frame,
                    persist=True,
                    conf=0.5,
                    iou=0.5,
                    verbose=False
                )
                
                if results and results[0].boxes:
                    # Create copy for annotation
                    annotated_frame = frame.copy()
                    boxes = results[0].boxes
                    
                    # Update object tracks
                    current_points, current_ids = tracker.update_tracks(frame, boxes)
                    
                    # Draw tracks
                    tracker.draw_tracks(annotated_frame)
                    
                    # Draw current detections
                    for box in boxes:
                        if box.id is None:
                            continue
                        
                        track_id = int(box.id[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        color = tracker.get_track_color(track_id)
                        
                        # Draw box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Add label with track ID
                        label = f"ID: {track_id}"
                        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Add FPS counter
                    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 40),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Show annotated frame
                    cv2.imshow("Real-time Tracking", annotated_frame)
                else:
                    cv2.imshow("Real-time Tracking", frame)
            
            except Exception as e:
                print(f"Error processing frame with YOLO: {e}")
                cv2.imshow("Real-time Tracking", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"Error occurred: {e}")
    
    finally:
        print("Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()