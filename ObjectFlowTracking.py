import cv2
import numpy as np
import time
import argparse

class OptimizedOpticalFlowTracker:
    def __init__(self):
        # Grid parameters
        self.grid_spacing = 20
        self.prev_gray = None
        
        # Farneback parameters for sub-pixel precision
        self.fb_params = dict(
            pyr_scale=0.5,
            levels=5,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        )
        
        # Visualization parameters
        self.min_flow_threshold = 0.5
        self.flow_scale = 2.0
        self.grid_color = (0, 255, 0)      # Green for grid
        self.flow_color = (0, 255, 255)    # Yellow for flow
        
        # Transparency settings
        self.overlay_alpha = 0.7
        
    def create_grid_points(self, shape):
        """Create a uniform grid of points"""
        h, w = shape
        y, x = np.mgrid[self.grid_spacing:h-self.grid_spacing:self.grid_spacing,
                       self.grid_spacing:w-self.grid_spacing:self.grid_spacing]
        return np.stack((x, y), axis=-1).reshape(-1, 2)
        
    def calculate_flow(self, frame_gray):
        """Calculate dense optical flow at sub-pixel precision"""
        if self.prev_gray is None:
            self.prev_gray = frame_gray.copy()
            return None
            
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            frame_gray,
            None,
            **self.fb_params
        )
        
        self.prev_gray = frame_gray.copy()
        return flow
    
    def create_flow_overlay(self, shape, flow):
        """Create a transparent overlay for flow visualization"""
        overlay = np.zeros(shape, dtype=np.uint8)
        h, w = flow.shape[:2]
        grid_points = self.create_grid_points((h, w))
        
        # Draw background grid
        for pt in grid_points:
            x, y = pt.astype(int)
            cv2.circle(overlay, (x, y), 1, self.grid_color, -1)
        
        # Draw flow vectors
        for pt in grid_points:
            x, y = pt.astype(int)
            
            # Ensure point is within flow bounds
            if y < flow.shape[0] and x < flow.shape[1]:
                flow_x = flow[y, x, 0]
                flow_y = flow[y, x, 1]
                magnitude = np.sqrt(flow_x**2 + flow_y**2)
                
                if magnitude > self.min_flow_threshold:
                    end_x = x + int(flow_x * self.flow_scale)
                    end_y = y + int(flow_y * self.flow_scale)
                    
                    # Draw flow vector
                    cv2.line(overlay, (x, y), (end_x, end_y), self.flow_color, 2)
                    
                    # Draw arrow head
                    angle = np.arctan2(flow_y, flow_x)
                    tip_length = 5
                    tip_x1 = end_x - tip_length * np.cos(angle + np.pi/6)
                    tip_y1 = end_y - tip_length * np.sin(angle + np.pi/6)
                    tip_x2 = end_x - tip_length * np.cos(angle - np.pi/6)
                    tip_y2 = end_y - tip_length * np.sin(angle - np.pi/6)
                    
                    cv2.line(overlay, (end_x, end_y), (int(tip_x1), int(tip_y1)), self.flow_color, 2)
                    cv2.line(overlay, (end_x, end_y), (int(tip_x2), int(tip_y2)), self.flow_color, 2)
        
        return overlay
    
    def process_frame(self, frame, fps):
        # Convert to grayscale for flow calculation
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Create a copy of the original frame
        output = frame.copy()
        
        # Calculate optical flow
        flow = self.calculate_flow(frame_gray)
        
        if flow is not None:
            # Create flow visualization overlay
            flow_overlay = self.create_flow_overlay(frame.shape, flow)
            
            # Blend the overlay with the original frame
            mask = (flow_overlay != 0).any(axis=2)
            output[mask] = cv2.addWeighted(output[mask], 1 - self.overlay_alpha, 
                                         flow_overlay[mask], self.overlay_alpha, 0)
            
            # Add FPS counter on a dark background for better visibility
            fps_text = f"FPS: {fps:.1f}"
            (text_width, text_height), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(output, (5, 5), (text_width + 15, text_height + 15), (0, 0, 0), -1)
            cv2.putText(output, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return output

def main(input_source, output_file, max_frames=None):
    # Initialize visualizer
    visualizer = OptimizedOpticalFlowTracker()
    
    # Open video source
    cap = cv2.VideoCapture(int(input_source) if input_source.isdigit() else input_source)
    if not cap.isOpened():
        print(f"Error: Could not open input source: {input_source}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate current FPS
            elapsed_time = time.time() - start_time
            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # Process frame
            processed_frame = visualizer.process_frame(frame, current_fps)
            out.write(processed_frame)
            
            frame_count += 1
            if max_frames and frame_count >= max_frames:
                break
            
            if frame_count % 30 == 0:
                print(f"\rProcessed {frame_count} frames. FPS: {current_fps:.2f}", end="")
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        cap.release()
        out.release()
        print(f"\nProcessing complete. Output saved as '{output_file}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optical Flow Grid Visualization")
    parser.add_argument("--input", default="0", help="Input source (0 for webcam or video path)")
    parser.add_argument("--output", default="output.mp4", help="Output video path")
    parser.add_argument("--max_frames", type=int, help="Maximum frames to process")
    args = parser.parse_args()
    
    main(args.input, args.output, args.max_frames)
