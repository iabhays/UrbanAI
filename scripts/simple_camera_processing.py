#!/usr/bin/env python3
"""
Simple Real-time Camera Processing with Person Detection and Risk Assessment.

This script processes live camera feeds with actual YOLO detection.
Features:
- Real-time person detection using YOLOv8
- Bounding box visualization
- Risk percentage calculation
- Person tracking
"""

import cv2
import numpy as np
import time
from dataclasses import dataclass
from loguru import logger
import argparse

try:
    from ultralytics import YOLO
except ImportError:
    logger.error("ultralytics not installed. Run: pip install ultralytics")
    exit(1)


@dataclass
class CameraFrameResult:
    """Results from processing a single camera frame."""
    camera_id: str
    timestamp: float
    frame_id: int
    detections_count: int
    tracks_count: int
    crowd_density: float
    risk_percentage: float
    risk_level: str
    alerts: list
    fps: float


class SimpleTrack:
    """Simple track representation."""
    
    def __init__(self, track_id: int, bbox: np.ndarray, confidence: float):
        self.track_id = track_id
        self.bbox = bbox
        self.confidence = confidence
        self.last_seen = time.time()


class SimpleTracker:
    """Simple person tracker using IoU matching."""
    
    def __init__(self, max_disappeared: int = 30):
        self.tracks = []
        self.next_id = 1
        self.max_disappeared = max_disappeared
    
    def update(self, detections: list) -> list:
        """Update tracks with new detections."""
        if not detections:
            return self.tracks
        
        # Convert detections to format
        det_bboxes = [np.array(d['bbox']) for d in detections]
        det_confs = [d['confidence'] for d in detections]
        
        # Simple IoU matching with existing tracks
        matched_tracks = []
        matched_detections = []
        
        if self.tracks:
            track_bboxes = [track.bbox for track in self.tracks]
            
            for i, det_bbox in enumerate(det_bboxes):
                best_iou = 0
                best_track_idx = -1
                
                for j, track_bbox in enumerate(track_bboxes):
                    iou = self._calculate_iou(det_bbox, track_bbox)
                    if iou > best_iou and iou > 0.3:  # IoU threshold
                        best_iou = iou
                        best_track_idx = j
                
                if best_track_idx >= 0:
                    # Update existing track
                    track = self.tracks[best_track_idx]
                    track.bbox = det_bbox
                    track.confidence = det_confs[i]
                    track.last_seen = time.time()
                    matched_tracks.append(track)
                    matched_detections.append(i)
        
        # Create new tracks for unmatched detections
        for i, det_bbox in enumerate(det_bboxes):
            if i not in matched_detections:
                new_track = SimpleTrack(self.next_id, det_bbox, det_confs[i])
                self.tracks.append(new_track)
                self.next_id += 1
        
        # Remove old tracks
        current_time = time.time()
        self.tracks = [t for t in self.tracks 
                     if current_time - t.last_seen < self.max_disappeared]
        
        return self.tracks
    
    def _calculate_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate IoU between two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


class SimpleCameraProcessor:
    """Simple real-time camera processor with YOLO detection."""
    
    def __init__(self, camera_source, camera_id="camera_1"):
        """
        Initialize the camera processor.
        
        Args:
            camera_source: Camera source (camera index, RTSP URL, or video file)
            camera_id: Unique camera identifier
        """
        self.camera_source = camera_source
        self.camera_id = camera_id
        
        # Initialize YOLO model
        logger.info("Loading YOLOv8 model...")
        self.yolo_model = YOLO('yolov8n.pt')
        
        # Initialize tracker
        self.tracker = SimpleTracker()
        
        # Camera setup
        self.cap = None
        self.frame_count = 0
        self.start_time = time.time()
        self.last_fps_time = time.time()
        self.fps_frames = 0
        
        # Risk calculation history
        self.detection_history = []
        self.alert_history = []
        self.max_history = 30  # Keep last 30 frames for risk calculation
        
        logger.info(f"Initialized camera processor for {camera_id}")
    
    def calculate_risk_percentage(self, detections_count: int, crowd_density: float, alerts_count: int):
        """Calculate real-time risk percentage."""
        # Add current frame to history
        self.detection_history.append(detections_count)
        self.alert_history.append(alerts_count)
        
        # Keep history limited
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)
        if len(self.alert_history) > self.max_history:
            self.alert_history.pop(0)
        
        # Calculate averages over history
        avg_detections = np.mean(self.detection_history) if self.detection_history else 0
        avg_alerts = np.mean(self.alert_history) if self.alert_history else 0
        
        # Base risk from crowd density
        density_risk = crowd_density * 40  # Max 40% from density
        
        # Risk from number of people (more people = higher risk)
        people_risk = min(avg_detections * 5, 30)  # Max 30% from people count
        
        # Risk from alerts (falls, panic, etc.)
        alert_risk = min(avg_alerts * 15, 30)  # Max 30% from alerts
        
        # Calculate total risk
        total_risk = density_risk + people_risk + alert_risk
        total_risk = min(total_risk, 100)  # Cap at 100%
        
        # Determine risk level
        if total_risk >= 80:
            risk_level = "Critical"
        elif total_risk >= 60:
            risk_level = "High"
        elif total_risk >= 40:
            risk_level = "Medium"
        elif total_risk >= 20:
            risk_level = "Low"
        else:
            risk_level = "Very Low"
        
        return round(total_risk, 1), risk_level
    
    def initialize_camera(self):
        """Initialize camera capture."""
        try:
            # Handle different camera sources
            if isinstance(self.camera_source, str) and self.camera_source.startswith(('http://', 'rtsp://', 'rtmp://')):
                # RTSP/Network camera
                self.cap = cv2.VideoCapture(self.camera_source)
                # Set buffer size for RTSP
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            elif isinstance(self.camera_source, int) or self.camera_source.isdigit():
                # USB camera index
                camera_idx = int(self.camera_source) if isinstance(self.camera_source, str) else self.camera_source
                self.cap = cv2.VideoCapture(camera_idx)
            else:
                # Video file or other source
                self.cap = cv2.VideoCapture(self.camera_source)
            
            if not self.cap.isOpened():
                logger.error(f"Could not open camera source: {self.camera_source}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            logger.info(f"Camera initialized: {self.camera_source}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            return False
    
    def calculate_fps(self):
        """Calculate current FPS."""
        current_time = time.time()
        self.fps_frames += 1
        
        if current_time - self.last_fps_time >= 1.0:  # Update every second
            fps = self.fps_frames / (current_time - self.last_fps_time)
            self.fps_frames = 0
            self.last_fps_time = current_time
            return fps
        
        return 0.0  # Return 0 if not enough time passed
    
    def detect_persons(self, frame: np.ndarray) -> list:
        """Detect persons in frame using YOLO."""
        results = self.yolo_model(frame)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Filter for persons only (class 0)
                    if int(box.cls) == 0:  # Person class
                        confidence = float(box.conf)
                        if confidence > 0.5:  # Confidence threshold
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            detections.append({
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'confidence': confidence,
                                'class': 'person',
                                'class_id': 0
                            })
        
        return detections
    
    def draw_bounding_boxes(self, frame: np.ndarray, detections: list, tracks: list) -> np.ndarray:
        """Draw bounding boxes and tracking info on frame."""
        processed_frame = frame.copy()
        
        # Draw detections
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Choose color based on confidence
            if confidence > 0.8:
                color = (0, 255, 0)  # Green - high confidence
            elif confidence > 0.6:
                color = (0, 255, 255)  # Yellow - medium confidence
            else:
                color = (0, 0, 255)  # Red - low confidence
            
            # Draw rectangle
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"Person: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Background for label
            cv2.rectangle(processed_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Text
            cv2.putText(processed_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw tracking IDs
        for track in tracks:
            x1, y1, x2, y2 = map(int, track.bbox)
            
            # Draw track ID
            id_label = f"ID: {track.track_id}"
            cv2.putText(processed_frame, id_label, (x1, y2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return processed_frame
    
    def process_frame(self, frame: np.ndarray):
        """Process a single frame from camera."""
        self.frame_count += 1
        
        # Detect persons
        detections = self.detect_persons(frame)
        
        # Update tracker
        tracks = self.tracker.update(detections)
        
        # Draw bounding boxes
        processed_frame = self.draw_bounding_boxes(frame, detections, tracks)
        
        # Calculate crowd density for this frame
        frame_area = frame.shape[0] * frame.shape[1]
        person_area = 0
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            person_area += (x2 - x1) * (y2 - y1)
        
        crowd_density = min(person_area / frame_area, 1.0) if frame_area > 0 else 0
        
        # Calculate risk
        risk_percentage, risk_level = self.calculate_risk_percentage(
            len(detections),
            crowd_density,
            0  # No alerts in simple version
        )
        
        # Calculate FPS
        fps = self.calculate_fps()
        
        # Create result
        frame_result = CameraFrameResult(
            camera_id=self.camera_id,
            timestamp=time.time(),
            frame_id=self.frame_count,
            detections_count=len(detections),
            tracks_count=len(tracks),
            crowd_density=round(crowd_density, 3),
            risk_percentage=risk_percentage,
            risk_level=risk_level,
            alerts=[],
            fps=fps
        )
        
        return frame_result, processed_frame
    
    def run_display_mode(self):
        """Run camera processing with display window."""
        if not self.initialize_camera():
            return
        
        logger.info("Starting camera display mode. Press 'q' to quit.")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Could not read frame from camera")
                    break
                
                # Process frame
                result, processed_frame = self.process_frame(frame)
                
                # Add info overlay
                info_text = [
                    f"Camera: {self.camera_id}",
                    f"People: {result.detections_count}",
                    f"Risk: {result.risk_percentage}% ({result.risk_level})",
                    f"FPS: {result.fps:.1f}"
                ]
                
                y_offset = 30
                for text in info_text:
                    cv2.putText(processed_frame, text, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y_offset += 25
                
                # Show frame
                cv2.imshow(f"Camera Feed - {self.camera_id}", processed_frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera processor cleaned up")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Simple Real-time Camera Processing")
    parser.add_argument("--source", default="0", 
                       help="Camera source (index, RTSP URL, or video file)")
    parser.add_argument("--camera-id", default="camera_1",
                       help="Camera identifier")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = SimpleCameraProcessor(args.source, args.camera_id)
    
    # Run display mode
    processor.run_display_mode()


if __name__ == "__main__":
    main()
