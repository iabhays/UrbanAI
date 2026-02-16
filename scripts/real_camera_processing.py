#!/usr/bin/env python3
"""
Real-time Camera Processing with Person Detection and Risk Assessment.

This script processes live camera feeds with actual YOLO detection.
Features:
- Real-time person detection using YOLOv8
- Bounding box visualization
- Risk percentage calculation
- Person tracking
- Behavior analysis
- WebSocket streaming to dashboard
"""

import cv2
import numpy as np
import asyncio
import websockets
import json
from typing import Dict, List, Optional
import time
from dataclasses import dataclass, asdict
from loguru import logger
import argparse

# Add sentient_city to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from sentient_city.perception.pipeline import PerceptionPipeline
from sentient_city.perception.tracking import OCSortTracker


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
    alerts: List[str]
    fps: float


class RealCameraProcessor:
    """Real-time camera processor with YOLO detection."""
    
    def __init__(self, camera_source: str, camera_id: str = "camera_1"):
        """
        Initialize the camera processor.
        
        Args:
            camera_source: Camera source (camera index, RTSP URL, or video file)
            camera_id: Unique camera identifier
        """
        self.camera_source = camera_source
        self.camera_id = camera_id
        
        # Initialize perception pipeline
        self.tracker = OCSortTracker()
        self.pipeline = PerceptionPipeline(
            tracker=self.tracker,
            camera_id=camera_id
        )
        
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
    
    def calculate_risk_percentage(
        self, 
        detections_count: int, 
        crowd_density: float, 
        alerts_count: int
    ) -> tuple[float, str]:
        """
        Calculate real-time risk percentage.
        
        Args:
            detections_count: Number of person detections in current frame
            crowd_density: Current frame crowd density (0-1)
            alerts_count: Number of alerts in current frame
        
        Returns:
            Tuple of (risk_percentage, risk_level)
        """
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
    
    def initialize_camera(self) -> bool:
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
    
    def calculate_fps(self) -> float:
        """Calculate current FPS."""
        current_time = time.time()
        self.fps_frames += 1
        
        if current_time - self.last_fps_time >= 1.0:  # Update every second
            fps = self.fps_frames / (current_time - self.last_fps_time)
            self.fps_frames = 0
            self.last_fps_time = current_time
            return fps
        
        return 0.0  # Return 0 if not enough time passed
    
    def process_frame(self, frame: np.ndarray) -> tuple[CameraFrameResult, np.ndarray]:
        """
        Process a single frame from camera.
        
        Args:
            frame: Input frame from camera
        
        Returns:
            Tuple of (frame_result, processed_frame_with_bboxes)
        """
        self.frame_count += 1
        
        # Process frame with detection and bounding boxes
        results = self.pipeline.process_frame(
            frame, 
            frame_id=self.frame_count,
            draw_bboxes=True
        )
        
        # Extract metrics
        detections = results.get("detections", [])
        tracks = results.get("tracks", [])
        alerts = results.get("alerts", [])
        processed_frame = results.get("processed_frame", frame)
        
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
            len(alerts)
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
            alerts=[alert.get('type', 'unknown') for alert in alerts],
            fps=fps
        )
        
        return frame_result, processed_frame
    
    async def stream_to_websocket(self, websocket_url: str = "ws://localhost:8765"):
        """Stream results to WebSocket for dashboard."""
        try:
            async with websockets.connect(websocket_url) as websocket:
                logger.info(f"Connected to WebSocket: {websocket_url}")
                
                while True:
                    if not self.cap or not self.cap.isOpened():
                        await asyncio.sleep(1)
                        continue
                    
                    ret, frame = self.cap.read()
                    if not ret:
                        await asyncio.sleep(0.1)
                        continue
                    
                    # Process frame
                    result, processed_frame = self.process_frame(frame)
                    
                    # Send result to WebSocket
                    await websocket.send(json.dumps(asdict(result)))
                    
                    # Control frame rate (target 10 FPS for streaming)
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"WebSocket streaming error: {e}")
    
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


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Real-time Camera Processing")
    parser.add_argument("--source", default="0", 
                       help="Camera source (index, RTSP URL, or video file)")
    parser.add_argument("--camera-id", default="camera_1",
                       help="Camera identifier")
    parser.add_argument("--mode", choices=["display", "stream"], default="display",
                       help="Processing mode: display or stream")
    parser.add_argument("--websocket", default="ws://localhost:8765",
                       help="WebSocket URL for streaming mode")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = RealCameraProcessor(args.source, args.camera_id)
    
    if args.mode == "display":
        processor.run_display_mode()
    elif args.mode == "stream":
        await processor.stream_to_websocket(args.websocket)


if __name__ == "__main__":
    asyncio.run(main())
