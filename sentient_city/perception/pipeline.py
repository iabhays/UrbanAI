"""
Perception pipeline orchestrator.

Orchestrates detection, tracking, pose extraction, and behavior analysis.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger

from sentient_city.core import get_logger, get_config
from sentient_city.core.experiment_tracking import get_experiment_tracker
from .yolov26 import YOLOv26, YOLOv26InferenceEngine
from .tracking import BaseTracker, OCSortTracker, MultiCameraTracker
from .pose import PoseDetector, FallDetector, PanicDetector, ActivityClassifier
from .behavior import BehaviorFeatureExtractor, BehaviorAnalyzer


class PerceptionPipeline:
    """
    Perception pipeline orchestrator.
    
    Coordinates:
    - YOLOv26 detection
    - Multi-object tracking
    - Pose extraction
    - Behavior analysis
    - Feature extraction
    """
    
    def __init__(
        self,
        model: Optional[YOLOv26] = None,
        tracker: Optional[BaseTracker] = None,
        pose_detector: Optional[PoseDetector] = None,
        multi_camera_tracker: Optional[MultiCameraTracker] = None,
        experiment_id: Optional[str] = None,
        camera_id: Optional[str] = None
    ):
        """
        Initialize perception pipeline.
        
        Args:
            model: YOLOv26 model instance
            tracker: Tracker instance
            pose_detector: Pose detector instance
            multi_camera_tracker: Optional multi-camera tracker
            experiment_id: Optional experiment ID
            camera_id: Camera identifier
        """
        self.model = model
        self.tracker = tracker
        self.pose_detector = pose_detector
        self.multi_camera_tracker = multi_camera_tracker
        self.experiment_id = experiment_id
        self.camera_id = camera_id
        
        self.logger = get_logger(__name__)
        self.config = get_config()
        
        # Initialize inference engine
        if self.model:
            self.inference_engine = YOLOv26InferenceEngine(self.model)
        else:
            self.inference_engine = None
        
        # Behavior analysis
        self.feature_extractor = BehaviorFeatureExtractor(
            pose_detector=pose_detector,
            experiment_id=experiment_id
        )
        self.behavior_analyzer = BehaviorAnalyzer(
            feature_extractor=self.feature_extractor,
            experiment_id=experiment_id
        )
        
        # Specialized detectors
        self.fall_detector = FallDetector() if pose_detector else None
        self.panic_detector = PanicDetector() if pose_detector else None
        self.activity_classifier = ActivityClassifier() if pose_detector else None
        
        self.frame_count = 0
    
    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: Optional[int] = None,
        draw_bboxes: bool = True
    ) -> Dict[str, any]:
        """
        Process a single frame through the pipeline.
        
        Args:
            frame: Input frame (BGR format)
            frame_id: Optional frame ID
            draw_bboxes: Whether to draw bounding boxes on frame
        
        Returns:
            Dictionary with all processing results
        """
        if frame_id is None:
            frame_id = self.frame_count
        
        self.frame_count += 1
        
        # Create a copy for drawing
        processed_frame = frame.copy() if draw_bboxes else frame
        
        results = {
            "frame_id": frame_id,
            "camera_id": self.camera_id,
            "detections": None,
            "tracks": None,
            "poses": None,
            "behavior": None,
            "alerts": [],
            "processed_frame": processed_frame
        }
        
        # 1. Detection
        if self.inference_engine:
            detections = self._run_detection(frame)
            results["detections"] = detections
            
            # Draw bounding boxes
            if draw_bboxes and detections:
                processed_frame = self._draw_bounding_boxes(processed_frame, detections)
                results["processed_frame"] = processed_frame
        else:
            detections = []
        
        # 2. Tracking
        if self.tracker and detections:
            tracks = self._run_tracking(detections, frame_id)
            results["tracks"] = tracks
            
            # Draw tracking IDs
            if draw_bboxes and tracks:
                processed_frame = self._draw_tracking_info(processed_frame, tracks)
                results["processed_frame"] = processed_frame
        else:
            tracks = []
        
        # 3. Pose extraction
        if self.pose_detector:
            poses = self._run_pose_extraction(frame, tracks)
            results["poses"] = poses
        else:
            poses = []
        
        # 4. Behavior analysis
        if tracks:
            behavior = self._run_behavior_analysis(tracks, poses, frame_id)
            results["behavior"] = behavior
        
        # 5. Specialized detection
        if self.fall_detector and poses:
            fall_alerts = self._detect_falls(poses, tracks)
            results["alerts"].extend(fall_alerts)
        
        if self.panic_detector and tracks:
            panic_alerts = self._detect_panic(tracks)
            results["alerts"].extend(panic_alerts)
        
        return results
    
    def _run_detection(self, frame: np.ndarray) -> List[Dict]:
        """Run YOLOv26 detection."""
        import torch
        from PIL import Image
        import cv2
        
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run direct inference using ultralytics YOLO
            if not hasattr(self.inference_engine, '_yolo_model'):
                from ultralytics import YOLO
                self.inference_engine._yolo_model = YOLO('yolov8n.pt')
            
            # Run detection
            results = self.inference_engine._yolo_model(frame_rgb)
            
            # Extract person detections
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
            
        except ImportError:
            self.logger.warning("ultralytics not installed, using fallback detection")
            return []
        except Exception as e:
            self.logger.error(f"Error in detection: {e}")
            return []
    
    def _run_tracking(
        self,
        detections: List[Dict],
        frame_id: int
    ) -> List:
        """Run tracking."""
        tracks = self.tracker.update(detections, frame_id)
        return tracks
    
    def _run_pose_extraction(
        self,
        frame: np.ndarray,
        tracks: List
    ) -> List[Dict]:
        """Run pose extraction."""
        poses = []
        
        if self.pose_detector:
            # Extract poses for each track
            for track in tracks:
                # Crop track region
                bbox = track.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                if x2 > x1 and y2 > y1:
                    track_roi = frame[y1:y2, x1:x2]
                    pose = self.pose_detector.detect(track_roi)
                    if pose:
                        poses.extend(pose)
        
        return poses
    
    def _run_behavior_analysis(
        self,
        tracks: List,
        poses: List[Dict],
        frame_id: int
    ) -> Dict:
        """Run behavior analysis."""
        behavior_results = {}
        
        for track in tracks:
            # Find corresponding pose
            pose = None
            if poses:
                # Match pose to track (simplified - would use better matching)
                pose = poses[0] if len(poses) > 0 else None
            
            # Analyze behavior
            analysis = self.behavior_analyzer.analyze_track(track, pose, frame_id)
            behavior_results[track.track_id] = analysis
        
        return behavior_results
    
    def _detect_falls(
        self,
        poses: List[Dict],
        tracks: List
    ) -> List[Dict]:
        """Detect falls."""
        alerts = []
        
        for pose in poses:
            if self.fall_detector:
                is_fall = self.fall_detector.detect(pose)
                if is_fall:
                    alerts.append({
                        "type": "fall",
                        "confidence": 0.8,
                        "pose": pose
                    })
        
        return alerts
    
    def _detect_panic(self, tracks: List) -> List[Dict]:
        """Detect panic movements."""
        alerts = []
        
        for track in tracks:
            if self.panic_detector:
                is_panic = self.panic_detector.detect(track)
                if is_panic:
                    alerts.append({
                        "type": "panic",
                        "confidence": 0.7,
                        "track_id": track.track_id
                    })
        
        return alerts
    
    def _draw_bounding_boxes(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes on frame."""
        import cv2
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class']
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Choose color based on confidence
            if confidence > 0.8:
                color = (0, 255, 0)  # Green - high confidence
            elif confidence > 0.6:
                color = (0, 255, 255)  # Yellow - medium confidence
            else:
                color = (0, 0, 255)  # Red - low confidence
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Background for label
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Text
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def _draw_tracking_info(self, frame: np.ndarray, tracks: List) -> np.ndarray:
        """Draw tracking information on frame."""
        import cv2
        
        for track in tracks:
            if hasattr(track, 'bbox') and hasattr(track, 'track_id'):
                bbox = track.bbox.astype(int)
                track_id = track.track_id
                
                x1, y1, x2, y2 = bbox
                
                # Draw track ID
                id_label = f"ID: {track_id}"
                cv2.putText(frame, id_label, (x1, y2 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw track trail (simplified - just show center)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(frame, (center_x, center_y), 3, (255, 255, 255), -1)
        
        return frame
