"""
Enhanced Person Detector with Skeleton Wireframe Support.

This module extends the base person detector to include MediaPipe pose estimation
for skeleton wireframe detection and enhanced person analysis.
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import mediapipe as mp
from ultralytics import YOLO

from .person_detector import PersonDetector, Detection


@dataclass
class SkeletonDetection:
    """Enhanced detection with skeleton keypoints."""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    keypoints: np.ndarray  # MediaPipe pose landmarks (33, 3) - (x, y, visibility)
    keypoints_confidence: np.ndarray  # Confidence scores for each keypoint
    skeleton_connections: List[Tuple[int, int]]  # Connections between keypoints
    class_id: int = 0
    class_name: str = "person"
    center: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.center is None:
            self.center = np.array([
                (self.bbox[0] + self.bbox[2]) / 2,
                (self.bbox[1] + self.bbox[3]) / 2
            ])


class EnhancedPersonDetector:
    """
    Enhanced person detector with skeleton wireframe capabilities.
    
    Features:
    - YOLOv8-based person detection
    - MediaPipe pose estimation for skeleton detection
    - Combined confidence scoring
    - Optimized for real-time processing
    - Skeleton visualization support
    """
    
    # MediaPipe pose connections for visualization
    POSE_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 7),  # Head to body
        (0, 4), (4, 5), (5, 6), (6, 8),  # Head to body (other side)
        (9, 10),  # Shoulders
        (11, 12),  # Hips
        (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),  # Left arm
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),  # Right arm
        (11, 23), (12, 24),  # Torso
        (23, 24),  # Hips connection
        (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),  # Left leg
        (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),  # Right leg
    ]
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        pose_confidence_threshold: float = 0.5,
        enable_skeleton: bool = True,
        device: Optional[str] = None,
        max_detections: int = 500
    ):
        """
        Initialize enhanced person detector.
        
        Args:
            model_path: Path to YOLO model
            confidence_threshold: Minimum confidence for person detection
            pose_confidence_threshold: Minimum confidence for pose detection
            enable_skeleton: Whether to enable skeleton detection
            device: Device to run inference on
            max_detections: Maximum number of detections to process
        """
        self.confidence_threshold = confidence_threshold
        self.pose_confidence_threshold = pose_confidence_threshold
        self.enable_skeleton = enable_skeleton
        self.max_detections = max_detections
        
        # Initialize base person detector
        self.person_detector = PersonDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device=device,
            max_detections=max_detections
        )
        
        # Initialize MediaPipe pose estimation
        if self.enable_skeleton:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,  # Balance between speed and accuracy
                enable_segmentation=False,
                min_detection_confidence=pose_confidence_threshold,
                min_tracking_confidence=0.5
            )
            
            print("MediaPipe pose estimation initialized")
        
        # Performance metrics
        self.frame_count = 0
        self.total_detections = 0
        self.total_skeletons_detected = 0
        
        print("Enhanced Person Detector initialized successfully")
    
    def detect(
        self,
        frame: np.ndarray,
        frame_id: Optional[int] = None
    ) -> List[SkeletonDetection]:
        """
        Detect persons with skeleton wireframes in frame.
        
        Args:
            frame: Input frame (BGR format)
            frame_id: Optional frame ID for tracking
            
        Returns:
            List of enhanced person detections with skeletons
        """
        if frame_id is None:
            frame_id = self.frame_count
            
        self.frame_count += 1
        
        try:
            # Step 1: Detect persons using YOLO
            person_detections = self.person_detector.detect(frame, frame_id)
            
            if not person_detections:
                return []
            
            # Step 2: Extract skeleton wireframes for each detection
            skeleton_detections = []
            
            if self.enable_skeleton:
                skeleton_detections = self._extract_skeletons(frame, person_detections)
            else:
                # Convert to skeleton detections without keypoints
                for detection in person_detections:
                    skeleton_detection = SkeletonDetection(
                        bbox=detection.bbox,
                        confidence=detection.confidence,
                        keypoints=np.array([]),
                        keypoints_confidence=np.array([]),
                        skeleton_connections=[],
                        center=detection.center
                    )
                    skeleton_detections.append(skeleton_detection)
            
            self.total_detections += len(skeleton_detections)
            self.total_skeletons_detected += sum(1 for d in skeleton_detections if len(d.keypoints) > 0)
            
            print(f"Frame {frame_id}: Detected {len(skeleton_detections)} persons with skeletons")
            
            return skeleton_detections
            
        except Exception as e:
            print(f"Error in enhanced person detection: {e}")
            return []
    
    def _extract_skeletons(
        self,
        frame: np.ndarray,
        person_detections: List[Detection]
    ) -> List[SkeletonDetection]:
        """
        Extract skeleton wireframes for detected persons.
        
        Args:
            frame: Input frame
            person_detections: List of person detections from YOLO
            
        Returns:
            List of skeleton detections
        """
        skeleton_detections = []
        
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process pose estimation on the entire frame
        pose_results = self.pose.process(frame_rgb)
        
        if pose_results.pose_landmarks:
            # Get all pose landmarks
            landmarks = pose_results.pose_landmarks.landmark
            h, w = frame.shape[:2]
            
            # Convert landmarks to numpy array
            keypoints = np.array([[lm.x * w, lm.y * h, lm.visibility] for lm in landmarks])
            keypoints_confidence = np.array([lm.visibility for lm in landmarks])
            
            # Match skeletons to person detections
            matched_skeletons = self._match_skeletons_to_detections(
                keypoints, person_detections
            )
            
            for detection, skeleton_keypoints, skeleton_confidence in matched_skeletons:
                skeleton_detection = SkeletonDetection(
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    keypoints=skeleton_keypoints,
                    keypoints_confidence=skeleton_confidence,
                    skeleton_connections=self.POSE_CONNECTIONS,
                    center=detection.center
                )
                skeleton_detections.append(skeleton_detection)
        else:
            # No pose detected, return detections without skeletons
            for detection in person_detections:
                skeleton_detection = SkeletonDetection(
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    keypoints=np.array([]),
                    keypoints_confidence=np.array([]),
                    skeleton_connections=[],
                    center=detection.center
                )
                skeleton_detections.append(skeleton_detection)
        
        return skeleton_detections
    
    def _match_skeletons_to_detections(
        self,
        keypoints: np.ndarray,
        detections: List[Detection]
    ) -> List[Tuple[Detection, np.ndarray, np.ndarray]]:
        """
        Match skeleton keypoints to person detections.
        
        Args:
            keypoints: MediaPipe pose landmarks (33, 3)
            detections: List of person detections
            
        Returns:
            List of (detection, keypoints, confidence) tuples
        """
        matched = []
        
        if len(detections) == 0:
            return matched
        
        # Calculate skeleton center point (using visible keypoints)
        visible_keypoints = keypoints[keypoints[:, 2] > 0.5]  # Visibility > 0.5
        if len(visible_keypoints) > 0:
            skeleton_center = np.mean(visible_keypoints[:, :2], axis=0)
        else:
            skeleton_center = np.mean(keypoints[:, :2], axis=0)
        
        # Find the best matching detection
        best_detection = None
        min_distance = float('inf')
        
        for detection in detections:
            # Calculate distance between skeleton center and detection center
            distance = np.linalg.norm(skeleton_center - detection.center)
            
            # Check if skeleton is within detection bounds
            if (detection.bbox[0] <= skeleton_center[0] <= detection.bbox[2] and
                detection.bbox[1] <= skeleton_center[1] <= detection.bbox[3]):
                
                if distance < min_distance:
                    min_distance = distance
                    best_detection = detection
        
        # If we found a match, add it
        if best_detection is not None:
            matched.append((best_detection, keypoints, keypoints[:, 2]))
        else:
            # No match found, add detections without skeletons
            for detection in detections:
                empty_keypoints = np.array([])
                empty_confidence = np.array([])
                matched.append((detection, empty_keypoints, empty_confidence))
        
        return matched
    
    def draw_skeleton(
        self,
        frame: np.ndarray,
        skeleton_detection: SkeletonDetection,
        bbox_color: Tuple[int, int, int] = (0, 255, 0),
        skeleton_color: Tuple[int, int, int] = (255, 0, 0),
        keypoint_color: Tuple[int, int, int] = (0, 0, 255),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw skeleton wireframe and bounding box on frame.
        
        Args:
            frame: Input frame
            skeleton_detection: Skeleton detection result
            bbox_color: Color for bounding box (BGR)
            skeleton_color: Color for skeleton connections (BGR)
            keypoint_color: Color for keypoints (BGR)
            thickness: Line thickness
            
        Returns:
            Frame with drawn skeleton and bounding box
        """
        annotated_frame = frame.copy()
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, skeleton_detection.bbox)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), bbox_color, thickness)
        
        # Draw confidence score
        conf_text = f"Person: {skeleton_detection.confidence:.2f}"
        cv2.putText(annotated_frame, conf_text, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 1)
        
        # Draw skeleton if available
        if len(skeleton_detection.keypoints) > 0:
            keypoints = skeleton_detection.keypoints
            
            # Draw skeleton connections
            for connection in skeleton_detection.skeleton_connections:
                start_idx, end_idx = connection
                
                if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                    keypoints[start_idx, 2] > 0.5 and keypoints[end_idx, 2] > 0.5):
                    
                    start_point = (int(keypoints[start_idx, 0]), int(keypoints[start_idx, 1]))
                    end_point = (int(keypoints[end_idx, 0]), int(keypoints[end_idx, 1]))
                    
                    cv2.line(annotated_frame, start_point, end_point, 
                            skeleton_color, thickness)
            
            # Draw keypoints
            for i, keypoint in enumerate(keypoints):
                if keypoint[2] > 0.5:  # Visible keypoints
                    center = (int(keypoint[0]), int(keypoint[1]))
                    cv2.circle(annotated_frame, center, 3, keypoint_color, -1)
        
        return annotated_frame
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics."""
        base_metrics = self.person_detector.get_performance_metrics()
        
        skeleton_success_rate = 0.0
        if self.total_detections > 0:
            skeleton_success_rate = self.total_skeletons_detected / self.total_detections
        
        enhanced_metrics = {
            **base_metrics,
            "total_skeletons_detected": self.total_skeletons_detected,
            "skeleton_success_rate": skeleton_success_rate,
            "skeleton_enabled": self.enable_skeleton
        }
        
        return enhanced_metrics
    
    def reset_metrics(self):
        """Reset performance metrics."""
        self.frame_count = 0
        self.total_detections = 0
        self.total_skeletons_detected = 0
        self.person_detector.reset_metrics()
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'pose'):
            self.pose.close()
