"""
Pose detection using MediaPipe or OpenPose.

Extracts human skeleton keypoints for behavior analysis.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import mediapipe as mp
from loguru import logger

from ...utils.config import get_config


class PoseDetector:
    """
    Pose detection using MediaPipe.
    
    Extracts 33 body keypoints for each person in the frame.
    """
    
    # MediaPipe pose landmark indices
    LANDMARK_INDICES = {
        "nose": 0,
        "left_eye": 2,
        "right_eye": 5,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28
    }
    
    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize pose detector.
        
        Args:
            model_complexity: MediaPipe model complexity (0, 1, or 2)
            min_detection_confidence: Minimum detection confidence
            min_tracking_confidence: Minimum tracking confidence
        """
        self.config = get_config()
        pose_config = self.config.get_section("pose")
        
        self.model_complexity = pose_config.get("model_complexity", model_complexity)
        self.min_detection_confidence = pose_config.get(
            "min_detection_confidence", min_detection_confidence
        )
        self.min_tracking_confidence = pose_config.get(
            "min_tracking_confidence", min_tracking_confidence
        )
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect poses in image.
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            List of pose dictionaries, each containing:
            - keypoints: List of (x, y, visibility) tuples
            - landmarks: MediaPipe landmarks object
            - bbox: Bounding box [x1, y1, x2, y2]
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.pose.process(rgb_image)
        
        poses = []
        
        if results.pose_landmarks:
            h, w = image.shape[:2]
            
            # Extract keypoints
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                visibility = landmark.visibility
                keypoints.append((x, y, visibility))
            
            # Calculate bounding box
            visible_points = [kp for kp in keypoints if kp[2] > 0.5]
            if visible_points:
                xs = [p[0] for p in visible_points]
                ys = [p[1] for p in visible_points]
                bbox = [
                    max(0, min(xs) - 10),
                    max(0, min(ys) - 10),
                    min(w, max(xs) + 10),
                    min(h, max(ys) + 10)
                ]
            else:
                bbox = [0, 0, w, h]
            
            poses.append({
                "keypoints": keypoints,
                "landmarks": results.pose_landmarks,
                "bbox": bbox,
                "num_keypoints": len([kp for kp in keypoints if kp[2] > 0.5])
            })
        
        return poses
    
    def get_keypoint(self, pose: Dict, landmark_name: str) -> Optional[Tuple[int, int, float]]:
        """
        Get specific keypoint from pose.
        
        Args:
            pose: Pose dictionary
            landmark_name: Name of landmark (e.g., "left_shoulder")
        
        Returns:
            (x, y, visibility) tuple or None
        """
        if landmark_name not in self.LANDMARK_INDICES:
            return None
        
        idx = self.LANDMARK_INDICES[landmark_name]
        keypoints = pose["keypoints"]
        
        if idx < len(keypoints):
            return keypoints[idx]
        return None
    
    def calculate_angle(
        self,
        pose: Dict,
        point1: str,
        point2: str,
        point3: str
    ) -> Optional[float]:
        """
        Calculate angle between three keypoints.
        
        Args:
            pose: Pose dictionary
            point1: First point name
            point2: Middle point name (vertex)
            point3: Third point name
        
        Returns:
            Angle in degrees or None
        """
        p1 = self.get_keypoint(pose, point1)
        p2 = self.get_keypoint(pose, point2)
        p3 = self.get_keypoint(pose, point3)
        
        if p1 is None or p2 is None or p3 is None:
            return None
        
        # Calculate vectors
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180 / np.pi
        
        return angle
    
    def draw_pose(self, image: np.ndarray, pose: Dict) -> np.ndarray:
        """
        Draw pose landmarks on image.
        
        Args:
            image: Input image
            pose: Pose dictionary
        
        Returns:
            Image with drawn landmarks
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if pose["landmarks"]:
            self.mp_drawing.draw_landmarks(
                rgb_image,
                pose["landmarks"],
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2
                ),
                self.mp_drawing.DrawingSpec(
                    color=(0, 0, 255), thickness=2
                )
            )
        
        return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    
    def extract_features(self, pose: Dict) -> np.ndarray:
        """
        Extract feature vector from pose.
        
        Args:
            pose: Pose dictionary
        
        Returns:
            Feature vector
        """
        keypoints = pose["keypoints"]
        
        # Extract normalized keypoint coordinates
        features = []
        for kp in keypoints:
            features.extend([kp[0], kp[1], kp[2]])
        
        return np.array(features)
