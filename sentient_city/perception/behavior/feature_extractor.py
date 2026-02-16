"""
Behavior feature extraction.

Extracts behavior-aware features from tracks and poses for downstream analysis.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger

from sentient_city.core import get_logger, get_config
from ..tracking import Track
from ..pose import PoseDetector


class BehaviorFeatureExtractor:
    """
    Behavior feature extractor.
    
    Extracts features from tracks and poses for behavior analysis:
    - Movement patterns
    - Speed and acceleration
    - Pose-based features
    - Trajectory features
    - Interaction features
    """
    
    def __init__(
        self,
        pose_detector: Optional[PoseDetector] = None,
        experiment_id: Optional[str] = None
    ):
        """
        Initialize behavior feature extractor.
        
        Args:
            pose_detector: Optional pose detector for pose-based features
            experiment_id: Optional experiment ID for tracking
        """
        self.pose_detector = pose_detector
        self.experiment_id = experiment_id
        
        self.logger = get_logger(__name__)
        self.config = get_config()
    
    def extract_track_features(
        self,
        track: Track,
        frame_id: int = 0
    ) -> Dict[str, np.ndarray]:
        """
        Extract features from a track.
        
        Args:
            track: Track object
            frame_id: Current frame ID
        
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Basic track features
        features["bbox"] = track.bbox
        features["confidence"] = np.array([track.confidence])
        features["age"] = np.array([track.age])
        features["hits"] = np.array([track.hits])
        
        # Trajectory features
        if len(track.trajectory) > 0:
            trajectory = np.array(track.trajectory)
            features["trajectory"] = trajectory
            
            # Speed features
            if track.velocity is not None:
                features["velocity"] = track.velocity
                features["speed"] = np.array([np.linalg.norm(track.velocity)])
            else:
                features["velocity"] = np.array([0.0, 0.0])
                features["speed"] = np.array([0.0])
            
            # Trajectory statistics
            if len(trajectory) > 1:
                centers = np.array([self._get_center(bbox) for bbox in trajectory])
                features["trajectory_centers"] = centers
                features["trajectory_length"] = np.array([len(trajectory)])
                
                # Displacement
                displacement = centers[-1] - centers[0]
                features["displacement"] = displacement
                features["displacement_magnitude"] = np.array([np.linalg.norm(displacement)])
                
                # Direction
                if np.linalg.norm(displacement) > 0:
                    features["direction"] = displacement / np.linalg.norm(displacement)
                else:
                    features["direction"] = np.array([0.0, 0.0])
        else:
            features["trajectory"] = np.array([])
            features["velocity"] = np.array([0.0, 0.0])
            features["speed"] = np.array([0.0])
        
        # Embedding features
        if track.embedding is not None:
            features["embedding"] = track.embedding
        
        if track.reid_embedding is not None:
            features["reid_embedding"] = track.reid_embedding
        
        if track.behavior_features is not None:
            features["behavior_features"] = track.behavior_features
        
        return features
    
    def extract_pose_features(
        self,
        pose: Dict,
        bbox: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract features from pose keypoints.
        
        Args:
            pose: Pose dictionary with keypoints
            bbox: Optional bounding box for normalization
        
        Returns:
            Dictionary of pose features
        """
        features = {}
        
        keypoints = pose.get("keypoints", [])
        if len(keypoints) == 0:
            return features
        
        # Convert to numpy array
        kp_array = np.array(keypoints)  # [N, 3] (x, y, visibility)
        
        # Visible keypoints
        visible_mask = kp_array[:, 2] > 0.5
        visible_kps = kp_array[visible_mask, :2]
        
        features["keypoints"] = kp_array
        features["num_visible_keypoints"] = np.array([np.sum(visible_mask)])
        
        if len(visible_kps) > 0:
            # Bounding box of keypoints
            kp_bbox = np.array([
                np.min(visible_kps[:, 0]),
                np.min(visible_kps[:, 1]),
                np.max(visible_kps[:, 0]),
                np.max(visible_kps[:, 1])
            ])
            features["keypoint_bbox"] = kp_bbox
            
            # Center of keypoints
            kp_center = np.mean(visible_kps, axis=0)
            features["keypoint_center"] = kp_center
            
            # Keypoint spread (standard deviation)
            kp_std = np.std(visible_kps, axis=0)
            features["keypoint_spread"] = kp_std
            
            # Normalize if bbox provided
            if bbox is not None:
                bbox_w = bbox[2] - bbox[0]
                bbox_h = bbox[3] - bbox[1]
                if bbox_w > 0 and bbox_h > 0:
                    normalized_kps = (visible_kps - bbox[:2]) / np.array([bbox_w, bbox_h])
                    features["normalized_keypoints"] = normalized_kps
        
        # Extract specific keypoint features
        if self.pose_detector:
            # Head position
            nose = self.pose_detector.get_keypoint(pose, "nose")
            if nose:
                features["head_position"] = np.array([nose[0], nose[1]])
            
            # Shoulder positions
            left_shoulder = self.pose_detector.get_keypoint(pose, "left_shoulder")
            right_shoulder = self.pose_detector.get_keypoint(pose, "right_shoulder")
            if left_shoulder and right_shoulder:
                shoulder_center = np.array([
                    (left_shoulder[0] + right_shoulder[0]) / 2,
                    (left_shoulder[1] + right_shoulder[1]) / 2
                ])
                features["shoulder_center"] = shoulder_center
                features["shoulder_width"] = np.array([
                    np.linalg.norm(np.array([left_shoulder[0], left_shoulder[1]]) - 
                                 np.array([right_shoulder[0], right_shoulder[1]]))
                ])
        
        return features
    
    def extract_combined_features(
        self,
        track: Track,
        pose: Optional[Dict] = None,
        frame_id: int = 0
    ) -> Dict[str, np.ndarray]:
        """
        Extract combined features from track and pose.
        
        Args:
            track: Track object
            pose: Optional pose dictionary
            frame_id: Current frame ID
        
        Returns:
            Dictionary of combined features
        """
        # Extract track features
        features = self.extract_track_features(track, frame_id)
        
        # Extract pose features if available
        if pose is not None:
            pose_features = self.extract_pose_features(pose, track.bbox)
            # Prefix pose features
            for key, value in pose_features.items():
                features[f"pose_{key}"] = value
        
        return features
    
    def _get_center(self, bbox: np.ndarray) -> np.ndarray:
        """Get center of bounding box."""
        return np.array([
            (bbox[0] + bbox[2]) / 2,
            (bbox[1] + bbox[3]) / 2
        ])
