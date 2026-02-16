"""
Behavior analyzer.

Analyzes behavior patterns from extracted features.
"""

import numpy as np
from typing import Dict, List, Optional
from collections import deque
from loguru import logger

from sentient_city.core import get_logger, get_config
from .feature_extractor import BehaviorFeatureExtractor


class BehaviorAnalyzer:
    """
    Behavior analyzer.
    
    Analyzes behavior patterns from extracted features:
    - Movement patterns
    - Speed analysis
    - Pose-based behavior
    - Trajectory analysis
    - Anomaly detection
    """
    
    def __init__(
        self,
        feature_extractor: Optional[BehaviorFeatureExtractor] = None,
        window_size: int = 30,
        experiment_id: Optional[str] = None
    ):
        """
        Initialize behavior analyzer.
        
        Args:
            feature_extractor: Behavior feature extractor
            window_size: Temporal window size for analysis
            experiment_id: Optional experiment ID for tracking
        """
        self.feature_extractor = feature_extractor or BehaviorFeatureExtractor()
        self.window_size = window_size
        self.experiment_id = experiment_id
        
        self.logger = get_logger(__name__)
        self.config = get_config()
        
        # Feature history: {track_id: deque of features}
        self.feature_history: Dict[int, deque] = {}
    
    def analyze_track(
        self,
        track,
        pose: Optional[Dict] = None,
        frame_id: int = 0
    ) -> Dict[str, any]:
        """
        Analyze behavior for a track.
        
        Args:
            track: Track object
            pose: Optional pose dictionary
            frame_id: Current frame ID
        
        Returns:
            Dictionary of behavior analysis results
        """
        # Extract features
        features = self.feature_extractor.extract_combined_features(track, pose, frame_id)
        
        # Update history
        if track.track_id not in self.feature_history:
            self.feature_history[track.track_id] = deque(maxlen=self.window_size)
        self.feature_history[track.track_id].append(features)
        
        # Analyze current behavior
        analysis = {
            "track_id": track.track_id,
            "frame_id": frame_id,
            "features": features,
            "movement": self._analyze_movement(features),
            "speed": self._analyze_speed(features),
            "pose": self._analyze_pose(features) if pose else None
        }
        
        # Temporal analysis if history available
        if len(self.feature_history[track.track_id]) > 1:
            analysis["temporal"] = self._analyze_temporal(track.track_id)
        
        return analysis
    
    def _analyze_movement(self, features: Dict) -> Dict:
        """Analyze movement patterns."""
        movement = {}
        
        if "speed" in features:
            speed = features["speed"][0]
            movement["speed"] = float(speed)
            movement["is_moving"] = speed > 5.0  # Threshold in pixels/frame
            movement["movement_category"] = (
                "fast" if speed > 20.0 else
                "medium" if speed > 10.0 else
                "slow" if speed > 5.0 else
                "stationary"
            )
        
        if "direction" in features:
            direction = features["direction"]
            movement["direction"] = direction.tolist()
            # Convert to angle
            angle = np.arctan2(direction[1], direction[0]) * 180 / np.pi
            movement["direction_angle"] = float(angle)
        
        return movement
    
    def _analyze_speed(self, features: Dict) -> Dict:
        """Analyze speed patterns."""
        speed_analysis = {}
        
        if "speed" in features:
            speed = features["speed"][0]
            speed_analysis["current_speed"] = float(speed)
            speed_analysis["speed_category"] = (
                "very_fast" if speed > 30.0 else
                "fast" if speed > 20.0 else
                "medium" if speed > 10.0 else
                "slow" if speed > 5.0 else
                "stationary"
            )
        
        if "velocity" in features:
            velocity = features["velocity"]
            speed_analysis["velocity"] = velocity.tolist()
            speed_analysis["velocity_magnitude"] = float(np.linalg.norm(velocity))
        
        return speed_analysis
    
    def _analyze_pose(self, features: Dict) -> Dict:
        """Analyze pose-based behavior."""
        pose_analysis = {}
        
        if "pose_num_visible_keypoints" in features:
            num_kps = features["pose_num_visible_keypoints"][0]
            pose_analysis["num_visible_keypoints"] = int(num_kps)
            pose_analysis["pose_quality"] = (
                "good" if num_kps > 10 else
                "medium" if num_kps > 5 else
                "poor"
            )
        
        if "pose_keypoint_spread" in features:
            spread = features["pose_keypoint_spread"]
            pose_analysis["keypoint_spread"] = spread.tolist()
            pose_analysis["spread_magnitude"] = float(np.linalg.norm(spread))
        
        return pose_analysis
    
    def _analyze_temporal(self, track_id: int) -> Dict:
        """Analyze temporal patterns from history."""
        if track_id not in self.feature_history:
            return {}
        
        history = list(self.feature_history[track_id])
        if len(history) < 2:
            return {}
        
        temporal = {}
        
        # Speed trend
        speeds = [h.get("speed", np.array([0.0]))[0] for h in history if "speed" in h]
        if len(speeds) > 1:
            temporal["speed_trend"] = "increasing" if speeds[-1] > speeds[0] else "decreasing" if speeds[-1] < speeds[0] else "stable"
            temporal["speed_variance"] = float(np.var(speeds))
            temporal["avg_speed"] = float(np.mean(speeds))
        
        # Trajectory consistency
        if "trajectory" in history[-1] and len(history[-1]["trajectory"]) > 1:
            trajectory = history[-1]["trajectory"]
            if len(trajectory) > 1:
                centers = np.array([self._get_center(bbox) for bbox in trajectory])
                temporal["trajectory_consistency"] = float(np.std(centers, axis=0).mean())
        
        return temporal
    
    def _get_center(self, bbox: np.ndarray) -> np.ndarray:
        """Get center of bounding box."""
        return np.array([
            (bbox[0] + bbox[2]) / 2,
            (bbox[1] + bbox[3]) / 2
        ])
    
    def clear_history(self, track_id: Optional[int] = None):
        """Clear feature history."""
        if track_id is None:
            self.feature_history.clear()
        elif track_id in self.feature_history:
            del self.feature_history[track_id]
