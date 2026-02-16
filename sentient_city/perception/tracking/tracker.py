"""
Base tracker interface with research lab integration.

Enhanced tracking system with experiment tracking and metrics.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field
from loguru import logger

from sentient_city.core import get_logger, get_config
from sentient_city.core.experiment_tracking import get_experiment_tracker


@dataclass
class Track:
    """
    Single track representation with enhanced features.
    
    Represents a tracked object across frames with trajectory,
    embeddings, and behavior features.
    """
    
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    embedding: Optional[np.ndarray] = None
    reid_embedding: Optional[np.ndarray] = None
    behavior_features: Optional[np.ndarray] = None
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    state: str = "tentative"  # tentative, confirmed, deleted
    camera_id: Optional[str] = None
    trajectory: List[np.ndarray] = field(default_factory=list)
    velocity: Optional[np.ndarray] = None
    last_seen_frame: int = 0
    
    def update(
        self,
        bbox: np.ndarray,
        confidence: float,
        embedding: Optional[np.ndarray] = None,
        reid_embedding: Optional[np.ndarray] = None,
        behavior_features: Optional[np.ndarray] = None,
        frame_id: int = 0
    ):
        """
        Update track with new detection.
        
        Args:
            bbox: New bounding box
            confidence: Detection confidence
            embedding: Detection embedding
            reid_embedding: ReID embedding
            behavior_features: Behavior features
            frame_id: Current frame ID
        """
        # Update velocity
        if len(self.trajectory) > 0:
            prev_center = self._get_center(self.bbox)
            curr_center = self._get_center(bbox)
            self.velocity = curr_center - prev_center
        
        self.bbox = bbox
        self.confidence = confidence
        if embedding is not None:
            self.embedding = embedding
        if reid_embedding is not None:
            self.reid_embedding = reid_embedding
        if behavior_features is not None:
            self.behavior_features = behavior_features
        self.hits += 1
        self.time_since_update = 0
        self.trajectory.append(bbox.copy())
        self.last_seen_frame = frame_id
        
        if self.state == "tentative" and self.hits >= 3:
            self.state = "confirmed"
    
    def predict(self):
        """Predict next state (placeholder for Kalman filter)."""
        self.age += 1
        self.time_since_update += 1
    
    def _get_center(self, bbox: np.ndarray) -> np.ndarray:
        """Get center of bounding box."""
        return np.array([
            (bbox[0] + bbox[2]) / 2,
            (bbox[1] + bbox[3]) / 2
        ])
    
    def get_trajectory_length(self) -> int:
        """Get trajectory length."""
        return len(self.trajectory)
    
    def get_speed(self) -> float:
        """Get current speed (pixels per frame)."""
        if self.velocity is None:
            return 0.0
        return np.linalg.norm(self.velocity)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "track_id": self.track_id,
            "bbox": self.bbox.tolist(),
            "confidence": float(self.confidence),
            "class_id": int(self.class_id),
            "age": self.age,
            "hits": self.hits,
            "state": self.state,
            "camera_id": self.camera_id,
            "trajectory_length": self.get_trajectory_length(),
            "speed": self.get_speed(),
            "last_seen_frame": self.last_seen_frame
        }


class BaseTracker(ABC):
    """
    Base tracker interface with research lab integration.
    
    Provides common tracking functionality and experiment tracking.
    """
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        camera_id: Optional[str] = None,
        experiment_id: Optional[str] = None
    ):
        """
        Initialize tracker.
        
        Args:
            max_age: Maximum frames to keep track without update
            min_hits: Minimum hits to confirm track
            iou_threshold: IoU threshold for association
            camera_id: Camera identifier
            experiment_id: Optional experiment ID for tracking
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.camera_id = camera_id
        self.experiment_id = experiment_id
        
        self.tracks: List[Track] = []
        self.next_id = 1
        self.frame_count = 0
        
        self.logger = get_logger(__name__)
        self.config = get_config()
        
        # Tracking metrics
        self.metrics = {
            "total_tracks": 0,
            "active_tracks": 0,
            "confirmed_tracks": 0,
            "lost_tracks": 0
        }
    
    @abstractmethod
    def update(self, detections: List[Dict], frame_id: int = 0) -> List[Track]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dictionaries with 'bbox', 'confidence', 'class_id'
            frame_id: Current frame ID
        
        Returns:
            List of active tracks
        """
        pass
    
    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _get_centroid(self, bbox: np.ndarray) -> np.ndarray:
        """Get centroid of bounding box."""
        return np.array([
            (bbox[0] + bbox[2]) / 2,
            (bbox[1] + bbox[3]) / 2
        ])
    
    def get_active_tracks(self) -> List[Track]:
        """Get all active (confirmed) tracks."""
        return [t for t in self.tracks if t.state == "confirmed"]
    
    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """Get track by ID."""
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None
    
    def update_metrics(self) -> Dict[str, int]:
        """
        Update tracking metrics.
        
        Returns:
            Dictionary of metrics
        """
        self.metrics["active_tracks"] = len([t for t in self.tracks if t.time_since_update < self.max_age])
        self.metrics["confirmed_tracks"] = len([t for t in self.tracks if t.state == "confirmed"])
        self.metrics["lost_tracks"] = len([t for t in self.tracks if t.time_since_update >= self.max_age])
        
        return self.metrics.copy()
    
    def log_metrics(self, frame_id: int) -> None:
        """
        Log metrics to experiment tracker.
        
        Args:
            frame_id: Current frame ID
        """
        if self.experiment_id:
            metrics = self.update_metrics()
            tracker = get_experiment_tracker()
            tracker.log_metrics(
                self.experiment_id,
                metrics={f"tracking_{k}": v for k, v in metrics.items()},
                step=frame_id
            )
    
    def reset(self):
        """Reset tracker state."""
        self.tracks.clear()
        self.next_id = 1
        self.frame_count = 0
        self.metrics = {
            "total_tracks": 0,
            "active_tracks": 0,
            "confirmed_tracks": 0,
            "lost_tracks": 0
        }
        self.logger.info("Tracker reset")
