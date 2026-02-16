"""
Base tracker interface and track data structures.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class Track:
    """Single track representation."""
    
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    embedding: Optional[np.ndarray] = None
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    state: str = "tentative"  # tentative, confirmed, deleted
    camera_id: Optional[str] = None
    trajectory: List[np.ndarray] = field(default_factory=list)
    
    def update(
        self,
        bbox: np.ndarray,
        confidence: float,
        embedding: Optional[np.ndarray] = None
    ):
        """Update track with new detection."""
        self.bbox = bbox
        self.confidence = confidence
        if embedding is not None:
            self.embedding = embedding
        self.hits += 1
        self.time_since_update = 0
        self.trajectory.append(bbox.copy())
        
        if self.state == "tentative" and self.hits >= 3:
            self.state = "confirmed"
    
    def predict(self):
        """Predict next state (placeholder for Kalman filter)."""
        self.age += 1
        self.time_since_update += 1
    
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
            "trajectory": [t.tolist() for t in self.trajectory]
        }


class Tracker(ABC):
    """Base tracker interface."""
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3
    ):
        """
        Initialize tracker.
        
        Args:
            max_age: Maximum frames to keep track without update
            min_hits: Minimum hits to confirm track
            iou_threshold: IoU threshold for association
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[Track] = []
        self.next_id = 1
    
    @abstractmethod
    def update(self, detections: List[Dict]) -> List[Track]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dictionaries with 'bbox', 'confidence', 'class_id'
        
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
    
    def reset(self):
        """Reset tracker state."""
        self.tracks.clear()
        self.next_id = 1
