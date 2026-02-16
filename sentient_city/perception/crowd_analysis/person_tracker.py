"""
Enhanced Person Tracker with ByteTrack integration.

Robust multi-person tracking optimized for crowded environments
with occlusion handling, re-entry support, and ID stability.
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from loguru import logger
import torch

from .person_detector import Detection
from sentient_city.core import get_logger, get_config


@dataclass
class TrackedPerson:
    """Enhanced tracked person with comprehensive features."""
    
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    center: np.ndarray
    velocity: Optional[np.ndarray] = None
    acceleration: Optional[np.ndarray] = None
    trajectory: deque = field(default_factory=lambda: deque(maxlen=30))
    age: int = 0
    time_since_update: int = 0
    hits: int = 0
    misses: int = 0
    state: str = "tentative"  # tentative, confirmed, deleted
    risk_level: str = "Low"
    risk_score: float = 0.0
    movement_speed: float = 0.0
    direction_variance: float = 0.0
    last_seen_frame: int = 0
    occluded_frames: int = 0
    re_entry_count: int = 0
    
    def update(self, detection: Detection, frame_id: int):
        """Update track with new detection."""
        # Calculate velocity and acceleration
        if len(self.trajectory) > 0:
            prev_center = self.trajectory[-1]['center']
            curr_center = detection.center
            new_velocity = curr_center - prev_center
            
            if self.velocity is not None:
                self.acceleration = new_velocity - self.velocity
            self.velocity = new_velocity
            self.movement_speed = float(np.linalg.norm(new_velocity))
        
        # Update position
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.center = detection.center
        
        # Update tracking info
        self.hits += 1
        self.time_since_update = 0
        self.age += 1
        self.last_seen_frame = frame_id
        self.occluded_frames = 0
        
        # Add to trajectory
        self.trajectory.append({
            'center': self.center.copy(),
            'bbox': self.bbox.copy(),
            'frame_id': frame_id,
            'velocity': self.velocity.copy() if self.velocity is not None else None
        })
        
        # Update state
        if self.state == "tentative" and self.hits >= 3:
            self.state = "confirmed"
    
    def predict(self):
        """Predict next position (simple linear prediction)."""
        self.age += 1
        self.time_since_update += 1
        self.occluded_frames += 1
        
        # Simple position prediction based on velocity
        if self.velocity is not None and len(self.trajectory) > 0:
            predicted_center = self.trajectory[-1]['center'] + self.velocity
            # Update bbox center while maintaining size
            bbox_width = self.bbox[2] - self.bbox[0]
            bbox_height = self.bbox[3] - self.bbox[1]
            self.bbox = np.array([
                predicted_center[0] - bbox_width / 2,
                predicted_center[1] - bbox_height / 2,
                predicted_center[0] + bbox_width / 2,
                predicted_center[1] + bbox_height / 2
            ])
            self.center = predicted_center
    
    def get_direction_variance(self) -> float:
        """Calculate direction variance from trajectory."""
        if len(self.trajectory) < 3:
            return 0.0
        
        directions = []
        for i in range(1, len(self.trajectory)):
            if self.trajectory[i]['velocity'] is not None:
                vel = self.trajectory[i]['velocity']
                if np.linalg.norm(vel) > 0:
                    direction = vel / np.linalg.norm(vel)
                    directions.append(direction)
        
        if len(directions) < 2:
            return 0.0
        
        directions = np.array(directions)
        # Calculate variance of direction vectors
        mean_direction = np.mean(directions, axis=0)
        variance = np.mean([np.linalg.norm(d - mean_direction)**2 for d in directions])
        
        return float(variance)
    
    def is_occluded(self) -> bool:
        """Check if track is likely occluded."""
        return self.occluded_frames > 5 and self.state == "confirmed"
    
    def should_delete(self) -> bool:
        """Check if track should be deleted."""
        return (self.time_since_update > 30 or 
                (self.state == "tentative" and self.time_since_update > 10))


class ByteTrackTracker:
    """
    ByteTrack-inspired tracker optimized for crowded scenes.
    
    Features:
    - Robust ID assignment and maintenance
    - Occlusion handling
    - Re-entry detection
    - Dense crowd overlap management
    - Motion-based prediction
    """
    
    def __init__(
        self,
        track_buffer: int = 30,
        confirmation_threshold: int = 3,
        deletion_threshold: int = 30,
        iou_threshold: float = 0.5,
        max_disappeared: int = 10
    ):
        """
        Initialize ByteTrack tracker.
        
        Args:
            track_buffer: Maximum trajectory length to store
            confirmation_threshold: Hits needed to confirm track
            deletion_threshold: Frames without update before deletion
            iou_threshold: IoU threshold for matching
            max_disappeared: Max frames a track can disappear
        """
        self.track_buffer = track_buffer
        self.confirmation_threshold = confirmation_threshold
        self.deletion_threshold = deletion_threshold
        self.iou_threshold = iou_threshold
        self.max_disappeared = max_disappeared
        
        self.logger = get_logger(__name__)
        
        # Tracking state
        self.tracks: Dict[int, TrackedPerson] = {}
        self.next_track_id = 1
        self.frame_count = 0
        
        # Re-entry detection
        self.appeared_features: Dict[int, np.ndarray] = {}
        self.disappeared_tracks: Dict[int, TrackedPerson] = {}
        
        # Performance metrics
        self.total_matches = 0
        self.total_new_tracks = 0
        
    def update(self, detections: List[Detection], frame_id: Optional[int] = None) -> List[TrackedPerson]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of person detections
            frame_id: Current frame ID
            
        Returns:
            List of active tracked persons
        """
        if frame_id is None:
            frame_id = self.frame_count
            
        self.frame_count += 1
        
        # Predict all tracks
        self._predict_all_tracks()
        
        if not detections:
            # No detections, just update track states
            self._update_track_states()
            return list(self.tracks.values())
        
        # Match detections to existing tracks
        matched_tracks, unmatched_detections = self._match_detections(detections)
        
        # Update matched tracks
        for track_id, detection in matched_tracks:
            if track_id in self.tracks:
                self.tracks[track_id].update(detection, frame_id)
                self.total_matches += 1
        
        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            self._create_new_track(detection, frame_id)
        
        # Update track states and remove old tracks
        self._update_track_states()
        
        active_tracks = [track for track in self.tracks.values() if track.state == "confirmed"]
        
        self.logger.debug(f"Frame {frame_id}: {len(active_tracks)} active tracks")
        
        return active_tracks
    
    def _predict_all_tracks(self):
        """Predict next state for all tracks."""
        for track in self.tracks.values():
            track.predict()
    
    def _match_detections(self, detections: List[Detection]) -> Tuple[List[Tuple[int, Detection]], List[Detection]]:
        """Match detections to existing tracks using IoU and motion cues."""
        matched_tracks = []
        unmatched_detections = detections.copy()
        
        if not self.tracks:
            return matched_tracks, unmatched_detections
        
        # Calculate IoU matrix
        track_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(track_ids), len(detections)))
        
        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            for j, detection in enumerate(detections):
                iou = self._calculate_iou(track.bbox, detection.bbox)
                # Add motion-based scoring
                motion_score = self._calculate_motion_score(track, detection)
                combined_score = 0.7 * iou + 0.3 * motion_score
                iou_matrix[i, j] = combined_score
        
        # Hungarian algorithm for optimal matching
        from scipy.optimize import linear_sum_assignment
        row_indices, col_indices = linear_sum_assignment(-iou_matrix)
        
        # Apply threshold
        for i, j in zip(row_indices, col_indices):
            if iou_matrix[i, j] >= self.iou_threshold:
                track_id = track_ids[i]
                detection = detections[j]
                matched_tracks.append((track_id, detection))
                unmatched_detections.remove(detection)
        
        # Try re-entry matching for remaining detections
        if unmatched_detections:
            re_entry_matches = self._try_re_entry_matching(unmatched_detections)
            matched_tracks.extend(re_entry_matches)
            for track_id, detection in re_entry_matches:
                if detection in unmatched_detections:
                    unmatched_detections.remove(detection)
        
        return matched_tracks, unmatched_detections
    
    def _calculate_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
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
    
    def _calculate_motion_score(self, track: TrackedPerson, detection: Detection) -> float:
        """Calculate motion-based matching score."""
        if track.velocity is None or len(track.trajectory) < 2:
            return 0.5  # Neutral score for new tracks
        
        # Predict next position based on velocity
        predicted_center = track.center + track.velocity
        
        # Calculate distance from predicted position
        distance = np.linalg.norm(detection.center - predicted_center)
        
        # Convert to score (higher is better)
        max_distance = 100.0  # Maximum expected distance in pixels
        score = max(0.0, 1.0 - distance / max_distance)
        
        return float(score)
    
    def _try_re_entry_matching(self, detections: List[Detection]) -> List[Tuple[int, Detection]]:
        """Try to match detections with previously disappeared tracks."""
        re_entry_matches = []
        
        for detection in detections:
            best_match_id = None
            best_score = 0.0
            
            for track_id, disappeared_track in self.disappeared_tracks.items():
                # Check if enough time has passed for re-entry
                if self.frame_count - disappeared_track.last_seen_frame < 10:
                    continue
                
                # Calculate position similarity
                distance = np.linalg.norm(detection.center - disappeared_track.center)
                if distance < 50:  # Threshold for re-entry
                    score = 1.0 - (distance / 50.0)
                    if score > best_score:
                        best_score = score
                        best_match_id = track_id
            
            if best_match_id and best_score > 0.6:
                # Re-activate the disappeared track
                disappeared_track = self.disappeared_tracks.pop(best_match_id)
                disappeared_track.re_entry_count += 1
                disappeared_track.state = "tentative"
                disappeared_track.time_since_update = 0
                self.tracks[best_match_id] = disappeared_track
                re_entry_matches.append((best_match_id, detection))
        
        return re_entry_matches
    
    def _create_new_track(self, detection: Detection, frame_id: int):
        """Create a new track from detection."""
        track = TrackedPerson(
            track_id=self.next_track_id,
            bbox=detection.bbox.copy(),
            confidence=detection.confidence,
            center=detection.center.copy(),
            last_seen_frame=frame_id
        )
        
        self.tracks[self.next_track_id] = track
        self.next_track_id += 1
        self.total_new_tracks += 1
        
        self.logger.debug(f"Created new track {track.track_id}")
    
    def _update_track_states(self):
        """Update track states and remove old tracks."""
        tracks_to_delete = []
        
        for track_id, track in self.tracks.items():
            if track.should_delete():
                tracks_to_delete.append(track_id)
                # Move to disappeared tracks for potential re-entry
                if track.state == "confirmed":
                    self.disappeared_tracks[track_id] = track
            elif track.time_since_update > self.max_disappeared:
                track.state = "tentative"
        
        # Delete old tracks
        for track_id in tracks_to_delete:
            del self.tracks[track_id]
        
        # Clean old disappeared tracks
        disappeared_to_delete = []
        for track_id, track in self.disappeared_tracks.items():
            if self.frame_count - track.last_seen_frame > 60:  # 2 seconds at 30fps
                disappeared_to_delete.append(track_id)
        
        for track_id in disappeared_to_delete:
            del self.disappeared_tracks[track_id]
    
    def get_track_count(self) -> Dict[str, int]:
        """Get track count by state."""
        counts = {"tentative": 0, "confirmed": 0, "deleted": 0}
        for track in self.tracks.values():
            counts[track.state] += 1
        return counts
    
    def get_performance_metrics(self) -> Dict:
        """Get tracking performance metrics."""
        return {
            "frame_count": self.frame_count,
            "active_tracks": len(self.tracks),
            "disappeared_tracks": len(self.disappeared_tracks),
            "total_matches": self.total_matches,
            "total_new_tracks": self.total_new_tracks,
            "track_states": self.get_track_count()
        }
    
    def reset(self):
        """Reset tracker state."""
        self.tracks.clear()
        self.disappeared_tracks.clear()
        self.next_track_id = 1
        self.frame_count = 0
        self.total_matches = 0
        self.total_new_tracks = 0
