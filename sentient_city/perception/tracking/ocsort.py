"""
OC-SORT tracker implementation.

Online and Real-time Tracking by Associating Every Detection Box
with a High-Score Track.
"""

import numpy as np
from typing import List, Dict, Optional
from loguru import logger

from .tracker import BaseTracker, Track


class OCSortTracker(BaseTracker):
    """
    OC-SORT tracker.
    
    Implements observation-centric tracking with online smoothing.
    """
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        delta_t: int = 3,
        camera_id: Optional[str] = None,
        experiment_id: Optional[str] = None
    ):
        """
        Initialize OC-SORT tracker.
        
        Args:
            max_age: Maximum frames to keep track without update
            min_hits: Minimum hits to confirm track
            iou_threshold: IoU threshold for association
            delta_t: Temporal window for observation-centric association
            camera_id: Camera identifier
            experiment_id: Optional experiment ID for tracking
        """
        super().__init__(max_age, min_hits, iou_threshold, camera_id, experiment_id)
        self.delta_t = delta_t
    
    def update(self, detections: List[Dict], frame_id: int = 0) -> List[Track]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dictionaries
        
        Returns:
            List of active tracks
        """
        # Predict all tracks
        for track in self.tracks:
            track.predict()
        
        # Extract detection boxes and features
        det_boxes = np.array([d["bbox"] for d in detections])
        det_confidences = np.array([d["confidence"] for d in detections])
        det_classes = np.array([d.get("class_id", 0) for d in detections])
        det_embeddings = [d.get("embedding") for d in detections]
        
        if len(detections) == 0:
            # No detections, mark all tracks as stale
            self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]
            return self.get_active_tracks()
        
        # Associate detections with tracks
        matched, unmatched_dets, unmatched_trks = self._associate(
            det_boxes, det_confidences
        )
        
        # Update matched tracks
        for m in matched:
            track = self.tracks[m[0]]
            det_idx = m[1]
            track.update(
                det_boxes[det_idx],
                det_confidences[det_idx],
                embedding=det_embeddings[det_idx] if det_embeddings[det_idx] is not None else None,
                frame_id=frame_id
            )
        
        # Create new tracks for unmatched detections
        for i in unmatched_dets:
            new_track = Track(
                track_id=self.next_id,
                bbox=det_boxes[i],
                confidence=det_confidences[i],
                class_id=det_classes[i],
                embedding=det_embeddings[i] if det_embeddings[i] is not None else None
            )
            self.tracks.append(new_track)
            self.next_id += 1
        
        # Remove stale tracks
        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]
        
        # Update frame count and log metrics
        self.frame_count += 1
        if self.frame_count % 100 == 0:
            self.log_metrics(frame_id)
        
        return self.get_active_tracks()
    
    def _associate(
        self,
        det_boxes: np.ndarray,
        det_confidences: np.ndarray
    ) -> tuple:
        """
        Associate detections with tracks using IoU and observation-centric matching.
        
        Returns:
            Tuple of (matched pairs, unmatched detections, unmatched tracks)
        """
        if len(self.tracks) == 0:
            return [], list(range(len(det_boxes))), []
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(self.tracks), len(det_boxes)))
        for i, track in enumerate(self.tracks):
            for j, det_box in enumerate(det_boxes):
                iou_matrix[i, j] = self._calculate_iou(track.bbox, det_box)
        
        # Observation-centric association: prefer recent observations
        cost_matrix = 1 - iou_matrix
        
        # Apply observation-centric weighting
        for i, track in enumerate(self.tracks):
            if track.time_since_update <= self.delta_t:
                # Recent tracks get higher priority
                cost_matrix[i] *= 0.5
        
        # Hungarian algorithm (simplified greedy matching)
        matched = []
        unmatched_dets = list(range(len(det_boxes)))
        unmatched_trks = list(range(len(self.tracks)))
        
        # Greedy matching (in production, use Hungarian algorithm)
        while True:
            if len(unmatched_trks) == 0 or len(unmatched_dets) == 0:
                break
            
            # Find best match
            min_cost = float('inf')
            best_match = None
            
            for i in unmatched_trks:
                for j in unmatched_dets:
                    if iou_matrix[i, j] > self.iou_threshold:
                        if cost_matrix[i, j] < min_cost:
                            min_cost = cost_matrix[i, j]
                            best_match = (i, j)
            
            if best_match is None:
                break
            
            trk_idx, det_idx = best_match
            matched.append((trk_idx, det_idx))
            unmatched_trks.remove(trk_idx)
            unmatched_dets.remove(det_idx)
        
        return matched, unmatched_dets, unmatched_trks
