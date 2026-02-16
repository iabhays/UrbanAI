"""
DeepSORT tracker implementation.

Simple Online and Realtime Tracking with a Deep Association Metric.
"""

import numpy as np
from typing import List, Dict, Optional
from loguru import logger

from .tracker import Tracker, Track
from .reid import ReIDModel


class DeepSortTracker(Tracker):
    """
    DeepSORT tracker with deep association metric.
    
    Uses appearance features for association in addition to IoU.
    """
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        max_distance: float = 0.2,
        reid_model: Optional[ReIDModel] = None
    ):
        """
        Initialize DeepSORT tracker.
        
        Args:
            max_age: Maximum frames to keep track without update
            min_hits: Minimum hits to confirm track
            iou_threshold: IoU threshold for association
            max_distance: Maximum cosine distance for appearance matching
            reid_model: ReID model for feature extraction
        """
        super().__init__(max_age, min_hits, iou_threshold)
        self.max_distance = max_distance
        self.reid_model = reid_model
    
    def update(self, detections: List[Dict]) -> List[Track]:
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
        
        if len(detections) == 0:
            self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]
            return self.get_active_tracks()
        
        # Extract detection data
        det_boxes = np.array([d["bbox"] for d in detections])
        det_confidences = np.array([d["confidence"] for d in detections])
        det_classes = np.array([d.get("class_id", 0) for d in detections])
        det_embeddings = []
        
        # Extract or compute embeddings
        for det in detections:
            if "embedding" in det and det["embedding"] is not None:
                det_embeddings.append(np.array(det["embedding"]))
            elif self.reid_model is not None:
                # Extract embedding using ReID model (would need image crop)
                # For now, use placeholder
                det_embeddings.append(None)
            else:
                det_embeddings.append(None)
        
        # Associate detections with tracks
        matched, unmatched_dets, unmatched_trks = self._associate(
            det_boxes, det_confidences, det_embeddings
        )
        
        # Update matched tracks
        for m in matched:
            track = self.tracks[m[0]]
            det_idx = m[1]
            track.update(
                det_boxes[det_idx],
                det_confidences[det_idx],
                det_embeddings[det_idx] if det_embeddings[det_idx] is not None else None
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
        
        return self.get_active_tracks()
    
    def _associate(
        self,
        det_boxes: np.ndarray,
        det_confidences: np.ndarray,
        det_embeddings: List[Optional[np.ndarray]]
    ) -> tuple:
        """
        Associate detections with tracks using IoU and appearance features.
        
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
        
        # Calculate appearance distance matrix
        appearance_matrix = np.ones((len(self.tracks), len(det_boxes)))
        
        for i, track in enumerate(self.tracks):
            if track.embedding is not None:
                for j, det_emb in enumerate(det_embeddings):
                    if det_emb is not None:
                        # Cosine distance
                        track_emb_norm = track.embedding / (np.linalg.norm(track.embedding) + 1e-8)
                        det_emb_norm = det_emb / (np.linalg.norm(det_emb) + 1e-8)
                        appearance_matrix[i, j] = 1 - np.dot(track_emb_norm, det_emb_norm)
        
        # Combined cost matrix (IoU + appearance)
        cost_matrix = (1 - iou_matrix) * 0.5 + appearance_matrix * 0.5
        
        # Greedy matching
        matched = []
        unmatched_dets = list(range(len(det_boxes)))
        unmatched_trks = list(range(len(self.tracks)))
        
        while True:
            if len(unmatched_trks) == 0 or len(unmatched_dets) == 0:
                break
            
            # Find best match
            best_match = None
            best_iou = 0
            
            for i in unmatched_trks:
                for j in unmatched_dets:
                    iou = iou_matrix[i, j]
                    appearance_dist = appearance_matrix[i, j]
                    
                    # Match if IoU is good OR appearance is good
                    if (iou > self.iou_threshold or appearance_dist < self.max_distance):
                        if iou > best_iou:
                            best_iou = iou
                            best_match = (i, j)
            
            if best_match is None:
                break
            
            trk_idx, det_idx = best_match
            matched.append((trk_idx, det_idx))
            unmatched_trks.remove(trk_idx)
            unmatched_dets.remove(det_idx)
        
        return matched, unmatched_dets, unmatched_trks
