"""
Multi-camera tracking coordinator.

Coordinates tracking across multiple cameras with cross-camera re-identification.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from loguru import logger

from sentient_city.core import get_logger, get_config
from .tracker import BaseTracker, Track
from .reid import ReIDModel


class MultiCameraTracker:
    """
    Multi-camera tracking coordinator.
    
    Manages tracking across multiple cameras with cross-camera
    identity persistence using ReID.
    """
    
    def __init__(
        self,
        reid_model: Optional[ReIDModel] = None,
        reid_threshold: float = 0.7,
        experiment_id: Optional[str] = None
    ):
        """
        Initialize multi-camera tracker.
        
        Args:
            reid_model: ReID model for cross-camera matching
            reid_threshold: Similarity threshold for ReID matching
            experiment_id: Optional experiment ID for tracking
        """
        self.reid_model = reid_model
        self.reid_threshold = reid_threshold
        self.experiment_id = experiment_id
        
        # Camera trackers: {camera_id: BaseTracker}
        self.camera_trackers: Dict[str, BaseTracker] = {}
        
        # Global identity mapping: {global_id: {camera_id: track_id}}
        self.global_identities: Dict[int, Dict[str, int]] = {}
        self.next_global_id = 1
        
        # ReID feature cache: {global_id: embedding}
        self.reid_cache: Dict[int, np.ndarray] = {}
        
        self.logger = get_logger(__name__)
        self.config = get_config()
    
    def register_camera(
        self,
        camera_id: str,
        tracker: BaseTracker
    ) -> None:
        """
        Register a camera tracker.
        
        Args:
            camera_id: Camera identifier
            tracker: Tracker instance for this camera
        """
        self.camera_trackers[camera_id] = tracker
        tracker.camera_id = camera_id
        self.logger.info(f"Registered camera tracker: {camera_id}")
    
    def update_camera(
        self,
        camera_id: str,
        detections: List[Dict],
        frame_id: int = 0
    ) -> List[Track]:
        """
        Update tracking for a specific camera.
        
        Args:
            camera_id: Camera identifier
            detections: Detections for this camera
            frame_id: Current frame ID
        
        Returns:
            List of tracks with global IDs assigned
        """
        if camera_id not in self.camera_trackers:
            self.logger.warning(f"Camera {camera_id} not registered")
            return []
        
        tracker = self.camera_trackers[camera_id]
        tracks = tracker.update(detections, frame_id)
        
        # Assign global IDs
        tracks_with_global_ids = self._assign_global_ids(camera_id, tracks)
        
        return tracks_with_global_ids
    
    def _assign_global_ids(
        self,
        camera_id: str,
        tracks: List[Track]
    ) -> List[Track]:
        """
        Assign global IDs to tracks using ReID.
        
        Args:
            camera_id: Camera identifier
            tracks: Local tracks from camera
        
        Returns:
            Tracks with global IDs assigned
        """
        for track in tracks:
            # Check if track already has global ID
            if hasattr(track, 'global_id') and track.global_id is not None:
                continue
            
            # Try to match with existing global identities
            global_id = self._match_with_global_identities(track, camera_id)
            
            if global_id is None:
                # Create new global identity
                global_id = self.next_global_id
                self.next_global_id += 1
                self.global_identities[global_id] = {}
            
            # Assign global ID
            track.global_id = global_id
            self.global_identities[global_id][camera_id] = track.track_id
            
            # Cache ReID embedding
            if track.reid_embedding is not None:
                self.reid_cache[global_id] = track.reid_embedding
        
        return tracks
    
    def _match_with_global_identities(
        self,
        track: Track,
        camera_id: str
    ) -> Optional[int]:
        """
        Match track with existing global identities using ReID.
        
        Args:
            track: Track to match
            camera_id: Current camera ID
        
        Returns:
            Matched global ID or None
        """
        if track.reid_embedding is None or self.reid_model is None:
            return None
        
        best_match_id = None
        best_similarity = 0.0
        
        for global_id, cached_embedding in self.reid_cache.items():
            # Skip if already seen in this camera
            if camera_id in self.global_identities.get(global_id, {}):
                continue
            
            # Compute cosine similarity
            similarity = np.dot(track.reid_embedding, cached_embedding) / (
                np.linalg.norm(track.reid_embedding) * np.linalg.norm(cached_embedding)
            )
            
            if similarity > self.reid_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match_id = global_id
        
        return best_match_id if best_similarity >= self.reid_threshold else None
    
    def get_global_track(self, global_id: int) -> Optional[Dict]:
        """
        Get track information across all cameras.
        
        Args:
            global_id: Global identity ID
        
        Returns:
            Dictionary with track information across cameras
        """
        if global_id not in self.global_identities:
            return None
        
        camera_tracks = {}
        for camera_id, local_track_id in self.global_identities[global_id].items():
            tracker = self.camera_trackers.get(camera_id)
            if tracker:
                track = tracker.get_track_by_id(local_track_id)
                if track:
                    camera_tracks[camera_id] = track.to_dict()
        
        return {
            "global_id": global_id,
            "cameras": camera_tracks,
            "reid_embedding": self.reid_cache.get(global_id)
        }
    
    def get_all_global_tracks(self) -> List[Dict]:
        """Get all global tracks."""
        return [
            self.get_global_track(global_id)
            for global_id in self.global_identities.keys()
        ]
    
    def get_statistics(self) -> Dict:
        """Get multi-camera tracking statistics."""
        return {
            "num_cameras": len(self.camera_trackers),
            "num_global_identities": len(self.global_identities),
            "camera_tracks": {
                camera_id: len(tracker.get_active_tracks())
                for camera_id, tracker in self.camera_trackers.items()
            }
        }
