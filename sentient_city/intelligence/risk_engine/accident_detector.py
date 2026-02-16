"""
Traffic accident detection engine.

Detects vehicle accidents and collision risks.
"""

import numpy as np
from typing import Dict, List, Optional
from loguru import logger

from ...utils.config import get_config


class AccidentDetector:
    """
    Traffic accident detector.
    
    Detects vehicle accidents and predicts collision risks.
    """
    
    def __init__(
        self,
        probability_threshold: float = 0.8,
        vehicle_classes: Optional[List[int]] = None
    ):
        """
        Initialize accident detector.
        
        Args:
            probability_threshold: Accident probability threshold
            vehicle_classes: List of vehicle class IDs (COCO format)
        """
        self.config = get_config()
        intelligence_config = self.config.get_section("intelligence")
        accident_config = intelligence_config.get("accident_detection", {})
        
        self.probability_threshold = accident_config.get(
            "probability_threshold", probability_threshold
        )
        self.vehicle_classes = accident_config.get(
            "vehicle_classes", vehicle_classes or [2, 3, 5, 7]
        )  # car, motorcycle, bus, truck
        
        self.track_history: Dict[int, List[Dict]] = {}
    
    def detect(
        self,
        detections: List[Dict],
        tracks: List[Dict]
    ) -> Dict[str, any]:
        """
        Detect accidents and collision risks.
        
        Args:
            detections: List of detection dictionaries
            tracks: List of track dictionaries
        
        Returns:
            Dictionary with:
            - is_accident: Boolean indicating accident detected
            - accident_probability: Probability of accident (0-1)
            - collision_risks: List of collision risk events
            - vehicles_involved: List of vehicle IDs involved
        """
        # Filter vehicle tracks
        vehicle_tracks = [
            t for t in tracks
            if t.get("class_id", -1) in self.vehicle_classes
        ]
        
        if len(vehicle_tracks) < 2:
            return {
                "is_accident": False,
                "accident_probability": 0.0,
                "collision_risks": [],
                "vehicles_involved": []
            }
        
        # Detect collisions
        collisions = self._detect_collisions(vehicle_tracks)
        
        # Detect collision risks
        collision_risks = self._detect_collision_risks(vehicle_tracks)
        
        # Calculate overall accident probability
        accident_probability = self._calculate_accident_probability(
            collisions, collision_risks
        )
        
        is_accident = accident_probability >= self.probability_threshold
        
        # Get vehicles involved
        vehicles_involved = []
        if collisions:
            for collision in collisions:
                vehicles_involved.extend(collision.get("vehicles", []))
        vehicles_involved = list(set(vehicles_involved))
        
        return {
            "is_accident": bool(is_accident),
            "accident_probability": float(accident_probability),
            "collisions": collisions,
            "collision_risks": collision_risks,
            "vehicles_involved": vehicles_involved
        }
    
    def _detect_collisions(self, tracks: List[Dict]) -> List[Dict]:
        """Detect actual collisions between vehicles."""
        collisions = []
        
        for i, track1 in enumerate(tracks):
            for j, track2 in enumerate(tracks[i+1:], start=i+1):
                # Check if bounding boxes overlap significantly
                bbox1 = track1.get("bbox", [])
                bbox2 = track2.get("bbox", [])
                
                if len(bbox1) < 4 or len(bbox2) < 4:
                    continue
                
                iou = self._calculate_iou(bbox1, bbox2)
                
                # High IoU indicates collision
                if iou > 0.3:
                    # Check if velocities indicate collision
                    velocity1 = self._calculate_velocity(track1)
                    velocity2 = self._calculate_velocity(track2)
                    
                    # Check if vehicles are moving toward each other
                    if velocity1 > 0.1 and velocity2 > 0.1:
                        collisions.append({
                            "vehicles": [track1.get("track_id"), track2.get("track_id")],
                            "iou": float(iou),
                            "confidence": float(min(1.0, iou * 2)),
                            "type": "collision"
                        })
        
        return collisions
    
    def _detect_collision_risks(self, tracks: List[Dict]) -> List[Dict]:
        """Detect potential collision risks."""
        risks = []
        
        for i, track1 in enumerate(tracks):
            for j, track2 in enumerate(tracks[i+1:], start=i+1):
                # Predict future positions
                future_pos1 = self._predict_position(track1, frames_ahead=5)
                future_pos2 = self._predict_position(track2, frames_ahead=5)
                
                # Check if future positions overlap
                if future_pos1 and future_pos2:
                    future_bbox1 = [
                        future_pos1[0] - 20, future_pos1[1] - 20,
                        future_pos1[0] + 20, future_pos1[1] + 20
                    ]
                    future_bbox2 = [
                        future_pos2[0] - 20, future_pos2[1] - 20,
                        future_pos2[0] + 20, future_pos2[1] + 20
                    ]
                    
                    future_iou = self._calculate_iou(future_bbox1, future_bbox2)
                    
                    if future_iou > 0.1:
                        # Calculate time to collision
                        ttc = self._calculate_time_to_collision(track1, track2)
                        
                        if ttc and ttc < 2.0:  # Less than 2 seconds
                            risks.append({
                                "vehicles": [track1.get("track_id"), track2.get("track_id")],
                                "time_to_collision": float(ttc),
                                "probability": float(1.0 / (1.0 + ttc)),
                                "type": "collision_risk"
                            })
        
        return risks
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU between two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_velocity(self, track: Dict) -> float:
        """Calculate velocity from track trajectory."""
        trajectory = track.get("trajectory", [])
        if len(trajectory) < 2:
            return 0.0
        
        # Get last two positions
        pos1 = trajectory[-2]
        pos2 = trajectory[-1]
        
        if isinstance(pos1, list):
            pos1 = np.array(pos1[:2])
        else:
            pos1 = np.array([pos1[0], pos1[1]])
        
        if isinstance(pos2, list):
            pos2 = np.array(pos2[:2])
        else:
            pos2 = np.array([pos2[0], pos2[1]])
        
        velocity = np.linalg.norm(pos2 - pos1)
        return velocity
    
    def _predict_position(self, track: Dict, frames_ahead: int = 5) -> Optional[np.ndarray]:
        """Predict future position based on velocity."""
        trajectory = track.get("trajectory", [])
        if len(trajectory) < 2:
            return None
        
        # Get current position and velocity
        current_pos = trajectory[-1]
        if isinstance(current_pos, list):
            current_pos = np.array(current_pos[:2])
        else:
            current_pos = np.array([current_pos[0], current_pos[1]])
        
        velocity = self._calculate_velocity(track)
        if velocity < 0.1:
            return None
        
        # Estimate direction
        if len(trajectory) >= 2:
            prev_pos = trajectory[-2]
            if isinstance(prev_pos, list):
                prev_pos = np.array(prev_pos[:2])
            else:
                prev_pos = np.array([prev_pos[0], prev_pos[1]])
            
            direction = (current_pos - prev_pos) / (np.linalg.norm(current_pos - prev_pos) + 1e-8)
            future_pos = current_pos + direction * velocity * frames_ahead
            return future_pos
        
        return None
    
    def _calculate_time_to_collision(self, track1: Dict, track2: Dict) -> Optional[float]:
        """Calculate time to collision."""
        pos1 = self._predict_position(track1, frames_ahead=10)
        pos2 = self._predict_position(track2, frames_ahead=10)
        
        if pos1 is None or pos2 is None:
            return None
        
        # Get current positions
        traj1 = track1.get("trajectory", [])
        traj2 = track2.get("trajectory", [])
        
        if len(traj1) == 0 or len(traj2) == 0:
            return None
        
        curr_pos1 = traj1[-1]
        curr_pos2 = traj2[-1]
        
        if isinstance(curr_pos1, list):
            curr_pos1 = np.array(curr_pos1[:2])
        else:
            curr_pos1 = np.array([curr_pos1[0], curr_pos1[1]])
        
        if isinstance(curr_pos2, list):
            curr_pos2 = np.array(curr_pos2[:2])
        else:
            curr_pos2 = np.array([curr_pos2[0], curr_pos2[1]])
        
        # Distance between vehicles
        distance = np.linalg.norm(curr_pos1 - curr_pos2)
        
        # Relative velocity
        vel1 = self._calculate_velocity(track1)
        vel2 = self._calculate_velocity(track2)
        
        if vel1 + vel2 < 0.1:
            return None
        
        # Estimate time to collision
        ttc = distance / (vel1 + vel2 + 1e-8)
        return ttc
    
    def _calculate_accident_probability(
        self,
        collisions: List[Dict],
        collision_risks: List[Dict]
    ) -> float:
        """Calculate overall accident probability."""
        # Base probability from collisions
        collision_prob = 0.0
        if collisions:
            collision_prob = max(c["confidence"] for c in collisions)
        
        # Risk probability
        risk_prob = 0.0
        if collision_risks:
            risk_prob = max(r["probability"] for r in collision_risks)
        
        # Combine probabilities
        accident_prob = max(collision_prob, risk_prob * 0.7)
        
        return min(1.0, accident_prob)
