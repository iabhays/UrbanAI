"""
Advanced Motion Analyzer for Crowd Risk Assessment.

Implements optical flow analysis, movement vector variance,
crowd entropy metrics, and motion intensity heatmaps.
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque
from loguru import logger

from .person_tracker import TrackedPerson
from sentient_city.core import get_logger, get_config


@dataclass
class MotionFeatures:
    """Motion features for a person or crowd region."""
    
    velocity_magnitude: float
    direction_angle: float
    acceleration: float
    direction_variance: float
    motion_consistency: float
    chaotic_index: float
    flow_density: float
    
    
@dataclass
class CrowdMotionMetrics:
    """Global crowd motion metrics."""
    
    avg_speed: float
    speed_variance: float
    direction_variance: float
    crowd_entropy: float
    flow_coherence: float
    chaotic_regions: List[Tuple[int, int, float]]  # (x, y, intensity)
    compression_density: float
    expansion_rate: float


class MotionAnalyzer:
    """
    Advanced motion analyzer for crowd risk assessment.
    
    Features:
    - Farneback optical flow for dense motion analysis
    - Movement vector variance scoring
    - Crowd entropy calculation
    - Motion intensity heatmaps
    - Compression and expansion detection
    """
    
    def __init__(
        self,
        flow_window_size: int = 15,
        grid_size: int = 20,
        entropy_bins: int = 8,
        chaos_threshold: float = 0.7
    ):
        """
        Initialize motion analyzer.
        
        Args:
            flow_window_size: Window size for optical flow
            grid_size: Grid size for crowd analysis
            entropy_bins: Number of bins for entropy calculation
            chaos_threshold: Threshold for chaos detection
        """
        self.flow_window_size = flow_window_size
        self.grid_size = grid_size
        self.entropy_bins = entropy_bins
        self.chaos_threshold = chaos_threshold
        
        self.logger = get_logger(__name__)
        
        # Optical flow state
        self.prev_gray = None
        self.flow_history = deque(maxlen=10)
        
        # Motion metrics history
        self.motion_history = deque(maxlen=30)
        self.heatmap_history = deque(maxlen=5)
        
        # Performance optimization
        self.flow_step = 2  # Sample every 2 pixels for faster processing
        
    def analyze_frame_motion(
        self,
        frame: np.ndarray,
        tracked_persons: List[TrackedPerson]
    ) -> Tuple[CrowdMotionMetrics, Dict[int, MotionFeatures]]:
        """
        Analyze motion in current frame.
        
        Args:
            frame: Current frame (BGR format)
            tracked_persons: List of tracked persons
            
        Returns:
            Tuple of (crowd metrics, per-person motion features)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        if self.prev_gray is not None:
            flow = self._calculate_optical_flow(self.prev_gray, gray)
            self.flow_history.append(flow)
        else:
            flow = None
        
        # Analyze per-person motion
        person_features = self._analyze_person_motion(tracked_persons, flow)
        
        # Calculate crowd metrics
        crowd_metrics = self._calculate_crowd_metrics(tracked_persons, flow, frame.shape)
        
        # Update state
        self.prev_gray = gray.copy()
        
        return crowd_metrics, person_features
    
    def _calculate_optical_flow(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
        """Calculate dense optical flow using Farneback algorithm."""
        try:
            # Farneback optical flow for dense motion field
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                curr_gray,
                None,
                0.5,  # pyr_scale
                3,    # levels
                self.flow_window_size,  # winsize
                3,    # iterations
                5,    # poly_n
                1.2,  # poly_sigma
                0     # flags
            )
            
            return flow
            
        except Exception as e:
            self.logger.error(f"Error calculating optical flow: {e}")
            return np.zeros((prev_gray.shape[0], prev_gray.shape[1], 2), dtype=np.float32)
    
    def _analyze_person_motion(
        self,
        tracked_persons: List[TrackedPerson],
        flow: Optional[np.ndarray]
    ) -> Dict[int, MotionFeatures]:
        """Analyze motion features for each tracked person."""
        person_features = {}
        
        for person in tracked_persons:
            # Extract motion from track data
            velocity_magnitude = person.movement_speed
            
            # Direction angle
            if person.velocity is not None:
                direction_angle = np.arctan2(person.velocity[1], person.velocity[0]) * 180 / np.pi
            else:
                direction_angle = 0.0
            
            # Acceleration
            acceleration = float(np.linalg.norm(person.acceleration)) if person.acceleration is not None else 0.0
            
            # Direction variance from trajectory
            direction_variance = person.get_direction_variance()
            
            # Motion consistency (inverse of variance)
            motion_consistency = 1.0 / (1.0 + direction_variance)
            
            # Chaotic index (combination of speed variance and direction variance)
            chaotic_index = self._calculate_chaotic_index(person)
            
            # Flow density from optical flow
            flow_density = self._calculate_flow_density(person, flow) if flow is not None else 0.0
            
            features = MotionFeatures(
                velocity_magnitude=velocity_magnitude,
                direction_angle=direction_angle,
                acceleration=acceleration,
                direction_variance=direction_variance,
                motion_consistency=motion_consistency,
                chaotic_index=chaotic_index,
                flow_density=flow_density
            )
            
            person_features[person.track_id] = features
        
        return person_features
    
    def _calculate_chaotic_index(self, person: TrackedPerson) -> float:
        """Calculate chaotic index for a person based on trajectory."""
        if len(person.trajectory) < 5:
            return 0.0
        
        # Extract velocity vectors from trajectory
        velocities = []
        for i in range(1, len(person.trajectory)):
            if person.trajectory[i]['velocity'] is not None:
                velocities.append(person.trajectory[i]['velocity'])
        
        if len(velocities) < 3:
            return 0.0
        
        velocities = np.array(velocities)
        
        # Speed variance
        speeds = np.linalg.norm(velocities, axis=1)
        speed_variance = np.var(speeds)
        
        # Direction changes
        direction_changes = 0
        for i in range(1, len(velocities)):
            v1, v2 = velocities[i-1], velocities[i]
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle_change = np.arccos(cos_angle)
                if angle_change > np.pi/4:  # 45 degree threshold
                    direction_changes += 1
        
        direction_change_rate = direction_changes / len(velocities)
        
        # Combine metrics
        chaotic_index = 0.5 * min(speed_variance / 100.0, 1.0) + 0.5 * direction_change_rate
        
        return float(chaotic_index)
    
    def _calculate_flow_density(
        self,
        person: TrackedPerson,
        flow: np.ndarray
    ) -> float:
        """Calculate optical flow density around person."""
        x1, y1, x2, y2 = person.bbox.astype(int)
        
        # Ensure bounds are within image
        h, w = flow.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        # Extract flow in person region
        region_flow = flow[y1:y2, x1:x2]
        
        # Calculate flow magnitude
        flow_magnitude = np.sqrt(region_flow[:,:,0]**2 + region_flow[:,:,1]**2)
        
        # Average flow density
        density = np.mean(flow_magnitude)
        
        return float(density)
    
    def _calculate_crowd_metrics(
        self,
        tracked_persons: List[TrackedPerson],
        flow: Optional[np.ndarray],
        frame_shape: Tuple[int, int, int]
    ) -> CrowdMotionMetrics:
        """Calculate global crowd motion metrics."""
        if not tracked_persons:
            return CrowdMotionMetrics(
                avg_speed=0.0,
                speed_variance=0.0,
                direction_variance=0.0,
                crowd_entropy=0.0,
                flow_coherence=0.0,
                chaotic_regions=[],
                compression_density=0.0,
                expansion_rate=0.0
            )
        
        # Extract speeds and directions
        speeds = [person.movement_speed for person in tracked_persons]
        directions = []
        
        for person in tracked_persons:
            if person.velocity is not None and np.linalg.norm(person.velocity) > 0:
                direction = person.velocity / np.linalg.norm(person.velocity)
                directions.append(direction)
        
        # Basic statistics
        avg_speed = np.mean(speeds)
        speed_variance = np.var(speeds)
        
        # Direction variance
        if directions:
            directions = np.array(directions)
            mean_direction = np.mean(directions, axis=0)
            direction_variance = np.mean([np.linalg.norm(d - mean_direction)**2 for d in directions])
        else:
            direction_variance = 0.0
        
        # Crowd entropy
        crowd_entropy = self._calculate_crowd_entropy(tracked_persons, frame_shape)
        
        # Flow coherence
        flow_coherence = self._calculate_flow_coherence(directions)
        
        # Chaotic regions detection
        chaotic_regions = self._detect_chaotic_regions(flow, frame_shape) if flow is not None else []
        
        # Compression and expansion
        compression_density, expansion_rate = self._calculate_compression_expansion(
            tracked_persons, frame_shape
        )
        
        return CrowdMotionMetrics(
            avg_speed=float(avg_speed),
            speed_variance=float(speed_variance),
            direction_variance=float(direction_variance),
            crowd_entropy=float(crowd_entropy),
            flow_coherence=float(flow_coherence),
            chaotic_regions=chaotic_regions,
            compression_density=float(compression_density),
            expansion_rate=float(expansion_rate)
        )
    
    def _calculate_crowd_entropy(
        self,
        tracked_persons: List[TrackedPerson],
        frame_shape: Tuple[int, int, int]
    ) -> float:
        """Calculate crowd entropy based on spatial distribution and motion."""
        h, w = frame_shape[:2]
        
        # Create grid
        grid_x = w // self.grid_size
        grid_y = h // self.grid_size
        
        # Count people in each grid cell
        grid_counts = np.zeros((grid_y, grid_x))
        
        for person in tracked_persons:
            cx, cy = person.center
            grid_x_idx = min(int(cx // self.grid_size), grid_x - 1)
            grid_y_idx = min(int(cy // self.grid_size), grid_y - 1)
            grid_counts[grid_y_idx, grid_x_idx] += 1
        
        # Calculate entropy
        total_people = len(tracked_persons)
        if total_people == 0:
            return 0.0
        
        probabilities = grid_counts.flatten() / total_people
        probabilities = probabilities[probabilities > 0]  # Remove zeros
        
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        return float(entropy)
    
    def _calculate_flow_coherence(self, directions: List[np.ndarray]) -> float:
        """Calculate flow coherence (how aligned movements are)."""
        if len(directions) < 2:
            return 1.0
        
        directions = np.array(directions)
        
        # Calculate pairwise alignment
        alignments = []
        for i in range(len(directions)):
            for j in range(i + 1, len(directions)):
                alignment = np.dot(directions[i], directions[j])
                alignments.append(alignment)
        
        coherence = np.mean(alignments)
        
        return float(coherence)
    
    def _detect_chaotic_regions(
        self,
        flow: np.ndarray,
        frame_shape: Tuple[int, int, int]
    ) -> List[Tuple[int, int, float]]:
        """Detect regions with chaotic motion using optical flow."""
        h, w = frame_shape[:2]
        chaotic_regions = []
        
        # Sample grid points
        for y in range(0, h, self.grid_size * 2):
            for x in range(0, w, self.grid_size * 2):
                # Extract local flow
                x1, y1 = x, y
                x2, y2 = min(x + self.grid_size * 2, w), min(y + self.grid_size * 2, h)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                local_flow = flow[y1:y2, x1:x2]
                
                # Calculate flow variance
                flow_magnitude = np.sqrt(local_flow[:,:,0]**2 + local_flow[:,:,1]**2)
                flow_variance = np.var(flow_magnitude)
                
                # Check if chaotic
                if flow_variance > self.chaos_threshold * 10:
                    chaotic_regions.append((x + self.grid_size, y + self.grid_size, flow_variance))
        
        return chaotic_regions
    
    def _calculate_compression_expansion(
        self,
        tracked_persons: List[TrackedPerson],
        frame_shape: Tuple[int, int, int]
    ) -> Tuple[float, float]:
        """Calculate crowd compression and expansion rates."""
        if len(tracked_persons) < 2:
            return 0.0, 0.0
        
        # Calculate center of mass
        positions = np.array([person.center for person in tracked_persons])
        center_of_mass = np.mean(positions, axis=0)
        
        # Calculate distances from center
        distances = np.linalg.norm(positions - center_of_mass, axis=1)
        avg_distance = np.mean(distances)
        
        # Calculate compression density (people per unit area)
        frame_area = frame_shape[0] * frame_shape[1]
        compression_density = len(tracked_persons) / (frame_area / 10000)  # per 100x100 pixels
        
        # Calculate expansion rate from velocities
        expansion_rates = []
        for person in tracked_persons:
            if person.velocity is not None:
                # Radial velocity component
                to_center = center_of_mass - person.center
                if np.linalg.norm(to_center) > 0:
                    radial_direction = to_center / np.linalg.norm(to_center)
                    radial_velocity = np.dot(person.velocity, radial_direction)
                    expansion_rates.append(-radial_velocity)  # Negative for expansion
        
        expansion_rate = np.mean(expansion_rates) if expansion_rates else 0.0
        
        return float(compression_density), float(expansion_rate)
    
    def generate_motion_heatmap(self, flow: np.ndarray, frame_shape: Tuple[int, int, int]) -> np.ndarray:
        """Generate motion intensity heatmap."""
        if flow is None:
            return np.zeros((frame_shape[0], frame_shape[1]), dtype=np.uint8)
        
        # Calculate flow magnitude
        flow_magnitude = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
        
        # Normalize to 0-255
        if flow_magnitude.max() > 0:
            heatmap = (flow_magnitude / flow_magnitude.max() * 255).astype(np.uint8)
        else:
            heatmap = np.zeros_like(flow_magnitude, dtype=np.uint8)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        return heatmap_colored
    
    def get_motion_trends(self) -> Dict:
        """Get motion trends over time."""
        if len(self.motion_history) < 2:
            return {}
        
        recent_metrics = list(self.motion_history)[-10:]  # Last 10 frames
        
        # Calculate trends
        speed_trend = np.mean([m.avg_speed for m in recent_metrics])
        chaos_trend = np.mean([len(m.chaotic_regions) for m in recent_metrics])
        entropy_trend = np.mean([m.crowd_entropy for m in recent_metrics])
        
        return {
            "avg_speed_trend": float(speed_trend),
            "chaos_trend": float(chaos_trend),
            "entropy_trend": float(entropy_trend),
            "stability_score": float(1.0 / (1.0 + chaos_trend))
        }
