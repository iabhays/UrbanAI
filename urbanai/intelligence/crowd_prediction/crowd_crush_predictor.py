"""
Crowd crush risk prediction engine.

Predicts risk of crowd crush incidents based on density, movement, and behavior.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger

from ...utils.config import get_config


class CrowdCrushPredictor:
    """
    Crowd crush risk predictor.
    
    Analyzes crowd density, movement patterns, and behavior to predict
    crush risk.
    """
    
    def __init__(
        self,
        density_threshold: float = 0.8,
        movement_threshold: float = 0.6,
        risk_threshold: float = 0.7,
        lookback_window: int = 30
    ):
        """
        Initialize crowd crush predictor.
        
        Args:
            density_threshold: Density threshold for high risk
            movement_threshold: Movement threshold for risk
            risk_threshold: Overall risk threshold
            lookback_window: Number of frames to analyze
        """
        self.config = get_config()
        intelligence_config = self.config.get_section("intelligence")
        crush_config = intelligence_config.get("crowd_crush", {})
        
        self.density_threshold = crush_config.get("density_threshold", density_threshold)
        self.movement_threshold = crush_config.get("movement_threshold", movement_threshold)
        self.risk_threshold = crush_config.get("risk_threshold", risk_threshold)
        self.lookback_window = crush_config.get("lookback_window", lookback_window)
        
        self.history: List[Dict] = []
    
    def predict(
        self,
        density_map: np.ndarray,
        tracks: List[Dict],
        movement_vectors: Optional[np.ndarray] = None
    ) -> Dict[str, any]:
        """
        Predict crowd crush risk.
        
        Args:
            density_map: Crowd density map
            tracks: List of track dictionaries
            movement_vectors: Optional movement vectors for each track
        
        Returns:
            Dictionary with:
            - risk_score: Risk score (0-1)
            - is_high_risk: Boolean indicating high risk
            - factors: Dictionary of contributing factors
            - recommendation: Recommended action
        """
        # Calculate density metrics
        mean_density = np.mean(density_map)
        max_density = np.max(density_map)
        density_variance = np.var(density_map)
        
        # Calculate movement metrics
        movement_score = self._calculate_movement_score(tracks, movement_vectors)
        
        # Calculate spatial distribution
        spatial_score = self._calculate_spatial_distribution(tracks)
        
        # Calculate temporal trends
        temporal_score = self._calculate_temporal_trends(mean_density)
        
        # Combine factors
        factors = {
            "density": {
                "mean": float(mean_density),
                "max": float(max_density),
                "variance": float(density_variance),
                "score": min(1.0, mean_density / self.density_threshold)
            },
            "movement": {
                "score": float(movement_score),
                "is_restricted": movement_score < self.movement_threshold
            },
            "spatial": {
                "score": float(spatial_score),
                "is_clustered": spatial_score > 0.7
            },
            "temporal": {
                "score": float(temporal_score),
                "is_increasing": temporal_score > 0.6
            }
        }
        
        # Calculate overall risk score
        risk_score = self._calculate_risk_score(factors)
        
        is_high_risk = risk_score >= self.risk_threshold
        
        # Generate recommendation
        recommendation = self._generate_recommendation(risk_score, factors)
        
        # Add to history
        self.history.append({
            "risk_score": risk_score,
            "density": mean_density,
            "timestamp": None  # Will be set by caller
        })
        if len(self.history) > self.lookback_window:
            self.history.pop(0)
        
        return {
            "risk_score": float(risk_score),
            "is_high_risk": is_high_risk,
            "factors": factors,
            "recommendation": recommendation
        }
    
    def _calculate_movement_score(
        self,
        tracks: List[Dict],
        movement_vectors: Optional[np.ndarray]
    ) -> float:
        """Calculate movement score."""
        if len(tracks) < 2:
            return 0.5
        
        if movement_vectors is not None:
            # Use provided movement vectors
            avg_speed = np.mean(np.linalg.norm(movement_vectors, axis=1))
            return min(1.0, avg_speed / 10.0)
        
        # Calculate from track trajectories
        speeds = []
        for track in tracks:
            trajectory = track.get("trajectory", [])
            if len(trajectory) >= 2:
                # Calculate speed from last two positions
                pos1 = np.array(trajectory[-2][:2]) if isinstance(trajectory[-2], list) else trajectory[-2][:2]
                pos2 = np.array(trajectory[-1][:2]) if isinstance(trajectory[-1], list) else trajectory[-1][:2]
                speed = np.linalg.norm(pos2 - pos1)
                speeds.append(speed)
        
        if speeds:
            avg_speed = np.mean(speeds)
            return min(1.0, avg_speed / 10.0)
        
        return 0.5
    
    def _calculate_spatial_distribution(self, tracks: List[Dict]) -> float:
        """Calculate spatial distribution score (clustering)."""
        if len(tracks) < 2:
            return 0.5
        
        # Extract track positions
        positions = []
        for track in tracks:
            bbox = track.get("bbox", [])
            if len(bbox) >= 4:
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                positions.append([center_x, center_y])
        
        if len(positions) < 2:
            return 0.5
        
        positions = np.array(positions)
        
        # Calculate average distance to centroid
        centroid = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - centroid, axis=1)
        avg_distance = np.mean(distances)
        
        # Low average distance indicates clustering
        clustering_score = 1.0 / (1.0 + avg_distance / 100.0)
        
        return clustering_score
    
    def _calculate_temporal_trends(self, current_density: float) -> float:
        """Calculate temporal trend score."""
        if len(self.history) < 3:
            return 0.5
        
        # Calculate density trend
        densities = [h["density"] for h in self.history[-10:]]
        densities.append(current_density)
        
        # Linear regression to get trend
        x = np.arange(len(densities))
        coeffs = np.polyfit(x, densities, 1)
        slope = coeffs[0]
        
        # Normalize slope
        trend_score = (slope + 0.01) / 0.02  # Normalize to 0-1
        trend_score = np.clip(trend_score, 0.0, 1.0)
        
        return trend_score
    
    def _calculate_risk_score(self, factors: Dict) -> float:
        """Calculate overall risk score from factors."""
        weights = {
            "density": 0.4,
            "movement": 0.2,
            "spatial": 0.2,
            "temporal": 0.2
        }
        
        risk_score = (
            factors["density"]["score"] * weights["density"] +
            factors["movement"]["score"] * weights["movement"] +
            factors["spatial"]["score"] * weights["spatial"] +
            factors["temporal"]["score"] * weights["temporal"]
        )
        
        # Boost risk if multiple factors are high
        high_factors = sum([
            factors["density"]["score"] > 0.7,
            factors["movement"]["is_restricted"],
            factors["spatial"]["is_clustered"],
            factors["temporal"]["is_increasing"]
        ])
        
        if high_factors >= 3:
            risk_score = min(1.0, risk_score * 1.2)
        
        return risk_score
    
    def _generate_recommendation(self, risk_score: float, factors: Dict) -> str:
        """Generate recommendation based on risk."""
        if risk_score >= 0.9:
            return "IMMEDIATE EVACUATION REQUIRED - Critical crowd crush risk detected"
        elif risk_score >= 0.7:
            return "HIGH RISK - Deploy crowd control measures immediately"
        elif risk_score >= 0.5:
            return "MODERATE RISK - Monitor closely and prepare intervention"
        else:
            return "LOW RISK - Normal crowd conditions"
