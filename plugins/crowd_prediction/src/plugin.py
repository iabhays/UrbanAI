"""
SENTIENTCITY AI - Crowd Prediction Plugin
Predicts crowd crush risk and flow patterns
"""

from typing import Any
import numpy as np

from sentientcity.schemas.events import PluginResult


class Plugin:
    """Crowd prediction and crush risk analysis plugin."""

    name = "crowd_prediction"
    version = "0.1.0"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.density_threshold = config.get("density_threshold", 4.0)  # people/m²
        self.velocity_threshold = config.get("velocity_threshold", 0.5)  # m/s
        self.crush_risk_threshold = config.get("crush_risk_threshold", 0.7)

    async def initialize(self) -> None:
        """Initialize plugin resources."""
        pass

    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        pass

    async def analyze(self, context: Any) -> PluginResult:
        """
        Analyze crowd conditions for crush risk.
        
        Factors considered:
        - Local density
        - Velocity variance
        - Flow convergence
        - Historical patterns
        """
        findings = []
        risk_contribution = 0.0
        confidence = 0.8
        details = {}
        
        tracks = context.tracks
        
        if len(tracks) < 3:
            return PluginResult(
                plugin_name=self.name,
                plugin_version=self.version,
                confidence=0.5,
                risk_contribution=0.0,
                findings=["Insufficient data for crowd analysis"],
                details={},
                processing_time_ms=0.0,
            )
        
        # Calculate crowd metrics
        density = self._calculate_density(tracks)
        velocity_variance = self._calculate_velocity_variance(tracks)
        flow_convergence = self._calculate_flow_convergence(tracks)
        
        details["density"] = density
        details["velocity_variance"] = velocity_variance
        details["flow_convergence"] = flow_convergence
        
        # Density analysis
        if density > self.density_threshold:
            risk_contribution += 30
            findings.append(f"High crowd density detected: {density:.1f} people/m²")
        elif density > self.density_threshold * 0.7:
            risk_contribution += 15
            findings.append(f"Elevated crowd density: {density:.1f} people/m²")
        
        # Velocity analysis
        if velocity_variance > self.velocity_threshold:
            risk_contribution += 20
            findings.append("Irregular crowd movement detected")
        
        # Flow convergence (people moving toward same point)
        if flow_convergence > 0.7:
            risk_contribution += 25
            findings.append("Crowd flow convergence detected - potential bottleneck")
        
        # Calculate crush risk score
        crush_risk = self._calculate_crush_risk(density, velocity_variance, flow_convergence)
        details["crush_risk"] = crush_risk
        
        if crush_risk > self.crush_risk_threshold:
            risk_contribution += 25
            findings.append(f"CRITICAL: High crush risk ({crush_risk:.2f})")
            confidence = 0.9
        
        return PluginResult(
            plugin_name=self.name,
            plugin_version=self.version,
            confidence=confidence,
            risk_contribution=min(100, risk_contribution),
            findings=findings,
            details=details,
            processing_time_ms=0.0,
        )

    def _calculate_density(self, tracks: list[dict]) -> float:
        """Estimate local crowd density."""
        if len(tracks) < 2:
            return 0.0
        
        # Get positions from recent tracks
        positions = []
        for track in tracks[-50:]:
            if "current_bbox" in track:
                bbox = track["current_bbox"]
                cx = (bbox.get("x1", 0) + bbox.get("x2", 0)) / 2
                cy = (bbox.get("y1", 0) + bbox.get("y2", 0)) / 2
                positions.append((cx, cy))
        
        if len(positions) < 2:
            return 0.0
        
        # Calculate area covered
        positions = np.array(positions)
        min_x, min_y = positions.min(axis=0)
        max_x, max_y = positions.max(axis=0)
        
        area = max((max_x - min_x) * (max_y - min_y), 1.0)
        
        # Rough conversion: assume 1 pixel = 0.01m
        area_m2 = area * 0.0001
        
        return len(positions) / max(area_m2, 1.0)

    def _calculate_velocity_variance(self, tracks: list[dict]) -> float:
        """Calculate variance in movement velocities."""
        velocities = []
        
        for track in tracks[-50:]:
            if "velocity" in track and track["velocity"]:
                vx, vy = track["velocity"]
                speed = (vx**2 + vy**2) ** 0.5
                velocities.append(speed)
        
        if len(velocities) < 2:
            return 0.0
        
        return float(np.std(velocities))

    def _calculate_flow_convergence(self, tracks: list[dict]) -> float:
        """Calculate how much the crowd is converging to a single point."""
        if len(tracks) < 5:
            return 0.0
        
        # Get velocity vectors
        vectors = []
        positions = []
        
        for track in tracks[-30:]:
            if "velocity" in track and track["velocity"] and "current_bbox" in track:
                vx, vy = track["velocity"]
                bbox = track["current_bbox"]
                cx = (bbox.get("x1", 0) + bbox.get("x2", 0)) / 2
                cy = (bbox.get("y1", 0) + bbox.get("y2", 0)) / 2
                
                if vx != 0 or vy != 0:
                    vectors.append((vx, vy))
                    positions.append((cx, cy))
        
        if len(vectors) < 3:
            return 0.0
        
        # Check if vectors point toward a common center
        center_x = np.mean([p[0] for p in positions])
        center_y = np.mean([p[1] for p in positions])
        
        convergent = 0
        for (vx, vy), (px, py) in zip(vectors, positions):
            # Direction to center
            dx = center_x - px
            dy = center_y - py
            
            # Dot product (positive means moving toward center)
            dot = vx * dx + vy * dy
            if dot > 0:
                convergent += 1
        
        return convergent / len(vectors)

    def _calculate_crush_risk(
        self,
        density: float,
        velocity_variance: float,
        flow_convergence: float,
    ) -> float:
        """Calculate overall crush risk score."""
        # Normalize factors
        density_factor = min(density / self.density_threshold, 2.0) / 2.0
        variance_factor = min(velocity_variance / self.velocity_threshold, 2.0) / 2.0
        convergence_factor = flow_convergence
        
        # Weighted combination
        risk = (
            0.5 * density_factor +
            0.25 * variance_factor +
            0.25 * convergence_factor
        )
        
        return min(1.0, risk)
