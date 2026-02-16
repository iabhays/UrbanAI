"""
SENTIENTCITY AI - Anomaly Detection Plugin
Detects behavioral anomalies using statistical and ML methods
"""

from typing import Any
from collections import deque
import numpy as np

from sentientcity.schemas.events import PluginResult


class Plugin:
    """Anomaly detection plugin using behavior analysis."""

    name = "anomaly_detection"
    version = "0.1.0"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.anomaly_threshold = config.get("anomaly_threshold", 2.5)  # std deviations
        self.loitering_threshold = config.get("loitering_threshold", 60)  # seconds
        self.speed_threshold = config.get("speed_threshold", 50.0)  # pixels/frame
        
        # Historical patterns for baseline
        self._velocity_history: deque = deque(maxlen=1000)
        self._position_history: deque = deque(maxlen=1000)

    async def initialize(self) -> None:
        """Initialize plugin resources."""
        pass

    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        pass

    async def analyze(self, context: Any) -> PluginResult:
        """
        Detect behavioral anomalies.
        
        Anomaly types:
        - Unusual speed (running, very slow movement)
        - Loitering (staying in same area too long)
        - Erratic movement patterns
        - Against-flow movement
        """
        findings = []
        risk_contribution = 0.0
        confidence = 0.7
        details = {}
        
        tracks = context.tracks
        
        if len(tracks) < 2:
            return PluginResult(
                plugin_name=self.name,
                plugin_version=self.version,
                confidence=0.5,
                risk_contribution=0.0,
                findings=["Insufficient data for anomaly detection"],
                details={},
                processing_time_ms=0.0,
            )
        
        # Analyze each track
        anomalies = []
        
        for track in tracks[-30:]:
            track_anomalies = self._analyze_track(track)
            anomalies.extend(track_anomalies)
        
        # Update historical data
        self._update_history(tracks)
        
        # Process anomalies
        speed_anomalies = [a for a in anomalies if a["type"] == "speed"]
        loitering_anomalies = [a for a in anomalies if a["type"] == "loitering"]
        erratic_anomalies = [a for a in anomalies if a["type"] == "erratic"]
        against_flow = [a for a in anomalies if a["type"] == "against_flow"]
        
        details["anomaly_counts"] = {
            "speed": len(speed_anomalies),
            "loitering": len(loitering_anomalies),
            "erratic": len(erratic_anomalies),
            "against_flow": len(against_flow),
        }
        
        # Speed anomalies
        if speed_anomalies:
            risk_contribution += min(30, len(speed_anomalies) * 10)
            findings.append(f"Unusual speed detected: {len(speed_anomalies)} instances")
        
        # Loitering
        if loitering_anomalies:
            risk_contribution += min(25, len(loitering_anomalies) * 8)
            findings.append(f"Loitering detected: {len(loitering_anomalies)} individuals")
        
        # Erratic movement
        if erratic_anomalies:
            risk_contribution += min(20, len(erratic_anomalies) * 7)
            findings.append(f"Erratic movement patterns: {len(erratic_anomalies)} instances")
        
        # Against flow movement
        if against_flow:
            risk_contribution += min(25, len(against_flow) * 8)
            findings.append(f"Against-flow movement: {len(against_flow)} individuals")
            confidence = 0.8
        
        return PluginResult(
            plugin_name=self.name,
            plugin_version=self.version,
            confidence=confidence,
            risk_contribution=min(100, risk_contribution),
            findings=findings,
            details=details,
            processing_time_ms=0.0,
        )

    def _analyze_track(self, track: dict) -> list[dict]:
        """Analyze individual track for anomalies."""
        anomalies = []
        track_id = track.get("track_id", "unknown")
        
        # Speed analysis
        if "velocity" in track and track["velocity"]:
            vx, vy = track["velocity"]
            speed = (vx**2 + vy**2) ** 0.5
            
            if speed > self.speed_threshold:
                anomalies.append({
                    "type": "speed",
                    "track_id": track_id,
                    "value": speed,
                    "severity": "high" if speed > self.speed_threshold * 1.5 else "medium",
                })
            elif self._velocity_history and speed > 0:
                mean_speed = np.mean(list(self._velocity_history))
                std_speed = np.std(list(self._velocity_history)) + 0.1
                z_score = abs(speed - mean_speed) / std_speed
                
                if z_score > self.anomaly_threshold:
                    anomalies.append({
                        "type": "speed",
                        "track_id": track_id,
                        "value": speed,
                        "z_score": z_score,
                        "severity": "medium",
                    })
        
        # Loitering analysis
        if "trajectory" in track:
            trajectory = track["trajectory"]
            if len(trajectory) > 10:
                displacement = self._calculate_displacement(trajectory)
                if displacement < 50 and len(trajectory) > self.loitering_threshold:
                    anomalies.append({
                        "type": "loitering",
                        "track_id": track_id,
                        "duration": len(trajectory),
                        "displacement": displacement,
                    })
        
        # Erratic movement
        if "trajectory" in track:
            trajectory = track["trajectory"]
            if len(trajectory) > 5:
                direction_changes = self._count_direction_changes(trajectory)
                if direction_changes > len(trajectory) * 0.4:
                    anomalies.append({
                        "type": "erratic",
                        "track_id": track_id,
                        "direction_changes": direction_changes,
                    })
        
        return anomalies

    def _calculate_displacement(self, trajectory: list[dict]) -> float:
        """Calculate total displacement from trajectory."""
        if len(trajectory) < 2:
            return 0.0
        
        start = trajectory[0]
        end = trajectory[-1]
        
        dx = end.get("x", 0) - start.get("x", 0)
        dy = end.get("y", 0) - start.get("y", 0)
        
        return (dx**2 + dy**2) ** 0.5

    def _count_direction_changes(self, trajectory: list[dict]) -> int:
        """Count significant direction changes in trajectory."""
        if len(trajectory) < 3:
            return 0
        
        changes = 0
        prev_dir = None
        
        for i in range(1, len(trajectory)):
            dx = trajectory[i].get("x", 0) - trajectory[i-1].get("x", 0)
            dy = trajectory[i].get("y", 0) - trajectory[i-1].get("y", 0)
            
            if abs(dx) > 1 or abs(dy) > 1:
                direction = np.arctan2(dy, dx)
                if prev_dir is not None:
                    angle_diff = abs(direction - prev_dir)
                    if angle_diff > np.pi / 4:  # More than 45 degrees
                        changes += 1
                prev_dir = direction
        
        return changes

    def _update_history(self, tracks: list[dict]) -> None:
        """Update historical baseline data."""
        for track in tracks:
            if "velocity" in track and track["velocity"]:
                vx, vy = track["velocity"]
                speed = (vx**2 + vy**2) ** 0.5
                if speed > 0:
                    self._velocity_history.append(speed)
