"""
Panic movement detection based on pose analysis.

Detects panic behavior through rapid, erratic movements.
"""

import numpy as np
from typing import Dict, List
from loguru import logger

from .pose_detector import PoseDetector


class PanicDetector:
    """
    Panic movement detector.
    
    Detects panic through:
    - Rapid arm movements
    - Erratic body movements
    - High movement variance
    """
    
    def __init__(self, threshold: float = 0.6, window_size: int = 10):
        """
        Initialize panic detector.
        
        Args:
            threshold: Panic detection threshold (0-1)
            window_size: Number of frames to analyze
        """
        self.threshold = threshold
        self.window_size = window_size
        self.pose_detector = PoseDetector()
        self.pose_history: List[Dict] = []
        self.velocity_history: List[np.ndarray] = []
    
    def detect(self, pose: Dict) -> Dict[str, any]:
        """
        Detect panic from pose.
        
        Args:
            pose: Pose dictionary from PoseDetector
        
        Returns:
            Dictionary with:
            - is_panic: Boolean indicating if panic detected
            - confidence: Confidence score (0-1)
            - reason: Reason for detection
        """
        if len(pose["keypoints"]) == 0:
            return {
                "is_panic": False,
                "confidence": 0.0,
                "reason": "No pose detected"
            }
        
        # Add to history
        self.pose_history.append(pose)
        if len(self.pose_history) > self.window_size:
            self.pose_history.pop(0)
        
        # Need at least 3 frames for analysis
        if len(self.pose_history) < 3:
            return {
                "is_panic": False,
                "confidence": 0.0,
                "reason": "Insufficient history"
            }
        
        # Calculate panic indicators
        indicators = self._calculate_indicators()
        
        # Combine indicators
        confidence = self._combine_indicators(indicators)
        
        is_panic = confidence >= self.threshold
        
        reason = "Normal"
        if is_panic:
            reason = self._get_reason(indicators)
        
        return {
            "is_panic": is_panic,
            "confidence": float(confidence),
            "reason": reason,
            "indicators": indicators
        }
    
    def _calculate_indicators(self) -> Dict[str, float]:
        """Calculate panic detection indicators."""
        indicators = {}
        
        # Calculate velocities for keypoints
        velocities = []
        for i in range(1, len(self.pose_history)):
            prev_pose = self.pose_history[i - 1]
            curr_pose = self.pose_history[i]
            
            prev_keypoints = prev_pose["keypoints"]
            curr_keypoints = curr_pose["keypoints"]
            
            frame_velocities = []
            for j in range(min(len(prev_keypoints), len(curr_keypoints))):
                if prev_keypoints[j][2] > 0.5 and curr_keypoints[j][2] > 0.5:
                    vel = np.sqrt(
                        (curr_keypoints[j][0] - prev_keypoints[j][0])**2 +
                        (curr_keypoints[j][1] - prev_keypoints[j][1])**2
                    )
                    frame_velocities.append(vel)
            
            if frame_velocities:
                velocities.append(np.array(frame_velocities))
        
        if not velocities:
            return {
                "rapid_arm_movement": 0.0,
                "high_variance": 0.0,
                "erratic_movement": 0.0
            }
        
        velocities = np.array(velocities)
        
        # Indicator 1: Rapid arm movement
        # Focus on wrist keypoints (indices 15, 16)
        wrist_indices = [15, 16]
        arm_velocities = []
        
        for i in range(1, len(self.pose_history)):
            prev_pose = self.pose_history[i - 1]
            curr_pose = self.pose_history[i]
            
            for wrist_idx in wrist_indices:
                prev_wrist = prev_pose["keypoints"][wrist_idx] if wrist_idx < len(prev_pose["keypoints"]) else None
                curr_wrist = curr_pose["keypoints"][wrist_idx] if wrist_idx < len(curr_pose["keypoints"]) else None
                
                if prev_wrist and curr_wrist and prev_wrist[2] > 0.5 and curr_wrist[2] > 0.5:
                    vel = np.sqrt(
                        (curr_wrist[0] - prev_wrist[0])**2 +
                        (curr_wrist[1] - prev_wrist[1])**2
                    )
                    arm_velocities.append(vel)
        
        if arm_velocities:
            avg_arm_velocity = np.mean(arm_velocities)
            indicators["rapid_arm_movement"] = min(1.0, avg_arm_velocity / 30.0)
        else:
            indicators["rapid_arm_movement"] = 0.0
        
        # Indicator 2: High movement variance
        if len(velocities) > 1:
            mean_velocities = np.mean(velocities, axis=1)
            variance = np.var(mean_velocities)
            indicators["high_variance"] = min(1.0, variance / 100.0)
        else:
            indicators["high_variance"] = 0.0
        
        # Indicator 3: Erratic movement (high frequency changes)
        if len(velocities) > 2:
            mean_velocities = np.mean(velocities, axis=1)
            # Count direction changes
            direction_changes = 0
            for i in range(1, len(mean_velocities) - 1):
                if (mean_velocities[i] > mean_velocities[i-1]) != (mean_velocities[i+1] > mean_velocities[i]):
                    direction_changes += 1
            
            change_rate = direction_changes / len(mean_velocities)
            indicators["erratic_movement"] = min(1.0, change_rate * 2)
        else:
            indicators["erratic_movement"] = 0.0
        
        return indicators
    
    def _combine_indicators(self, indicators: Dict[str, float]) -> float:
        """Combine indicators into final confidence score."""
        weights = {
            "rapid_arm_movement": 0.4,
            "high_variance": 0.3,
            "erratic_movement": 0.3
        }
        
        confidence = sum(
            indicators.get(key, 0.0) * weights.get(key, 0.0)
            for key in weights
        )
        
        return min(1.0, confidence)
    
    def _get_reason(self, indicators: Dict[str, float]) -> str:
        """Get reason for panic detection."""
        max_indicator = max(indicators.items(), key=lambda x: x[1])
        
        reasons = {
            "rapid_arm_movement": "Rapid arm movements detected",
            "high_variance": "High movement variance detected",
            "erratic_movement": "Erratic movement patterns detected"
        }
        
        return reasons.get(max_indicator[0], "Multiple indicators detected")
    
    def reset(self):
        """Reset pose history."""
        self.pose_history.clear()
        self.velocity_history.clear()
