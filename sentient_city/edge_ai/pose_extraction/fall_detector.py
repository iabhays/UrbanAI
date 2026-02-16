"""
Fall detection based on pose analysis.

Detects falls by analyzing pose keypoints and movement patterns.
"""

import numpy as np
from typing import Dict, List, Optional
from loguru import logger

from .pose_detector import PoseDetector


class FallDetector:
    """
    Fall detection using pose analysis.
    
    Detects falls by analyzing:
    - Vertical position of keypoints
    - Body orientation
    - Movement velocity
    """
    
    def __init__(self, threshold: float = 0.7):
        """
        Initialize fall detector.
        
        Args:
            threshold: Fall detection threshold (0-1)
        """
        self.threshold = threshold
        self.pose_detector = PoseDetector()
        self.pose_history: List[Dict] = []
        self.history_size = 10
    
    def detect(self, pose: Dict, image_height: int) -> Dict[str, any]:
        """
        Detect fall from pose.
        
        Args:
            pose: Pose dictionary from PoseDetector
            image_height: Image height for normalization
        
        Returns:
            Dictionary with:
            - is_fall: Boolean indicating if fall detected
            - confidence: Confidence score (0-1)
            - reason: Reason for detection
        """
        if len(pose["keypoints"]) == 0:
            return {
                "is_fall": False,
                "confidence": 0.0,
                "reason": "No pose detected"
            }
        
        # Add to history
        self.pose_history.append(pose)
        if len(self.pose_history) > self.history_size:
            self.pose_history.pop(0)
        
        # Calculate fall indicators
        indicators = self._calculate_indicators(pose, image_height)
        
        # Combine indicators
        confidence = self._combine_indicators(indicators)
        
        is_fall = confidence >= self.threshold
        
        reason = "Normal"
        if is_fall:
            reason = self._get_reason(indicators)
        
        return {
            "is_fall": is_fall,
            "confidence": float(confidence),
            "reason": reason,
            "indicators": indicators
        }
    
    def _calculate_indicators(self, pose: Dict, image_height: int) -> Dict[str, float]:
        """Calculate fall detection indicators."""
        indicators = {}
        
        # Get key keypoints
        left_hip = self.pose_detector.get_keypoint(pose, "left_hip")
        right_hip = self.pose_detector.get_keypoint(pose, "right_hip")
        left_shoulder = self.pose_detector.get_keypoint(pose, "left_shoulder")
        right_shoulder = self.pose_detector.get_keypoint(pose, "right_shoulder")
        left_ankle = self.pose_detector.get_keypoint(pose, "left_ankle")
        right_ankle = self.pose_detector.get_keypoint(pose, "right_ankle")
        
        # Indicator 1: Vertical position (hips low relative to image)
        if left_hip and right_hip:
            avg_hip_y = (left_hip[1] + right_hip[1]) / 2
            normalized_y = avg_hip_y / image_height
            indicators["low_position"] = max(0, (normalized_y - 0.5) * 2)  # Higher if hips are low
        else:
            indicators["low_position"] = 0.0
        
        # Indicator 2: Body orientation (horizontal vs vertical)
        if left_shoulder and right_shoulder and left_hip and right_hip:
            shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
            hip_center_y = (left_hip[1] + right_hip[1]) / 2
            
            vertical_distance = abs(shoulder_center_y - hip_center_y)
            horizontal_distance = abs(left_shoulder[0] - right_shoulder[0])
            
            if horizontal_distance > 0:
                orientation_ratio = vertical_distance / horizontal_distance
                # Low ratio indicates horizontal (lying down)
                indicators["horizontal_orientation"] = max(0, 1 - orientation_ratio / 2)
            else:
                indicators["horizontal_orientation"] = 0.0
        else:
            indicators["horizontal_orientation"] = 0.0
        
        # Indicator 3: Ankle position relative to hips
        if left_ankle and right_ankle and left_hip and right_hip:
            avg_ankle_y = (left_ankle[1] + right_ankle[1]) / 2
            avg_hip_y = (left_hip[1] + right_hip[1]) / 2
            
            # If ankles are above hips, person might be lying down
            if avg_ankle_y < avg_hip_y:
                indicators["ankle_position"] = 0.8
            else:
                indicators["ankle_position"] = 0.0
        else:
            indicators["ankle_position"] = 0.0
        
        # Indicator 4: Movement velocity (if history available)
        if len(self.pose_history) >= 2:
            prev_pose = self.pose_history[-2]
            curr_pose = pose
            
            prev_hip = self.pose_detector.get_keypoint(prev_pose, "left_hip")
            curr_hip = self.pose_detector.get_keypoint(curr_pose, "left_hip")
            
            if prev_hip and curr_hip:
                velocity = np.sqrt(
                    (curr_hip[0] - prev_hip[0])**2 + (curr_hip[1] - prev_hip[1])**2
                )
                # High velocity might indicate fall
                indicators["high_velocity"] = min(1.0, velocity / 50.0)
            else:
                indicators["high_velocity"] = 0.0
        else:
            indicators["high_velocity"] = 0.0
        
        return indicators
    
    def _combine_indicators(self, indicators: Dict[str, float]) -> float:
        """Combine indicators into final confidence score."""
        # Weighted combination
        weights = {
            "low_position": 0.3,
            "horizontal_orientation": 0.3,
            "ankle_position": 0.2,
            "high_velocity": 0.2
        }
        
        confidence = sum(
            indicators.get(key, 0.0) * weights.get(key, 0.0)
            for key in weights
        )
        
        return min(1.0, confidence)
    
    def _get_reason(self, indicators: Dict[str, float]) -> str:
        """Get reason for fall detection."""
        max_indicator = max(indicators.items(), key=lambda x: x[1])
        
        reasons = {
            "low_position": "Person appears to be on ground level",
            "horizontal_orientation": "Body is in horizontal position",
            "ankle_position": "Unusual ankle position detected",
            "high_velocity": "Rapid movement detected"
        }
        
        return reasons.get(max_indicator[0], "Multiple indicators detected")
    
    def reset(self):
        """Reset pose history."""
        self.pose_history.clear()
