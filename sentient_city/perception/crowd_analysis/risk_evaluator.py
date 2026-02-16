"""
Advanced Risk Evaluator for Crowd Safety Analysis.

Implements crowd-aware risk detection with per-person and aggregated
risk scoring, dynamic thresholds, and behavior-based assessment.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque
from loguru import logger

from .person_tracker import TrackedPerson
from .motion_analyzer import CrowdMotionMetrics, MotionFeatures
from sentient_city.core import get_logger, get_config


@dataclass
class RiskThresholds:
    """Dynamic risk thresholds for different crowd scenarios."""
    
    # Large crowd thresholds
    large_crowd_min_people: int = 20
    large_crowd_speed_spike_threshold: float = 25.0
    large_crowd_direction_variance_threshold: float = 0.8
    large_crowd_compression_threshold: float = 0.5
    large_crowd_chaos_threshold: float = 0.7
    
    # Small crowd thresholds
    small_crowd_max_people: int = 10
    small_crowd_abnormal_speed_threshold: float = 30.0
    small_crowd_collision_threshold: float = 0.6
    small_crowd_fighting_threshold: float = 0.8
    
    # General thresholds
    global_risk_threshold: float = 0.6
    alert_sustain_frames: int = 5
    alert_cooldown_frames: int = 30


@dataclass
class PersonRisk:
    """Risk assessment for a single person."""
    
    track_id: int
    risk_level: str  # Low, Medium, High
    risk_score: float
    risk_factors: Dict[str, float]
    is_anomalous: bool
    collision_risk: float
    panic_probability: float
    
    
@dataclass
class CrowdRisk:
    """Global crowd risk assessment."""
    
    risk_level: str  # Low, Medium, High, Critical
    risk_score: float
    crowd_size: int
    risk_factors: Dict[str, float]
    alert_triggered: bool
    alert_confidence: float
    recommended_actions: List[str]


class RiskEvaluator:
    """
    Advanced risk evaluator for crowd safety analysis.
    
    Features:
    - Crowd-size aware risk detection
    - Per-person and aggregated risk scoring
    - Dynamic threshold adjustment
    - Behavior-based anomaly detection
    - Collision and panic detection
    """
    
    def __init__(
        self,
        thresholds: Optional[RiskThresholds] = None,
        risk_history_length: int = 30,
        anomaly_threshold: float = 0.7
    ):
        """
        Initialize risk evaluator.
        
        Args:
            thresholds: Risk thresholds for different scenarios
            risk_history_length: Length of risk history to maintain
            anomaly_threshold: Threshold for anomaly detection
        """
        self.thresholds = thresholds or RiskThresholds()
        self.risk_history_length = risk_history_length
        self.anomaly_threshold = anomaly_threshold
        
        self.logger = get_logger(__name__)
        
        # Risk history
        self.risk_history = deque(maxlen=risk_history_length)
        self.alert_history = deque(maxlen=100)
        
        # Alert state
        self.alert_cooldown_counter = 0
        self.sustained_risk_counter = 0
        self.last_alert_frame = -1000
        
        # Risk factors weights
        self.risk_weights = {
            'speed': 0.25,
            'direction_variance': 0.20,
            'compression': 0.20,
            'chaos': 0.15,
            'collisions': 0.10,
            'anomalies': 0.10
        }
    
    def evaluate_risk(
        self,
        tracked_persons: List[TrackedPerson],
        crowd_metrics: CrowdMotionMetrics,
        person_features: Dict[int, MotionFeatures],
        frame_id: int
    ) -> Tuple[CrowdRisk, Dict[int, PersonRisk]]:
        """
        Evaluate risk for current frame.
        
        Args:
            tracked_persons: List of tracked persons
            crowd_metrics: Global crowd motion metrics
            person_features: Per-person motion features
            frame_id: Current frame ID
            
        Returns:
            Tuple of (crowd risk, per-person risks)
        """
        crowd_size = len(tracked_persons)
        
        # Determine crowd scenario
        is_large_crowd = crowd_size >= self.thresholds.large_crowd_min_people
        is_small_crowd = crowd_size <= self.thresholds.small_crowd_max_people
        
        # Evaluate per-person risk
        person_risks = self._evaluate_person_risks(
            tracked_persons, person_features, is_small_crowd
        )
        
        # Evaluate crowd risk
        crowd_risk = self._evaluate_crowd_risk(
            tracked_persons, crowd_metrics, person_risks, is_large_crowd, is_small_crowd
        )
        
        # Update alert state
        self._update_alert_state(crowd_risk, frame_id)
        
        # Store in history
        self.risk_history.append({
            'frame_id': frame_id,
            'crowd_risk': crowd_risk,
            'person_risks': person_risks,
            'crowd_size': crowd_size
        })
        
        return crowd_risk, person_risks
    
    def _evaluate_person_risks(
        self,
        tracked_persons: List[TrackedPerson],
        person_features: Dict[int, MotionFeatures],
        is_small_crowd: bool
    ) -> Dict[int, PersonRisk]:
        """Evaluate risk for each person."""
        person_risks = {}
        
        for person in tracked_persons:
            features = person_features.get(person.track_id)
            if not features:
                continue
            
            # Calculate risk factors
            risk_factors = self._calculate_person_risk_factors(person, features, is_small_crowd)
            
            # Calculate overall risk score
            risk_score = self._calculate_person_risk_score(risk_factors, is_small_crowd)
            
            # Determine risk level
            risk_level = self._determine_risk_level(risk_score)
            
            # Detect anomalies
            is_anomalous = self._detect_person_anomaly(person, features)
            
            # Calculate collision risk
            collision_risk = self._calculate_collision_risk(person, tracked_persons)
            
            # Calculate panic probability
            panic_probability = self._calculate_panic_probability(features)
            
            person_risk = PersonRisk(
                track_id=person.track_id,
                risk_level=risk_level,
                risk_score=risk_score,
                risk_factors=risk_factors,
                is_anomalous=is_anomalous,
                collision_risk=collision_risk,
                panic_probability=panic_probability
            )
            
            person_risks[person.track_id] = person_risk
            
            # Update person's risk level
            person.risk_level = risk_level
            person.risk_score = risk_score
        
        return person_risks
    
    def _calculate_person_risk_factors(
        self,
        person: TrackedPerson,
        features: MotionFeatures,
        is_small_crowd: bool
    ) -> Dict[str, float]:
        """Calculate risk factors for a person."""
        factors = {}
        
        # Speed risk
        if is_small_crowd:
            # In small crowds, only very high speeds are risky
            factors['speed'] = min(features.velocity_magnitude / self.thresholds.small_crowd_abnormal_speed_threshold, 1.0)
        else:
            # In large crowds, moderate speeds can be risky
            factors['speed'] = min(features.velocity_magnitude / 20.0, 1.0)
        
        # Direction variance (chaotic movement)
        factors['direction_variance'] = min(features.direction_variance, 1.0)
        
        # Acceleration (sudden movements)
        factors['acceleration'] = min(features.acceleration / 50.0, 1.0)
        
        # Chaotic index
        factors['chaotic_index'] = features.chaotic_index
        
        # Flow density
        factors['flow_density'] = min(features.flow_density / 30.0, 1.0)
        
        return factors
    
    def _calculate_person_risk_score(
        self,
        risk_factors: Dict[str, float],
        is_small_crowd: bool
    ) -> float:
        """Calculate overall risk score for a person."""
        if is_small_crowd:
            # In small crowds, focus on abnormal behavior
            weights = {
                'speed': 0.3,
                'direction_variance': 0.15,
                'acceleration': 0.25,
                'chaotic_index': 0.2,
                'flow_density': 0.1
            }
        else:
            # In large crowds, focus on chaotic movement
            weights = {
                'speed': 0.2,
                'direction_variance': 0.3,
                'acceleration': 0.15,
                'chaotic_index': 0.25,
                'flow_density': 0.1
            }
        
        # Calculate weighted sum
        risk_score = sum(weights.get(factor, 0) * value for factor, value in risk_factors.items())
        
        return float(risk_score)
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level from risk score."""
        if risk_score >= 0.8:
            return "High"
        elif risk_score >= 0.5:
            return "Medium"
        else:
            return "Low"
    
    def _detect_person_anomaly(self, person: TrackedPerson, features: MotionFeatures) -> bool:
        """Detect anomalous behavior for a person."""
        # Check for sudden direction changes
        if features.chaotic_index > self.anomaly_threshold:
            return True
        
        # Check for unusual speed patterns
        if features.velocity_magnitude > 40.0:  # Very fast movement
            return True
        
        # Check for erratic acceleration
        if features.acceleration > 30.0:
            return True
        
        return False
    
    def _calculate_collision_risk(self, person: TrackedPerson, all_persons: List[TrackedPerson]) -> float:
        """Calculate collision risk with other persons."""
        collision_risk = 0.0
        
        for other in all_persons:
            if other.track_id == person.track_id:
                continue
            
            # Calculate distance
            distance = np.linalg.norm(person.center - other.center)
            
            # Check if too close
            min_distance = 50.0  # Minimum safe distance in pixels
            if distance < min_distance:
                # Calculate relative velocity
                if person.velocity is not None and other.velocity is not None:
                    relative_velocity = person.velocity - other.velocity
                    closing_speed = np.linalg.norm(relative_velocity)
                    
                    # Higher risk if moving towards each other
                    if closing_speed > 10.0:
                        collision_risk += (1.0 - distance / min_distance) * (closing_speed / 30.0)
                else:
                    collision_risk += (1.0 - distance / min_distance) * 0.5
        
        return min(collision_risk, 1.0)
    
    def _calculate_panic_probability(self, features: MotionFeatures) -> float:
        """Calculate panic probability based on motion features."""
        panic_score = 0.0
        
        # High speed
        if features.velocity_magnitude > 25.0:
            panic_score += 0.3
        
        # High direction variance
        if features.direction_variance > 0.6:
            panic_score += 0.3
        
        # High chaotic index
        if features.chaotic_index > 0.7:
            panic_score += 0.2
        
        # High acceleration
        if features.acceleration > 20.0:
            panic_score += 0.2
        
        return min(panic_score, 1.0)
    
    def _evaluate_crowd_risk(
        self,
        tracked_persons: List[TrackedPerson],
        crowd_metrics: CrowdMotionMetrics,
        person_risks: Dict[int, PersonRisk],
        is_large_crowd: bool,
        is_small_crowd: bool
    ) -> CrowdRisk:
        """Evaluate overall crowd risk."""
        crowd_size = len(tracked_persons)
        
        # Calculate crowd risk factors
        risk_factors = {}
        
        if is_large_crowd:
            # Large crowd risk factors
            risk_factors['speed_spike'] = min(crowd_metrics.avg_speed / self.thresholds.large_crowd_speed_spike_threshold, 1.0)
            risk_factors['direction_variance'] = min(crowd_metrics.direction_variance / self.thresholds.large_crowd_direction_variance_threshold, 1.0)
            risk_factors['compression'] = min(crowd_metrics.compression_density / self.thresholds.large_crowd_compression_threshold, 1.0)
            risk_factors['chaos'] = min(len(crowd_metrics.chaotic_regions) / 10.0, 1.0)
            risk_factors['entropy'] = min(crowd_metrics.crowd_entropy / 3.0, 1.0)
            
        elif is_small_crowd:
            # Small crowd risk factors (only trigger for abnormal behavior)
            high_risk_persons = sum(1 for risk in person_risks.values() if risk.risk_level == "High")
            anomalous_persons = sum(1 for risk in person_risks.values() if risk.is_anomalous)
            
            risk_factors['abnormal_behavior'] = min(high_risk_persons / max(crowd_size, 1), 1.0)
            risk_factors['anomalies'] = min(anomalous_persons / max(crowd_size, 1), 1.0)
            risk_factors['collisions'] = min(np.mean([risk.collision_risk for risk in person_risks.values()]) if person_risks else 0, 1.0)
            
        else:
            # Medium crowd - balanced approach
            risk_factors['speed'] = min(crowd_metrics.avg_speed / 20.0, 1.0)
            risk_factors['direction_variance'] = min(crowd_metrics.direction_variance / 0.5, 1.0)
            risk_factors['compression'] = min(crowd_metrics.compression_density / 0.3, 1.0)
            risk_factors['chaos'] = min(len(crowd_metrics.chaotic_regions) / 5.0, 1.0)
        
        # Calculate overall crowd risk score
        crowd_risk_score = self._calculate_crowd_risk_score(risk_factors, is_large_crowd, is_small_crowd)
        
        # Determine crowd risk level
        if crowd_risk_score >= 0.9:
            crowd_risk_level = "Critical"
        elif crowd_risk_score >= 0.7:
            crowd_risk_level = "High"
        elif crowd_risk_score >= 0.4:
            crowd_risk_level = "Medium"
        else:
            crowd_risk_level = "Low"
        
        # Generate recommended actions
        recommended_actions = self._generate_recommendations(crowd_risk_level, risk_factors, crowd_size)
        
        return CrowdRisk(
            risk_level=crowd_risk_level,
            risk_score=crowd_risk_score,
            crowd_size=crowd_size,
            risk_factors=risk_factors,
            alert_triggered=False,  # Will be updated by alert manager
            alert_confidence=0.0,   # Will be updated by alert manager
            recommended_actions=recommended_actions
        )
    
    def _calculate_crowd_risk_score(
        self,
        risk_factors: Dict[str, float],
        is_large_crowd: bool,
        is_small_crowd: bool
    ) -> float:
        """Calculate overall crowd risk score."""
        if is_large_crowd:
            weights = {
                'speed_spike': 0.25,
                'direction_variance': 0.25,
                'compression': 0.20,
                'chaos': 0.20,
                'entropy': 0.10
            }
        elif is_small_crowd:
            weights = {
                'abnormal_behavior': 0.4,
                'anomalies': 0.3,
                'collisions': 0.3
            }
        else:
            weights = {
                'speed': 0.25,
                'direction_variance': 0.25,
                'compression': 0.25,
                'chaos': 0.25
            }
        
        # Calculate weighted sum
        risk_score = sum(weights.get(factor, 0) * value for factor, value in risk_factors.items())
        
        return float(risk_score)
    
    def _generate_recommendations(
        self,
        risk_level: str,
        risk_factors: Dict[str, float],
        crowd_size: int
    ) -> List[str]:
        """Generate recommended actions based on risk assessment."""
        recommendations = []
        
        if risk_level == "Critical":
            recommendations.extend([
                "IMMEDIATE ATTENTION REQUIRED",
                "Consider crowd dispersal procedures",
                "Emergency services on standby",
                "Monitor for stampede conditions"
            ])
        elif risk_level == "High":
            recommendations.extend([
                "High risk detected - monitor closely",
                "Prepare crowd management interventions",
                "Increase surveillance density",
                "Alert security personnel"
            ])
        elif risk_level == "Medium":
            recommendations.extend([
                "Moderate risk - continue monitoring",
                "Review crowd flow patterns",
                "Consider preventive measures"
            ])
        
        # Specific recommendations based on risk factors
        if risk_factors.get('compression', 0) > 0.7:
            recommendations.append("High compression detected - create space")
        
        if risk_factors.get('chaos', 0) > 0.6:
            recommendations.append("Chaotic movement detected - restore order")
        
        if risk_factors.get('collisions', 0) > 0.5:
            recommendations.append("High collision risk - separate groups")
        
        return recommendations
    
    def _update_alert_state(self, crowd_risk: CrowdRisk, frame_id: int):
        """Update alert state based on crowd risk."""
        # Check cooldown
        if frame_id - self.last_alert_frame < self.thresholds.alert_cooldown_frames:
            self.alert_cooldown_counter -= 1
            return
        
        # Check if risk exceeds threshold
        if crowd_risk.risk_score >= self.thresholds.global_risk_threshold:
            self.sustained_risk_counter += 1
            
            # Trigger alert if sustained
            if self.sustained_risk_counter >= self.thresholds.alert_sustain_frames:
                crowd_risk.alert_triggered = True
                crowd_risk.alert_confidence = min(crowd_risk.risk_score, 1.0)
                self.last_alert_frame = frame_id
                self.sustained_risk_counter = 0
                self.alert_cooldown_counter = self.thresholds.alert_cooldown_frames
                
                self.logger.warning(f"ALERT TRIGGERED: {crowd_risk.risk_level} risk detected")
        else:
            self.sustained_risk_counter = max(0, self.sustained_risk_counter - 1)
    
    def get_risk_trends(self) -> Dict:
        """Get risk trends over time."""
        if len(self.risk_history) < 2:
            return {}
        
        recent_history = list(self.risk_history)[-10:]  # Last 10 frames
        
        # Calculate trends
        risk_scores = [h['crowd_risk'].risk_score for h in recent_history]
        crowd_sizes = [h['crowd_size'] for h in recent_history]
        
        return {
            'avg_risk_score': float(np.mean(risk_scores)),
            'risk_trend': 'increasing' if risk_scores[-1] > risk_scores[0] else 'decreasing',
            'avg_crowd_size': float(np.mean(crowd_sizes)),
            'risk_stability': float(1.0 / (1.0 + np.var(risk_scores))),
            'alert_frequency': len([h for h in self.risk_history if h['crowd_risk'].alert_triggered])
        }
    
    def reset(self):
        """Reset risk evaluator state."""
        self.risk_history.clear()
        self.alert_history.clear()
        self.alert_cooldown_counter = 0
        self.sustained_risk_counter = 0
        self.last_alert_frame = -1000
