"""
Unified risk engine.

Combines all risk assessment modules into a unified risk scoring system.
"""

from typing import Dict, List, Optional
from loguru import logger

from ..crowd_prediction.crowd_crush_predictor import CrowdCrushPredictor
from ..anomaly_detection.anomaly_detector import AnomalyDetector
from .accident_detector import AccidentDetector
from ...utils.config import get_config


class RiskEngine:
    """
    Unified risk engine.
    
    Combines crowd crush, anomaly, and accident detection into
    a unified risk assessment system.
    """
    
    def __init__(self):
        """Initialize risk engine."""
        self.crowd_crush_predictor = CrowdCrushPredictor()
        self.anomaly_detector = AnomalyDetector()
        self.accident_detector = AccidentDetector()
        
        self.config = get_config()
    
    def assess_risk(
        self,
        detections: List[Dict],
        tracks: List[Dict],
        density_map: Optional[any] = None,
        behavior_embeddings: Optional[List] = None,
        context: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Assess overall risk from multiple sources.
        
        Args:
            detections: List of detections
            tracks: List of tracks
            density_map: Optional crowd density map
            behavior_embeddings: Optional behavior embeddings
            context: Optional context dictionary
        
        Returns:
            Dictionary with comprehensive risk assessment
        """
        risks = {}
        
        # Crowd crush risk
        if density_map is not None:
            crush_result = self.crowd_crush_predictor.predict(
                density_map, tracks
            )
            risks["crowd_crush"] = crush_result
        
        # Anomaly detection
        if behavior_embeddings and len(behavior_embeddings) > 0:
            # Use average embedding
            avg_embedding = sum(behavior_embeddings) / len(behavior_embeddings)
            anomaly_result = self.anomaly_detector.detect_behavior_anomaly(
                avg_embedding, behavior_embeddings
            )
            risks["anomaly"] = anomaly_result
        
        # Accident detection
        accident_result = self.accident_detector.detect(detections, tracks)
        risks["accident"] = accident_result
        
        # Calculate overall risk score
        overall_risk = self._calculate_overall_risk(risks)
        
        # Determine risk level
        risk_level = self._determine_risk_level(overall_risk)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(risks, overall_risk)
        
        return {
            "overall_risk_score": overall_risk,
            "risk_level": risk_level,
            "component_risks": risks,
            "recommendations": recommendations,
            "timestamp": None  # Will be set by caller
        }
    
    def _calculate_overall_risk(self, risks: Dict) -> float:
        """Calculate overall risk score from component risks."""
        weights = {
            "crowd_crush": 0.4,
            "anomaly": 0.3,
            "accident": 0.3
        }
        
        overall_risk = 0.0
        
        if "crowd_crush" in risks:
            overall_risk += risks["crowd_crush"]["risk_score"] * weights["crowd_crush"]
        
        if "anomaly" in risks:
            overall_risk += risks["anomaly"]["anomaly_score"] * weights["anomaly"]
        
        if "accident" in risks:
            overall_risk += risks["accident"]["accident_probability"] * weights["accident"]
        
        return min(1.0, overall_risk)
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level from score."""
        if risk_score >= 0.8:
            return "CRITICAL"
        elif risk_score >= 0.6:
            return "HIGH"
        elif risk_score >= 0.4:
            return "MODERATE"
        elif risk_score >= 0.2:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _generate_recommendations(
        self,
        risks: Dict,
        overall_risk: float
    ) -> List[str]:
        """Generate recommendations based on risks."""
        recommendations = []
        
        if overall_risk >= 0.8:
            recommendations.append("IMMEDIATE ACTION REQUIRED - Deploy emergency response")
        
        if "crowd_crush" in risks and risks["crowd_crush"]["is_high_risk"]:
            recommendations.append(risks["crowd_crush"]["recommendation"])
        
        if "anomaly" in risks and risks["anomaly"]["is_anomaly"]:
            recommendations.append("Anomalous behavior detected - Investigate immediately")
        
        if "accident" in risks and risks["accident"]["is_accident"]:
            recommendations.append("ACCIDENT DETECTED - Dispatch emergency services")
        
        if not recommendations:
            recommendations.append("No immediate action required - Continue monitoring")
        
        return recommendations
