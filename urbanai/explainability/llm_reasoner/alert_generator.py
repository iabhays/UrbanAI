"""
Alert generation module.

Generates natural language alerts from risk assessments.
"""

from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger

from .llm_reasoner import LLMReasoner
from ...utils.config import get_config


class AlertGenerator:
    """
    Alert generator.
    
    Generates natural language alerts from risk assessments.
    """
    
    def __init__(self, llm_reasoner: Optional[LLMReasoner] = None):
        """
        Initialize alert generator.
        
        Args:
            llm_reasoner: Optional LLM reasoner instance
        """
        self.llm_reasoner = llm_reasoner or LLMReasoner()
        self.config = get_config()
    
    def generate_alert(
        self,
        risk_assessment: Dict,
        camera_id: Optional[str] = None,
        location: Optional[str] = None
    ) -> Dict:
        """
        Generate alert from risk assessment.
        
        Args:
            risk_assessment: Risk assessment dictionary
            camera_id: Optional camera ID
            location: Optional location name
        
        Returns:
            Alert dictionary
        """
        risk_level = risk_assessment.get("risk_level", "UNKNOWN")
        risk_score = risk_assessment.get("overall_risk_score", 0.0)
        component_risks = risk_assessment.get("component_risks", {})
        recommendations = risk_assessment.get("recommendations", [])
        
        # Determine alert severity
        severity = self._determine_severity(risk_level, risk_score)
        
        # Generate explanation
        explanation = self._generate_explanation(
            risk_assessment, camera_id, location
        )
        
        # Build alert
        alert = {
            "alert_id": f"alert_{datetime.now().timestamp()}",
            "timestamp": datetime.now().isoformat(),
            "severity": severity,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "camera_id": camera_id,
            "location": location,
            "explanation": explanation,
            "component_risks": component_risks,
            "recommendations": recommendations,
            "acknowledged": False
        }
        
        return alert
    
    def _determine_severity(self, risk_level: str, risk_score: float) -> str:
        """Determine alert severity."""
        if risk_level == "CRITICAL" or risk_score >= 0.9:
            return "CRITICAL"
        elif risk_level == "HIGH" or risk_score >= 0.7:
            return "HIGH"
        elif risk_level == "MODERATE" or risk_score >= 0.5:
            return "MODERATE"
        else:
            return "LOW"
    
    def _generate_explanation(
        self,
        risk_assessment: Dict,
        camera_id: Optional[str],
        location: Optional[str]
    ) -> str:
        """Generate explanation for alert."""
        risk_level = risk_assessment.get("risk_level", "UNKNOWN")
        component_risks = risk_assessment.get("component_risks", {})
        
        # Build explanation
        explanation_parts = []
        
        if location:
            explanation_parts.append(f"Location: {location}")
        if camera_id:
            explanation_parts.append(f"Camera: {camera_id}")
        
        explanation_parts.append(f"Risk Level: {risk_level}")
        
        # Add component explanations
        if "crowd_crush" in component_risks:
            crush_data = component_risks["crowd_crush"]
            if crush_data.get("is_high_risk"):
                explanation_parts.append(
                    f"High crowd crush risk detected (score: {crush_data.get('risk_score', 0.0):.2f})"
                )
        
        if "accident" in component_risks:
            accident_data = component_risks["accident"]
            if accident_data.get("is_accident"):
                explanation_parts.append("Traffic accident detected")
        
        if "anomaly" in component_risks:
            anomaly_data = component_risks["anomaly"]
            if anomaly_data.get("is_anomaly"):
                explanation_parts.append("Anomalous behavior detected")
        
        # Use LLM if available
        if self.llm_reasoner and self.llm_reasoner.client:
            try:
                llm_explanation = self.llm_reasoner.explain(
                    "risk_assessment",
                    risk_assessment,
                    {"camera_id": camera_id, "location": location}
                )
                explanation_parts.append(llm_explanation)
            except Exception as e:
                logger.error(f"LLM explanation error: {e}")
        
        return " | ".join(explanation_parts)
