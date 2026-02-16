"""
Advanced Alert Manager for Crowd Safety System.

Implements dynamic alert triggering with cooldown periods,
confidence scoring, and multi-level alert classification.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta
from loguru import logger

from .risk_evaluator import CrowdRisk, PersonRisk
from sentient_city.core import get_logger, get_config


@dataclass
class Alert:
    """Alert data structure."""
    
    alert_id: str
    timestamp: datetime
    alert_type: str  # Risk, Anomaly, System
    severity: str  # Low, Medium, High, Critical
    confidence: float
    message: str
    crowd_risk: CrowdRisk
    person_risks: Dict[int, PersonRisk]
    frame_id: int
    location: Optional[Tuple[float, float]] = None
    duration_frames: int = 0
    acknowledged: bool = False
    resolved: bool = False
    
    
@dataclass
class AlertConfig:
    """Alert system configuration."""
    
    # Alert thresholds
    min_confidence_threshold: float = 0.6
    high_risk_threshold: float = 0.7
    critical_risk_threshold: float = 0.9
    
    # Timing
    cooldown_seconds: int = 30
    sustained_risk_frames: int = 5
    max_alert_duration_seconds: int = 300
    
    # Alert types
    enable_risk_alerts: bool = True
    enable_anomaly_alerts: bool = True
    enable_system_alerts: bool = True
    
    # Filtering
    min_crowd_size_for_alerts: int = 3
    false_positive_filter_threshold: float = 0.3


class AlertManager:
    """
    Advanced alert manager for crowd safety monitoring.
    
    Features:
    - Dynamic alert triggering with confidence scoring
    - Cooldown system to prevent spam
    - Multi-level severity classification
    - Alert persistence and history
    - Custom alert callbacks
    - False positive filtering
    """
    
    def __init__(
        self,
        config: Optional[AlertConfig] = None,
        max_alert_history: int = 1000,
        alert_callbacks: Optional[List[Callable]] = None
    ):
        """
        Initialize alert manager.
        
        Args:
            config: Alert configuration
            max_alert_history: Maximum number of alerts to keep in history
            alert_callbacks: List of callback functions for alerts
        """
        self.config = config or AlertConfig()
        self.max_alert_history = max_alert_history
        self.alert_callbacks = alert_callbacks or []
        
        self.logger = get_logger(__name__)
        
        # Alert state
        self.alert_history: deque = deque(maxlen=max_alert_history)
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_counter = 0
        
        # Cooldown tracking
        self.last_alert_times: Dict[str, datetime] = {}
        self.sustained_risk_counter = 0
        
        # Statistics
        self.total_alerts = 0
        self.false_positive_count = 0
        self.alert_resolution_times: List[float] = []
        
        # Alert patterns
        self.alert_patterns: Dict[str, List[datetime]] = {}
        
    def process_frame_alerts(
        self,
        crowd_risk: CrowdRisk,
        person_risks: Dict[int, PersonRisk],
        frame_id: int,
        frame_timestamp: Optional[datetime] = None
    ) -> List[Alert]:
        """
        Process alerts for current frame.
        
        Args:
            crowd_risk: Current crowd risk assessment
            person_risks: Per-person risk assessments
            frame_id: Current frame ID
            frame_timestamp: Frame timestamp
            
        Returns:
            List of new alerts triggered
        """
        if frame_timestamp is None:
            frame_timestamp = datetime.now()
        
        new_alerts = []
        
        # Update sustained risk counter
        if crowd_risk.risk_score >= self.config.high_risk_threshold:
            self.sustained_risk_counter += 1
        else:
            self.sustained_risk_counter = max(0, self.sustained_risk_counter - 1)
        
        # Check for risk alerts
        if self.config.enable_risk_alerts:
            risk_alert = self._check_risk_alert(crowd_risk, person_risks, frame_id, frame_timestamp)
            if risk_alert:
                new_alerts.append(risk_alert)
        
        # Check for anomaly alerts
        if self.config.enable_anomaly_alerts:
            anomaly_alert = self._check_anomaly_alert(crowd_risk, person_risks, frame_id, frame_timestamp)
            if anomaly_alert:
                new_alerts.append(anomaly_alert)
        
        # Update active alerts
        self._update_active_alerts(frame_id, frame_timestamp)
        
        # Process callbacks
        for alert in new_alerts:
            self._process_alert_callbacks(alert)
        
        return new_alerts
    
    def _check_risk_alert(
        self,
        crowd_risk: CrowdRisk,
        person_risks: Dict[int, PersonRisk],
        frame_id: int,
        timestamp: datetime
    ) -> Optional[Alert]:
        """Check if risk alert should be triggered."""
        # Check minimum crowd size
        if crowd_risk.crowd_size < self.config.min_crowd_size_for_alerts:
            return None
        
        # Check confidence threshold
        if crowd_risk.risk_score < self.config.min_confidence_threshold:
            return None
        
        # Check sustained risk
        if self.sustained_risk_counter < self.config.sustained_risk_frames:
            return None
        
        # Check cooldown
        if not self._check_cooldown("risk", timestamp):
            return None
        
        # Apply false positive filtering
        if self._is_likely_false_positive(crowd_risk, person_risks):
            self.false_positive_count += 1
            return None
        
        # Determine severity
        if crowd_risk.risk_score >= self.config.critical_risk_threshold:
            severity = "Critical"
        elif crowd_risk.risk_score >= self.config.high_risk_threshold:
            severity = "High"
        else:
            severity = "Medium"
        
        # Generate alert message
        message = self._generate_risk_alert_message(crowd_risk, person_risks)
        
        # Create alert
        alert = Alert(
            alert_id=self._generate_alert_id(),
            timestamp=timestamp,
            alert_type="Risk",
            severity=severity,
            confidence=crowd_risk.risk_score,
            message=message,
            crowd_risk=crowd_risk,
            person_risks=person_risks,
            frame_id=frame_id
        )
        
        # Add to active alerts
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        self.total_alerts += 1
        
        # Update cooldown
        self._update_cooldown("risk", timestamp)
        
        self.logger.warning(f"Risk Alert triggered: {severity} - {message}")
        
        return alert
    
    def _check_anomaly_alert(
        self,
        crowd_risk: CrowdRisk,
        person_risks: Dict[int, PersonRisk],
        frame_id: int,
        timestamp: datetime
    ) -> Optional[Alert]:
        """Check if anomaly alert should be triggered."""
        # Count anomalous persons
        anomalous_count = sum(1 for risk in person_risks.values() if risk.is_anomalous)
        
        if anomalous_count == 0:
            return None
        
        # Check if anomaly rate is significant
        anomaly_rate = anomalous_count / max(len(person_risks), 1)
        if anomaly_rate < 0.3:  # Less than 30% anomalous
            return None
        
        # Check cooldown
        if not self._check_cooldown("anomaly", timestamp):
            return None
        
        # Determine severity based on anomaly rate
        if anomaly_rate >= 0.7:
            severity = "High"
        elif anomaly_rate >= 0.5:
            severity = "Medium"
        else:
            severity = "Low"
        
        # Generate alert message
        message = f"Anomalous behavior detected: {anomalous_count} persons showing unusual patterns"
        
        # Create alert
        alert = Alert(
            alert_id=self._generate_alert_id(),
            timestamp=timestamp,
            alert_type="Anomaly",
            severity=severity,
            confidence=anomaly_rate,
            message=message,
            crowd_risk=crowd_risk,
            person_risks=person_risks,
            frame_id=frame_id
        )
        
        # Add to active alerts
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        self.total_alerts += 1
        
        # Update cooldown
        self._update_cooldown("anomaly", timestamp)
        
        self.logger.warning(f"Anomaly Alert triggered: {severity} - {message}")
        
        return alert
    
    def _generate_risk_alert_message(self, crowd_risk: CrowdRisk, person_risks: Dict[int, PersonRisk]) -> str:
        """Generate descriptive risk alert message."""
        messages = []
        
        # Base message
        messages.append(f"{crowd_risk.risk_level} crowd risk detected")
        messages.append(f"Risk score: {crowd_risk.risk_score:.2f}")
        messages.append(f"Crowd size: {crowd_risk.crowd_size} persons")
        
        # Add specific risk factors
        high_risk_factors = [factor for factor, value in crowd_risk.risk_factors.items() if value > 0.7]
        if high_risk_factors:
            messages.append(f"High risk factors: {', '.join(high_risk_factors)}")
        
        # Add person risk summary
        high_risk_persons = sum(1 for risk in person_risks.values() if risk.risk_level == "High")
        if high_risk_persons > 0:
            messages.append(f"{high_risk_persons} persons at high risk")
        
        return " | ".join(messages)
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID."""
        self.alert_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"ALERT_{timestamp}_{self.alert_counter:04d}"
    
    def _check_cooldown(self, alert_type: str, timestamp: datetime) -> bool:
        """Check if alert type is in cooldown period."""
        if alert_type not in self.last_alert_times:
            return True
        
        time_since_last = timestamp - self.last_alert_times[alert_type]
        return time_since_last.total_seconds() >= self.config.cooldown_seconds
    
    def _update_cooldown(self, alert_type: str, timestamp: datetime):
        """Update cooldown timestamp for alert type."""
        self.last_alert_times[alert_type] = timestamp
    
    def _is_likely_false_positive(
        self,
        crowd_risk: CrowdRisk,
        person_risks: Dict[int, PersonRisk]
    ) -> bool:
        """Apply false positive filtering logic."""
        # Check for inconsistent risk patterns
        if crowd_risk.risk_score > 0.8 and len(person_risks) < 3:
            return True
        
        # Check for sudden risk spikes without supporting factors
        high_risk_persons = sum(1 for risk in person_risks.values() if risk.risk_score > 0.7)
        if crowd_risk.risk_score > 0.7 and high_risk_persons == 0:
            return True
        
        # Check for low confidence in all risk factors
        max_factor_confidence = max(crowd_risk.risk_factors.values()) if crowd_risk.risk_factors else 0
        if max_factor_confidence < self.config.false_positive_filter_threshold:
            return True
        
        return False
    
    def _update_active_alerts(self, frame_id: int, timestamp: datetime):
        """Update active alerts and check for resolution."""
        alerts_to_remove = []
        
        for alert_id, alert in self.active_alerts.items():
            alert.duration_frames += 1
            
            # Check if alert should be resolved
            duration_seconds = (timestamp - alert.timestamp).total_seconds()
            if duration_seconds > self.config.max_alert_duration_seconds:
                alert.resolved = True
                alerts_to_remove.append(alert_id)
                self.alert_resolution_times.append(duration_seconds)
        
        # Remove resolved alerts
        for alert_id in alerts_to_remove:
            del self.active_alerts[alert_id]
    
    def _process_alert_callbacks(self, alert: Alert):
        """Process alert through registered callbacks."""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an active alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            self.logger.info(f"Alert {alert_id} acknowledged")
            return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Manually resolve an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            duration = (datetime.now() - alert.timestamp).total_seconds()
            self.alert_resolution_times.append(duration)
            del self.active_alerts[alert_id]
            self.logger.info(f"Alert {alert_id} resolved after {duration:.1f} seconds")
            return True
        return False
    
    def get_alert_statistics(self) -> Dict:
        """Get alert system statistics."""
        # Calculate alert frequency by hour
        current_time = datetime.now()
        recent_alerts = [
            alert for alert in self.alert_history
            if (current_time - alert.timestamp).total_seconds() < 3600  # Last hour
        ]
        
        # Calculate average resolution time
        avg_resolution_time = (
            np.mean(self.alert_resolution_times) if self.alert_resolution_times else 0
        )
        
        # False positive rate
        false_positive_rate = (
            self.false_positive_count / max(self.total_alerts, 1) * 100
        )
        
        return {
            "total_alerts": self.total_alerts,
            "active_alerts": len(self.active_alerts),
            "alerts_last_hour": len(recent_alerts),
            "false_positive_count": self.false_positive_count,
            "false_positive_rate": false_positive_rate,
            "avg_resolution_time_seconds": avg_resolution_time,
            "sustained_risk_counter": self.sustained_risk_counter,
            "alert_types": {
                alert_type: len([a for a in self.alert_history if a.alert_type == alert_type])
                for alert_type in ["Risk", "Anomaly", "System"]
            },
            "severity_distribution": {
                severity: len([a for a in self.alert_history if a.severity == severity])
                for severity in ["Low", "Medium", "High", "Critical"]
            }
        }
    
    def get_recent_alerts(self, limit: int = 10) -> List[Alert]:
        """Get most recent alerts."""
        return list(self.alert_history)[-limit:]
    
    def get_active_alerts(self) -> List[Alert]:
        """Get currently active alerts."""
        return list(self.active_alerts.values())
    
    def add_alert_callback(self, callback: Callable):
        """Add a new alert callback function."""
        self.alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: Callable):
        """Remove an alert callback function."""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
    
    def reset(self):
        """Reset alert manager state."""
        self.alert_history.clear()
        self.active_alerts.clear()
        self.last_alert_times.clear()
        self.sustained_risk_counter = 0
        self.alert_counter = 0
        self.total_alerts = 0
        self.false_positive_count = 0
        self.alert_resolution_times.clear()
        self.alert_patterns.clear()
