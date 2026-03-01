"""
Event schemas for Kafka message serialization.

Defines event schemas for type-safe event publishing.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import json


@dataclass
class DetectionEvent:
    """Detection event schema."""
    event_id: str
    timestamp: str
    camera_id: str
    frame_id: int
    detections: List[Dict[str, Any]]
    crowd_density: Optional[Dict[str, Any]] = None
    behavior_embeddings: Optional[List[List[float]]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict())


@dataclass
class TrackEvent:
    """Track event schema."""
    event_id: str
    timestamp: str
    camera_id: str
    tracks: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict())


@dataclass
class AlertEvent:
    """Alert event schema."""
    event_id: str
    timestamp: str
    alert_id: str
    severity: str
    risk_level: str
    risk_score: float
    camera_id: Optional[str] = None
    location: Optional[str] = None
    explanation: Optional[str] = None
    recommendations: Optional[List[str]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict())


@dataclass
class BehaviorEvent:
    """Behavior event schema."""
    event_id: str
    timestamp: str
    camera_id: str
    track_id: int
    behavior_type: str
    confidence: float
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict())


class EventSchemaRegistry:
    """
    Event schema registry.
    
    Manages event schemas and validation.
    """
    
    def __init__(self):
        """Initialize schema registry."""
        self.schemas = {
            "detection": DetectionEvent,
            "track": TrackEvent,
            "alert": AlertEvent,
            "behavior": BehaviorEvent
        }
    
    def validate_event(self, event_type: str, data: Dict) -> bool:
        """
        Validate event data against schema.
        
        Args:
            event_type: Event type
            data: Event data
        
        Returns:
            True if valid
        """
        if event_type not in self.schemas:
            return False
        
        schema_class = self.schemas[event_type]
        
        try:
            # Try to create instance (validates required fields)
            schema_class(**data)
            return True
        except Exception:
            return False
    
    def create_event(self, event_type: str, **kwargs) -> Any:
        """
        Create event instance.
        
        Args:
            event_type: Event type
            **kwargs: Event data
        
        Returns:
            Event instance
        """
        if event_type not in self.schemas:
            raise ValueError(f"Unknown event type: {event_type}")
        
        schema_class = self.schemas[event_type]
        
        # Generate event_id if not provided
        if "event_id" not in kwargs:
            kwargs["event_id"] = f"{event_type}_{datetime.utcnow().timestamp()}"
        
        # Generate timestamp if not provided
        if "timestamp" not in kwargs:
            kwargs["timestamp"] = datetime.utcnow().isoformat()
        
        return schema_class(**kwargs)
