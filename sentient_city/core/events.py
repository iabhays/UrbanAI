"""
Event schema definitions and validation.

Provides type-safe event schemas for Kafka messaging
with validation and serialization support.
"""

import json
from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid


class EventType(str, Enum):
    """Event type enumeration."""
    DETECTION = "detection"
    TRACK = "track"
    POSE = "pose"
    BEHAVIOR = "behavior"
    ALERT = "alert"
    RISK_ASSESSMENT = "risk_assessment"
    EXPERIMENT = "experiment"
    MODEL_UPDATE = "model_update"


class EventPriority(str, Enum):
    """Event priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class EventMetadata(BaseModel):
    """Event metadata."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    event_type: EventType = EventType.DETECTION
    priority: EventPriority = EventPriority.NORMAL
    source: str = "unknown"
    correlation_id: Optional[str] = None
    version: str = "1.0.0"
    
    class Config:
        use_enum_values = True


class EventSchema(BaseModel):
    """
    Base event schema with validation.
    
    All events inherit from this base schema.
    """
    metadata: EventMetadata = Field(default_factory=lambda: EventMetadata())
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.dict()
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        return self.json()
    
    @classmethod
    def from_json(cls, json_str: str) -> "EventSchema":
        """Deserialize from JSON."""
        data = json.loads(json_str)
        return cls(**data)


class DetectionEvent(EventSchema):
    """Detection event schema."""
    camera_id: str
    frame_id: int
    detections: List[Dict[str, Any]] = Field(default_factory=list)
    crowd_density: Optional[Dict[str, Any]] = None
    behavior_embeddings: Optional[List[List[float]]] = None
    
    metadata: EventMetadata = Field(
        default_factory=lambda: EventMetadata(event_type=EventType.DETECTION)
    )


class TrackEvent(EventSchema):
    """Track event schema."""
    camera_id: str
    tracks: List[Dict[str, Any]] = Field(default_factory=list)
    frame_id: int
    
    metadata: EventMetadata = Field(
        default_factory=lambda: EventMetadata(event_type=EventType.TRACK)
    )


class PoseEvent(EventSchema):
    """Pose event schema."""
    camera_id: str
    track_id: Optional[int] = None
    poses: List[Dict[str, Any]] = Field(default_factory=list)
    behaviors: List[Dict[str, Any]] = Field(default_factory=list)
    
    metadata: EventMetadata = Field(
        default_factory=lambda: EventMetadata(event_type=EventType.POSE)
    )


class BehaviorEvent(EventSchema):
    """Behavior event schema."""
    camera_id: str
    track_id: int
    behavior_type: str
    confidence: float
    details: Optional[Dict[str, Any]] = None
    
    metadata: EventMetadata = Field(
        default_factory=lambda: EventMetadata(event_type=EventType.BEHAVIOR)
    )


class AlertEvent(EventSchema):
    """Alert event schema."""
    alert_id: str
    severity: str
    risk_level: str
    risk_score: float
    camera_id: Optional[str] = None
    location: Optional[str] = None
    explanation: Optional[str] = None
    recommendations: Optional[List[str]] = Field(default_factory=list)
    
    metadata: EventMetadata = Field(
        default_factory=lambda: EventMetadata(
            event_type=EventType.ALERT,
            priority=EventPriority.HIGH
        )
    )


class RiskAssessmentEvent(EventSchema):
    """Risk assessment event schema."""
    overall_risk_score: float
    risk_level: str
    component_risks: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    camera_id: Optional[str] = None
    
    metadata: EventMetadata = Field(
        default_factory=lambda: EventMetadata(event_type=EventType.RISK_ASSESSMENT)
    )


class ExperimentEvent(EventSchema):
    """Research experiment event schema."""
    experiment_id: str
    experiment_name: str
    status: str  # "started", "running", "completed", "failed"
    metrics: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    
    metadata: EventMetadata = Field(
        default_factory=lambda: EventMetadata(event_type=EventType.EXPERIMENT)
    )


class ModelUpdateEvent(EventSchema):
    """Model update event schema."""
    model_id: str
    version: str
    action: str  # "registered", "promoted", "deployed", "retired"
    metrics: Optional[Dict[str, Any]] = None
    
    metadata: EventMetadata = Field(
        default_factory=lambda: EventMetadata(event_type=EventType.MODEL_UPDATE)
    )


class EventValidator:
    """
    Event validation service.
    
    Validates events against schemas and provides error reporting.
    """
    
    def __init__(self):
        """Initialize event validator."""
        self.schemas: Dict[EventType, Type[EventSchema]] = {
            EventType.DETECTION: DetectionEvent,
            EventType.TRACK: TrackEvent,
            EventType.POSE: PoseEvent,
            EventType.BEHAVIOR: BehaviorEvent,
            EventType.ALERT: AlertEvent,
            EventType.RISK_ASSESSMENT: RiskAssessmentEvent,
            EventType.EXPERIMENT: ExperimentEvent,
            EventType.MODEL_UPDATE: ModelUpdateEvent
        }
    
    def validate(
        self,
        event_type: EventType,
        data: Dict[str, Any]
    ) -> tuple[bool, Optional[EventSchema], Optional[str]]:
        """
        Validate event data against schema.
        
        Args:
            event_type: Type of event
            data: Event data dictionary
        
        Returns:
            Tuple of (is_valid, event_instance, error_message)
        """
        if event_type not in self.schemas:
            return False, None, f"Unknown event type: {event_type}"
        
        schema_class = self.schemas[event_type]
        
        try:
            event = schema_class(**data)
            return True, event, None
        except Exception as e:
            return False, None, str(e)
    
    def validate_json(
        self,
        event_type: EventType,
        json_str: str
    ) -> tuple[bool, Optional[EventSchema], Optional[str]]:
        """
        Validate JSON event string.
        
        Args:
            event_type: Type of event
            json_str: JSON string
        
        Returns:
            Tuple of (is_valid, event_instance, error_message)
        """
        try:
            data = json.loads(json_str)
            return self.validate(event_type, data)
        except json.JSONDecodeError as e:
            return False, None, f"Invalid JSON: {e}"


class EventRegistry:
    """
    Event schema registry.
    
    Manages event schemas and provides factory methods.
    """
    
    def __init__(self):
        """Initialize event registry."""
        self.validator = EventValidator()
        self.event_counters: Dict[EventType, int] = {}
    
    def create_event(
        self,
        event_type: EventType,
        **kwargs
    ) -> EventSchema:
        """
        Create event instance.
        
        Args:
            event_type: Type of event
            **kwargs: Event data
        
        Returns:
            Event instance
        """
        if event_type not in self.validator.schemas:
            raise ValueError(f"Unknown event type: {event_type}")
        
        schema_class = self.validator.schemas[event_type]
        
        # Generate event ID if not provided
        if "metadata" not in kwargs:
            metadata = EventMetadata(event_type=event_type)
            kwargs["metadata"] = metadata
        
        event = schema_class(**kwargs)
        
        # Update counter
        self.event_counters[event_type] = self.event_counters.get(event_type, 0) + 1
        
        return event
    
    def get_schema(self, event_type: EventType) -> Optional[Type[EventSchema]]:
        """
        Get schema class for event type.
        
        Args:
            event_type: Event type
        
        Returns:
            Schema class or None
        """
        return self.validator.schemas.get(event_type)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get event statistics.
        
        Returns:
            Dictionary with event counts
        """
        return {
            "total_events": sum(self.event_counters.values()),
            "by_type": {k.value: v for k, v in self.event_counters.items()}
        }
