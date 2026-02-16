"""
SENTIENTCITY AI - Event Schemas
Pydantic models for all Kafka event types
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Event type enumeration."""

    DETECTION = "detection"
    TRACK = "track"
    POSE = "pose"
    ALERT = "alert"
    METRIC = "metric"
    AUDIT = "audit"
    SYSTEM = "system"


class Severity(str, Enum):
    """Alert severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TrackState(str, Enum):
    """Track lifecycle states."""

    CREATED = "created"
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    DELETED = "deleted"


class BaseEvent(BaseModel):
    """Base event schema with common fields."""

    event_id: UUID = Field(default_factory=uuid4)
    event_type: EventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_service: str
    correlation_id: UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class BoundingBox(BaseModel):
    """Bounding box coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return self.width * self.height


class Detection(BaseModel):
    """Single detection result."""

    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox
    embedding: list[float] | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class DetectionEvent(BaseEvent):
    """Detection event from edge inference."""

    event_type: EventType = EventType.DETECTION
    camera_id: str
    frame_id: int
    frame_timestamp: datetime
    image_width: int
    image_height: int
    detections: list[Detection]
    density_map: list[list[float]] | None = None
    inference_time_ms: float


class Keypoint(BaseModel):
    """Single pose keypoint."""

    x: float
    y: float
    confidence: float = Field(ge=0.0, le=1.0)
    name: str


class PoseData(BaseModel):
    """Pose estimation result."""

    track_id: str
    keypoints: list[Keypoint]
    skeleton_confidence: float = Field(ge=0.0, le=1.0)
    action_features: list[float] | None = None


class PoseEvent(BaseEvent):
    """Pose estimation event."""

    event_type: EventType = EventType.POSE
    camera_id: str
    frame_id: int
    poses: list[PoseData]


class TrajectoryPoint(BaseModel):
    """Single point in a trajectory."""

    x: float
    y: float
    timestamp: datetime
    confidence: float = Field(ge=0.0, le=1.0)


class TrackEvent(BaseEvent):
    """Object track event."""

    event_type: EventType = EventType.TRACK
    camera_id: str
    track_id: str
    global_id: str | None = None
    state: TrackState
    class_id: int
    class_name: str
    current_bbox: BoundingBox
    trajectory: list[TrajectoryPoint]
    embedding: list[float] | None = None
    velocity: tuple[float, float] | None = None
    age_frames: int
    time_since_update: int


class Evidence(BaseModel):
    """Evidence for an alert."""

    frame_ids: list[int] = Field(default_factory=list)
    track_ids: list[str] = Field(default_factory=list)
    keyframes: list[str] = Field(default_factory=list)  # Base64 or URLs
    video_clip_url: str | None = None


class Location(BaseModel):
    """Location information."""

    camera_id: str
    zone_id: str | None = None
    zone_name: str | None = None
    geo_lat: float | None = None
    geo_lon: float | None = None
    floor: str | None = None
    area: str | None = None


class AlertEvent(BaseEvent):
    """Alert event from intelligence engine."""

    event_type: EventType = EventType.ALERT
    alert_type: str
    severity: Severity
    risk_score: float = Field(ge=0.0, le=100.0)
    location: Location
    evidence: Evidence
    description: str
    explanation: str | None = None
    recommended_actions: list[str] = Field(default_factory=list)
    acknowledged: bool = False
    resolved: bool = False
    false_positive: bool | None = None


class MetricEvent(BaseEvent):
    """System metric event."""

    event_type: EventType = EventType.METRIC
    metric_name: str
    metric_value: float
    metric_unit: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)


class AuditAction(str, Enum):
    """Audit action types."""

    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LOGIN = "login"
    LOGOUT = "logout"
    ACCESS = "access"
    EXPORT = "export"


class AuditEvent(BaseEvent):
    """Audit log event."""

    event_type: EventType = EventType.AUDIT
    action: AuditAction
    actor_id: str | None = None
    actor_type: str  # user, service, system
    resource_type: str
    resource_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    ip_address: str | None = None
    user_agent: str | None = None
    success: bool = True
    error_message: str | None = None


class PluginResult(BaseModel):
    """Result from a plugin analysis."""

    plugin_name: str
    plugin_version: str
    confidence: float = Field(ge=0.0, le=1.0)
    risk_contribution: float = Field(ge=0.0, le=100.0)
    findings: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)
    processing_time_ms: float


class IntelligenceEvent(BaseEvent):
    """Combined intelligence analysis event."""

    event_type: EventType = EventType.SYSTEM
    camera_id: str
    frame_range: tuple[int, int]
    plugin_results: list[PluginResult]
    aggregated_risk_score: float = Field(ge=0.0, le=100.0)
    behavior_embedding: list[float] | None = None
    anomaly_score: float | None = None
    generate_alert: bool = False
