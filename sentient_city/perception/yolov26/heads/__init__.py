"""Detection heads for YOLOv26."""

from .detection_head import DetectionHead, MultiScaleDetectionHead
from .crowd_density_head import CrowdDensityHead
from .behavior_embedding_head import BehaviorEmbeddingHead

__all__ = [
    "DetectionHead",
    "MultiScaleDetectionHead",
    "CrowdDensityHead",
    "BehaviorEmbeddingHead"
]
