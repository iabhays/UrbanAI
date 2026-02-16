"""
SENTIENTCITY AI - Tracking Service
Multi-object tracking and re-identification
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

from sentientcity.core.logging import get_logger
from sentientcity.core.service import (
    BaseService,
    KafkaConsumerMixin,
    KafkaProducerMixin,
    RedisMixin,
)
from sentientcity.schemas.events import (
    BoundingBox,
    DetectionEvent,
    TrackEvent,
    TrackState,
    TrajectoryPoint,
)

logger = get_logger(__name__)


@dataclass
class Track:
    """Internal track representation."""
    
    track_id: str
    global_id: str | None = None
    class_id: int = 0
    class_name: str = "unknown"
    state: TrackState = TrackState.CREATED
    bbox: list[float] = field(default_factory=lambda: [0, 0, 0, 0])
    embedding: list[float] | None = None
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    velocity: tuple[float, float] = (0.0, 0.0)
    
    def predict(self) -> list[float]:
        """Predict next position using velocity."""
        cx = (self.bbox[0] + self.bbox[2]) / 2 + self.velocity[0]
        cy = (self.bbox[1] + self.bbox[3]) / 2 + self.velocity[1]
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
    
    def update(self, bbox: list[float], embedding: list[float] | None = None) -> None:
        """Update track with new detection."""
        # Update velocity
        old_cx = (self.bbox[0] + self.bbox[2]) / 2
        old_cy = (self.bbox[1] + self.bbox[3]) / 2
        new_cx = (bbox[0] + bbox[2]) / 2
        new_cy = (bbox[1] + bbox[3]) / 2
        self.velocity = (new_cx - old_cx, new_cy - old_cy)
        
        # Update bbox
        self.bbox = bbox
        
        # Update embedding with EMA
        if embedding is not None:
            if self.embedding is None:
                self.embedding = embedding
            else:
                alpha = 0.9
                self.embedding = [
                    alpha * e + (1 - alpha) * ne
                    for e, ne in zip(self.embedding, embedding)
                ]
        
        self.hits += 1
        self.time_since_update = 0
        
        # Update state
        if self.state == TrackState.CREATED:
            self.state = TrackState.TENTATIVE
        elif self.state == TrackState.TENTATIVE and self.hits >= 3:
            self.state = TrackState.CONFIRMED


class ByteTracker:
    """ByteTrack-style multi-object tracker."""

    def __init__(
        self,
        high_thresh: float = 0.6,
        low_thresh: float = 0.1,
        match_thresh: float = 0.8,
        max_time_lost: int = 30,
    ) -> None:
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_time_lost = max_time_lost
        
        self.tracks: dict[str, Track] = {}
        self.track_counter = 0

    def update(
        self,
        detections: list[dict[str, Any]],
        timestamp: datetime,
    ) -> list[Track]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of detection dicts with bbox, confidence, embedding
            timestamp: Frame timestamp
            
        Returns:
            List of active tracks
        """
        # Split detections by confidence
        high_dets = [d for d in detections if d["confidence"] >= self.high_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d["confidence"] < self.high_thresh]
        
        # Predict track positions
        for track in self.tracks.values():
            track.age += 1
            track.time_since_update += 1
        
        # First association: high confidence detections with confirmed tracks
        confirmed_tracks = [t for t in self.tracks.values() if t.state == TrackState.CONFIRMED]
        matched, unmatched_tracks, unmatched_dets = self._associate(
            confirmed_tracks, high_dets
        )
        
        # Update matched tracks
        for track, det in matched:
            track.update(det["bbox"], det.get("embedding"))
            track.trajectory.append({
                "x": (det["bbox"][0] + det["bbox"][2]) / 2,
                "y": (det["bbox"][1] + det["bbox"][3]) / 2,
                "timestamp": timestamp.isoformat(),
            })
        
        # Second association: remaining tracks with low confidence detections
        remaining_tracks = [self.tracks[tid] for tid in unmatched_tracks]
        matched2, unmatched_tracks2, _ = self._associate(remaining_tracks, low_dets)
        
        for track, det in matched2:
            track.update(det["bbox"], det.get("embedding"))
        
        # Handle unmatched detections - create new tracks
        for det_idx in unmatched_dets:
            det = high_dets[det_idx]
            self._create_track(det, timestamp)
        
        # Handle lost tracks
        for track_id in unmatched_tracks + unmatched_tracks2:
            track = self.tracks[track_id]
            if track.time_since_update > self.max_time_lost:
                track.state = TrackState.DELETED
            elif track.state == TrackState.CONFIRMED:
                track.state = TrackState.LOST
        
        # Remove deleted tracks
        self.tracks = {
            tid: t for tid, t in self.tracks.items()
            if t.state != TrackState.DELETED
        }
        
        return list(self.tracks.values())

    def _associate(
        self,
        tracks: list[Track],
        detections: list[dict[str, Any]],
    ) -> tuple[list[tuple[Track, dict]], list[str], list[int]]:
        """Associate tracks with detections using IoU."""
        if not tracks or not detections:
            return [], [t.track_id for t in tracks], list(range(len(detections)))
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._iou(track.bbox, det["bbox"])
        
        # Simple greedy matching
        matched = []
        unmatched_tracks = set(t.track_id for t in tracks)
        unmatched_dets = set(range(len(detections)))
        
        while True:
            if iou_matrix.size == 0:
                break
            max_iou = iou_matrix.max()
            if max_iou < self.match_thresh:
                break
            
            i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            matched.append((tracks[i], detections[j]))
            unmatched_tracks.discard(tracks[i].track_id)
            unmatched_dets.discard(j)
            
            iou_matrix[i, :] = 0
            iou_matrix[:, j] = 0
        
        return matched, list(unmatched_tracks), list(unmatched_dets)

    def _create_track(self, detection: dict[str, Any], timestamp: datetime) -> Track:
        """Create a new track from detection."""
        self.track_counter += 1
        track_id = f"track_{self.track_counter}"
        
        track = Track(
            track_id=track_id,
            class_id=detection.get("class_id", 0),
            class_name=detection.get("class_name", "unknown"),
            bbox=detection["bbox"],
            embedding=detection.get("embedding"),
            trajectory=[{
                "x": (detection["bbox"][0] + detection["bbox"][2]) / 2,
                "y": (detection["bbox"][1] + detection["bbox"][3]) / 2,
                "timestamp": timestamp.isoformat(),
            }],
        )
        
        self.tracks[track_id] = track
        return track

    @staticmethod
    def _iou(box1: list[float], box2: list[float]) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0


class ReIDMatcher:
    """Cross-camera re-identification matcher."""

    def __init__(self, similarity_threshold: float = 0.7) -> None:
        self.similarity_threshold = similarity_threshold
        self.global_gallery: dict[str, list[float]] = {}

    def match(self, embedding: list[float]) -> str | None:
        """Match embedding against global gallery."""
        if not self.global_gallery or embedding is None:
            return None
        
        best_match = None
        best_sim = self.similarity_threshold
        
        for global_id, gallery_emb in self.global_gallery.items():
            sim = self._cosine_similarity(embedding, gallery_emb)
            if sim > best_sim:
                best_sim = sim
                best_match = global_id
        
        return best_match

    def register(self, global_id: str, embedding: list[float]) -> None:
        """Register new global identity."""
        self.global_gallery[global_id] = embedding

    def update(self, global_id: str, embedding: list[float], alpha: float = 0.9) -> None:
        """Update gallery embedding with EMA."""
        if global_id in self.global_gallery:
            old = self.global_gallery[global_id]
            self.global_gallery[global_id] = [
                alpha * o + (1 - alpha) * n
                for o, n in zip(old, embedding)
            ]
        else:
            self.register(global_id, embedding)

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0


class TrackingService(BaseService, KafkaConsumerMixin, KafkaProducerMixin, RedisMixin):
    """Multi-object tracking service."""

    def __init__(self) -> None:
        super().__init__(
            service_name="tracking",
            version="0.1.0",
        )
        
        self.trackers: dict[str, ByteTracker] = defaultdict(ByteTracker)
        self.reid_matcher = ReIDMatcher()
        self._consumer_task: asyncio.Task | None = None
        
        self._register_routes()

    def _register_routes(self) -> None:
        """Register API routes."""
        router = APIRouter(prefix="/api/v1", tags=["tracking"])

        @router.get("/tracks/{camera_id}")
        async def get_tracks(camera_id: str) -> list[dict[str, Any]]:
            """Get active tracks for a camera."""
            if camera_id not in self.trackers:
                return []
            return [
                {
                    "track_id": t.track_id,
                    "global_id": t.global_id,
                    "state": t.state.value,
                    "bbox": t.bbox,
                    "age": t.age,
                }
                for t in self.trackers[camera_id].tracks.values()
            ]

        @router.get("/global/{global_id}")
        async def get_global_track(global_id: str) -> dict[str, Any] | None:
            """Get global track across cameras."""
            # Would query Redis/DB for global track info
            return None

        self.app.include_router(router)

    async def startup(self) -> None:
        """Initialize service."""
        await self.start_consumer(["sentient.detections"])
        await self.start_producer()
        await self.connect_redis()
        
        self._consumer_task = asyncio.create_task(self._consume_loop())
        self.logger.info("service_started")

    async def shutdown(self) -> None:
        """Cleanup service."""
        if self._consumer_task:
            self._consumer_task.cancel()
        await self.stop_consumer()
        await self.stop_producer()
        await self.disconnect_redis()
        self.logger.info("service_stopped")

    async def _consume_loop(self) -> None:
        """Consume detection events and update tracks."""
        async for msg in self.kafka_consumer:
            try:
                event = DetectionEvent.model_validate_json(msg.value)
                await self._process_detections(event)
            except Exception as e:
                logger.error("processing_error", error=str(e))

    async def _process_detections(self, event: DetectionEvent) -> None:
        """Process detection event and update tracks."""
        camera_id = event.camera_id
        tracker = self.trackers[camera_id]
        
        # Convert to tracker format
        detections = [
            {
                "bbox": [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2],
                "confidence": d.confidence,
                "class_id": d.class_id,
                "class_name": d.class_name,
                "embedding": d.embedding,
            }
            for d in event.detections
        ]
        
        # Update tracker
        tracks = tracker.update(detections, event.frame_timestamp)
        
        # Cross-camera ReID
        for track in tracks:
            if track.embedding and track.state == TrackState.CONFIRMED:
                global_id = self.reid_matcher.match(track.embedding)
                if global_id:
                    track.global_id = global_id
                    self.reid_matcher.update(global_id, track.embedding)
                elif track.global_id is None:
                    track.global_id = str(uuid4())
                    self.reid_matcher.register(track.global_id, track.embedding)
        
        # Publish track events
        for track in tracks:
            if track.state in [TrackState.CONFIRMED, TrackState.LOST]:
                await self._publish_track(camera_id, track, event.frame_timestamp)

    async def _publish_track(
        self,
        camera_id: str,
        track: Track,
        timestamp: datetime,
    ) -> None:
        """Publish track event to Kafka."""
        trajectory = [
            TrajectoryPoint(
                x=p["x"],
                y=p["y"],
                timestamp=datetime.fromisoformat(p["timestamp"]),
                confidence=1.0,
            )
            for p in track.trajectory[-50:]  # Last 50 points
        ]
        
        event = TrackEvent(
            source_service=self.service_name,
            camera_id=camera_id,
            track_id=track.track_id,
            global_id=track.global_id,
            state=track.state,
            class_id=track.class_id,
            class_name=track.class_name,
            current_bbox=BoundingBox(
                x1=track.bbox[0],
                y1=track.bbox[1],
                x2=track.bbox[2],
                y2=track.bbox[3],
            ),
            trajectory=trajectory,
            embedding=track.embedding,
            velocity=track.velocity,
            age_frames=track.age,
            time_since_update=track.time_since_update,
        )
        
        await self.publish(
            "sentient.tracks",
            event.model_dump_json().encode(),
            key=camera_id.encode(),
        )


def create_app() -> TrackingService:
    """Create service instance."""
    return TrackingService()


if __name__ == "__main__":
    service = create_app()
    service.run(port=8002)
