"""
SENTIENTCITY AI - Tracking Service Unit Tests
"""

import pytest
from datetime import datetime

from services.tracking.src.service import ByteTracker, Track, ReIDMatcher
from sentientcity.schemas.events import TrackState


class TestByteTracker:
    """Tests for ByteTracker class."""

    def test_tracker_initialization(self):
        """Test tracker initializes correctly."""
        tracker = ByteTracker()
        assert tracker.high_thresh == 0.6
        assert tracker.low_thresh == 0.1
        assert len(tracker.tracks) == 0

    def test_create_track(self):
        """Test track creation from detection."""
        tracker = ByteTracker()
        detection = {
            "bbox": [100, 100, 200, 200],
            "confidence": 0.8,
            "class_id": 0,
            "class_name": "person",
        }
        
        track = tracker._create_track(detection, datetime.utcnow())
        
        assert track.track_id == "track_1"
        assert track.bbox == [100, 100, 200, 200]
        assert track.class_name == "person"
        assert track.state == TrackState.CREATED

    def test_update_creates_tracks(self):
        """Test that update creates tracks for new detections."""
        tracker = ByteTracker()
        detections = [
            {"bbox": [100, 100, 200, 200], "confidence": 0.8},
            {"bbox": [300, 300, 400, 400], "confidence": 0.7},
        ]
        
        tracks = tracker.update(detections, datetime.utcnow())
        
        assert len(tracks) == 2
        assert all(t.state in [TrackState.CREATED, TrackState.TENTATIVE] for t in tracks)

    def test_iou_calculation(self):
        """Test IoU calculation between boxes."""
        box1 = [0, 0, 10, 10]
        box2 = [5, 5, 15, 15]
        
        iou = ByteTracker._iou(box1, box2)
        
        # Intersection: 5x5 = 25, Union: 200-25 = 175
        expected_iou = 25 / 175
        assert abs(iou - expected_iou) < 0.01

    def test_track_state_transitions(self):
        """Test track state transitions through updates."""
        tracker = ByteTracker()
        detection = {"bbox": [100, 100, 200, 200], "confidence": 0.8}
        
        # First detection creates track
        tracker.update([detection], datetime.utcnow())
        track = list(tracker.tracks.values())[0]
        assert track.state == TrackState.CREATED
        
        # Multiple updates should transition to confirmed
        for i in range(5):
            shifted_det = {
                "bbox": [100 + i*5, 100 + i*5, 200 + i*5, 200 + i*5],
                "confidence": 0.8
            }
            tracker.update([shifted_det], datetime.utcnow())
        
        track = list(tracker.tracks.values())[0]
        assert track.state == TrackState.CONFIRMED


class TestTrack:
    """Tests for Track dataclass."""

    def test_track_predict(self):
        """Test track position prediction."""
        track = Track(
            track_id="test",
            bbox=[100, 100, 200, 200],
            velocity=(10, 5),
        )
        
        predicted = track.predict()
        
        # Center should move by velocity
        assert predicted[0] == 110 - 50  # x1
        assert predicted[1] == 105 - 50  # y1

    def test_track_update(self):
        """Test track update with new detection."""
        track = Track(
            track_id="test",
            bbox=[100, 100, 200, 200],
        )
        
        track.update([110, 105, 210, 205])
        
        assert track.bbox == [110, 105, 210, 205]
        assert track.hits == 1
        assert track.time_since_update == 0

    def test_embedding_ema_update(self):
        """Test embedding exponential moving average."""
        track = Track(
            track_id="test",
            bbox=[100, 100, 200, 200],
            embedding=[1.0, 0.0, 0.0],
        )
        
        track.update([100, 100, 200, 200], embedding=[0.0, 1.0, 0.0])
        
        # EMA with alpha=0.9
        assert track.embedding[0] == pytest.approx(0.9)
        assert track.embedding[1] == pytest.approx(0.1)


class TestReIDMatcher:
    """Tests for ReIDMatcher class."""

    def test_register_and_match(self):
        """Test identity registration and matching."""
        matcher = ReIDMatcher(similarity_threshold=0.5)
        
        # Register identity
        matcher.register("id_1", [1.0, 0.0, 0.0])
        
        # Match similar embedding
        result = matcher.match([0.9, 0.1, 0.0])
        
        assert result == "id_1"

    def test_no_match_below_threshold(self):
        """Test no match when similarity is below threshold."""
        matcher = ReIDMatcher(similarity_threshold=0.9)
        
        matcher.register("id_1", [1.0, 0.0, 0.0])
        
        # Very different embedding
        result = matcher.match([0.0, 1.0, 0.0])
        
        assert result is None

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        
        sim = ReIDMatcher._cosine_similarity(a, b)
        
        assert sim == pytest.approx(1.0)

    def test_update_gallery(self):
        """Test gallery embedding update with EMA."""
        matcher = ReIDMatcher()
        
        matcher.register("id_1", [1.0, 0.0])
        matcher.update("id_1", [0.0, 1.0], alpha=0.5)
        
        assert matcher.global_gallery["id_1"][0] == pytest.approx(0.5)
        assert matcher.global_gallery["id_1"][1] == pytest.approx(0.5)
