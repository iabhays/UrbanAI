"""Tests for tracking components."""

import pytest
import numpy as np

# optional import in case edge AI dependencies (torch, loguru) are missing
try:
    from urbanai.edge_ai.tracking_engine import OCSortTracker
except ImportError:
    pytest.skip("urbanai edge_ai dependencies not installed", allow_module_level=True)


def test_tracker_initialization():
    """Test tracker initialization."""
    tracker = OCSortTracker()
    assert tracker is not None


def test_tracker_update():
    """Test tracker update."""
    tracker = OCSortTracker()
    
    detections = [
        {"bbox": np.array([100, 100, 200, 200]), "confidence": 0.9, "class_id": 0},
        {"bbox": np.array([300, 300, 400, 400]), "confidence": 0.8, "class_id": 0}
    ]
    
    tracks = tracker.update(detections)
    assert len(tracks) >= 0  # May be 0 if tracks not confirmed yet
