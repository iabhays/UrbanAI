"""Tests for edge AI components."""

import pytest
import numpy as np

try:
    from urbanai.edge_ai import EdgeDetector
except ImportError:
    pytest.skip("urbanai edge_ai dependencies not installed", allow_module_level=True)


def test_detector_initialization():
    """Test detector initialization."""
    detector = EdgeDetector(device="cpu")
    assert detector is not None


def test_detector_preprocess():
    """Test image preprocessing."""
    detector = EdgeDetector(device="cpu")
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    tensor = detector.preprocess(image)
    assert tensor.shape[0] == 1  # Batch dimension
    assert tensor.shape[1] == 3  # Channels


@pytest.mark.skip(reason="Requires model weights")
def test_detector_detect():
    """Test detection."""
    detector = EdgeDetector(device="cpu")
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = detector.detect(image)
    assert "detections" in result
    assert "num_detections" in result
