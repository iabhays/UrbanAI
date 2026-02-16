"""Pytest configuration and fixtures."""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_image():
    """Create sample test image."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_detections():
    """Create sample detections."""
    return [
        {
            "bbox": np.array([100, 100, 200, 200]),
            "confidence": 0.9,
            "class_id": 0,
            "class_name": "person"
        },
        {
            "bbox": np.array([300, 300, 400, 400]),
            "confidence": 0.8,
            "class_id": 0,
            "class_name": "person"
        }
    ]


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory."""
    return tmp_path
