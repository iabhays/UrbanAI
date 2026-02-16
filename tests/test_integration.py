"""Integration tests for SENTIENTCITY AI."""

import pytest
import numpy as np
from sentient_city.edge_ai.edge_inference_runner import EdgeDetector
from sentient_city.tracking_engine import OCSortTracker
from sentient_city.intelligence.risk_engine import RiskEngine


@pytest.mark.integration
def test_detection_to_tracking_pipeline():
    """Test detection to tracking pipeline."""
    # Initialize components
    detector = EdgeDetector(device="cpu")
    tracker = OCSortTracker()
    
    # Create dummy image
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Run detection
    detection_result = detector.detect(image)
    detections = detection_result.get("detections", [])
    
    # Convert to tracking format
    tracking_detections = [
        {
            "bbox": np.array(det["bbox"]),
            "confidence": det["confidence"],
            "class_id": det["class_id"]
        }
        for det in detections
    ]
    
    # Run tracking
    tracks = tracker.update(tracking_detections)
    
    assert isinstance(tracks, list)


@pytest.mark.integration
def test_risk_assessment_pipeline():
    """Test risk assessment pipeline."""
    risk_engine = RiskEngine()
    
    # Create dummy data
    detections = [
        {"bbox": [100, 100, 200, 200], "confidence": 0.9, "class_id": 0}
    ]
    
    tracks = []
    density_map = np.random.rand(480, 640).astype(np.float32)
    
    # Assess risk
    risk_assessment = risk_engine.assess_risk(
        detections=detections,
        tracks=tracks,
        density_map=density_map
    )
    
    assert "overall_risk_score" in risk_assessment
    assert "risk_level" in risk_assessment
    assert 0.0 <= risk_assessment["overall_risk_score"] <= 1.0
