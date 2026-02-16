"""Tests for MLOps modules."""

import pytest
from sentient_city.mlops import ModelRegistry, ModelMonitor, MetricsCollector
from sentient_city.mlops.model_monitor import ModelMetrics


def test_model_registry():
    """Test model registry."""
    registry = ModelRegistry(registry_path="data/test_registry.json")
    
    # Register model
    metadata = registry.register_model(
        model_id="test_model",
        model_path="models/test.pt",
        model_type="yolov26",
        version="v1",
        description="Test model"
    )
    
    assert metadata.model_id == "test_model"
    assert metadata.version == "v1"
    
    # Get model
    retrieved = registry.get_model("test_model", "v1")
    assert retrieved is not None
    assert retrieved.model_id == "test_model"
    
    # Set production
    registry.set_production("test_model", "v1", True)
    prod_model = registry.get_production_model("test_model")
    assert prod_model is not None
    assert prod_model.is_production


def test_model_monitor():
    """Test model monitor."""
    monitor = ModelMonitor()
    
    # Record metrics
    metrics = ModelMetrics(
        timestamp="2024-01-01T00:00:00",
        model_id="test_model",
        version="v1",
        inference_time_ms=50.0,
        accuracy=0.95
    )
    monitor.record_metrics(metrics)
    
    # Check drift
    drift_result = monitor.check_drift("test_model", "v1", 0.95)
    assert "drift_detected" in drift_result
    
    # Get stats
    stats = monitor.get_performance_stats("test_model", "v1")
    assert "avg_inference_time_ms" in stats


def test_metrics_collector():
    """Test metrics collector."""
    collector = MetricsCollector()
    
    # Collect system metrics
    sys_metrics = collector.collect_system_metrics()
    assert sys_metrics is not None
    
    # Collect application metrics
    app_metrics = collector.collect_application_metrics(
        requests_per_second=100.0,
        error_rate=0.01,
        avg_response_time_ms=50.0,
        active_connections=10
    )
    assert app_metrics.requests_per_second == 100.0
    
    # Record custom metric
    collector.record_custom_metric("custom_metric", 42.0)
    assert "custom_metric" in collector.custom_metrics
