"""
Model monitoring service.

Monitors model performance, drift, and degradation.
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    timestamp: str
    model_id: str
    version: str
    inference_time_ms: float
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    error_rate: Optional[float] = None
    throughput_fps: Optional[float] = None


class ModelMonitor:
    """
    Model monitoring service.
    
    Tracks model performance and detects degradation.
    """
    
    def __init__(
        self,
        drift_threshold: float = 0.1,
        performance_window: int = 100
    ):
        """
        Initialize model monitor.
        
        Args:
            drift_threshold: Accuracy drift threshold
            performance_window: Window size for performance analysis
        """
        self.drift_threshold = drift_threshold
        self.performance_window = performance_window
        self.metrics_history: Dict[str, List[ModelMetrics]] = {}
    
    def record_metrics(self, metrics: ModelMetrics):
        """
        Record model metrics.
        
        Args:
            metrics: Model metrics
        """
        model_key = f"{metrics.model_id}_{metrics.version}"
        
        if model_key not in self.metrics_history:
            self.metrics_history[model_key] = []
        
        self.metrics_history[model_key].append(metrics)
        
        # Keep only recent metrics
        if len(self.metrics_history[model_key]) > self.performance_window * 2:
            self.metrics_history[model_key] = self.metrics_history[model_key][-self.performance_window:]
        
        logger.debug(f"Recorded metrics for {model_key}")
    
    def check_drift(
        self,
        model_id: str,
        version: str,
        baseline_accuracy: float
    ) -> Dict[str, any]:
        """
        Check for model drift.
        
        Args:
            model_id: Model identifier
            version: Model version
            baseline_accuracy: Baseline accuracy
        
        Returns:
            Drift analysis results
        """
        model_key = f"{model_id}_{version}"
        
        if model_key not in self.metrics_history:
            return {
                "drift_detected": False,
                "reason": "Insufficient data"
            }
        
        metrics = self.metrics_history[model_key]
        
        if len(metrics) < 10:
            return {
                "drift_detected": False,
                "reason": "Insufficient data"
            }
        
        # Get recent metrics
        recent_metrics = metrics[-self.performance_window:]
        recent_accuracies = [m.accuracy for m in recent_metrics if m.accuracy is not None]
        
        if not recent_accuracies:
            return {
                "drift_detected": False,
                "reason": "No accuracy metrics"
            }
        
        # Calculate drift
        current_accuracy = np.mean(recent_accuracies)
        drift = baseline_accuracy - current_accuracy
        
        drift_detected = drift > self.drift_threshold
        
        return {
            "drift_detected": drift_detected,
            "baseline_accuracy": baseline_accuracy,
            "current_accuracy": current_accuracy,
            "drift": drift,
            "drift_percentage": (drift / baseline_accuracy) * 100 if baseline_accuracy > 0 else 0
        }
    
    def get_performance_stats(
        self,
        model_id: str,
        version: str
    ) -> Dict[str, any]:
        """
        Get performance statistics.
        
        Args:
            model_id: Model identifier
            version: Model version
        
        Returns:
            Performance statistics
        """
        model_key = f"{model_id}_{version}"
        
        if model_key not in self.metrics_history:
            return {}
        
        metrics = self.metrics_history[model_key]
        
        if not metrics:
            return {}
        
        # Extract metrics
        inference_times = [m.inference_time_ms for m in metrics]
        accuracies = [m.accuracy for m in metrics if m.accuracy is not None]
        throughputs = [m.throughput_fps for m in metrics if m.throughput_fps is not None]
        
        stats = {
            "total_samples": len(metrics),
            "avg_inference_time_ms": np.mean(inference_times) if inference_times else None,
            "p95_inference_time_ms": np.percentile(inference_times, 95) if inference_times else None,
            "avg_accuracy": np.mean(accuracies) if accuracies else None,
            "avg_throughput_fps": np.mean(throughputs) if throughputs else None
        }
        
        return stats
    
    def alert_on_degradation(
        self,
        model_id: str,
        version: str,
        baseline_accuracy: float
    ) -> bool:
        """
        Check if alert should be raised for degradation.
        
        Args:
            model_id: Model identifier
            version: Model version
            baseline_accuracy: Baseline accuracy
        
        Returns:
            True if alert should be raised
        """
        drift_analysis = self.check_drift(model_id, version, baseline_accuracy)
        
        if drift_analysis.get("drift_detected"):
            logger.warning(
                f"Model degradation detected: {model_id} {version} | "
                f"Drift: {drift_analysis.get('drift_percentage', 0):.2f}%"
            )
            return True
        
        return False
