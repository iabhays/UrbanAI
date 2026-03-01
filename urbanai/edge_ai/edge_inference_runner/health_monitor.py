"""
Health monitoring for edge inference.

Monitors device health, inference performance, and system status.
"""

import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque
from loguru import logger

from .device_manager import DeviceManager, DeviceInfo


@dataclass
class HealthStatus:
    """Health status."""
    timestamp: str
    is_healthy: bool
    device_health: Dict[str, DeviceInfo]
    inference_latency_ms: float
    error_rate: float
    warnings: List[str]


class HealthMonitor:
    """
    Health monitoring service.
    
    Monitors device health, inference performance, and system status.
    """
    
    def __init__(
        self,
        device_manager: DeviceManager,
        latency_threshold_ms: float = 100.0,
        error_rate_threshold: float = 0.05,
        health_check_interval: int = 60
    ):
        """
        Initialize health monitor.
        
        Args:
            device_manager: Device manager instance
            latency_threshold_ms: Maximum acceptable latency
            error_rate_threshold: Maximum acceptable error rate
            health_check_interval: Health check interval in seconds
        """
        self.device_manager = device_manager
        self.latency_threshold_ms = latency_threshold_ms
        self.error_rate_threshold = error_rate_threshold
        self.health_check_interval = health_check_interval
        
        self.inference_times: deque = deque(maxlen=1000)
        self.errors: deque = deque(maxlen=1000)
        self.last_health_check: Optional[datetime] = None
    
    def record_inference(self, latency_ms: float, success: bool = True):
        """
        Record inference performance.
        
        Args:
            latency_ms: Inference latency in milliseconds
            success: Whether inference succeeded
        """
        self.inference_times.append(latency_ms)
        self.errors.append(0 if success else 1)
    
    def check_health(self) -> HealthStatus:
        """
        Perform health check.
        
        Returns:
            Health status
        """
        warnings = []
        
        # Check device health
        device_health = self.device_manager.monitor_health()
        
        # Check for unhealthy devices
        unhealthy_devices = [
            device_id for device_id, info in device_health.items()
            if not info.is_healthy
        ]
        
        if unhealthy_devices:
            warnings.append(f"Unhealthy devices: {unhealthy_devices}")
        
        # Check inference latency
        if self.inference_times:
            avg_latency = sum(self.inference_times) / len(self.inference_times)
            p95_latency = sorted(self.inference_times)[int(len(self.inference_times) * 0.95)]
            
            if p95_latency > self.latency_threshold_ms:
                warnings.append(
                    f"High latency detected: p95={p95_latency:.2f}ms "
                    f"(threshold={self.latency_threshold_ms}ms)"
                )
        else:
            avg_latency = 0.0
        
        # Check error rate
        if self.errors:
            error_rate = sum(self.errors) / len(self.errors)
            if error_rate > self.error_rate_threshold:
                warnings.append(
                    f"High error rate: {error_rate:.2%} "
                    f"(threshold={self.error_rate_threshold:.2%})"
                )
        else:
            error_rate = 0.0
        
        # Check device memory
        for device_id, info in device_health.items():
            if info.device_type == "cuda":
                memory_usage = info.memory_used_mb / info.memory_total_mb if info.memory_total_mb > 0 else 0
                if memory_usage > 0.9:
                    warnings.append(
                        f"High memory usage on {device_id}: {memory_usage:.1%}"
                    )
        
        # Overall health
        is_healthy = len(warnings) == 0 and len(unhealthy_devices) == 0
        
        status = HealthStatus(
            timestamp=datetime.utcnow().isoformat(),
            is_healthy=is_healthy,
            device_health=device_health,
            inference_latency_ms=avg_latency,
            error_rate=error_rate,
            warnings=warnings
        )
        
        self.last_health_check = datetime.utcnow()
        
        if not is_healthy:
            logger.warning(f"Health check failed: {warnings}")
        
        return status
    
    def get_metrics(self) -> Dict:
        """
        Get performance metrics.
        
        Returns:
            Performance metrics dictionary
        """
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "inference_count": len(self.inference_times),
            "error_count": sum(self.errors) if self.errors else 0
        }
        
        if self.inference_times:
            metrics.update({
                "avg_latency_ms": sum(self.inference_times) / len(self.inference_times),
                "min_latency_ms": min(self.inference_times),
                "max_latency_ms": max(self.inference_times),
                "p95_latency_ms": sorted(self.inference_times)[int(len(self.inference_times) * 0.95)]
            })
        
        if self.errors:
            metrics["error_rate"] = sum(self.errors) / len(self.errors)
        
        return metrics
