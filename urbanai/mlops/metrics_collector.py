"""
Metrics collection service.

Collects and aggregates system and model metrics.
"""

import time
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from loguru import logger


@dataclass
class SystemMetrics:
    """System metrics."""
    timestamp: str
    cpu_usage_percent: float
    memory_usage_percent: float
    gpu_usage_percent: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    disk_usage_percent: Optional[float] = None


@dataclass
class ApplicationMetrics:
    """Application metrics."""
    timestamp: str
    requests_per_second: float
    error_rate: float
    avg_response_time_ms: float
    active_connections: int


class MetricsCollector:
    """
    Metrics collection service.
    
    Collects system, application, and model metrics.
    """
    
    def __init__(self, metrics_dir: str = "data/metrics"):
        """
        Initialize metrics collector.
        
        Args:
            metrics_dir: Directory to store metrics
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        self.system_metrics: List[SystemMetrics] = []
        self.application_metrics: List[ApplicationMetrics] = []
        self.custom_metrics: Dict[str, List[Dict]] = defaultdict(list)
    
    def collect_system_metrics(self) -> SystemMetrics:
        """
        Collect system metrics.
        
        Returns:
            System metrics
        """
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # GPU metrics (if available)
            gpu_usage = None
            gpu_memory = None
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_usage = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                gpu_mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory = (gpu_mem_info.used / gpu_mem_info.total) * 100
            except Exception:
                pass
            
            # Disk usage
            disk = psutil.disk_usage("/")
            disk_percent = (disk.used / disk.total) * 100
            
            metrics = SystemMetrics(
                timestamp=datetime.utcnow().isoformat(),
                cpu_usage_percent=cpu_percent,
                memory_usage_percent=memory_percent,
                gpu_usage_percent=gpu_usage,
                gpu_memory_percent=gpu_memory,
                disk_usage_percent=disk_percent
            )
            
            self.system_metrics.append(metrics)
            
            # Keep only recent metrics
            if len(self.system_metrics) > 1000:
                self.system_metrics = self.system_metrics[-1000:]
            
            return metrics
            
        except ImportError:
            logger.warning("psutil not available for system metrics")
            return SystemMetrics(
                timestamp=datetime.utcnow().isoformat(),
                cpu_usage_percent=0.0,
                memory_usage_percent=0.0
            )
    
    def collect_application_metrics(
        self,
        requests_per_second: float,
        error_rate: float,
        avg_response_time_ms: float,
        active_connections: int
    ) -> ApplicationMetrics:
        """
        Collect application metrics.
        
        Args:
            requests_per_second: Request rate
            error_rate: Error rate (0-1)
            avg_response_time_ms: Average response time
            active_connections: Active connections
        
        Returns:
            Application metrics
        """
        metrics = ApplicationMetrics(
            timestamp=datetime.utcnow().isoformat(),
            requests_per_second=requests_per_second,
            error_rate=error_rate,
            avg_response_time_ms=avg_response_time_ms,
            active_connections=active_connections
        )
        
        self.application_metrics.append(metrics)
        
        # Keep only recent metrics
        if len(self.application_metrics) > 1000:
            self.application_metrics = self.application_metrics[-1000:]
        
        return metrics
    
    def record_custom_metric(
        self,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Record custom metric.
        
        Args:
            metric_name: Metric name
            value: Metric value
            tags: Optional tags
        """
        metric_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "value": value,
            "tags": tags or {}
        }
        
        self.custom_metrics[metric_name].append(metric_entry)
        
        # Keep only recent metrics
        if len(self.custom_metrics[metric_name]) > 1000:
            self.custom_metrics[metric_name] = self.custom_metrics[metric_name][-1000:]
    
    def export_metrics(self, output_path: Optional[str] = None) -> Dict:
        """
        Export all metrics.
        
        Args:
            output_path: Optional path to save JSON
        
        Returns:
            Dictionary with all metrics
        """
        metrics_data = {
            "system_metrics": [asdict(m) for m in self.system_metrics],
            "application_metrics": [asdict(m) for m in self.application_metrics],
            "custom_metrics": {
                name: entries
                for name, entries in self.custom_metrics.items()
            }
        }
        
        if output_path:
            with open(output_path, "w") as f:
                json.dump(metrics_data, f, indent=2)
            logger.info(f"Exported metrics to {output_path}")
        
        return metrics_data
