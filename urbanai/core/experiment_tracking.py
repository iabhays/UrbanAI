"""
Experiment tracking infrastructure.

Provides experiment tracking, metrics logging, and research lab integration.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict, field
from enum import Enum
from loguru import logger


class ExperimentStatus(str, Enum):
    """Experiment status."""
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    experiment_id: str
    experiment_name: str
    description: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    created_by: str = "system"


@dataclass
class ExperimentMetrics:
    """Experiment metrics."""
    experiment_id: str
    timestamp: str
    step: Optional[int] = None
    epoch: Optional[int] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    loss: Optional[float] = None


@dataclass
class ExperimentResult:
    """Experiment result."""
    experiment_id: str
    status: ExperimentStatus
    start_time: str
    end_time: Optional[str] = None
    final_metrics: Dict[str, float] = field(default_factory=dict)
    best_metrics: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None
    artifacts: List[str] = field(default_factory=list)


class ExperimentTracker:
    """
    Experiment tracking service.
    
    Tracks experiments, metrics, and results for research lab.
    """
    
    def __init__(self, workspace: str = "experiments/"):
        """
        Initialize experiment tracker.
        
        Args:
            workspace: Workspace directory for experiments
        """
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.metrics_history: Dict[str, List[ExperimentMetrics]] = {}
        self.results: Dict[str, ExperimentResult] = {}
    
    def create_experiment(
        self,
        experiment_name: str,
        description: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create new experiment.
        
        Args:
            experiment_name: Name of experiment
            description: Experiment description
            config: Experiment configuration
            hyperparameters: Hyperparameters
        
        Returns:
            Experiment ID
        """
        experiment_id = f"exp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{experiment_name[:20]}"
        
        experiment_config = ExperimentConfig(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            description=description,
            config=config or {},
            hyperparameters=hyperparameters or {}
        )
        
        self.experiments[experiment_id] = experiment_config
        
        # Create experiment directory
        exp_dir = self.workspace / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save experiment config
        self._save_experiment_config(experiment_id, experiment_config)
        
        logger.info(f"Created experiment: {experiment_id} - {experiment_name}")
        
        return experiment_id
    
    def log_metrics(
        self,
        experiment_id: str,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        loss: Optional[float] = None
    ) -> None:
        """
        Log experiment metrics.
        
        Args:
            experiment_id: Experiment ID
            metrics: Metrics dictionary
            step: Training step
            epoch: Training epoch
            loss: Loss value
        """
        if experiment_id not in self.metrics_history:
            self.metrics_history[experiment_id] = []
        
        metric_entry = ExperimentMetrics(
            experiment_id=experiment_id,
            timestamp=datetime.utcnow().isoformat(),
            step=step,
            epoch=epoch,
            metrics=metrics,
            loss=loss
        )
        
        self.metrics_history[experiment_id].append(metric_entry)
        
        # Save metrics periodically
        if len(self.metrics_history[experiment_id]) % 100 == 0:
            self._save_metrics(experiment_id)
    
    def complete_experiment(
        self,
        experiment_id: str,
        final_metrics: Dict[str, float],
        best_metrics: Optional[Dict[str, float]] = None,
        artifacts: Optional[List[str]] = None
    ) -> None:
        """
        Mark experiment as completed.
        
        Args:
            experiment_id: Experiment ID
            final_metrics: Final metrics
            best_metrics: Best metrics during training
            artifacts: List of artifact paths
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        result = ExperimentResult(
            experiment_id=experiment_id,
            status=ExperimentStatus.COMPLETED,
            start_time=self.experiments[experiment_id].created_at,
            end_time=datetime.utcnow().isoformat(),
            final_metrics=final_metrics,
            best_metrics=best_metrics or final_metrics,
            artifacts=artifacts or []
        )
        
        self.results[experiment_id] = result
        self._save_result(experiment_id, result)
        
        logger.info(f"Experiment completed: {experiment_id}")
    
    def fail_experiment(
        self,
        experiment_id: str,
        error_message: str
    ) -> None:
        """
        Mark experiment as failed.
        
        Args:
            experiment_id: Experiment ID
            error_message: Error message
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        result = ExperimentResult(
            experiment_id=experiment_id,
            status=ExperimentStatus.FAILED,
            start_time=self.experiments[experiment_id].created_at,
            end_time=datetime.utcnow().isoformat(),
            error_message=error_message
        )
        
        self.results[experiment_id] = result
        self._save_result(experiment_id, result)
        
        logger.error(f"Experiment failed: {experiment_id} - {error_message}")
    
    def get_experiment(self, experiment_id: str) -> Optional[ExperimentConfig]:
        """Get experiment configuration."""
        return self.experiments.get(experiment_id)
    
    def get_metrics(self, experiment_id: str) -> List[ExperimentMetrics]:
        """Get experiment metrics."""
        return self.metrics_history.get(experiment_id, [])
    
    def get_result(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Get experiment result."""
        return self.results.get(experiment_id)
    
    def _save_experiment_config(self, experiment_id: str, config: ExperimentConfig) -> None:
        """Save experiment configuration."""
        exp_dir = self.workspace / experiment_id
        config_file = exp_dir / "config.json"
        
        with open(config_file, "w") as f:
            json.dump(asdict(config), f, indent=2)
    
    def _save_metrics(self, experiment_id: str) -> None:
        """Save experiment metrics."""
        exp_dir = self.workspace / experiment_id
        metrics_file = exp_dir / "metrics.json"
        
        metrics = [asdict(m) for m in self.metrics_history[experiment_id]]
        
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
    
    def _save_result(self, experiment_id: str, result: ExperimentResult) -> None:
        """Save experiment result."""
        exp_dir = self.workspace / experiment_id
        result_file = exp_dir / "result.json"
        
        with open(result_file, "w") as f:
            json.dump(asdict(result), f, indent=2)
    
    def list_experiments(self) -> List[str]:
        """List all experiment IDs."""
        return list(self.experiments.keys())


# Global experiment tracker
_experiment_tracker: Optional[ExperimentTracker] = None


def get_experiment_tracker(workspace: Optional[str] = None) -> ExperimentTracker:
    """
    Get global experiment tracker instance.
    
    Args:
        workspace: Optional workspace directory
    
    Returns:
        ExperimentTracker instance
    """
    global _experiment_tracker
    if _experiment_tracker is None:
        _experiment_tracker = ExperimentTracker(workspace or "experiments/")
    return _experiment_tracker
