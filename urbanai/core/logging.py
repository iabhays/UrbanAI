"""
Comprehensive logging infrastructure.

Provides structured logging with research lab integration,
correlation IDs, and log aggregation support.
"""

import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from contextvars import ContextVar
from loguru import logger
import uuid

# Context variable for correlation IDs
correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


class StructuredLogger:
    """
    Structured logger with research lab support.
    
    Provides JSON-structured logging for log aggregation systems.
    """
    
    def __init__(
        self,
        name: str,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        json_logs: bool = False
    ):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name (typically module name)
            log_level: Logging level
            log_file: Optional log file path
            json_logs: Whether to output JSON logs
        """
        self.name = name
        self.log_level = log_level
        self.log_file = log_file
        self.json_logs = json_logs
        
        # Configure logger
        self._configure()
    
    def _configure(self) -> None:
        """Configure logger."""
        # Remove default handler
        logger.remove()
        
        # Format string
        if self.json_logs:
            format_string = "{message}"
        else:
            format_string = (
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            )
        
        # Console handler
        logger.add(
            sys.stderr,
            format=self._format_log if self.json_logs else format_string,
            level=self.log_level,
            colorize=not self.json_logs,
            backtrace=True,
            diagnose=True,
            filter=self._add_context
        )
        
        # File handler
        if self.log_file:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.add(
                log_file,
                format=self._format_log if self.json_logs else format_string,
                level=self.log_level,
                rotation="10 MB",
                retention="7 days",
                compression="zip",
                backtrace=True,
                diagnose=True,
                filter=self._add_context
            )
    
    def _format_log(self, record: Dict[str, Any]) -> str:
        """Format log as JSON."""
        log_entry = {
            "timestamp": record["time"].isoformat(),
            "level": record["level"].name,
            "logger": self.name,
            "module": record.get("module", ""),
            "function": record.get("function", ""),
            "line": record.get("line", 0),
            "message": record["message"],
            "correlation_id": correlation_id.get(),
            "process_id": record.get("process", {}).get("id"),
            "thread_id": record.get("thread", {}).get("id")
        }
        
        # Add exception info if present
        if "exception" in record:
            log_entry["exception"] = {
                "type": record["exception"].type.__name__,
                "value": str(record["exception"].value),
                "traceback": record["exception"].traceback
            }
        
        return json.dumps(log_entry)
    
    def _add_context(self, record: Dict[str, Any]) -> bool:
        """Add context to log record."""
        # Add correlation ID if available
        cid = correlation_id.get()
        if cid:
            record["extra"]["correlation_id"] = cid
        
        return True
    
    def get_logger(self):
        """Get logger instance."""
        return logger.bind(name=self.name)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    json_logs: bool = False,
    research_logs: bool = True
) -> None:
    """
    Setup global logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
        json_logs: Whether to output JSON logs
        research_logs: Whether to enable research lab logging
    """
    # Remove default handler
    logger.remove()
    
    # Format string
    if json_logs:
        format_string = "{message}"
    else:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # Console handler
    logger.add(
        sys.stderr,
        format=format_string,
        level=log_level,
        colorize=not json_logs,
        backtrace=True,
        diagnose=True
    )
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format=format_string,
            level=log_level,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            backtrace=True,
            diagnose=True
        )
    
    # Research lab log file
    if research_logs:
        research_log_file = Path(log_file).parent / "research.log" if log_file else "logs/research.log"
        research_log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            str(research_log_file),
            format=format_string,
            level="DEBUG",  # More verbose for research
            rotation="50 MB",
            retention="30 days",
            compression="zip",
            filter=lambda record: "research" in record["name"].lower()
        )
    
    logger.info(f"Logging configured: level={log_level}, file={log_file}")


def get_logger(name: str) -> Any:
    """
    Get logger instance for a module.
    
    Args:
        name: Module name (typically __name__)
    
    Returns:
        Logger instance with correlation ID support
    """
    return logger.bind(name=name)


def set_correlation_id(cid: Optional[str] = None) -> str:
    """
    Set correlation ID for request tracing.
    
    Args:
        cid: Correlation ID (generates new if None)
    
    Returns:
        Correlation ID
    """
    if cid is None:
        cid = str(uuid.uuid4())
    
    correlation_id.set(cid)
    return cid


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID."""
    return correlation_id.get()


class ResearchLogger:
    """
    Research-specific logger.
    
    Provides specialized logging for research lab activities.
    """
    
    def __init__(self, experiment_id: Optional[str] = None):
        """
        Initialize research logger.
        
        Args:
            experiment_id: Optional experiment ID
        """
        self.experiment_id = experiment_id
        self.logger = logger.bind(research=True, experiment_id=experiment_id)
    
    def log_experiment_start(self, experiment_name: str, config: Dict[str, Any]):
        """Log experiment start."""
        self.logger.info(
            f"Experiment started: {experiment_name}",
            experiment_config=config
        )
    
    def log_experiment_end(self, experiment_name: str, metrics: Dict[str, Any]):
        """Log experiment end."""
        self.logger.info(
            f"Experiment completed: {experiment_name}",
            metrics=metrics
        )
    
    def log_training_step(self, step: int, loss: float, metrics: Dict[str, Any]):
        """Log training step."""
        self.logger.debug(
            f"Training step {step}",
            step=step,
            loss=loss,
            **metrics
        )
    
    def log_evaluation(self, metrics: Dict[str, Any]):
        """Log evaluation results."""
        self.logger.info(
            "Evaluation completed",
            **metrics
        )
