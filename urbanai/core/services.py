"""
Base service interfaces and abstractions.

Provides base classes for all services with common functionality
like health checks, metrics, and lifecycle management.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from loguru import logger


class ServiceStatus(str, Enum):
    """Service status enumeration."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class HealthCheck:
    """Health check result."""
    service_name: str
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: str
    checks: Dict[str, Any] = None
    message: Optional[str] = None
    
    def __post_init__(self):
        """Initialize defaults."""
        if self.checks is None:
            self.checks = {}
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()


class BaseService(ABC):
    """
    Base service interface.
    
    All services inherit from this base class.
    Provides common functionality:
    - Lifecycle management
    - Health checks
    - Metrics collection
    - Error handling
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base service.
        
        Args:
            name: Service name
            config: Service configuration
        """
        self.name = name
        self.config = config or {}
        self.status = ServiceStatus.INITIALIZING
        self.start_time: Optional[datetime] = None
        self.metrics: Dict[str, Any] = {}
        self.error_count = 0
        self.request_count = 0
        
        logger.info(f"Initializing service: {name}")
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize service (must be implemented by subclasses).
        
        This method should:
        - Load configurations
        - Initialize connections
        - Prepare resources
        """
        pass
    
    @abstractmethod
    async def start(self) -> None:
        """
        Start service (must be implemented by subclasses).
        
        This method should:
        - Start background tasks
        - Begin processing
        - Register with service registry
        """
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """
        Stop service (must be implemented by subclasses).
        
        This method should:
        - Stop background tasks
        - Clean up resources
        - Unregister from service registry
        """
        pass
    
    async def health_check(self) -> HealthCheck:
        """
        Perform health check.
        
        Returns:
            HealthCheck result
        """
        checks = {
            "status": self.status.value,
            "uptime_seconds": self.get_uptime(),
            "error_count": self.error_count,
            "request_count": self.request_count
        }
        
        # Determine overall health
        if self.status == ServiceStatus.RUNNING:
            health_status = "healthy"
        elif self.status == ServiceStatus.ERROR:
            health_status = "unhealthy"
        else:
            health_status = "degraded"
        
        return HealthCheck(
            service_name=self.name,
            status=health_status,
            timestamp=datetime.utcnow().isoformat(),
            checks=checks
        )
    
    def get_uptime(self) -> float:
        """
        Get service uptime in seconds.
        
        Returns:
            Uptime in seconds
        """
        if self.start_time is None:
            return 0.0
        
        return (datetime.utcnow() - self.start_time).total_seconds()
    
    def record_error(self, error: Exception) -> None:
        """
        Record service error.
        
        Args:
            error: Exception that occurred
        """
        self.error_count += 1
        logger.error(f"Service {self.name} error: {error}", exc_info=error)
    
    def record_request(self) -> None:
        """Record service request."""
        self.request_count += 1
    
    def update_metric(self, key: str, value: Any) -> None:
        """
        Update service metric.
        
        Args:
            key: Metric key
            value: Metric value
        """
        self.metrics[key] = value
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get service metrics.
        
        Returns:
            Dictionary of metrics
        """
        return {
            "service_name": self.name,
            "status": self.status.value,
            "uptime_seconds": self.get_uptime(),
            "error_count": self.error_count,
            "request_count": self.request_count,
            **self.metrics
        }


class ServiceRegistry:
    """
    Service registry for managing all services.
    
    Provides service discovery, health monitoring, and lifecycle management.
    """
    
    def __init__(self):
        """Initialize service registry."""
        self.services: Dict[str, BaseService] = {}
        self.health_check_interval = 60  # seconds
    
    def register(self, service: BaseService) -> None:
        """
        Register service.
        
        Args:
            service: Service instance
        """
        if service.name in self.services:
            logger.warning(f"Service {service.name} already registered, overwriting")
        
        self.services[service.name] = service
        logger.info(f"Registered service: {service.name}")
    
    def unregister(self, service_name: str) -> None:
        """
        Unregister service.
        
        Args:
            service_name: Service name
        """
        if service_name in self.services:
            del self.services[service_name]
            logger.info(f"Unregistered service: {service_name}")
    
    def get_service(self, service_name: str) -> Optional[BaseService]:
        """
        Get service by name.
        
        Args:
            service_name: Service name
        
        Returns:
            Service instance or None
        """
        return self.services.get(service_name)
    
    def get_all_services(self) -> List[BaseService]:
        """
        Get all registered services.
        
        Returns:
            List of services
        """
        return list(self.services.values())
    
    async def health_check_all(self) -> Dict[str, HealthCheck]:
        """
        Perform health check on all services.
        
        Returns:
            Dictionary of health checks by service name
        """
        health_checks = {}
        
        for name, service in self.services.items():
            try:
                health_check = await service.health_check()
                health_checks[name] = health_check
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                health_checks[name] = HealthCheck(
                    service_name=name,
                    status="unhealthy",
                    message=str(e)
                )
        
        return health_checks
    
    def get_service_status(self) -> Dict[str, str]:
        """
        Get status of all services.
        
        Returns:
            Dictionary of statuses by service name
        """
        return {
            name: service.status.value
            for name, service in self.services.items()
        }


# Global service registry
_service_registry: Optional[ServiceRegistry] = None


def get_service_registry() -> ServiceRegistry:
    """
    Get global service registry instance.
    
    Returns:
        ServiceRegistry instance
    """
    global _service_registry
    if _service_registry is None:
        _service_registry = ServiceRegistry()
    return _service_registry
