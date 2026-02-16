"""
SENTIENTCITY AI - Base Service Module
Abstract base class for all microservices
"""

from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI
from prometheus_client import Counter, Histogram, Info

from sentientcity.core.logging import get_logger, setup_logging
from sentientcity.core.settings import Settings, get_settings


class BaseService(ABC):
    """
    Abstract base class for all SENTIENTCITY microservices.
    
    Provides common functionality:
    - Configuration loading
    - Logging setup
    - Health check endpoints
    - Metrics collection
    - Lifecycle management
    """

    def __init__(
        self,
        service_name: str,
        version: str = "0.1.0",
        settings: Settings | None = None,
    ) -> None:
        self.service_name = service_name
        self.version = version
        self.settings = settings or get_settings()
        self.logger = get_logger(service_name)

        # Prometheus metrics
        self._setup_metrics()

        # FastAPI app
        self.app = self._create_app()

    def _setup_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        self.request_counter = Counter(
            f"{self.service_name}_requests_total",
            "Total requests",
            ["method", "endpoint", "status"],
        )
        self.request_latency = Histogram(
            f"{self.service_name}_request_latency_seconds",
            "Request latency in seconds",
            ["method", "endpoint"],
        )
        self.service_info = Info(
            f"{self.service_name}_info",
            "Service information",
        )
        self.service_info.info({
            "version": self.version,
            "environment": self.settings.environment,
        })

    def _create_app(self) -> FastAPI:
        """Create FastAPI application with lifecycle management."""

        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
            # Startup
            setup_logging()
            self.logger.info(
                "service_starting",
                service=self.service_name,
                version=self.version,
            )
            await self.startup()
            yield
            # Shutdown
            self.logger.info("service_stopping", service=self.service_name)
            await self.shutdown()

        app = FastAPI(
            title=f"SENTIENTCITY - {self.service_name}",
            version=self.version,
            lifespan=lifespan,
        )

        # Register common routes
        self._register_health_routes(app)

        return app

    def _register_health_routes(self, app: FastAPI) -> None:
        """Register health check endpoints."""

        @app.get("/health")
        async def health() -> dict[str, Any]:
            return {
                "status": "healthy",
                "service": self.service_name,
                "version": self.version,
            }

        @app.get("/health/ready")
        async def ready() -> dict[str, Any]:
            is_ready = await self.check_readiness()
            return {
                "ready": is_ready,
                "service": self.service_name,
            }

        @app.get("/health/live")
        async def live() -> dict[str, Any]:
            is_alive = await self.check_liveness()
            return {
                "alive": is_alive,
                "service": self.service_name,
            }

        @app.get("/metrics")
        async def metrics() -> dict[str, Any]:
            return await self.get_metrics()

    @abstractmethod
    async def startup(self) -> None:
        """
        Service startup hook.
        Override to initialize connections, load models, etc.
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Service shutdown hook.
        Override to close connections, cleanup resources, etc.
        """
        pass

    async def check_readiness(self) -> bool:
        """
        Check if service is ready to accept traffic.
        Override to add custom readiness checks.
        """
        return True

    async def check_liveness(self) -> bool:
        """
        Check if service is alive.
        Override to add custom liveness checks.
        """
        return True

    async def get_metrics(self) -> dict[str, Any]:
        """
        Get service metrics.
        Override to add custom metrics.
        """
        return {
            "service": self.service_name,
            "version": self.version,
        }

    def run(
        self,
        host: str | None = None,
        port: int | None = None,
        workers: int | None = None,
    ) -> None:
        """Run the service with uvicorn."""
        import uvicorn

        uvicorn.run(
            self.app,
            host=host or self.settings.api_host,
            port=port or self.settings.api_port,
            workers=workers or self.settings.api_workers,
            log_level=self.settings.log_level.lower(),
        )


class KafkaConsumerMixin:
    """Mixin for services that consume Kafka messages."""

    kafka_consumer: Any = None

    async def start_consumer(
        self,
        topics: list[str],
        group_id: str | None = None,
    ) -> None:
        """Start Kafka consumer for given topics."""
        from aiokafka import AIOKafkaConsumer

        settings = get_settings()
        self.kafka_consumer = AIOKafkaConsumer(
            *topics,
            bootstrap_servers=settings.kafka.bootstrap_servers,
            group_id=group_id or settings.kafka.consumer_group,
            auto_offset_reset=settings.kafka.auto_offset_reset,
        )
        await self.kafka_consumer.start()

    async def stop_consumer(self) -> None:
        """Stop Kafka consumer."""
        if self.kafka_consumer:
            await self.kafka_consumer.stop()


class KafkaProducerMixin:
    """Mixin for services that produce Kafka messages."""

    kafka_producer: Any = None

    async def start_producer(self) -> None:
        """Start Kafka producer."""
        from aiokafka import AIOKafkaProducer

        settings = get_settings()
        self.kafka_producer = AIOKafkaProducer(
            bootstrap_servers=settings.kafka.bootstrap_servers,
        )
        await self.kafka_producer.start()

    async def stop_producer(self) -> None:
        """Stop Kafka producer."""
        if self.kafka_producer:
            await self.kafka_producer.stop()

    async def publish(self, topic: str, message: bytes, key: bytes | None = None) -> None:
        """Publish message to Kafka topic."""
        if self.kafka_producer:
            await self.kafka_producer.send_and_wait(topic, message, key=key)


class RedisMixin:
    """Mixin for services that use Redis."""

    redis_client: Any = None

    async def connect_redis(self) -> None:
        """Connect to Redis."""
        import redis.asyncio as redis

        settings = get_settings()
        self.redis_client = redis.from_url(
            settings.redis.url,
            max_connections=settings.redis.max_connections,
        )

    async def disconnect_redis(self) -> None:
        """Disconnect from Redis."""
        if self.redis_client:
            await self.redis_client.close()
