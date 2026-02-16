"""
SENTIENTCITY AI - Intelligence Engine
Plugin-based analysis orchestration and risk scoring
"""

import asyncio
import importlib
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from sentientcity.core.logging import get_logger
from sentientcity.core.service import (
    BaseService,
    KafkaConsumerMixin,
    KafkaProducerMixin,
)
from sentientcity.schemas.events import (
    AlertEvent,
    Evidence,
    Location,
    PluginResult,
    Severity,
    TrackEvent,
)

logger = get_logger(__name__)


class PluginConfig(BaseModel):
    """Plugin configuration."""
    name: str
    version: str
    enabled: bool = True
    priority: int = 0
    config: dict[str, Any] = {}


class AnalysisContext(BaseModel):
    """Context passed to plugins for analysis."""
    camera_id: str
    timestamp: datetime
    tracks: list[dict[str, Any]]
    poses: list[dict[str, Any]] | None = None
    density_map: list[list[float]] | None = None
    historical_data: dict[str, Any] = {}


class BasePlugin(ABC):
    """Abstract base class for intelligence plugins."""

    name: str = "base_plugin"
    version: str = "0.1.0"
    
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = get_logger(self.name)

    @abstractmethod
    async def analyze(self, context: AnalysisContext) -> PluginResult:
        """
        Analyze context and return result.
        
        Args:
            context: Analysis context with tracks, poses, etc.
            
        Returns:
            PluginResult with findings and risk contribution
        """
        pass

    async def initialize(self) -> None:
        """Initialize plugin resources."""
        pass

    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        pass


class PluginManager:
    """Manages plugin loading and execution."""

    def __init__(self) -> None:
        self.plugins: dict[str, BasePlugin] = {}
        self.plugin_configs: dict[str, PluginConfig] = {}
        self.logger = get_logger("plugin_manager")

    async def load_plugin(self, config: PluginConfig) -> None:
        """Load a plugin from configuration."""
        try:
            # Dynamic import from plugins directory
            module_path = f"plugins.{config.name}.src.plugin"
            module = importlib.import_module(module_path)
            plugin_class = getattr(module, "Plugin")
            
            plugin = plugin_class(config.config)
            await plugin.initialize()
            
            self.plugins[config.name] = plugin
            self.plugin_configs[config.name] = config
            
            self.logger.info("plugin_loaded", name=config.name, version=config.version)
        except Exception as e:
            self.logger.error("plugin_load_failed", name=config.name, error=str(e))

    async def unload_plugin(self, name: str) -> None:
        """Unload a plugin."""
        if name in self.plugins:
            await self.plugins[name].cleanup()
            del self.plugins[name]
            del self.plugin_configs[name]
            self.logger.info("plugin_unloaded", name=name)

    def get_active_plugins(self) -> list[BasePlugin]:
        """Get list of active plugins sorted by priority."""
        active = [
            (self.plugin_configs[name], plugin)
            for name, plugin in self.plugins.items()
            if self.plugin_configs[name].enabled
        ]
        active.sort(key=lambda x: x[0].priority)
        return [p[1] for p in active]

    async def run_analysis(self, context: AnalysisContext) -> list[PluginResult]:
        """Run all active plugins on context."""
        results = []
        
        for plugin in self.get_active_plugins():
            try:
                start = datetime.utcnow()
                result = await plugin.analyze(context)
                result.processing_time_ms = (
                    datetime.utcnow() - start
                ).total_seconds() * 1000
                results.append(result)
            except Exception as e:
                self.logger.error(
                    "plugin_analysis_failed",
                    plugin=plugin.name,
                    error=str(e),
                )
        
        return results


class RiskEngine:
    """Calculates aggregated risk scores from plugin results."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.weights = self.config.get("weights", {})
        self.threshold = self.config.get("alert_threshold", 70.0)

    def calculate(self, plugin_results: list[PluginResult]) -> float:
        """
        Calculate aggregated risk score.
        
        Args:
            plugin_results: Results from all plugins
            
        Returns:
            Aggregated risk score (0-100)
        """
        if not plugin_results:
            return 0.0
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for result in plugin_results:
            weight = self.weights.get(result.plugin_name, 1.0)
            # Weight by confidence as well
            effective_weight = weight * result.confidence
            
            weighted_sum += result.risk_contribution * effective_weight
            total_weight += effective_weight
        
        if total_weight == 0:
            return 0.0
        
        return min(100.0, weighted_sum / total_weight)

    def should_alert(self, risk_score: float) -> bool:
        """Determine if risk score warrants an alert."""
        return risk_score >= self.threshold

    def get_severity(self, risk_score: float) -> Severity:
        """Map risk score to severity level."""
        if risk_score >= 90:
            return Severity.CRITICAL
        elif risk_score >= 70:
            return Severity.HIGH
        elif risk_score >= 50:
            return Severity.MEDIUM
        else:
            return Severity.LOW


class IntelligenceEngineService(BaseService, KafkaConsumerMixin, KafkaProducerMixin):
    """Intelligence engine orchestrating plugin-based analysis."""

    def __init__(self) -> None:
        super().__init__(
            service_name="intelligence-engine",
            version="0.1.0",
        )
        
        self.plugin_manager = PluginManager()
        self.risk_engine = RiskEngine()
        self._consumer_task: asyncio.Task | None = None
        self._track_buffer: dict[str, list[dict]] = {}
        
        self._register_routes()

    def _register_routes(self) -> None:
        """Register API routes."""
        router = APIRouter(prefix="/api/v1", tags=["intelligence"])

        @router.get("/plugins")
        async def list_plugins() -> list[dict[str, Any]]:
            """List all loaded plugins."""
            return [
                {
                    "name": cfg.name,
                    "version": cfg.version,
                    "enabled": cfg.enabled,
                    "priority": cfg.priority,
                }
                for cfg in self.plugin_manager.plugin_configs.values()
            ]

        @router.post("/plugins/{name}/enable")
        async def enable_plugin(name: str) -> dict[str, str]:
            """Enable a plugin."""
            if name in self.plugin_manager.plugin_configs:
                self.plugin_manager.plugin_configs[name].enabled = True
                return {"status": "enabled", "plugin": name}
            return {"status": "not_found", "plugin": name}

        @router.post("/plugins/{name}/disable")
        async def disable_plugin(name: str) -> dict[str, str]:
            """Disable a plugin."""
            if name in self.plugin_manager.plugin_configs:
                self.plugin_manager.plugin_configs[name].enabled = False
                return {"status": "disabled", "plugin": name}
            return {"status": "not_found", "plugin": name}

        @router.get("/risk/threshold")
        async def get_threshold() -> dict[str, float]:
            """Get alert threshold."""
            return {"threshold": self.risk_engine.threshold}

        @router.post("/risk/threshold")
        async def set_threshold(threshold: float) -> dict[str, float]:
            """Set alert threshold."""
            self.risk_engine.threshold = max(0, min(100, threshold))
            return {"threshold": self.risk_engine.threshold}

        self.app.include_router(router)

    async def startup(self) -> None:
        """Initialize service."""
        await self.start_consumer(["sentient.tracks"])
        await self.start_producer()
        
        # Load default plugins
        await self._load_default_plugins()
        
        self._consumer_task = asyncio.create_task(self._consume_loop())
        self.logger.info("service_started")

    async def shutdown(self) -> None:
        """Cleanup service."""
        if self._consumer_task:
            self._consumer_task.cancel()
        
        # Cleanup plugins
        for name in list(self.plugin_manager.plugins.keys()):
            await self.plugin_manager.unload_plugin(name)
        
        await self.stop_consumer()
        await self.stop_producer()
        self.logger.info("service_stopped")

    async def _load_default_plugins(self) -> None:
        """Load default analysis plugins."""
        default_plugins = [
            PluginConfig(name="crowd_prediction", version="0.1.0", priority=1),
            PluginConfig(name="anomaly_detection", version="0.1.0", priority=2),
            PluginConfig(name="defense_monitoring", version="0.1.0", priority=3),
            PluginConfig(name="traffic_analysis", version="0.1.0", priority=4),
            PluginConfig(name="disaster_detection", version="0.1.0", priority=5),
        ]
        
        for config in default_plugins:
            try:
                await self.plugin_manager.load_plugin(config)
            except Exception as e:
                self.logger.warning(
                    "default_plugin_not_loaded",
                    plugin=config.name,
                    error=str(e),
                )

    async def _consume_loop(self) -> None:
        """Consume track events and run analysis."""
        async for msg in self.kafka_consumer:
            try:
                event = TrackEvent.model_validate_json(msg.value)
                await self._process_track(event)
            except Exception as e:
                logger.error("processing_error", error=str(e))

    async def _process_track(self, event: TrackEvent) -> None:
        """Process track event and run intelligence analysis."""
        camera_id = event.camera_id
        
        # Buffer tracks per camera
        if camera_id not in self._track_buffer:
            self._track_buffer[camera_id] = []
        
        self._track_buffer[camera_id].append(event.model_dump())
        
        # Keep buffer size limited
        if len(self._track_buffer[camera_id]) > 100:
            self._track_buffer[camera_id] = self._track_buffer[camera_id][-100:]
        
        # Create analysis context
        context = AnalysisContext(
            camera_id=camera_id,
            timestamp=event.timestamp,
            tracks=self._track_buffer[camera_id],
        )
        
        # Run plugin analysis
        results = await self.plugin_manager.run_analysis(context)
        
        # Calculate risk
        risk_score = self.risk_engine.calculate(results)
        
        # Generate alert if needed
        if self.risk_engine.should_alert(risk_score):
            await self._generate_alert(event, results, risk_score)

    async def _generate_alert(
        self,
        trigger_event: TrackEvent,
        plugin_results: list[PluginResult],
        risk_score: float,
    ) -> None:
        """Generate and publish alert event."""
        # Determine alert type from plugin findings
        alert_type = "anomaly"
        findings = []
        for result in plugin_results:
            findings.extend(result.findings)
            if result.risk_contribution > 50:
                alert_type = result.plugin_name
        
        alert = AlertEvent(
            source_service=self.service_name,
            alert_type=alert_type,
            severity=self.risk_engine.get_severity(risk_score),
            risk_score=risk_score,
            location=Location(
                camera_id=trigger_event.camera_id,
            ),
            evidence=Evidence(
                track_ids=[trigger_event.track_id],
            ),
            description="; ".join(findings[:5]) if findings else "Risk threshold exceeded",
            recommended_actions=self._get_recommendations(alert_type, risk_score),
        )
        
        await self.publish(
            "sentient.alerts",
            alert.model_dump_json().encode(),
            key=trigger_event.camera_id.encode(),
        )
        
        self.logger.info(
            "alert_generated",
            alert_type=alert_type,
            severity=alert.severity.value,
            risk_score=risk_score,
        )

    def _get_recommendations(self, alert_type: str, risk_score: float) -> list[str]:
        """Get recommended actions based on alert type."""
        recommendations = {
            "crowd_prediction": [
                "Monitor crowd density in affected area",
                "Consider crowd control measures",
                "Alert security personnel",
            ],
            "anomaly_detection": [
                "Review camera feed for suspicious activity",
                "Dispatch security if warranted",
            ],
            "defense_monitoring": [
                "Immediate perimeter check required",
                "Alert security response team",
                "Review access logs",
            ],
            "traffic_analysis": [
                "Check for traffic incidents",
                "Consider traffic redirection",
            ],
            "disaster_detection": [
                "Initiate emergency response protocol",
                "Alert emergency services",
                "Begin evacuation if needed",
            ],
        }
        
        base = recommendations.get(alert_type, ["Review situation and respond accordingly"])
        
        if risk_score >= 90:
            base.insert(0, "CRITICAL: Immediate action required")
        
        return base


def create_app() -> IntelligenceEngineService:
    """Create service instance."""
    return IntelligenceEngineService()


if __name__ == "__main__":
    service = create_app()
    service.run(port=8005)
