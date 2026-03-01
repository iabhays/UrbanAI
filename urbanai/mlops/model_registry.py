"""
Model registry for versioning and management.

Provides model versioning, storage, and retrieval.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from loguru import logger


@dataclass
class ModelMetadata:
    """Model metadata."""
    model_id: str
    version: str
    model_type: str
    model_path: str
    config_path: Optional[str]
    created_at: str
    created_by: str
    description: Optional[str]
    metrics: Dict
    tags: List[str]
    is_production: bool = False


class ModelRegistry:
    """
    Model registry for versioning and management.
    
    Tracks model versions, metadata, and production deployments.
    """
    
    def __init__(self, registry_path: str = "data/model_registry.json"):
        """
        Initialize model registry.
        
        Args:
            registry_path: Path to registry JSON file
        """
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, List[ModelMetadata]] = {}
        self.load_registry()
    
    def load_registry(self):
        """Load registry from file."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r") as f:
                    data = json.load(f)
                    self.models = {
                        model_id: [
                            ModelMetadata(**meta_dict)
                            for meta_dict in versions
                        ]
                        for model_id, versions in data.items()
                    }
                logger.info(f"Loaded model registry with {len(self.models)} models")
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
                self.models = {}
        else:
            self.models = {}
    
    def save_registry(self):
        """Save registry to file."""
        try:
            data = {
                model_id: [asdict(meta) for meta in versions]
                for model_id, versions in self.models.items()
            }
            with open(self.registry_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug("Saved model registry")
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def register_model(
        self,
        model_id: str,
        model_path: str,
        model_type: str,
        version: Optional[str] = None,
        config_path: Optional[str] = None,
        created_by: str = "system",
        description: Optional[str] = None,
        metrics: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
        is_production: bool = False
    ) -> ModelMetadata:
        """
        Register a new model version.
        
        Args:
            model_id: Model identifier
            model_path: Path to model weights
            model_type: Type of model (e.g., "yolov26")
            version: Version string (auto-generated if None)
            config_path: Path to model config
            created_by: Creator identifier
            description: Model description
            metrics: Model performance metrics
            tags: Model tags
            is_production: Whether this is production model
        
        Returns:
            Model metadata
        """
        if version is None:
            version = self._generate_version(model_id)
        
        metadata = ModelMetadata(
            model_id=model_id,
            version=version,
            model_type=model_type,
            model_path=model_path,
            config_path=config_path,
            created_at=datetime.utcnow().isoformat(),
            created_by=created_by,
            description=description,
            metrics=metrics or {},
            tags=tags or [],
            is_production=is_production
        )
        
        if model_id not in self.models:
            self.models[model_id] = []
        
        self.models[model_id].append(metadata)
        self.save_registry()
        
        logger.info(f"Registered model {model_id} version {version}")
        
        return metadata
    
    def _generate_version(self, model_id: str) -> str:
        """Generate version string."""
        if model_id in self.models:
            existing_versions = len(self.models[model_id])
            return f"v{existing_versions + 1}"
        return "v1"
    
    def get_model(
        self,
        model_id: str,
        version: Optional[str] = None
    ) -> Optional[ModelMetadata]:
        """
        Get model metadata.
        
        Args:
            model_id: Model identifier
            version: Version string (None for latest)
        
        Returns:
            Model metadata or None
        """
        if model_id not in self.models:
            return None
        
        versions = self.models[model_id]
        
        if version is None:
            # Return latest version
            return versions[-1] if versions else None
        
        # Find specific version
        for meta in versions:
            if meta.version == version:
                return meta
        
        return None
    
    def get_production_model(self, model_id: str) -> Optional[ModelMetadata]:
        """
        Get production model.
        
        Args:
            model_id: Model identifier
        
        Returns:
            Production model metadata or None
        """
        if model_id not in self.models:
            return None
        
        for meta in reversed(self.models[model_id]):
            if meta.is_production:
                return meta
        
        return None
    
    def set_production(
        self,
        model_id: str,
        version: str,
        is_production: bool = True
    ):
        """
        Set model as production.
        
        Args:
            model_id: Model identifier
            version: Version string
            is_production: Whether to set as production
        """
        meta = self.get_model(model_id, version)
        if meta:
            # Unset other production models
            if is_production:
                for m in self.models[model_id]:
                    m.is_production = False
            
            meta.is_production = is_production
            self.save_registry()
            logger.info(f"Set {model_id} {version} as production: {is_production}")
    
    def list_models(self) -> List[str]:
        """List all model IDs."""
        return list(self.models.keys())
    
    def list_versions(self, model_id: str) -> List[ModelMetadata]:
        """List all versions of a model."""
        return self.models.get(model_id, [])
