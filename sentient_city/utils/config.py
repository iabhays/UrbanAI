"""
Configuration management utilities.

Provides centralized configuration loading and validation.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from loguru import logger


class ConfigLoader:
    """Loads and manages configuration from YAML files and environment variables."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to configuration YAML file. If None, uses default.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self.load()
    
    def load(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Configuration dictionary.
        """
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}. Using defaults.")
            return {}
        
        with open(self.config_path, "r") as f:
            self._config = yaml.safe_load(f) or {}
        
        # Override with environment variables
        self._override_from_env()
        
        logger.info(f"Configuration loaded from {self.config_path}")
        return self._config
    
    def _override_from_env(self) -> None:
        """Override configuration values with environment variables."""
        for key, value in os.environ.items():
            if key.startswith("SENTIENT_"):
                # Convert SENTIENT_EDGE_AI__MODEL__WEIGHTS_PATH to nested dict
                parts = key.replace("SENTIENT_", "").lower().split("__")
                self._set_nested(self._config, parts, value)
    
    def _set_nested(self, d: Dict, keys: list, value: Any) -> None:
        """Set nested dictionary value."""
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., "edge_ai.model.weights_path")
            default: Default value if key not found
        
        Returns:
            Configuration value or default.
        """
        keys = key.split(".")
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Section name (e.g., "edge_ai")
        
        Returns:
            Section configuration dictionary.
        """
        return self._config.get(section, {})
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get full configuration dictionary."""
        return self._config


# Global configuration instance
_config_loader: Optional[ConfigLoader] = None


def get_config(config_path: Optional[str] = None) -> ConfigLoader:
    """
    Get global configuration loader instance.
    
    Args:
        config_path: Optional path to config file.
    
    Returns:
        ConfigLoader instance.
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_path)
    return _config_loader
