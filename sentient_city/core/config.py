"""
Enhanced configuration management system.

Provides centralized configuration with research lab support,
environment variable overrides, and validation.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, List, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, ValidationError
from loguru import logger


@dataclass
class ConfigSection:
    """Configuration section wrapper."""
    data: Dict[str, Any]
    name: str
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from section."""
        keys = key.split(".")
        value = self.data
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set value in section."""
        keys = key.split(".")
        d = self.data
        
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        
        d[keys[-1]] = value


class ConfigManager:
    """
    Enhanced configuration manager.
    
    Supports:
    - YAML configuration files
    - Environment variable overrides
    - Research lab configurations
    - Configuration validation
    - Runtime updates
    """
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        research_config_path: Optional[Union[str, Path]] = None,
        validate: bool = True
    ):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to main configuration file
            research_config_path: Path to research lab configuration
            validate: Whether to validate configuration
        """
        # Default paths
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
        
        if research_config_path is None:
            research_config_path = Path(__file__).parent.parent.parent / "configs" / "research_config.yaml"
        
        self.config_path = Path(config_path)
        self.research_config_path = Path(research_config_path)
        self.validate = validate
        
        # Configuration storage
        self._config: Dict[str, Any] = {}
        self._research_config: Dict[str, Any] = {}
        self._sections: Dict[str, ConfigSection] = {}
        
        # Load configurations
        self.load()
    
    def load(self) -> None:
        """Load all configuration files."""
        # Load main config
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                self._config = yaml.safe_load(f) or {}
            logger.info(f"Loaded main config from {self.config_path}")
        else:
            logger.warning(f"Config file not found: {self.config_path}")
            self._config = {}
        
        # Load research config
        if self.research_config_path.exists():
            with open(self.research_config_path, "r") as f:
                self._research_config = yaml.safe_load(f) or {}
            logger.info(f"Loaded research config from {self.research_config_path}")
        else:
            logger.debug(f"Research config not found: {self.research_config_path}")
            self._research_config = {}
        
        # Override with environment variables
        self._override_from_env()
        
        # Create section wrappers
        self._create_sections()
        
        # Validate if requested
        if self.validate:
            self._validate()
    
    def _override_from_env(self) -> None:
        """Override configuration with environment variables."""
        for key, value in os.environ.items():
            if key.startswith("SENTIENT_"):
                # Convert SENTIENT_EDGE_AI__MODEL__DEVICE to nested dict
                parts = key.replace("SENTIENT_", "").lower().split("__")
                self._set_nested(self._config, parts, value)
            
            elif key.startswith("RESEARCH_"):
                # Research lab environment variables
                parts = key.replace("RESEARCH_", "").lower().split("__")
                self._set_nested(self._research_config, parts, value)
    
    def _set_nested(self, d: Dict, keys: List[str], value: Any) -> None:
        """Set nested dictionary value."""
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = self._parse_value(value)
    
    def _parse_value(self, value: str) -> Any:
        """Parse environment variable value."""
        # Try to parse as boolean
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        
        # Try to parse as number
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _create_sections(self) -> None:
        """Create section wrappers for easy access."""
        for key, value in self._config.items():
            if isinstance(value, dict):
                self._sections[key] = ConfigSection(data=value, name=key)
    
    def _validate(self) -> None:
        """Validate configuration structure."""
        # Basic validation - check required sections
        required_sections = ["system", "perception", "streaming"]
        
        for section in required_sections:
            if section not in self._config:
                logger.warning(f"Missing configuration section: {section}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., "perception.yolov26.model.device")
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        """
        # Check research config first
        research_value = self._get_nested(self._research_config, key)
        if research_value is not None:
            return research_value
        
        # Check main config
        main_value = self._get_nested(self._config, key)
        if main_value is not None:
            return main_value
        
        return default
    
    def _get_nested(self, d: Dict, key: str) -> Any:
        """Get nested dictionary value."""
        keys = key.split(".")
        value = d
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return None
            else:
                return None
        
        return value
    
    def get_section(self, section: str) -> ConfigSection:
        """
        Get configuration section.
        
        Args:
            section: Section name (e.g., "perception")
        
        Returns:
            ConfigSection wrapper
        """
        if section in self._sections:
            return self._sections[section]
        
        # Create new section if not exists
        data = self._config.get(section, {})
        section_wrapper = ConfigSection(data=data, name=section)
        self._sections[section] = section_wrapper
        return section_wrapper
    
    def get_research_config(self, key: str, default: Any = None) -> Any:
        """
        Get research lab configuration.
        
        Args:
            key: Configuration key
            default: Default value
        
        Returns:
            Research configuration value
        """
        return self._get_nested(self._research_config, key) or default
    
    def update(self, key: str, value: Any) -> None:
        """
        Update configuration at runtime.
        
        Args:
            key: Configuration key
            value: New value
        """
        keys = key.split(".")
        d = self._config
        
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        
        d[keys[-1]] = value
        logger.info(f"Updated configuration: {key} = {value}")
    
    def save(self, path: Optional[Path] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            path: Path to save (default: original config path)
        """
        save_path = path or self.config_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, "w") as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved configuration to {save_path}")
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get full configuration dictionary."""
        return self._config.copy()
    
    @property
    def research_config(self) -> Dict[str, Any]:
        """Get research configuration dictionary."""
        return self._research_config.copy()


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def get_config(
    config_path: Optional[Union[str, Path]] = None,
    research_config_path: Optional[Union[str, Path]] = None
) -> ConfigManager:
    """
    Get global configuration manager instance.
    
    Args:
        config_path: Optional path to config file
        research_config_path: Optional path to research config file
    
    Returns:
        ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path, research_config_path)
    return _config_manager
