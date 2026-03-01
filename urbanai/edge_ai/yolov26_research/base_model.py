"""
Base model interface for YOLOv26 research variant.

Provides abstract base class and common utilities for detection models.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import yaml
from loguru import logger


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for detection models.
    
    Provides common interface and utilities for all detection models.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize base model.
        
        Args:
            config_path: Path to model configuration YAML
        """
        super().__init__()
        self.config_path = config_path
        self.config: Dict = {}
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> Dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
        
        Returns:
            Configuration dictionary
        """
        if Path(config_path).exists():
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f) or {}
            logger.info(f"Loaded config from {config_path}")
        else:
            logger.warning(f"Config file not found: {config_path}")
            self.config = {}
        
        return self.config
    
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass (must be implemented by subclasses).
        
        Args:
            x: Input tensor
            **kwargs: Additional arguments
        
        Returns:
            Dictionary of outputs
        """
        pass
    
    def load_weights(self, weights_path: str, strict: bool = True) -> None:
        """
        Load model weights.
        
        Args:
            weights_path: Path to weights file
            strict: Whether to strictly enforce state dict matching
        """
        if not Path(weights_path).exists():
            logger.warning(f"Weights file not found: {weights_path}")
            return
        
        try:
            state_dict = torch.load(weights_path, map_location="cpu")
            
            # Handle different state dict formats
            if isinstance(state_dict, dict) and "model" in state_dict:
                state_dict = state_dict["model"]
            
            self.load_state_dict(state_dict, strict=strict)
            logger.info(f"Loaded weights from {weights_path}")
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
            if strict:
                raise
    
    def save_weights(self, weights_path: str) -> None:
        """
        Save model weights.
        
        Args:
            weights_path: Path to save weights
        """
        Path(weights_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), weights_path)
        logger.info(f"Saved weights to {weights_path}")
    
    def get_model_size(self) -> Dict[str, int]:
        """
        Get model size statistics.
        
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params
        }
    
    def export_onnx(
        self,
        output_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 640, 640),
        opset_version: int = 11
    ) -> None:
        """
        Export model to ONNX format.
        
        Args:
            output_path: Path to save ONNX model
            input_shape: Input tensor shape
            opset_version: ONNX opset version
        """
        try:
            import torch.onnx
            
            self.eval()
            dummy_input = torch.randn(*input_shape)
            
            torch.onnx.export(
                self,
                dummy_input,
                output_path,
                input_names=["input"],
                output_names=["output"],
                opset_version=opset_version,
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
            )
            
            logger.info(f"Exported ONNX model to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export ONNX: {e}")
            raise
    
    def freeze_backbone(self) -> None:
        """Freeze backbone parameters for fine-tuning."""
        # Subclasses should implement this
        logger.warning("freeze_backbone not implemented for this model")
    
    def unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
