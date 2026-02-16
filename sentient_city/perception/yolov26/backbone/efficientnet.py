"""
EfficientNet backbone for YOLOv26 (optional alternative).

Lightweight and efficient backbone option.
"""

import torch
import torch.nn as nn
from typing import List
from loguru import logger


class EfficientNetBackbone(nn.Module):
    """
    EfficientNet backbone (placeholder for future implementation).
    
    This is a placeholder - full EfficientNet implementation would go here.
    """
    
    def __init__(
        self,
        model_name: str = "efficientnet-b0",
        pretrained: bool = True
    ):
        """
        Initialize EfficientNet backbone.
        
        Args:
            model_name: EfficientNet variant name
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        logger.warning("EfficientNetBackbone is a placeholder - not fully implemented")
        
        # Placeholder - would use timm or torchvision EfficientNet
        self.out_channels = [256, 512, 1024]
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass (placeholder)."""
        B, C, H, W = x.shape
        # Placeholder outputs
        return [
            torch.randn(B, 256, H // 8, W // 8, device=x.device),
            torch.randn(B, 512, H // 16, W // 16, device=x.device),
            torch.randn(B, 1024, H // 32, W // 32, device=x.device)
        ]
    
    def get_output_channels(self) -> List[int]:
        """Get output channels."""
        return self.out_channels
