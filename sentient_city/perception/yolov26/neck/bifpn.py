"""
BiFPN (Bidirectional Feature Pyramid Network) neck.

Efficient bidirectional feature fusion with learnable weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from loguru import logger


class BiFPN(nn.Module):
    """
    BiFPN neck architecture.
    
    Bidirectional feature fusion with weighted connections.
    """
    
    def __init__(
        self,
        in_channels: List[int],
        out_channels: List[int],
        num_layers: int = 1
    ):
        """
        Initialize BiFPN.
        
        Args:
            in_channels: Input channels for each scale
            out_channels: Output channels for each scale
            num_layers: Number of BiFPN layers
        """
        super().__init__()
        logger.warning("BiFPN is a simplified placeholder - full implementation needed")
        
        # For now, use PAN-FPN as fallback
        from .pan_fpn import PANFPN
        self.neck = PANFPN(in_channels, out_channels)
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass (placeholder)."""
        return self.neck(features)
