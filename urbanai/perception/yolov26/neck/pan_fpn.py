"""
PAN-FPN (Path Aggregation Network - Feature Pyramid Network) neck.

Combines top-down and bottom-up pathways for multi-scale feature fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ConvBNSiLU(nn.Module):
    """Convolution + BatchNorm + SiLU."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bn(self.conv(x)))


class PANFPN(nn.Module):
    """
    PAN-FPN neck architecture.
    
    Combines FPN (top-down) and PAN (bottom-up) pathways.
    """
    
    def __init__(
        self,
        in_channels: List[int],
        out_channels: List[int]
    ):
        """
        Initialize PAN-FPN.
        
        Args:
            in_channels: Input channels for each scale [P3, P4, P5]
            out_channels: Output channels for each scale
        """
        super().__init__()
        assert len(in_channels) == len(out_channels), "Input and output channels must match"
        
        self.num_scales = len(in_channels)
        
        # Top-down pathway (FPN)
        self.top_down_convs = nn.ModuleList([
            ConvBNSiLU(in_channels[i], out_channels[i], 1)
            for i in range(self.num_scales)
        ])
        
        # Bottom-up pathway (PAN)
        self.bottom_up_convs = nn.ModuleList([
            ConvBNSiLU(out_channels[i], out_channels[i], 3)
            for i in range(self.num_scales)
        ])
        
        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass through PAN-FPN.
        
        Args:
            features: List of input features [P3, P4, P5]
        
        Returns:
            List of enhanced features [P3, P4, P5]
        """
        # Top-down pathway (FPN)
        top_down = []
        for i, feat in enumerate(features):
            top_down.append(self.top_down_convs[i](feat))
        
        # Upsample and merge (top-down)
        for i in range(self.num_scales - 2, -1, -1):
            upsampled = self.upsample(top_down[i + 1])
            top_down[i] = top_down[i] + upsampled
        
        # Bottom-up pathway (PAN)
        bottom_up = [top_down[0]]
        for i in range(1, self.num_scales):
            downsampled = F.avg_pool2d(bottom_up[-1], 2, 2)
            merged = top_down[i] + downsampled
            bottom_up.append(self.bottom_up_convs[i](merged))
        
        return bottom_up
