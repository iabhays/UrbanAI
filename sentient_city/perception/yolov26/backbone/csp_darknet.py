"""
CSPDarknet backbone for YOLOv26.

Modular CSPDarknet implementation with configurable depth and width.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from loguru import logger


class ConvBNSiLU(nn.Module):
    """Convolution + BatchNorm + SiLU activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bn(self.conv(x)))


class CSPDarknetBlock(nn.Module):
    """CSP Darknet block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        shortcut: bool = True,
        expansion: float = 0.5
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        
        self.conv1 = ConvBNSiLU(in_channels, hidden_channels, 1, 1, 0)
        self.conv2 = ConvBNSiLU(in_channels, hidden_channels, 1, 1, 0)
        self.conv3 = ConvBNSiLU(2 * hidden_channels, out_channels, 1, 1, 0)
        
        self.blocks = nn.Sequential(*[
            self._make_conv_block(hidden_channels, hidden_channels, shortcut)
            for _ in range(num_blocks)
        ])
    
    def _make_conv_block(self, in_ch: int, out_ch: int, shortcut: bool) -> nn.Module:
        """Create a convolutional block."""
        layers = [
            ConvBNSiLU(in_ch, out_ch, 3, 1, 1),
            ConvBNSiLU(out_ch, out_ch, 3, 1, 1)
        ]
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x1 = self.blocks(x1)
        out = torch.cat([x1, x2], dim=1)
        out = self.conv3(out)
        return out


class CSPDarknet(nn.Module):
    """
    CSPDarknet backbone for YOLOv26.
    
    Configurable depth and width multipliers for different model sizes.
    """
    
    def __init__(
        self,
        depth_multiple: float = 1.0,
        width_multiple: float = 1.0,
        activation: str = "SiLU",
        input_channels: int = 3
    ):
        """
        Initialize CSPDarknet.
        
        Args:
            depth_multiple: Depth multiplier for scaling number of blocks
            width_multiple: Width multiplier for scaling channels
            activation: Activation function name
            input_channels: Input image channels
        """
        super().__init__()
        
        # Base channel configuration
        base_channels = [64, 128, 256, 512, 1024]
        base_depths = [1, 2, 8, 8, 4]
        
        # Apply width multiplier
        channels = [int(c * width_multiple) for c in base_channels]
        # Apply depth multiplier
        depths = [max(round(d * depth_multiple), 1) for d in base_depths]
        
        # Stem
        self.stem = ConvBNSiLU(input_channels, channels[0], 6, 2, 2)
        
        # Stage 1
        self.stage1 = nn.Sequential(
            ConvBNSiLU(channels[0], channels[1], 3, 2, 1),
            CSPDarknetBlock(channels[1], channels[1], depths[0], shortcut=True)
        )
        
        # Stage 2
        self.stage2 = nn.Sequential(
            ConvBNSiLU(channels[1], channels[2], 3, 2, 1),
            CSPDarknetBlock(channels[2], channels[2], depths[1], shortcut=True)
        )
        
        # Stage 3
        self.stage3 = nn.Sequential(
            ConvBNSiLU(channels[2], channels[3], 3, 2, 1),
            CSPDarknetBlock(channels[3], channels[3], depths[2], shortcut=True)
        )
        
        # Stage 4
        self.stage4 = nn.Sequential(
            ConvBNSiLU(channels[3], channels[4], 3, 2, 1),
            CSPDarknetBlock(channels[4], channels[4], depths[3], shortcut=True)
        )
        
        self.out_channels = [channels[2], channels[3], channels[4]]  # P3, P4, P5
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, 3, H, W]
        
        Returns:
            List of feature maps at different scales [P3, P4, P5]
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        p3 = x  # P3: 1/8 scale
        
        x = self.stage3(x)
        p4 = x  # P4: 1/16 scale
        
        x = self.stage4(x)
        p5 = x  # P5: 1/32 scale
        
        return [p3, p4, p5]
    
    def get_output_channels(self) -> List[int]:
        """Get output channels for each scale."""
        return self.out_channels
