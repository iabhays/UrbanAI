"""
SENTIENTCITY AI - YOLOv26 Base Model
Research-grade multi-head detection architecture
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from pydantic import BaseModel


class ModelConfig(BaseModel):
    """Configuration for YOLOv26 model."""

    # Backbone
    backbone: str = "cspdarknet"
    depth_multiple: float = 1.0
    width_multiple: float = 1.0
    
    # Input
    input_channels: int = 3
    input_size: tuple[int, int] = (640, 640)
    
    # Detection head
    num_classes: int = 80
    anchors_per_scale: int = 3
    num_scales: int = 3
    
    # Crowd density head
    enable_density: bool = True
    density_output_stride: int = 8
    
    # Behavior embedding head
    enable_behavior: bool = True
    behavior_embedding_dim: int = 256
    
    # Pose fusion
    enable_pose_fusion: bool = True
    num_keypoints: int = 17
    
    # Temporal
    enable_temporal: bool = True
    temporal_buffer_size: int = 16
    temporal_hidden_dim: int = 512


class ConvBlock(nn.Module):
    """Standard convolution block with BatchNorm and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        
        if padding is None:
            padding = kernel_size // 2
            
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        
        if activation == "silu":
            self.act = nn.SiLU(inplace=True)
        elif activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "leaky":
            self.act = nn.LeakyReLU(0.1, inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        expansion: float = 0.5,
    ) -> None:
        super().__init__()
        
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvBlock(in_channels, hidden_channels, 1)
        self.conv2 = ConvBlock(hidden_channels, out_channels, 3)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        return x + out if self.add else out


class CSPBlock(nn.Module):
    """Cross Stage Partial block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        shortcut: bool = True,
        expansion: float = 0.5,
    ) -> None:
        super().__init__()
        
        hidden_channels = int(out_channels * expansion)
        
        self.conv1 = ConvBlock(in_channels, hidden_channels, 1)
        self.conv2 = ConvBlock(in_channels, hidden_channels, 1)
        self.conv3 = ConvBlock(2 * hidden_channels, out_channels, 1)
        
        self.blocks = nn.Sequential(
            *[
                Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.blocks(self.conv1(x))
        x2 = self.conv2(x)
        return self.conv3(torch.cat([x1, x2], dim=1))


class SPPFBlock(nn.Module):
    """Spatial Pyramid Pooling - Fast."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        
        hidden_channels = in_channels // 2
        self.conv1 = ConvBlock(in_channels, hidden_channels, 1)
        self.conv2 = ConvBlock(hidden_channels * 4, out_channels, 1)
        self.pool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], dim=1))


class BaseBackbone(nn.Module, ABC):
    """Abstract base class for backbones."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Returns multi-scale features."""
        pass

    @property
    @abstractmethod
    def out_channels(self) -> list[int]:
        """Returns output channels for each scale."""
        pass


class CSPDarknet(BaseBackbone):
    """CSPDarknet backbone for YOLOv26."""

    def __init__(
        self,
        depth_multiple: float = 1.0,
        width_multiple: float = 1.0,
    ) -> None:
        super().__init__()
        
        # Base channel counts
        base_channels = [64, 128, 256, 512, 1024]
        channels = [int(c * width_multiple) for c in base_channels]
        
        # Base depths
        base_depths = [3, 6, 9, 3]
        depths = [max(1, int(d * depth_multiple)) for d in base_depths]
        
        self._out_channels = channels[2:]  # P3, P4, P5
        
        # Stem
        self.stem = ConvBlock(3, channels[0], 6, 2, 2)
        
        # Stages
        self.stage1 = nn.Sequential(
            ConvBlock(channels[0], channels[1], 3, 2),
            CSPBlock(channels[1], channels[1], depths[0]),
        )
        
        self.stage2 = nn.Sequential(
            ConvBlock(channels[1], channels[2], 3, 2),
            CSPBlock(channels[2], channels[2], depths[1]),
        )
        
        self.stage3 = nn.Sequential(
            ConvBlock(channels[2], channels[3], 3, 2),
            CSPBlock(channels[3], channels[3], depths[2]),
        )
        
        self.stage4 = nn.Sequential(
            ConvBlock(channels[3], channels[4], 3, 2),
            CSPBlock(channels[4], channels[4], depths[3]),
            SPPFBlock(channels[4], channels[4]),
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return [p3, p4, p5]

    @property
    def out_channels(self) -> list[int]:
        return self._out_channels


class YOLOv26Base(nn.Module):
    """
    YOLOv26 Base Model.
    
    Multi-head architecture supporting:
    - Object detection
    - Crowd density estimation
    - Behavior embeddings
    - Pose-aware fusion
    - Temporal feature buffering
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        
        # Backbone
        self.backbone = CSPDarknet(
            depth_multiple=config.depth_multiple,
            width_multiple=config.width_multiple,
        )
        
        # Feature Pyramid Network neck will be added in the full model
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Dictionary containing outputs from all heads
        """
        features = self.backbone(x)
        
        return {
            "backbone_features": features,
        }

    @classmethod
    def from_config(cls, config_path: str) -> "YOLOv26Base":
        """Load model from config file."""
        import yaml
        
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        
        config = ModelConfig(**config_dict)
        return cls(config)
