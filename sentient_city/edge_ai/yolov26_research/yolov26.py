"""
YOLOv26 Research Variant Implementation.

Multi-head detection architecture with crowd density estimation,
behavior embedding extraction, and pose fusion capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import yaml
from loguru import logger

from ultralytics import YOLO
from ultralytics.models import YOLO as YOLOBase


class CSPDarknetBlock(nn.Module):
    """CSP Darknet block for YOLOv26 backbone."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        shortcut: bool = True
    ):
        super().__init__()
        hidden_channels = out_channels // 2
        
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, 1, 0)
        self.conv2 = nn.Conv2d(in_channels, hidden_channels, 1, 1, 0)
        self.conv3 = nn.Conv2d(2 * hidden_channels, out_channels, 1, 1, 0)
        
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.blocks = nn.Sequential(*[
            self._make_conv_block(hidden_channels, hidden_channels, shortcut)
            for _ in range(num_blocks)
        ])
        
        self.activation = nn.SiLU()
    
    def _make_conv_block(self, in_ch: int, out_ch: int, shortcut: bool) -> nn.Module:
        """Create a convolutional block."""
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU()
        ]
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x1 = self.activation(self.bn1(self.conv1(x)))
        x2 = self.activation(self.bn2(self.conv2(x)))
        x1 = self.blocks(x1)
        out = torch.cat([x1, x2], dim=1)
        out = self.activation(self.bn3(self.conv3(out)))
        return out


class PANFPN(nn.Module):
    """PAN-FPN neck architecture."""
    
    def __init__(self, in_channels: List[int], out_channels: List[int]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Top-down pathway
        self.top_down_convs = nn.ModuleList([
            nn.Conv2d(in_channels[i], out_channels[i], 1, 1, 0)
            for i in range(len(in_channels))
        ])
        
        # Bottom-up pathway
        self.bottom_up_convs = nn.ModuleList([
            nn.Conv2d(out_channels[i], out_channels[i], 3, 1, 1)
            for i in range(len(in_channels))
        ])
        
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.activation = nn.SiLU()
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass through PAN-FPN."""
        # Top-down pathway
        top_down = []
        for i, feat in enumerate(features):
            top_down.append(self.activation(self.top_down_convs[i](feat)))
        
        # Upsample and merge
        for i in range(len(top_down) - 2, -1, -1):
            top_down[i] = top_down[i] + self.upsample(top_down[i + 1])
        
        # Bottom-up pathway
        bottom_up = [top_down[0]]
        for i in range(1, len(top_down)):
            downsampled = F.avg_pool2d(bottom_up[-1], 2, 2)
            merged = top_down[i] + downsampled
            bottom_up.append(self.activation(self.bottom_up_convs[i](merged)))
        
        return bottom_up


class DetectionHead(nn.Module):
    """Standard object detection head."""
    
    def __init__(self, in_channels: int, num_classes: int, num_anchors: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Bounding box regression (x, y, w, h, conf)
        self.bbox_conv = nn.Conv2d(in_channels, num_anchors * 5, 1, 1, 0)
        # Class prediction
        self.cls_conv = nn.Conv2d(in_channels, num_anchors * num_classes, 1, 1, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        bbox = self.bbox_conv(x)  # [B, anchors*5, H, W]
        cls = self.cls_conv(x)    # [B, anchors*classes, H, W]
        return bbox, cls


class CrowdDensityHead(nn.Module):
    """Crowd density estimation head."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels // 4, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_channels // 4, 1, 1, 1, 0)
        self.activation = nn.SiLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        density = self.sigmoid(self.conv3(x))  # [B, 1, H, W]
        return density


class BehaviorEmbeddingHead(nn.Module):
    """Behavior embedding extraction head."""
    
    def __init__(self, in_channels: int, embedding_dim: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels // 2, embedding_dim, 1, 1, 0)
        self.activation = nn.SiLU()
        self.normalize = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.activation(self.conv1(x))
        embedding = self.conv2(x)  # [B, embedding_dim, H, W]
        
        if self.normalize:
            embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding


class PoseFusionHead(nn.Module):
    """Pose-aware fusion head."""
    
    def __init__(self, in_channels: int, pose_dim: int = 17, fusion_method: str = "concat"):
        super().__init__()
        self.pose_dim = pose_dim
        self.fusion_method = fusion_method
        
        if fusion_method == "concat":
            self.fusion_conv = nn.Conv2d(in_channels + pose_dim, in_channels, 1, 1, 0)
        elif fusion_method == "attention":
            self.attention = nn.MultiheadAttention(in_channels, num_heads=8, batch_first=True)
            self.pose_proj = nn.Linear(pose_dim, in_channels)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        self.activation = nn.SiLU()
    
    def forward(
        self,
        features: torch.Tensor,
        pose_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with pose fusion."""
        if pose_features is None:
            return features
        
        if self.fusion_method == "concat":
            fused = torch.cat([features, pose_features], dim=1)
            return self.activation(self.fusion_conv(fused))
        else:  # attention
            B, C, H, W = features.shape
            feat_flat = features.view(B, C, H * W).permute(0, 2, 1)
            pose_flat = self.pose_proj(pose_features.view(B, self.pose_dim, H * W).permute(0, 2, 1))
            fused, _ = self.attention(feat_flat, pose_flat, pose_flat)
            return fused.permute(0, 2, 1).view(B, C, H, W)


class TemporalBuffer(nn.Module):
    """Temporal feature buffer for temporal smoothing."""
    
    def __init__(self, buffer_size: int = 30, alpha: float = 0.7):
        super().__init__()
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.buffer: List[torch.Tensor] = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add to buffer and return smoothed output."""
        self.buffer.append(x.detach())
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        
        if len(self.buffer) == 1:
            return x
        
        # Exponential moving average
        smoothed = self.buffer[0]
        for i in range(1, len(self.buffer)):
            smoothed = self.alpha * smoothed + (1 - self.alpha) * self.buffer[i]
        
        return smoothed
    
    def reset(self):
        """Reset buffer."""
        self.buffer.clear()


class YOLOv26Detector(nn.Module):
    """
    YOLOv26 Research Variant Model.
    
    Multi-head detection architecture with:
    - Object detection
    - Crowd density estimation
    - Behavior embedding extraction
    - Pose fusion capabilities
    - Temporal feature buffering
    """
    
    def __init__(
        self,
        num_classes: int = 80,
        input_size: Tuple[int, int] = (640, 640),
        embedding_dim: int = 512,
        pose_dim: int = 17,
        config_path: Optional[str] = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        
        # Load configuration if provided
        if config_path:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        else:
            config = {}
        
        model_config = config.get("model", {})
        backbone_config = model_config.get("backbone", {})
        neck_config = model_config.get("neck", {})
        heads_config = model_config.get("heads", {})
        
        # Backbone (simplified CSPDarknet)
        self.backbone = self._build_backbone(backbone_config)
        
        # Neck (PAN-FPN)
        in_channels = [256, 512, 1024]
        out_channels = neck_config.get("out_channels", in_channels)
        self.neck = PANFPN(in_channels, out_channels)
        
        # Detection heads
        self.detection_head = DetectionHead(
            out_channels[-1],
            num_classes,
            heads_config.get("detection", {}).get("anchors", 3)
        )
        
        # Crowd density head
        if heads_config.get("crowd_density", {}).get("enabled", True):
            self.crowd_density_head = CrowdDensityHead(out_channels[-1])
        else:
            self.crowd_density_head = None
        
        # Behavior embedding head
        if heads_config.get("behavior_embedding", {}).get("enabled", True):
            self.behavior_embedding_head = BehaviorEmbeddingHead(
                out_channels[-1],
                heads_config.get("behavior_embedding", {}).get("embedding_dim", embedding_dim)
            )
        else:
            self.behavior_embedding_head = None
        
        # Pose fusion head
        if heads_config.get("pose_fusion", {}).get("enabled", True):
            fusion_method = heads_config.get("pose_fusion", {}).get("fusion_method", "concat")
            self.pose_fusion_head = PoseFusionHead(
                out_channels[-1],
                heads_config.get("pose_fusion", {}).get("pose_dim", pose_dim),
                fusion_method
            )
        else:
            self.pose_fusion_head = None
        
        # Temporal buffer
        inference_config = config.get("inference", {})
        temporal_config = inference_config.get("temporal", {})
        if temporal_config.get("enable_temporal_smoothing", True):
            self.temporal_buffer = TemporalBuffer(
                temporal_config.get("buffer_size", 30),
                temporal_config.get("temporal_alpha", 0.7)
            )
        else:
            self.temporal_buffer = None
    
    def _build_backbone(self, config: Dict) -> nn.Module:
        """Build backbone network."""
        # Simplified backbone - in production, use full CSPDarknet
        layers = [
            nn.Conv2d(3, 64, 6, 2, 2),  # Initial conv
            nn.BatchNorm2d(64),
            nn.SiLU(),
            CSPDarknetBlock(64, 128, 1),
            nn.Conv2d(128, 128, 3, 2, 1),  # Downsample
            CSPDarknetBlock(128, 256, 2),
            nn.Conv2d(256, 256, 3, 2, 1),  # Downsample
            CSPDarknetBlock(256, 512, 8),
            nn.Conv2d(512, 512, 3, 2, 1),  # Downsample
            CSPDarknetBlock(512, 1024, 4),
        ]
        return nn.Sequential(*layers)
    
    def forward(
        self,
        x: torch.Tensor,
        pose_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input image tensor [B, 3, H, W]
            pose_features: Optional pose features [B, pose_dim, H, W]
        
        Returns:
            Dictionary containing:
            - detections: Detection outputs
            - crowd_density: Crowd density map (if enabled)
            - behavior_embedding: Behavior embeddings (if enabled)
        """
        # Backbone
        features = self.backbone(x)
        
        # Extract multi-scale features (simplified - in production, extract properly)
        feat_small = features
        feat_medium = F.avg_pool2d(features, 2, 2)
        feat_large = F.avg_pool2d(features, 4, 4)
        
        # Neck
        neck_features = self.neck([feat_small, feat_medium, feat_large])
        
        # Main feature (use largest scale)
        main_feat = neck_features[-1]
        
        # Pose fusion
        if self.pose_fusion_head is not None and pose_features is not None:
            main_feat = self.pose_fusion_head(main_feat, pose_features)
        
        # Temporal smoothing
        if self.temporal_buffer is not None:
            main_feat = self.temporal_buffer(main_feat)
        
        # Detection head
        bbox, cls = self.detection_head(main_feat)
        
        outputs = {
            "detections": {
                "bbox": bbox,
                "cls": cls
            }
        }
        
        # Crowd density
        if self.crowd_density_head is not None:
            outputs["crowd_density"] = self.crowd_density_head(main_feat)
        
        # Behavior embedding
        if self.behavior_embedding_head is not None:
            outputs["behavior_embedding"] = self.behavior_embedding_head(main_feat)
        
        return outputs
    
    def load_weights(self, weights_path: str) -> None:
        """Load model weights."""
        if Path(weights_path).exists():
            state_dict = torch.load(weights_path, map_location="cpu")
            self.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded weights from {weights_path}")
        else:
            logger.warning(f"Weights file not found: {weights_path}. Using random initialization.")
    
    def reset_temporal_buffer(self):
        """Reset temporal buffer."""
        if self.temporal_buffer is not None:
            self.temporal_buffer.reset()
