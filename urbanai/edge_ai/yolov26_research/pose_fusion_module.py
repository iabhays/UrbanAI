"""
Pose fusion module for YOLOv26.

Fuses pose information with detection features for pose-aware detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Literal


class PoseFusionModule(nn.Module):
    """
    Pose-aware feature fusion module.
    
    Fuses pose keypoint information with detection features.
    """
    
    def __init__(
        self,
        in_channels: int,
        pose_dim: int = 17,  # MediaPipe has 17 keypoints
        fusion_method: Literal["concat", "attention", "gated"] = "attention",
        hidden_dim: Optional[int] = None
    ):
        """
        Initialize pose fusion module.
        
        Args:
            in_channels: Input feature channels
            pose_dim: Pose feature dimension (number of keypoints)
            fusion_method: Fusion method ("concat", "attention", "gated")
            hidden_dim: Hidden dimension for fusion (default: in_channels)
        """
        super().__init__()
        self.in_channels = in_channels
        self.pose_dim = pose_dim
        self.fusion_method = fusion_method
        self.hidden_dim = hidden_dim or in_channels
        
        if fusion_method == "concat":
            # Simple concatenation
            self.fusion_conv = nn.Conv2d(
                in_channels + pose_dim,
                in_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )
            self.bn = nn.BatchNorm2d(in_channels)
        
        elif fusion_method == "attention":
            # Attention-based fusion
            self.pose_proj = nn.Conv2d(pose_dim, in_channels, 1, 1, 0)
            self.attention = nn.MultiheadAttention(
                in_channels,
                num_heads=8,
                batch_first=False,
                dropout=0.1
            )
            self.norm = nn.LayerNorm(in_channels)
        
        elif fusion_method == "gated":
            # Gated fusion
            self.pose_proj = nn.Conv2d(pose_dim, in_channels, 1, 1, 0)
            self.gate = nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels, 1, 1, 0),
                nn.Sigmoid()
            )
            self.fusion_conv = nn.Conv2d(in_channels * 2, in_channels, 1, 1, 0)
            self.bn = nn.BatchNorm2d(in_channels)
        
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        self.activation = nn.ReLU(inplace=True)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        features: torch.Tensor,
        pose_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with pose fusion.
        
        Args:
            features: Detection features [B, C, H, W]
            pose_features: Pose features [B, pose_dim, H, W] or None
        
        Returns:
            Fused features [B, C, H, W]
        """
        if pose_features is None:
            return features
        
        B, C, H, W = features.shape
        
        if self.fusion_method == "concat":
            # Concatenate and project
            fused = torch.cat([features, pose_features], dim=1)  # [B, C+pose_dim, H, W]
            fused = self.fusion_conv(fused)  # [B, C, H, W]
            fused = self.bn(fused)
            fused = self.activation(fused)
        
        elif self.fusion_method == "attention":
            # Reshape for attention: [B, C, H, W] -> [H*W, B, C]
            feat_flat = features.view(B, C, H * W).permute(2, 0, 1)  # [H*W, B, C]
            
            # Project pose features
            pose_proj = self.pose_proj(pose_features)  # [B, C, H, W]
            pose_flat = pose_proj.view(B, C, H * W).permute(2, 0, 1)  # [H*W, B, C]
            
            # Self-attention: features attend to pose
            fused_flat, _ = self.attention(feat_flat, pose_flat, pose_flat)  # [H*W, B, C]
            fused_flat = self.norm(fused_flat + feat_flat)  # Residual + norm
            
            # Reshape back
            fused = fused_flat.permute(1, 2, 0).view(B, C, H, W)
        
        elif self.fusion_method == "gated":
            # Project pose
            pose_proj = self.pose_proj(pose_features)  # [B, C, H, W]
            
            # Concatenate
            concat = torch.cat([features, pose_proj], dim=1)  # [B, 2*C, H, W]
            
            # Gating mechanism
            gate = self.gate(concat)  # [B, C, H, W]
            
            # Gated fusion
            fused = self.fusion_conv(concat)  # [B, C, H, W]
            fused = fused * gate  # Apply gate
            fused = self.bn(fused)
            fused = self.activation(fused)
        
        return fused
    
    def compute_fusion_weights(
        self,
        features: torch.Tensor,
        pose_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute fusion weights (for visualization/analysis).
        
        Args:
            features: Detection features
            pose_features: Pose features
        
        Returns:
            Fusion weights [B, 1, H, W]
        """
        if self.fusion_method == "gated":
            pose_proj = self.pose_proj(pose_features)
            concat = torch.cat([features, pose_proj], dim=1)
            weights = self.gate(concat)
            return weights
        else:
            # Return uniform weights for other methods
            B, C, H, W = features.shape
            return torch.ones(B, 1, H, W, device=features.device)
