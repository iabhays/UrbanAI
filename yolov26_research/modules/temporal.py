"""
SENTIENTCITY AI - YOLOv26 Temporal Modules
Pose fusion and temporal feature buffering
"""

from collections import deque
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseFusionModule(nn.Module):
    """
    Pose-Aware Feature Fusion Module.
    
    Fuses pose skeleton information with visual features
    for enhanced behavior understanding.
    """

    def __init__(
        self,
        visual_channels: int = 256,
        pose_channels: int = 17 * 3,  # 17 keypoints * (x, y, conf)
        hidden_channels: int = 256,
        num_keypoints: int = 17,
    ) -> None:
        super().__init__()
        
        self.num_keypoints = num_keypoints
        
        # Pose encoder
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
        )
        
        # Spatial pose features (for spatial attention)
        self.pose_spatial = nn.Sequential(
            nn.Linear(pose_channels, hidden_channels),
            nn.ReLU(inplace=True),
        )
        
        # Cross-attention: visual attends to pose
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(visual_channels + hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, visual_channels),
        )
        
        # Learnable pose positional encoding
        self.pose_pe = nn.Parameter(torch.randn(1, num_keypoints, hidden_channels))

    def forward(
        self,
        visual_features: torch.Tensor,
        pose_keypoints: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse pose information with visual features.
        
        Args:
            visual_features: Visual features [B, C, H, W] or [B, C]
            pose_keypoints: Pose keypoints [B, K, 3] (x, y, conf)
            
        Returns:
            Fused features with same shape as visual_features
        """
        B = visual_features.shape[0]
        is_spatial = len(visual_features.shape) == 4
        
        if is_spatial:
            _, C, H, W = visual_features.shape
            visual_flat = visual_features.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        else:
            visual_flat = visual_features.unsqueeze(1)  # [B, 1, C]
        
        # Encode pose
        pose_flat = pose_keypoints.flatten(1)  # [B, K*3]
        pose_encoded = self.pose_encoder(pose_flat)  # [B, hidden]
        
        # Create pose tokens for attention
        pose_tokens = self.pose_spatial(pose_flat).unsqueeze(1)  # [B, 1, hidden]
        pose_tokens = pose_tokens.expand(-1, self.num_keypoints, -1)
        pose_tokens = pose_tokens + self.pose_pe
        
        # Cross-attention
        attended, _ = self.cross_attention(
            visual_flat, pose_tokens, pose_tokens
        )
        
        # Fuse
        pose_broadcast = pose_encoded.unsqueeze(1).expand(-1, visual_flat.shape[1], -1)
        fused = self.fusion(torch.cat([attended, pose_broadcast], dim=-1))
        
        # Residual connection
        output = visual_flat + fused
        
        if is_spatial:
            output = output.permute(0, 2, 1).reshape(B, C, H, W)
        else:
            output = output.squeeze(1)
        
        return output


class TemporalFeatureBuffer(nn.Module):
    """
    Temporal Feature Buffer with Memory.
    
    Maintains a buffer of past features for temporal reasoning.
    Uses transformer-based temporal aggregation.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        buffer_size: int = 16,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        
        self.feature_dim = feature_dim
        self.buffer_size = buffer_size
        
        # Temporal positional encoding
        self.temporal_pe = nn.Parameter(
            torch.randn(1, buffer_size, feature_dim)
        )
        
        # Temporal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Output projection
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        
        # Buffer storage (not a parameter, managed externally)
        self._buffer: dict[str, deque] = {}

    def init_buffer(self, buffer_id: str) -> None:
        """Initialize a new buffer for a track/camera."""
        self._buffer[buffer_id] = deque(maxlen=self.buffer_size)

    def update_buffer(
        self,
        buffer_id: str,
        features: torch.Tensor,
    ) -> None:
        """
        Update buffer with new features.
        
        Args:
            buffer_id: Identifier for the buffer
            features: New features [D] or [B, D]
        """
        if buffer_id not in self._buffer:
            self.init_buffer(buffer_id)
        
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        
        self._buffer[buffer_id].append(features.detach().cpu())

    def get_temporal_features(
        self,
        buffer_id: str,
        current_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get temporally aggregated features.
        
        Args:
            buffer_id: Identifier for the buffer
            current_features: Current frame features [B, D]
            
        Returns:
            Temporally aggregated features [B, D]
        """
        device = current_features.device
        B = current_features.shape[0]
        
        if buffer_id not in self._buffer or len(self._buffer[buffer_id]) == 0:
            # No history, return current features
            return current_features
        
        # Gather history
        history = list(self._buffer[buffer_id])
        history_tensor = torch.cat(history, dim=0).to(device)  # [T, D]
        
        # Pad if needed
        T = history_tensor.shape[0]
        if T < self.buffer_size:
            padding = torch.zeros(
                self.buffer_size - T, self.feature_dim, device=device
            )
            history_tensor = torch.cat([padding, history_tensor], dim=0)
        else:
            history_tensor = history_tensor[-self.buffer_size:]
        
        # Add current features
        sequence = history_tensor.unsqueeze(0).expand(B, -1, -1)  # [B, T, D]
        
        # Add positional encoding
        sequence = sequence + self.temporal_pe
        
        # Apply transformer
        encoded = self.temporal_encoder(sequence)
        
        # Use last position (most recent)
        output = encoded[:, -1, :]
        
        # Project and add residual
        output = self.output_proj(output) + current_features
        
        return output

    def forward(
        self,
        features: torch.Tensor,
        buffer_ids: list[str] | None = None,
    ) -> torch.Tensor:
        """
        Process features with temporal context.
        
        Args:
            features: Input features [B, D]
            buffer_ids: Optional buffer IDs per batch item
            
        Returns:
            Temporally enhanced features [B, D]
        """
        if buffer_ids is None:
            # No temporal context, return as is
            return features
        
        B = features.shape[0]
        outputs = []
        
        for i in range(B):
            out = self.get_temporal_features(buffer_ids[i], features[i:i+1])
            self.update_buffer(buffer_ids[i], features[i])
            outputs.append(out)
        
        return torch.cat(outputs, dim=0)

    def clear_buffer(self, buffer_id: str) -> None:
        """Clear a specific buffer."""
        if buffer_id in self._buffer:
            del self._buffer[buffer_id]

    def clear_all_buffers(self) -> None:
        """Clear all buffers."""
        self._buffer.clear()


class TemporalAttentionPooling(nn.Module):
    """Attention-based temporal pooling."""

    def __init__(
        self,
        feature_dim: int = 256,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        
        # Learnable query for pooling
        self.query = nn.Parameter(torch.randn(1, 1, feature_dim))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Pool temporal sequence to single vector.
        
        Args:
            sequence: Temporal sequence [B, T, D]
            
        Returns:
            Pooled features [B, D]
        """
        B = sequence.shape[0]
        query = self.query.expand(B, -1, -1)
        
        pooled, _ = self.attention(query, sequence, sequence)
        
        return pooled.squeeze(1)
