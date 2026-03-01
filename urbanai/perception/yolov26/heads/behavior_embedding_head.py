"""
Behavior embedding head for YOLOv26.

Extracts behavior-aware embeddings for downstream analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class BehaviorEmbeddingHead(nn.Module):
    """
    Behavior embedding extraction head.
    
    Produces discriminative embeddings for behavior analysis.
    """
    
    def __init__(
        self,
        in_channels: int,
        embedding_dim: int = 512,
        normalize: bool = True,
        use_temporal: bool = True
    ):
        """
        Initialize behavior embedding head.
        
        Args:
            in_channels: Input feature channels
            embedding_dim: Output embedding dimension
            normalize: Whether to L2 normalize embeddings
            use_temporal: Whether to use temporal context
        """
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.normalize = normalize
        self.use_temporal = use_temporal
        
        # Feature projection
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels // 4, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(in_channels // 4)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Embedding projection
        self.embedding_proj = nn.Sequential(
            nn.Linear(in_channels // 4, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Temporal context (optional)
        if self.use_temporal:
            self.temporal_conv = nn.Conv1d(
                embedding_dim,
                embedding_dim,
                kernel_size=3,
                padding=1
            )
        
        self.activation = nn.ReLU(inplace=True)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        temporal_context: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input features [B, C, H, W]
            temporal_context: Optional temporal context [B, T, embedding_dim]
        
        Returns:
            Dictionary with:
            - embedding: Behavior embeddings [B, embedding_dim]
            - spatial_embedding: Spatial embedding map [B, embedding_dim, H, W]
        """
        # Feature extraction
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        
        # Global embedding
        x_pooled = self.global_pool(x)  # [B, C//4, 1, 1]
        x_pooled = x_pooled.view(x_pooled.size(0), -1)  # [B, C//4]
        
        embedding = self.embedding_proj(x_pooled)  # [B, embedding_dim]
        
        # Spatial embedding map
        spatial_embedding = self._extract_spatial_embedding(x)
        
        # Temporal context fusion (if enabled)
        if self.use_temporal and temporal_context is not None:
            embedding = self._apply_temporal_context(embedding, temporal_context)
        
        # Normalize if requested
        if self.normalize:
            embedding = F.normalize(embedding, p=2, dim=1)
            spatial_embedding = F.normalize(spatial_embedding, p=2, dim=1)
        
        return {
            "embedding": embedding,
            "spatial_embedding": spatial_embedding
        }
    
    def _extract_spatial_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial embedding map.
        
        Args:
            x: Feature map [B, C, H, W]
        
        Returns:
            Spatial embedding [B, embedding_dim, H, W]
        """
        B, C, H, W = x.shape
        
        # Project to embedding dimension
        x_flat = x.view(B, C, H * W)  # [B, C, H*W]
        
        # Simple projection (in production, use more sophisticated method)
        # For now, use global embedding broadcasted
        # Full implementation would use conv layers
        
        return x_flat  # Placeholder
    
    def _apply_temporal_context(
        self,
        embedding: torch.Tensor,
        temporal_context: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply temporal context to embedding.
        
        Args:
            embedding: Current embedding [B, embedding_dim]
            temporal_context: Temporal context [B, T, embedding_dim]
        
        Returns:
            Temporally enhanced embedding [B, embedding_dim]
        """
        # Concatenate current with temporal context
        B, T, D = temporal_context.shape
        
        # Stack: [current, context]
        stacked = torch.cat([
            embedding.unsqueeze(1),  # [B, 1, D]
            temporal_context  # [B, T, D]
        ], dim=1)  # [B, T+1, D]
        
        # Apply temporal convolution
        stacked = stacked.transpose(1, 2)  # [B, D, T+1]
        enhanced = self.temporal_conv(stacked)  # [B, D, T+1]
        enhanced = enhanced.transpose(1, 2)  # [B, T+1, D]
        
        # Use last timestep
        return enhanced[:, -1, :]  # [B, D]
