"""
SENTIENTCITY AI - YOLOv26 Behavior Embedding Head
Generates embeddings for behavior understanding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolov26_research.models.base import ConvBlock


class BehaviorEmbeddingHead(nn.Module):
    """
    Behavior Embedding Head for action/behavior representation.
    
    Generates dense embeddings that capture behavioral patterns
    useful for anomaly detection and action recognition.
    """

    def __init__(
        self,
        in_channels: list[int],
        embedding_dim: int = 256,
        hidden_channels: int = 512,
        num_attention_heads: int = 8,
        use_temporal: bool = True,
    ) -> None:
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.use_temporal = use_temporal
        
        # Feature aggregation from multiple scales
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                ConvBlock(in_ch, hidden_channels, 1),
                ConvBlock(hidden_channels, hidden_channels, 3),
            )
            for in_ch in in_channels
        ])
        
        # Global context aggregation
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Self-attention for spatial relationships
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=num_attention_heads,
            dropout=0.1,
            batch_first=True,
        )
        
        # Embedding projection
        self.embed_proj = nn.Sequential(
            nn.Linear(hidden_channels * len(in_channels), hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels, embedding_dim),
        )
        
        # Per-location embeddings (for dense prediction)
        self.dense_embed = nn.Sequential(
            ConvBlock(hidden_channels, hidden_channels, 3),
            nn.Conv2d(hidden_channels, embedding_dim, 1),
        )

    def forward(
        self,
        features: list[torch.Tensor],
        return_dense: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            features: Multi-scale features [P3, P4, P5]
            return_dense: Whether to return dense embeddings
            
        Returns:
            Dictionary with global and optionally dense embeddings
        """
        B = features[0].shape[0]
        
        # Process each scale
        scale_features = []
        for conv, feat in zip(self.scale_convs, features):
            processed = conv(feat)
            scale_features.append(processed)
        
        # Global features via pooling
        global_features = []
        for feat in scale_features:
            pooled = self.global_pool(feat).flatten(1)
            global_features.append(pooled)
        
        # Concatenate global features
        global_concat = torch.cat(global_features, dim=1)
        
        # Project to embedding space
        global_embedding = self.embed_proj(global_concat)
        global_embedding = F.normalize(global_embedding, p=2, dim=1)
        
        result = {"global_embedding": global_embedding}
        
        if return_dense:
            # Use finest scale for dense embeddings
            finest = scale_features[0]
            
            # Apply self-attention
            B, C, H, W = finest.shape
            flat = finest.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
            attended, _ = self.attention(flat, flat, flat)
            attended = attended.permute(0, 2, 1).reshape(B, C, H, W)
            
            # Generate dense embeddings
            dense_embedding = self.dense_embed(attended)
            dense_embedding = F.normalize(dense_embedding, p=2, dim=1)
            
            result["dense_embedding"] = dense_embedding
        
        return result


class BehaviorContrastiveLoss(nn.Module):
    """Contrastive loss for behavior embeddings."""

    def __init__(
        self,
        temperature: float = 0.07,
        margin: float = 0.5,
    ) -> None:
        super().__init__()
        
        self.temperature = temperature
        self.margin = margin

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            embeddings: Normalized embeddings [B, D]
            labels: Class labels [B]
            
        Returns:
            Contrastive loss
        """
        # Compute similarity matrix
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create positive/negative masks
        labels = labels.unsqueeze(0)
        positive_mask = (labels == labels.T).float()
        negative_mask = (labels != labels.T).float()
        
        # Remove self-similarity
        eye = torch.eye(embeddings.shape[0], device=embeddings.device)
        positive_mask = positive_mask - eye
        
        # InfoNCE loss
        exp_sim = torch.exp(similarity)
        
        # Positive pairs
        pos_sim = (exp_sim * positive_mask).sum(dim=1)
        
        # All pairs (for denominator)
        all_sim = (exp_sim * (1 - eye)).sum(dim=1)
        
        # Loss
        loss = -torch.log(pos_sim / (all_sim + 1e-8) + 1e-8)
        
        # Only count samples with positive pairs
        valid_mask = positive_mask.sum(dim=1) > 0
        if valid_mask.sum() > 0:
            loss = loss[valid_mask].mean()
        else:
            loss = torch.tensor(0.0, device=embeddings.device)
        
        return loss


class ActionClassifier(nn.Module):
    """Action classification head using behavior embeddings."""

    def __init__(
        self,
        embedding_dim: int = 256,
        num_actions: int = 60,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_actions),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Classify actions from embeddings.
        
        Args:
            embeddings: Behavior embeddings [B, D]
            
        Returns:
            Action logits [B, num_actions]
        """
        return self.classifier(embeddings)
