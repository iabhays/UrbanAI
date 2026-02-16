"""
Transformer-based behavior model.

Analyzes temporal behavior patterns using transformer architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger

from ...utils.config import get_config


class BehaviorTransformer(nn.Module):
    """
    Transformer-based behavior analysis model.
    
    Processes temporal sequences of behavior embeddings to understand
    behavior patterns and predict future behaviors.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_seq_length: int = 128
    ):
        """
        Initialize behavior transformer.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        self.behavior_classifier = nn.Linear(hidden_dim, 10)  # 10 behavior classes
        self.risk_predictor = nn.Linear(hidden_dim, 1)
        self.embedding_extractor = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input sequence [B, T, input_dim]
            mask: Optional attention mask [B, T]
        
        Returns:
            Dictionary with:
            - behavior_logits: Behavior classification logits
            - risk_score: Risk prediction score
            - embeddings: Behavior embeddings
        """
        B, T, _ = x.shape
        
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Create padding mask if needed
        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=x.device)
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=~mask)
        
        # Use last valid token for prediction
        # Get last non-padded token for each sequence
        lengths = mask.sum(dim=1)
        last_indices = (lengths - 1).clamp(min=0)
        last_hidden = x[torch.arange(B), last_indices]
        
        # Outputs
        behavior_logits = self.behavior_classifier(last_hidden)
        risk_score = torch.sigmoid(self.risk_predictor(last_hidden))
        embeddings = F.normalize(self.embedding_extractor(last_hidden), p=2, dim=1)
        
        return {
            "behavior_logits": behavior_logits,
            "risk_score": risk_score,
            "embeddings": embeddings,
            "hidden_states": x
        }
    
    def load_weights(self, weights_path: str) -> None:
        """Load model weights."""
        if Path(weights_path).exists():
            state_dict = torch.load(weights_path, map_location="cpu")
            self.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded behavior model weights from {weights_path}")
        else:
            logger.warning(f"Weights not found: {weights_path}")


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding.
        
        Args:
            x: Input tensor [B, T, d_model]
        
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return x
