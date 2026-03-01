"""
LSTM-based memory model.

Maintains long-term memory of behaviors and patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger

from ...utils.config import get_config


class MemoryLSTM(nn.Module):
    """
    LSTM-based memory model.
    
    Maintains temporal memory of behaviors and patterns for long-term analysis.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize memory LSTM.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Output projections
        self.pattern_predictor = nn.Linear(hidden_dim, input_dim)
        self.anomaly_detector = nn.Linear(hidden_dim, 1)
        self.memory_embedding = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input sequence [B, T, input_dim]
            hidden: Optional hidden state (h, c)
        
        Returns:
            Dictionary with:
            - pattern_prediction: Predicted next pattern
            - anomaly_score: Anomaly detection score
            - memory_embedding: Memory embedding
            - hidden: Updated hidden state
        """
        # LSTM forward
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Use last output
        last_output = lstm_out[:, -1, :]
        
        # Outputs
        pattern_prediction = self.pattern_predictor(last_output)
        anomaly_score = torch.sigmoid(self.anomaly_detector(last_output))
        memory_embedding = F.normalize(
            self.memory_embedding(last_output), p=2, dim=1
        )
        
        return {
            "pattern_prediction": pattern_prediction,
            "anomaly_score": anomaly_score,
            "memory_embedding": memory_embedding,
            "hidden": hidden,
            "lstm_output": lstm_out
        }
    
    def reset_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reset hidden state.
        
        Args:
            batch_size: Batch size
            device: Device
        
        Returns:
            Zero hidden state (h, c)
        """
        h = torch.zeros(
            self.num_layers, batch_size, self.hidden_dim, device=device
        )
        c = torch.zeros(
            self.num_layers, batch_size, self.hidden_dim, device=device
        )
        return (h, c)
    
    def load_weights(self, weights_path: str) -> None:
        """Load model weights."""
        if Path(weights_path).exists():
            state_dict = torch.load(weights_path, map_location="cpu")
            self.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded memory model weights from {weights_path}")
        else:
            logger.warning(f"Weights not found: {weights_path}")
