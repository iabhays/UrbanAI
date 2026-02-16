"""
Temporal buffer module for YOLOv26.

Provides temporal smoothing and feature buffering for temporal consistency.
"""

import torch
import torch.nn as nn
from typing import List, Optional
from collections import deque
from loguru import logger


class TemporalBufferModule(nn.Module):
    """
    Temporal feature buffer for temporal smoothing.
    
    Maintains a buffer of recent features and applies temporal smoothing.
    """
    
    def __init__(
        self,
        buffer_size: int = 30,
        smoothing_method: str = "ema",  # "ema", "moving_avg", "weighted"
        alpha: float = 0.7,
        device: Optional[torch.device] = None
    ):
        """
        Initialize temporal buffer.
        
        Args:
            buffer_size: Maximum buffer size (frames)
            smoothing_method: Smoothing method ("ema", "moving_avg", "weighted")
            alpha: EMA alpha parameter (higher = more weight to recent)
            device: Device to store buffer on
        """
        super().__init__()
        self.buffer_size = buffer_size
        self.smoothing_method = smoothing_method
        self.alpha = alpha
        self.device = device
        
        # Buffer storage (not part of model parameters)
        self.buffer: deque = deque(maxlen=buffer_size)
        self.is_initialized = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with temporal smoothing.
        
        Args:
            x: Current features [B, C, H, W]
        
        Returns:
            Smoothed features [B, C, H, W]
        """
        # Add to buffer
        x_detached = x.detach().clone()
        self.buffer.append(x_detached)
        
        if len(self.buffer) == 1:
            # First frame, no smoothing
            return x
        
        # Apply temporal smoothing
        if self.smoothing_method == "ema":
            return self._exponential_moving_average(x)
        elif self.smoothing_method == "moving_avg":
            return self._moving_average()
        elif self.smoothing_method == "weighted":
            return self._weighted_average()
        else:
            logger.warning(f"Unknown smoothing method: {self.smoothing_method}")
            return x
    
    def _exponential_moving_average(self, current: torch.Tensor) -> torch.Tensor:
        """
        Exponential moving average smoothing.
        
        Args:
            current: Current frame features
        
        Returns:
            EMA-smoothed features
        """
        if len(self.buffer) < 2:
            return current
        
        # Start with oldest frame
        smoothed = self.buffer[0].clone()
        
        # Apply EMA recursively
        for i in range(1, len(self.buffer)):
            smoothed = self.alpha * smoothed + (1 - self.alpha) * self.buffer[i]
        
        return smoothed.to(current.device)
    
    def _moving_average(self) -> torch.Tensor:
        """
        Simple moving average smoothing.
        
        Returns:
            Moving average smoothed features
        """
        if len(self.buffer) == 0:
            return None
        
        # Average all frames in buffer
        stacked = torch.stack(list(self.buffer), dim=0)  # [T, B, C, H, W]
        averaged = stacked.mean(dim=0)  # [B, C, H, W]
        
        return averaged
    
    def _weighted_average(self) -> torch.Tensor:
        """
        Weighted average (more weight to recent frames).
        
        Returns:
            Weighted average smoothed features
        """
        if len(self.buffer) == 0:
            return None
        
        # Create weights (exponential decay)
        T = len(self.buffer)
        weights = torch.exp(torch.linspace(-2, 0, T))  # More weight to recent
        weights = weights / weights.sum()  # Normalize
        weights = weights.view(-1, 1, 1, 1).to(self.buffer[0].device)
        
        # Weighted average
        stacked = torch.stack(list(self.buffer), dim=0)  # [T, B, C, H, W]
        weighted = (stacked * weights).sum(dim=0)  # [B, C, H, W]
        
        return weighted
    
    def reset(self):
        """Reset buffer."""
        self.buffer.clear()
        self.is_initialized = False
        logger.debug("Temporal buffer reset")
    
    def get_buffer_stats(self) -> dict:
        """
        Get buffer statistics.
        
        Returns:
            Dictionary with buffer statistics
        """
        return {
            "buffer_size": len(self.buffer),
            "max_size": self.buffer_size,
            "smoothing_method": self.smoothing_method,
            "alpha": self.alpha
        }
    
    def get_smoothed_history(self, num_frames: Optional[int] = None) -> List[torch.Tensor]:
        """
        Get smoothed history for visualization.
        
        Args:
            num_frames: Number of frames to return (None for all)
        
        Returns:
            List of smoothed frames
        """
        if num_frames is None:
            num_frames = len(self.buffer)
        
        frames = list(self.buffer)[-num_frames:]
        
        # Apply smoothing to each frame
        smoothed_frames = []
        temp_buffer = deque(maxlen=self.buffer_size)
        
        for frame in frames:
            temp_buffer.append(frame)
            if len(temp_buffer) > 1:
                if self.smoothing_method == "ema":
                    smoothed = self._exponential_moving_average(frame)
                elif self.smoothing_method == "moving_avg":
                    stacked = torch.stack(list(temp_buffer), dim=0)
                    smoothed = stacked.mean(dim=0)
                else:
                    smoothed = frame
                smoothed_frames.append(smoothed)
            else:
                smoothed_frames.append(frame)
        
        return smoothed_frames
