"""
Crowd density estimation head for YOLOv26.

Estimates crowd density maps from feature representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class CrowdDensityHead(nn.Module):
    """
    Crowd density estimation head.
    
    Produces density maps indicating crowd density at each spatial location.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        use_attention: bool = True
    ):
        """
        Initialize crowd density head.
        
        Args:
            in_channels: Input feature channels
            hidden_channels: Hidden layer channels (default: in_channels // 2)
            use_attention: Whether to use attention mechanism
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels or in_channels // 2
        self.use_attention = use_attention
        
        # Feature refinement
        self.conv1 = nn.Conv2d(in_channels, self.hidden_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(self.hidden_channels)
        
        self.conv2 = nn.Conv2d(self.hidden_channels, self.hidden_channels // 2, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(self.hidden_channels // 2)
        
        # Attention mechanism (optional)
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Conv2d(self.hidden_channels // 2, self.hidden_channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.hidden_channels // 4, 1, 1),
                nn.Sigmoid()
            )
        
        # Density output
        self.density_conv = nn.Conv2d(
            self.hidden_channels // 2,
            1,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        self.activation = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input features [B, C, H, W]
        
        Returns:
            Dictionary with:
            - density_map: Density map [B, 1, H, W] (0-1 normalized)
            - density_score: Global density score [B, 1]
        """
        # Feature refinement
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        
        # Attention (if enabled)
        if self.use_attention:
            attn = self.attention(x)
            x = x * attn
        
        # Density prediction
        density_map = self.sigmoid(self.density_conv(x))  # [B, 1, H, W]
        
        # Global density score (mean of density map)
        density_score = density_map.mean(dim=[2, 3], keepdim=True)  # [B, 1, 1, 1]
        density_score = density_score.squeeze(-1).squeeze(-1)  # [B, 1]
        
        return {
            "density_map": density_map,
            "density_score": density_score
        }
    
    def compute_density_statistics(
        self,
        density_map: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute density statistics.
        
        Args:
            density_map: Density map [B, 1, H, W]
        
        Returns:
            Dictionary with density statistics
        """
        density_map_np = density_map.detach().cpu().numpy()
        
        return {
            "mean_density": float(density_map_np.mean()),
            "max_density": float(density_map_np.max()),
            "min_density": float(density_map_np.min()),
            "std_density": float(density_map_np.std()),
            "high_density_ratio": float((density_map_np > 0.7).mean())
        }
