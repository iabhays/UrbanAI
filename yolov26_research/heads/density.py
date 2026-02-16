"""
SENTIENTCITY AI - YOLOv26 Crowd Density Head
Density estimation for crowd analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolov26_research.models.base import ConvBlock


class CrowdDensityHead(nn.Module):
    """
    Crowd Density Estimation Head.
    
    Generates density maps for crowd counting and flow analysis.
    Uses multi-scale feature aggregation for accurate density estimation.
    """

    def __init__(
        self,
        in_channels: list[int],
        output_channels: int = 1,
        hidden_channels: int = 256,
        output_stride: int = 4,
    ) -> None:
        super().__init__()
        
        self.output_stride = output_stride
        
        # Multi-scale feature fusion
        self.lateral_convs = nn.ModuleList([
            ConvBlock(in_ch, hidden_channels, 1)
            for in_ch in in_channels
        ])
        
        # Feature pyramid upsampling
        self.fpn_convs = nn.ModuleList([
            ConvBlock(hidden_channels, hidden_channels, 3)
            for _ in in_channels
        ])
        
        # Density regression
        self.density_conv = nn.Sequential(
            ConvBlock(hidden_channels, hidden_channels, 3),
            ConvBlock(hidden_channels, hidden_channels // 2, 3),
            nn.Conv2d(hidden_channels // 2, output_channels, 1),
            nn.ReLU(inplace=True),  # Density must be non-negative
        )
        
        # Uncertainty estimation (optional)
        self.uncertainty_conv = nn.Sequential(
            ConvBlock(hidden_channels, hidden_channels // 2, 3),
            nn.Conv2d(hidden_channels // 2, output_channels, 1),
            nn.Softplus(),  # Positive uncertainty
        )

    def forward(
        self,
        features: list[torch.Tensor],
        return_uncertainty: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            features: Multi-scale features [P3, P4, P5]
            return_uncertainty: Whether to return uncertainty maps
            
        Returns:
            Dictionary with density map and optionally uncertainty
        """
        # Apply lateral convolutions
        laterals = [
            conv(feat) for conv, feat in zip(self.lateral_convs, features)
        ]
        
        # Top-down pathway with upsampling
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        
        # Apply FPN convolutions
        fpn_features = [
            conv(lat) for conv, lat in zip(self.fpn_convs, laterals)
        ]
        
        # Use finest scale for density prediction
        finest_feature = fpn_features[0]
        
        # Generate density map
        density_map = self.density_conv(finest_feature)
        
        result = {"density": density_map}
        
        if return_uncertainty:
            uncertainty = self.uncertainty_conv(finest_feature)
            result["uncertainty"] = uncertainty
        
        return result

    def get_count(self, density_map: torch.Tensor) -> torch.Tensor:
        """
        Get crowd count from density map.
        
        Args:
            density_map: Density map [B, 1, H, W]
            
        Returns:
            Crowd count per batch [B]
        """
        # Scale factor based on output stride
        scale = self.output_stride ** 2
        return density_map.sum(dim=(1, 2, 3)) * scale


class DensityLoss(nn.Module):
    """Loss function for density estimation."""

    def __init__(
        self,
        use_ssim: bool = True,
        ssim_weight: float = 0.1,
        count_weight: float = 0.01,
    ) -> None:
        super().__init__()
        
        self.use_ssim = use_ssim
        self.ssim_weight = ssim_weight
        self.count_weight = count_weight
        
        self.mse = nn.MSELoss()

    def forward(
        self,
        pred_density: torch.Tensor,
        gt_density: torch.Tensor,
        gt_count: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute density loss.
        
        Args:
            pred_density: Predicted density map
            gt_density: Ground truth density map
            gt_count: Optional ground truth count
            
        Returns:
            Dictionary of loss components
        """
        # MSE loss
        loss_mse = self.mse(pred_density, gt_density)
        
        total_loss = loss_mse
        losses = {"loss_mse": loss_mse}
        
        # SSIM loss for structural similarity
        if self.use_ssim:
            loss_ssim = 1 - self._ssim(pred_density, gt_density)
            total_loss = total_loss + self.ssim_weight * loss_ssim
            losses["loss_ssim"] = loss_ssim
        
        # Count loss
        if gt_count is not None:
            pred_count = pred_density.sum(dim=(1, 2, 3))
            loss_count = F.l1_loss(pred_count, gt_count)
            total_loss = total_loss + self.count_weight * loss_count
            losses["loss_count"] = loss_count
        
        losses["loss"] = total_loss
        return losses

    @staticmethod
    def _ssim(
        pred: torch.Tensor,
        target: torch.Tensor,
        window_size: int = 11,
    ) -> torch.Tensor:
        """Compute SSIM between two images."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Gaussian window
        sigma = 1.5
        gauss = torch.exp(
            -torch.arange(window_size).float().sub(window_size // 2).pow(2) / (2 * sigma ** 2)
        )
        gauss = gauss / gauss.sum()
        window = gauss.unsqueeze(0) * gauss.unsqueeze(1)
        window = window.unsqueeze(0).unsqueeze(0).to(pred.device)
        
        mu1 = F.conv2d(pred, window, padding=window_size // 2)
        mu2 = F.conv2d(target, window, padding=window_size // 2)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred ** 2, window, padding=window_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(target ** 2, window, padding=window_size // 2) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=window_size // 2) - mu1_mu2
        
        ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim.mean()
