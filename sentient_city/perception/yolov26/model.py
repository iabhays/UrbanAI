"""
YOLOv26 Research Variant - Main Model.

Enhanced implementation with research lab integration, experiment tracking,
and production-ready features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from loguru import logger

from sentient_city.core import get_config, get_logger
from sentient_city.core.experiment_tracking import get_experiment_tracker
from .backbone import CSPDarknet
from .neck import PANFPN
from .heads import (
    DetectionHead,
    MultiScaleDetectionHead,
    CrowdDensityHead,
    BehaviorEmbeddingHead
)
from .modules import PoseFusionModule, TemporalBufferModule


class YOLOv26(nn.Module):
    """
    YOLOv26 Research Variant Model.
    
    Multi-head detection architecture with:
    - Object detection
    - Crowd density estimation
    - Behavior embedding extraction
    - Pose fusion capabilities
    - Temporal feature buffering
    - Research lab integration
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        num_classes: int = 80,
        input_size: Tuple[int, int] = (640, 640),
        embedding_dim: int = 512,
        pose_dim: int = 17,
        experiment_id: Optional[str] = None
    ):
        """
        Initialize YOLOv26 model.
        
        Args:
            config_path: Path to model configuration YAML
            num_classes: Number of detection classes
            input_size: Input image size (H, W)
            embedding_dim: Behavior embedding dimension
            pose_dim: Pose feature dimension
            experiment_id: Optional experiment ID for tracking
        """
        super().__init__()
        
        self.logger = get_logger(__name__)
        self.config_manager = get_config()
        self.experiment_id = experiment_id
        
        # Load configuration
        if config_path:
            self.config = self._load_config(config_path)
        else:
            self.config = self.config_manager.get_section("perception.yolov26").data
        
        model_config = self.config.get("model", {})
        backbone_config = model_config.get("backbone", {})
        neck_config = model_config.get("neck", {})
        heads_config = model_config.get("heads", {})
        
        self.num_classes = num_classes
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.pose_dim = pose_dim
        
        # Build backbone
        backbone_type = backbone_config.get("type", "CSPDarknet")
        self.backbone = self._build_backbone(backbone_type, backbone_config)
        
        # Get backbone output channels
        backbone_out_channels = self._get_backbone_output_channels(backbone_type)
        
        # Build neck
        neck_type = neck_config.get("type", "PAN-FPN")
        in_channels = neck_config.get("in_channels", backbone_out_channels)
        out_channels = neck_config.get("out_channels", in_channels)
        self.neck = self._build_neck(neck_type, in_channels, out_channels)
        
        # Build detection heads
        detection_config = heads_config.get("detection", {})
        if detection_config.get("enabled", True):
            self.detection_head = self._build_detection_head(
                out_channels,
                detection_config
            )
        else:
            self.detection_head = None
        
        # Build crowd density head
        crowd_config = heads_config.get("crowd_density", {})
        if crowd_config.get("enabled", True):
            self.crowd_density_head = CrowdDensityHead(
                out_channels[-1],
                output_dim=crowd_config.get("output_dim", 1)
            )
        else:
            self.crowd_density_head = None
        
        # Build behavior embedding head
        behavior_config = heads_config.get("behavior_embedding", {})
        if behavior_config.get("enabled", True):
            self.behavior_embedding_head = BehaviorEmbeddingHead(
                out_channels[-1],
                embedding_dim=behavior_config.get("embedding_dim", embedding_dim),
                normalize=behavior_config.get("normalize", True)
            )
        else:
            self.behavior_embedding_head = None
        
        # Build pose fusion module
        pose_config = heads_config.get("pose_fusion", {})
        if pose_config.get("enabled", True):
            self.pose_fusion_module = PoseFusionModule(
                in_channels=out_channels[-1],
                pose_dim=pose_config.get("pose_dim", pose_dim),
                fusion_method=pose_config.get("fusion_method", "concat")
            )
        else:
            self.pose_fusion_module = None
        
        # Build temporal buffer
        inference_config = self.config.get("inference", {})
        temporal_config = inference_config.get("temporal", {})
        if temporal_config.get("enable_temporal_smoothing", True):
            self.temporal_buffer = TemporalBufferModule(
                buffer_size=temporal_config.get("buffer_size", 30),
                alpha=temporal_config.get("temporal_alpha", 0.7)
            )
        else:
            self.temporal_buffer = None
        
        # Training state
        self.training_step = 0
        self.training_epoch = 0
        
        self.logger.info(f"Initialized YOLOv26 model: {num_classes} classes, input_size={input_size}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        import yaml
        if Path(config_path).exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
            self.logger.info(f"Loaded config from {config_path}")
            return config
        else:
            self.logger.warning(f"Config file not found: {config_path}")
            return {}
    
    def _build_backbone(self, backbone_type: str, config: Dict) -> nn.Module:
        """Build backbone network."""
        if backbone_type == "CSPDarknet":
            return CSPDarknet(
                depth_multiple=config.get("depth_multiple", 1.0),
                width_multiple=config.get("width_multiple", 1.0),
                activation=config.get("activation", "SiLU")
            )
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")
    
    def _get_backbone_output_channels(self, backbone_type: str) -> List[int]:
        """Get backbone output channels for each scale."""
        if backbone_type == "CSPDarknet":
            # CSPDarknet outputs at 3 scales: P3, P4, P5
            return [256, 512, 1024]
        else:
            return [256, 512, 1024]  # Default
    
    def _build_neck(self, neck_type: str, in_channels: List[int], out_channels: List[int]) -> nn.Module:
        """Build neck network."""
        if neck_type == "PAN-FPN":
            return PANFPN(in_channels, out_channels)
        elif neck_type == "BiFPN":
            from .neck import BiFPN
            return BiFPN(in_channels, out_channels)
        else:
            raise ValueError(f"Unknown neck type: {neck_type}")
    
    def _build_detection_head(
        self,
        out_channels: List[int],
        config: Dict
    ) -> nn.Module:
        """Build detection head."""
        num_anchors = config.get("anchors", 3)
        strides = config.get("stride", [8, 16, 32])
        
        if len(out_channels) > 1:
            # Multi-scale detection head
            return MultiScaleDetectionHead(
                in_channels_list=out_channels,
                num_classes=self.num_classes,
                num_anchors=num_anchors,
                strides=strides
            )
        else:
            # Single-scale detection head
            return DetectionHead(
                in_channels=out_channels[-1],
                num_classes=self.num_classes,
                num_anchors=num_anchors,
                stride=strides[-1]
            )
    
    def forward(
        self,
        x: torch.Tensor,
        pose_features: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input image tensor [B, 3, H, W]
            pose_features: Optional pose features [B, pose_dim, H, W]
            return_features: Whether to return intermediate features
        
        Returns:
            Dictionary containing:
            - detections: Detection outputs
            - crowd_density: Crowd density map (if enabled)
            - behavior_embedding: Behavior embeddings (if enabled)
            - features: Intermediate features (if return_features=True)
        """
        # Backbone
        backbone_features = self.backbone(x)
        
        # Extract multi-scale features
        if isinstance(backbone_features, (list, tuple)):
            multi_scale_features = backbone_features
        else:
            # Extract from single feature map
            B, C, H, W = backbone_features.shape
            multi_scale_features = [
                backbone_features,  # P5
                F.interpolate(backbone_features, scale_factor=2, mode="nearest"),  # P4
                F.interpolate(backbone_features, scale_factor=4, mode="nearest")  # P3
            ]
        
        # Neck
        neck_features = self.neck(multi_scale_features)
        
        # Main feature (use largest scale for auxiliary heads)
        main_feat = neck_features[-1]
        
        # Pose fusion
        if self.pose_fusion_module is not None and pose_features is not None:
            main_feat = self.pose_fusion_module(main_feat, pose_features)
        
        # Temporal smoothing (only during inference)
        if not self.training and self.temporal_buffer is not None:
            main_feat = self.temporal_buffer(main_feat)
        
        # Detection head
        if self.detection_head is not None:
            if isinstance(self.detection_head, MultiScaleDetectionHead):
                detection_outputs = self.detection_head(neck_features)
            else:
                detection_outputs = self.detection_head(main_feat)
        else:
            detection_outputs = None
        
        # Build outputs
        outputs = {}
        
        if detection_outputs is not None:
            outputs["detections"] = detection_outputs
        
        # Crowd density
        if self.crowd_density_head is not None:
            outputs["crowd_density"] = self.crowd_density_head(main_feat)
        
        # Behavior embedding
        if self.behavior_embedding_head is not None:
            outputs["behavior_embedding"] = self.behavior_embedding_head(main_feat)
        
        # Intermediate features (for research/debugging)
        if return_features:
            outputs["features"] = {
                "backbone": backbone_features,
                "neck": neck_features,
                "main": main_feat
            }
        
        return outputs
    
    def load_weights(self, weights_path: str, strict: bool = False) -> None:
        """
        Load model weights.
        
        Args:
            weights_path: Path to weights file
            strict: Whether to strictly enforce state dict matching
        """
        if not Path(weights_path).exists():
            self.logger.warning(f"Weights file not found: {weights_path}")
            return
        
        try:
            state_dict = torch.load(weights_path, map_location="cpu")
            
            # Handle different state dict formats
            if isinstance(state_dict, dict):
                if "model" in state_dict:
                    state_dict = state_dict["model"]
                elif "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
            
            self.load_state_dict(state_dict, strict=strict)
            self.logger.info(f"Loaded weights from {weights_path}")
        except Exception as e:
            self.logger.error(f"Failed to load weights: {e}")
            if strict:
                raise
    
    def save_weights(self, weights_path: str) -> None:
        """
        Save model weights.
        
        Args:
            weights_path: Path to save weights
        """
        Path(weights_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), weights_path)
        self.logger.info(f"Saved weights to {weights_path}")
    
    def freeze_backbone(self) -> None:
        """Freeze backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.logger.info("Frozen backbone parameters")
    
    def unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
        self.logger.info("Unfrozen all parameters")
    
    def reset_temporal_buffer(self) -> None:
        """Reset temporal buffer."""
        if self.temporal_buffer is not None:
            self.temporal_buffer.reset()
            self.logger.debug("Reset temporal buffer")
    
    def get_model_size(self) -> Dict[str, int]:
        """
        Get model size statistics.
        
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024)  # Assuming float32
        }
    
    def log_training_step(
        self,
        loss: float,
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Log training step to experiment tracker.
        
        Args:
            loss: Training loss
            metrics: Additional metrics
        """
        if self.experiment_id:
            tracker = get_experiment_tracker()
            tracker.log_metrics(
                self.experiment_id,
                metrics={"loss": loss, **(metrics or {})},
                step=self.training_step,
                epoch=self.training_epoch,
                loss=loss
            )
        
        self.training_step += 1
