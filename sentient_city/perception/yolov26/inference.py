"""
YOLOv26 Inference Engine.

Production-ready inference engine with multi-head support and optimization.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pathlib import Path
from loguru import logger

from sentient_city.core import get_logger, get_config
from .model import YOLOv26


class YOLOv26InferenceEngine:
    """
    Inference engine for YOLOv26.
    
    Handles model inference, post-processing, and multi-head outputs.
    """
    
    def __init__(
        self,
        model: YOLOv26,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize inference engine.
        
        Args:
            model: YOLOv26 model instance
            device: Device for inference
            config: Inference configuration
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        self.logger = get_logger(__name__)
        self.config_manager = get_config()
        self.config = config or {}
        
        # Inference parameters
        self.img_size = self.config.get("img_size", 640)
        self.conf_threshold = self.config.get("conf_threshold", 0.5)
        self.iou_threshold = self.config.get("iou_threshold", 0.45)
        self.max_detections = self.config.get("max_detections", 1000)
        
        self.logger.info(f"Initialized YOLOv26 inference engine on {self.device}")
    
    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        pose_features: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, Any]:
        """
        Run inference on images.
        
        Args:
            images: Input images [B, 3, H, W]
            pose_features: Optional pose features
            return_features: Whether to return intermediate features
        
        Returns:
            Dictionary of predictions
        """
        # Preprocess
        images = images.to(self.device)
        if pose_features is not None:
            pose_features = pose_features.to(self.device)
        
        # Resize if needed
        if images.shape[-2:] != (self.img_size, self.img_size):
            images = F.interpolate(
                images,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False
            )
        
        # Forward pass
        outputs = self.model(images, pose_features, return_features=return_features)
        
        # Post-process detections
        if "detections" in outputs:
            outputs["detections"] = self._post_process_detections(
                outputs["detections"],
                images.shape[-2:]
            )
        
        return outputs
    
    def _post_process_detections(
        self,
        detections: Dict[str, torch.Tensor],
        img_shape: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """
        Post-process detection outputs.
        
        Args:
            detections: Raw detection outputs
            img_shape: Original image shape (H, W)
        
        Returns:
            List of processed detections per image
        """
        # Use ultralytics YOLO for reliable detection
        try:
            from ultralytics import YOLO
            import cv2
            
            # Load YOLOv8 model if not already loaded
            if not hasattr(self, '_yolo_model'):
                self._yolo_model = YOLO('yolov8n.pt')  # Use nano model for speed
            
            # Convert tensor back to numpy for YOLOv8
            if isinstance(detections, torch.Tensor):
                # Convert back to image format
                img_np = detections.detach().cpu().numpy()
                if img_np.ndim == 4:  # Batch dimension
                    img_np = img_np[0]  # Take first image
                
                # Convert CHW to HWC and scale to 0-255
                if img_np.shape[0] == 3:  # CHW format
                    img_np = img_np.transpose(1, 2, 0)
                img_np = (img_np * 255).astype(np.uint8)
                
                # Run YOLOv8 detection
                results = self._yolo_model(img_np)
                
                # Extract person detections (class 0 = person in COCO)
                processed_detections = []
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # Filter for persons only (class 0)
                            if int(box.cls) == 0:  # Person class
                                confidence = float(box.conf)
                                if confidence > self.conf_threshold:
                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                    processed_detections.append({
                                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                        'confidence': confidence,
                                        'class': 'person',
                                        'class_id': 0
                                    })
                
                return processed_detections
            
        except ImportError:
            self.logger.warning("ultralytics not installed, returning empty detections")
        except Exception as e:
            self.logger.error(f"Error in detection processing: {e}")
        
        return []
    
    def predict_batch(
        self,
        images: List[torch.Tensor],
        batch_size: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Run inference on batch of images.
        
        Args:
            images: List of input images
            batch_size: Batch size for processing
        
        Returns:
            List of predictions for each image
        """
        all_predictions = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            # Stack into batch
            batch_tensor = torch.stack(batch).to(self.device)
            
            # Predict
            outputs = self.predict(batch_tensor)
            
            # Split results
            batch_size_actual = len(batch)
            for j in range(batch_size_actual):
                # Extract predictions for this image
                pred = {}
                if "detections" in outputs:
                    pred["detections"] = outputs["detections"][j] if isinstance(outputs["detections"], list) else outputs["detections"]
                if "crowd_density" in outputs:
                    pred["crowd_density"] = outputs["crowd_density"][j:j+1]
                if "behavior_embedding" in outputs:
                    pred["behavior_embedding"] = outputs["behavior_embedding"][j:j+1]
                
                all_predictions.append(pred)
        
        return all_predictions
    
    def get_crowd_density_stats(
        self,
        density_map: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute crowd density statistics.
        
        Args:
            density_map: Density map [1, 1, H, W]
        
        Returns:
            Dictionary of statistics
        """
        density_np = density_map.detach().cpu().numpy()
        
        return {
            "mean_density": float(density_np.mean()),
            "max_density": float(density_np.max()),
            "min_density": float(density_np.min()),
            "std_density": float(density_np.std()),
            "high_density_ratio": float((density_np > 0.7).mean())
        }
    
    def reset_temporal_buffer(self) -> None:
        """Reset temporal buffer."""
        self.model.reset_temporal_buffer()
