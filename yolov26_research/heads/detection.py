"""
SENTIENTCITY AI - YOLOv26 Detection Head
Multi-scale anchor-free detection head
"""

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolov26_research.models.base import ConvBlock


class DetectionHead(nn.Module):
    """
    YOLOv26 Detection Head.
    
    Anchor-free detection with decoupled classification and regression branches.
    Outputs per scale:
    - Classification: [B, num_classes, H, W]
    - Bounding box: [B, 4, H, W]  (x, y, w, h)
    - Objectness: [B, 1, H, W]
    - Embedding: [B, embedding_dim, H, W] (for re-identification)
    """

    def __init__(
        self,
        in_channels: list[int],
        num_classes: int = 80,
        embedding_dim: int = 128,
        num_convs: int = 2,
        use_depthwise: bool = False,
    ) -> None:
        super().__init__()
        
        self.num_classes = num_classes
        self.num_scales = len(in_channels)
        self.embedding_dim = embedding_dim
        
        # Per-scale heads
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.emb_preds = nn.ModuleList()
        
        for in_ch in in_channels:
            # Classification branch
            cls_conv = nn.Sequential(
                *[
                    ConvBlock(in_ch if i == 0 else in_ch, in_ch, 3)
                    for i in range(num_convs)
                ]
            )
            self.cls_convs.append(cls_conv)
            self.cls_preds.append(
                nn.Conv2d(in_ch, num_classes, 1, bias=True)
            )
            
            # Regression branch
            reg_conv = nn.Sequential(
                *[
                    ConvBlock(in_ch if i == 0 else in_ch, in_ch, 3)
                    for i in range(num_convs)
                ]
            )
            self.reg_convs.append(reg_conv)
            self.reg_preds.append(
                nn.Conv2d(in_ch, 4, 1, bias=True)  # x, y, w, h
            )
            self.obj_preds.append(
                nn.Conv2d(in_ch, 1, 1, bias=True)
            )
            
            # Embedding branch for re-identification
            self.emb_preds.append(
                nn.Sequential(
                    ConvBlock(in_ch, in_ch, 3),
                    nn.Conv2d(in_ch, embedding_dim, 1, bias=True),
                )
            )
        
        self._init_bias()

    def _init_bias(self) -> None:
        """Initialize biases for stable training."""
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        
        for cls_pred in self.cls_preds:
            nn.init.constant_(cls_pred.bias, bias_value)
        
        for obj_pred in self.obj_preds:
            nn.init.constant_(obj_pred.bias, bias_value)

    def forward(
        self,
        features: list[torch.Tensor],
    ) -> dict[str, list[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            features: List of feature maps from neck [P3, P4, P5]
            
        Returns:
            Dictionary with predictions per scale
        """
        cls_outputs = []
        reg_outputs = []
        obj_outputs = []
        emb_outputs = []
        
        for i, feat in enumerate(features):
            # Classification
            cls_feat = self.cls_convs[i](feat)
            cls_out = self.cls_preds[i](cls_feat)
            cls_outputs.append(cls_out)
            
            # Regression
            reg_feat = self.reg_convs[i](feat)
            reg_out = self.reg_preds[i](reg_feat)
            obj_out = self.obj_preds[i](reg_feat)
            reg_outputs.append(reg_out)
            obj_outputs.append(obj_out)
            
            # Embeddings
            emb_out = self.emb_preds[i](feat)
            emb_out = F.normalize(emb_out, p=2, dim=1)
            emb_outputs.append(emb_out)
        
        return {
            "cls": cls_outputs,
            "reg": reg_outputs,
            "obj": obj_outputs,
            "emb": emb_outputs,
        }

    def decode_predictions(
        self,
        predictions: dict[str, list[torch.Tensor]],
        strides: list[int] = [8, 16, 32],
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.45,
    ) -> list[dict[str, Any]]:
        """
        Decode predictions to bounding boxes.
        
        Args:
            predictions: Raw model predictions
            strides: Stride for each scale
            conf_threshold: Confidence threshold
            nms_threshold: NMS IoU threshold
            
        Returns:
            List of detections per batch item
        """
        batch_size = predictions["cls"][0].shape[0]
        device = predictions["cls"][0].device
        
        all_boxes = []
        all_scores = []
        all_classes = []
        all_embeddings = []
        
        for scale_idx, stride in enumerate(strides):
            cls = predictions["cls"][scale_idx].sigmoid()
            reg = predictions["reg"][scale_idx]
            obj = predictions["obj"][scale_idx].sigmoid()
            emb = predictions["emb"][scale_idx]
            
            B, _, H, W = cls.shape
            
            # Generate grid
            yv, xv = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing="ij",
            )
            grid = torch.stack([xv, yv], dim=-1).float()
            
            # Decode boxes
            xy = (grid + reg[:, :2].permute(0, 2, 3, 1).sigmoid() * 2 - 0.5) * stride
            wh = (reg[:, 2:].permute(0, 2, 3, 1).sigmoid() * 2) ** 2 * stride * 4
            
            boxes = torch.cat([xy, wh], dim=-1)  # [B, H, W, 4]
            
            # Compute scores
            scores = (cls * obj).permute(0, 2, 3, 1)  # [B, H, W, num_classes]
            
            all_boxes.append(boxes.reshape(B, -1, 4))
            all_scores.append(scores.reshape(B, -1, self.num_classes))
            all_embeddings.append(emb.permute(0, 2, 3, 1).reshape(B, -1, self.embedding_dim))
        
        # Concatenate all scales
        boxes = torch.cat(all_boxes, dim=1)  # [B, N, 4]
        scores = torch.cat(all_scores, dim=1)  # [B, N, num_classes]
        embeddings = torch.cat(all_embeddings, dim=1)  # [B, N, emb_dim]
        
        # Post-process per batch
        results = []
        for b in range(batch_size):
            # Get max class scores
            max_scores, class_ids = scores[b].max(dim=-1)
            
            # Filter by confidence
            mask = max_scores > conf_threshold
            filtered_boxes = boxes[b][mask]
            filtered_scores = max_scores[mask]
            filtered_classes = class_ids[mask]
            filtered_embeddings = embeddings[b][mask]
            
            # Apply NMS per class
            if len(filtered_boxes) > 0:
                # Convert xywh to xyxy for NMS
                xyxy = self._xywh_to_xyxy(filtered_boxes)
                
                # Simple class-agnostic NMS
                keep = self._nms(xyxy, filtered_scores, nms_threshold)
                
                results.append({
                    "boxes": filtered_boxes[keep],
                    "scores": filtered_scores[keep],
                    "classes": filtered_classes[keep],
                    "embeddings": filtered_embeddings[keep],
                })
            else:
                results.append({
                    "boxes": torch.empty(0, 4, device=device),
                    "scores": torch.empty(0, device=device),
                    "classes": torch.empty(0, dtype=torch.long, device=device),
                    "embeddings": torch.empty(0, self.embedding_dim, device=device),
                })
        
        return results

    @staticmethod
    def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
        xy = boxes[..., :2]
        wh = boxes[..., 2:]
        return torch.cat([xy - wh / 2, xy + wh / 2], dim=-1)

    @staticmethod
    def _nms(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        threshold: float,
    ) -> torch.Tensor:
        """Non-maximum suppression."""
        from torchvision.ops import nms
        return nms(boxes, scores, threshold)


class DetectionLoss(nn.Module):
    """Loss function for detection head."""

    def __init__(
        self,
        num_classes: int = 80,
        reg_weight: float = 5.0,
        obj_weight: float = 1.0,
        cls_weight: float = 1.0,
        emb_weight: float = 0.5,
    ) -> None:
        super().__init__()
        
        self.num_classes = num_classes
        self.reg_weight = reg_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.emb_weight = emb_weight
        
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.ciou_loss = CIoULoss()

    def forward(
        self,
        predictions: dict[str, list[torch.Tensor]],
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """
        Compute detection loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary of loss components
        """
        # Simplified loss computation - full implementation would include
        # proper target assignment (SimOTA, etc.)
        device = predictions["cls"][0].device
        
        loss_cls = torch.tensor(0.0, device=device)
        loss_reg = torch.tensor(0.0, device=device)
        loss_obj = torch.tensor(0.0, device=device)
        loss_emb = torch.tensor(0.0, device=device)
        
        # Placeholder - actual implementation requires target assignment
        total_loss = (
            self.cls_weight * loss_cls
            + self.reg_weight * loss_reg
            + self.obj_weight * loss_obj
            + self.emb_weight * loss_emb
        )
        
        return {
            "loss": total_loss,
            "loss_cls": loss_cls,
            "loss_reg": loss_reg,
            "loss_obj": loss_obj,
            "loss_emb": loss_emb,
        }


class CIoULoss(nn.Module):
    """Complete IoU loss for bounding box regression."""

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CIoU loss."""
        # Convert to xyxy
        pred_xyxy = self._xywh_to_xyxy(pred)
        target_xyxy = self._xywh_to_xyxy(target)
        
        # Intersection
        inter_x1 = torch.max(pred_xyxy[..., 0], target_xyxy[..., 0])
        inter_y1 = torch.max(pred_xyxy[..., 1], target_xyxy[..., 1])
        inter_x2 = torch.min(pred_xyxy[..., 2], target_xyxy[..., 2])
        inter_y2 = torch.min(pred_xyxy[..., 3], target_xyxy[..., 3])
        
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h
        
        # Union
        pred_area = pred[..., 2] * pred[..., 3]
        target_area = target[..., 2] * target[..., 3]
        union_area = pred_area + target_area - inter_area + self.eps
        
        iou = inter_area / union_area
        
        # Enclosing box
        enc_x1 = torch.min(pred_xyxy[..., 0], target_xyxy[..., 0])
        enc_y1 = torch.min(pred_xyxy[..., 1], target_xyxy[..., 1])
        enc_x2 = torch.max(pred_xyxy[..., 2], target_xyxy[..., 2])
        enc_y2 = torch.max(pred_xyxy[..., 3], target_xyxy[..., 3])
        
        c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + self.eps
        
        # Center distance
        pred_cx = pred[..., 0]
        pred_cy = pred[..., 1]
        target_cx = target[..., 0]
        target_cy = target[..., 1]
        
        rho2 = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
        
        # Aspect ratio
        v = (4 / (math.pi ** 2)) * torch.pow(
            torch.atan(target[..., 2] / (target[..., 3] + self.eps))
            - torch.atan(pred[..., 2] / (pred[..., 3] + self.eps)),
            2,
        )
        
        with torch.no_grad():
            alpha = v / (1 - iou + v + self.eps)
        
        ciou = iou - rho2 / c2 - alpha * v
        
        return 1 - ciou

    @staticmethod
    def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
        xy = boxes[..., :2]
        wh = boxes[..., 2:]
        return torch.cat([xy - wh / 2, xy + wh / 2], dim=-1)
