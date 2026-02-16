"""
Edge detector interface and implementation.

Provides unified interface for detection operations.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import cv2
from loguru import logger

from ...utils.config import get_config
from ..yolov26_research.yolov26 import YOLOv26Detector


class Detection:
    """Single detection result."""
    
    def __init__(
        self,
        bbox: np.ndarray,
        confidence: float,
        class_id: int,
        class_name: str,
        embedding: Optional[np.ndarray] = None
    ):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.embedding = embedding
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "bbox": self.bbox.tolist(),
            "confidence": float(self.confidence),
            "class_id": int(self.class_id),
            "class_name": self.class_name
        }
        if self.embedding is not None:
            result["embedding"] = self.embedding.tolist()
        return result


class EdgeDetector:
    """
    Edge AI detector interface.
    
    Handles detection, crowd density estimation, and behavior embedding extraction.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize edge detector.
        
        Args:
            model_path: Path to model weights
            config_path: Path to model configuration
            device: Device to run inference on (cuda:0, cpu, etc.)
        """
        self.config = get_config()
        edge_config = self.config.get_section("edge_ai")
        
        self.model_path = model_path or edge_config.get("model", {}).get("weights_path")
        self.config_path = config_path or "configs/yolov26_config.yaml"
        self.device = device or edge_config.get("model", {}).get("device", "cuda:0")
        
        if not torch.cuda.is_available() and "cuda" in self.device:
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"
        
        self.device = torch.device(self.device)
        
        # Detection parameters
        det_config = edge_config.get("detection", {})
        self.conf_threshold = edge_config.get("model", {}).get("conf_threshold", 0.5)
        self.iou_threshold = edge_config.get("model", {}).get("iou_threshold", 0.45)
        self.max_detections = det_config.get("max_detections", 1000)
        self.target_classes = det_config.get("classes", [0])  # Person class
        
        # Feature extraction flags
        self.enable_crowd_density = det_config.get("enable_crowd_density", True)
        self.enable_behavior_embedding = det_config.get("enable_behavior_embedding", True)
        
        # Initialize model
        self.model = None
        self._load_model()
        
        # COCO class names (simplified)
        self.class_names = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
            "truck", "boat", "traffic light", "fire hydrant", "stop sign",
            "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
            "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
            "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
            "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
            "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv",
            "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
            "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush"
        ]
    
    def _load_model(self) -> None:
        """Load detection model."""
        try:
            self.model = YOLOv26Detector(
                num_classes=80,
                input_size=(640, 640),
                config_path=self.config_path
            )
            
            if self.model_path and Path(self.model_path).exists():
                self.model.load_weights(self.model_path)
            
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model loaded on device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            Preprocessed tensor [1, 3, H, W]
        """
        # Resize to model input size
        h, w = image.shape[:2]
        target_size = (640, 640)
        
        # Resize maintaining aspect ratio
        scale = min(target_size[0] / h, target_size[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to target size
        pad_h = target_size[0] - new_h
        pad_w = target_size[1] - new_w
        
        padded = cv2.copyMakeBorder(
            resized, 0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114)
        )
        
        # Convert to RGB and normalize
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)
    
    def postprocess(
        self,
        outputs: Dict[str, torch.Tensor],
        original_shape: Tuple[int, int],
        input_shape: Tuple[int, int] = (640, 640)
    ) -> Tuple[List[Detection], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Postprocess model outputs.
        
        Args:
            outputs: Model outputs dictionary
            original_shape: Original image shape (H, W)
            input_shape: Model input shape (H, W)
        
        Returns:
            Tuple of (detections, crowd_density_map, behavior_embeddings)
        """
        detections = []
        crowd_density = None
        behavior_embeddings = None
        
        # Process detections
        det_outputs = outputs.get("detections", {})
        bbox = det_outputs.get("bbox", None)
        cls = det_outputs.get("cls", None)
        
        if bbox is not None and cls is not None:
            # Decode detections (simplified - in production, use proper NMS)
            detections = self._decode_detections(
                bbox, cls, original_shape, input_shape
            )
        
        # Process crowd density
        if "crowd_density" in outputs:
            density_map = outputs["crowd_density"].cpu().numpy()[0, 0]
            # Resize to original image size
            h, w = original_shape
            crowd_density = cv2.resize(density_map, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Process behavior embeddings
        if "behavior_embedding" in outputs:
            embeddings = outputs["behavior_embedding"].cpu().numpy()[0]
            # Resize to original image size
            h, w = original_shape
            behavior_embeddings = np.zeros((embeddings.shape[0], h, w), dtype=np.float32)
            for i in range(embeddings.shape[0]):
                behavior_embeddings[i] = cv2.resize(
                    embeddings[i], (w, h), interpolation=cv2.INTER_LINEAR
                )
        
        return detections, crowd_density, behavior_embeddings
    
    def _decode_detections(
        self,
        bbox: torch.Tensor,
        cls: torch.Tensor,
        original_shape: Tuple[int, int],
        input_shape: Tuple[int, int]
    ) -> List[Detection]:
        """Decode detection outputs."""
        # Simplified decoding - in production, implement proper anchor-based decoding
        # This is a placeholder for the actual NMS and decoding logic
        detections = []
        
        # Convert to numpy
        bbox_np = bbox.cpu().numpy()[0]
        cls_np = cls.cpu().numpy()[0]
        
        # Scale factors
        scale_h = original_shape[0] / input_shape[0]
        scale_w = original_shape[1] / input_shape[1]
        
        # Decode boxes (simplified - assumes format [x, y, w, h, conf])
        # In production, implement proper anchor decoding
        # For now, return empty list as placeholder
        # Actual implementation would:
        # 1. Reshape bbox and cls according to grid
        # 2. Apply sigmoid/softmax
        # 3. Decode anchor boxes
        # 4. Apply NMS
        # 5. Filter by confidence and class
        
        return detections
    
    def detect(
        self,
        image: np.ndarray,
        pose_features: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Perform detection on image.
        
        Args:
            image: Input image (BGR format)
            pose_features: Optional pose features for fusion
        
        Returns:
            Detection results dictionary
        """
        original_shape = image.shape[:2]
        
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Prepare pose features if provided
        pose_tensor = None
        if pose_features is not None:
            pose_tensor = torch.from_numpy(pose_features).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor, pose_tensor)
        
        # Postprocess
        detections, crowd_density, behavior_embeddings = self.postprocess(
            outputs, original_shape
        )
        
        result = {
            "detections": [det.to_dict() for det in detections],
            "num_detections": len(detections),
            "timestamp": None  # Will be set by caller
        }
        
        if crowd_density is not None:
            result["crowd_density"] = {
                "map": crowd_density.tolist(),
                "mean_density": float(np.mean(crowd_density))
            }
        
        if behavior_embeddings is not None:
            result["behavior_embeddings"] = behavior_embeddings.tolist()
        
        return result
    
    def reset_temporal_buffer(self):
        """Reset temporal buffer."""
        if self.model is not None:
            self.model.reset_temporal_buffer()
