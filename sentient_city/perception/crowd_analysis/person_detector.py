"""
Enhanced Person Detector for Crowd Risk Analysis.

Optimized YOLO-based person detection with confidence filtering
and performance optimizations for dense crowd scenarios.
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import torch
from ultralytics import YOLO

from sentient_city.core import get_logger, get_config


@dataclass
class Detection:
    """Single person detection result."""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int = 0  # Person class
    class_name: str = "person"
    center: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.center is None:
            self.center = np.array([
                (self.bbox[0] + self.bbox[2]) / 2,
                (self.bbox[1] + self.bbox[3]) / 2
            ])


class PersonDetector:
    """
    Enhanced person detector optimized for crowd analysis.
    
    Features:
    - YOLOv8-based person-only detection
    - Confidence filtering with adaptive thresholds
    - GPU optimization for real-time processing
    - Non-maximum suppression optimization
    """
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        device: Optional[str] = None,
        max_detections: int = 500
    ):
        """
        Initialize person detector.
        
        Args:
            model_path: Path to YOLO model
            confidence_threshold: Minimum confidence for detections
            nms_threshold: Non-maximum suppression threshold
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
            max_detections: Maximum number of detections to process
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        
        self.logger = get_logger(__name__)
        
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                try:
                    # Test if CUDA is actually working
                    torch.cuda.init()
                    self.device = 'cuda'
                except:
                    self.device = 'cpu'
            else:
                self.device = 'cpu'
        else:
            self.device = device
            
        self.logger.info(f"Initializing PersonDetector on device: {self.device}")
        
        # Load YOLO model
        self.model = self._load_model()
        
        # Performance metrics
        self.frame_count = 0
        self.total_detections = 0
        self.processing_times = []
    
    def _load_model(self) -> YOLO:
        """Load YOLO model with optimizations."""
        try:
            model = YOLO(self.model_path)
            
            # Optimize for inference
            if self.device == 'cuda':
                try:
                    model.to(self.device)
                    # Enable half-precision for faster inference
                    if hasattr(model.model, 'half'):
                        model.model.half()
                except Exception as e:
                    self.logger.warning(f"Failed to move model to CUDA: {e}, using CPU")
                    self.device = 'cpu'
            
            self.logger.info(f"Successfully loaded YOLO model: {self.model_path}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def detect(
        self,
        frame: np.ndarray,
        frame_id: Optional[int] = None
    ) -> List[Detection]:
        """
        Detect persons in frame.
        
        Args:
            frame: Input frame (BGR format)
            frame_id: Optional frame ID for tracking
            
        Returns:
            List of person detections
        """
        if frame_id is None:
            frame_id = self.frame_count
            
        self.frame_count += 1
        
        try:
            # Convert BGR to RGB for YOLO
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run inference with optimizations
            with torch.no_grad():
                results = self.model(
                    frame_rgb,
                    conf=self.confidence_threshold,
                    iou=self.nms_threshold,
                    max_det=self.max_detections,
                    classes=[0],  # Person class only
                    verbose=False,
                    device=self.device
                )
            
            # Extract detections
            detections = self._extract_detections(results[0])
            
            # Update metrics
            self.total_detections += len(detections)
            
            self.logger.debug(f"Frame {frame_id}: Detected {len(detections)} persons")
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Error in person detection: {e}")
            return []
    
    def _extract_detections(self, result) -> List[Detection]:
        """Extract and filter person detections from YOLO result."""
        detections = []
        
        if result.boxes is not None:
            boxes = result.boxes
            
            # Convert to numpy for faster processing
            if hasattr(boxes, 'xyxy'):
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy()
                
                for i in range(len(xyxy)):
                    if cls[i] == 0:  # Person class
                        bbox = xyxy[i]
                        confidence = float(conf[i])
                        
                        # Additional confidence filtering
                        if confidence >= self.confidence_threshold:
                            detection = Detection(
                                bbox=bbox,
                                confidence=confidence,
                                class_id=int(cls[i])
                            )
                            detections.append(detection)
        
        return detections
    
    def detect_batch(
        self,
        frames: List[np.ndarray],
        batch_size: int = 8
    ) -> List[List[Detection]]:
        """
        Detect persons in batch of frames for better GPU utilization.
        
        Args:
            frames: List of input frames
            batch_size: Batch size for processing
            
        Returns:
            List of detection lists for each frame
        """
        all_detections = []
        
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            batch_detections = []
            
            for frame in batch:
                detections = self.detect(frame)
                batch_detections.append(detections)
            
            all_detections.extend(batch_detections)
        
        return all_detections
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics."""
        avg_detections = self.total_detections / max(self.frame_count, 1)
        
        return {
            "frame_count": self.frame_count,
            "total_detections": self.total_detections,
            "avg_detections_per_frame": avg_detections,
            "device": self.device,
            "model_path": self.model_path,
            "confidence_threshold": self.confidence_threshold
        }
    
    def reset_metrics(self):
        """Reset performance metrics."""
        self.frame_count = 0
        self.total_detections = 0
        self.processing_times = []
    
    def update_confidence_threshold(self, threshold: float):
        """Update confidence threshold dynamically."""
        self.confidence_threshold = max(0.1, min(1.0, threshold))
        self.logger.info(f"Updated confidence threshold to {self.confidence_threshold}")
    
    def enable_adaptive_threshold(self, crowd_density: float):
        """
        Enable adaptive confidence threshold based on crowd density.
        
        Args:
            crowd_density: Current crowd density (people per frame)
        """
        if crowd_density > 50:  # Very dense crowd
            new_threshold = 0.3
        elif crowd_density > 20:  # Dense crowd
            new_threshold = 0.4
        else:  # Normal crowd
            new_threshold = 0.5
            
        self.update_confidence_threshold(new_threshold)
