"""
Privacy masking module.

Provides privacy-preserving data masking for faces, license plates, and PII.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from loguru import logger


@dataclass
class MaskingConfig:
    """Privacy masking configuration."""
    blur_faces: bool = True
    blur_license_plates: bool = True
    blur_region: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2
    blur_kernel_size: int = 51
    blur_sigma: float = 0


class PrivacyMasking:
    """
    Privacy masking service.
    
    Applies privacy-preserving masking to images.
    """
    
    def __init__(self, config: Optional[MaskingConfig] = None):
        """
        Initialize privacy masking.
        
        Args:
            config: Masking configuration
        """
        self.config = config or MaskingConfig()
        self.face_cascade = None
        self._load_face_detector()
    
    def _load_face_detector(self):
        """Load face detection cascade."""
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                logger.warning("Failed to load face cascade")
        except Exception as e:
            logger.warning(f"Could not load face detector: {e}")
    
    def mask_image(
        self,
        image: np.ndarray,
        detections: Optional[List[Dict]] = None
    ) -> np.ndarray:
        """
        Apply privacy masking to image.
        
        Args:
            image: Input image (BGR)
            detections: Optional detection results with face/plate regions
        
        Returns:
            Masked image
        """
        masked_image = image.copy()
        
        # Mask faces
        if self.config.blur_faces:
            masked_image = self._mask_faces(masked_image, detections)
        
        # Mask license plates
        if self.config.blur_license_plates:
            masked_image = self._mask_license_plates(masked_image, detections)
        
        # Mask custom region
        if self.config.blur_region:
            masked_image = self._mask_region(masked_image, self.config.blur_region)
        
        return masked_image
    
    def _mask_faces(
        self,
        image: np.ndarray,
        detections: Optional[List[Dict]] = None
    ) -> np.ndarray:
        """Mask faces in image."""
        if detections:
            # Use provided detections
            for det in detections:
                if det.get("class_name") == "person":
                    bbox = det.get("bbox", [])
                    if len(bbox) >= 4:
                        x1, y1, x2, y2 = map(int, bbox[:4])
                        image = self._blur_region(
                            image,
                            (x1, y1, x2, y2)
                        )
        else:
            # Use face detector
            if self.face_cascade is not None:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                for (x, y, w, h) in faces:
                    image = self._blur_region(image, (x, y, x + w, y + h))
        
        return image
    
    def _mask_license_plates(
        self,
        image: np.ndarray,
        detections: Optional[List[Dict]] = None
    ) -> np.ndarray:
        """Mask license plates in image."""
        if detections:
            for det in detections:
                if det.get("class_name") in ["car", "truck", "bus", "motorcycle"]:
                    bbox = det.get("bbox", [])
                    if len(bbox) >= 4:
                        x1, y1, x2, y2 = map(int, bbox[:4])
                        # Assume license plate in lower portion of vehicle
                        plate_y1 = y1 + int((y2 - y1) * 0.6)
                        image = self._blur_region(
                            image,
                            (x1, plate_y1, x2, y2)
                        )
        
        return image
    
    def _mask_region(
        self,
        image: np.ndarray,
        region: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Mask specific region."""
        return self._blur_region(image, region)
    
    def _blur_region(
        self,
        image: np.ndarray,
        region: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Blur specific region.
        
        Args:
            image: Input image
            region: (x1, y1, x2, y2) bounding box
        
        Returns:
            Image with blurred region
        """
        x1, y1, x2, y2 = region
        
        # Clamp to image bounds
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        if x2 <= x1 or y2 <= y1:
            return image
        
        # Extract region
        roi = image[y1:y2, x1:x2]
        
        # Apply Gaussian blur
        blurred_roi = cv2.GaussianBlur(
            roi,
            (self.config.blur_kernel_size, self.config.blur_kernel_size),
            self.config.blur_sigma
        )
        
        # Replace region
        image[y1:y2, x1:x2] = blurred_roi
        
        return image
    
    def detect_pii(self, image: np.ndarray) -> List[Dict]:
        """
        Detect PII in image (placeholder for OCR/text detection).
        
        Args:
            image: Input image
        
        Returns:
            List of PII regions
        """
        # Placeholder - in production, use OCR/text detection
        # to identify text regions that might contain PII
        return []
