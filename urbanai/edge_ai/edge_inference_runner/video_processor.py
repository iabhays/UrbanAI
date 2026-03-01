"""
Video processing pipeline for real-time inference.

Handles video input, frame processing, and result streaming.
"""

import cv2
import numpy as np
from typing import Optional, Callable, Dict, Any, Iterator
from pathlib import Path
import asyncio
from loguru import logger

from .detector import EdgeDetector


class VideoProcessor:
    """
    Real-time video processor.
    
    Handles video input from various sources and processes frames
    through the detection pipeline.
    """
    
    def __init__(
        self,
        detector: EdgeDetector,
        source: Optional[str] = None,
        fps: Optional[int] = None,
        frame_skip: int = 1
    ):
        """
        Initialize video processor.
        
        Args:
            detector: EdgeDetector instance
            source: Video source (RTSP URL, file path, or camera index)
            fps: Target FPS (None for source FPS)
            frame_skip: Process every Nth frame
        """
        self.detector = detector
        self.source = source
        self.fps = fps
        self.frame_skip = frame_skip
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.frame_count = 0
    
    def open(self, source: Optional[str] = None) -> bool:
        """
        Open video source.
        
        Args:
            source: Video source (overrides instance source)
        
        Returns:
            True if opened successfully
        """
        source = source or self.source
        if source is None:
            logger.error("No video source specified")
            return False
        
        try:
            # Try as file path first
            if Path(source).exists():
                self.cap = cv2.VideoCapture(str(source))
            # Try as camera index
            elif source.isdigit():
                self.cap = cv2.VideoCapture(int(source))
            # Assume RTSP/network stream
            else:
                self.cap = cv2.VideoCapture(source)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open video source: {source}")
                return False
            
            logger.info(f"Video source opened: {source}")
            return True
        except Exception as e:
            logger.error(f"Error opening video source: {e}")
            return False
    
    def close(self):
        """Close video source."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_running = False
        logger.info("Video source closed")
    
    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read next frame from video source.
        
        Returns:
            Frame as numpy array (BGR) or None if failed
        """
        if self.cap is None or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        return frame
    
    def process_frame(
        self,
        frame: np.ndarray,
        pose_features: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Process single frame.
        
        Args:
            frame: Input frame (BGR)
            pose_features: Optional pose features
        
        Returns:
            Detection results
        """
        import time
        timestamp = time.time()
        
        result = self.detector.detect(frame, pose_features)
        result["timestamp"] = timestamp
        result["frame_id"] = self.frame_count
        
        return result
    
    def process_stream(
        self,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Process video stream.
        
        Args:
            callback: Optional callback function for each result
        
        Yields:
            Detection results for each processed frame
        """
        if self.cap is None:
            logger.error("Video source not opened")
            return
        
        self.is_running = True
        self.frame_count = 0
        
        while self.is_running:
            frame = self.read_frame()
            if frame is None:
                logger.warning("Failed to read frame, stopping")
                break
            
            # Skip frames if needed
            if self.frame_count % self.frame_skip != 0:
                self.frame_count += 1
                continue
            
            # Process frame
            try:
                result = self.process_frame(frame)
                result["source"] = self.source
                
                if callback:
                    callback(result)
                
                yield result
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
            
            self.frame_count += 1
    
    async def process_stream_async(
        self,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        Process video stream asynchronously.
        
        Args:
            callback: Optional async callback function
        """
        loop = asyncio.get_event_loop()
        
        def sync_callback(result):
            if callback:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(result))
                else:
                    callback(result)
        
        for result in self.process_stream(sync_callback):
            await asyncio.sleep(0)  # Yield control
    
    def stop(self):
        """Stop processing."""
        self.is_running = False
    
    def get_properties(self) -> Dict[str, Any]:
        """
        Get video properties.
        
        Returns:
            Dictionary with video properties
        """
        if self.cap is None:
            return {}
        
        return {
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_count": int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "current_frame": self.frame_count
        }
