"""
Real-time Camera Processing System with Skeleton Detection and Crowd Analysis.

This module provides a complete real-time processing system that integrates
person detection, skeleton wireframe tracking, crowd density mapping, and
comprehensive visualization for live camera feeds and video files.
"""

import cv2
import numpy as np
import time
import threading
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime
from queue import Queue, Empty

from .enhanced_person_detector import EnhancedPersonDetector, SkeletonDetection
from .crowd_density_mapper import CrowdDensityMapper, DensityMetrics, HeatmapConfig


@dataclass
class ProcessingConfig:
    """Configuration for real-time processing."""
    # Camera settings
    camera_id: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    
    # Processing settings
    enable_skeleton: bool = True
    enable_heatmap: bool = True
    enable_density_zones: bool = True
    enable_flow_analysis: bool = True
    
    # Detection thresholds
    person_confidence_threshold: float = 0.5
    pose_confidence_threshold: float = 0.5
    
    # Performance settings
    max_processing_threads: int = 2
    frame_queue_size: int = 10
    processing_timeout: float = 0.1
    
    # Visualization settings
    show_fps: bool = True
    show_metrics: bool = True
    save_output: bool = False
    output_path: str = "crowd_analysis_output.mp4"


@dataclass
class ProcessingResults:
    """Results from frame processing."""
    frame_id: int
    timestamp: datetime
    detections: List[SkeletonDetection]
    density_metrics: DensityMetrics
    processing_time: float
    fps: float
    annotated_frame: Optional[np.ndarray] = None
    heatmap_frame: Optional[np.ndarray] = None


class RealTimeProcessor:
    """
    Real-time camera processing system with skeleton detection and crowd analysis.
    
    Features:
    - Multi-threaded processing for real-time performance
    - Live camera feed and video file support
    - Skeleton wireframe detection and tracking
    - Crowd density mapping and heatmap generation
    - Real-time visualization and metrics
    - Configurable processing pipelines
    - Performance monitoring and optimization
    """
    
    def __init__(
        self,
        config: Optional[ProcessingConfig] = None,
        result_callbacks: Optional[List[Callable]] = None
    ):
        """
        Initialize real-time processor.
        
        Args:
            config: Processing configuration
            result_callbacks: List of callback functions for processing results
        """
        self.config = config or ProcessingConfig()
        self.result_callbacks = result_callbacks or []
        
        # Initialize components
        self._initialize_components()
        
        # Processing state
        self.is_running = False
        self.is_paused = False
        self.current_frame_id = 0
        
        # Threading
        self.frame_queue = Queue(maxsize=self.config.frame_queue_size)
        self.result_queue = Queue()
        self.processing_threads = []
        
        # Performance metrics
        self.processing_times = []
        self.last_frame_time = time.time()
        self.fps_history = []
        
        # Video capture
        self.cap = None
        self.video_writer = None
        
        print("RealTimeProcessor initialized successfully")
    
    def _initialize_components(self):
        """Initialize all processing components."""
        # Enhanced person detector with skeleton support
        self.person_detector = EnhancedPersonDetector(
            confidence_threshold=self.config.person_confidence_threshold,
            pose_confidence_threshold=self.config.pose_confidence_threshold,
            enable_skeleton=self.config.enable_skeleton,
            max_detections=500
        )
        
        # Initialize crowd density mapper
        heatmap_config = HeatmapConfig(
            grid_size=(20, 20),
            sigma=2.0,
            alpha=0.6,
            normalize=True
        )
        self.density_mapper = CrowdDensityMapper(
            frame_size=(self.config.camera_width, self.config.camera_height),
            config=heatmap_config
        )
        
        print("Processing components initialized")
    
    def start_camera(self, camera_id: Optional[int] = None) -> bool:
        """
        Start camera capture.
        
        Args:
            camera_id: Camera ID (uses config if None)
            
        Returns:
            True if camera started successfully
        """
        if camera_id is None:
            camera_id = self.config.camera_id
        
        try:
            self.cap = cv2.VideoCapture(camera_id)
            
            if not self.cap.isOpened():
                print(f"Failed to open camera {camera_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.camera_fps)
            
            # Get actual camera properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"Camera started: {actual_width}x{actual_height} @ {actual_fps} FPS")
            
            return True
            
        except Exception as e:
            print(f"Error starting camera: {e}")
            return False
    
    def start_video_file(self, video_path: str) -> bool:
        """
        Start processing video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            True if video opened successfully
        """
        try:
            self.cap = cv2.VideoCapture(video_path)
            
            if not self.cap.isOpened():
                print(f"Failed to open video file: {video_path}")
                return False
            
            # Get video properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Video opened: {width}x{height} @ {fps} FPS, {total_frames} frames")
            
            return True
            
        except Exception as e:
            print(f"Error opening video file: {e}")
            return False
    
    def start_processing(self) -> bool:
        """
        Start real-time processing.
        
        Returns:
            True if processing started successfully
        """
        if self.cap is None or not self.cap.isOpened():
            print("No camera or video source available")
            return False
        
        self.is_running = True
        self.is_paused = False
        
        # Initialize video writer if saving output
        if self.config.save_output:
            self._initialize_video_writer()
        
        # Start processing threads
        self._start_processing_threads()
        
        # Start frame capture thread
        capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        capture_thread.start()
        
        print("Real-time processing started")
        return True
    
    def stop_processing(self):
        """Stop real-time processing."""
        self.is_running = False
        
        # Wait for threads to finish
        for thread in self.processing_threads:
            thread.join(timeout=1.0)
        
        # Release resources
        if self.cap:
            self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        
        print("Real-time processing stopped")
    
    def _initialize_video_writer(self):
        """Initialize video writer for output saving."""
        if self.cap is None:
            return
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.video_writer = cv2.VideoWriter(
            self.config.output_path,
            fourcc,
            fps,
            (width, height)
        )
        
        print(f"Video writer initialized: {self.config.output_path}")
    
    def _start_processing_threads(self):
        """Start worker threads for frame processing."""
        for i in range(self.config.max_processing_threads):
            thread = threading.Thread(
                target=self._process_frames,
                args=(i,),
                daemon=True
            )
            thread.start()
            self.processing_threads.append(thread)
        
        print(f"Started {len(self.processing_threads)} processing threads")
    
    def _capture_frames(self):
        """Capture frames from camera/video and add to queue."""
        while self.is_running:
            if self.is_paused:
                time.sleep(0.1)
                continue
            
            ret, frame = self.cap.read()
            if not ret:
                print("No more frames available")
                break
            
            # Add frame to processing queue
            try:
                self.frame_queue.put((self.current_frame_id, frame), timeout=0.1)
                self.current_frame_id += 1
            except:
                # Queue is full, skip frame
                continue
            
            # Control frame rate
            time.sleep(1.0 / self.config.camera_fps)
    
    def _process_frames(self, thread_id: int):
        """Process frames from queue."""
        while self.is_running:
            try:
                # Get frame from queue
                frame_id, frame = self.frame_queue.get(timeout=self.config.processing_timeout)
                
                # Process frame
                start_time = time.time()
                results = self._process_single_frame(frame, frame_id)
                processing_time = time.time() - start_time
                
                # Update results with timing info
                results.processing_time = processing_time
                results.fps = self._calculate_fps()
                
                # Add to result queue
                self.result_queue.put(results)
                
                # Call result callbacks
                for callback in self.result_callbacks:
                    try:
                        callback(results)
                    except Exception as e:
                        print(f"Error in result callback: {e}")
                
                # Save frame if configured
                if self.config.save_output and self.video_writer and results.annotated_frame is not None:
                    self.video_writer.write(results.annotated_frame)
                
            except Empty:
                continue
            except Exception as e:
                print(f"Error processing frame: {e}")
    
    def _process_single_frame(self, frame: np.ndarray, frame_id: int) -> ProcessingResults:
        """
        Process a single frame with all analysis components.
        
        Args:
            frame: Input frame
            frame_id: Frame identifier
            
        Returns:
            ProcessingResults with all analysis data
        """
        timestamp = datetime.now()
        
        # Step 1: Detect persons with skeletons
        detections = self.person_detector.detect(frame, frame_id)
        
        # Step 2: Calculate crowd density metrics
        density_metrics = self.density_mapper.calculate_density(detections, frame_id)
        
        # Step 3: Create visualizations
        annotated_frame = self._create_annotated_frame(frame, detections, density_metrics)
        heatmap_frame = None
        
        if self.config.enable_heatmap:
            heatmap = self.density_mapper.generate_heatmap()
            heatmap_frame = self.density_mapper.overlay_heatmap(frame, heatmap)
        
        return ProcessingResults(
            frame_id=frame_id,
            timestamp=timestamp,
            detections=detections,
            density_metrics=density_metrics,
            processing_time=0.0,  # Will be set by caller
            fps=0.0,  # Will be set by caller
            annotated_frame=annotated_frame,
            heatmap_frame=heatmap_frame
        )
    
    def _create_annotated_frame(
        self,
        frame: np.ndarray,
        detections: List[SkeletonDetection],
        density_metrics: DensityMetrics
    ) -> np.ndarray:
        """
        Create annotated frame with detections and metrics.
        
        Args:
            frame: Original frame
            detections: Person detections with skeletons
            density_metrics: Crowd density analysis
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Draw skeleton detections
        for detection in detections:
            if self.config.enable_skeleton and len(detection.keypoints) > 0:
                annotated = self.person_detector.draw_skeleton(annotated, detection)
            else:
                # Draw simple bounding box
                x1, y1, x2, y2 = map(int, detection.bbox)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add confidence score
                conf_text = f"Person: {detection.confidence:.2f}"
                cv2.putText(annotated, conf_text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw density zones if enabled
        if self.config.enable_density_zones:
            annotated = self.density_mapper.draw_density_zones(
                annotated, density_metrics.density_zones
            )
        
        # Add metrics overlay if enabled
        if self.config.show_metrics:
            annotated = self._add_metrics_overlay(annotated, density_metrics)
        
        return annotated
    
    def _add_metrics_overlay(self, frame: np.ndarray, metrics: DensityMetrics) -> np.ndarray:
        """Add metrics overlay to frame."""
        overlay = frame.copy()
        
        # Create semi-transparent background for text
        h, w = frame.shape[:2]
        cv2.rectangle(overlay, (10, 10), (350, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add metrics text
        y_offset = 30
        metrics_text = [
            f"People Count: {metrics.crowd_count}",
            f"Overall Density: {metrics.overall_density:.2f}/m²",
            f"Peak Density: {metrics.peak_density:.2f}",
            f"Area Coverage: {metrics.area_coverage:.1%}",
        ]
        
        for i, text in enumerate(metrics_text):
            cv2.putText(frame, text, (20, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # Add FPS if enabled
        if self.config.show_fps:
            current_fps = self._calculate_fps()
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (w - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame
    
    def _calculate_fps(self) -> float:
        """Calculate current FPS."""
        current_time = time.time()
        if self.last_frame_time > 0:
            fps = 1.0 / (current_time - self.last_frame_time)
            self.fps_history.append(fps)
            if len(self.fps_history) > 30:  # Keep last 30 FPS values
                self.fps_history.pop(0)
        self.last_frame_time = current_time
        
        return np.mean(self.fps_history) if self.fps_history else 0.0
    
    def get_latest_results(self) -> Optional[ProcessingResults]:
        """Get the latest processing results."""
        try:
            return self.result_queue.get_nowait()
        except Empty:
            return None
    
    def pause_processing(self):
        """Pause processing."""
        self.is_paused = True
        print("Processing paused")
    
    def resume_processing(self):
        """Resume processing."""
        self.is_paused = False
        print("Processing resumed")
    
    def get_performance_metrics(self) -> Dict:
        """Get comprehensive performance metrics."""
        detector_metrics = self.person_detector.get_performance_metrics()
        density_stats = self.density_mapper.get_density_statistics()
        
        return {
            'processing': {
                'is_running': self.is_running,
                'is_paused': self.is_paused,
                'current_fps': self._calculate_fps(),
                'frames_processed': self.current_frame_id,
                'queue_size': self.frame_queue.qsize()
            },
            'detector': detector_metrics,
            'density_mapper': density_stats,
            'config': {
                'camera_width': self.config.camera_width,
                'camera_height': self.config.camera_height,
                'camera_fps': self.config.camera_fps,
                'enable_skeleton': self.config.enable_skeleton,
                'enable_heatmap': self.config.enable_heatmap
            }
        }
    
    def __del__(self):
        """Cleanup resources."""
        self.stop_processing()
