"""
Main Crowd Analysis System - Orchestrates all components.

Comprehensive crowd risk detection system with person detection,
tracking, motion analysis, risk evaluation, alert management,
and visualization capabilities.
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime
import time
from loguru import logger

from .person_detector import PersonDetector, Detection
from .person_tracker import ByteTrackTracker, TrackedPerson
from .motion_analyzer import MotionAnalyzer, CrowdMotionMetrics, MotionFeatures
from .risk_evaluator import RiskEvaluator, CrowdRisk, PersonRisk
from .alert_manager import AlertManager, Alert, AlertConfig
from .visualizer import CrowdVisualizer, VisualizationConfig
from sentient_city.core import get_logger, get_config


@dataclass
class SystemConfig:
    """Main system configuration."""
    
    # Detection settings
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.5
    max_detections: int = 500
    
    # Tracking settings
    track_buffer: int = 30
    confirmation_threshold: int = 3
    
    # Risk evaluation settings
    enable_adaptive_thresholds: bool = True
    risk_history_length: int = 30
    
    # Alert settings
    enable_alerts: bool = True
    alert_cooldown_seconds: int = 30
    
    # Visualization settings
    enable_visualization: bool = True
    debug_mode: bool = False
    
    # Performance settings
    target_fps: int = 30
    enable_gpu: bool = True


class CrowdAnalysisSystem:
    """
    Main crowd analysis system orchestrating all components.
    
    Features:
    - Real-time person detection and tracking
    - Advanced motion analysis and risk evaluation
    - Dynamic alert management
    - Comprehensive visualization
    - Performance optimization
    - Debugging capabilities
    """
    
    def __init__(
        self,
        config: Optional[SystemConfig] = None,
        alert_callbacks: Optional[List[Callable]] = None
    ):
        """
        Initialize crowd analysis system.
        
        Args:
            config: System configuration
            alert_callbacks: List of alert callback functions
        """
        self.config = config or SystemConfig()
        self.alert_callbacks = alert_callbacks or []
        
        self.logger = get_logger(__name__)
        
        # Initialize components
        self._initialize_components()
        
        # System state
        self.frame_count = 0
        self.is_running = False
        self.start_time = None
        
        # Performance metrics
        self.processing_times: List[float] = []
        self.last_frame_time = 0
        
        self.logger.info("Crowd Analysis System initialized successfully")
    
    def _initialize_components(self):
        """Initialize all system components."""
        # Person Detector
        self.person_detector = PersonDetector(
            model_path=self.config.model_path,
            confidence_threshold=self.config.confidence_threshold,
            max_detections=self.config.max_detections,
            device='cuda' if self.config.enable_gpu else 'cpu'
        )
        
        # Person Tracker
        self.person_tracker = ByteTrackTracker(
            track_buffer=self.config.track_buffer,
            confirmation_threshold=self.config.confirmation_threshold
        )
        
        # Motion Analyzer
        self.motion_analyzer = MotionAnalyzer()
        
        # Risk Evaluator
        self.risk_evaluator = RiskEvaluator(
            risk_history_length=self.config.risk_history_length
        )
        
        # Alert Manager
        if self.config.enable_alerts:
            alert_config = AlertConfig(
                cooldown_seconds=self.config.alert_cooldown_seconds
            )
            self.alert_manager = AlertManager(
                config=alert_config,
                alert_callbacks=self.alert_callbacks
            )
        else:
            self.alert_manager = None
        
        # Visualizer
        if self.config.enable_visualization:
            viz_config = VisualizationConfig(
                show_debug_info=self.config.debug_mode
            )
            self.visualizer = CrowdVisualizer(
                config=viz_config,
                debug_mode=self.config.debug_mode
            )
        else:
            self.visualizer = None
    
    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: Optional[int] = None,
        timestamp: Optional[datetime] = None
    ) -> Dict:
        """
        Process a single frame through the complete pipeline.
        
        Args:
            frame: Input frame (BGR format)
            frame_id: Optional frame ID
            timestamp: Optional frame timestamp
            
        Returns:
            Dictionary with all processing results
        """
        if frame_id is None:
            frame_id = self.frame_count
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Start timing
        start_time = time.time()
        
        try:
            # 1. Person Detection
            detections = self.person_detector.detect(frame, frame_id)
            
            # 2. Person Tracking
            tracked_persons = self.person_tracker.update(detections, frame_id)
            
            # 3. Motion Analysis
            crowd_metrics, person_features = self.motion_analyzer.analyze_frame_motion(
                frame, tracked_persons
            )
            
            # 4. Risk Evaluation
            crowd_risk, person_risks = self.risk_evaluator.evaluate_risk(
                tracked_persons, crowd_metrics, person_features, frame_id
            )
            
            # 5. Alert Management
            new_alerts = []
            if self.alert_manager:
                new_alerts = self.alert_manager.process_frame_alerts(
                    crowd_risk, person_risks, frame_id, timestamp
                )
            
            # 6. Visualization
            visualized_frame = None
            if self.visualizer:
                motion_heatmap = None
                if hasattr(self.motion_analyzer, 'flow_history') and self.motion_analyzer.flow_history:
                    latest_flow = self.motion_analyzer.flow_history[-1]
                    motion_heatmap = self.motion_analyzer.generate_motion_heatmap(
                        latest_flow, frame.shape
                    )
                
                visualized_frame = self.visualizer.visualize_frame(
                    frame, tracked_persons, crowd_metrics, person_risks,
                    crowd_risk, new_alerts, motion_heatmap
                )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Prepare results
            results = {
                'frame_id': frame_id,
                'timestamp': timestamp,
                'detections': detections,
                'tracked_persons': tracked_persons,
                'crowd_metrics': crowd_metrics,
                'person_features': person_features,
                'person_risks': person_risks,
                'crowd_risk': crowd_risk,
                'new_alerts': new_alerts,
                'visualized_frame': visualized_frame,
                'processing_time': processing_time,
                'fps': self._calculate_fps()
            }
            
            self.frame_count += 1
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing frame {frame_id}: {e}")
            return {
                'frame_id': frame_id,
                'timestamp': timestamp,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def process_video_stream(
        self,
        video_source: str,
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        show_live: bool = True
    ) -> Dict:
        """
        Process video stream through the complete pipeline.
        
        Args:
            video_source: Video file path or camera index
            output_path: Optional output video path
            max_frames: Maximum number of frames to process
            show_live: Whether to show live visualization
            
        Returns:
            Processing statistics and results
        """
        # Open video source
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {video_source}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Processing loop
        self.is_running = True
        self.start_time = time.time()
        
        processed_frames = 0
        total_alerts = 0
        processing_times = []
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if max_frames and processed_frames >= max_frames:
                    break
                
                # Process frame
                results = self.process_frame(frame)
                
                if 'error' not in results:
                    processed_frames += 1
                    processing_times.append(results['processing_time'])
                    total_alerts += len(results.get('new_alerts', []))
                    
                    # Write frame if writer available
                    if writer and results['visualized_frame'] is not None:
                        writer.write(results['visualized_frame'])
                    
                    # Show live visualization
                    if show_live and results['visualized_frame'] is not None:
                        cv2.imshow('Crowd Analysis', results['visualized_frame'])
                        
                        # Check for key presses
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord('d'):
                            self.toggle_debug_mode()
                        elif key == ord('s'):
                            self.save_current_frame(results['visualized_frame'])
                
                # Print progress
                if processed_frames % 100 == 0:
                    avg_fps = processed_frames / (time.time() - self.start_time)
                    self.logger.info(f"Processed {processed_frames} frames, Avg FPS: {avg_fps:.1f}")
        
        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            if show_live:
                cv2.destroyAllWindows()
            
            self.is_running = False
        
        # Calculate statistics
        total_time = time.time() - self.start_time
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        avg_fps = processed_frames / total_time if total_time > 0 else 0
        
        statistics = {
            'total_frames': processed_frames,
            'total_time': total_time,
            'avg_processing_time': avg_processing_time,
            'avg_fps': avg_fps,
            'total_alerts': total_alerts,
            'video_properties': {
                'fps': fps,
                'width': width,
                'height': height,
                'total_frames': total_frames
            }
        }
        
        self.logger.info(f"Video processing complete: {statistics}")
        
        return statistics
    
    def get_system_performance(self) -> Dict:
        """Get comprehensive system performance metrics."""
        # Component metrics
        detector_metrics = self.person_detector.get_performance_metrics()
        tracker_metrics = self.person_tracker.get_performance_metrics()
        risk_trends = self.risk_evaluator.get_risk_trends()
        
        # Alert statistics
        alert_stats = {}
        if self.alert_manager:
            alert_stats = self.alert_manager.get_alert_statistics()
        
        # System metrics
        avg_processing_time = np.mean(self.processing_times[-100:]) if self.processing_times else 0
        current_fps = self._calculate_fps()
        
        return {
            'system': {
                'frame_count': self.frame_count,
                'is_running': self.is_running,
                'avg_processing_time': avg_processing_time,
                'current_fps': current_fps,
                'uptime': time.time() - self.start_time if self.start_time else 0
            },
            'detector': detector_metrics,
            'tracker': tracker_metrics,
            'risk_trends': risk_trends,
            'alerts': alert_stats
        }
    
    def _calculate_fps(self) -> float:
        """Calculate current FPS."""
        if not self.processing_times:
            return 0.0
        
        recent_times = self.processing_times[-30:]  # Last 30 frames
        avg_time = np.mean(recent_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def toggle_debug_mode(self):
        """Toggle debug mode for all components."""
        self.config.debug_mode = not self.config.debug_mode
        
        if self.visualizer:
            self.visualizer.toggle_debug_mode()
        
        self.logger.info(f"Debug mode: {'ON' if self.config.debug_mode else 'OFF'}")
    
    def save_current_frame(self, frame: np.ndarray, filename: Optional[str] = None):
        """Save current frame with timestamp."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"crowd_analysis_frame_{timestamp}.jpg"
        
        cv2.imwrite(filename, frame)
        self.logger.info(f"Frame saved: {filename}")
    
    def update_confidence_threshold(self, threshold: float):
        """Update detection confidence threshold dynamically."""
        self.person_detector.update_confidence_threshold(threshold)
        self.config.confidence_threshold = threshold
        self.logger.info(f"Updated confidence threshold to {threshold}")
    
    def enable_adaptive_detection(self, crowd_density: float):
        """Enable adaptive detection based on crowd density."""
        self.person_detector.enable_adaptive_threshold(crowd_density)
        self.logger.info(f"Enabled adaptive detection for crowd density: {crowd_density}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get currently active alerts."""
        return self.alert_manager.get_active_alerts() if self.alert_manager else []
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an active alert."""
        return self.alert_manager.acknowledge_alert(alert_id) if self.alert_manager else False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Manually resolve an alert."""
        return self.alert_manager.resolve_alert(alert_id) if self.alert_manager else False
    
    def stop_processing(self):
        """Stop video processing."""
        self.is_running = False
        self.logger.info("Processing stopped")
    
    def reset(self):
        """Reset all system components."""
        self.frame_count = 0
        self.processing_times.clear()
        self.last_frame_time = 0
        self.start_time = None
        
        # Reset components
        self.person_tracker.reset()
        self.risk_evaluator.reset()
        if self.alert_manager:
            self.alert_manager.reset()
        if self.visualizer:
            self.visualizer.reset()
        
        self.logger.info("System reset complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_processing()
