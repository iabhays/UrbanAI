"""
Enhanced Visualization System for Crowd Analysis.

This module provides comprehensive visualization capabilities including
bounding boxes, skeleton wireframes, heatmaps, and real-time metrics
display for crowd analysis systems.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
import time

from .enhanced_person_detector import SkeletonDetection
from .crowd_density_mapper import DensityMetrics, HeatmapConfig
from .real_time_processor import ProcessingResults


@dataclass
class VisualizationConfig:
    """Configuration for visualization system."""
    # Bounding box settings
    bbox_thickness: int = 2
    bbox_color: Tuple[int, int, int] = (0, 255, 0)  # Green
    bbox_confidence_threshold: float = 0.5
    
    # Skeleton settings
    skeleton_thickness: int = 2
    skeleton_color: Tuple[int, int, int] = (255, 0, 0)  # Red
    keypoint_radius: int = 3
    keypoint_color: Tuple[int, int, int] = (0, 0, 255)  # Blue
    keypoint_visibility_threshold: float = 0.5
    
    # Heatmap settings
    heatmap_alpha: float = 0.6
    heatmap_colormap: int = cv2.COLORMAP_JET
    show_grid: bool = False
    
    # Metrics display settings
    show_fps: bool = True
    show_metrics: bool = True
    show_timestamp: bool = True
    metrics_position: str = "top-left"  # top-left, top-right, bottom-left, bottom-right
    metrics_bg_alpha: float = 0.7
    metrics_text_color: Tuple[int, int, int] = (0, 255, 0)
    metrics_bg_color: Tuple[int, int, int] = (0, 0, 0)
    
    # Zone visualization
    show_zones: bool = True
    zone_colors: Dict[str, Tuple[int, int, int]] = None
    
    # Flow visualization
    show_flow: bool = True
    flow_arrow_color: Tuple[int, int, int] = (255, 255, 0)  # Yellow
    flow_arrow_thickness: int = 3
    
    # Alert visualization
    show_alerts: bool = True
    alert_color: Tuple[int, int, int] = (0, 0, 255)  # Red
    alert_flash_frequency: float = 2.0  # Hz


class EnhancedVisualizer:
    """
    Enhanced visualization system for crowd analysis.
    
    Features:
    - Multi-layered visualization with bounding boxes and skeletons
    - Real-time metrics and performance display
    - Interactive heatmap overlays
    - Zone-based density visualization
    - Flow direction indicators
    - Alert and warning displays
    - Customizable color schemes and layouts
    """
    
    # MediaPipe pose connections for skeleton drawing
    POSE_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 7),  # Head to body
        (0, 4), (4, 5), (5, 6), (6, 8),  # Head to body (other side)
        (9, 10),  # Shoulders
        (11, 12),  # Hips
        (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),  # Left arm
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),  # Right arm
        (11, 23), (12, 24),  # Torso
        (23, 24),  # Hips connection
        (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),  # Left leg
        (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),  # Right leg
    ]
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize enhanced visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        
        # Set default zone colors if not provided
        if self.config.zone_colors is None:
            self.config.zone_colors = {
                'low': (0, 255, 0),      # Green
                'medium': (0, 255, 255), # Yellow
                'high': (0, 165, 255),   # Orange
                'critical': (0, 0, 255)   # Red
            }
        
        # Alert flashing state
        self.alert_flash_time = 0
        
        print("EnhancedVisualizer initialized")
    
    def create_comprehensive_visualization(
        self,
        frame: np.ndarray,
        results: ProcessingResults,
        heatmap: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Create comprehensive visualization with all elements.
        
        Args:
            frame: Original frame
            results: Processing results with detections and metrics
            heatmap: Optional heatmap overlay
            
        Returns:
            Comprehensively annotated frame
        """
        vis_frame = frame.copy()
        
        # Step 1: Add heatmap overlay if available
        if heatmap is not None:
            vis_frame = self._overlay_heatmap(vis_frame, heatmap)
        
        # Step 2: Draw density zones
        if self.config.show_zones:
            vis_frame = self._draw_density_zones(vis_frame, results.density_metrics)
        
        # Step 3: Draw detections with skeletons
        vis_frame = self._draw_detections_with_skeletons(vis_frame, results.detections)
        
        # Step 4: Draw flow indicators
        if self.config.show_flow and results.density_metrics.flow_direction is not None:
            vis_frame = self._draw_flow_indicators(vis_frame, results.density_metrics)
        
        # Step 5: Add metrics overlay
        if self.config.show_metrics:
            vis_frame = self._add_metrics_overlay(vis_frame, results)
        
        # Step 6: Add timestamp
        if self.config.show_timestamp:
            vis_frame = self._add_timestamp(vis_frame, results.timestamp)
        
        # Step 7: Add alerts if any
        if self.config.show_alerts:
            vis_frame = self._add_alerts(vis_frame, results)
        
        return vis_frame
    
    def _draw_detections_with_skeletons(
        self,
        frame: np.ndarray,
        detections: List[SkeletonDetection]
    ) -> np.ndarray:
        """
        Draw detections with bounding boxes and skeleton wireframes.
        
        Args:
            frame: Input frame
            detections: List of skeleton detections
            
        Returns:
            Frame with drawn detections and skeletons
        """
        annotated = frame.copy()
        
        for detection in detections:
            # Only draw if confidence meets threshold
            if detection.confidence < self.config.bbox_confidence_threshold:
                continue
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, detection.bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), 
                          self.config.bbox_color, self.config.bbox_thickness)
            
            # Draw confidence score and ID
            conf_text = f"Person: {detection.confidence:.2f}"
            cv2.putText(annotated, conf_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.config.bbox_color, 1)
            
            # Draw skeleton if available
            if len(detection.keypoints) > 0:
                annotated = self._draw_skeleton(annotated, detection.keypoints, detection)
            else:
                # Draw simple person indicator
                center = map(int, detection.center)
                cv2.circle(annotated, tuple(center), 5, self.config.keypoint_color, -1)
        
        return annotated
    
    def _draw_skeleton(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        detection: SkeletonDetection
    ) -> np.ndarray:
        """
        Draw skeleton wireframe on frame.
        
        Args:
            frame: Input frame
            keypoints: MediaPipe pose landmarks
            detection: Detection object for additional info
            
        Returns:
            Frame with drawn skeleton
        """
        annotated = frame.copy()
        
        # Draw skeleton connections
        for connection in self.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            
            if (start_idx < len(keypoints) and end_idx < len(keypoints)):
                start_point = keypoints[start_idx]
                end_point = keypoints[end_idx]
                
                # Check visibility
                if (start_point[2] > self.config.keypoint_visibility_threshold and
                    end_point[2] > self.config.keypoint_visibility_threshold):
                    
                    start_coords = (int(start_point[0]), int(start_point[1]))
                    end_coords = (int(end_point[0]), int(end_point[1]))
                    
                    cv2.line(annotated, start_coords, end_coords,
                            self.config.skeleton_color, self.config.skeleton_thickness)
        
        # Draw keypoints
        for i, keypoint in enumerate(keypoints):
            if keypoint[2] > self.config.keypoint_visibility_threshold:
                center = (int(keypoint[0]), int(keypoint[1]))
                
                # Different colors for different body parts
                if i in [0, 1, 2, 3, 4, 5, 6, 7, 8]:  # Face
                    color = (255, 0, 255)  # Magenta
                elif i in [9, 10]:  # Shoulders
                    color = (0, 255, 255)  # Yellow
                elif i in [11, 12, 23, 24]:  # Torso
                    color = (255, 255, 0)  # Cyan
                elif i in [13, 15, 17, 19, 21]:  # Left arm
                    color = (255, 165, 0)  # Orange
                elif i in [14, 16, 18, 20, 22]:  # Right arm
                    color = (255, 0, 0)  # Red
                elif i in [25, 27, 29, 31]:  # Left leg
                    color = (0, 255, 0)  # Green
                elif i in [26, 28, 30, 32]:  # Right leg
                    color = (0, 0, 255)  # Blue
                else:
                    color = self.config.keypoint_color
                
                cv2.circle(annotated, center, self.config.keypoint_radius, color, -1)
        
        return annotated
    
    def _draw_density_zones(
        self,
        frame: np.ndarray,
        density_metrics: DensityMetrics
    ) -> np.ndarray:
        """
        Draw density zones with color coding.
        
        Args:
            frame: Input frame
            density_metrics: Density metrics
            
        Returns:
            Frame with drawn zones
        """
        zone_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Define zone boundaries
        zones = {
            'top_left': [(0, 0), (w//2, h//2)],
            'top_right': [(w//2, 0), (w, h//2)],
            'bottom_left': [(0, h//2), (w//2, h)],
            'bottom_right': [(w//2, h//2), (w, h)],
        }
        
        for zone_name, ((x1, y1), (x2, y2)) in zones.items():
            density = density_metrics.density_zones.get(zone_name, 0.0)
            
            # Determine color based on density
            if density < 0.5:
                color = self.config.zone_colors['low']
            elif density < 1.0:
                color = self.config.zone_colors['medium']
            elif density < 2.0:
                color = self.config.zone_colors['high']
            else:
                color = self.config.zone_colors['critical']
            
            # Draw zone with transparency
            overlay = zone_frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.2, zone_frame, 0.8, 0, zone_frame)
            
            # Draw zone border
            cv2.rectangle(zone_frame, (x1, y1), (x2, y2), color, 2)
            
            # Add zone label
            label = f"{zone_name.replace('_', ' ').title()}: {density:.2f}/m²"
            cv2.putText(zone_frame, label, (x1 + 5, y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return zone_frame
    
    def _draw_flow_indicators(
        self,
        frame: np.ndarray,
        density_metrics: DensityMetrics
    ) -> np.ndarray:
        """
        Draw crowd flow direction indicators.
        
        Args:
            frame: Input frame
            density_metrics: Density metrics with flow info
            
        Returns:
            Frame with flow indicators
        """
        flow_frame = frame.copy()
        h, w = frame.shape[:2]
        
        if density_metrics.flow_direction is not None and density_metrics.flow_speed is not None:
            # Calculate center of frame
            center_x, center_y = w // 2, h // 2
            
            # Calculate flow arrow end point
            flow_scale = 100  # Scale factor for visualization
            end_x = int(center_x + density_metrics.flow_direction[0] * flow_scale)
            end_y = int(center_y + density_metrics.flow_direction[1] * flow_scale)
            
            # Draw flow arrow
            cv2.arrowedLine(flow_frame, (center_x, center_y), (end_x, end_y),
                          self.config.flow_arrow_color, self.config.flow_arrow_thickness,
                          tipLength=0.3)
            
            # Add flow speed text
            speed_text = f"Flow: {density_metrics.flow_speed:.1f} px/frame"
            cv2.putText(flow_frame, speed_text, (center_x - 50, center_y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.config.flow_arrow_color, 2)
        
        return flow_frame
    
    def _add_metrics_overlay(
        self,
        frame: np.ndarray,
        results: ProcessingResults
    ) -> np.ndarray:
        """
        Add comprehensive metrics overlay to frame.
        
        Args:
            frame: Input frame
            results: Processing results
            
        Returns:
            Frame with metrics overlay
        """
        metrics_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Prepare metrics text
        metrics_text = [
            f"People Count: {results.density_metrics.crowd_count}",
            f"Overall Density: {results.density_metrics.overall_density:.2f}/m²",
            f"Peak Density: {results.density_metrics.peak_density:.2f}",
            f"Area Coverage: {results.density_metrics.area_coverage:.1%}",
        ]
        
        if results.density_metrics.flow_speed is not None:
            metrics_text.append(f"Flow Speed: {results.density_metrics.flow_speed:.1f} px/frame")
        
        if self.config.show_fps:
            metrics_text.append(f"FPS: {results.fps:.1f}")
        
        metrics_text.append(f"Processing: {results.processing_time*1000:.1f}ms")
        
        # Calculate text dimensions
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        text_size = cv2.getTextSize("Test", font, font_scale, thickness)[0]
        line_height = text_size[1] + 10
        
        # Calculate background rectangle size
        bg_width = max([cv2.getTextSize(text, font, font_scale, thickness)[0][0] 
                       for text in metrics_text]) + 20
        bg_height = len(metrics_text) * line_height + 20
        
        # Determine position
        if self.config.metrics_position == "top-left":
            x, y = 10, 10
        elif self.config.metrics_position == "top-right":
            x, y = w - bg_width - 10, 10
        elif self.config.metrics_position == "bottom-left":
            x, y = 10, h - bg_height - 10
        else:  # bottom-right
            x, y = w - bg_width - 10, h - bg_height - 10
        
        # Draw background rectangle
        overlay = metrics_frame.copy()
        cv2.rectangle(overlay, (x, y), (x + bg_width, y + bg_height),
                     self.config.metrics_bg_color, -1)
        cv2.addWeighted(overlay, self.config.metrics_bg_alpha, metrics_frame, 
                       1 - self.config.metrics_bg_alpha, 0, metrics_frame)
        
        # Draw text
        for i, text in enumerate(metrics_text):
            text_y = y + 20 + i * line_height
            cv2.putText(metrics_frame, text, (x + 10, text_y),
                       font, font_scale, self.config.metrics_text_color, thickness)
        
        return metrics_frame
    
    def _add_timestamp(self, frame: np.ndarray, timestamp: datetime) -> np.ndarray:
        """Add timestamp to frame."""
        timestamp_text = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp_text, (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame
    
    def _add_alerts(self, frame: np.ndarray, results: ProcessingResults) -> np.ndarray:
        """Add alert overlays to frame."""
        # This would integrate with the alert system
        # For now, just a placeholder implementation
        alert_frame = frame.copy()
        
        # Check for high density alert
        if results.density_metrics.overall_density > 2.0:
            # Flash alert
            current_time = time.time()
            if int(current_time * self.config.alert_flash_frequency) % 2 == 0:
                # Draw red border
                h, w = alert_frame.shape[:2]
                cv2.rectangle(alert_frame, (0, 0), (w-1, h-1), 
                            self.config.alert_color, 10)
                
                # Add alert text
                alert_text = "HIGH CROWD DENSITY ALERT"
                text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = (w - text_size[0]) // 2
                text_y = 50
                
                # Background for alert text
                cv2.rectangle(alert_frame, 
                            (text_x - 10, text_y - 30),
                            (text_x + text_size[0] + 10, text_y + 10),
                            self.config.alert_color, -1)
                
                cv2.putText(alert_frame, alert_text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return alert_frame
    
    def _overlay_heatmap(self, frame: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        """Overlay heatmap on frame."""
        # Ensure frame and heatmap have same size
        if frame.shape[:2] != heatmap.shape[:2]:
            heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        
        # Blend frame with heatmap
        overlayed = cv2.addWeighted(frame, 1 - self.config.heatmap_alpha, 
                                   heatmap, self.config.heatmap_alpha, 0)
        return overlayed
    
    def create_analysis_dashboard(
        self,
        results: ProcessingResults,
        frame: np.ndarray,
        heatmap: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Create a comprehensive analysis dashboard with multiple views.
        
        Args:
            results: Processing results
            frame: Original frame
            heatmap: Optional heatmap
            
        Returns:
            Dashboard image with multiple analysis views
        """
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Crowd Analysis Dashboard - {results.timestamp.strftime("%Y-%m-%d %H:%M:%S")}', 
                    fontsize=16)
        
        # Top-left: Original frame with detections
        axes[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Frame with Detections')
        axes[0, 0].axis('off')
        
        # Top-right: Heatmap
        if heatmap is not None:
            axes[0, 1].imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
            axes[0, 1].set_title('Crowd Density Heatmap')
        else:
            axes[0, 1].text(0.5, 0.5, 'No Heatmap Available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Crowd Density Heatmap')
        axes[0, 1].axis('off')
        
        # Bottom-left: Density metrics bar chart
        zone_names = list(results.density_metrics.density_zones.keys())
        zone_values = list(results.density_metrics.density_zones.values())
        axes[1, 0].bar(zone_names, zone_values, color='skyblue')
        axes[1, 0].set_title('Density by Zone')
        axes[1, 0].set_ylabel('Density (people/m²)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Bottom-right: Metrics summary
        metrics_text = f"""
        People Count: {results.density_metrics.crowd_count}
        Overall Density: {results.density_metrics.overall_density:.2f}/m²
        Peak Density: {results.density_metrics.peak_density:.2f}
        Area Coverage: {results.density_metrics.area_coverage:.1%}
        Flow Speed: {results.density_metrics.flow_speed:.1f} px/frame
        Processing Time: {results.processing_time*1000:.1f}ms
        FPS: {results.fps:.1f}
        """
        axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='center')
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Convert matplotlib figure to numpy array
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        dashboard = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        dashboard = dashboard.reshape(canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return dashboard
