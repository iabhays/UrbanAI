"""
Advanced Visualizer for Crowd Risk Analysis System.

Comprehensive visualization with bounding boxes, risk levels,
debug information, motion heatmaps, and alert overlays.
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import math

from .person_tracker import TrackedPerson
from .motion_analyzer import CrowdMotionMetrics, MotionFeatures
from .risk_evaluator import CrowdRisk, PersonRisk
from .alert_manager import Alert
from sentient_city.core import get_logger, get_config


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    
    # Display options
    show_bounding_boxes: bool = True
    show_track_ids: bool = True
    show_risk_levels: bool = True
    show_confidence_scores: bool = True
    show_trajectories: bool = True
    show_motion_heatmap: bool = False
    show_debug_info: bool = False
    show_alert_banner: bool = True
    
    # Colors (BGR format)
    colors = {
        'low_risk': (0, 255, 0),      # Green
        'medium_risk': (0, 255, 255),  # Yellow
        'high_risk': (0, 0, 255),      # Red
        'critical_risk': (0, 0, 139),  # Dark Red
        'trajectory': (255, 255, 255), # White
        'text': (255, 255, 255),       # White
        'alert_bg': (0, 0, 139),       # Dark Red
        'debug_bg': (50, 50, 50),      # Dark Gray
    }
    
    # Font settings
    font_scale: float = 0.6
    font_thickness: int = 2
    box_thickness: int = 2
    
    # Layout
    info_panel_height: int = 150
    alert_banner_height: int = 60


class CrowdVisualizer:
    """
    Advanced visualizer for crowd risk analysis.
    
    Features:
    - Person bounding boxes with risk-based coloring
    - Track IDs and confidence scores
    - Risk level indicators
    - Motion trajectories
    - Debug information overlay
    - Alert banners
    - Motion heatmaps
    - Crowd metrics display
    """
    
    def __init__(
        self,
        config: Optional[VisualizationConfig] = None,
        debug_mode: bool = False
    ):
        """
        Initialize visualizer.
        
        Args:
            config: Visualization configuration
            debug_mode: Enable debug mode by default
        """
        self.config = config or VisualizationConfig()
        self.debug_mode = debug_mode
        
        self.logger = get_logger(__name__)
        
        # Visualization state
        self.frame_count = 0
        self.heatmap_alpha = 0.4
        
        # Performance metrics
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = 0
        
    def visualize_frame(
        self,
        frame: np.ndarray,
        tracked_persons: List[TrackedPerson],
        crowd_metrics: CrowdMotionMetrics,
        person_risks: Dict[int, PersonRisk],
        crowd_risk: CrowdRisk,
        active_alerts: List[Alert],
        motion_heatmap: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Create comprehensive visualization of frame analysis.
        
        Args:
            frame: Original frame
            tracked_persons: List of tracked persons
            crowd_metrics: Crowd motion metrics
            person_risks: Per-person risk assessments
            crowd_risk: Global crowd risk
            active_alerts: Active alerts
            motion_heatmap: Optional motion heatmap overlay
            
        Returns:
            Visualized frame
        """
        # Create copy for drawing
        vis_frame = frame.copy()
        
        # Add motion heatmap if enabled
        if self.config.show_motion_heatmap and motion_heatmap is not None:
            vis_frame = self._add_motion_heatmap(vis_frame, motion_heatmap)
        
        # Draw person bounding boxes and information
        if self.config.show_bounding_boxes:
            vis_frame = self._draw_persons(vis_frame, tracked_persons, person_risks)
        
        # Draw trajectories if enabled
        if self.config.show_trajectories:
            vis_frame = self._draw_trajectories(vis_frame, tracked_persons)
        
        # Draw alert banner if enabled and alerts exist
        if self.config.show_alert_banner and active_alerts:
            vis_frame = self._draw_alert_banner(vis_frame, active_alerts)
        
        # Draw information panel
        vis_frame = self._draw_info_panel(
            vis_frame, crowd_metrics, crowd_risk, len(tracked_persons)
        )
        
        # Draw debug information if enabled
        if self.config.show_debug_info or self.debug_mode:
            vis_frame = self._draw_debug_info(
                vis_frame, crowd_metrics, person_risks, crowd_risk
            )
        
        # Draw FPS counter
        vis_frame = self._draw_fps_counter(vis_frame)
        
        self.frame_count += 1
        
        return vis_frame
    
    def _draw_persons(
        self,
        frame: np.ndarray,
        tracked_persons: List[TrackedPerson],
        person_risks: Dict[int, PersonRisk]
    ) -> np.ndarray:
        """Draw persons with bounding boxes and risk information."""
        for person in tracked_persons:
            risk = person_risks.get(person.track_id)
            if not risk:
                continue
            
            # Get color based on risk level
            color = self._get_risk_color(risk.risk_level)
            
            # Draw bounding box
            bbox = person.bbox.astype(int)
            cv2.rectangle(
                frame,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color,
                self.config.box_thickness
            )
            
            # Prepare label text
            label_lines = []
            
            if self.config.show_track_ids:
                label_lines.append(f"ID:{person.track_id}")
            
            if self.config.show_risk_levels:
                label_lines.append(f"{risk.risk_level}")
            
            if self.config.show_confidence_scores:
                label_lines.append(f"C:{person.confidence:.2f}")
            
            # Draw label background and text
            if label_lines:
                label_text = " ".join(label_lines)
                self._draw_label(frame, bbox, label_text, color)
            
            # Draw risk indicator (small circle)
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = bbox[1] - 10
            cv2.circle(frame, (center_x, center_y), 5, color, -1)
        
        return frame
    
    def _draw_trajectories(self, frame: np.ndarray, tracked_persons: List[TrackedPerson]) -> np.ndarray:
        """Draw motion trajectories for tracked persons."""
        for person in tracked_persons:
            if len(person.trajectory) < 2:
                continue
            
            # Extract trajectory points
            points = []
            for traj_point in person.trajectory:
                center = traj_point['center']
                points.append((int(center[0]), int(center[1])))
            
            # Draw trajectory with fading effect
            for i in range(1, len(points)):
                # Calculate alpha for fading effect
                alpha = i / len(points)
                thickness = max(1, int(3 * alpha))
                
                # Get color based on person's risk level
                risk_color = self._get_risk_color(person.risk_level)
                
                # Apply fading to color
                faded_color = tuple(int(c * alpha) for c in risk_color)
                
                cv2.line(frame, points[i-1], points[i], faded_color, thickness)
            
            # Draw trajectory points
            for i, point in enumerate(points):
                alpha = i / len(points)
                radius = max(1, int(3 * alpha))
                cv2.circle(frame, point, radius, self.config.colors['trajectory'], -1)
        
        return frame
    
    def _draw_alert_banner(self, frame: np.ndarray, active_alerts: List[Alert]) -> np.ndarray:
        """Draw alert banner at the top of the frame."""
        if not active_alerts:
            return frame
        
        # Get highest severity alert
        highest_alert = max(active_alerts, key=lambda a: self._severity_priority(a.severity))
        
        # Banner dimensions
        h, w = frame.shape[:2]
        banner_y = h - self.config.alert_banner_height
        
        # Draw banner background
        banner_color = self._get_alert_color(highest_alert.severity)
        cv2.rectangle(
            frame,
            (0, banner_y),
            (w, h),
            banner_color,
            -1
        )
        
        # Prepare alert text
        alert_text = f"⚠ {highest_alert.severity} ALERT: {highest_alert.message}"
        
        # Truncate text if too long
        max_text_width = w - 40
        text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale, self.config.font_thickness)[0]
        if text_size[0] > max_text_width:
            # Simple truncation
            while text_size[0] > max_text_width and len(alert_text) > 20:
                alert_text = alert_text[:-1]
                text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale, self.config.font_thickness)[0]
            alert_text += "..."
        
        # Draw alert text
        text_x = 20
        text_y = banner_y + 35
        
        cv2.putText(
            frame,
            alert_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.font_scale,
            self.config.colors['text'],
            self.config.font_thickness
        )
        
        # Draw timestamp
        timestamp_text = highest_alert.timestamp.strftime("%H:%M:%S")
        cv2.putText(
            frame,
            timestamp_text,
            (w - 150, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.font_scale * 0.8,
            self.config.colors['text'],
            1
        )
        
        return frame
    
    def _draw_info_panel(
        self,
        frame: np.ndarray,
        crowd_metrics: CrowdMotionMetrics,
        crowd_risk: CrowdRisk,
        crowd_size: int
    ) -> np.ndarray:
        """Draw information panel with crowd metrics."""
        # Panel dimensions
        h, w = frame.shape[:2]
        panel_height = self.config.info_panel_height
        
        # Draw panel background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (0, 0),
            (w, panel_height),
            self.config.colors['debug_bg'],
            -1
        )
        
        # Apply transparency
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Prepare information lines
        info_lines = [
            f"Crowd Size: {crowd_size} persons",
            f"Risk Level: {crowd_risk.risk_level} ({crowd_risk.risk_score:.2f})",
            f"Avg Speed: {crowd_metrics.avg_speed:.1f} px/frame",
            f"Direction Variance: {crowd_metrics.direction_variance:.3f}",
            f"Chaos Regions: {len(crowd_metrics.chaotic_regions)}",
            f"Compression: {crowd_metrics.compression_density:.3f}"
        ]
        
        # Draw information text
        y_offset = 25
        for i, line in enumerate(info_lines):
            x = 10 + (i % 3) * 250
            y = y_offset + (i // 3) * 30
            
            cv2.putText(
                frame,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.font_scale * 0.8,
                self.config.colors['text'],
                1
            )
        
        return frame
    
    def _draw_debug_info(
        self,
        frame: np.ndarray,
        crowd_metrics: CrowdMotionMetrics,
        person_risks: Dict[int, PersonRisk],
        crowd_risk: CrowdRisk
    ) -> np.ndarray:
        """Draw detailed debug information."""
        # Debug panel on the right side
        h, w = frame.shape[:2]
        panel_width = 300
        
        # Draw panel background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (w - panel_width, 0),
            (w, h),
            self.config.colors['debug_bg'],
            -1
        )
        
        # Apply transparency
        alpha = 0.8
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Prepare debug information
        debug_lines = [
            "=== DEBUG INFO ===",
            f"Frame: {self.frame_count}",
            f"FPS: {self._get_current_fps():.1f}",
            "",
            "=== CROWD METRICS ===",
            f"Density: {crowd_metrics.compression_density:.3f}",
            f"Entropy: {crowd_metrics.crowd_entropy:.3f}",
            f"Flow Coherence: {crowd_metrics.flow_coherence:.3f}",
            f"Expansion Rate: {crowd_metrics.expansion_rate:.3f}",
            "",
            "=== RISK FACTORS ===",
        ]
        
        # Add risk factors
        for factor, value in crowd_risk.risk_factors.items():
            debug_lines.append(f"{factor}: {value:.3f}")
        
        debug_lines.extend([
            "",
            "=== PERSON RISKS ===",
        ])
        
        # Add person risk summary
        risk_counts = {"Low": 0, "Medium": 0, "High": 0}
        for risk in person_risks.values():
            risk_counts[risk.risk_level] += 1
        
        for level, count in risk_counts.items():
            debug_lines.append(f"{level}: {count}")
        
        # Draw debug text
        y_offset = 20
        for line in debug_lines:
            if line.startswith("==="):
                color = (0, 255, 255)  # Yellow for headers
            else:
                color = self.config.colors['text']
            
            cv2.putText(
                frame,
                line,
                (w - panel_width + 10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.font_scale * 0.6,
                color,
                1
            )
            
            y_offset += 18
            if y_offset > h - 20:
                break
        
        return frame
    
    def _draw_fps_counter(self, frame: np.ndarray) -> np.ndarray:
        """Draw FPS counter in corner."""
        fps_text = f"FPS: {self._get_current_fps():.1f}"
        
        # Position in top-right corner
        h, w = frame.shape[:2]
        text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale, 1)[0]
        
        # Draw background
        cv2.rectangle(
            frame,
            (w - text_size[0] - 20, 10),
            (w - 10, text_size[1] + 20),
            (0, 0, 0),
            -1
        )
        
        # Draw text
        cv2.putText(
            frame,
            fps_text,
            (w - text_size[0] - 15, text_size[1] + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.font_scale,
            (0, 255, 0),
            1
        )
        
        return frame
    
    def _add_motion_heatmap(self, frame: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        """Add motion heatmap overlay to frame."""
        if heatmap.shape[:2] != frame.shape[:2]:
            # Resize heatmap to match frame
            heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        
        # Apply heatmap with transparency
        result = cv2.addWeighted(frame, 1 - self.heatmap_alpha, heatmap, self.heatmap_alpha, 0)
        
        return result
    
    def _draw_label(self, frame: np.ndarray, bbox: np.ndarray, text: str, color: Tuple[int, int, int]):
        """Draw label background and text."""
        # Get text size
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale, self.config.font_thickness)[0]
        
        # Label position
        label_x = bbox[0]
        label_y = bbox[1] - 10
        
        # Ensure label is within frame bounds
        if label_y < text_size[1] + 10:
            label_y = bbox[1] + text_size[1] + 10
        
        # Draw background rectangle
        cv2.rectangle(
            frame,
            (label_x, label_y - text_size[1] - 5),
            (label_x + text_size[0] + 10, label_y + 5),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            frame,
            text,
            (label_x + 5, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.font_scale,
            self.config.colors['text'],
            self.config.font_thickness
        )
    
    def _get_risk_color(self, risk_level: str) -> Tuple[int, int, int]:
        """Get color based on risk level."""
        color_map = {
            "Low": self.config.colors['low_risk'],
            "Medium": self.config.colors['medium_risk'],
            "High": self.config.colors['high_risk'],
            "Critical": self.config.colors['critical_risk']
        }
        return color_map.get(risk_level, self.config.colors['low_risk'])
    
    def _get_alert_color(self, severity: str) -> Tuple[int, int, int]:
        """Get color based on alert severity."""
        color_map = {
            "Low": (255, 255, 0),      # Yellow
            "Medium": (0, 165, 255),   # Orange
            "High": (0, 0, 255),       # Red
            "Critical": (139, 0, 0)    # Dark Red
        }
        return color_map.get(severity, self.config.colors['alert_bg'])
    
    def _severity_priority(self, severity: str) -> int:
        """Get priority for alert severity."""
        priority_map = {
            "Low": 1,
            "Medium": 2,
            "High": 3,
            "Critical": 4
        }
        return priority_map.get(severity, 0)
    
    def _get_current_fps(self) -> float:
        """Calculate current FPS."""
        import time
        current_time = time.time()
        
        if self.last_frame_time > 0:
            fps = 1.0 / (current_time - self.last_frame_time)
            self.fps_history.append(fps)
        
        self.last_frame_time = current_time
        
        return np.mean(self.fps_history) if self.fps_history else 0.0
    
    def create_summary_visualization(
        self,
        frames: List[np.ndarray],
        metrics_list: List[Dict],
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """Create summary visualization with multiple frames and metrics."""
        if not frames:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create grid layout
        grid_size = int(np.ceil(np.sqrt(len(frames))))
        cell_height = 240
        cell_width = 320
        
        summary_frame = np.zeros((grid_size * cell_height, grid_size * cell_width, 3), dtype=np.uint8)
        
        for i, frame in enumerate(frames):
            if i >= grid_size * grid_size:
                break
            
            # Resize frame to cell size
            resized_frame = cv2.resize(frame, (cell_width, cell_height))
            
            # Calculate position in grid
            row = i // grid_size
            col = i % grid_size
            
            # Place frame in grid
            y_start = row * cell_height
            y_end = y_start + cell_height
            x_start = col * cell_width
            x_end = x_start + cell_width
            
            summary_frame[y_start:y_end, x_start:x_end] = resized_frame
            
            # Add frame number
            cv2.putText(
                summary_frame,
                f"Frame {i+1}",
                (x_start + 10, y_start + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        
        # Save if path provided
        if output_path:
            cv2.imwrite(output_path, summary_frame)
        
        return summary_frame
    
    def toggle_debug_mode(self):
        """Toggle debug mode on/off."""
        self.debug_mode = not self.debug_mode
        self.config.show_debug_info = self.debug_mode
        self.logger.info(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
    
    def update_config(self, **kwargs):
        """Update visualization configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.info(f"Updated config: {key} = {value}")
    
    def reset(self):
        """Reset visualizer state."""
        self.frame_count = 0
        self.fps_history.clear()
        self.last_frame_time = 0
