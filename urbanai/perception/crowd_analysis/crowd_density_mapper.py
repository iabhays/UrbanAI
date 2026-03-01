"""
Crowd Density Mapping and Heatmap Generation System.

This module provides comprehensive crowd density analysis with heatmap visualization,
density maps, and crowd flow analysis for intelligent monitoring systems.
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import scipy.ndimage as ndimage
from scipy.spatial.distance import cdist

from .enhanced_person_detector import SkeletonDetection


@dataclass
class DensityMetrics:
    """Crowd density metrics for analysis."""
    overall_density: float  # People per square meter
    local_density: np.ndarray  # Local density map
    peak_density: float  # Maximum density point
    density_zones: Dict[str, float]  # Density by zones
    crowd_count: int  # Total number of people
    area_coverage: float  # Percentage of area covered by crowd
    flow_direction: Optional[np.ndarray]  # Primary flow direction
    flow_speed: Optional[float]  # Average flow speed


@dataclass
class HeatmapConfig:
    """Configuration for heatmap generation."""
    grid_size: Tuple[int, int] = (20, 20)  # Grid resolution
    sigma: float = 2.0  # Gaussian blur sigma
    colormap: int = cv2.COLORMAP_JET  # OpenCV colormap
    alpha: float = 0.6  # Overlay transparency
    density_threshold: float = 0.1  # Minimum density threshold
    show_grid: bool = False  # Show density grid lines
    normalize: bool = True  # Normalize heatmap values


class CrowdDensityMapper:
    """
    Advanced crowd density mapping and heatmap generation system.
    
    Features:
    - Real-time density calculation
    - Adaptive grid resolution
    - Multiple visualization modes
    - Flow analysis and direction detection
    - Zone-based density analysis
    - Historical density tracking
    """
    
    def __init__(
        self,
        frame_size: Tuple[int, int] = (640, 480),
        config: Optional[HeatmapConfig] = None
    ):
        """
        Initialize crowd density mapper.
        
        Args:
            frame_size: Video frame dimensions (width, height)
            config: Heatmap configuration
        """
        self.frame_width, self.frame_height = frame_size
        self.config = config or HeatmapConfig()
        
        # Initialize density grid
        self.grid_width, self.grid_height = self.config.grid_size
        self.density_grid = np.zeros((self.grid_height, self.grid_width))
        self.accumulated_density = np.zeros((self.grid_height, self.grid_width))
        
        # Historical data for flow analysis
        self.position_history = []
        self.max_history_length = 30  # Keep last 30 frames
        
        # Zone definitions (can be customized)
        self.zones = self._create_default_zones()
        
        # Performance metrics
        self.frame_count = 0
        self.total_people_processed = 0
        
        print(f"CrowdDensityMapper initialized for {frame_size} frames")
    
    def _create_default_zones(self) -> Dict[str, np.ndarray]:
        """Create default zone definitions."""
        zones = {
            'top_left': np.array([0, 0, self.frame_width//2, self.frame_height//2]),
            'top_right': np.array([self.frame_width//2, 0, self.frame_width, self.frame_height//2]),
            'bottom_left': np.array([0, self.frame_height//2, self.frame_width//2, self.frame_height]),
            'bottom_right': np.array([self.frame_width//2, self.frame_height//2, self.frame_width, self.frame_height]),
            'center': np.array([self.frame_width//4, self.frame_height//4, 
                              3*self.frame_width//4, 3*self.frame_height//4])
        }
        return zones
    
    def calculate_density(
        self,
        detections: List[SkeletonDetection],
        frame_id: Optional[int] = None
    ) -> DensityMetrics:
        """
        Calculate crowd density metrics from detections.
        
        Args:
            detections: List of person detections with skeletons
            frame_id: Optional frame ID
            
        Returns:
            DensityMetrics object with comprehensive density information
        """
        if frame_id is None:
            frame_id = self.frame_count
            
        self.frame_count += 1
        
        if not detections:
            return DensityMetrics(
                overall_density=0.0,
                local_density=np.zeros(self.config.grid_size),
                peak_density=0.0,
                density_zones={zone: 0.0 for zone in self.zones.keys()},
                crowd_count=0,
                area_coverage=0.0,
                flow_direction=None,
                flow_speed=None
            )
        
        # Extract person positions
        positions = np.array([detection.center for detection in detections])
        self.total_people_processed += len(positions)
        
        # Update position history for flow analysis
        self.position_history.append(positions.copy())
        if len(self.position_history) > self.max_history_length:
            self.position_history.pop(0)
        
        # Calculate density grid
        density_grid = self._calculate_density_grid(positions)
        
        # Calculate overall density (people per square meter)
        frame_area_m2 = (self.frame_width * self.frame_height) / (100 * 100)  # Convert to m²
        overall_density = len(detections) / frame_area_m2
        
        # Calculate peak density
        peak_density = np.max(density_grid)
        
        # Calculate zone densities
        density_zones = self._calculate_zone_densities(positions)
        
        # Calculate area coverage
        area_coverage = np.sum(density_grid > self.config.density_threshold) / density_grid.size
        
        # Calculate flow metrics
        flow_direction, flow_speed = self._calculate_flow_metrics()
        
        metrics = DensityMetrics(
            overall_density=overall_density,
            local_density=density_grid,
            peak_density=peak_density,
            density_zones=density_zones,
            crowd_count=len(detections),
            area_coverage=area_coverage,
            flow_direction=flow_direction,
            flow_speed=flow_speed
        )
        
        # Update accumulated density
        self.accumulated_density = 0.9 * self.accumulated_density + 0.1 * density_grid
        
        return metrics
    
    def _calculate_density_grid(self, positions: np.ndarray) -> np.ndarray:
        """
        Calculate density grid from person positions.
        
        Args:
            positions: Array of person positions (N, 2)
            
        Returns:
            Density grid of shape (grid_height, grid_width)
        """
        density_grid = np.zeros((self.grid_height, self.grid_width))
        
        if len(positions) == 0:
            return density_grid
        
        # Convert positions to grid coordinates
        grid_x = (positions[:, 0] * self.grid_width / self.frame_width).astype(int)
        grid_y = (positions[:, 1] * self.grid_height / self.frame_height).astype(int)
        
        # Clamp to grid bounds
        grid_x = np.clip(grid_x, 0, self.grid_width - 1)
        grid_y = np.clip(grid_y, 0, self.grid_height - 1)
        
        # Add person contributions to grid
        for gx, gy in zip(grid_x, grid_y):
            density_grid[gy, gx] += 1
        
        # Apply Gaussian smoothing for more natural density
        density_grid = ndimage.gaussian_filter(density_grid, sigma=self.config.sigma)
        
        # Normalize if requested
        if self.config.normalize:
            max_density = np.max(density_grid)
            if max_density > 0:
                density_grid = density_grid / max_density
        
        return density_grid
    
    def _calculate_zone_densities(self, positions: np.ndarray) -> Dict[str, float]:
        """
        Calculate density for predefined zones.
        
        Args:
            positions: Array of person positions
            
        Returns:
            Dictionary of zone densities
        """
        zone_densities = {}
        
        for zone_name, zone_bounds in self.zones.items():
            x1, y1, x2, y2 = zone_bounds
            
            # Count people in zone
            in_zone = ((positions[:, 0] >= x1) & (positions[:, 0] <= x2) &
                      (positions[:, 1] >= y1) & (positions[:, 1] <= y2))
            
            zone_area = ((x2 - x1) * (y2 - y1)) / (100 * 100)  # Convert to m²
            zone_people = np.sum(in_zone)
            
            zone_density = zone_people / zone_area if zone_area > 0 else 0.0
            zone_densities[zone_name] = zone_density
        
        return zone_densities
    
    def _calculate_flow_metrics(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Calculate crowd flow direction and speed.
        
        Returns:
            Tuple of (flow_direction, flow_speed)
        """
        if len(self.position_history) < 2:
            return None, None
        
        # Calculate average displacement between consecutive frames
        displacements = []
        for i in range(1, len(self.position_history)):
            if len(self.position_history[i]) > 0 and len(self.position_history[i-1]) > 0:
                # Simple approach: compare centroids
                curr_centroid = np.mean(self.position_history[i], axis=0)
                prev_centroid = np.mean(self.position_history[i-1], axis=0)
                displacement = curr_centroid - prev_centroid
                displacements.append(displacement)
        
        if not displacements:
            return None, None
        
        # Calculate average flow direction and speed
        avg_displacement = np.mean(displacements, axis=0)
        flow_speed = np.linalg.norm(avg_displacement)
        
        if flow_speed > 0:
            flow_direction = avg_displacement / flow_speed
        else:
            flow_direction = np.array([0, 0])
        
        return flow_direction, flow_speed
    
    def generate_heatmap(
        self,
        density_grid: Optional[np.ndarray] = None,
        use_accumulated: bool = False
    ) -> np.ndarray:
        """
        Generate heatmap visualization from density grid.
        
        Args:
            density_grid: Density grid to visualize (uses current if None)
            use_accumulated: Use accumulated density instead of current
            
        Returns:
            Heatmap image
        """
        if density_grid is None:
            density_grid = self.density_grid
        
        if use_accumulated:
            density_grid = self.accumulated_density
        
        # Resize density grid to frame size
        heatmap = cv2.resize(
            density_grid,
            (self.frame_width, self.frame_height),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Convert to 8-bit and apply colormap
        heatmap_8bit = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_8bit, self.config.colormap)
        
        return heatmap_colored
    
    def overlay_heatmap(
        self,
        frame: np.ndarray,
        heatmap: Optional[np.ndarray] = None,
        alpha: Optional[float] = None
    ) -> np.ndarray:
        """
        Overlay heatmap on original frame.
        
        Args:
            frame: Original frame
            heatmap: Heatmap to overlay (generates if None)
            alpha: Overlay transparency (uses config if None)
            
        Returns:
            Frame with heatmap overlay
        """
        if heatmap is None:
            heatmap = self.generate_heatmap()
        
        if alpha is None:
            alpha = self.config.alpha
        
        # Ensure frame and heatmap have same size
        if frame.shape[:2] != heatmap.shape[:2]:
            heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        
        # Blend frame with heatmap
        overlayed = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
        
        # Add grid lines if requested
        if self.config.show_grid:
            overlayed = self._draw_grid_lines(overlayed)
        
        return overlayed
    
    def _draw_grid_lines(self, frame: np.ndarray) -> np.ndarray:
        """Draw density grid lines on frame."""
        grid_frame = frame.copy()
        
        # Vertical lines
        for i in range(1, self.grid_width):
            x = int(i * self.frame_width / self.grid_width)
            cv2.line(grid_frame, (x, 0), (x, self.frame_height), (255, 255, 255), 1)
        
        # Horizontal lines
        for i in range(1, self.grid_height):
            y = int(i * self.frame_height / self.grid_height)
            cv2.line(grid_frame, (0, y), (self.frame_width, y), (255, 255, 255), 1)
        
        return grid_frame
    
    def draw_density_zones(
        self,
        frame: np.ndarray,
        density_zones: Dict[str, float],
        threshold_colors: Dict[str, Tuple[int, int, int]] = None
    ) -> np.ndarray:
        """
        Draw density zones with color coding based on density levels.
        
        Args:
            frame: Original frame
            density_zones: Dictionary of zone densities
            threshold_colors: Color thresholds for density levels
            
        Returns:
            Frame with zone overlays
        """
        if threshold_colors is None:
            threshold_colors = {
                'low': (0, 255, 0),      # Green
                'medium': (0, 255, 255), # Yellow
                'high': (0, 0, 255),     # Red
                'critical': (255, 0, 0)  # Dark Red
            }
        
        zone_frame = frame.copy()
        
        for zone_name, zone_bounds in self.zones.items():
            density = density_zones.get(zone_name, 0.0)
            
            # Determine color based on density
            if density < 0.5:
                color = threshold_colors['low']
            elif density < 1.0:
                color = threshold_colors['medium']
            elif density < 2.0:
                color = threshold_colors['high']
            else:
                color = threshold_colors['critical']
            
            # Draw zone rectangle with transparency
            x1, y1, x2, y2 = map(int, zone_bounds)
            overlay = zone_frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.2, zone_frame, 0.8, 0, zone_frame)
            
            # Draw zone border
            cv2.rectangle(zone_frame, (x1, y1), (x2, y2), color, 2)
            
            # Add zone label
            label = f"{zone_name}: {density:.2f}/m²"
            cv2.putText(zone_frame, label, (x1 + 5, y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return zone_frame
    
    def get_density_statistics(self) -> Dict:
        """Get comprehensive density statistics."""
        if self.frame_count == 0:
            return {}
        
        avg_people_per_frame = self.total_people_processed / self.frame_count
        current_peak_density = np.max(self.density_grid)
        accumulated_peak_density = np.max(self.accumulated_density)
        
        return {
            'frames_processed': self.frame_count,
            'total_people_processed': self.total_people_processed,
            'avg_people_per_frame': avg_people_per_frame,
            'current_peak_density': current_peak_density,
            'accumulated_peak_density': accumulated_peak_density,
            'grid_resolution': self.config.grid_size,
            'frame_size': (self.frame_width, self.frame_height)
        }
    
    def reset_accumulated_density(self):
        """Reset accumulated density history."""
        self.accumulated_density = np.zeros((self.grid_height, self.grid_width))
        self.position_history = []
        print("Accumulated density reset")
    
    def update_config(self, config: HeatmapConfig):
        """Update heatmap configuration."""
        self.config = config
        
        # Reinitialize density grid if size changed
        if config.grid_size != (self.grid_width, self.grid_height):
            self.grid_width, self.grid_height = config.grid_size
            self.density_grid = np.zeros((self.grid_height, self.grid_width))
            self.accumulated_density = np.zeros((self.grid_height, self.grid_width))
            
        print(f"Heatmap configuration updated: {config}")
