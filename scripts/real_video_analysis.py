#!/usr/bin/env python3
"""
Real Video Analysis Script with Person Detection and Risk Assessment.

This script processes videos with actual YOLO detection instead of mock results.
Features:
- Real person detection using YOLOv8
- Bounding box visualization
- Risk percentage calculation
- Person tracking
- Behavior analysis
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json
import time
from dataclasses import dataclass
from loguru import logger

# Add sentient_city to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from sentient_city.perception.pipeline import PerceptionPipeline
from sentient_city.perception.tracking import OCSortTracker


@dataclass
class VideoAnalysisResult:
    """Results from video analysis."""
    video_id: str
    total_frames: int
    total_detections: int
    total_tracks: int
    avg_crowd_density: float
    risk_percentage: float
    risk_level: str
    alerts: List[str]
    processing_time_seconds: float
    frames_with_detections: int


class RealVideoAnalyzer:
    """Real video analyzer with YOLO detection."""
    
    def __init__(self):
        """Initialize the analyzer."""
        self.tracker = OCSortTracker()
        self.pipeline = PerceptionPipeline(
            tracker=self.tracker,
            camera_id="video_analysis"
        )
        
    def calculate_risk_percentage(
        self, 
        detections_count: int, 
        crowd_density: float, 
        alerts_count: int,
        total_frames: int
    ) -> tuple[float, str]:
        """
        Calculate risk percentage based on multiple factors.
        
        Args:
            detections_count: Total number of person detections
            crowd_density: Average crowd density (0-1)
            alerts_count: Number of safety alerts
            total_frames: Total frames processed
        
        Returns:
            Tuple of (risk_percentage, risk_level)
        """
        # Base risk from crowd density
        density_risk = crowd_density * 40  # Max 40% from density
        
        # Risk from number of people (more people = higher risk)
        people_risk = min(detections_count / total_frames * 2, 30)  # Max 30% from people count
        
        # Risk from alerts (falls, panic, etc.)
        alert_risk = min(alerts_count * 10, 30)  # Max 30% from alerts
        
        # Calculate total risk
        total_risk = density_risk + people_risk + alert_risk
        total_risk = min(total_risk, 100)  # Cap at 100%
        
        # Determine risk level
        if total_risk >= 80:
            risk_level = "Critical"
        elif total_risk >= 60:
            risk_level = "High"
        elif total_risk >= 40:
            risk_level = "Medium"
        elif total_risk >= 20:
            risk_level = "Low"
        else:
            risk_level = "Very Low"
        
        return round(total_risk, 1), risk_level
    
    def analyze_video(self, video_path: str, progress_callback=None) -> VideoAnalysisResult:
        """
        Analyze a video file with real detection.
        
        Args:
            video_path: Path to video file
            progress_callback: Optional callback for progress updates
        
        Returns:
            VideoAnalysisResult with comprehensive metrics
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        video_id = video_path.stem
        
        logger.info(f"Starting real analysis of video: {video_id}")
        start_time = time.time()
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video: {width}x{height}, {fps:.2f}fps, {total_frames} frames")
        
        # Analysis metrics
        total_detections = 0
        total_tracks = set()
        crowd_densities = []
        all_alerts = []
        frames_with_detections = 0
        
        # Process every 10th frame for efficiency
        frame_skip = 10
        processed_frames = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames for efficiency
                if processed_frames % frame_skip != 0:
                    processed_frames += 1
                    continue
                
                # Process frame with detection and bounding boxes
                results = self.pipeline.process_frame(
                    frame, 
                    frame_id=processed_frames,
                    draw_bboxes=True
                )
                
                # Extract metrics
                detections = results.get("detections", [])
                tracks = results.get("tracks", [])
                alerts = results.get("alerts", [])
                
                if detections:
                    frames_with_detections += 1
                    total_detections += len(detections)
                
                # Track unique persons
                for track in tracks:
                    if hasattr(track, 'track_id'):
                        total_tracks.add(track.track_id)
                
                # Calculate crowd density for this frame
                frame_area = width * height
                person_area = 0
                for detection in detections:
                    bbox = detection['bbox']
                    x1, y1, x2, y2 = bbox
                    person_area += (x2 - x1) * (y2 - y1)
                
                density = min(person_area / frame_area, 1.0) if frame_area > 0 else 0
                crowd_densities.append(density)
                
                # Collect alerts
                for alert in alerts:
                    alert_type = alert.get('type', 'unknown')
                    confidence = alert.get('confidence', 0)
                    all_alerts.append(f"{alert_type.title()} (confidence: {confidence:.2f})")
                
                processed_frames += 1
                
                # Progress callback
                if progress_callback and processed_frames % 50 == 0:
                    progress = min((processed_frames / total_frames) * 100, 100)
                    progress_callback(video_id, progress, f"Processed {processed_frames}/{total_frames} frames")
                
                # Break if we've processed enough frames
                if processed_frames >= total_frames:
                    break
        
        finally:
            cap.release()
        
        # Calculate final metrics
        processing_time = time.time() - start_time
        avg_crowd_density = np.mean(crowd_densities) if crowd_densities else 0
        risk_percentage, risk_level = self.calculate_risk_percentage(
            total_detections,
            avg_crowd_density,
            len(all_alerts),
            frames_with_detections
        )
        
        # Create result
        result = VideoAnalysisResult(
            video_id=video_id,
            total_frames=total_frames,
            total_detections=total_detections,
            total_tracks=len(total_tracks),
            avg_crowd_density=round(avg_crowd_density, 3),
            risk_percentage=risk_percentage,
            risk_level=risk_level,
            alerts=all_alerts[:10],  # Limit to first 10 alerts
            processing_time_seconds=round(processing_time, 2),
            frames_with_detections=frames_with_detections
        )
        
        logger.info(f"Analysis complete for {video_id}: "
                   f"{total_detections} detections, {risk_percentage}% risk")
        
        return result


def main():
    """Main function for standalone testing."""
    analyzer = RealVideoAnalyzer()
    
    # Test with first video
    video_dir = Path(__file__).parent.parent / "datasets" / "raw"
    videos = list(video_dir.glob("*.mp4"))
    
    if not videos:
        logger.error("No videos found in datasets/raw directory")
        return
    
    video_path = videos[0]  # Use first video for testing
    logger.info(f"Testing with video: {video_path.name}")
    
    def progress_callback(video_id, progress, message):
        logger.info(f"[{video_id}] {progress:.1f}%: {message}")
    
    try:
        result = analyzer.analyze_video(str(video_path), progress_callback)
        
        # Print results
        print("\n" + "="*50)
        print(f"VIDEO ANALYSIS RESULTS: {result.video_id}")
        print("="*50)
        print(f"Total Frames: {result.total_frames}")
        print(f"Frames with Detections: {result.frames_with_detections}")
        print(f"Total Person Detections: {result.total_detections}")
        print(f"Unique Persons Tracked: {result.total_tracks}")
        print(f"Average Crowd Density: {result.avg_crowd_density}")
        print(f"Risk Percentage: {result.risk_percentage}%")
        print(f"Risk Level: {result.risk_level}")
        print(f"Processing Time: {result.processing_time_seconds}s")
        
        if result.alerts:
            print(f"\nAlerts ({len(result.alerts)}):")
            for alert in result.alerts:
                print(f"  - {alert}")
        
        print("="*50)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
