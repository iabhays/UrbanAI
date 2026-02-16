#!/usr/bin/env python3
"""
Simple Video Analysis with Person Detection and Risk Assessment.

This script processes videos with actual YOLO detection without complex dependencies.
Features:
- Real person detection using YOLOv8
- Bounding box visualization
- Risk percentage calculation
- Person tracking
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import time
from dataclasses import dataclass
from loguru import logger

try:
    from ultralytics import YOLO
except ImportError:
    logger.error("ultralytics not installed. Run: pip install ultralytics")
    exit(1)


@dataclass
class VideoAnalysisResult:
    """Results from video analysis."""
    video_id: str
    total_frames: int
    total_detections: int
    unique_tracks: int
    avg_crowd_density: float
    risk_percentage: float
    risk_level: str
    alerts: List[str]
    processing_time_seconds: float
    frames_with_detections: int


class SimpleTrack:
    """Simple track representation."""
    
    def __init__(self, track_id: int, bbox: np.ndarray, confidence: float):
        self.track_id = track_id
        self.bbox = bbox
        self.confidence = confidence
        self.last_seen = time.time()


class SimpleTracker:
    """Simple person tracker using IoU matching."""
    
    def __init__(self, max_disappeared: int = 30):
        self.tracks = []
        self.next_id = 1
        self.max_disappeared = max_disappeared
    
    def update(self, detections: List[Dict]) -> List[SimpleTrack]:
        """Update tracks with new detections."""
        if not detections:
            # Mark all tracks as disappeared
            for track in self.tracks:
                track.last_seen = time.time()
            return self.tracks
        
        # Convert detections to format
        det_bboxes = [np.array(d['bbox']) for d in detections]
        det_confs = [d['confidence'] for d in detections]
        
        # Simple IoU matching with existing tracks
        matched_tracks = []
        matched_detections = []
        
        if self.tracks:
            track_bboxes = [track.bbox for track in self.tracks]
            
            for i, det_bbox in enumerate(det_bboxes):
                best_iou = 0
                best_track_idx = -1
                
                for j, track_bbox in enumerate(track_bboxes):
                    iou = self._calculate_iou(det_bbox, track_bbox)
                    if iou > best_iou and iou > 0.3:  # IoU threshold
                        best_iou = iou
                        best_track_idx = j
                
                if best_track_idx >= 0:
                    # Update existing track
                    track = self.tracks[best_track_idx]
                    track.bbox = det_bbox
                    track.confidence = det_confs[i]
                    track.last_seen = time.time()
                    matched_tracks.append(track)
                    matched_detections.append(i)
        
        # Create new tracks for unmatched detections
        for i, det_bbox in enumerate(det_bboxes):
            if i not in matched_detections:
                new_track = SimpleTrack(self.next_id, det_bbox, det_confs[i])
                self.tracks.append(new_track)
                self.next_id += 1
        
        # Remove old tracks
        current_time = time.time()
        self.tracks = [t for t in self.tracks 
                     if current_time - t.last_seen < self.max_disappeared]
        
        return self.tracks
    
    def _calculate_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate IoU between two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


class SimpleVideoAnalyzer:
    """Simple video analyzer with YOLO detection."""
    
    def __init__(self):
        """Initialize the analyzer."""
        logger.info("Loading YOLOv8 model...")
        self.yolo_model = YOLO('yolov8n.pt')
        self.tracker = SimpleTracker()
        
    def calculate_risk_percentage(
        self, 
        detections_count: int, 
        crowd_density: float, 
        alerts_count: int,
        total_frames: int
    ) -> Tuple[float, str]:
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
        people_risk = min(detections_count / max(total_frames, 1) * 2, 30)  # Max 30% from people count
        
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
    
    def detect_persons(self, frame: np.ndarray) -> List[Dict]:
        """Detect persons in frame using YOLO."""
        results = self.yolo_model(frame)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Filter for persons only (class 0)
                    if int(box.cls) == 0:  # Person class
                        confidence = float(box.conf)
                        if confidence > 0.5:  # Confidence threshold
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            detections.append({
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'confidence': confidence,
                                'class': 'person',
                                'class_id': 0
                            })
        
        return detections
    
    def draw_bounding_boxes(self, frame: np.ndarray, detections: List[Dict], tracks: List[SimpleTrack]) -> np.ndarray:
        """Draw bounding boxes and tracking info on frame."""
        processed_frame = frame.copy()
        
        # Draw detections
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Choose color based on confidence
            if confidence > 0.8:
                color = (0, 255, 0)  # Green - high confidence
            elif confidence > 0.6:
                color = (0, 255, 255)  # Yellow - medium confidence
            else:
                color = (0, 0, 255)  # Red - low confidence
            
            # Draw rectangle
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"Person: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Background for label
            cv2.rectangle(processed_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Text
            cv2.putText(processed_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw tracking IDs
        for track in tracks:
            x1, y1, x2, y2 = map(int, track.bbox)
            
            # Draw track ID
            id_label = f"ID: {track.track_id}"
            cv2.putText(processed_frame, id_label, (x1, y2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return processed_frame
    
    def analyze_video(self, video_path: str, progress_callback=None, save_output: bool = False) -> VideoAnalysisResult:
        """
        Analyze a video file with real detection.
        
        Args:
            video_path: Path to video file
            progress_callback: Optional callback for progress updates
            save_output: Whether to save processed video
        
        Returns:
            VideoAnalysisResult with comprehensive metrics
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        video_id = video_path.stem
        
        logger.info(f"Starting analysis of video: {video_id}")
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
        
        # Setup video writer if saving output
        video_writer = None
        if save_output:
            output_path = video_path.parent / f"{video_id}_processed.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Analysis metrics
        total_detections = 0
        all_track_ids = set()
        crowd_densities = []
        all_alerts = []
        frames_with_detections = 0
        
        # Process every 5th frame for efficiency
        frame_skip = 5
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
                
                # Detect persons
                detections = self.detect_persons(frame)
                
                # Update tracker
                tracks = self.tracker.update(detections)
                
                # Draw bounding boxes
                processed_frame = self.draw_bounding_boxes(frame, detections, tracks)
                
                # Save frame if requested
                if video_writer:
                    video_writer.write(processed_frame)
                
                # Extract metrics
                if detections:
                    frames_with_detections += 1
                    total_detections += len(detections)
                
                # Track unique persons
                for track in tracks:
                    all_track_ids.add(track.track_id)
                
                # Calculate crowd density for this frame
                frame_area = width * height
                person_area = 0
                for detection in detections:
                    bbox = detection['bbox']
                    x1, y1, x2, y2 = bbox
                    person_area += (x2 - x1) * (y2 - y1)
                
                density = min(person_area / frame_area, 1.0) if frame_area > 0 else 0
                crowd_densities.append(density)
                
                # Simple alert generation (high crowd density)
                if density > 0.3:  # 30% coverage
                    all_alerts.append("High crowd density detected")
                
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
            if video_writer:
                video_writer.release()
        
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
            unique_tracks=len(all_track_ids),
            avg_crowd_density=round(avg_crowd_density, 3),
            risk_percentage=risk_percentage,
            risk_level=risk_level,
            alerts=list(set(all_alerts))[:10],  # Unique alerts, limit to 10
            processing_time_seconds=round(processing_time, 2),
            frames_with_detections=frames_with_detections
        )
        
        logger.info(f"Analysis complete for {video_id}: "
                   f"{total_detections} detections, {risk_percentage}% risk")
        
        return result


def main():
    """Main function for standalone testing."""
    analyzer = SimpleVideoAnalyzer()
    
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
        result = analyzer.analyze_video(str(video_path), progress_callback, save_output=True)
        
        # Print results
        print("\n" + "="*50)
        print(f"VIDEO ANALYSIS RESULTS: {result.video_id}")
        print("="*50)
        print(f"Total Frames: {result.total_frames}")
        print(f"Frames with Detections: {result.frames_with_detections}")
        print(f"Total Person Detections: {result.total_detections}")
        print(f"Unique Persons Tracked: {result.unique_tracks}")
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
