"""
Crowd Analysis Module for Advanced Risk Detection.

This module provides a comprehensive system for detecting and analyzing
risk in crowded environments using advanced computer vision techniques.

Components:
- PersonDetector: YOLO-based person detection optimized for crowds
- PersonTracker: ByteTrack-inspired multi-person tracking
- MotionAnalyzer: Optical flow and motion pattern analysis
- RiskEvaluator: Crowd-aware risk assessment logic
- AlertManager: Dynamic alert triggering with cooldown
- Visualizer: Comprehensive visualization and debugging
- CrowdAnalysisSystem: Main orchestrator class

Example Usage:
    from sentient_city.perception.crowd_analysis import CrowdAnalysisSystem
    
    # Initialize system
    system = CrowdAnalysisSystem()
    
    # Process single frame
    results = system.process_frame(frame)
    
    # Process video stream
    stats = system.process_video_stream("video.mp4", "output.mp4")
"""

from .person_detector import PersonDetector, Detection
from .person_tracker import ByteTrackTracker, TrackedPerson
from .motion_analyzer import MotionAnalyzer, CrowdMotionMetrics, MotionFeatures
from .risk_evaluator import RiskEvaluator, CrowdRisk, PersonRisk, RiskThresholds
from .alert_manager import AlertManager, Alert, AlertConfig
from .visualizer import CrowdVisualizer, VisualizationConfig
from .crowd_analysis_system import CrowdAnalysisSystem, SystemConfig

__version__ = "1.0.0"
__author__ = "Crowd Analysis Team"

__all__ = [
    # Core classes
    "CrowdAnalysisSystem",
    "SystemConfig",
    
    # Detection
    "PersonDetector",
    "Detection",
    
    # Tracking
    "ByteTrackTracker", 
    "TrackedPerson",
    
    # Motion Analysis
    "MotionAnalyzer",
    "CrowdMotionMetrics",
    "MotionFeatures",
    
    # Risk Evaluation
    "RiskEvaluator",
    "CrowdRisk",
    "PersonRisk",
    "RiskThresholds",
    
    # Alert Management
    "AlertManager",
    "Alert",
    "AlertConfig",
    
    # Visualization
    "CrowdVisualizer",
    "VisualizationConfig",
]
