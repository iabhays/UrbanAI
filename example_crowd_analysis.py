#!/usr/bin/env python3
"""
Crowd Analysis System - Example Usage

This script demonstrates the complete refactored crowd analysis system
with proper risk detection in crowded environments.

Features demonstrated:
- Person detection and tracking
- Crowd-aware risk evaluation
- Dynamic alert triggering
- Comprehensive visualization
- Performance monitoring
"""

import cv2
import numpy as np
import argparse
import time
from datetime import datetime
from loguru import logger

from sentient_city.perception.crowd_analysis import (
    CrowdAnalysisSystem,
    SystemConfig,
    AlertConfig,
    VisualizationConfig
)


def setup_logging():
    """Setup logging configuration."""
    logger.remove()
    logger.add(
        "crowd_analysis.log",
        rotation="10 MB",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO",
        format="{time:HH:mm:ss} | {level} | {message}\n"
    )


def custom_alert_callback(alert):
    """Custom alert callback function."""
    print(f"\n🚨 ALERT RECEIVED: {alert.severity}")
    print(f"   Type: {alert.alert_type}")
    print(f"   Message: {alert.message}")
    print(f"   Confidence: {alert.confidence:.2f}")
    print(f"   Crowd Risk Level: {alert.crowd_risk.risk_level}")
    print(f"   Timestamp: {alert.timestamp}")
    print("=" * 50)


def create_test_frame_with_crowd(frame_size=(640, 480), num_people=15):
    """Create a test frame with simulated crowd for testing."""
    frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
    
    # Add background
    frame[:] = (50, 50, 50)  # Dark gray background
    
    # Simulate crowd with random positions
    np.random.seed(42)  # For reproducible results
    
    for i in range(num_people):
        # Random position
        x = np.random.randint(50, frame_size[0] - 50)
        y = np.random.randint(50, frame_size[1] - 50)
        
        # Person size
        w = np.random.randint(20, 40)
        h = np.random.randint(40, 80)
        
        # Draw person (rectangle)
        color = (np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
        
        # Add some movement indication (direction arrow)
        if i % 3 == 0:  # Some people moving
            dx = np.random.randint(-20, 20)
            dy = np.random.randint(-20, 20)
            cv2.arrowedLine(frame, (x + w//2, y + h//2), 
                           (x + w//2 + dx, y + h//2 + dy), 
                           (255, 255, 0), 2)
    
    return frame


def demonstrate_single_frame_processing():
    """Demonstrate single frame processing capabilities."""
    print("\n" + "="*60)
    print("SINGLE FRAME PROCESSING DEMONSTRATION")
    print("="*60)
    
    # Initialize system with debug mode
    config = SystemConfig(
        debug_mode=True,
        enable_alerts=True,
        confidence_threshold=0.4,
        enable_visualization=True,
        enable_gpu=False  # Force CPU for compatibility
    )
    
    system = CrowdAnalysisSystem(
        config=config,
        alert_callbacks=[custom_alert_callback]
    )
    
    # Create test frame with crowd
    frame = create_test_frame_with_crowd(num_people=20)
    
    print(f"Processing test frame with crowd...")
    
    # Process frame
    start_time = time.time()
    results = system.process_frame(frame)
    processing_time = time.time() - start_time
    
    # Display results
    print(f"\n📊 PROCESSING RESULTS:")
    print(f"   Processing Time: {processing_time:.3f} seconds")
    print(f"   FPS: {results.get('fps', 0):.1f}")
    
    if 'detections' in results:
        print(f"   Persons Detected: {len(results['detections'])}")
    
    if 'tracked_persons' in results:
        print(f"   Persons Tracked: {len(results['tracked_persons'])}")
        
        # Show risk distribution
        risk_counts = {"Low": 0, "Medium": 0, "High": 0}
        for person in results['tracked_persons']:
            risk_counts[person.risk_level] += 1
        
        print(f"   Risk Distribution: {risk_counts}")
    
    if 'crowd_risk' in results:
        crowd_risk = results['crowd_risk']
        print(f"   Crowd Risk Level: {crowd_risk.risk_level}")
        print(f"   Crowd Risk Score: {crowd_risk.risk_score:.3f}")
        print(f"   Crowd Size: {crowd_risk.crowd_size}")
    
    if 'new_alerts' in results:
        print(f"   New Alerts: {len(results['new_alerts'])}")
    
    # Save visualized frame
    if results.get('visualized_frame') is not None:
        cv2.imwrite("crowd_analysis_result.jpg", results['visualized_frame'])
        print(f"   Result saved to: crowd_analysis_result.jpg")
    
    # Show performance metrics
    performance = system.get_system_performance()
    print(f"\n⚡ PERFORMANCE METRICS:")
    print(f"   Detector: {performance['detector']['avg_detections_per_frame']:.1f} avg detections/frame")
    print(f"   Tracker: {performance['tracker']['active_tracks']} active tracks")
    print(f"   System FPS: {performance['system']['current_fps']:.1f}")
    
    print("\n✅ Single frame processing completed successfully!")
    
    return system


def demonstrate_video_processing(video_path=None):
    """Demonstrate video stream processing."""
    print("\n" + "="*60)
    print("VIDEO PROCESSING DEMONSTRATION")
    print("="*60)
    
    # Initialize system
    config = SystemConfig(
        debug_mode=False,
        enable_alerts=True,
        confidence_threshold=0.5,
        target_fps=30,
        enable_gpu=False  # Force CPU for compatibility
    )
    
    system = CrowdAnalysisSystem(
        config=config,
        alert_callbacks=[custom_alert_callback]
    )
    
    # Use test video if none provided
    if video_path is None:
        print("No video path provided, creating simulated video stream...")
        # For demo, we'll process a few test frames
        frames = []
        for i in range(100):
            # Create varying crowd sizes
            crowd_size = 10 + i % 20  # Vary between 10-30 people
            frame = create_test_frame_with_crowd(num_people=crowd_size)
            frames.append(frame)
        
        print(f"Processing {len(frames)} simulated frames...")
        
        # Process frames
        start_time = time.time()
        total_alerts = 0
        
        for i, frame in enumerate(frames):
            results = system.process_frame(frame)
            total_alerts += len(results.get('new_alerts', []))
            
            if i % 20 == 0:
                print(f"Processed {i+1}/{len(frames)} frames...")
        
        processing_time = time.time() - start_time
        
        print(f"\n📊 VIDEO PROCESSING RESULTS:")
        print(f"   Total Frames: {len(frames)}")
        print(f"   Processing Time: {processing_time:.2f} seconds")
        print(f"   Average FPS: {len(frames) / processing_time:.1f}")
        print(f"   Total Alerts: {total_alerts}")
        
    else:
        print(f"Processing video: {video_path}")
        try:
            stats = system.process_video_stream(
                video_path=video_path,
                output_path="crowd_analysis_output.mp4",
                show_live=True
            )
            
            print(f"\n📊 VIDEO PROCESSING RESULTS:")
            print(f"   Total Frames: {stats['total_frames']}")
            print(f"   Average FPS: {stats['avg_fps']:.1f}")
            print(f"   Average Processing Time: {stats['avg_processing_time']:.3f} seconds")
            print(f"   Total Alerts: {stats['total_alerts']}")
            
        except Exception as e:
            print(f"❌ Error processing video: {e}")
            return None
    
    # Get final performance metrics
    performance = system.get_system_performance()
    print(f"\n⚡ FINAL PERFORMANCE METRICS:")
    print(f"   Total Frames Processed: {performance['system']['frame_count']}")
    print(f"   Average FPS: {performance['system']['current_fps']:.1f}")
    print(f"   Uptime: {performance['system']['uptime']:.1f} seconds")
    
    if 'alerts' in performance:
        print(f"   Total Alerts Generated: {performance['alerts'].get('total_alerts', 0)}")
        print(f"   False Positive Rate: {performance['alerts'].get('false_positive_rate', 0):.1f}%")
    
    print("\n✅ Video processing completed successfully!")
    
    return system


def demonstrate_risk_scenarios():
    """Demonstrate different risk scenarios."""
    print("\n" + "="*60)
    print("RISK SCENARIO DEMONSTRATION")
    print("="*60)
    
    system = CrowdAnalysisSystem(
        config=SystemConfig(debug_mode=True, enable_gpu=False)
    )
    
    scenarios = [
        {"name": "Small Calm Crowd", "people": 5, "description": "Low risk scenario"},
        {"name": " Medium Crowd", "people": 15, "description": "Normal scenario"},
        {"name": "Large Dense Crowd", "people": 35, "description": "High density scenario"},
        {"name": "Very Large Crowd", "people": 50, "description": "Critical density scenario"}
    ]
    
    for scenario in scenarios:
        print(f"\n🎭 Testing Scenario: {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        print(f"   People Count: {scenario['people']}")
        
        # Create test frame
        frame = create_test_frame_with_crowd(num_people=scenario['people'])
        
        # Process frame
        results = system.process_frame(frame)
        
        # Display results
        if 'crowd_risk' in results:
            crowd_risk = results['crowd_risk']
            print(f"   📊 Results:")
            print(f"      Risk Level: {crowd_risk.risk_level}")
            print(f"      Risk Score: {crowd_risk.risk_score:.3f}")
            print(f"      Crowd Size: {crowd_risk.crowd_size}")
            
            # Show top risk factors
            if crowd_risk.risk_factors:
                top_factors = sorted(crowd_risk.risk_factors.items(), 
                                   key=lambda x: x[1], reverse=True)[:3]
                print(f"      Top Risk Factors: {top_factors}")
        
        if 'new_alerts' in results and results['new_alerts']:
            print(f"      🚨 Alerts Triggered: {len(results['new_alerts'])}")
        
        # Small delay between scenarios
        time.sleep(0.5)
    
    print("\n✅ Risk scenario testing completed!")


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="Crowd Analysis System Demo")
    parser.add_argument("--video", type=str, help="Video file path for processing")
    parser.add_argument("--mode", type=str, choices=["single", "video", "scenarios", "all"],
                       default="all", help="Demo mode to run")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    print("🚀 CROWD ANALYSIS SYSTEM DEMONSTRATION")
    print("=" * 60)
    print("This demo showcases the refactored crowd analysis system")
    print("with advanced risk detection capabilities.")
    print()
    
    try:
        if args.mode in ["single", "all"]:
            demonstrate_single_frame_processing()
        
        if args.mode in ["video", "all"]:
            demonstrate_video_processing(args.video)
        
        if args.mode in ["scenarios", "all"]:
            demonstrate_risk_scenarios()
        
        print("\n" + "="*60)
        print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("✅ Person detection and tracking")
        print("✅ Crowd-aware risk evaluation")
        print("✅ Dynamic alert triggering")
        print("✅ Comprehensive visualization")
        print("✅ Performance monitoring")
        print("✅ Debugging capabilities")
        print("\nThe system is now ready for production use!")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        logger.exception("Demonstration failed")


if __name__ == "__main__":
    main()
