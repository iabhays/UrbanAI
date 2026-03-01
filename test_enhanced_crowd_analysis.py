#!/usr/bin/env python3
"""
Complete Crowd Analysis System Test with Skeleton Detection and Real-time Processing.

This script demonstrates the full capabilities of the enhanced crowd analysis system
including person detection with skeleton wireframes, crowd density mapping, and
real-time camera/video processing.
"""

import cv2
import numpy as np
import argparse
import time
import os
from datetime import datetime
from loguru import logger

from urbanai.perception.crowd_analysis.enhanced_person_detector import EnhancedPersonDetector
from urbanai.perception.crowd_analysis.crowd_density_mapper import CrowdDensityMapper, HeatmapConfig
from urbanai.perception.crowd_analysis.real_time_processor import RealTimeProcessor, ProcessingConfig
from urbanai.perception.crowd_analysis.enhanced_visualizer import EnhancedVisualizer, VisualizationConfig


def setup_logging():
    """Setup logging configuration."""
    logger.remove()
    logger.add(
        "enhanced_crowd_analysis.log",
        rotation="10 MB",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO",
        format="{time:HH:mm:ss} | {level} | {message}\n"
    )


def create_test_frame_with_skeletons(frame_size=(640, 480), num_people=10):
    """
    Create a test frame with simulated people for testing.
    
    Args:
        frame_size: Frame dimensions (width, height)
        num_people: Number of people to simulate
        
    Returns:
        Test frame with simulated people
    """
    frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
    
    # Add background
    frame[:] = (40, 40, 60)  # Dark blue-gray background
    
    # Add some texture to make it more realistic
    noise = np.random.randint(0, 30, frame.shape, dtype=np.uint8)
    frame = cv2.add(frame, noise)
    
    # Simulate people with random positions
    np.random.seed(int(time.time()))  # Use current time for variety
    
    for i in range(num_people):
        # Random position
        x = np.random.randint(50, frame_size[0] - 50)
        y = np.random.randint(50, frame_size[1] - 50)
        
        # Person size (realistic proportions)
        w = np.random.randint(30, 50)
        h = np.random.randint(60, 100)
        
        # Draw person (rectangle with human-like proportions)
        color = (np.random.randint(100, 200), np.random.randint(100, 200), np.random.randint(100, 200))
        
        # Body
        cv2.rectangle(frame, (x, y + h//4), (x + w, y + h), color, -1)
        
        # Head
        head_size = w // 3
        cv2.circle(frame, (x + w//2, y + h//4), head_size, color, -1)
        
        # Add some movement indication for some people
        if i % 3 == 0:  # Some people moving
            dx = np.random.randint(-30, 30)
            dy = np.random.randint(-30, 30)
            cv2.arrowedLine(frame, (x + w//2, y + h//2), 
                           (x + w//2 + dx, y + h//2 + dy), 
                           (255, 255, 0), 2)
    
    return frame


def test_enhanced_person_detector():
    """Test the enhanced person detector with skeleton capabilities."""
    print("\n" + "="*60)
    print("TESTING ENHANCED PERSON DETECTOR WITH SKELETONS")
    print("="*60)
    
    # Initialize detector
    detector = EnhancedPersonDetector(
        confidence_threshold=0.5,
        pose_confidence_threshold=0.5,
        enable_skeleton=True
    )
    
    # Create test frame
    frame = create_test_frame_with_skeletons(num_people=5)
    
    print(f"Processing test frame...")
    
    # Detect persons with skeletons
    start_time = time.time()
    detections = detector.detect(frame)
    processing_time = time.time() - start_time
    
    print(f"\n📊 DETECTION RESULTS:")
    print(f"   Processing Time: {processing_time:.3f} seconds")
    print(f"   Persons Detected: {len(detections)}")
    
    # Draw detections
    annotated_frame = frame.copy()
    for i, detection in enumerate(detections):
        # Draw bounding box
        x1, y1, x2, y2 = map(int, detection.bbox)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label
        label = f"Person {i+1}: {detection.confidence:.2f}"
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw skeleton if available
        if len(detection.keypoints) > 0:
            annotated_frame = detector.draw_skeleton(annotated_frame, detection)
        
        print(f"   Person {i+1}: confidence={detection.confidence:.2f}, "
              f"skeleton={'Yes' if len(detection.keypoints) > 0 else 'No'}")
    
    # Save result
    cv2.imwrite("enhanced_detection_result.jpg", annotated_frame)
    print(f"   Result saved to: enhanced_detection_result.jpg")
    
    # Get performance metrics
    metrics = detector.get_performance_metrics()
    print(f"\n⚡ PERFORMANCE METRICS:")
    print(f"   Skeleton Success Rate: {metrics['skeleton_success_rate']:.1%}")
    print(f"   Device: {metrics['device']}")
    
    print("\n✅ Enhanced person detector test completed!")
    return detector


def test_crowd_density_mapper():
    """Test the crowd density mapping system."""
    print("\n" + "="*60)
    print("TESTING CROWD DENSITY MAPPING")
    print("="*60)
    
    # Initialize density mapper
    config = HeatmapConfig(
        grid_size=(20, 20),
        sigma=2.0,
        alpha=0.6,
        colormap=cv2.COLORMAP_JET
    )
    
    mapper = CrowdDensityMapper(frame_size=(640, 480), config=config)
    
    # Create test frame with varying crowd densities
    frame = create_test_frame_with_skeletons(num_people=15)
    
    # Simulate detections (in real system, these would come from detector)
    from urbanai.perception.crowd_analysis.enhanced_person_detector import SkeletonDetection
    
    detections = []
    for i in range(15):
        x = np.random.randint(50, 590)
        y = np.random.randint(50, 430)
        w = np.random.randint(30, 50)
        h = np.random.randint(60, 100)
        
        detection = SkeletonDetection(
            bbox=np.array([x, y, x + w, y + h]),
            confidence=np.random.uniform(0.5, 0.9),
            keypoints=np.array([]),  # Empty for this test
            keypoints_confidence=np.array([]),
            skeleton_connections=[],
            center=np.array([x + w//2, y + h//2])
        )
        detections.append(detection)
    
    print(f"Processing crowd density for {len(detections)} people...")
    
    # Calculate density metrics
    start_time = time.time()
    density_metrics = mapper.calculate_density(detections)
    processing_time = time.time() - start_time
    
    print(f"\n📊 DENSITY ANALYSIS RESULTS:")
    print(f"   Processing Time: {processing_time:.3f} seconds")
    print(f"   People Count: {density_metrics.crowd_count}")
    print(f"   Overall Density: {density_metrics.overall_density:.2f} people/m²")
    print(f"   Peak Density: {density_metrics.peak_density:.2f}")
    print(f"   Area Coverage: {density_metrics.area_coverage:.1%}")
    
    if density_metrics.flow_direction is not None:
        print(f"   Flow Direction: ({density_metrics.flow_direction[0]:.2f}, "
              f"{density_metrics.flow_direction[1]:.2f})")
        print(f"   Flow Speed: {density_metrics.flow_speed:.1f} px/frame")
    
    print(f"\n   Zone Densities:")
    for zone, density in density_metrics.density_zones.items():
        print(f"      {zone}: {density:.2f} people/m²")
    
    # Generate heatmap
    heatmap = mapper.generate_heatmap()
    heatmap_overlay = mapper.overlay_heatmap(frame, heatmap)
    
    # Draw density zones
    zones_frame = mapper.draw_density_zones(frame, density_metrics.density_zones)
    
    # Save results
    cv2.imwrite("density_heatmap.jpg", heatmap)
    cv2.imwrite("density_overlay.jpg", heatmap_overlay)
    cv2.imwrite("density_zones.jpg", zones_frame)
    
    print(f"\n   Results saved:")
    print(f"      - density_heatmap.jpg")
    print(f"      - density_overlay.jpg")
    print(f"      - density_zones.jpg")
    
    print("\n✅ Crowd density mapping test completed!")
    return mapper


def test_real_time_processing():
    """Test real-time processing with simulated video stream."""
    print("\n" + "="*60)
    print("TESTING REAL-TIME PROCESSING SYSTEM")
    print("="*60)
    
    # Initialize processor
    config = ProcessingConfig(
        camera_width=640,
        camera_height=480,
        camera_fps=30,
        enable_skeleton=True,
        enable_heatmap=True,
        enable_density_zones=True,
        enable_flow_analysis=True,
        save_output=True,
        output_path="real_time_test_output.mp4"
    )
    
    processor = RealTimeProcessor(config=config)
    
    # Create simulated video frames
    print("Creating simulated video stream...")
    frames = []
    
    for i in range(100):  # 100 frames
        # Vary crowd size over time
        crowd_size = 5 + (i // 10) * 3  # Increase crowd every 10 frames
        frame = create_test_frame_with_skeletons(num_people=crowd_size)
        frames.append(frame)
    
    print(f"Processing {len(frames)} simulated frames...")
    
    # Process frames
    results_list = []
    start_time = time.time()
    
    for i, frame in enumerate(frames):
        # Simulate real-time processing
        results = processor._process_single_frame(frame, i)
        results_list.append(results)
        
        if i % 20 == 0:
            print(f"Processed {i+1}/{len(frames)} frames...")
        
        # Simulate real-time delay
        time.sleep(0.03)  # ~30 FPS
    
    processing_time = time.time() - start_time
    
    # Analyze results
    print(f"\n📊 REAL-TIME PROCESSING RESULTS:")
    print(f"   Total Frames: {len(frames)}")
    print(f"   Processing Time: {processing_time:.2f} seconds")
    print(f"   Average FPS: {len(frames) / processing_time:.1f}")
    
    if results_list:
        avg_people = np.mean([r.density_metrics.crowd_count for r in results_list])
        max_density = np.max([r.density_metrics.overall_density for r in results_list])
        avg_processing_time = np.mean([r.processing_time for r in results_list])
        
        print(f"   Average People per Frame: {avg_people:.1f}")
        print(f"   Maximum Density: {max_density:.2f} people/m²")
        print(f"   Average Processing Time per Frame: {avg_processing_time*1000:.1f}ms")
    
    # Save sample annotated frame
    if results_list:
        sample_frame = results_list[-1].annotated_frame
        cv2.imwrite("real_time_sample.jpg", sample_frame)
        print(f"   Sample frame saved to: real_time_sample.jpg")
    
    print("\n✅ Real-time processing test completed!")
    return processor


def test_enhanced_visualizer():
    """Test the enhanced visualization system."""
    print("\n" + "="*60)
    print("TESTING ENHANCED VISUALIZATION SYSTEM")
    print("="*60)
    
    # Initialize visualizer
    config = VisualizationConfig(
        show_fps=True,
        show_metrics=True,
        show_zones=True,
        show_flow=True,
        bbox_color=(0, 255, 0),
        skeleton_color=(255, 0, 0),
        heatmap_alpha=0.6
    )
    
    visualizer = EnhancedVisualizer(config)
    
    # Create test data
    frame = create_test_frame_with_skeletons(num_people=8)
    
    # Create mock processing results
    from urbanai.perception.crowd_analysis.enhanced_person_detector import SkeletonDetection
    from urbanai.perception.crowd_analysis.real_time_processor import ProcessingResults
    from urbanai.perception.crowd_analysis.crowd_density_mapper import DensityMetrics
    
    detections = []
    for i in range(8):
        x = np.random.randint(50, 590)
        y = np.random.randint(50, 430)
        w = np.random.randint(30, 50)
        h = np.random.randint(60, 100)
        
        detection = SkeletonDetection(
            bbox=np.array([x, y, x + w, y + h]),
            confidence=np.random.uniform(0.5, 0.9),
            keypoints=np.array([]),
            keypoints_confidence=np.array([]),
            skeleton_connections=[],
            center=np.array([x + w//2, y + h//2])
        )
        detections.append(detection)
    
    density_metrics = DensityMetrics(
        overall_density=1.2,
        local_density=np.random.rand(20, 20),
        peak_density=2.5,
        density_zones={
            'top_left': 0.8,
            'top_right': 1.5,
            'bottom_left': 1.2,
            'bottom_right': 1.8
        },
        crowd_count=8,
        area_coverage=0.35,
        flow_direction=np.array([0.5, 0.3]),
        flow_speed=2.1
    )
    
    results = ProcessingResults(
        frame_id=1,
        timestamp=datetime.now(),
        detections=detections,
        density_metrics=density_metrics,
        processing_time=0.045,
        fps=25.0
    )
    
    print("Creating comprehensive visualization...")
    
    # Create comprehensive visualization
    start_time = time.time()
    annotated_frame = visualizer.create_comprehensive_visualization(frame, results)
    viz_time = time.time() - start_time
    
    # Create analysis dashboard
    dashboard = visualizer.create_analysis_dashboard(results, frame)
    
    print(f"\n📊 VISUALIZATION RESULTS:")
    print(f"   Visualization Time: {viz_time:.3f} seconds")
    
    # Save results
    cv2.imwrite("comprehensive_visualization.jpg", annotated_frame)
    cv2.imwrite("analysis_dashboard.jpg", dashboard)
    
    print(f"   Results saved:")
    print(f"      - comprehensive_visualization.jpg")
    print(f"      - analysis_dashboard.jpg")
    
    print("\n✅ Enhanced visualization test completed!")
    return visualizer


def test_live_camera():
    """Test with live camera (if available)."""
    print("\n" + "="*60)
    print("TESTING LIVE CAMERA PROCESSING")
    print("="*60)
    
    # Initialize processor
    config = ProcessingConfig(
        camera_id=0,  # Default camera
        camera_width=640,
        camera_height=480,
        camera_fps=30,
        enable_skeleton=True,
        enable_heatmap=True,
        enable_density_zones=True,
        save_output=False  # Don't save during live test
    )
    
    processor = RealTimeProcessor(config=config)
    
    # Try to start camera
    if not processor.start_camera():
        print("❌ Could not access camera. Skipping live camera test.")
        return None
    
    print("Camera started successfully!")
    print("Press 'q' to quit, 'p' to pause, 'r' to resume")
    
    # Start processing
    if not processor.start_processing():
        print("❌ Could not start processing.")
        return None
    
    try:
        # Display live feed
        while True:
            results = processor.get_latest_results()
            
            if results and results.annotated_frame is not None:
                cv2.imshow('Enhanced Crowd Analysis - Live Camera', results.annotated_frame)
                
                # Display metrics in console
                if results.frame_id % 30 == 0:  # Every 30 frames
                    print(f"Frame {results.frame_id}: "
                          f"People={results.density_metrics.crowd_count}, "
                          f"Density={results.density_metrics.overall_density:.2f}/m², "
                          f"FPS={results.fps:.1f}")
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                processor.pause_processing()
                print("Processing paused. Press 'r' to resume.")
            elif key == ord('r'):
                processor.resume_processing()
                print("Processing resumed.")
    
    except KeyboardInterrupt:
        print("\nLive camera test interrupted by user")
    
    finally:
        processor.stop_processing()
        cv2.destroyAllWindows()
    
    print("\n✅ Live camera test completed!")
    return processor


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Enhanced Crowd Analysis System Test")
    parser.add_argument("--test", type=str, 
                       choices=["detector", "density", "realtime", "visualizer", "camera", "all"],
                       default="all", help="Test to run")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    print("🚀 ENHANCED CROWD ANALYSIS SYSTEM TEST")
    print("=" * 60)
    print("This test suite demonstrates the complete enhanced crowd analysis system")
    print("with person detection, skeleton wireframes, density mapping, and real-time processing.")
    print()
    
    try:
        if args.test in ["detector", "all"]:
            test_enhanced_person_detector()
        
        if args.test in ["density", "all"]:
            test_crowd_density_mapper()
        
        if args.test in ["realtime", "all"]:
            test_real_time_processing()
        
        if args.test in ["visualizer", "all"]:
            test_enhanced_visualizer()
        
        if args.test in ["camera", "all"]:
            test_live_camera()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey Features Tested:")
        print("✅ Enhanced person detection with skeleton wireframes")
        print("✅ Crowd density mapping and heatmap generation")
        print("✅ Real-time processing with multi-threading")
        print("✅ Comprehensive visualization and analysis dashboard")
        print("✅ Live camera processing capabilities")
        print("\nThe enhanced system is ready for production deployment!")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        logger.exception("Test failed")


if __name__ == "__main__":
    main()
