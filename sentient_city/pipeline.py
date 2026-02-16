"""
Main processing pipeline.

Orchestrates the complete video processing pipeline from input to alerts.
"""

import asyncio
import cv2
import numpy as np
from typing import Optional, Dict, List
from loguru import logger

from .edge_ai.edge_inference_runner import EdgeDetector, VideoProcessor
from .edge_ai.tracking_engine import OCSortTracker
from .edge_ai.pose_extraction import PoseDetector, FallDetector, PanicDetector
from .streaming.kafka_producer import KafkaProducer
from .streaming.event_router import RedisCache
from .intelligence.risk_engine import RiskEngine
from .memory_engine.vector_store import BehavioralMemory, IdentityMemory
from .explainability.llm_reasoner import AlertGenerator
from .utils.config import get_config


class SentientCityPipeline:
    """
    Main processing pipeline.
    
    Orchestrates:
    - Video input
    - Edge detection
    - Tracking
    - Pose extraction
    - Behavior analysis
    - Risk assessment
    - Alert generation
    """
    
    def __init__(self, camera_source: str, camera_id: Optional[str] = None):
        """
        Initialize pipeline.
        
        Args:
            camera_source: Video source (RTSP URL, file path, or camera index)
            camera_id: Optional camera ID
        """
        self.camera_source = camera_source
        self.camera_id = camera_id or f"camera_{hash(camera_source)}"
        
        # Initialize components
        self.detector = EdgeDetector()
        self.tracker = OCSortTracker()
        self.pose_detector = PoseDetector()
        self.fall_detector = FallDetector()
        self.panic_detector = PanicDetector()
        
        # Streaming
        self.kafka_producer = KafkaProducer()
        self.redis_cache = RedisCache()
        
        # Intelligence
        self.risk_engine = RiskEngine()
        
        # Memory
        self.behavioral_memory = BehavioralMemory()
        self.identity_memory = IdentityMemory()
        
        # Explainability
        self.alert_generator = AlertGenerator()
        
        # Video processor
        self.video_processor = VideoProcessor(
            detector=self.detector,
            source=camera_source
        )
        
        logger.info(f"Pipeline initialized for camera: {self.camera_id}")
    
    async def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process single frame through pipeline.
        
        Args:
            frame: Input frame (BGR)
        
        Returns:
            Processing results dictionary
        """
        results = {
            "camera_id": self.camera_id,
            "frame_id": self.video_processor.frame_count
        }
        
        # 1. Detection
        detection_result = self.detector.detect(frame)
        detections = detection_result.get("detections", [])
        density_map = detection_result.get("crowd_density", {}).get("map")
        behavior_embeddings = detection_result.get("behavior_embeddings")
        
        results["detections"] = detections
        results["density_map"] = density_map
        
        # 2. Tracking
        tracks = self.tracker.update(detections)
        results["tracks"] = [t.to_dict() for t in tracks]
        
        # 3. Pose extraction
        poses = self.pose_detector.detect(frame)
        results["poses"] = poses
        
        # 4. Behavior analysis (fall, panic)
        behavior_results = []
        for pose in poses:
            # Fall detection
            fall_result = self.fall_detector.detect(pose, frame.shape[0])
            if fall_result["is_fall"]:
                behavior_results.append({
                    "type": "fall",
                    "confidence": fall_result["confidence"],
                    "reason": fall_result["reason"]
                })
            
            # Panic detection
            panic_result = self.panic_detector.detect(pose)
            if panic_result["is_panic"]:
                behavior_results.append({
                    "type": "panic",
                    "confidence": panic_result["confidence"],
                    "reason": panic_result["reason"]
                })
        
        results["behaviors"] = behavior_results
        
        # 5. Risk assessment
        risk_assessment = self.risk_engine.assess_risk(
            detections=detections,
            tracks=[t.to_dict() for t in tracks],
            density_map=density_map,
            behavior_embeddings=behavior_embeddings
        )
        results["risk_assessment"] = risk_assessment
        
        # 6. Store in memory
        if behavior_embeddings:
            for i, embedding in enumerate(behavior_embeddings):
                if i < len(tracks):
                    track_id = tracks[i].track_id if tracks else None
                    self.behavioral_memory.store(
                        embedding=np.array(embedding),
                        track_id=track_id,
                        camera_id=self.camera_id
                    )
        
        # 7. Generate alerts if high risk
        if risk_assessment["overall_risk_score"] >= 0.7:
            alert = self.alert_generator.generate_alert(
                risk_assessment,
                camera_id=self.camera_id
            )
            results["alert"] = alert
            
            # Publish alert
            self.kafka_producer.publish_alert(alert, camera_id=self.camera_id)
        
        # 8. Publish to Kafka
        self.kafka_producer.publish_detection(detection_result, camera_id=self.camera_id)
        self.kafka_producer.publish_track(
            {"tracks": results["tracks"]},
            camera_id=self.camera_id
        )
        
        # 9. Cache in Redis
        self.redis_cache.set(
            f"frame:{self.camera_id}:{self.video_processor.frame_count}",
            results,
            ttl=3600
        )
        
        return results
    
    async def run(self):
        """Run pipeline continuously."""
        if not self.video_processor.open():
            logger.error(f"Failed to open video source: {self.camera_source}")
            return
        
        logger.info(f"Starting pipeline for camera: {self.camera_id}")
        
        try:
            for result in self.video_processor.process_stream():
                frame = result.get("frame") or result  # Handle different result formats
                if isinstance(frame, dict):
                    # If result is already processed, extract frame
                    frame_data = frame.get("frame") or frame.get("image")
                    if frame_data is None:
                        continue
                    frame = frame_data
                await self.process_frame(frame)
        except KeyboardInterrupt:
            logger.info("Pipeline stopped by user")
        finally:
            self.video_processor.close()
            self.kafka_producer.close()
            self.redis_cache.close()
    
    def stop(self):
        """Stop pipeline."""
        self.video_processor.stop()


async def main():
    """Main entry point."""
    config = get_config()
    cameras_config = config.get_section("cameras")
    
    # Start pipeline for each camera
    pipelines = []
    for camera in cameras_config:
        if camera.get("enabled", True):
            pipeline = SentientCityPipeline(
                camera_source=camera["source"],
                camera_id=camera["id"]
            )
            pipelines.append(pipeline)
    
    # Run all pipelines concurrently
    await asyncio.gather(*[p.run() for p in pipelines])


if __name__ == "__main__":
    asyncio.run(main())
