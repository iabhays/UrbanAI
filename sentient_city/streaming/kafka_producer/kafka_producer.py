"""
Kafka producer for event streaming.

Publishes detection, tracking, and alert events to Kafka topics.
"""

import json
from typing import Dict, Any, Optional
from kafka import KafkaProducer as KafkaProducerClient
from kafka.errors import KafkaError
from loguru import logger

from ...utils.config import get_config


class KafkaProducer:
    """
    Kafka producer for event streaming.
    
    Handles publishing events to Kafka topics.
    """
    
    def __init__(
        self,
        bootstrap_servers: Optional[str] = None,
        topics: Optional[Dict[str, str]] = None
    ):
        """
        Initialize Kafka producer.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers (comma-separated)
            topics: Dictionary mapping event types to topic names
        """
        self.config = get_config()
        streaming_config = self.config.get_section("streaming")
        kafka_config = streaming_config.get("kafka", {})
        
        self.bootstrap_servers = bootstrap_servers or kafka_config.get(
            "bootstrap_servers", "localhost:9092"
        )
        self.topics = topics or kafka_config.get("topics", {})
        
        # Initialize producer
        try:
            self.producer = KafkaProducerClient(
                bootstrap_servers=self.bootstrap_servers.split(","),
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks="all",
                retries=3,
                max_in_flight_requests_per_connection=1
            )
            logger.info(f"Kafka producer initialized: {self.bootstrap_servers}")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            raise
        
        # Event schema registry
        self.schema_registry = EventSchemaRegistry()
        
        # Event schema registry
        self.schema_registry = EventSchemaRegistry()
    
    def publish(
        self,
        event_type: str,
        data: Dict[str, Any],
        key: Optional[str] = None
    ) -> bool:
        """
        Publish event to Kafka topic.
        
        Args:
            event_type: Type of event (detections, tracks, alerts, etc.)
            data: Event data dictionary
            key: Optional partition key
        
        Returns:
            True if published successfully
        """
        topic = self.topics.get(event_type)
        if topic is None:
            logger.warning(f"Unknown event type: {event_type}")
            return False
        
        try:
            future = self.producer.send(topic, value=data, key=key)
            # Wait for acknowledgment
            record_metadata = future.get(timeout=10)
            logger.debug(
                f"Published to topic {record_metadata.topic} "
                f"partition {record_metadata.partition} "
                f"offset {record_metadata.offset}"
            )
            return True
        except KafkaError as e:
            logger.error(f"Failed to publish event: {e}")
            return False
    
    def publish_detection(self, detection_data: Dict[str, Any], camera_id: Optional[str] = None):
        """Publish detection event."""
        return self.publish("detections", detection_data, key=camera_id)
    
    def publish_track(self, track_data: Dict[str, Any], camera_id: Optional[str] = None):
        """Publish track event."""
        return self.publish("tracks", track_data, key=camera_id)
    
    def publish_pose(self, pose_data: Dict[str, Any], camera_id: Optional[str] = None):
        """Publish pose event."""
        return self.publish("poses", pose_data, key=camera_id)
    
    def publish_behavior(self, behavior_data: Dict[str, Any], camera_id: Optional[str] = None):
        """Publish behavior event."""
        return self.publish("behaviors", behavior_data, key=camera_id)
    
    def publish_alert(self, alert_data: Dict[str, Any], camera_id: Optional[str] = None):
        """Publish alert event."""
        return self.publish("alerts", alert_data, key=camera_id)
    
    def flush(self):
        """Flush pending messages."""
        self.producer.flush()
    
    def close(self):
        """Close producer."""
        self.producer.close()
        logger.info("Kafka producer closed")
