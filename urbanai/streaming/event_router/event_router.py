"""
Event router for distributing events to multiple handlers.

Routes events from Kafka to appropriate processors.
"""

from typing import Dict, Callable, List, Optional
from loguru import logger

from ..kafka_producer.kafka_consumer import KafkaConsumer
from .redis_cache import RedisCache


class EventRouter:
    """
    Event router for distributing events.
    
    Routes events from Kafka to registered handlers.
    """
    
    def __init__(
        self,
        kafka_consumer: Optional[KafkaConsumer] = None,
        redis_cache: Optional[RedisCache] = None
    ):
        """
        Initialize event router.
        
        Args:
            kafka_consumer: Kafka consumer instance
            redis_cache: Redis cache instance
        """
        self.kafka_consumer = kafka_consumer
        self.redis_cache = redis_cache
        self.handlers: Dict[str, List[Callable]] = {}
        self.is_running = False
    
    def register_handler(self, event_type: str, handler: Callable):
        """
        Register handler for event type.
        
        Args:
            event_type: Type of event (detections, tracks, alerts, etc.)
            handler: Handler function(event_data)
        """
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        
        self.handlers[event_type].append(handler)
        logger.info(f"Registered handler for event type: {event_type}")
    
    def route(self, topic: str, data: Dict):
        """
        Route event to appropriate handlers.
        
        Args:
            topic: Kafka topic name
            data: Event data
        """
        # Map topic to event type
        event_type = self._topic_to_event_type(topic)
        
        if event_type not in self.handlers:
            logger.debug(f"No handlers registered for event type: {event_type}")
            return
        
        # Call all registered handlers
        for handler in self.handlers[event_type]:
            try:
                handler(data)
            except Exception as e:
                logger.error(f"Handler error for {event_type}: {e}")
    
    def _topic_to_event_type(self, topic: str) -> str:
        """Map Kafka topic to event type."""
        # Extract event type from topic name
        # Topics are like "detections", "tracks", etc.
        return topic
    
    def start(self):
        """Start routing events."""
        if self.kafka_consumer is None:
            logger.error("Kafka consumer not initialized")
            return
        
        self.is_running = True
        logger.info("Event router started")
        
        # Start consuming
        self.kafka_consumer.consume(self.route)
    
    def stop(self):
        """Stop routing events."""
        self.is_running = False
        if self.kafka_consumer:
            self.kafka_consumer.close()
        logger.info("Event router stopped")
