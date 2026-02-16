"""
Kafka consumer for event streaming.

Consumes events from Kafka topics for processing.
"""

import json
from typing import Callable, Optional, List
from kafka import KafkaConsumer as KafkaConsumerClient
from kafka.errors import KafkaError
from loguru import logger

from ...utils.config import get_config


class KafkaConsumer:
    """
    Kafka consumer for event streaming.
    
    Handles consuming events from Kafka topics.
    """
    
    def __init__(
        self,
        bootstrap_servers: Optional[str] = None,
        topics: Optional[List[str]] = None,
        consumer_group: Optional[str] = None,
        auto_offset_reset: str = "latest"
    ):
        """
        Initialize Kafka consumer.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers (comma-separated)
            topics: List of topic names to consume from
            consumer_group: Consumer group ID
            auto_offset_reset: Offset reset policy (earliest, latest)
        """
        self.config = get_config()
        streaming_config = self.config.get_section("streaming")
        kafka_config = streaming_config.get("kafka", {})
        
        self.bootstrap_servers = bootstrap_servers or kafka_config.get(
            "bootstrap_servers", "localhost:9092"
        )
        self.topics = topics or list(kafka_config.get("topics", {}).values())
        self.consumer_group = consumer_group or kafka_config.get(
            "consumer_group", "sentient_city_group"
        )
        self.auto_offset_reset = auto_offset_reset
        
        # Initialize consumer
        try:
            self.consumer = KafkaConsumerClient(
                *self.topics,
                bootstrap_servers=self.bootstrap_servers.split(","),
                group_id=self.consumer_group,
                auto_offset_reset=self.auto_offset_reset,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                key_deserializer=lambda k: k.decode("utf-8") if k else None,
                enable_auto_commit=True,
                auto_commit_interval_ms=1000
            )
            logger.info(
                f"Kafka consumer initialized: {self.bootstrap_servers}, "
                f"topics: {self.topics}, group: {self.consumer_group}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Kafka consumer: {e}")
            raise
    
    def consume(
        self,
        callback: Callable[[str, Dict], None],
        timeout_ms: int = 1000
    ):
        """
        Consume messages and call callback.
        
        Args:
            callback: Callback function(topic, message_data)
            timeout_ms: Timeout in milliseconds
        """
        try:
            for message in self.consumer:
                topic = message.topic
                data = message.value
                key = message.key
                
                logger.debug(f"Received message from topic {topic}, key: {key}")
                callback(topic, data)
        except KafkaError as e:
            logger.error(f"Kafka consumer error: {e}")
        except KeyboardInterrupt:
            logger.info("Consumer interrupted")
        finally:
            self.close()
    
    def consume_async(
        self,
        callback: Callable[[str, Dict], None]
    ):
        """
        Consume messages asynchronously.
        
        Args:
            callback: Async callback function(topic, message_data)
        """
        import asyncio
        
        async def async_consume():
            try:
                for message in self.consumer:
                    topic = message.topic
                    data = message.value
                    
                    if asyncio.iscoroutinefunction(callback):
                        await callback(topic, data)
                    else:
                        callback(topic, data)
            except Exception as e:
                logger.error(f"Async consumer error: {e}")
        
        asyncio.run(async_consume())
    
    def close(self):
        """Close consumer."""
        self.consumer.close()
        logger.info("Kafka consumer closed")
