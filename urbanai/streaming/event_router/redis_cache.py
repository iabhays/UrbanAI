"""
Redis cache for fast data access.

Provides caching layer for frequently accessed data.
"""

import json
from typing import Optional, Any, Dict, List
import redis
import aioredis
from loguru import logger

from ...utils.config import get_config


class RedisCache:
    """
    Redis cache interface.
    
    Provides caching for detections, tracks, and other data.
    """
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db: int = 0,
        password: Optional[str] = None,
        ttl: int = 3600
    ):
        """
        Initialize Redis cache.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            ttl: Default TTL in seconds
        """
        self.config = get_config()
        streaming_config = self.config.get_section("streaming")
        redis_config = streaming_config.get("redis", {})
        
        self.host = host or redis_config.get("host", "localhost")
        self.port = port or redis_config.get("port", 6379)
        self.db = db or redis_config.get("db", 0)
        self.password = password or redis_config.get("password")
        self.ttl = ttl or redis_config.get("ttl", 3600)
        
        # Initialize synchronous Redis client
        try:
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"Redis cache initialized: {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {e}")
            raise
        
        # Initialize async Redis client
        self._async_client: Optional[aioredis.Redis] = None
    
    async def get_async_client(self) -> aioredis.Redis:
        """Get or create async Redis client."""
        if self._async_client is None:
            self._async_client = await aioredis.from_url(
                f"redis://{self.host}:{self.port}/{self.db}",
                password=self.password,
                decode_responses=True
            )
        return self._async_client
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized)
            ttl: TTL in seconds (uses default if None)
        
        Returns:
            True if successful
        """
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            ttl = ttl or self.ttl
            self.redis_client.setex(key, ttl, value)
            return True
        except Exception as e:
            logger.error(f"Failed to set cache key {key}: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None
        """
        try:
            value = self.redis_client.get(key)
            if value is None:
                return None
            
            # Try to parse as JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        except Exception as e:
            logger.error(f"Failed to get cache key {key}: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key
        
        Returns:
            True if deleted
        """
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            logger.error(f"Failed to delete cache key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """
        Check if key exists.
        
        Args:
            key: Cache key
        
        Returns:
            True if exists
        """
        try:
            return bool(self.redis_client.exists(key))
        except Exception as e:
            logger.error(f"Failed to check cache key {key}: {e}")
            return False
    
    def set_hash(
        self,
        key: str,
        mapping: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set hash in cache.
        
        Args:
            key: Cache key
            mapping: Dictionary to store
            ttl: TTL in seconds
        
        Returns:
            True if successful
        """
        try:
            # Serialize values
            serialized = {
                k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                for k, v in mapping.items()
            }
            
            self.redis_client.hset(key, mapping=serialized)
            
            if ttl:
                self.redis_client.expire(key, ttl)
            
            return True
        except Exception as e:
            logger.error(f"Failed to set hash {key}: {e}")
            return False
    
    def get_hash(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get hash from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Dictionary or None
        """
        try:
            data = self.redis_client.hgetall(key)
            if not data:
                return None
            
            # Deserialize values
            result = {}
            for k, v in data.items():
                try:
                    result[k] = json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    result[k] = v
            
            return result
        except Exception as e:
            logger.error(f"Failed to get hash {key}: {e}")
            return None
    
    def add_to_list(self, key: str, value: Any, max_length: Optional[int] = None) -> bool:
        """
        Add value to list (left push).
        
        Args:
            key: Cache key
            value: Value to add
            max_length: Maximum list length (FIFO)
        
        Returns:
            True if successful
        """
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            self.redis_client.lpush(key, value)
            
            if max_length:
                self.redis_client.ltrim(key, 0, max_length - 1)
            
            return True
        except Exception as e:
            logger.error(f"Failed to add to list {key}: {e}")
            return False
    
    def get_list(self, key: str, start: int = 0, end: int = -1) -> List[Any]:
        """
        Get list from cache.
        
        Args:
            key: Cache key
            start: Start index
            end: End index
        
        Returns:
            List of values
        """
        try:
            values = self.redis_client.lrange(key, start, end)
            
            # Try to deserialize
            result = []
            for v in values:
                try:
                    result.append(json.loads(v))
                except (json.JSONDecodeError, TypeError):
                    result.append(v)
            
            return result
        except Exception as e:
            logger.error(f"Failed to get list {key}: {e}")
            return []
    
    def close(self):
        """Close Redis connection."""
        self.redis_client.close()
        logger.info("Redis cache closed")
    
    async def close_async(self):
        """Close async Redis connection."""
        if self._async_client:
            await self._async_client.close()
            self._async_client = None
