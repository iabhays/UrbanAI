"""
Behavioral memory storage.

Stores and retrieves behavioral patterns and history.
"""

import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from loguru import logger

from .vector_db import VectorDatabase
from ...utils.config import get_config


class BehavioralMemory:
    """
    Behavioral memory storage.
    
    Stores behavioral embeddings and patterns for analysis.
    """
    
    def __init__(self, vector_db: Optional[VectorDatabase] = None):
        """
        Initialize behavioral memory.
        
        Args:
            vector_db: Vector database instance
        """
        self.config = get_config()
        memory_config = self.config.get_section("memory")
        storage_config = memory_config.get("storage", {})
        
        self.ttl = storage_config.get("behavioral_history_ttl", 86400)  # 24 hours
        
        self.vector_db = vector_db or VectorDatabase(
            dimension=512,
            index_type="IVF_FLAT"
        )
        
        # In-memory storage for metadata
        self.metadata: Dict[int, Dict] = {}
        self.next_id = 0
    
    def store(
        self,
        embedding: np.ndarray,
        track_id: Optional[int] = None,
        camera_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Store behavioral embedding.
        
        Args:
            embedding: Behavior embedding vector
            track_id: Optional track ID
            camera_id: Optional camera ID
            timestamp: Optional timestamp
            metadata: Optional metadata dictionary
        
        Returns:
            Stored ID
        """
        if timestamp is None:
            timestamp = datetime.now().timestamp()
        
        # Add to vector database
        self.vector_db.add(embedding.reshape(1, -1))
        
        # Store metadata
        storage_id = self.next_id
        self.metadata[storage_id] = {
            "track_id": track_id,
            "camera_id": camera_id,
            "timestamp": timestamp,
            "expires_at": timestamp + self.ttl,
            "metadata": metadata or {}
        }
        
        self.next_id += 1
        
        logger.debug(f"Stored behavioral embedding: ID={storage_id}, track_id={track_id}")
        
        return storage_id
    
    def search_similar(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        min_similarity: float = 0.7
    ) -> List[Dict]:
        """
        Search for similar behaviors.
        
        Args:
            query_embedding: Query embedding
            k: Number of results
            min_similarity: Minimum similarity threshold
        
        Returns:
            List of similar behavior records
        """
        # Search in vector database
        distances, indices = self.vector_db.search(query_embedding, k=k)
        
        # Convert distances to similarities (for L2 metric)
        similarities = 1.0 / (1.0 + distances[0])
        
        # Filter by similarity and get metadata
        results = []
        for idx, sim in zip(indices[0], similarities):
            if sim >= min_similarity and idx in self.metadata:
                result = self.metadata[idx].copy()
                result["similarity"] = float(sim)
                result["index"] = int(idx)
                results.append(result)
        
        return results
    
    def get_by_track(
        self,
        track_id: int,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Get behaviors by track ID.
        
        Args:
            track_id: Track ID
            limit: Optional limit
        
        Returns:
            List of behavior records
        """
        results = [
            meta.copy() for meta in self.metadata.values()
            if meta.get("track_id") == track_id
        ]
        
        # Sort by timestamp
        results.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        
        if limit:
            results = results[:limit]
        
        return results
    
    def cleanup_expired(self):
        """Remove expired entries."""
        current_time = datetime.now().timestamp()
        
        expired_ids = [
            idx for idx, meta in self.metadata.items()
            if meta.get("expires_at", 0) < current_time
        ]
        
        for idx in expired_ids:
            del self.metadata[idx]
        
        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired behavioral records")
    
    def get_statistics(self) -> Dict:
        """Get memory statistics."""
        return {
            "total_vectors": self.vector_db.get_size(),
            "total_metadata": len(self.metadata),
            "unique_tracks": len(set(
                meta.get("track_id") for meta in self.metadata.values()
                if meta.get("track_id") is not None
            ))
        }
