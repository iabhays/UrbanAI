"""
Identity memory storage.

Stores and retrieves identity embeddings for re-identification.
"""

import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from loguru import logger

from .vector_db import VectorDatabase
from ...utils.config import get_config


class IdentityMemory:
    """
    Identity memory storage.
    
    Stores identity embeddings for person re-identification.
    """
    
    def __init__(self, vector_db: Optional[VectorDatabase] = None):
        """
        Initialize identity memory.
        
        Args:
            vector_db: Vector database instance
        """
        self.config = get_config()
        memory_config = self.config.get_section("memory")
        storage_config = memory_config.get("storage", {})
        
        self.ttl = storage_config.get("identity_embedding_ttl", 604800)  # 7 days
        
        self.vector_db = vector_db or VectorDatabase(
            dimension=512,
            index_type="HNSW",  # HNSW is better for identity search
            metric="IP"  # Inner product for cosine similarity
        )
        
        # Identity storage
        self.identities: Dict[int, Dict] = {}
        self.track_to_identity: Dict[int, int] = {}  # track_id -> identity_id
        self.next_identity_id = 1
    
    def register_identity(
        self,
        embedding: np.ndarray,
        track_id: int,
        camera_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Register new identity or update existing.
        
        Args:
            embedding: Identity embedding
            track_id: Track ID
            camera_id: Camera ID
            metadata: Optional metadata
        
        Returns:
            Identity ID
        """
        # Check if track already has identity
        if track_id in self.track_to_identity:
            identity_id = self.track_to_identity[track_id]
            # Update existing identity
            self.update_identity(identity_id, embedding, track_id, camera_id)
            return identity_id
        
        # Search for similar identity
        similar = self.search_similar(embedding, k=1, min_similarity=0.8)
        
        if similar:
            # Match found, use existing identity
            identity_id = similar[0]["identity_id"]
            self.track_to_identity[track_id] = identity_id
            self.update_identity(identity_id, embedding, track_id, camera_id)
            return identity_id
        
        # Create new identity
        identity_id = self.next_identity_id
        self.next_identity_id += 1
        
        # Store embedding
        self.vector_db.add(embedding.reshape(1, -1))
        
        # Store identity metadata
        self.identities[identity_id] = {
            "identity_id": identity_id,
            "tracks": [track_id],
            "cameras": [camera_id] if camera_id else [],
            "first_seen": datetime.now().timestamp(),
            "last_seen": datetime.now().timestamp(),
            "expires_at": datetime.now().timestamp() + self.ttl,
            "metadata": metadata or {},
            "embedding_index": self.vector_db.get_size() - 1
        }
        
        self.track_to_identity[track_id] = identity_id
        
        logger.debug(f"Registered new identity: ID={identity_id}, track_id={track_id}")
        
        return identity_id
    
    def update_identity(
        self,
        identity_id: int,
        embedding: np.ndarray,
        track_id: int,
        camera_id: Optional[str] = None
    ):
        """Update existing identity."""
        if identity_id not in self.identities:
            logger.warning(f"Identity {identity_id} not found")
            return
        
        identity = self.identities[identity_id]
        
        # Update tracks
        if track_id not in identity["tracks"]:
            identity["tracks"].append(track_id)
        
        # Update cameras
        if camera_id and camera_id not in identity["cameras"]:
            identity["cameras"].append(camera_id)
        
        # Update timestamps
        identity["last_seen"] = datetime.now().timestamp()
        identity["expires_at"] = datetime.now().timestamp() + self.ttl
        
        # Update embedding (in production, maintain average or use separate storage)
        # For now, we'll just update the metadata
    
    def search_similar(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        min_similarity: float = 0.7
    ) -> List[Dict]:
        """
        Search for similar identities.
        
        Args:
            query_embedding: Query embedding
            k: Number of results
            min_similarity: Minimum similarity threshold
        
        Returns:
            List of similar identity records
        """
        # Search in vector database
        distances, indices = self.vector_db.search(query_embedding, k=k)
        
        # For inner product metric, higher is better
        if self.vector_db.metric == "IP":
            similarities = distances[0]  # Already similarity scores
        else:
            similarities = 1.0 / (1.0 + distances[0])
        
        # Get identity metadata
        results = []
        for idx, sim in zip(indices[0], similarities):
            if sim >= min_similarity:
                # Find identity by embedding index
                for identity_id, identity_data in self.identities.items():
                    if identity_data.get("embedding_index") == idx:
                        result = identity_data.copy()
                        result["similarity"] = float(sim)
                        results.append(result)
                        break
        
        return results
    
    def get_identity(self, identity_id: int) -> Optional[Dict]:
        """Get identity by ID."""
        return self.identities.get(identity_id)
    
    def get_identity_by_track(self, track_id: int) -> Optional[Dict]:
        """Get identity by track ID."""
        identity_id = self.track_to_identity.get(track_id)
        if identity_id:
            return self.get_identity(identity_id)
        return None
    
    def cleanup_expired(self):
        """Remove expired identities."""
        current_time = datetime.now().timestamp()
        
        expired_ids = [
            idx for idx, identity in self.identities.items()
            if identity.get("expires_at", 0) < current_time
        ]
        
        for identity_id in expired_ids:
            # Remove track mappings
            tracks = self.identities[identity_id]["tracks"]
            for track_id in tracks:
                self.track_to_identity.pop(track_id, None)
            
            del self.identities[identity_id]
        
        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired identities")
