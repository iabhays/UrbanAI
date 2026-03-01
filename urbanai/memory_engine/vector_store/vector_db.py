"""
Vector database interface using FAISS.

Provides abstraction for vector storage and similarity search.
"""

import numpy as np
import faiss
from typing import List, Optional, Tuple, Dict
from pathlib import Path
from loguru import logger

from ...utils.config import get_config


class VectorDatabase:
    """
    Vector database using FAISS.
    
    Provides efficient vector storage and similarity search.
    """
    
    def __init__(
        self,
        dimension: int = 512,
        index_type: str = "IVF_FLAT",
        metric: str = "L2",
        index_path: Optional[str] = None
    ):
        """
        Initialize vector database.
        
        Args:
            dimension: Vector dimension
            index_type: Index type (IVF_FLAT, HNSW, FLAT)
            metric: Distance metric (L2, IP for inner product)
            index_path: Path to save/load index
        """
        self.config = get_config()
        memory_config = self.config.get_section("memory")
        vector_config = memory_config.get("vector_db", {})
        
        self.dimension = vector_config.get("dimension", dimension)
        self.index_type = vector_config.get("index_type", index_type)
        self.metric = vector_config.get("metric", metric)
        self.index_path = index_path or vector_config.get("index_path")
        
        # Initialize FAISS index
        self.index = self._create_index()
        self.is_trained = False
    
    def _create_index(self):
        """Create FAISS index."""
        if self.metric == "L2":
            metric = faiss.METRIC_L2
        elif self.metric == "IP":
            metric = faiss.METRIC_INNER_PRODUCT
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        
        if self.index_type == "FLAT":
            index = faiss.IndexFlat(self.dimension, metric)
        elif self.index_type == "IVF_FLAT":
            # Use IVF with 100 clusters
            quantizer = faiss.IndexFlat(self.dimension, metric)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, 100, metric)
        elif self.index_type == "HNSW":
            index = faiss.IndexHNSWFlat(self.dimension, 32, metric)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        return index
    
    def add(self, vectors: np.ndarray, ids: Optional[List[int]] = None):
        """
        Add vectors to index.
        
        Args:
            vectors: Array of vectors [N, dimension]
            ids: Optional list of IDs
        """
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        # Ensure float32
        vectors = vectors.astype(np.float32)
        
        # Train index if needed
        if not self.is_trained and hasattr(self.index, "train"):
            if self.index.ntotal == 0:
                # Need at least some vectors for training
                if vectors.shape[0] >= 100:
                    self.index.train(vectors)
                    self.is_trained = True
        
        # Add vectors
        if ids is not None:
            # FAISS doesn't directly support IDs, so we'll use sequential IDs
            # In production, maintain a separate ID mapping
            self.index.add(vectors)
        else:
            self.index.add(vectors)
        
        logger.debug(f"Added {vectors.shape[0]} vectors to index. Total: {self.index.ntotal}")
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors.
        
        Args:
            query: Query vector [dimension] or [N, dimension]
            k: Number of results
        
        Returns:
            Tuple of (distances, indices)
        """
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        query = query.astype(np.float32)
        
        # Ensure index is trained
        if hasattr(self.index, "is_trained") and not self.index.is_trained:
            if self.index.ntotal > 0:
                # Use existing vectors for training
                vectors = self.index.reconstruct_n(0, min(100, self.index.ntotal))
                self.index.train(vectors)
                self.index.is_trained = True
        
        distances, indices = self.index.search(query, k)
        
        return distances, indices
    
    def remove(self, indices: List[int]):
        """
        Remove vectors by indices.
        
        Note: FAISS doesn't support direct removal. This is a placeholder.
        In production, maintain a separate index or use ID mapping.
        """
        logger.warning("FAISS doesn't support direct removal. Consider rebuilding index.")
    
    def save(self, path: Optional[str] = None):
        """Save index to disk."""
        save_path = path or self.index_path
        if save_path is None:
            logger.warning("No index path specified")
            return
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, save_path)
        logger.info(f"Index saved to {save_path}")
    
    def load(self, path: Optional[str] = None):
        """Load index from disk."""
        load_path = path or self.index_path
        if load_path is None or not Path(load_path).exists():
            logger.warning(f"Index file not found: {load_path}")
            return
        
        self.index = faiss.read_index(load_path)
        self.is_trained = True
        logger.info(f"Index loaded from {load_path}")
    
    def get_size(self) -> int:
        """Get number of vectors in index."""
        return self.index.ntotal
    
    def reset(self):
        """Reset index."""
        self.index = self._create_index()
        self.is_trained = False
