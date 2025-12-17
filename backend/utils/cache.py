"""Embedding cache for storing and retrieving paper embeddings"""

import pickle
import json
import logging
from pathlib import Path
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Simple pickle-based cache for embeddings with model-aware keys"""

    def __init__(self, cache_dir: str, model_name: str = "default"):
        """
        Initialize embedding cache

        Args:
            cache_dir: Directory path for cache storage
            model_name: Name of the embedding model (used in cache keys)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.embeddings_dir = self.cache_dir / "embeddings"
        self.embeddings_dir.mkdir(exist_ok=True)

        # Sanitize model name for filesystem (replace / with _)
        self.model_name = model_name.replace("/", "_").replace("\\", "_")

        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()

    def get(self, arxiv_id: str) -> Optional[np.ndarray]:
        """
        Retrieve cached embedding for a paper

        Args:
            arxiv_id: ArXiv ID of the paper

        Returns:
            Cached embedding vector or None if not found
        """
        cache_file = self.embeddings_dir / f"{self.model_name}_{arxiv_id}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    embedding = pickle.load(f)
                logger.debug(f"Cache hit: {arxiv_id} (model: {self.model_name})")
                return embedding
            except Exception as e:
                logger.warning(f"Failed to load cache for {arxiv_id}: {e}")
                return None

        logger.debug(f"Cache miss: {arxiv_id} (model: {self.model_name})")
        return None

    def set(self, arxiv_id: str, embedding: np.ndarray) -> None:
        """
        Store embedding in cache

        Args:
            arxiv_id: ArXiv ID of the paper
            embedding: Embedding vector to cache
        """
        cache_file = self.embeddings_dir / f"{self.model_name}_{arxiv_id}.pkl"

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)

            # Update metadata with model-aware key
            cache_key = f"{self.model_name}_{arxiv_id}"
            self.metadata[cache_key] = {
                'model': self.model_name,
                'dim': len(embedding),
                'cached_at': str(cache_file.stat().st_mtime)
            }
            self._save_metadata()

            logger.debug(f"Cached embedding: {arxiv_id} (model={self.model_name}, dim={len(embedding)})")
        except Exception as e:
            logger.error(f"Failed to cache {arxiv_id}: {e}")

    def _load_metadata(self) -> dict:
        """Load metadata JSON"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
                return {}
        return {}

    def _save_metadata(self) -> None:
        """Save metadata JSON"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def get_stats(self) -> dict:
        """Get cache statistics"""
        num_cached = len(list(self.embeddings_dir.glob("*.pkl")))
        total_size = sum(f.stat().st_size for f in self.embeddings_dir.glob("*.pkl"))

        return {
            "num_cached": num_cached,
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir)
        }
