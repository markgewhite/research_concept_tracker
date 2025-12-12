"""Embedding service using Qwen3 model"""

import numpy as np
import logging
from sentence_transformers import SentenceTransformer
from backend.models import Paper
from backend.config import settings
from backend.utils.cache import EmbeddingCache

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generate and cache embeddings using local Qwen3 model"""

    def __init__(self):
        """Initialize embedding model and cache"""
        logger.info(f"Loading embedding model: {settings.embedding_model}")

        try:
            self.model = SentenceTransformer(settings.embedding_model)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

        self.cache = EmbeddingCache(settings.cache_dir)
        cache_stats = self.cache.get_stats()
        logger.info(f"Cache initialized: {cache_stats['num_cached']} embeddings cached ({cache_stats['total_size_mb']:.2f} MB)")

    def embed_paper(self, paper: Paper) -> np.ndarray:
        """
        Generate embedding for a paper (title + abstract)

        Args:
            paper: Paper object to embed

        Returns:
            Normalized embedding vector
        """
        # Check cache first
        cached = self.cache.get(paper.arxiv_id)
        if cached is not None:
            return cached

        # Generate embedding
        text = f"{paper.title}\n\n{paper.abstract}"
        logger.debug(f"Generating embedding for {paper.arxiv_id}")

        try:
            # Generate and normalize embedding
            embedding = self.model.encode(text, normalize_embeddings=True)

            # Ensure it's a numpy array
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)

            # Cache the result
            self.cache.set(paper.arxiv_id, embedding)

            logger.debug(f"Generated embedding for {paper.arxiv_id} (dim={len(embedding)}, norm={np.linalg.norm(embedding):.3f})")

            return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding for {paper.arxiv_id}: {e}")
            raise

    def embed_papers(self, papers: list[Paper]) -> np.ndarray:
        """
        Batch embed multiple papers (checks cache individually)

        Args:
            papers: List of Paper objects

        Returns:
            Array of embeddings (num_papers x embedding_dim)
        """
        logger.info(f"Embedding {len(papers)} papers")

        embeddings = []
        num_cached = 0
        num_generated = 0

        for paper in papers:
            cached = self.cache.get(paper.arxiv_id)
            if cached is not None:
                embeddings.append(cached)
                num_cached += 1
            else:
                embedding = self.embed_paper(paper)
                embeddings.append(embedding)
                num_generated += 1

        logger.info(f"Embeddings complete: {num_cached} from cache, {num_generated} generated")

        return np.array(embeddings)

    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        return self.embedding_dim
