"""
Embeddings management for EOL RAG Context.
"""

import asyncio
import hashlib

# from sentence_transformers import SentenceTransformer  # Optional dependency
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .config import EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbeddingProvider:
    """Base class for embedding providers."""

    async def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text(s)."""
        raise NotImplementedError

    async def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings in batches."""
        raise NotImplementedError


class SentenceTransformerProvider(EmbeddingProvider):
    """Sentence Transformers embedding provider."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(config.model_name)
        except ImportError:
            # Fallback to mock embeddings for testing
            self.model = None
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings using Sentence Transformers."""
        if isinstance(texts, str):
            texts = [texts]

        if self.model is None:
            # Return mock embeddings for testing
            return np.random.randn(len(texts), self.config.dimension).astype(np.float32)

        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            self.executor,
            lambda: self.model.encode(
                texts, normalize_embeddings=self.config.normalize, show_progress_bar=False
            ),
        )

        return embeddings

    async def embed_batch(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Generate embeddings in batches."""
        batch_size = batch_size or self.config.batch_size

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = await self.embed(batch)
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)


class OpenAIProvider(EmbeddingProvider):
    """OpenAI embeddings provider."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config

        if not config.openai_api_key:
            raise ValueError("OpenAI API key required for OpenAI embeddings")

        try:
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(api_key=config.openai_api_key)
        except ImportError:
            raise ImportError("openai package required for OpenAI embeddings")

    async def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings using OpenAI API."""
        if isinstance(texts, str):
            texts = [texts]

        response = await self.client.embeddings.create(model=self.config.openai_model, input=texts)

        embeddings = np.array([e.embedding for e in response.data])

        if self.config.normalize:
            # Normalize embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms

        return embeddings

    async def embed_batch(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Generate embeddings in batches."""
        batch_size = batch_size or self.config.batch_size

        # OpenAI has a limit on input size, so batch accordingly
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = await self.embed(batch)
            all_embeddings.append(embeddings)

            # Rate limiting
            if i + batch_size < len(texts):
                await asyncio.sleep(0.1)

        return np.vstack(all_embeddings)


class EmbeddingManager:
    """Manages embeddings with caching and multiple providers."""

    def __init__(self, config: EmbeddingConfig, redis_client=None):
        self.config = config
        self.redis = redis_client
        self.provider = self._init_provider()
        self.cache_stats = {"hits": 0, "misses": 0, "total": 0}

    def _init_provider(self) -> EmbeddingProvider:
        """Initialize the configured embedding provider."""
        if self.config.provider == "sentence-transformers":
            return SentenceTransformerProvider(self.config)
        elif self.config.provider == "openai":
            return OpenAIProvider(self.config)
        else:
            raise ValueError(f"Unknown embedding provider: {self.config.provider}")

    def _cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return f"emb:{hashlib.md5(text.encode()).hexdigest()}"

    async def get_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Get embedding for text with caching."""
        self.cache_stats["total"] += 1

        if use_cache and self.redis:
            # Check cache
            cache_key = self._cache_key(text)
            cached = await self.redis.get(cache_key)

            if cached:
                self.cache_stats["hits"] += 1
                return np.frombuffer(cached, dtype=np.float32).reshape(1, -1)

        # Generate embedding
        self.cache_stats["misses"] += 1
        embedding = await self.provider.embed(text)

        if isinstance(embedding, np.ndarray) and embedding.ndim == 2:
            embedding = embedding[0]  # Get first embedding if batch result

        # Cache the result
        if use_cache and self.redis:
            cache_key = self._cache_key(text)
            await self.redis.setex(
                cache_key, 3600, embedding.astype(np.float32).tobytes()  # 1 hour TTL
            )

        return embedding

    async def get_embeddings(self, texts: List[str], use_cache: bool = True) -> np.ndarray:
        """Get embeddings for multiple texts with caching."""
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        if use_cache and self.redis:
            # Check cache for each text
            for i, text in enumerate(texts):
                self.cache_stats["total"] += 1
                cache_key = self._cache_key(text)
                cached = await self.redis.get(cache_key)

                if cached:
                    self.cache_stats["hits"] += 1
                    embedding = np.frombuffer(cached, dtype=np.float32)
                    embeddings.append((i, embedding))
                else:
                    self.cache_stats["misses"] += 1
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))

        # Generate embeddings for uncached texts
        if uncached_texts:
            new_embeddings = await self.provider.embed_batch(uncached_texts)

            # Cache the new embeddings
            if use_cache and self.redis:
                for text, embedding in zip(uncached_texts, new_embeddings):
                    cache_key = self._cache_key(text)
                    await self.redis.setex(cache_key, 3600, embedding.astype(np.float32).tobytes())

            # Add to results
            for idx, embedding in zip(uncached_indices, new_embeddings):
                embeddings.append((idx, embedding))

        # Sort by original index and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        return np.vstack([e[1] for e in embeddings])

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self.cache_stats.copy()
        if stats["total"] > 0:
            stats["hit_rate"] = stats["hits"] / stats["total"]
        else:
            stats["hit_rate"] = 0.0
        return stats

    async def clear_cache(self) -> None:
        """Clear embedding cache."""
        if self.redis:
            # Find and delete all embedding cache keys
            cursor = 0
            while True:
                cursor, keys = await self.redis.scan(cursor, match="emb:*", count=100)

                if keys:
                    await self.redis.delete(*keys)

                if cursor == 0:
                    break

        # Reset stats
        self.cache_stats = {"hits": 0, "misses": 0, "total": 0}
