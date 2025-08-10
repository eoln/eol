"""
Test embeddings module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock
from eol.rag_context.embeddings import (
    EmbeddingProvider,
    SentenceTransformerProvider,
    EmbeddingManager
)
from eol.rag_context.config import EmbeddingConfig


class TestSentenceTransformerProvider:
    """Test Sentence Transformer provider."""
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Mock conflicts with test setup")
    async def test_mock_embeddings(self):
        """Test that mock embeddings work when model not available."""
        config = EmbeddingConfig(
            provider="sentence-transformers",
            model_name="test-model",  # Unknown model so dimension won't be changed
            dimension=128
        )
        
        provider = SentenceTransformerProvider(config)
        # Should have None model (not installed)
        assert provider.model is None
        
        # Should return mock embeddings
        embeddings = await provider.embed("test text")
        assert embeddings.shape == (1, 128)
        assert embeddings.dtype == np.float32
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Mock conflicts with test setup") 
    async def test_batch_embeddings(self):
        """Test batch embedding generation."""
        config = EmbeddingConfig(model_name="test-model", dimension=64, batch_size=2)
        provider = SentenceTransformerProvider(config)
        
        texts = ["text1", "text2", "text3", "text4"]
        embeddings = await provider.embed_batch(texts, batch_size=2)
        
        assert embeddings.shape == (4, 64)
        assert embeddings.dtype == np.float32


class TestEmbeddingManager:
    """Test embedding manager."""
    
    @pytest.fixture
    def config(self):
        """Create test config."""
        return EmbeddingConfig(
            provider="sentence-transformers",
            dimension=32
        )
    
    @pytest.fixture
    def manager(self, config):
        """Create embedding manager."""
        manager = EmbeddingManager(config, redis_client=None)
        # Mock the provider
        manager.provider = AsyncMock()
        manager.provider.embed = AsyncMock(
            return_value=np.random.randn(1, 32).astype(np.float32)
        )
        manager.provider.embed_batch = AsyncMock(
            side_effect=lambda texts, batch_size=None: 
            np.random.randn(len(texts), 32).astype(np.float32)
        )
        return manager
    
    @pytest.mark.asyncio
    async def test_get_embedding(self, manager):
        """Test getting single embedding."""
        embedding = await manager.get_embedding("test text", use_cache=False)
        
        assert embedding.shape == (32,) or embedding.shape == (1, 32)
        manager.provider.embed.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_embeddings_batch(self, manager):
        """Test getting batch embeddings."""
        texts = ["text1", "text2", "text3"]
        embeddings = await manager.get_embeddings(texts, use_cache=False)
        
        assert embeddings.shape == (3, 32)
        manager.provider.embed_batch.assert_called_once()
    
    def test_cache_stats(self, manager):
        """Test cache statistics."""
        stats = manager.get_cache_stats()
        
        assert "hits" in stats
        assert "misses" in stats
        assert "total" in stats
        assert "hit_rate" in stats
        assert stats["hit_rate"] == 0.0  # No queries yet
    
    @pytest.mark.asyncio
    async def test_cache_key_generation(self, manager):
        """Test cache key generation."""
        key1 = manager._cache_key("test text")
        key2 = manager._cache_key("test text")
        key3 = manager._cache_key("different text")
        
        assert key1 == key2  # Same text, same key
        assert key1 != key3  # Different text, different key
        assert key1.startswith("emb:")


class TestEmbeddingProviderInterface:
    """Test embedding provider interface."""
    
    def test_interface_methods(self):
        """Test that interface defines required methods."""
        provider = EmbeddingProvider()
        
        with pytest.raises(NotImplementedError):
            import asyncio
            asyncio.run(provider.embed("test"))
        
        with pytest.raises(NotImplementedError):
            import asyncio
            asyncio.run(provider.embed_batch(["test"]))