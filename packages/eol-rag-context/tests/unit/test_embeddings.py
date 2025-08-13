"""
Test embeddings module.

Tests cover both sentence-transformers and mock embedding providers.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest

from eol.rag_context.config import EmbeddingConfig
from eol.rag_context.embeddings import (
    EmbeddingManager,
    EmbeddingProvider,
    SentenceTransformerProvider,
)


class TestSentenceTransformerProvider:
    """Test Sentence Transformer provider."""

    @pytest.mark.asyncio
    async def test_mock_embeddings(self):
        """Test that mock embeddings work when model not available."""
        config = EmbeddingConfig(
            provider="sentence-transformers",
            model_name="test-model",  # Unknown model so dimension won't be changed
            dimension=128,
        )

        # Mock the import to raise ImportError during provider initialization
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            provider = SentenceTransformerProvider(config)
            # Should have None model (import failed)
            assert provider.model is None

            # Should return mock embeddings
            embeddings = await provider.embed("test text")
            assert embeddings.shape == (1, 128)
            assert embeddings.dtype == np.float32

    @pytest.mark.asyncio
    async def test_batch_embeddings(self):
        """Test batch embedding generation."""
        config = EmbeddingConfig(model_name="test-model", dimension=64, batch_size=2)

        # Mock sentence_transformers import to force mock embeddings
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            provider = SentenceTransformerProvider(config)

            texts = ["text1", "text2", "text3", "text4"]
            embeddings = await provider.embed_batch(texts, batch_size=2)

            assert embeddings.shape == (4, 64)
            assert embeddings.dtype == np.float32

    @pytest.mark.asyncio
    async def test_single_text_to_list_conversion(self):
        """Test that single text is converted to list."""
        config = EmbeddingConfig(provider="sentence-transformers", model_name="test-model", dimension=32)
        
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            provider = SentenceTransformerProvider(config)
            
            # Test single string input
            result = await provider.embed("single text")
            assert result.shape == (1, 32)
            
            # Test list input
            result = await provider.embed(["text1", "text2"])
            assert result.shape == (2, 32)


class TestEmbeddingManager:
    """Test embedding manager."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return EmbeddingConfig(provider="sentence-transformers", dimension=32)

    @pytest.fixture
    def manager(self, config):
        """Create embedding manager."""
        manager = EmbeddingManager(config, redis_client=None)
        # Mock the provider
        manager.provider = AsyncMock()
        manager.provider.embed = AsyncMock(return_value=np.random.randn(1, 32).astype(np.float32))
        manager.provider.embed_batch = AsyncMock(
            side_effect=lambda texts, batch_size=None: np.random.randn(len(texts), 32).astype(
                np.float32
            )
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

    @pytest.mark.asyncio
    async def test_get_embeddings_no_cache(self, manager):
        """Test getting embeddings without cache."""
        texts = ["text1", "text2"]
        embeddings = await manager.get_embeddings(texts, use_cache=False)
        
        assert embeddings.shape == (2, 32)
        # Should go directly to provider without cache checks
        manager.provider.embed_batch.assert_called_once_with(texts)

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

    def test_init_provider_sentence_transformers(self, config):
        """Test provider initialization for sentence-transformers."""
        manager = EmbeddingManager(config)
        assert isinstance(manager.provider, SentenceTransformerProvider)
        
    def test_cache_stats_with_data(self, manager):
        """Test cache statistics with some data."""
        # Simulate some cache activity
        manager.cache_stats = {"hits": 7, "misses": 3, "total": 10}
        
        stats = manager.get_cache_stats()
        assert stats["hits"] == 7
        assert stats["misses"] == 3
        assert stats["total"] == 10
        assert stats["hit_rate"] == 0.7


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


class TestOpenAIProvider:
    """Test OpenAI embedding provider."""

    @pytest.mark.asyncio
    async def test_openai_provider_init_no_api_key(self):
        """Test OpenAI provider initialization without API key."""
        config = EmbeddingConfig(provider="openai", openai_api_key=None)
        
        with pytest.raises(ValueError, match="OpenAI API key required"):
            from eol.rag_context.embeddings import OpenAIProvider
            OpenAIProvider(config)

    @pytest.mark.asyncio
    async def test_openai_provider_with_mock(self):
        """Test OpenAI provider with mocked client."""
        config = EmbeddingConfig(
            provider="openai",
            openai_api_key="test-key",
            openai_model="text-embedding-ada-002",
            dimension=1536,
            normalize=True
        )
        
        # Mock OpenAI import and client
        mock_single_response = MagicMock()
        mock_single_response.data = [MagicMock(embedding=[0.1] * 1536)]
        
        mock_batch_response = MagicMock()
        mock_batch_response.data = [
            MagicMock(embedding=[0.1] * 1536),
            MagicMock(embedding=[0.2] * 1536)
        ]
        
        mock_client = AsyncMock()
        # Return different responses based on input length
        def create_side_effect(**kwargs):
            input_texts = kwargs['input']
            if len(input_texts) == 1:
                return mock_single_response
            else:
                return mock_batch_response
        
        mock_client.embeddings.create = AsyncMock(side_effect=create_side_effect)
        
        with patch.dict('sys.modules', {'openai': MagicMock(AsyncOpenAI=MagicMock(return_value=mock_client))}):
            from eol.rag_context.embeddings import OpenAIProvider
            provider = OpenAIProvider(config)
            
            # Test single text embedding
            result = await provider.embed("test text")
            assert result.shape == (1, 1536)
            
            # Test batch embedding
            result = await provider.embed(["test1", "test2"])
            assert result.shape == (2, 1536)
            
            # Test batch processing with rate limiting
            texts = ["text1", "text2", "text3", "text4"]
            result = await provider.embed_batch(texts, batch_size=2)
            assert result.shape == (4, 1536)

    @pytest.mark.asyncio
    async def test_openai_provider_import_error(self):
        """Test OpenAI provider when openai package not available."""
        config = EmbeddingConfig(provider="openai", openai_api_key="test-key")
        
        with patch.dict('sys.modules', {'openai': None}):
            with pytest.raises(ImportError, match="openai package required"):
                from eol.rag_context.embeddings import OpenAIProvider
                OpenAIProvider(config)


class TestEmbeddingManagerAdvanced:
    """Advanced tests for embedding manager."""

    @pytest.fixture
    def config_with_cache(self):
        """Create config for caching tests."""
        return EmbeddingConfig(
            provider="sentence-transformers",
            dimension=128
        )

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        redis_mock = AsyncMock()
        redis_mock.get = AsyncMock(return_value=None)
        redis_mock.setex = AsyncMock()
        redis_mock.delete = AsyncMock()
        redis_mock.scan = AsyncMock(return_value=(0, []))
        return redis_mock

    @pytest.mark.asyncio
    async def test_embedding_manager_with_cache(self, config_with_cache, mock_redis):
        """Test embedding manager with Redis cache."""
        manager = EmbeddingManager(config_with_cache, redis_client=mock_redis)
        
        # Mock the provider
        manager.provider = AsyncMock()
        test_embedding = np.random.randn(128).astype(np.float32)
        manager.provider.embed = AsyncMock(return_value=test_embedding)
        
        # Test cache miss
        result = await manager.get_embedding("test text", use_cache=True)
        assert result.shape == (128,)
        mock_redis.get.assert_called_once()
        mock_redis.setex.assert_called_once()
        
        # Verify cache stats
        stats = manager.get_cache_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0
        assert stats["total"] == 1
        assert stats["hit_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_embedding_manager_cache_hit(self, config_with_cache, mock_redis):
        """Test embedding manager cache hit."""
        # Mock cached embedding
        cached_embedding = np.random.randn(128).astype(np.float32)
        mock_redis.get = AsyncMock(return_value=cached_embedding.tobytes())
        
        manager = EmbeddingManager(config_with_cache, redis_client=mock_redis)
        manager.provider = AsyncMock()  # Should not be called
        
        result = await manager.get_embedding("test text", use_cache=True)
        assert result.shape == (1, 128)
        
        # Provider should not be called on cache hit
        manager.provider.embed.assert_not_called()
        
        # Verify cache stats
        stats = manager.get_cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0

    @pytest.mark.asyncio
    async def test_batch_embeddings_with_cache(self, config_with_cache, mock_redis):
        """Test batch embeddings with partial cache hits."""
        # Mock partial cache hits
        cached_embedding = np.random.randn(128).astype(np.float32)
        mock_redis.get = AsyncMock(side_effect=[
            cached_embedding.tobytes(),  # First text cached
            None,  # Second text not cached
            None   # Third text not cached
        ])
        
        manager = EmbeddingManager(config_with_cache, redis_client=mock_redis)
        
        # Mock provider for uncached texts
        new_embeddings = np.random.randn(2, 128).astype(np.float32)
        manager.provider = AsyncMock()
        manager.provider.embed_batch = AsyncMock(return_value=new_embeddings)
        
        texts = ["cached text", "new text 1", "new text 2"]
        result = await manager.get_embeddings(texts, use_cache=True)
        
        assert result.shape == (3, 128)
        # Should call embed_batch for uncached texts only
        manager.provider.embed_batch.assert_called_once_with(["new text 1", "new text 2"])
        
        # Should cache the new embeddings
        assert mock_redis.setex.call_count == 2

    @pytest.mark.asyncio
    async def test_clear_cache(self, config_with_cache, mock_redis):
        """Test cache clearing functionality."""
        # Mock scan to return some cache keys
        mock_redis.scan = AsyncMock(side_effect=[
            (10, ["emb:key1", "emb:key2"]),  # First scan
            (0, ["emb:key3"])  # Second scan, cursor 0 means done
        ])
        
        manager = EmbeddingManager(config_with_cache, redis_client=mock_redis)
        
        await manager.clear_cache()
        
        # Should scan for cache keys
        assert mock_redis.scan.call_count == 2
        # Should delete found keys
        assert mock_redis.delete.call_count == 2

    def test_init_provider_unknown(self):
        """Test initialization with unknown provider."""
        config = EmbeddingConfig(provider="unknown-provider")
        
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            EmbeddingManager(config)

    def test_init_openai_provider(self):
        """Test initialization with OpenAI provider."""
        config = EmbeddingConfig(provider="openai", openai_api_key="test-key")
        
        with patch('eol.rag_context.embeddings.OpenAIProvider') as mock_provider_class:
            manager = EmbeddingManager(config)
            mock_provider_class.assert_called_once_with(config)

    @pytest.mark.asyncio
    async def test_embedding_reshaping(self, config_with_cache):
        """Test embedding reshaping from 2D to 1D."""
        manager = EmbeddingManager(config_with_cache, redis_client=None)
        
        # Mock provider to return 2D array
        batch_embedding = np.random.randn(1, 128).astype(np.float32)
        manager.provider = AsyncMock()
        manager.provider.embed = AsyncMock(return_value=batch_embedding)
        
        result = await manager.get_embedding("test text", use_cache=False)
        
        # Should be reshaped to 1D
        assert result.shape == (128,)

    def test_cache_key_consistency(self, config_with_cache):
        """Test cache key generation consistency."""
        manager = EmbeddingManager(config_with_cache, redis_client=None)
        
        key1 = manager._cache_key("test text")
        key2 = manager._cache_key("test text")
        key3 = manager._cache_key("different text")
        
        assert key1 == key2  # Same text should generate same key
        assert key1 != key3  # Different text should generate different key
        assert key1.startswith("emb:")  # Should have proper prefix
        assert len(key1) > 4  # Should have hash after prefix


class TestSentenceTransformerProviderAdvanced:
    """Advanced tests for Sentence Transformer provider."""

    @pytest.mark.asyncio
    async def test_real_model_path(self):
        """Test with sentence transformers available (mocked)."""
        config = EmbeddingConfig(
            provider="sentence-transformers",
            model_name="all-MiniLM-L6-v2",
            dimension=384,
            normalize=True,
            batch_size=16
        )
        
        # Mock sentence_transformers module
        mock_model = MagicMock()
        mock_model.encode = MagicMock(return_value=np.random.randn(2, 384).astype(np.float32))
        
        mock_st_class = MagicMock(return_value=mock_model)
        
        with patch.dict('sys.modules', {'sentence_transformers': MagicMock(SentenceTransformer=mock_st_class)}):
            provider = SentenceTransformerProvider(config)
            
            # Should have real model, not None
            assert provider.model is not None
            
            # Test embedding
            result = await provider.embed(["text1", "text2"])
            assert result.shape == (2, 384)
            
            # Verify model.encode was called with correct parameters
            mock_model.encode.assert_called_once()
            args, kwargs = mock_model.encode.call_args
            assert args[0] == ["text1", "text2"]
            assert kwargs["normalize_embeddings"] == True
            assert kwargs["show_progress_bar"] == False

    @pytest.mark.asyncio
    async def test_batch_processing_large(self):
        """Test batch processing with large dataset."""
        config = EmbeddingConfig(
            provider="sentence-transformers",
            model_name="test-model",
            dimension=64,
            batch_size=3
        )
        
        # Force mock mode
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            provider = SentenceTransformerProvider(config)
            
            # Test with 10 texts, batch size 3
            texts = [f"text {i}" for i in range(10)]
            result = await provider.embed_batch(texts, batch_size=3)
            
            assert result.shape == (10, 64)
            assert result.dtype == np.float32

    def test_executor_initialization(self):
        """Test that ThreadPoolExecutor is initialized."""
        config = EmbeddingConfig(provider="sentence-transformers", model_name="test-model")
        
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            provider = SentenceTransformerProvider(config)
            
            assert provider.executor is not None
            assert hasattr(provider.executor, 'submit')


# Removed TestEmbeddingsAdditional class - tests were for non-existent methods
