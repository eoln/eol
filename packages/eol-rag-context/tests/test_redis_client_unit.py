"""Unit tests for redis_client module with improved coverage."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock, PropertyMock
import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eol.rag_context.redis_client import RedisVectorStore
from eol.rag_context.config import RedisConfig


class TestRedisVectorStoreUnit:
    """Unit tests for RedisVectorStore class."""
    
    @pytest.fixture
    def redis_config(self):
        """Create test Redis configuration."""
        return RedisConfig()
    
    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        mock = MagicMock()
        mock.ping = MagicMock(return_value=True)
        mock.hset = MagicMock(return_value=1)
        mock.hget = MagicMock(return_value=json.dumps({"test": "data"}))
        mock.exists = MagicMock(return_value=1)
        mock.delete = MagicMock(return_value=1)
        mock.scan_iter = MagicMock(return_value=iter(["key1", "key2"]))
        return mock
    
    @pytest.fixture
    def mock_async_redis(self):
        """Create mock async Redis client."""
        mock = AsyncMock()
        mock.ping = AsyncMock(return_value=True)
        mock.hset = AsyncMock(return_value=1)
        mock.hget = AsyncMock(return_value=json.dumps({"test": "data"}))
        mock.exists = AsyncMock(return_value=1)
        mock.delete = AsyncMock(return_value=1)
        mock.scan = AsyncMock(return_value=(0, [b"key1", b"key2"]))
        
        # Mock FT search
        mock_ft = AsyncMock()
        mock_search = AsyncMock()
        mock_search.search = AsyncMock(return_value=MagicMock(docs=[]))
        mock_ft.return_value = mock_search
        mock.ft = mock_ft
        
        return mock
    
    @pytest.fixture
    @patch('eol.rag_context.redis_client.Redis')
    @patch('eol.rag_context.redis_client.redis.asyncio.Redis')
    def redis_store(self, mock_async_redis_class, mock_redis_class, redis_config, mock_redis, mock_async_redis):
        """Create RedisVectorStore with mocked clients."""
        mock_redis_class.return_value = mock_redis
        mock_async_redis_class.return_value = mock_async_redis
        store = RedisVectorStore(redis_config)
        store.sync_redis = mock_redis
        store.async_redis = mock_async_redis
        return store
    
    def test_initialization(self, redis_config):
        """Test RedisVectorStore initialization."""
        store = RedisVectorStore(redis_config)
        assert store.config == redis_config
        assert store.sync_redis is not None
        assert store.async_redis is not None
    
    def test_connect_sync(self, redis_store, mock_redis):
        """Test synchronous connection."""
        redis_store.connect()
        mock_redis.ping.assert_called_once()
    
    def test_connect_sync_failure(self, redis_store, mock_redis):
        """Test synchronous connection failure."""
        mock_redis.ping.side_effect = Exception("Connection failed")
        with pytest.raises(Exception):
            redis_store.connect()
    
    @pytest.mark.asyncio
    async def test_connect_async(self, redis_store, mock_async_redis):
        """Test asynchronous connection."""
        await redis_store.connect_async()
        mock_async_redis.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connect_async_failure(self, redis_store, mock_async_redis):
        """Test asynchronous connection failure."""
        mock_async_redis.ping.side_effect = Exception("Connection failed")
        with pytest.raises(Exception):
            await redis_store.connect_async()
    
    def test_store_document_sync(self, redis_store, mock_redis):
        """Test storing document synchronously."""
        doc_id = "test_doc"
        document = {"content": "test content", "metadata": {}}
        embedding = np.array([0.1, 0.2, 0.3])
        
        redis_store.store_document(doc_id, document, embedding)
        
        # Check hset was called
        assert mock_redis.hset.called
        call_args = mock_redis.hset.call_args
        assert call_args[0][0] == f"doc:{doc_id}"
    
    @pytest.mark.asyncio
    async def test_store_document_async(self, redis_store, mock_async_redis):
        """Test storing document asynchronously."""
        doc_id = "test_doc"
        document = {"content": "test content", "metadata": {}}
        embedding = np.array([0.1, 0.2, 0.3])
        
        await redis_store.store_document_async(doc_id, document, embedding)
        
        # Check hset was called
        assert mock_async_redis.hset.called
    
    def test_get_document_sync(self, redis_store, mock_redis):
        """Test retrieving document synchronously."""
        doc_id = "test_doc"
        mock_redis.hget.return_value = json.dumps({"content": "test"}).encode()
        
        result = redis_store.get_document(doc_id)
        
        assert result is not None
        mock_redis.hget.assert_called()
    
    @pytest.mark.asyncio
    async def test_get_document_async(self, redis_store, mock_async_redis):
        """Test retrieving document asynchronously."""
        doc_id = "test_doc"
        mock_async_redis.hget.return_value = json.dumps({"content": "test"}).encode()
        
        result = await redis_store.get_document_async(doc_id)
        
        assert result is not None
        mock_async_redis.hget.assert_called()
    
    def test_delete_document_sync(self, redis_store, mock_redis):
        """Test deleting document synchronously."""
        doc_id = "test_doc"
        
        result = redis_store.delete_document(doc_id)
        
        assert result is True
        mock_redis.delete.assert_called_with(f"doc:{doc_id}")
    
    @pytest.mark.asyncio
    async def test_delete_document_async(self, redis_store, mock_async_redis):
        """Test deleting document asynchronously."""
        doc_id = "test_doc"
        
        result = await redis_store.delete_document_async(doc_id)
        
        assert result is True
        mock_async_redis.delete.assert_called()
    
    def test_check_exists_sync(self, redis_store, mock_redis):
        """Test checking document existence synchronously."""
        doc_id = "test_doc"
        
        result = redis_store.check_exists(doc_id)
        
        assert result is True
        mock_redis.exists.assert_called_with(f"doc:{doc_id}")
    
    @pytest.mark.asyncio
    async def test_check_exists_async(self, redis_store, mock_async_redis):
        """Test checking document existence asynchronously."""
        doc_id = "test_doc"
        
        result = await redis_store.check_exists_async(doc_id)
        
        assert result is True
        mock_async_redis.exists.assert_called()
    
    @pytest.mark.asyncio
    async def test_vector_search(self, redis_store, mock_async_redis):
        """Test vector search functionality."""
        query_embedding = np.array([0.1, 0.2, 0.3])
        
        # Setup mock search results
        mock_doc = MagicMock()
        mock_doc.id = "doc:test"
        mock_doc.score = 0.95
        
        mock_result = MagicMock()
        mock_result.docs = [mock_doc]
        
        mock_async_redis.ft.return_value.search.return_value = mock_result
        
        results = await redis_store.vector_search(query_embedding, k=5)
        
        assert results is not None
        mock_async_redis.ft.assert_called()
    
    @pytest.mark.asyncio
    async def test_list_documents_async(self, redis_store, mock_async_redis):
        """Test listing documents asynchronously."""
        mock_async_redis.scan.return_value = (0, [b"doc:1", b"doc:2"])
        
        docs = await redis_store.list_documents_async()
        
        assert len(docs) == 2
        mock_async_redis.scan.assert_called()
    
    def test_get_stats(self, redis_store, mock_redis):
        """Test getting statistics."""
        mock_redis.info.return_value = {
            "used_memory_human": "10M",
            "connected_clients": 5,
            "total_commands_processed": 1000
        }
        mock_redis.dbsize.return_value = 100
        
        stats = redis_store.get_stats()
        
        assert stats["total_documents"] == 100
        assert stats["memory_usage"] == "10M"
        mock_redis.info.assert_called()
        mock_redis.dbsize.assert_called()
    
    @pytest.mark.asyncio
    async def test_create_index_async(self, redis_store, mock_async_redis):
        """Test creating vector index asynchronously."""
        index_name = "test_index"
        
        # Mock ft.create_index
        mock_ft = AsyncMock()
        mock_create = AsyncMock()
        mock_ft.create_index = mock_create
        mock_async_redis.ft.return_value = mock_ft
        
        await redis_store.create_index_async(index_name, dimension=128)
        
        mock_async_redis.ft.assert_called_with(index_name)
    
    def test_batch_store_documents(self, redis_store, mock_redis):
        """Test batch storing documents."""
        documents = [
            ("doc1", {"content": "test1"}, np.array([0.1, 0.2])),
            ("doc2", {"content": "test2"}, np.array([0.3, 0.4]))
        ]
        
        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.hset = MagicMock()
        mock_pipeline.execute = MagicMock(return_value=[1, 1])
        mock_redis.pipeline.return_value = mock_pipeline
        
        redis_store.batch_store_documents(documents)
        
        mock_redis.pipeline.assert_called()
        assert mock_pipeline.hset.call_count == 2
        mock_pipeline.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_remove_source_async(self, redis_store, mock_async_redis):
        """Test removing source asynchronously."""
        source_id = "source_123"
        
        # Mock scan to return document keys
        mock_async_redis.scan.return_value = (0, [b"doc:source_123:1", b"doc:source_123:2"])
        
        result = await redis_store.remove_source_async(source_id)
        
        assert result > 0
        mock_async_redis.delete.assert_called()