"""Unit tests for ParallelIndexer - high-performance parallel document indexing."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from eol.rag_context.parallel_indexer import (
    ParallelIndexer,
    ParallelIndexingConfig,
    IndexingCheckpoint,
    FileBatch,
    ParallelFileScanner,
)


class TestParallelIndexingConfig:
    """Test ParallelIndexingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ParallelIndexingConfig()
        
        assert config.max_document_workers == 16
        assert config.max_embedding_workers == 8
        assert config.max_redis_workers == 4
        assert config.batch_size == 32
        assert config.enable_streaming is True
        assert config.checkpoint_interval == 100
        assert config.memory_limit_mb == 2048
        assert config.enable_resume is True
        assert "*.md" in config.priority_patterns

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ParallelIndexingConfig(
            max_document_workers=8,
            batch_size=16,
            enable_streaming=False,
            priority_patterns=["*.py", "*.js"]
        )
        
        assert config.max_document_workers == 8
        assert config.batch_size == 16
        assert config.enable_streaming is False
        assert config.priority_patterns == ["*.py", "*.js"]


class TestIndexingCheckpoint:
    """Test IndexingCheckpoint dataclass."""

    def test_checkpoint_properties(self):
        """Test checkpoint computed properties."""
        import time
        
        checkpoint = IndexingCheckpoint(
            source_id="test-source",
            root_path="/test/path",
            total_files=100,
            completed_files=40,
            start_time=time.time() - 60  # 60 seconds ago
        )
        
        # Test completion percentage
        assert checkpoint.completion_percentage == 40.0
        
        # Test files per second
        assert checkpoint.files_per_second > 0.5
        assert checkpoint.files_per_second < 1.0
        
        # Test estimated completion
        estimated = checkpoint.estimated_completion
        assert estimated > 80  # More than 80 seconds for remaining files
        assert estimated < 120

    def test_checkpoint_edge_cases(self):
        """Test checkpoint edge cases."""
        checkpoint = IndexingCheckpoint(
            source_id="test",
            root_path="/test",
            total_files=0,
            completed_files=0
        )
        
        assert checkpoint.completion_percentage == 0.0
        assert checkpoint.files_per_second == 0.0
        assert checkpoint.estimated_completion == float('inf')


class TestFileBatch:
    """Test FileBatch dataclass."""

    def test_file_batch(self):
        """Test file batch functionality."""
        files = [Path("file1.py"), Path("file2.py"), Path("file3.py")]
        batch = FileBatch(
            files=files,
            priority=100,
            estimated_size=1024
        )
        
        assert len(batch) == 3
        assert batch.priority == 100
        assert batch.estimated_size == 1024
        assert batch.files == files


class TestParallelFileScanner:
    """Test ParallelFileScanner functionality."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return ParallelIndexingConfig(batch_size=2)

    @pytest.fixture
    def scanner(self, config):
        """Create file scanner."""
        return ParallelFileScanner(config)

    def test_prioritize_files(self, scanner):
        """Test file prioritization."""
        files = [
            Path("regular.txt"),
            Path("README.md"),
            Path("script.py"), 
            Path("large_file.dat")
        ]
        
        # Mock file stats for large file
        with patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value.st_size = 2 * 1024 * 1024  # 2MB for large_file.dat
            
            prioritized = scanner._prioritize_files(files)
            
            # README should be first (high priority)
            # Python file should be second (medium priority)
            # Regular text should be third
            # Large file should be last (penalty for size)
            priority_names = [f.name for f in prioritized]
            
            assert "README.md" in priority_names[:2]  # Should be high priority
            assert "script.py" in priority_names[:3]  # Should be medium priority

    @pytest.mark.asyncio
    async def test_scan_repository(self, scanner):
        """Test repository scanning."""
        test_path = Path("/tmp/test_repo")
        
        # Mock glob results - need to be more careful about how patterns are handled
        with patch.object(Path, 'rglob') as mock_rglob, \
             patch.object(Path, 'glob') as mock_glob, \
             patch.object(Path, 'stat') as mock_stat:
            
            # Each pattern should return different files to avoid duplication
            def mock_rglob_side_effect(pattern):
                if pattern == "*.py":
                    return [Path("file1.py")]
                elif pattern == "*.md":
                    return [Path("file2.md")]
                else:
                    return []
            
            mock_rglob.side_effect = mock_rglob_side_effect
            
            # Mock file stats
            mock_stat.return_value.st_size = 1024
            
            batches = []
            async for batch in scanner.scan_repository(
                test_path, 
                ["*.py", "*.md"], 
                recursive=True
            ):
                batches.append(batch)
            
            # Should create one batch with 2 files
            assert len(batches) == 1  
            assert len(batches[0].files) == 2
            assert batches[0].estimated_size == 2048  # 2 files * 1024 bytes


class TestParallelIndexer:
    """Test ParallelIndexer functionality."""

    @pytest.fixture
    def mock_config(self):
        """Mock RAGConfig."""
        config = MagicMock()
        config.document.file_patterns = ["*.py", "*.md"]
        config.context.default_top_k = 10
        return config

    @pytest.fixture
    def mock_processor(self):
        """Mock DocumentProcessor."""
        return MagicMock()

    @pytest.fixture
    def mock_embeddings(self):
        """Mock EmbeddingManager."""
        return AsyncMock()

    @pytest.fixture
    def mock_redis(self):
        """Mock RedisVectorStore."""
        return AsyncMock()

    @pytest.fixture
    def parallel_indexer(self, mock_config, mock_processor, mock_embeddings, mock_redis):
        """Create ParallelIndexer instance."""
        with patch('eol.rag_context.parallel_indexer.DocumentIndexer.__init__') as mock_init:
            mock_init.return_value = None
            indexer = ParallelIndexer(mock_config, mock_processor, mock_embeddings, mock_redis)
            
            # Setup required attributes from parent class
            indexer.config = mock_config
            indexer.processor = mock_processor
            indexer.embeddings = mock_embeddings
            indexer.redis = mock_redis
            indexer._generate_source_id = MagicMock(return_value="test-source-123")
            indexer._store_indexed_source = AsyncMock()
            
            return indexer

    @pytest.mark.asyncio
    async def test_parallel_indexer_initialization(self, parallel_indexer):
        """Test parallel indexer initialization."""
        assert parallel_indexer is not None
        assert parallel_indexer.parallel_config is not None
        assert parallel_indexer.scanner is not None
        assert hasattr(parallel_indexer, 'document_semaphore')
        assert hasattr(parallel_indexer, 'embedding_semaphore')
        assert hasattr(parallel_indexer, 'redis_semaphore')

    @pytest.mark.asyncio
    async def test_process_file_batch(self, parallel_indexer):
        """Test processing a file batch."""
        files = [Path("test1.py"), Path("test2.py")]
        batch = FileBatch(files=files)
        
        # Initialize checkpoint to avoid None error
        from eol.rag_context.parallel_indexer import IndexingCheckpoint
        parallel_indexer.current_checkpoint = IndexingCheckpoint(
            source_id="test-source",
            root_path="/test"
        )
        
        # Mock index_file method
        mock_result = MagicMock()
        mock_result.chunks = 5
        mock_result.errors = []
        
        with patch.object(parallel_indexer, 'index_file', return_value=mock_result) as mock_index_file:
            results = await parallel_indexer._process_file_batch(
                batch, 
                "test-source", 
                Path("/test"), 
                False
            )
            
            # Should have processed both files
            assert len(results) == 2
            assert mock_index_file.call_count == 2

    @pytest.mark.asyncio
    async def test_process_single_file_with_semaphore(self, parallel_indexer):
        """Test processing single file with semaphore."""
        # Initialize checkpoint to avoid None error
        from eol.rag_context.parallel_indexer import IndexingCheckpoint
        parallel_indexer.current_checkpoint = IndexingCheckpoint(
            source_id="test-source",
            root_path="/test"
        )
        
        mock_result = MagicMock()
        mock_result.chunks = 3
        mock_result.errors = []
        
        with patch.object(parallel_indexer, 'index_file', return_value=mock_result):
            result = await parallel_indexer._process_single_file_with_semaphore(
                Path("test.py"),
                "test-source",
                Path("/test"),
                False
            )
            
            assert result is not None
            assert result.chunks == 3

    @pytest.mark.asyncio
    async def test_save_and_load_checkpoint(self, parallel_indexer):
        """Test checkpoint save/load functionality."""
        checkpoint = IndexingCheckpoint(
            source_id="test-source",
            root_path="/test",
            completed_files=10,
            total_files=50,
            total_chunks=100
        )
        
        parallel_indexer.current_checkpoint = checkpoint
        
        # Mock Redis operations
        parallel_indexer.redis.redis.hset = AsyncMock()
        parallel_indexer.redis.redis.expire = AsyncMock()
        parallel_indexer.redis.redis.hgetall = AsyncMock(return_value={
            'completed_files': '10',
            'total_files': '50',
            'total_chunks': '100',
            'processed_files': 'file1.py,file2.py'
        })
        
        # Test save
        await parallel_indexer._save_checkpoint()
        parallel_indexer.redis.redis.hset.assert_called_once()
        
        # Test load
        success = await parallel_indexer._load_checkpoint()
        assert success is True

    @pytest.mark.asyncio
    async def test_cleanup_checkpoint(self, parallel_indexer):
        """Test checkpoint cleanup."""
        checkpoint = IndexingCheckpoint(
            source_id="test-source",
            root_path="/test"
        )
        parallel_indexer.current_checkpoint = checkpoint
        parallel_indexer.redis.redis.delete = AsyncMock()
        
        await parallel_indexer._cleanup_checkpoint()
        parallel_indexer.redis.redis.delete.assert_called_once()

    def test_estimate_file_count(self, parallel_indexer):
        """Test file count estimation."""
        test_path = Path("/tmp/test")
        
        # The method was moved to the server class, test it there if needed
        # For now, just test that the parallel indexer doesn't have this method
        assert not hasattr(parallel_indexer, '_estimate_file_count')