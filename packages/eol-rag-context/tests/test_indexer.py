"""
Unit tests for document indexer.
"""

import pytest
from pathlib import Path
import hashlib

from eol.rag_context.indexer import DocumentIndexer, FolderScanner, DocumentMetadata


class TestFolderScanner:
    """Test folder scanner functionality."""
    
    @pytest.fixture
    def scanner(self, test_config):
        """Create folder scanner."""
        return FolderScanner(test_config)
    
    def test_default_ignore_patterns(self, scanner):
        """Test default ignore patterns."""
        patterns = scanner._default_ignore_patterns()
        
        assert "**/.git/**" in patterns
        assert "**/node_modules/**" in patterns
        assert "**/__pycache__/**" in patterns
    
    def test_should_ignore(self, scanner, temp_dir):
        """Test file ignore logic."""
        # Create test files
        git_file = temp_dir / ".git" / "config"
        git_file.parent.mkdir()
        git_file.touch()
        
        normal_file = temp_dir / "test.py"
        normal_file.touch()
        
        assert scanner._should_ignore(git_file)
        assert not scanner._should_ignore(normal_file)
    
    @pytest.mark.asyncio
    async def test_scan_folder(self, scanner, sample_documents):
        """Test folder scanning."""
        folder = sample_documents["markdown"].parent
        
        files = await scanner.scan_folder(folder, recursive=False)
        
        assert len(files) > 0
        assert all(f.exists() for f in files)
        # Resolve paths for comparison
        resolved_files = [f.resolve() for f in files]
        assert sample_documents["markdown"].resolve() in resolved_files
    
    @pytest.mark.asyncio
    async def test_scan_with_patterns(self, scanner, sample_documents):
        """Test scanning with file patterns."""
        folder = sample_documents["markdown"].parent
        
        # Only Python files
        files = await scanner.scan_folder(
            folder,
            recursive=False,
            file_patterns=["*.py"]
        )
        
        assert len(files) == 1
        assert files[0].suffix == ".py"
    
    def test_generate_source_id(self, scanner, temp_dir):
        """Test source ID generation."""
        source_id = scanner.generate_source_id(temp_dir)
        
        assert isinstance(source_id, str)
        assert len(source_id) == 16  # MD5 truncated to 16 chars
        
        # Should be deterministic
        source_id2 = scanner.generate_source_id(temp_dir)
        assert source_id == source_id2


class TestDocumentIndexer:
    """Test document indexer functionality."""
    
    @pytest.mark.asyncio
    async def test_index_file(self, indexed_documents, sample_documents):
        """Test file indexing."""
        indexer = indexed_documents
        stats = indexer.get_stats()
        
        assert stats["documents_indexed"] > 0
        assert stats["chunks_created"] > 0
        assert stats["errors"] == 0
    
    @pytest.mark.asyncio
    async def test_index_folder(
        self,
        redis_store,
        mock_embedding_manager,
        test_config,
        sample_documents
    ):
        """Test folder indexing."""
        from eol.rag_context.document_processor import DocumentProcessor
        
        processor = DocumentProcessor(test_config.document, test_config.chunking)
        indexer = DocumentIndexer(
            test_config,
            processor,
            mock_embedding_manager,
            redis_store
        )
        
        folder = sample_documents["markdown"].parent
        result = await indexer.index_folder(folder, recursive=False)
        
        assert result.file_count > 0
        assert result.total_chunks > 0
        assert result.path == folder
    
    @pytest.mark.asyncio
    async def test_extract_concepts(self, indexed_documents, sample_documents):
        """Test concept extraction."""
        indexer = indexed_documents
        stats = indexer.get_stats()
        
        # Should have extracted concepts
        assert stats["concepts_extracted"] > 0
    
    @pytest.mark.asyncio
    async def test_extract_sections(self, indexed_documents):
        """Test section extraction."""
        indexer = indexed_documents
        stats = indexer.get_stats()
        
        # Should have created sections
        assert stats["sections_created"] > 0
    
    @pytest.mark.asyncio
    async def test_metadata_tracking(self, indexed_documents, sample_documents):
        """Test metadata tracking."""
        # The metadata should include file information
        indexer = indexed_documents
        
        # Check that files were indexed with metadata
        assert indexer.stats["documents_indexed"] == len(sample_documents)
    
    @pytest.mark.asyncio
    async def test_remove_source(
        self,
        redis_store,
        mock_embedding_manager,
        test_config,
        sample_documents
    ):
        """Test removing indexed source."""
        from eol.rag_context.document_processor import DocumentProcessor
        
        processor = DocumentProcessor(test_config.document, test_config.chunking)
        indexer = DocumentIndexer(
            test_config,
            processor,
            mock_embedding_manager,
            redis_store
        )
        
        # Index a file
        py_file = sample_documents["python"]
        await indexer.index_file(py_file)
        
        # Get source ID
        source_id = indexer.scanner.generate_source_id(py_file.parent)
        
        # Remove source
        success = await indexer.remove_source(source_id)
        assert success
        
        # Try to remove again - should fail
        success = await indexer.remove_source(source_id)
        assert not success
    
    @pytest.mark.asyncio
    async def test_list_sources(
        self,
        redis_store,
        mock_embedding_manager,
        test_config,
        sample_documents
    ):
        """Test listing indexed sources."""
        from eol.rag_context.document_processor import DocumentProcessor
        
        processor = DocumentProcessor(test_config.document, test_config.chunking)
        indexer = DocumentIndexer(
            test_config,
            processor,
            mock_embedding_manager,
            redis_store
        )
        
        # Index folder
        folder = sample_documents["markdown"].parent
        await indexer.index_folder(folder)
        
        # List sources
        sources = await indexer.list_sources()
        
        assert len(sources) > 0
        assert sources[0].path == folder