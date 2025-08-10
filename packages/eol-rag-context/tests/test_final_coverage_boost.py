"""
Final coverage boost - comprehensive testing of all modules.
This file aims to reach 80% coverage by testing all remaining code paths.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock, patch, PropertyMock, ANY, call
import numpy as np
import json
import tempfile
import os
import asyncio
from dataclasses import asdict, fields
import hashlib
from collections import deque
import time
from io import StringIO, BytesIO

# Comprehensive mocking of all external dependencies
mock_modules = {
    'magic': MagicMock(),
    'pypdf': MagicMock(),
    'pypdf.PdfReader': MagicMock(),
    'docx': MagicMock(),
    'docx.Document': MagicMock(),
    'redis': MagicMock(),
    'redis.asyncio': MagicMock(),
    'redis.commands': MagicMock(),
    'redis.commands.search': MagicMock(),
    'redis.commands.search.field': MagicMock(),
    'redis.commands.search.indexDefinition': MagicMock(),
    'redis.commands.search.query': MagicMock(),
    'watchdog': MagicMock(),
    'watchdog.observers': MagicMock(),
    'watchdog.events': MagicMock(),
    'networkx': MagicMock(),
    'sentence_transformers': MagicMock(),
    'openai': MagicMock(),
    'tree_sitter': MagicMock(),
    'tree_sitter_python': MagicMock(),
    'yaml': MagicMock(),
    'bs4': MagicMock(),
    'aiofiles': MagicMock(),
    'typer': MagicMock(),
    'rich': MagicMock(),
    'rich.console': MagicMock(),
    'fastmcp': MagicMock(),
    'fastmcp.server': MagicMock(),
}

for name, mock in mock_modules.items():
    if name not in sys.modules:
        sys.modules[name] = mock

# Import all modules for comprehensive testing
from eol.rag_context import config
from eol.rag_context import embeddings
from eol.rag_context import document_processor
from eol.rag_context import indexer
from eol.rag_context import redis_client
from eol.rag_context import semantic_cache
from eol.rag_context import knowledge_graph
from eol.rag_context import file_watcher
from eol.rag_context import server
from eol.rag_context import main


# ============================================================================
# EMBEDDINGS MODULE - Complete Coverage
# ============================================================================

class TestEmbeddingsFinal:
    """Final comprehensive embeddings tests."""
    
    @pytest.mark.asyncio
    async def test_all_embedding_providers(self):
        """Test all embedding provider paths."""
        # Mock provider
        mock_cfg = config.EmbeddingConfig(dimension=64)
        mock_prov = embeddings.MockEmbeddingsProvider(mock_cfg)
        emb = await mock_prov.embed("test")
        assert emb.shape == (1, 64)
        embs = await mock_prov.embed_batch(["a", "b", "c"], batch_size=1)
        assert embs.shape == (3, 64)
        
        # Sentence Transformer without model
        st_cfg = config.EmbeddingConfig(model_name="unknown", dimension=128)
        st_prov = embeddings.SentenceTransformerProvider(st_cfg)
        assert st_prov.model is None
        emb = await st_prov.embed("test")
        assert emb.shape == (1, 128)
        embs = await st_prov.embed_batch(["a", "b"], batch_size=10)
        assert embs.shape == (2, 128)
        
        # Sentence Transformer with model
        with patch('eol.rag_context.embeddings.SentenceTransformer') as MockST:
            mock_model = MagicMock()
            mock_model.encode = MagicMock(return_value=np.random.randn(384))
            MockST.return_value = mock_model
            
            st_cfg2 = config.EmbeddingConfig(model_name="all-mpnet-base-v2", dimension=768)
            st_prov2 = embeddings.SentenceTransformerProvider(st_cfg2)
            st_prov2.model = mock_model
            
            emb = await st_prov2.embed("test")
            mock_model.encode.assert_called()
            
            mock_model.encode = MagicMock(return_value=np.random.randn(2, 384))
            embs = await st_prov2.embed_batch(["a", "b"], batch_size=10)
            mock_model.encode.assert_called()
    
    @pytest.mark.asyncio
    async def test_openai_provider_complete(self):
        """Test OpenAI provider completely."""
        # Without API key
        cfg = config.EmbeddingConfig(provider="openai")
        with pytest.raises(ValueError, match="OpenAI API key required"):
            embeddings.OpenAIProvider(cfg)
        
        # With API key
        cfg = config.EmbeddingConfig(
            provider="openai",
            openai_api_key="test-key",
            openai_model="text-embedding-3-large",
            dimension=3072,
            batch_size=10
        )
        
        with patch('eol.rag_context.embeddings.AsyncOpenAI') as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            
            provider = embeddings.OpenAIProvider(cfg)
            
            # Single embedding
            mock_resp = MagicMock()
            mock_resp.data = [MagicMock(embedding=[0.1] * 3072)]
            mock_client.embeddings.create = AsyncMock(return_value=mock_resp)
            
            emb = await provider.embed("test")
            assert emb.shape == (1, 3072)
            
            # Batch with multiple chunks
            texts = [f"text{i}" for i in range(25)]  # More than batch size
            mock_resp.data = [MagicMock(embedding=[0.1] * 3072) for _ in range(10)]
            
            embs = await provider.embed_batch(texts, batch_size=10)
            assert embs.shape == (25, 3072)
            assert mock_client.embeddings.create.call_count >= 3  # 10+10+5
    
    @pytest.mark.asyncio
    async def test_embedding_manager_complete(self):
        """Test EmbeddingManager completely."""
        # With Redis caching
        redis_mock = MagicMock()
        redis_mock.hget = AsyncMock(return_value=None)
        redis_mock.hset = AsyncMock()
        redis_mock.expire = AsyncMock()
        
        cfg = config.EmbeddingConfig(dimension=32, cache_embeddings=True)
        manager = embeddings.EmbeddingManager(cfg, redis_mock)
        
        # Mock provider
        manager.provider = AsyncMock()
        manager.provider.embed = AsyncMock(
            return_value=np.random.randn(1, 32).astype(np.float32)
        )
        manager.provider.embed_batch = AsyncMock(
            side_effect=lambda texts, **kwargs: np.random.randn(len(texts), 32).astype(np.float32)
        )
        
        # Test caching enabled
        emb = await manager.get_embedding("test", use_cache=True)
        redis_mock.hget.assert_called()
        redis_mock.hset.assert_called()
        assert manager.cache_stats["misses"] == 1
        
        # Test cache hit
        redis_mock.hget.return_value = np.random.randn(32).astype(np.float32).tobytes()
        emb = await manager.get_embedding("cached", use_cache=True)
        assert manager.cache_stats["hits"] == 1
        
        # Test batch processing
        embs = await manager.get_embeddings(["a", "b", "c"], use_cache=False, batch_size=2)
        assert embs.shape == (3, 32)
        
        # Test cache disabled
        manager2 = embeddings.EmbeddingManager(cfg)
        manager2.provider = manager.provider
        emb = await manager2.get_embedding("test", use_cache=False)
        assert emb is not None
        
        # Test cache error handling
        redis_mock.hset = AsyncMock(side_effect=Exception("Redis error"))
        emb = await manager.get_embedding("error", use_cache=True)
        assert emb is not None  # Should still work
        
        # Test stats
        stats = manager.get_cache_stats()
        assert stats["total"] > 0
        assert stats["hit_rate"] > 0


# ============================================================================
# DOCUMENT PROCESSOR MODULE - Complete Coverage
# ============================================================================

class TestDocumentProcessorFinal:
    """Final comprehensive document processor tests."""
    
    @pytest.mark.asyncio
    async def test_process_all_file_types_complete(self):
        """Test processing all supported file types."""
        proc = document_processor.DocumentProcessor(
            config.DocumentConfig(),
            config.ChunkingConfig()
        )
        
        # Mock aiofiles for text files
        with patch('eol.rag_context.document_processor.aiofiles') as mock_aiofiles:
            mock_file = MagicMock()
            mock_file.read = AsyncMock(return_value="Test content")
            mock_aiofiles.open = MagicMock(return_value=mock_file)
            mock_file.__aenter__ = AsyncMock(return_value=mock_file)
            mock_file.__aexit__ = AsyncMock()
            
            # Text file
            doc = await proc._process_text(Path("/test.txt"))
            assert doc.doc_type == "text"
            assert doc.content == "Test content"
            
            # Markdown file
            mock_file.read = AsyncMock(return_value="# Title\n\n## Section\n\nContent")
            doc = await proc._process_markdown(Path("/test.md"))
            assert doc.doc_type == "markdown"
            assert len(doc.chunks) > 0
            assert doc.metadata.get("headers") is not None
            
            # Code file
            mock_file.read = AsyncMock(return_value="def test():\n    pass\n\nclass Test:\n    pass")
            doc = await proc._process_code(Path("/test.py"))
            assert doc.doc_type == "code"
            assert doc.language == "python"
            
            # HTML file
            mock_file.read = AsyncMock(return_value="<html><body><h1>Title</h1><p>Content</p></body></html>")
            doc = await proc._process_html(Path("/test.html"))
            assert doc.doc_type == "html"
            assert len(doc.metadata.get("headers", [])) > 0
        
        # PDF file
        with patch('eol.rag_context.document_processor.PdfReader') as MockPdf:
            mock_reader = MagicMock()
            mock_page = MagicMock()
            mock_page.extract_text = MagicMock(return_value="Page content")
            mock_reader.pages = [mock_page, mock_page]
            MockPdf.return_value = mock_reader
            
            doc = await proc._process_pdf(Path("/test.pdf"))
            assert doc.doc_type == "pdf"
            assert "Page content" in doc.content
            assert len(doc.chunks) > 0
        
        # DOCX file
        with patch('eol.rag_context.document_processor.Document') as MockDocx:
            mock_doc = MagicMock()
            mock_para = MagicMock()
            mock_para.text = "Paragraph text"
            mock_doc.paragraphs = [mock_para]
            
            mock_table = MagicMock()
            mock_row = MagicMock()
            mock_cell = MagicMock()
            mock_cell.text = "Cell text"
            mock_row.cells = [mock_cell]
            mock_table.rows = [mock_row]
            mock_doc.tables = [mock_table]
            
            MockDocx.return_value = mock_doc
            
            doc = await proc._process_docx(Path("/test.docx"))
            assert doc.doc_type == "docx"
            assert "Paragraph text" in doc.content
            assert "Cell text" in doc.content
        
        # Structured files
        with tempfile.NamedTemporaryFile(suffix=".json", mode='w', delete=False) as f:
            json.dump({"key": "value", "list": [1, 2, 3], "nested": {"a": "b"}}, f)
            f.flush()
            doc = await proc._process_structured(Path(f.name))
            assert doc.doc_type == "json"
            assert len(doc.chunks) > 0
        Path(f.name).unlink()
        
        # YAML file
        with patch('eol.rag_context.document_processor.yaml') as mock_yaml:
            mock_yaml.safe_load = MagicMock(return_value={
                "key": "value",
                "list": [1, 2],
                "nested": {"a": "b"}
            })
            
            with tempfile.NamedTemporaryFile(suffix=".yaml", mode='w', delete=False) as f:
                f.write("key: value\nlist:\n  - 1\n  - 2")
                f.flush()
                doc = await proc._process_structured(Path(f.name))
                assert doc.doc_type == "yaml"
            Path(f.name).unlink()
    
    @pytest.mark.asyncio
    async def test_process_file_with_detection(self):
        """Test file processing with type detection."""
        proc = document_processor.DocumentProcessor(
            config.DocumentConfig(),
            config.ChunkingConfig()
        )
        
        with patch('eol.rag_context.document_processor.magic') as mock_magic, \
             patch('eol.rag_context.document_processor.aiofiles') as mock_aiofiles:
            
            mock_file = MagicMock()
            mock_file.read = AsyncMock(return_value="content")
            mock_aiofiles.open = MagicMock(return_value=mock_file)
            mock_file.__aenter__ = AsyncMock(return_value=mock_file)
            mock_file.__aexit__ = AsyncMock()
            
            # Test with various MIME types
            test_cases = [
                ("text/plain", "text"),
                ("text/markdown", "markdown"),
                ("text/html", "html"),
                ("application/pdf", "pdf"),
                ("application/json", "json"),
                ("application/x-yaml", "yaml"),
                ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "docx"),
            ]
            
            for mime_type, expected_type in test_cases:
                mock_magic.from_file = MagicMock(return_value=mime_type)
                
                # Create temp file with appropriate extension
                ext = f".{expected_type}" if expected_type != "yaml" else ".yml"
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                    f.write(b"test")
                    f.flush()
                    
                    if expected_type in ["pdf", "docx"]:
                        # Mock binary file processing
                        if expected_type == "pdf":
                            with patch('eol.rag_context.document_processor.PdfReader'):
                                doc = await proc.process_file(Path(f.name))
                        else:
                            with patch('eol.rag_context.document_processor.Document'):
                                doc = await proc.process_file(Path(f.name))
                    else:
                        doc = await proc.process_file(Path(f.name))
                    
                    assert doc is not None
                
                Path(f.name).unlink()
    
    def test_all_chunking_methods(self):
        """Test all chunking methods comprehensively."""
        # Test semantic chunking
        cfg = config.ChunkingConfig(
            use_semantic_chunking=True,
            max_chunk_size=50,
            min_chunk_size=10,
            chunk_overlap=5
        )
        proc = document_processor.DocumentProcessor(config.DocumentConfig(), cfg)
        
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = proc._chunk_text(text)
        assert all(c["type"] == "semantic" for c in chunks)
        
        # Test simple chunking
        cfg.use_semantic_chunking = False
        proc = document_processor.DocumentProcessor(config.DocumentConfig(), cfg)
        chunks = proc._chunk_text(text)
        assert all(c["type"] == "simple" for c in chunks)
        
        # Test markdown chunking
        md = "# H1\nContent\n## H2\nMore\n### H3\nEven more"
        chunks = proc._chunk_markdown_by_headers(md)
        assert len(chunks) >= 3
        assert chunks[0]["header"] == "H1"
        assert chunks[0]["level"] == 1
        
        # Test code chunking
        code = "def func1():\n    pass\n\ndef func2():\n    pass"
        chunks = proc._chunk_code_by_lines(code, "python")
        assert all(c["language"] == "python" for c in chunks)
        assert all("start_line" in c for c in chunks)
        
        # Test code with AST
        with patch('eol.rag_context.document_processor.Parser') as MockParser:
            mock_parser = MagicMock()
            mock_tree = MagicMock()
            mock_root = MagicMock()
            
            mock_func = MagicMock()
            mock_func.type = "function_definition"
            mock_func.start_point = (0, 0)
            mock_func.end_point = (2, 0)
            mock_func.text = b"def test():\n    pass"
            
            mock_class = MagicMock()
            mock_class.type = "class_definition"
            mock_class.start_point = (4, 0)
            mock_class.end_point = (6, 0)
            mock_class.text = b"class Test:\n    pass"
            
            mock_root.children = [mock_func, mock_class]
            mock_tree.root_node = mock_root
            mock_parser.parse = MagicMock(return_value=mock_tree)
            MockParser.return_value = mock_parser
            
            chunks = proc._chunk_code_by_ast(code, mock_parser, "python")
            assert len(chunks) >= 2
            assert any(c["type"] == "function" for c in chunks)
            assert any(c["type"] == "class" for c in chunks)
        
        # Test PDF chunking
        pages = ["Page 1 content\n\nParagraph", "Page 2 content"]
        chunks = proc._chunk_pdf_content(pages)
        assert len(chunks) == 2
        assert chunks[0]["page"] == 1
        
        # Test structured data chunking
        data = {
            "key1": "value1",
            "key2": {"nested": "value"},
            "list": [1, 2, 3]
        }
        chunks = proc._chunk_structured_data(data, "json")
        assert any(c["type"] == "object_field" for c in chunks)
        
        # Test list chunking
        data = ["item1", "item2", "item3"]
        chunks = proc._chunk_structured_data(data, "json")
        assert all(c["type"] == "array_item" for c in chunks)
        assert len(chunks) == 3
    
    def test_extraction_methods(self):
        """Test content extraction methods."""
        proc = document_processor.DocumentProcessor(
            config.DocumentConfig(),
            config.ChunkingConfig()
        )
        
        # Test header extraction
        from bs4 import BeautifulSoup
        html = "<h1>Title</h1><h2>Subtitle</h2><h3>Section</h3><h4>Subsection</h4>"
        soup = BeautifulSoup(html, 'html.parser')
        headers = proc._extract_headers(soup)
        assert len(headers) == 4
        assert headers[0]["level"] == 1
        assert headers[0]["text"] == "Title"
        
        # Test text content extraction
        html = "<p>Paragraph 1</p><div>Content</div><span>More</span>"
        soup = BeautifulSoup(html, 'html.parser')
        content = proc._extract_text_content(soup)
        assert "Paragraph 1" in content
        assert "Content" in content
        assert "More" in content
    
    def test_language_detection_complete(self):
        """Test complete language detection."""
        proc = document_processor.DocumentProcessor(
            config.DocumentConfig(),
            config.ChunkingConfig()
        )
        
        test_cases = [
            (".py", "python"),
            (".js", "javascript"),
            (".ts", "typescript"),
            (".java", "java"),
            (".go", "go"),
            (".rs", "rust"),
            (".cpp", "cpp"),
            (".c", "c"),
            (".cs", "csharp"),
            (".rb", "ruby"),
            (".php", "php"),
            (".swift", "swift"),
            (".kt", "kotlin"),
            (".scala", "scala"),
            (".r", "r"),
            (".m", "matlab"),
            (".jl", "julia"),
            (".sh", "bash"),
            (".ps1", "powershell"),
            (".lua", "lua"),
            (".pl", "perl"),
            (".unknown", "unknown"),
        ]
        
        for ext, expected in test_cases:
            assert proc._detect_language(ext) == expected


# ============================================================================
# INDEXER MODULE - Complete Coverage
# ============================================================================

class TestIndexerFinal:
    """Final comprehensive indexer tests."""
    
    @pytest.mark.asyncio
    async def test_complete_indexing_pipeline(self):
        """Test complete indexing pipeline with all features."""
        cfg = config.RAGConfig()
        proc = MagicMock()
        emb = MagicMock()
        redis = MagicMock()
        
        idx = indexer.DocumentIndexer(cfg, proc, emb, redis)
        
        # Mock document processing
        test_doc = document_processor.ProcessedDocument(
            file_path=Path("/test.md"),
            content="# Main Title\n\n" + "Test content " * 200,  # Long content
            doc_type="markdown",
            metadata={"headers": ["Main Title", "Section 1"]},
            chunks=[
                {"content": "chunk1", "header": "Main Title", "type": "header"},
                {"content": "chunk2", "header": "Section 1", "type": "paragraph"},
                {"content": "chunk3", "type": "paragraph"}
            ]
        )
        proc.process_file = AsyncMock(return_value=test_doc)
        
        # Mock embeddings
        emb.get_embedding = AsyncMock(return_value=np.random.randn(128))
        emb.get_embeddings = AsyncMock(
            side_effect=lambda texts, **kwargs: np.random.randn(len(texts), 128)
        )
        
        # Mock Redis
        redis.store_document = AsyncMock()
        redis.delete_by_source = AsyncMock()
        redis.list_sources = AsyncMock(return_value=[
            {"source_id": "src1", "path": "/path1"},
            {"source_id": "src2", "path": "/path2"}
        ])
        
        # Test file indexing
        await idx.index_file(Path("/test.md"), "src123")
        assert idx.stats["documents_indexed"] == 1
        assert idx.stats["chunks_created"] > 0
        
        # Test folder indexing
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "test1.md").write_text("# Doc 1")
            (tmpdir / "test2.py").write_text("def test(): pass")
            (tmpdir / "subdir").mkdir()
            (tmpdir / "subdir" / "test3.txt").write_text("text")
            
            # Index with patterns and recursion
            result = await idx.index_folder(
                tmpdir,
                recursive=True,
                file_patterns=["*.md", "*.py"]
            )
            
            assert result is not None
            assert result.source_id is not None
            assert result.file_count >= 2
        
        # Test source management
        success = await idx.remove_source("src123")
        assert success
        redis.delete_by_source.assert_called_with("src123")
        
        sources = await idx.list_sources()
        assert len(sources) == 2
        
        # Test stats
        stats = idx.get_stats()
        assert stats["documents_indexed"] > 0
        assert stats["chunks_created"] > 0
    
    @pytest.mark.asyncio
    async def test_hierarchical_extraction(self):
        """Test hierarchical document extraction."""
        idx = indexer.DocumentIndexer(
            config.RAGConfig(),
            MagicMock(),
            MagicMock(),
            MagicMock()
        )
        
        idx.embeddings.get_embedding = AsyncMock(return_value=np.random.randn(128))
        idx.embeddings.get_embeddings = AsyncMock(
            side_effect=lambda texts, **kwargs: np.random.randn(len(texts), 128)
        )
        idx.redis.store_document = AsyncMock()
        
        # Create document with enough content for all levels
        doc = document_processor.ProcessedDocument(
            file_path=Path("/test.md"),
            content="# Main\n\n" + "Content " * 300,  # Very long content
            doc_type="markdown",
            chunks=[
                {"content": "Section " + str(i), "header": f"Header {i}"}
                for i in range(10)
            ]
        )
        
        metadata = indexer.DocumentMetadata(
            source_path="/test.md",
            source_id="test",
            relative_path="test.md",
            file_type="markdown",
            file_size=1000,
            file_hash="abc",
            modified_time=0,
            indexed_at=0,
            chunk_index=0,
            total_chunks=10,
            hierarchy_level=1
        )
        
        # Extract all levels
        concepts = await idx._extract_concepts(doc, metadata)
        assert len(concepts) > 0
        assert all(c.hierarchy_level == 1 for c in concepts)
        
        if concepts:
            sections = await idx._extract_sections(doc, metadata, concepts[0].id)
            assert len(sections) > 0
            assert all(s.hierarchy_level == 2 for s in sections)
        
        chunks = await idx._extract_chunks(doc, metadata)
        assert len(chunks) == len(doc.chunks)
        assert all(c.hierarchy_level == 3 for c in chunks)
    
    def test_folder_scanner_complete(self):
        """Test folder scanner completely."""
        scanner = indexer.FolderScanner(config.RAGConfig())
        
        # Test source ID generation
        id1 = scanner.generate_source_id(Path("/test1"))
        id2 = scanner.generate_source_id(Path("/test1"))
        id3 = scanner.generate_source_id(Path("/test2"))
        assert id1 == id2
        assert id1 != id3
        assert len(id1) == 16
        
        # Test ignore patterns
        patterns = scanner._default_ignore_patterns()
        assert len(patterns) > 0
        
        # Test should ignore
        test_cases = [
            (Path(".git/config"), True),
            (Path("node_modules/package.json"), True),
            (Path("__pycache__/module.pyc"), True),
            (Path(".DS_Store"), True),
            (Path("src/main.py"), False),
            (Path("README.md"), False),
        ]
        
        for path, should_ignore in test_cases:
            assert scanner._should_ignore(path) == should_ignore
        
        # Test file size limit
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"x" * (101 * 1024 * 1024))  # 101MB
            f.flush()
            assert scanner._should_ignore(Path(f.name))
        Path(f.name).unlink()
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"x" * 1024)  # 1KB
            f.flush()
            assert not scanner._should_ignore(Path(f.name))
        Path(f.name).unlink()
        
        # Test git metadata
        with patch('eol.rag_context.indexer.subprocess.run') as mock_run:
            # Git repo exists
            mock_run.side_effect = [
                MagicMock(returncode=0, stdout="main"),
                MagicMock(returncode=0, stdout="abc123"),
                MagicMock(returncode=0, stdout="user@example.com"),
            ]
            meta = scanner._get_git_metadata(Path("/repo"))
            assert "git_branch" in meta
            assert meta["git_branch"] == "main"
            
            # Not a git repo
            mock_run.side_effect = [
                MagicMock(returncode=1, stdout=""),
            ]
            meta = scanner._get_git_metadata(Path("/not-repo"))
            assert meta == {}
    
    @pytest.mark.asyncio
    async def test_scan_folder_complete(self):
        """Test folder scanning completely."""
        scanner = indexer.FolderScanner(config.RAGConfig())
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create test structure
            (tmpdir / "file1.py").write_text("code1")
            (tmpdir / "file2.md").write_text("doc1")
            (tmpdir / "file3.txt").write_text("text1")
            (tmpdir / ".git").mkdir()
            (tmpdir / ".git" / "config").write_text("git")
            (tmpdir / "subdir").mkdir()
            (tmpdir / "subdir" / "file4.py").write_text("code2")
            (tmpdir / "subdir" / "file5.md").write_text("doc2")
            
            # Scan non-recursive
            files = await scanner.scan_folder(tmpdir, recursive=False)
            assert len(files) == 3  # Excludes .git
            
            # Scan recursive
            files = await scanner.scan_folder(tmpdir, recursive=True)
            assert len(files) == 5  # Excludes .git
            
            # Scan with patterns
            files = await scanner.scan_folder(
                tmpdir,
                recursive=True,
                file_patterns=["*.py"]
            )
            assert len(files) == 2
            assert all(f.suffix == ".py" for f in files)
            
            # Scan with ignore patterns
            files = await scanner.scan_folder(
                tmpdir,
                recursive=True,
                ignore_patterns=["**/subdir/**"]
            )
            assert len(files) == 3
            assert all("subdir" not in str(f) for f in files)
        
        # Test error cases
        with pytest.raises(ValueError, match="does not exist"):
            await scanner.scan_folder(Path("/nonexistent"))
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test")
            f.flush()
            with pytest.raises(ValueError, match="not a directory"):
                await scanner.scan_folder(Path(f.name))
        Path(f.name).unlink()
    
    def test_document_metadata_complete(self):
        """Test DocumentMetadata completely."""
        # Test with all fields
        meta = indexer.DocumentMetadata(
            source_path="/test/file.py",
            source_id="src123",
            relative_path="file.py",
            file_type="code",
            file_size=1024,
            file_hash="hash123",
            modified_time=123.0,
            indexed_at=124.0,
            chunk_index=5,
            total_chunks=10,
            hierarchy_level=3,
            language="python",
            line_start=10,
            line_end=20,
            parent_chunk_id="parent",
            chunk_type="function",
            semantic_density=0.8,
            git_commit="abc123",
            git_branch="main",
            tags=["test", "function"]
        )
        
        # Test as dict
        meta_dict = asdict(meta)
        assert meta_dict["source_id"] == "src123"
        assert meta_dict["language"] == "python"
        assert meta_dict["semantic_density"] == 0.8
        assert "test" in meta_dict["tags"]
    
    def test_indexed_source_complete(self):
        """Test IndexedSource completely."""
        source = indexer.IndexedSource(
            source_id="src123",
            path=Path("/project"),
            indexed_at=123.0,
            file_count=10,
            total_chunks=50,
            metadata={"version": "1.0", "author": "test"}
        )
        
        assert source.source_id == "src123"
        assert source.file_count == 10
        assert source.metadata["version"] == "1.0"


# ============================================================================
# Run all tests to maximize coverage
# ============================================================================

@pytest.mark.asyncio
async def test_run_all_final_tests():
    """Run all final tests to maximize coverage."""
    # This test ensures all test classes are instantiated and run
    test_classes = [
        TestEmbeddingsFinal(),
        TestDocumentProcessorFinal(),
        TestIndexerFinal(),
    ]
    
    for test_class in test_classes:
        # Run async test methods
        for attr_name in dir(test_class):
            if attr_name.startswith("test_") and asyncio.iscoroutinefunction(getattr(test_class, attr_name)):
                await getattr(test_class, attr_name)()
            elif attr_name.startswith("test_") and callable(getattr(test_class, attr_name)):
                getattr(test_class, attr_name)()