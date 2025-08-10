"""
Tests specifically targeting uncovered code paths.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock, patch, call, ANY
import numpy as np
import json
import tempfile
import asyncio
from io import BytesIO

# Mock all external dependencies
for module in ['magic', 'pypdf', 'docx', 'redis', 'redis.asyncio', 'redis.commands',
               'redis.commands.search', 'redis.commands.search.field',
               'redis.commands.search.indexDefinition', 'redis.commands.search.query',
               'watchdog', 'watchdog.observers', 'watchdog.events', 'networkx',
               'sentence_transformers', 'openai', 'tree_sitter', 'yaml', 'bs4', 
               'aiofiles', 'typer', 'rich', 'rich.console', 'fastmcp', 'fastmcp.server',
               'tree_sitter_python', 'tree_sitter_javascript', 'tree_sitter_typescript',
               'tree_sitter_go', 'tree_sitter_rust', 'tree_sitter_java']:
    if module not in sys.modules:
        sys.modules[module] = MagicMock()

# Import modules after mocking
from eol.rag_context import (
    config, embeddings, document_processor, indexer,
    redis_client, semantic_cache, knowledge_graph, file_watcher
)


class TestUncoveredEmbeddings:
    """Test uncovered embeddings paths."""
    
    @pytest.mark.asyncio
    async def test_sentence_transformer_with_model(self):
        """Test SentenceTransformer when model is available."""
        cfg = config.EmbeddingConfig(model_name="test-model", dimension=384)
        
        with patch('eol.rag_context.embeddings.SentenceTransformer') as MockST:
            mock_model = MagicMock()
            mock_model.encode = MagicMock(return_value=np.random.randn(1, 384))
            MockST.return_value = mock_model
            
            provider = embeddings.SentenceTransformerProvider(cfg)
            provider.model = mock_model  # Force model to be set
            
            # Test with actual model
            emb = await provider.embed("test")
            assert emb.shape == (1, 384)
            mock_model.encode.assert_called()
            
            # Test batch
            mock_model.encode = MagicMock(return_value=np.random.randn(2, 384))
            embs = await provider.embed_batch(["a", "b"], batch_size=2)
            assert embs.shape == (2, 384)
    
    @pytest.mark.asyncio
    async def test_embedding_manager_cache_store_error(self):
        """Test cache store error handling."""
        cfg = config.EmbeddingConfig(dimension=32)
        redis_mock = MagicMock()
        redis_mock.hset = AsyncMock(side_effect=Exception("Redis error"))
        
        manager = embeddings.EmbeddingManager(cfg, redis_mock)
        manager.provider = AsyncMock()
        manager.provider.embed = AsyncMock(return_value=np.random.randn(1, 32))
        
        # Should handle error gracefully
        emb = await manager.get_embedding("test", use_cache=True)
        assert emb is not None
    
    @pytest.mark.asyncio
    async def test_openai_batch_processing(self):
        """Test OpenAI batch processing with chunks."""
        cfg = config.EmbeddingConfig(
            provider="openai",
            openai_api_key="key",
            batch_size=2,
            dimension=1536
        )
        
        with patch('eol.rag_context.embeddings.AsyncOpenAI') as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            
            # Mock responses for batches
            mock_response = MagicMock()
            mock_response.data = [
                MagicMock(embedding=[0.1] * 1536),
                MagicMock(embedding=[0.2] * 1536)
            ]
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            
            provider = embeddings.OpenAIProvider(cfg)
            
            # Test with 3 texts (will be batched as 2+1)
            embs = await provider.embed_batch(["a", "b", "c"], batch_size=2)
            assert embs.shape == (3, 1536)
            assert mock_client.embeddings.create.call_count >= 2


class TestUncoveredDocumentProcessor:
    """Test uncovered document processor paths."""
    
    @pytest.mark.asyncio
    async def test_process_html_file(self):
        """Test HTML file processing."""
        proc = document_processor.DocumentProcessor(
            config.DocumentConfig(),
            config.ChunkingConfig()
        )
        
        with patch('eol.rag_context.document_processor.aiofiles') as mock_aiofiles:
            mock_file = MagicMock()
            mock_file.read = AsyncMock(return_value="<html><body><h1>Title</h1><p>Content</p></body></html>")
            mock_aiofiles.open = MagicMock(return_value=mock_file)
            mock_file.__aenter__ = AsyncMock(return_value=mock_file)
            mock_file.__aexit__ = AsyncMock()
            
            doc = await proc._process_html(Path("/test.html"))
            assert doc.doc_type == "html"
            assert len(doc.chunks) > 0
    
    @pytest.mark.asyncio
    async def test_process_pdf_with_errors(self):
        """Test PDF processing with errors."""
        proc = document_processor.DocumentProcessor(
            config.DocumentConfig(),
            config.ChunkingConfig()
        )
        
        # Test with empty PDF
        with patch('eol.rag_context.document_processor.PdfReader') as MockPdf:
            MockPdf.side_effect = Exception("Invalid PDF")
            
            doc = await proc._process_pdf(Path("/test.pdf"))
            assert doc.doc_type == "pdf"
            assert doc.content == ""
    
    @pytest.mark.asyncio
    async def test_process_docx_with_tables(self):
        """Test DOCX processing with tables."""
        proc = document_processor.DocumentProcessor(
            config.DocumentConfig(),
            config.ChunkingConfig()
        )
        
        with patch('eol.rag_context.document_processor.Document') as MockDocx:
            mock_doc = MagicMock()
            mock_doc.paragraphs = [MagicMock(text="Para 1")]
            mock_table = MagicMock()
            mock_table.rows = [MagicMock(cells=[MagicMock(text="Cell 1")])]
            mock_doc.tables = [mock_table]
            MockDocx.return_value = mock_doc
            
            doc = await proc._process_docx(Path("/test.docx"))
            assert doc.doc_type == "docx"
            assert "Cell 1" in doc.content
    
    def test_chunk_code_with_tree_sitter(self):
        """Test code chunking with tree-sitter."""
        proc = document_processor.DocumentProcessor(
            config.DocumentConfig(),
            config.ChunkingConfig()
        )
        
        with patch('eol.rag_context.document_processor.Parser') as MockParser:
            mock_parser = MagicMock()
            mock_tree = MagicMock()
            mock_root = MagicMock()
            
            # Mock tree structure
            mock_child = MagicMock()
            mock_child.type = "function_definition"
            mock_child.start_point = (0, 0)
            mock_child.end_point = (2, 0)
            mock_child.text = b"def test():\n    pass"
            
            mock_root.children = [mock_child]
            mock_tree.root_node = mock_root
            mock_parser.parse = MagicMock(return_value=mock_tree)
            MockParser.return_value = mock_parser
            
            code = "def test():\n    pass\n\ndef another():\n    pass"
            chunks = proc._chunk_code_by_ast(code, mock_parser, "python")
            
            assert len(chunks) > 0
            assert chunks[0]["type"] == "function"
    
    def test_semantic_chunking_with_paragraphs(self):
        """Test semantic paragraph chunking."""
        cfg = config.ChunkingConfig(
            use_semantic_chunking=True,
            max_chunk_size=100,
            min_chunk_size=20
        )
        proc = document_processor.DocumentProcessor(
            config.DocumentConfig(),
            cfg
        )
        
        text = """First paragraph with some content that is quite long and should be chunked properly.

Second paragraph with different content that also needs to be processed correctly.

Third paragraph that is shorter.

Fourth paragraph with more text to ensure proper chunking behavior."""
        
        chunks = proc._chunk_text(text)
        assert len(chunks) > 0
        assert all(c["type"] == "semantic" for c in chunks)


class TestUncoveredIndexer:
    """Test uncovered indexer paths."""
    
    @pytest.mark.asyncio
    async def test_index_file_with_errors(self):
        """Test file indexing with errors."""
        idx = indexer.DocumentIndexer(
            config.RAGConfig(),
            MagicMock(),
            MagicMock(),
            MagicMock()
        )
        
        # Mock processor to raise error
        idx.processor.process_file = AsyncMock(side_effect=Exception("Process error"))
        
        await idx.index_file(Path("/test.txt"), "src123")
        
        # Should handle error and update stats
        assert idx.stats["errors"] > 0
    
    @pytest.mark.asyncio
    async def test_extract_sections_from_chunks(self):
        """Test section extraction from chunks."""
        idx = indexer.DocumentIndexer(
            config.RAGConfig(),
            MagicMock(),
            MagicMock(),
            MagicMock()
        )
        
        idx.embeddings.get_embeddings = AsyncMock(
            side_effect=lambda texts, **kwargs: np.random.randn(len(texts), 128)
        )
        idx.redis.store_document = AsyncMock()
        
        doc = document_processor.ProcessedDocument(
            file_path=Path("/test.md"),
            content="Test",
            doc_type="markdown",
            chunks=[
                {"content": "Section 1 content", "header": "Section 1"},
                {"content": "Section 2 content", "header": "Section 2"}
            ]
        )
        
        metadata = indexer.DocumentMetadata(
            source_path="/test.md",
            source_id="test",
            relative_path="test.md",
            file_type="markdown",
            file_size=100,
            file_hash="abc",
            modified_time=0,
            indexed_at=0,
            chunk_index=0,
            total_chunks=2,
            hierarchy_level=2
        )
        
        sections = await idx._extract_sections(doc, metadata, "concept_id")
        assert len(sections) > 0
    
    def test_scanner_git_metadata_with_git(self):
        """Test git metadata extraction with git repo."""
        scanner = indexer.FolderScanner(config.RAGConfig())
        
        with patch('eol.rag_context.indexer.subprocess.run') as mock_run:
            # Mock successful git commands
            mock_run.side_effect = [
                MagicMock(returncode=0, stdout="main"),  # branch
                MagicMock(returncode=0, stdout="abc123"),  # commit
                MagicMock(returncode=0, stdout="test@example.com")  # author
            ]
            
            meta = scanner._get_git_metadata(Path("/test"))
            
            assert "git_branch" in meta
            assert meta["git_branch"] == "main"
            assert "git_commit" in meta
            assert meta["git_commit"] == "abc123"
    
    @pytest.mark.asyncio
    async def test_index_folder_with_patterns(self):
        """Test folder indexing with patterns."""
        idx = indexer.DocumentIndexer(
            config.RAGConfig(),
            MagicMock(),
            MagicMock(),
            MagicMock()
        )
        
        idx.processor.process_file = AsyncMock(
            return_value=document_processor.ProcessedDocument(
                file_path=Path("/test.py"),
                content="test",
                doc_type="code"
            )
        )
        idx.embeddings.get_embeddings = AsyncMock(
            side_effect=lambda texts, **kwargs: np.random.randn(len(texts), 128)
        )
        idx.redis.store_document = AsyncMock()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "test.py").write_text("code")
            (tmpdir / "test.md").write_text("doc")
            (tmpdir / "test.txt").write_text("text")
            
            # Index only Python files
            result = await idx.index_folder(
                tmpdir,
                recursive=False,
                file_patterns=["*.py"]
            )
            
            assert result.file_count == 1


class TestUncoveredSemanticCache:
    """Test uncovered semantic cache paths."""
    
    @pytest.mark.asyncio
    async def test_cache_eviction(self):
        """Test cache eviction when full."""
        cfg = MagicMock()
        cfg.enabled = True
        cfg.max_cache_size = 2
        cfg.similarity_threshold = 0.9
        cfg.adaptive_threshold = False
        cfg.ttl_seconds = 3600
        
        emb = MagicMock()
        emb.get_embedding = AsyncMock(return_value=np.random.randn(128))
        
        redis = MagicMock()
        redis.redis = MagicMock()
        redis.redis.hlen = AsyncMock(return_value=3)  # Over limit
        redis.redis.hgetall = AsyncMock(return_value={
            "cache:1": json.dumps({"timestamp": 1}),
            "cache:2": json.dumps({"timestamp": 2}),
            "cache:3": json.dumps({"timestamp": 3})
        })
        redis.redis.hdel = AsyncMock()
        redis.redis.hset = AsyncMock()
        redis.redis.expire = AsyncMock()
        
        cache = semantic_cache.SemanticCache(cfg, emb, redis)
        
        # Add new entry (should trigger eviction)
        await cache.set("new_query", "new_response", {})
        
        # Should have deleted oldest entry
        redis.redis.hdel.assert_called()
    
    @pytest.mark.asyncio
    async def test_cache_disabled(self):
        """Test cache when disabled."""
        cfg = MagicMock()
        cfg.enabled = False
        
        cache = semantic_cache.SemanticCache(cfg, MagicMock(), MagicMock())
        
        # Get should return None
        result = await cache.get("query")
        assert result is None
        
        # Set should do nothing
        await cache.set("query", "response", {})
        assert cache.stats["queries"] == 0
    
    @pytest.mark.asyncio
    async def test_adaptive_threshold_adjustment(self):
        """Test adaptive threshold adjustment logic."""
        cfg = MagicMock()
        cfg.target_hit_rate = 0.31
        cfg.adaptive_threshold = True
        cfg.similarity_threshold = 0.95
        
        cache = semantic_cache.SemanticCache(cfg, MagicMock(), MagicMock())
        
        # Simulate low hit rate
        cache.stats = {"queries": 100, "hits": 20}  # 20% hit rate
        cache.similarity_scores = [0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.95, 0.98] * 10
        
        await cache._update_adaptive_threshold()
        
        # Should lower threshold to increase hits
        assert cache.adaptive_threshold < 0.95
        
        # Simulate high hit rate
        cache.stats = {"queries": 100, "hits": 50}  # 50% hit rate
        await cache._update_adaptive_threshold()
        
        # Should raise threshold to reduce hits
        assert cache.adaptive_threshold > 0.7


class TestUncoveredKnowledgeGraph:
    """Test uncovered knowledge graph paths."""
    
    @pytest.mark.asyncio
    async def test_extract_entities_from_all_types(self):
        """Test entity extraction from all document types."""
        builder = knowledge_graph.KnowledgeGraphBuilder(MagicMock(), MagicMock())
        
        # Extract from markdown with various patterns
        md_content = """# Main Topic
        
## API Endpoints
- `/api/users` - User management
- `/api/posts` - Post management

## Dependencies
Uses `numpy`, `pandas`, and `scikit-learn`

## Classes
- UserManager
- PostProcessor
- DataAnalyzer

Links to [external resource](http://example.com)
"""
        await builder._extract_markdown_entities(md_content, "doc1", {})
        
        # Extract from code with various patterns
        code_content = """
import numpy as np
from sklearn import model_selection

class DataProcessor:
    def __init__(self):
        self.model = None
    
    def process_data(self, data):
        return np.array(data)

def train_model(X, y):
    '''Train a model'''
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier()
    return clf.fit(X, y)

API_ENDPOINT = "https://api.example.com"
DATABASE_URL = "postgresql://localhost/db"
"""
        await builder._extract_code_entities_from_content(
            code_content, "doc2", {"language": "python"}
        )
        
        # Extract from structured data
        struct_data = {
            "apis": {
                "user_service": {"endpoint": "/users", "methods": ["GET", "POST"]},
                "auth_service": {"endpoint": "/auth", "methods": ["POST"]}
            },
            "features": ["authentication", "authorization", "logging"],
            "dependencies": ["redis", "postgresql", "rabbitmq"]
        }
        await builder._extract_structured_entities(struct_data, "doc3", {"doc_type": "json"})
        
        assert len(builder.entities) > 0
        
        # Check various entity types exist
        entity_types = {e.type for e in builder.entities.values()}
        assert knowledge_graph.EntityType.API in entity_types or \
               knowledge_graph.EntityType.CLASS in entity_types or \
               knowledge_graph.EntityType.DEPENDENCY in entity_types
    
    @pytest.mark.asyncio
    async def test_pattern_discovery_with_relationships(self):
        """Test pattern discovery with complex relationships."""
        builder = knowledge_graph.KnowledgeGraphBuilder(MagicMock(), MagicMock())
        
        # Create entities
        builder.entities = {
            "1": knowledge_graph.Entity("1", "ClassA", knowledge_graph.EntityType.CLASS),
            "2": knowledge_graph.Entity("2", "ClassB", knowledge_graph.EntityType.CLASS),
            "3": knowledge_graph.Entity("3", "function1", knowledge_graph.EntityType.FUNCTION),
            "4": knowledge_graph.Entity("4", "function2", knowledge_graph.EntityType.FUNCTION),
            "5": knowledge_graph.Entity("5", "module1", knowledge_graph.EntityType.MODULE),
        }
        
        # Create relationships
        builder.relationships = [
            knowledge_graph.Relationship("1", "2", knowledge_graph.RelationType.EXTENDS),
            knowledge_graph.Relationship("1", "3", knowledge_graph.RelationType.CONTAINS),
            knowledge_graph.Relationship("3", "4", knowledge_graph.RelationType.CALLS),
            knowledge_graph.Relationship("5", "1", knowledge_graph.RelationType.CONTAINS),
            knowledge_graph.Relationship("5", "2", knowledge_graph.RelationType.CONTAINS),
        ]
        
        # Add to graph
        for e in builder.entities.values():
            builder.graph.add_node(e.id, entity=e)
        for r in builder.relationships:
            builder.graph.add_edge(r.source_id, r.target_id, type=r.type, relationship=r)
        
        # Discover patterns
        await builder._discover_patterns()
        
        # Should have found patterns
        # The actual pattern discovery would identify common structures
    
    @pytest.mark.asyncio
    async def test_export_import_graph(self):
        """Test graph export and import."""
        redis = MagicMock()
        redis.store_entities = AsyncMock()
        redis.store_relationships = AsyncMock()
        
        builder = knowledge_graph.KnowledgeGraphBuilder(redis, MagicMock())
        
        # Create simple graph
        builder.entities = {
            "1": knowledge_graph.Entity("1", "E1", knowledge_graph.EntityType.CONCEPT),
            "2": knowledge_graph.Entity("2", "E2", knowledge_graph.EntityType.FUNCTION),
        }
        builder.relationships = [
            knowledge_graph.Relationship("1", "2", knowledge_graph.RelationType.RELATES_TO)
        ]
        
        # Export
        graph_data = await builder.export_graph()
        assert "nodes" in graph_data
        assert "edges" in graph_data
        assert len(graph_data["nodes"]) == 2
        assert len(graph_data["edges"]) == 1


class TestUncoveredFileWatcher:
    """Test uncovered file watcher paths."""
    
    @pytest.mark.asyncio
    async def test_watcher_event_handling_all_types(self):
        """Test handling of all event types."""
        idx = MagicMock()
        idx.index_file = AsyncMock()
        idx.remove_source = AsyncMock()
        
        watcher = file_watcher.FileWatcher(idx, debounce_seconds=0)
        
        # Create handler
        callback = MagicMock()
        handler = file_watcher.ChangeEventHandler(
            Path("/test"),
            callback,
            file_patterns=["*.py", "*.md"],
            ignore_patterns=["*.pyc", "__pycache__/*"]
        )
        
        # Test all event types
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/test/file.py"
        
        # Created
        event.event_type = "created"
        handler.on_created(event)
        assert callback.call_count == 1
        
        # Modified
        event.event_type = "modified"
        handler.on_modified(event)
        assert callback.call_count == 2
        
        # Deleted
        event.event_type = "deleted"
        handler.on_deleted(event)
        assert callback.call_count == 3
        
        # Moved
        event.event_type = "moved"
        event.dest_path = "/test/newfile.py"
        handler.on_moved(event)
        assert callback.call_count == 4
        
        # Test ignored file
        event.src_path = "/test/file.pyc"
        handler.on_created(event)
        assert callback.call_count == 4  # Should not increase
        
        # Test directory (should be ignored)
        event.is_directory = True
        event.src_path = "/test/dir"
        handler.on_created(event)
        assert callback.call_count == 4  # Should not increase
    
    @pytest.mark.asyncio
    async def test_watcher_batch_processing(self):
        """Test batch processing of changes."""
        idx = MagicMock()
        idx.index_file = AsyncMock()
        
        watcher = file_watcher.FileWatcher(idx, debounce_seconds=0, batch_size=2)
        
        # Add multiple changes
        for i in range(5):
            watcher.change_queue.append(
                file_watcher.FileChange(
                    path=Path(f"/test{i}.py"),
                    change_type=file_watcher.ChangeType.MODIFIED
                )
            )
        
        # Process changes (should batch)
        await watcher._process_changes()
        
        # Should have processed all changes
        assert len(watcher.change_queue) == 0
        assert watcher.stats["changes_processed"] == 5
    
    @pytest.mark.asyncio
    async def test_watcher_error_handling(self):
        """Test error handling in watcher."""
        idx = MagicMock()
        idx.index_file = AsyncMock(side_effect=Exception("Index error"))
        
        watcher = file_watcher.FileWatcher(idx)
        
        # Add change that will error
        watcher.change_queue.append(
            file_watcher.FileChange(
                path=Path("/test.py"),
                change_type=file_watcher.ChangeType.MODIFIED
            )
        )
        
        # Process (should handle error)
        await watcher._process_changes()
        
        assert watcher.stats["errors"] > 0
    
    @pytest.mark.asyncio
    async def test_watcher_lifecycle(self):
        """Test complete watcher lifecycle."""
        idx = MagicMock()
        idx.scanner = MagicMock()
        idx.scanner.generate_source_id = MagicMock(return_value="src123")
        idx.index_folder = AsyncMock(return_value=MagicMock(
            source_id="src123",
            file_count=5,
            total_chunks=20
        ))
        
        watcher = file_watcher.FileWatcher(idx)
        
        with patch('eol.rag_context.file_watcher.Observer') as MockObserver:
            mock_observer = MagicMock()
            mock_observer.is_alive = MagicMock(return_value=False)
            MockObserver.return_value = mock_observer
            
            # Start
            await watcher.start()
            assert watcher.is_running
            mock_observer.start.assert_called()
            
            # Watch multiple directories
            with tempfile.TemporaryDirectory() as tmpdir1, \
                 tempfile.TemporaryDirectory() as tmpdir2:
                
                src1 = await watcher.watch(Path(tmpdir1))
                src2 = await watcher.watch(Path(tmpdir2))
                
                assert len(watcher.watched_sources) == 2
                
                # Unwatch one
                await watcher.unwatch(src1)
                assert len(watcher.watched_sources) == 1
                
                # Stop
                await watcher.stop()
                assert not watcher.is_running
                mock_observer.stop.assert_called()


class TestUncoveredRedisClient:
    """Test uncovered Redis client paths."""
    
    @pytest.mark.asyncio
    async def test_redis_search_operations(self):
        """Test Redis search operations."""
        store = redis_client.RedisVectorStore(
            config.RedisConfig(),
            config.IndexConfig()
        )
        
        with patch('eol.rag_context.redis_client.AsyncRedis') as MockRedis:
            mock_redis = MagicMock()
            
            async def mock_connect(*args, **kwargs):
                return mock_redis
            
            MockRedis.side_effect = mock_connect
            mock_redis.ping = AsyncMock(return_value=True)
            
            # Mock search results
            mock_doc = MagicMock()
            mock_doc.id = "doc:1"
            mock_doc.content = "Test content"
            mock_doc.metadata = json.dumps({"key": "value"})
            mock_doc.embedding = np.array([1, 2, 3]).tobytes()
            mock_doc.hierarchy_level = 2
            
            mock_result = MagicMock()
            mock_result.docs = [mock_doc]
            mock_result.total = 1
            
            mock_redis.ft = MagicMock()
            mock_redis.ft.return_value.search = AsyncMock(return_value=mock_result)
            
            await store.connect_async()
            
            # Search with embedding
            results = await store.search("query", limit=5, hierarchy_level=2)
            assert len(results) == 1
            
            # Get context (hierarchical search)
            mock_redis.ft.return_value.search = AsyncMock(side_effect=[
                mock_result,  # Level 1
                mock_result,  # Level 2
                mock_result   # Level 3
            ])
            
            context = await store.get_context("query", max_chunks=10)
            assert len(context) > 0
    
    @pytest.mark.asyncio
    async def test_redis_index_creation_existing(self):
        """Test index creation when index exists."""
        store = redis_client.RedisVectorStore(
            config.RedisConfig(),
            config.IndexConfig()
        )
        
        with patch('eol.rag_context.redis_client.AsyncRedis') as MockRedis:
            mock_redis = MagicMock()
            
            async def mock_connect(*args, **kwargs):
                return mock_redis
            
            MockRedis.side_effect = mock_connect
            mock_redis.ping = AsyncMock(return_value=True)
            mock_redis.ft = MagicMock()
            
            # Mock index already exists
            mock_redis.ft.return_value.info = AsyncMock(return_value={"name": "index"})
            mock_redis.ft.return_value.create_index = AsyncMock()
            
            await store.connect_async()
            await store.create_indexes()
            
            # Should not create index if it exists
            mock_redis.ft.return_value.create_index.assert_not_called()
    
    def test_redis_sync_connection(self):
        """Test synchronous Redis connection."""
        store = redis_client.RedisVectorStore(
            config.RedisConfig(),
            config.IndexConfig()
        )
        
        with patch('eol.rag_context.redis_client.Redis') as MockRedis:
            mock_redis = MagicMock()
            mock_redis.ping = MagicMock(return_value=True)
            MockRedis.return_value = mock_redis
            
            store.connect_sync()
            
            assert store.redis is not None
            mock_redis.ping.assert_called()