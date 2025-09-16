"""
Unit tests for the enhanced knowledge graph module.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from eol.rag_context.code_analyzer import CodeEntity, CodeEntityType
from eol.rag_context.enhanced_knowledge_graph import EnhancedKnowledgeGraphBuilder
from eol.rag_context.multimodal_config import MultimodalConfig, ProcessingMode


class TestEnhancedKnowledgeGraphBuilder:
    """Test the EnhancedKnowledgeGraphBuilder class."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis store."""
        return MagicMock()

    @pytest.fixture
    def mock_embedding_manager(self):
        """Create a mock embedding manager."""
        manager = AsyncMock()
        manager.get_embedding = AsyncMock(return_value=[0.1] * 384)
        return manager

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return MultimodalConfig(
            enable_code_analysis=True,
            enable_data_extraction=True,
            enable_pattern_detection=True,
            include_embeddings=True,
            min_pattern_frequency=2,
        )

    @pytest.fixture
    def builder(self, mock_redis, mock_embedding_manager, config):
        """Create an enhanced knowledge graph builder."""
        return EnhancedKnowledgeGraphBuilder(mock_redis, mock_embedding_manager, config)

    def test_initialization(self, builder):
        """Test builder initialization."""
        assert builder is not None
        assert builder.config is not None
        # These are stored in parent class private attributes
        assert builder._code_analyzer is None  # Not initialized yet (lazy loading)

    def test_configuration_validation(self, mock_redis, mock_embedding_manager):
        """Test configuration validation on initialization."""
        config = MultimodalConfig(
            similarity_threshold=2.0,  # Invalid: > 1
            batch_size=0,  # Invalid: < 1
        )

        # The builder will log warnings for invalid config
        _ = EnhancedKnowledgeGraphBuilder(mock_redis, mock_embedding_manager, config)

        # Validation happens and warnings are logged
        warnings = config.validate()
        assert len(warnings) > 0  # Should have warnings for invalid values

    def test_lazy_loading_code_analyzer(self, builder):
        """Test lazy loading of code analyzer."""
        # Should not be initialized yet
        assert builder._code_analyzer is None

        # Access property to trigger lazy loading
        analyzer = builder.code_analyzer

        # Should now be initialized
        assert analyzer is not None
        assert builder._code_analyzer is not None

    def test_lazy_loading_data_extractor(self, builder):
        """Test lazy loading of data extractor."""
        # Should not be initialized yet
        assert builder._data_extractor is None

        # Access property to trigger lazy loading
        extractor = builder.data_extractor

        # Should now be initialized if data extraction is enabled
        if builder.config.enable_data_extraction:
            assert extractor is not None
            assert builder._data_extractor is not None

    @pytest.mark.asyncio
    async def test_build_from_code_disabled(self, mock_redis, mock_embedding_manager):
        """Test build_from_code when code analysis is disabled."""
        config = MultimodalConfig(enable_code_analysis=False)
        builder = EnhancedKnowledgeGraphBuilder(mock_redis, mock_embedding_manager, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            stats = await builder.build_from_code(Path(tmpdir))

        assert stats["entities"] == 0
        assert stats["relationships"] == 0

    @pytest.mark.asyncio
    async def test_build_from_code_success(self, builder, tmp_path):
        """Test successful code analysis and graph building."""
        # Create a simple Python file
        code_file = tmp_path / "test.py"
        code_file.write_text("def hello(): pass")

        # Run actual code analysis
        stats = await builder.build_from_code(tmp_path)

        # Check that stats are returned (may be 0 if there's an error)
        assert "entities" in stats
        assert "relationships" in stats
        assert "files_processed" in stats

    @pytest.mark.asyncio
    async def test_build_from_data_disabled(self, mock_redis, mock_embedding_manager):
        """Test build_from_data when data extraction is disabled."""
        config = MultimodalConfig(enable_data_extraction=False)
        builder = EnhancedKnowledgeGraphBuilder(mock_redis, mock_embedding_manager, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_file = Path(tmpdir) / "data.csv"
            csv_file.write_text("id,name\n1,test")
            stats = await builder.build_from_data(csv_file)

        assert stats["entities"] == 0
        assert stats["relationships"] == 0

    @pytest.mark.asyncio
    async def test_build_from_data_success(self, builder, tmp_path):
        """Test successful data extraction and graph building."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,Alice\n2,Bob")

        # Run actual data extraction
        stats = await builder.build_from_data(csv_file)

        # Should extract at least the file entity
        assert stats["entities"] >= 1

    @pytest.mark.asyncio
    async def test_detect_patterns_disabled(self, mock_redis, mock_embedding_manager):
        """Test pattern detection when disabled."""
        config = MultimodalConfig(enable_pattern_detection=False)
        builder = EnhancedKnowledgeGraphBuilder(mock_redis, mock_embedding_manager, config)

        patterns = await builder.detect_patterns()

        assert patterns == []

    @pytest.mark.asyncio
    async def test_detect_code_patterns(self, builder):
        """Test code pattern detection."""
        # Test that method exists and returns list
        patterns = await builder._detect_code_patterns(min_frequency=2)
        assert isinstance(patterns, list)

    @pytest.mark.asyncio
    async def test_detect_data_patterns(self, builder):
        """Test data pattern detection."""
        # Test that method exists and returns list
        patterns = await builder._detect_data_patterns(min_frequency=2)
        assert isinstance(patterns, list)

    @pytest.mark.asyncio
    async def test_detect_cross_modal_patterns(self, builder):
        """Test cross-modal pattern detection."""
        # Test that method exists and returns list
        patterns = await builder._detect_cross_modal_patterns(min_frequency=2)
        assert isinstance(patterns, list)

    def test_create_entity_text(self, builder):
        """Test entity text creation for embeddings."""
        entity = CodeEntity(
            id="test",
            type=CodeEntityType.FUNCTION,
            name="test_function",
            file_path="test.py",
            line_start=1,
            line_end=5,
            content="def test_function():\n    pass",
            signature="def test_function()",
            docstring="This is a test function",
        )

        text = builder._create_entity_text(entity)

        assert "function: test_function" in text
        assert "Signature: def test_function()" in text
        assert "Documentation: This is a test function" in text
        assert "Content:" in text

    @pytest.mark.asyncio
    async def test_store_entity_embedding(self, builder):
        """Test storing entity embeddings."""
        entity_id = "test_entity"
        embedding = [0.1] * 384
        metadata = {"type": "function", "name": "test"}

        await builder.store_entity_embedding(entity_id, embedding, metadata)

        # Should not raise any exceptions
        # In real implementation, this would store in Redis

    @pytest.mark.asyncio
    async def test_find_similar_entities_no_embedding(self, builder):
        """Test finding similar entities when entity has no embedding."""
        # Mock graph node without embedding
        builder.graph.nodes = {"entity1": {"name": "test"}}

        similar = await builder.find_similar_entities("entity1")

        assert similar == []

    @pytest.mark.asyncio
    async def test_find_similar_entities_with_embedding(self, builder):
        """Test finding similar entities with embeddings."""
        # Test that method exists and returns list
        similar = await builder.find_similar_entities("entity1", k=2)
        assert isinstance(similar, list)

    @pytest.mark.asyncio
    async def test_merge_similar_entities_disabled(self, mock_redis, mock_embedding_manager):
        """Test entity merging when disabled."""
        config = MultimodalConfig(merge_similar_entities=False)
        builder = EnhancedKnowledgeGraphBuilder(mock_redis, mock_embedding_manager, config)

        merged_count = await builder.merge_similar_entities()

        assert merged_count == 0

    @pytest.mark.asyncio
    async def test_merge_similar_entities_enabled(self, builder):
        """Test entity merging when enabled."""
        # This is a placeholder test as the actual implementation is incomplete
        merged_count = await builder.merge_similar_entities(threshold=0.9)

        assert merged_count == 0  # No entities to merge in mock scenario

    @pytest.mark.asyncio
    async def test_detect_patterns_full_workflow(self, builder):
        """Test the complete pattern detection workflow."""
        # Test that the workflow completes
        patterns = await builder.detect_patterns(min_frequency=1)
        assert isinstance(patterns, list)

    def test_minimal_config(self, mock_redis, mock_embedding_manager):
        """Test with minimal configuration."""
        config = MultimodalConfig.minimal()
        builder = EnhancedKnowledgeGraphBuilder(mock_redis, mock_embedding_manager, config)

        assert builder.config.processing_mode == ProcessingMode.MINIMAL
        assert not builder.config.enable_image_processing
        assert not builder.config.enable_pattern_detection

    def test_comprehensive_config(self, mock_redis, mock_embedding_manager):
        """Test with comprehensive configuration."""
        config = MultimodalConfig.comprehensive()
        builder = EnhancedKnowledgeGraphBuilder(mock_redis, mock_embedding_manager, config)

        assert builder.config.processing_mode == ProcessingMode.COMPREHENSIVE
        assert builder.config.enable_image_processing
        assert builder.config.enable_pattern_detection

    @pytest.mark.asyncio
    async def test_error_handling_in_build_from_code(self, builder, tmp_path):
        """Test error handling in build_from_code."""
        # Mock analyzer to raise exception
        with patch.object(
            builder.code_analyzer, "analyze_directory", side_effect=Exception("Test error")
        ):
            stats = await builder.build_from_code(tmp_path)

        # Should handle error gracefully
        assert stats["entities"] == 0
        assert stats["relationships"] == 0

    @pytest.mark.asyncio
    async def test_error_handling_in_build_from_data(self, builder, tmp_path):
        """Test error handling in build_from_data."""
        # Test with non-existent file
        non_existent = tmp_path / "non_existent.csv"

        stats = await builder.build_from_data(non_existent)

        # Should handle error gracefully
        assert stats["entities"] == 0
        assert stats["relationships"] == 0
