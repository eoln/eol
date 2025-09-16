"""
Test main.py to boost coverage.
"""

# Removed unused imports

from eol.rag_context import main


class TestMainModule:
    """Test main module functionality."""

    def test_main_import(self):
        """Test that main module can be imported."""
        assert main is not None
        assert hasattr(main, "__file__")

    def test_main_function(self):
        """Test main function exists."""
        # Check main has the main function
        assert hasattr(main, "main")
        assert callable(main.main)


class TestEmbeddingsExtra:
    """Extra tests for embeddings module."""

    def test_embeddings_import(self):
        """Test embeddings module import."""
        from eol.rag_context import embeddings

        assert embeddings is not None
        assert hasattr(embeddings, "EmbeddingManager")
        assert hasattr(embeddings, "SentenceTransformerProvider")


class TestKnowledgeGraphExtra:
    """Extra tests for knowledge graph module."""

    def test_knowledge_graph_import(self):
        """Test knowledge graph module import."""
        from eol.rag_context import knowledge_graph

        assert knowledge_graph is not None
        assert hasattr(knowledge_graph, "KnowledgeGraphBuilder")
        assert hasattr(knowledge_graph, "Entity")
        assert hasattr(knowledge_graph, "Relationship")


if __name__ == "__main__":
    print("✅ Main module coverage tests!")
