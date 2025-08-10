"""
EOL RAG Context - Intelligent context management MCP server.

This module provides Redis 8-backed RAG system for dynamic context management,
replacing static context files with intelligent retrieval.
"""

__version__ = "0.1.0"

from .server import EOLRAGContextServer
from .config import RAGConfig

__all__ = ["EOLRAGContextServer", "RAGConfig"]