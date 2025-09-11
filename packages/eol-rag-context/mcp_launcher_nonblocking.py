#!/usr/bin/env python3
"""Non-blocking MCP Server launcher with async indexing capabilities."""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from eol.rag_context.server import EOLRAGContextServer
from eol.rag_context.config import RAGConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the non-blocking MCP server."""
    logger.info("🚀 Starting EOL RAG Context MCP Server with non-blocking indexing")
    
    try:
        # Load configuration
        config = RAGConfig()
        
        # Create server
        server = EOLRAGContextServer(config)
        
        logger.info("✅ Server created successfully with non-blocking capabilities")
        logger.info("📊 Non-blocking tools available: start_indexing, get_indexing_status, list_indexing_tasks, cancel_indexing_task, cleanup_old_indexing_tasks")
        
        # Use the server's blocking run method which handles async internally
        import asyncio
        asyncio.run(server.run())
        
    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()