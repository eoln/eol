"""Main entry point for EOL RAG Context MCP Server."""

import asyncio
import logging
import sys
from pathlib import Path

from .config import RAGConfig
from .server import EOLRAGContextServer


def main():
    """Main CLI entry point."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Parse arguments
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print("EOL RAG Context MCP Server")
        print("\nUsage:")
        print("  eol-rag-context [config_file]")
        print("\nOptions:")
        print("  config_file    Path to configuration file (JSON or YAML)")
        print("  -h, --help     Show this help message")
        print("\nEnvironment variables:")
        print("  REDIS_HOST     Redis host (default: localhost)")
        print("  REDIS_PORT     Redis port (default: 6379)")
        print(
            "  EMBEDDING_PROVIDER  Embedding provider (default: sentence-transformers)"
        )
        sys.exit(0)

    # Load configuration
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    try:
        if config_path:
            config = RAGConfig.from_file(config_path)
            logging.info(f"Loaded configuration from {config_path}")
        else:
            config = RAGConfig()
            logging.info("Using default configuration")
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Create and run server
    server = EOLRAGContextServer(config)

    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        logging.info("Server stopped by user")
    except Exception as e:
        logging.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
