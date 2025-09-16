# Basic Usage Examples

Practical, copy-and-paste examples for common EOL RAG Context tasks. All examples are tested and include complete code with explanations.

## Prerequisites

Before running these examples, ensure you have:

```bash
# Install EOL RAG Context
pip install eol-rag-context

# Start Redis Stack
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest

# Verify installation
python -c "from eol.rag_context import EOLRAGContextServer; print('✅ Ready')"
```

## Single File Indexing

### Index a Python File

Index a single Python file with detailed output:

```python
import asyncio
from pathlib import Path
from eol.rag_context import EOLRAGContextServer

async def index_python_file():
    """Index a single Python file and examine the results."""
    # Initialize server
    server = EOLRAGContextServer()
    await server.initialize()

    # Create a sample Python file
    sample_code = '''
"""User authentication module."""

class UserManager:
    """Manages user accounts and authentication."""

    def __init__(self, database):
        self.db = database
        self.active_sessions = {}

    def create_user(self, username, password, email):
        """Create a new user account."""
        user_data = {
            'username': username,
            'password_hash': self._hash_password(password),
            'email': email,
            'created_at': datetime.now()
        }
        return self.db.users.insert(user_data)

    def authenticate(self, username, password):
        """Authenticate user credentials."""
        user = self.db.users.find_one({'username': username})
        if user and self._verify_password(password, user['password_hash']):
            session_id = self._create_session(user['_id'])
            return {'user_id': user['_id'], 'session_id': session_id}
        return None

    def _hash_password(self, password):
        """Hash password using bcrypt."""
        import bcrypt
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    def _verify_password(self, password, hash):
        """Verify password against hash."""
        import bcrypt
        return bcrypt.checkpw(password.encode('utf-8'), hash)
'''

    # Save to temporary file
    temp_file = Path("sample_auth.py")
    temp_file.write_text(sample_code)

    try:
        # Index the file
        result = await server.index_file(
            file_path=str(temp_file),
            force_reindex=True
        )

        print("📄 File Indexing Results:")
        print(f"   File: {result['file_path']}")
        print(f"   Chunks created: {result['chunks_created']}")
        print(f"   Processing time: {result['processing_time']:.2f}s")
        print(f"   File size: {result['file_size']} bytes")

        # Show chunk breakdown
        chunks = result.get('chunk_details', [])
        print(f"\n🧩 Chunk Breakdown:")
        for i, chunk in enumerate(chunks, 1):
            print(f"   Chunk {i}: {chunk['type']} - {chunk['name']}")
            print(f"      Lines: {chunk['start_line']}-{chunk['end_line']}")
            print(f"      Size: {len(chunk['content'])} chars")

        # Test search for the indexed content
        search_result = await server.search_context({
            'query': 'how to authenticate a user',
            'max_results': 3
        }, None)

        print(f"\n🔍 Search Test Results:")
        for result in search_result['results']:
            print(f"   📍 {result['chunk_type']} (score: {result['similarity']:.3f})")
            print(f"      Content: {result['content'][:100]}...")

    finally:
        # Cleanup
        if temp_file.exists():
            temp_file.unlink()
        await server.close()

# Run the example
asyncio.run(index_python_file())
```

### Index a Markdown Document

Index documentation with header-based chunking:

```python
import asyncio
from pathlib import Path
from eol.rag_context import EOLRAGContextServer

async def index_markdown_file():
    """Index a markdown document with header-based chunking."""
    server = EOLRAGContextServer()
    await server.initialize()

    # Create sample markdown documentation
    markdown_content = '''# Database Configuration Guide

This guide covers database setup and configuration for your application.

## Overview

The database configuration system supports multiple database backends including PostgreSQL, MySQL, and SQLite. Configuration is managed through YAML files with environment variable overrides.

### Supported Databases

- **PostgreSQL** - Recommended for production
- **MySQL** - Good alternative with wide support
- **SQLite** - Perfect for development and testing

## Basic Configuration

### PostgreSQL Setup

To configure PostgreSQL:

```yaml
database:
  type: postgresql
  host: localhost
  port: 5432
  name: myapp
  user: dbuser
  password: ${DB_PASSWORD}
```

### Connection Pooling

Enable connection pooling for better performance:

```yaml
database:
  pool:
    min_connections: 5
    max_connections: 20
    timeout: 30
```

## Advanced Configuration

### SSL Configuration

For production deployments, enable SSL:

```yaml
database:
  ssl:
    enabled: true
    ca_file: /path/to/ca.pem
    cert_file: /path/to/client-cert.pem
    key_file: /path/to/client-key.pem
```

### Performance Tuning

Optimize performance with these settings:

```yaml
database:
  performance:
    statement_timeout: 60000
    idle_timeout: 300
    query_cache_size: 100
```

## Troubleshooting

### Connection Issues

If you can't connect to the database:

1. Check that the database server is running
2. Verify connection parameters
3. Test network connectivity
4. Check firewall settings

### Performance Problems

For slow queries:

1. Enable query logging
2. Analyze slow queries
3. Add appropriate indexes
4. Consider connection pooling

## Environment Variables

Override configuration with environment variables:

- `DB_HOST` - Database host
- `DB_PORT` - Database port
- `DB_NAME` - Database name
- `DB_USER` - Username
- `DB_PASSWORD` - Password
'''

  # Save to temporary file

    temp_file = Path("database_guide.md")
    temp_file.write_text(markdown_content)

    try:
        # Index the markdown file
        result = await server.index_file(
            file_path=str(temp_file),
            force_reindex=True
        )

        print("📄 Markdown Indexing Results:")
        print(f"   File: {result['file_path']}")
        print(f"   Chunks created: {result['chunks_created']}")
        print(f"   Processing time: {result['processing_time']:.2f}s")

        # Show hierarchical structure
        chunks = result.get('chunk_details', [])
        print(f"\n📋 Document Structure:")
        current_level = 0
        for chunk in chunks:
            # Determine indentation based on header level
            if chunk['type'] == 'header':
                level = chunk.get('header_level', 1)
                indent = "  " * (level - 1)
                print(f"{indent}📌 {chunk['name']}")
            else:
                print(f"    📝 {chunk['type']} ({len(chunk['content'])} chars)")

        # Test different types of searches
        queries = [
            "How to configure PostgreSQL?",
            "Connection pooling settings",
            "SSL configuration for production",
            "Troubleshooting database issues"
        ]

        print(f"\n🔍 Search Test Results:")
        for query in queries:
            search_result = await server.search_context({
                'query': query,
                'max_results': 2
            }, None)

            print(f"\n   Query: '{query}'")
            for result in search_result['results']:
                section = result['metadata'].get('section', 'Unknown')
                print(f"      📍 {section} (score: {result['similarity']:.3f})")

    finally:
        # Cleanup
        if temp_file.exists():
            temp_file.unlink()
        await server.close()

# Run the example

asyncio.run(index_markdown_file())

```

## Simple Directory Indexing

### Index a Project Directory

Index an entire project with smart file filtering:

```python
import asyncio
from pathlib import Path
from eol.rag_context import EOLRAGContextServer

async def index_project_directory():
    """Index a complete project directory with intelligent filtering."""
    server = EOLRAGContextServer()
    await server.initialize()

    # Create a sample project structure
    project_root = Path("sample_project")
    project_root.mkdir(exist_ok=True)

    # Create directory structure
    (project_root / "src").mkdir(exist_ok=True)
    (project_root / "docs").mkdir(exist_ok=True)
    (project_root / "tests").mkdir(exist_ok=True)
    (project_root / "config").mkdir(exist_ok=True)

    # Create sample files
    files_to_create = {
        "README.md": '''# Sample Project

This is a sample project demonstrating EOL RAG Context indexing.

## Features

- User authentication
- Database integration
- RESTful API
- Configuration management
''',
        "src/main.py": '''"""Main application entry point."""

from flask import Flask
from src.auth import UserAuth
from src.database import Database

app = Flask(__name__)

def create_app(config_path="config/app.yaml"):
    """Create and configure the Flask application."""
    db = Database(config_path)
    auth = UserAuth(db)

    @app.route("/api/login", methods=["POST"])
    def login():
        # Login implementation
        pass

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
''',
        "src/auth.py": '''"""Authentication module."""

import bcrypt
from datetime import datetime, timedelta

class UserAuth:
    def __init__(self, database):
        self.db = database

    def register_user(self, username, password, email):
        """Register a new user."""
        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        user_data = {
            "username": username,
            "password_hash": password_hash,
            "email": email,
            "created_at": datetime.now()
        }
        return self.db.create_user(user_data)
''',
        "docs/api.md": '''# API Documentation

## Authentication Endpoints

### POST /api/login

Login with username and password.

**Request:**
```json
{
  "username": "user@example.com",
  "password": "secretpassword"
}
```

**Response:**

```json
{
  "token": "jwt_token_here",
  "expires": "2024-01-01T00:00:00Z"
}
```

''',
        "config/app.yaml": '''# Application Configuration

database:
  type: postgresql
  host: localhost
  port: 5432
  name: sampleapp

auth:
  jwt_secret: ${JWT_SECRET}
  token_expiry: 3600

logging:
  level: INFO
  file: logs/app.log
'''
    }

    # Write files
    for file_path, content in files_to_create.items():
        full_path = project_root / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)

    try:
        # Index the entire project
        result = await server.index_directory(
            directory_path=str(project_root),
            recursive=True,
            file_patterns=["*.py", "*.md", "*.yaml", "*.yml", "*.json"],
            exclude_patterns=["*.pyc", "__pycache__/*", ".git/*", "*.log"]
        )

        print("📁 Project Indexing Results:")
        print(f"   Directory: {project_root}")
        print(f"   Indexed files: {result['indexed_files']}")
        print(f"   Total chunks: {result['total_chunks']}")
        print(f"   Skipped files: {result['skipped_files']}")
        print(f"   Processing time: {result['processing_time']:.2f}s")

        # Show file breakdown by type
        file_stats = result.get('file_stats', {})
        if file_stats:
            print(f"\n📊 File Type Breakdown:")
            for file_type, count in file_stats.items():
                print(f"   {file_type}: {count} files")

        # Test searches for different content types
        test_queries = [
            ("Code search", "how does user authentication work?"),
            ("API docs", "login endpoint documentation"),
            ("Configuration", "database connection settings"),
            ("General", "what does this project do?")
        ]

        print(f"\n🔍 Search Test Results:")
        for category, query in test_queries:
            search_result = await server.search_context({
                'query': query,
                'max_results': 2,
                'similarity_threshold': 0.6
            }, None)

            print(f"\n   {category}: '{query}'")
            print(f"   Found {len(search_result['results'])} results:")

            for result in search_result['results']:
                file_name = Path(result['source_path']).name
                print(f"      📄 {file_name} ({result['similarity']:.3f})")
                print(f"         {result['content'][:80]}...")

    finally:
        # Cleanup
        import shutil
        if project_root.exists():
            shutil.rmtree(project_root)
        await server.close()

# Run the example

asyncio.run(index_project_directory())

```

## Simple Search Queries

### Basic Search with Filtering

Search with various filters and result options:

```python
import asyncio
from eol.rag_context import EOLRAGContextServer
from eol.rag_context.server import SearchContextRequest

async def basic_search_examples():
    """Demonstrate basic search patterns with filtering."""
    server = EOLRAGContextServer()
    await server.initialize()

    # First, let's ensure we have some content indexed
    # (In real usage, you'd have already indexed your content)

    print("🔍 Basic Search Examples\n")

    # Example 1: Simple search
    print("1. Simple Search:")
    result = await server.search_context({
        'query': 'user authentication',
        'max_results': 3
    }, None)

    print(f"   Found {len(result['results'])} results")
    for r in result['results']:
        print(f"   📄 {r['source_path']} (score: {r['similarity']:.3f})")

    # Example 2: Search with similarity threshold
    print("\n2. High-quality results only (similarity > 0.8):")
    result = await server.search_context({
        'query': 'database configuration',
        'max_results': 5,
        'similarity_threshold': 0.8
    }, None)

    print(f"   High-quality results: {len(result['results'])}")
    for r in result['results']:
        print(f"   📄 {r['source_path']} (score: {r['similarity']:.3f})")

    # Example 3: Search with file type filtering
    print("\n3. Search only Python files:")
    result = await server.search_context({
        'query': 'class definition',
        'max_results': 3,
        'filters': {
            'file_types': ['.py']
        }
    }, None)

    print(f"   Python files: {len(result['results'])}")
    for r in result['results']:
        print(f"   🐍 {r['source_path']} ({r['chunk_type']})")

    # Example 4: Search with metadata
    print("\n4. Search with metadata included:")
    result = await server.search_context({
        'query': 'configuration settings',
        'max_results': 2,
        'include_metadata': True
    }, None)

    for i, r in enumerate(result['results'], 1):
        metadata = r['metadata']
        print(f"   Result {i}:")
        print(f"      File: {r['source_path']}")
        print(f"      Lines: {metadata.get('lines', 'N/A')}")
        print(f"      Language: {metadata.get('language', 'N/A')}")
        print(f"      Modified: {metadata.get('modified', 'N/A')}")

    await server.close()

# Run the example
asyncio.run(basic_search_examples())
```

### Search with Context Assembly

Get coherent context from multiple results:

```python
import asyncio
from eol.rag_context import EOLRAGContextServer

async def search_with_context():
    """Demonstrate context assembly for comprehensive answers."""
    server = EOLRAGContextServer()
    await server.initialize()

    # Search for comprehensive context on a topic
    result = await server.search_context({
        'query': 'how to set up user authentication system',
        'max_results': 8,
        'assemble_context': True,
        'max_context_size': 3000,
        'include_surrounding': True
    }, None)

    print("🎯 Context Assembly Example")
    print("=" * 50)

    # Show assembled context
    assembled_context = result.get('assembled_context')
    if assembled_context:
        print("📄 Assembled Context:")
        print(assembled_context)
        print(f"\nContext size: {len(assembled_context)} characters")

    # Show sources used
    sources = result.get('context_sources', [])
    print(f"\n📚 Sources Used ({len(sources)}):")
    for source in sources:
        print(f"   • {source['file']}")
        if 'lines' in source:
            print(f"     Lines: {source['lines']}")
        if 'relevance' in source:
            print(f"     Relevance: {source['relevance']:.3f}")

    await server.close()

# Run the example
asyncio.run(search_with_context())
```

## Configuration Examples

### Basic Configuration

Simple configuration for development:

```python
# config.py - Basic configuration example
import asyncio
from eol.rag_context import EOLRAGContextServer

async def basic_configuration():
    """Example of basic configuration setup."""

    # Configuration as dictionary
    config = {
        "redis": {
            "url": "redis://localhost:6379",
            "db": 0
        },
        "embedding": {
            "provider": "sentence_transformers",
            "model": "all-MiniLM-L6-v2",  # Fast, efficient model
            "batch_size": 16
        },
        "indexing": {
            "chunk_size": 800,
            "chunk_overlap": 100,
            "use_semantic_chunking": True
        },
        "caching": {
            "enabled": True,
            "ttl_seconds": 1800  # 30 minutes
        }
    }

    # Initialize with configuration
    server = EOLRAGContextServer(config=config)
    await server.initialize()

    print("✅ Server initialized with custom configuration")

    # Test the configuration
    info = await server.get_server_info()
    print(f"   Redis: {info['redis']['status']}")
    print(f"   Embedding model: {info['embedding']['model']}")
    print(f"   Cache enabled: {info['caching']['enabled']}")

    await server.close()

# Run the example
asyncio.run(basic_configuration())
```

### YAML Configuration File

Using a YAML configuration file:

```python
# yaml_config.py - YAML configuration example
import asyncio
import yaml
from pathlib import Path
from eol.rag_context import EOLRAGContextServer

async def yaml_configuration():
    """Example using YAML configuration file."""

    # Create a sample YAML configuration
    yaml_config = '''
# EOL RAG Context Configuration
redis:
  url: "redis://localhost:6379"
  db: 0

embedding:
  provider: "sentence_transformers"
  model: "all-MiniLM-L6-v2"
  batch_size: 32

indexing:
  chunk_size: 1000
  chunk_overlap: 200
  use_semantic_chunking: true
  parse_code_structure: true

chunking:
  markdown_split_headers: true
  preserve_code_blocks: true

caching:
  enabled: true
  ttl_seconds: 3600
  max_cache_size: 1000

context:
  max_context_size: 8000

logging:
  level: "INFO"
  format: "%(asctime)s - %(levelname)s - %(message)s"
'''

    # Save configuration to file
    config_file = Path("eol_config.yaml")
    config_file.write_text(yaml_config)

    try:
        # Initialize server with YAML config
        server = EOLRAGContextServer(config_path=str(config_file))
        await server.initialize()

        print("✅ Server initialized with YAML configuration")

        # Show active configuration
        info = await server.get_server_info()
        print("📋 Active Configuration:")
        print(f"   Chunk size: {info['indexing']['chunk_size']}")
        print(f"   Overlap: {info['indexing']['chunk_overlap']}")
        print(f"   Cache TTL: {info['caching']['ttl_seconds']}s")
        print(f"   Max context: {info['context']['max_context_size']}")

        await server.close()

    finally:
        # Cleanup
        if config_file.exists():
            config_file.unlink()

# Run the example
asyncio.run(yaml_configuration())
```

## Batch Processing

### Process Multiple Directories

Efficiently index large document collections:

```python
import asyncio
import time
from pathlib import Path
from eol.rag_context import EOLRAGContextServer

async def batch_processing_example():
    """Example of efficient batch processing for large collections."""

    # Create sample directory structure
    base_dir = Path("batch_test")
    directories = ["docs", "code", "config", "examples"]

    # Create directories and sample files
    for dir_name in directories:
        dir_path = base_dir / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)

        # Create sample files in each directory
        for i in range(5):  # 5 files per directory
            if dir_name == "code":
                file_path = dir_path / f"module_{i}.py"
                content = f'"""Module {i}"""\nclass Class{i}:\n    def method(self): pass'
            elif dir_name == "docs":
                file_path = dir_path / f"doc_{i}.md"
                content = f'# Document {i}\nThis is documentation for feature {i}.'
            else:
                file_path = dir_path / f"file_{i}.txt"
                content = f'Sample content for file {i} in {dir_name} directory.'

            file_path.write_text(content)

    server = EOLRAGContextServer()
    await server.initialize()

    try:
        print("🚀 Batch Processing Example")
        print(f"Processing directories: {directories}")

        # Configure batch processing
        batch_config = {
            "batch_size": 10,           # Process 10 files at once
            "parallel_workers": 2,      # Use 2 parallel workers
            "show_progress": True,      # Show progress updates
            "error_handling": "skip"    # Skip problematic files
        }

        # Track processing time
        start_time = time.time()

        # Process all directories
        total_files = 0
        for directory in directories:
            dir_path = base_dir / directory

            print(f"\n📁 Processing {directory}...")

            result = await server.index_directory(
                directory_path=str(dir_path),
                recursive=True,
                **batch_config
            )

            total_files += result['indexed_files']
            print(f"   ✅ Indexed: {result['indexed_files']} files")
            print(f"   📊 Chunks: {result['total_chunks']}")
            print(f"   ⏱️  Time: {result['processing_time']:.2f}s")

        processing_time = time.time() - start_time

        print(f"\n🎉 Batch Processing Complete!")
        print(f"   Total files: {total_files}")
        print(f"   Total time: {processing_time:.2f}s")
        print(f"   Files/second: {total_files/processing_time:.1f}")

        # Test search across all indexed content
        search_result = await server.search_context({
            'query': 'documentation example',
            'max_results': 3
        }, None)

        print(f"\n🔍 Cross-directory search test:")
        for result in search_result['results']:
            print(f"   📄 {result['source_path']} ({result['similarity']:.3f})")

    finally:
        # Cleanup
        import shutil
        if base_dir.exists():
            shutil.rmtree(base_dir)
        await server.close()

# Run the example
asyncio.run(batch_processing_example())
```

## Error Handling

### Robust Error Handling

Handle common errors gracefully:

```python
import asyncio
import logging
from pathlib import Path
from eol.rag_context import EOLRAGContextServer

# Set up logging to see error details
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def error_handling_example():
    """Demonstrate robust error handling patterns."""

    server = None

    try:
        # 1. Server initialization with error handling
        print("🔧 Initializing server...")
        server = EOLRAGContextServer()
        await server.initialize()
        print("✅ Server initialized successfully")

    except ConnectionError as e:
        print(f"❌ Redis connection failed: {e}")
        print("💡 Make sure Redis is running: docker run -d -p 6379:6379 redis/redis-stack")
        return
    except Exception as e:
        print(f"❌ Server initialization failed: {e}")
        return

    try:
        # 2. File indexing with error handling
        print("\n📄 Testing file indexing...")

        # Test with non-existent file
        try:
            result = await server.index_file("nonexistent_file.py")
        except FileNotFoundError:
            print("✅ Correctly handled non-existent file")

        # Test with unreadable file (create and remove permissions)
        test_file = Path("test_permissions.py")
        test_file.write_text("print('test')")

        try:
            # This should work
            result = await server.index_file(str(test_file))
            print(f"✅ Indexed test file: {result['chunks_created']} chunks")
        except Exception as e:
            print(f"❌ Unexpected error indexing readable file: {e}")
        finally:
            # Cleanup
            if test_file.exists():
                test_file.unlink()

        # 3. Directory indexing with mixed content
        print("\n📁 Testing directory indexing with problematic files...")

        # Create test directory with mixed file types
        test_dir = Path("error_test_dir")
        test_dir.mkdir(exist_ok=True)

        # Create various file types including problematic ones
        files_to_create = {
            "good_file.py": "def hello(): print('hello')",
            "empty_file.py": "",
            "binary_file.pdf": b"\\x00\\x01\\x02\\x03",  # Mock binary content
            "large_text.md": "# Title\n" + "This is content. " * 1000,  # Very large file
            "unicode_file.py": "# -*- coding: utf-8 -*-\ndef greet(): print('🌍 Hello')"
        }

        for filename, content in files_to_create.items():
            file_path = test_dir / filename
            if isinstance(content, bytes):
                file_path.write_bytes(content)
            else:
                file_path.write_text(content, encoding='utf-8')

        try:
            result = await server.index_directory(
                directory_path=str(test_dir),
                error_handling="skip",  # Skip problematic files
                show_progress=False
            )

            print(f"✅ Directory indexing completed:")
            print(f"   Indexed files: {result['indexed_files']}")
            print(f"   Skipped files: {result.get('skipped_files', 0)}")
            print(f"   Total chunks: {result['total_chunks']}")

        except Exception as e:
            print(f"❌ Directory indexing failed: {e}")

        # 4. Search with error handling
        print("\n🔍 Testing search error handling...")

        try:
            # Test search with good query
            result = await server.search_context({
                'query': 'hello function',
                'max_results': 5
            }, None)
            print(f"✅ Search successful: {len(result['results'])} results")

        except Exception as e:
            print(f"❌ Search failed: {e}")

        try:
            # Test search with very long query
            very_long_query = "test " * 1000  # Very long query
            result = await server.search_context({
                'query': very_long_query,
                'max_results': 1
            }, None)
            print(f"✅ Long query handled: {len(result['results'])} results")

        except Exception as e:
            print(f"⚠️ Long query issue (expected): {type(e).__name__}")

    except Exception as e:
        logger.error(f"Unexpected error in main processing: {e}", exc_info=True)

    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")

        # Clean up test directory
        if 'test_dir' in locals() and test_dir.exists():
            import shutil
            shutil.rmtree(test_dir)
            print("✅ Test directory cleaned up")

        # Close server connection
        if server:
            try:
                await server.close()
                print("✅ Server connection closed")
            except Exception as e:
                print(f"⚠️ Error closing server: {e}")

# Run the example
asyncio.run(error_handling_example())
```

## Performance Monitoring

### Basic Performance Tracking

Monitor indexing and search performance:

```python
import asyncio
import time
import psutil
from pathlib import Path
from eol.rag_context import EOLRAGContextServer

async def performance_monitoring():
    """Monitor performance during indexing and searching."""

    server = EOLRAGContextServer()
    await server.initialize()

    # Create test content
    test_dir = Path("perf_test")
    test_dir.mkdir(exist_ok=True)

    # Create various sized files
    file_sizes = {
        "small.py": 50,    # 50 lines
        "medium.py": 200,  # 200 lines
        "large.py": 1000   # 1000 lines
    }

    for filename, line_count in file_sizes.items():
        content = f'"""Test file with {line_count} lines."""\n'
        content += '\n'.join([f'def function_{i}():\n    """Function {i}."""\n    pass'
                             for i in range(line_count // 3)])
        (test_dir / filename).write_text(content)

    try:
        print("📊 Performance Monitoring Example\n")

        # Monitor system resources
        process = psutil.Process()

        def get_memory_usage():
            return process.memory_info().rss / 1024 / 1024  # MB

        def get_cpu_usage():
            return process.cpu_percent()

        # Baseline measurements
        baseline_memory = get_memory_usage()
        print(f"🔍 Baseline memory usage: {baseline_memory:.1f} MB")

        # Performance tracking during indexing
        print("\n📁 Indexing Performance:")

        start_time = time.time()
        start_memory = get_memory_usage()

        result = await server.index_directory(
            directory_path=str(test_dir),
            show_progress=True
        )

        end_time = time.time()
        end_memory = get_memory_usage()

        processing_time = end_time - start_time
        memory_increase = end_memory - start_memory

        print(f"   ⏱️  Total time: {processing_time:.2f}s")
        print(f"   💾 Memory increase: {memory_increase:.1f} MB")
        print(f"   📊 Files/second: {result['indexed_files']/processing_time:.1f}")
        print(f"   🧩 Chunks created: {result['total_chunks']}")

        # Performance tracking during search
        print("\n🔍 Search Performance:")

        queries = [
            "function definition",
            "test file documentation",
            "python code example",
            "medium sized functions"
        ]

        search_times = []

        for query in queries:
            start_time = time.time()

            search_result = await server.search_context({
                'query': query,
                'max_results': 5
            }, None)

            end_time = time.time()
            search_time = (end_time - start_time) * 1000  # Convert to ms
            search_times.append(search_time)

            cache_hit = search_result.get('cache_hit', False)
            print(f"   '{query}': {search_time:.1f}ms ({len(search_result['results'])} results) {'[CACHED]' if cache_hit else ''}")

        # Search performance statistics
        avg_search_time = sum(search_times) / len(search_times)
        max_search_time = max(search_times)
        min_search_time = min(search_times)

        print(f"\n📈 Search Statistics:")
        print(f"   Average: {avg_search_time:.1f}ms")
        print(f"   Min: {min_search_time:.1f}ms")
        print(f"   Max: {max_search_time:.1f}ms")

        # Test cache performance
        print(f"\n🚀 Cache Performance Test:")

        # Run same query multiple times to test caching
        test_query = "python function example"
        cache_times = []

        for i in range(3):
            start_time = time.time()
            result = await server.search_context({'query': test_query}, None)
            end_time = time.time()

            search_time = (end_time - start_time) * 1000
            cache_hit = result.get('cache_hit', False)
            cache_times.append(search_time)

            print(f"   Run {i+1}: {search_time:.1f}ms {'[CACHE HIT]' if cache_hit else '[CACHE MISS]'}")

        if len(cache_times) > 1:
            speed_improvement = cache_times[0] / min(cache_times[1:])
            print(f"   Cache speed improvement: {speed_improvement:.1f}x")

    finally:
        # Cleanup
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)
        await server.close()

# Run the example
asyncio.run(performance_monitoring())
```

## Next Steps

After trying these basic examples:

### Explore Advanced Features

→ **[Advanced Usage Examples](advanced-usage.md)** - Knowledge graphs, custom providers, production optimization

### Real-world Integrations

→ **[Integration Examples](integration-examples.md)** - Claude Desktop, custom MCP clients, API patterns

### Solve Problems

→ **[Troubleshooting Examples](troubleshooting.md)** - Debug issues with working solutions

### Customize for Your Needs

- Modify the chunk sizes and overlap settings
- Experiment with different file patterns
- Try different embedding models
- Adjust similarity thresholds for your content

### Production Readiness

- Add proper error handling and logging
- Implement monitoring and alerts
- Scale with Redis clustering
- Optimize for your specific use case

All examples include cleanup code and error handling. Feel free to modify them for your specific requirements!
