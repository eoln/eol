# Advanced Usage Examples

Sophisticated examples for complex scenarios, production optimization, and advanced features. These examples demonstrate the full power of EOL RAG Context for demanding use cases.

## Prerequisites

These examples assume you're familiar with basic usage and have:

```bash
# Install with all extras
pip install eol-rag-context[all]

# Redis Stack with persistence
docker run -d --name redis-stack-persistent \
  -p 6379:6379 -p 8001:8001 \
  -v redis-data:/data \
  redis/redis-stack:latest

# Optional: Install additional dependencies for advanced features
pip install networkx matplotlib plotly sentence-transformers[gpu]
```

## Knowledge Graph Construction

### Automatic Entity Extraction and Relationship Discovery

Build intelligent knowledge graphs from your documents:

```python
import asyncio
import json
from pathlib import Path
from eol.rag_context import EOLRAGContextServer
from eol.rag_context.server import QueryKnowledgeGraphRequest

async def knowledge_graph_construction():
    """Build and explore knowledge graphs from document collections."""

    # Create a comprehensive test project
    project_root = Path("knowledge_graph_test")
    project_root.mkdir(exist_ok=True)

    # Create interconnected documentation
    docs = {
        "architecture.md": '''# System Architecture

## Overview
The authentication system integrates with the user database and session management. It uses JWT tokens and Redis for session storage.

## Components
- **UserService**: Manages user accounts and profiles
- **AuthenticationService**: Handles login/logout and token validation
- **SessionManager**: Manages user sessions with Redis
- **TokenService**: Creates and validates JWT tokens

## Dependencies
The authentication system depends on:
- Database connection for user storage
- Redis for session management
- JWT library for token handling
- Encryption service for password hashing

## Integration Points
- API Gateway routes authentication requests
- User interface calls authentication endpoints
- Background services validate tokens
''',

        "user_service.md": '''# User Service Documentation

## Purpose
UserService manages user accounts, profiles, and user-related operations.

## Key Features
- User registration and profile management
- Password reset and email verification
- User preference storage
- Account deactivation and deletion

## Related Components
- Integrates with AuthenticationService for login
- Uses Database for persistent storage
- Connects to EmailService for notifications
- Works with PermissionService for authorization

## API Endpoints
- POST /users - Create new user
- GET /users/{id} - Get user profile
- PUT /users/{id} - Update user profile
- DELETE /users/{id} - Deactivate user account
''',

        "database_design.md": '''# Database Design

## Schema Overview
The system uses PostgreSQL with the following main tables:

### Users Table
Stores user account information:
- Primary key: user_id
- Fields: username, email, password_hash, created_at, profile_data

### Sessions Table
Manages user sessions:
- Links to Users table via user_id
- Contains session tokens and expiration times
- Used by SessionManager for validation

### Permissions Table
Defines user permissions and roles:
- Connected to Users through user_permissions junction table
- Referenced by PermissionService for authorization

## Relationships
- Users → Sessions (one-to-many)
- Users → Permissions (many-to-many)
- Sessions referenced by AuthenticationService
''',

        "api_documentation.md": '''# API Documentation

## Authentication Endpoints

### POST /auth/login
Authenticates users and returns JWT token.
- Requires: username, password
- Returns: JWT token, session information
- Used by: User interface, mobile apps

### POST /auth/logout
Invalidates user session.
- Requires: Valid JWT token
- Side effects: Removes session from Redis
- Calls: SessionManager to clean up

### GET /auth/validate
Validates JWT token.
- Used by: API Gateway for request validation
- Integrates with: TokenService for verification

## User Management Endpoints

### User CRUD operations
All user endpoints integrate with UserService:
- Authentication required via JWT tokens
- Permission checking through PermissionService
- Database operations via UserService
''',

        "deployment.md": '''# Deployment Guide

## Infrastructure Requirements

### Database Setup
- PostgreSQL 14+ for user data and sessions
- Redis Stack for session storage and caching
- Connection pooling configured

### Authentication Services
Deploy the following services:
- **AuthenticationService**: Handles login/logout
- **UserService**: Manages user accounts
- **TokenService**: JWT token operations
- **SessionManager**: Session lifecycle

### Dependencies
- All services require database connectivity
- Redis connection for session management
- Inter-service communication via API calls
- Load balancer for high availability

## Configuration
Services configured via environment variables:
- Database connection strings
- Redis configuration
- JWT signing keys
- Session timeout settings
'''
    }

    # Write the documentation files
    for filename, content in docs.items():
        (project_root / filename).write_text(content)

    # Initialize server with knowledge graph enabled
    config = {
        "embedding": {
            "provider": "sentence_transformers",
            "model": "all-MiniLM-L6-v2",
        },
        "indexing": {
            "chunk_size": 800,
            "chunk_overlap": 150,
            "create_hierarchy": True
        },
        "knowledge_graph": {
            "enabled": True,
            "extract_entities": True,
            "build_relationships": True,
            "entity_types": ["service", "component", "database", "api", "table", "endpoint"],
            "relationship_types": ["integrates_with", "depends_on", "uses", "manages", "calls"]
        }
    }

    server = EOLRAGContextServer(config=config)
    await server.initialize()

    try:
        print("🕸️ Knowledge Graph Construction Example\n")

        # Index the documentation with knowledge graph enabled
        print("📚 Indexing documentation with entity extraction...")
        result = await server.index_directory(
            directory_path=str(project_root),
            recursive=True,
            extract_knowledge_graph=True
        )

        print(f"   ✅ Indexed {result['indexed_files']} files")
        print(f"   🧩 Created {result['total_chunks']} chunks")
        print(f"   🏷️  Extracted {result.get('entities_extracted', 0)} entities")
        print(f"   🔗 Built {result.get('relationships_created', 0)} relationships")

        # Query the knowledge graph
        print("\n🔍 Querying Knowledge Graph:")

        # Find all entities related to authentication
        auth_query = QueryKnowledgeGraphRequest(
            query="authentication system components",
            max_depth=2,
            include_relationships=True,
            entity_types=["service", "component"],
            min_confidence=0.7
        )

        auth_results = await server.query_knowledge_graph(auth_query, None)

        print(f"   Found {len(auth_results['entities'])} authentication-related entities:")
        for entity in auth_results['entities'][:5]:  # Show top 5
            print(f"      🏷️  {entity['name']} ({entity['type']}) - confidence: {entity['confidence']:.3f}")

            # Show relationships
            for rel in entity.get('relationships', [])[:3]:  # Show top 3 relationships
                print(f"         → {rel['type']}: {rel['target']} (strength: {rel['confidence']:.3f})")

        # Explore database-related entities
        print("\n📊 Database-related entities:")
        db_query = QueryKnowledgeGraphRequest(
            query="database tables and storage",
            entity_types=["database", "table"],
            include_relationships=True
        )

        db_results = await server.query_knowledge_graph(db_query, None)

        for entity in db_results['entities']:
            print(f"   📊 {entity['name']} ({entity['type']})")
            print(f"      Source: {entity['source_path']}")

            # Find what uses this database entity
            for rel in entity.get('relationships', []):
                if rel['type'] in ['uses', 'depends_on']:
                    print(f"      Used by: {rel['source']} ({rel['type']})")

        # Generate knowledge graph visualization
        print("\n📈 Generating Knowledge Graph Visualization:")

        visualization = await server.generate_graph_visualization(
            focus_entity="AuthenticationService",
            max_nodes=20,
            include_labels=True,
            color_by_type=True,
            layout='force_directed'
        )

        # Save visualization (if supported)
        viz_file = project_root / "knowledge_graph.html"
        if 'html' in visualization:
            viz_file.write_text(visualization['html'])
            print(f"   📊 Visualization saved to: {viz_file}")
            print(f"   📈 Graph stats: {visualization['stats']['nodes']} nodes, {visualization['stats']['edges']} edges")

        # Analyze system architecture using knowledge graph
        print("\n🏗️ System Architecture Analysis:")

        # Find all services and their dependencies
        services_query = QueryKnowledgeGraphRequest(
            query="system services and dependencies",
            entity_types=["service"],
            relationship_types=["depends_on", "integrates_with"],
            max_depth=3
        )

        services_results = await server.query_knowledge_graph(services_query, None)

        print("   Services and their dependencies:")
        for entity in services_results['entities']:
            if entity['type'] == 'service':
                print(f"   🔧 {entity['name']}:")

                # Group relationships by type
                deps = [r for r in entity.get('relationships', []) if r['type'] == 'depends_on']
                integrations = [r for r in entity.get('relationships', []) if r['type'] == 'integrates_with']

                if deps:
                    print(f"      Dependencies: {', '.join([d['target'] for d in deps])}")
                if integrations:
                    print(f"      Integrations: {', '.join([i['target'] for i in integrations])}")

        # Find potential integration points
        print("\n🔌 Integration Points Analysis:")

        # Look for entities that have many relationships (central components)
        central_entities = []
        for entity in auth_results['entities'] + db_results['entities']:
            relationship_count = len(entity.get('relationships', []))
            if relationship_count >= 2:  # Components with 2+ relationships
                central_entities.append((entity, relationship_count))

        # Sort by relationship count
        central_entities.sort(key=lambda x: x[1], reverse=True)

        print("   Central components (high connectivity):")
        for entity, rel_count in central_entities[:5]:
            print(f"      🌟 {entity['name']} ({entity['type']}) - {rel_count} connections")

            # Show what it connects to
            connected_to = [r['target'] for r in entity.get('relationships', [])]
            if connected_to:
                print(f"         Connects: {', '.join(connected_to[:3])}")

    finally:
        # Cleanup
        import shutil
        if project_root.exists():
            shutil.rmtree(project_root)
        await server.close()

# Run the example
asyncio.run(knowledge_graph_construction())
```

## Advanced Semantic Caching

### Optimized Caching with Analytics

Implement research-backed 31% hit rate optimization:

```python
import asyncio
import time
import random
from collections import defaultdict
from eol.rag_context import EOLRAGContextServer

async def advanced_semantic_caching():
    """Implement and optimize semantic caching for maximum efficiency."""

    # Advanced caching configuration
    cache_config = {
        "caching": {
            "enabled": True,
            "target_hit_rate": 0.31,           # Research-backed optimum
            "similarity_threshold": 0.95,      # Start conservative
            "adaptive_threshold": True,        # Auto-adjust for target hit rate
            "ttl_seconds": 1800,              # 30 minute TTL
            "max_cache_size": 2000,           # Large cache
            "eviction_policy": "lru_with_quality",  # Smart eviction
            "quality_scoring": True,          # Score cached responses
            "analytics": {
                "track_patterns": True,
                "export_metrics": True,
                "optimize_threshold": True
            }
        },
        "embedding": {
            "provider": "sentence_transformers",
            "model": "all-MiniLM-L6-v2",
        }
    }

    server = EOLRAGContextServer(config=cache_config)
    await server.initialize()

    # Create test content for caching experiments
    test_queries = [
        # Similar queries that should benefit from semantic caching
        ("authentication", [
            "how to implement user authentication",
            "user authentication implementation guide",
            "implementing authentication for users",
            "authentication system setup",
            "user login implementation tutorial"
        ]),
        ("database", [
            "database connection configuration",
            "how to configure database connections",
            "setting up database connectivity",
            "database setup and configuration",
            "connecting to database tutorial"
        ]),
        ("api", [
            "REST API endpoint documentation",
            "API endpoints and documentation",
            "how to document API endpoints",
            "creating REST API documentation",
            "API reference guide creation"
        ]),
        ("deployment", [
            "application deployment guide",
            "how to deploy the application",
            "deployment and production setup",
            "deploying to production environment",
            "production deployment tutorial"
        ])
    ]

    try:
        print("🚀 Advanced Semantic Caching Example\n")

        # First, ensure we have content to search
        print("📚 Setting up test content...")

        # Create sample documents for testing
        sample_docs = {
            "auth_guide.md": """# Authentication Guide

This guide covers user authentication implementation including login, logout, session management, and security best practices.

## Implementation Steps
1. Set up authentication middleware
2. Create login/logout endpoints
3. Implement session management
4. Add security headers and validation

## Best Practices
- Use strong password hashing
- Implement rate limiting
- Enable session timeouts
- Log authentication events
""",
            "database_config.md": """# Database Configuration

Learn how to configure database connections for optimal performance and reliability.

## Connection Setup
Configure your database connection with proper pooling, timeouts, and error handling.

## Performance Optimization
- Use connection pooling
- Set appropriate timeouts
- Monitor connection metrics
- Implement retry logic
""",
            "api_docs.md": """# API Documentation

Complete API reference with endpoints, parameters, and response formats.

## REST Endpoints
Document all API endpoints with examples, parameters, and response schemas.

## Authentication
All API endpoints require proper authentication tokens for access.
"""
        }

        # Write test documents
        for filename, content in sample_docs.items():
            Path(filename).write_text(content)

        # Index the test documents
        for filename in sample_docs.keys():
            await server.index_file(filename)

        print("✅ Test content indexed")

        # Phase 1: Baseline performance without caching
        print("\n📊 Phase 1: Measuring baseline performance (cache disabled)...")

        await server.configure_cache(enabled=False)

        baseline_times = []
        for category, queries in test_queries:
            for query in queries[:2]:  # Test 2 queries per category
                start_time = time.time()
                result = await server.search_context({'query': query, 'max_results': 3}, None)
                end_time = time.time()

                baseline_times.append((end_time - start_time) * 1000)  # Convert to ms

        baseline_avg = sum(baseline_times) / len(baseline_times)
        print(f"   Average baseline time: {baseline_avg:.1f}ms")

        # Phase 2: Enable caching and test performance
        print(f"\n🔥 Phase 2: Testing semantic cache performance...")

        await server.configure_cache(enabled=True, target_hit_rate=0.31)

        cache_performance = defaultdict(list)
        query_results = []

        # Test each query category multiple times
        for round_num in range(3):  # 3 rounds of testing
            print(f"   Round {round_num + 1}:")

            for category, queries in test_queries:
                category_times = []

                for query in queries:
                    start_time = time.time()
                    result = await server.search_context({'query': query, 'max_results': 3}, None)
                    end_time = time.time()

                    search_time = (end_time - start_time) * 1000
                    cache_hit = result.get('cache_hit', False)

                    category_times.append(search_time)
                    cache_performance[category].append({
                        'time': search_time,
                        'cache_hit': cache_hit,
                        'results': len(result['results'])
                    })

                    query_results.append({
                        'query': query,
                        'category': category,
                        'time': search_time,
                        'cache_hit': cache_hit,
                        'round': round_num
                    })

                avg_time = sum(category_times) / len(category_times)
                hit_rate = sum(1 for p in cache_performance[category] if p['cache_hit']) / len(cache_performance[category])

                print(f"      {category}: {avg_time:.1f}ms avg, {hit_rate:.1%} hit rate")

        # Phase 3: Analyze caching effectiveness
        print(f"\n📈 Phase 3: Cache Performance Analysis")

        cache_stats = await server.get_cache_stats()

        print(f"   Overall Statistics:")
        print(f"      Total queries: {cache_stats.get('total_queries', 0)}")
        print(f"      Cache hit rate: {cache_stats.get('hit_rate', 0):.1%}")
        print(f"      Target hit rate: 31%")
        print(f"      Cache size: {cache_stats.get('cache_size', 0)} entries")
        print(f"      Average response time: {cache_stats.get('avg_response_time_ms', 0):.1f}ms")

        # Calculate performance improvements
        cache_hits = [r for r in query_results if r['cache_hit']]
        cache_misses = [r for r in query_results if not r['cache_hit']]

        if cache_hits and cache_misses:
            hit_avg_time = sum(r['time'] for r in cache_hits) / len(cache_hits)
            miss_avg_time = sum(r['time'] for r in cache_misses) / len(cache_misses)
            speed_improvement = miss_avg_time / hit_avg_time

            print(f"   Performance Improvement:")
            print(f"      Cache hits: {hit_avg_time:.1f}ms average")
            print(f"      Cache misses: {miss_avg_time:.1f}ms average")
            print(f"      Speed improvement: {speed_improvement:.1f}x faster")

        # Phase 4: Cache optimization recommendations
        print(f"\n🔧 Phase 4: Cache Optimization Analysis")

        current_hit_rate = cache_stats.get('hit_rate', 0)
        target_hit_rate = 0.31

        if abs(current_hit_rate - target_hit_rate) > 0.05:  # More than 5% off target
            recommendations = await server.get_cache_recommendations()

            print(f"   Optimization Recommendations:")
            for rec in recommendations:
                print(f"      • {rec['action']}: {rec['description']}")
                if 'new_threshold' in rec:
                    print(f"        Suggested threshold: {rec['new_threshold']:.3f}")
        else:
            print(f"   ✅ Cache performance is optimal (within 5% of target)")

        # Phase 5: Advanced cache analytics
        print(f"\n📊 Phase 5: Advanced Cache Analytics")

        analytics = await server.get_cache_analytics()

        if analytics:
            print(f"   Query Patterns:")
            for pattern, stats in analytics.get('query_patterns', {}).items():
                print(f"      {pattern}: {stats['count']} queries, {stats['hit_rate']:.1%} hit rate")

            print(f"   Quality Metrics:")
            quality_metrics = analytics.get('quality_metrics', {})
            print(f"      Average relevance: {quality_metrics.get('avg_relevance', 0):.3f}")
            print(f"      False positive rate: {quality_metrics.get('false_positive_rate', 0):.1%}")
            print(f"      User satisfaction: {quality_metrics.get('user_satisfaction', 0):.1%}")

        # Phase 6: Cache warming strategy
        print(f"\n🔥 Phase 6: Intelligent Cache Warming")

        # Identify common query patterns for warming
        warming_queries = [
            "getting started guide",
            "installation instructions",
            "configuration examples",
            "troubleshooting common issues",
            "API reference documentation",
            "best practices and guidelines"
        ]

        print(f"   Warming cache with {len(warming_queries)} common patterns...")

        warmed_count = 0
        for query in warming_queries:
            try:
                result = await server.search_context({'query': query, 'max_results': 3}, None)
                if result['results']:
                    warmed_count += 1
            except Exception as e:
                print(f"      Warning: Could not warm query '{query}': {e}")

        print(f"   ✅ Cache warmed with {warmed_count}/{len(warming_queries)} queries")

        # Final cache statistics
        final_stats = await server.get_cache_stats()
        print(f"   Final cache size: {final_stats.get('cache_size', 0)} entries")

    finally:
        # Cleanup test files
        for filename in sample_docs.keys():
            file_path = Path(filename)
            if file_path.exists():
                file_path.unlink()

        await server.close()

# Run the example
asyncio.run(advanced_semantic_caching())
```

## Custom Embedding Providers

### Multi-Provider Setup with Specialized Models

Use different embedding models for different content types:

```python
import asyncio
import numpy as np
from pathlib import Path
from eol.rag_context import EOLRAGContextServer
from eol.rag_context.embeddings import EmbeddingProvider

class SpecializedCodeEmbedding(EmbeddingProvider):
    """Custom embedding provider optimized for code content."""

    def __init__(self, model_config: dict):
        super().__init__(model_config)
        self.model_name = model_config.get('model', 'code-specialized')
        self.dimension = 384  # Match sentence-transformers dimension

        # In a real implementation, you'd load a code-specialized model
        # For this example, we'll simulate with modified sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            # Use a model that's been fine-tuned on code or has code understanding
            self.model = SentenceTransformer('microsoft/codebert-base')
            self.dimension = self.model.get_sentence_embedding_dimension()
        except ImportError:
            print("⚠️ sentence-transformers not available, using mock embeddings")
            self.model = None

    async def get_embedding(self, text: str) -> np.ndarray:
        """Generate code-optimized embedding."""
        if self.model is None:
            # Mock embedding for demonstration
            return np.random.rand(self.dimension).astype(np.float32)

        # Preprocess code text for better embeddings
        processed_text = self._preprocess_code(text)
        embedding = self.model.encode(processed_text)
        return embedding.astype(np.float32)

    def _preprocess_code(self, text: str) -> str:
        """Preprocess code for better semantic understanding."""
        # Remove excessive whitespace
        lines = text.split('\n')
        processed_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):  # Skip empty lines and comments
                processed_lines.append(stripped)

        # Join with single spaces to create dense representation
        return ' '.join(processed_lines)

    async def get_embeddings_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple code snippets."""
        if self.model is None:
            return [await self.get_embedding(text) for text in texts]

        processed_texts = [self._preprocess_code(text) for text in texts]
        embeddings = self.model.encode(processed_texts)
        return [emb.astype(np.float32) for emb in embeddings]

class DocumentationEmbedding(EmbeddingProvider):
    """Custom embedding provider optimized for documentation."""

    def __init__(self, model_config: dict):
        super().__init__(model_config)
        self.model_name = model_config.get('model', 'docs-specialized')
        self.dimension = 384

        try:
            from sentence_transformers import SentenceTransformer
            # Use a model optimized for longer text and documentation
            self.model = SentenceTransformer('all-mpnet-base-v2')
            self.dimension = self.model.get_sentence_embedding_dimension()
        except ImportError:
            print("⚠️ sentence-transformers not available, using mock embeddings")
            self.model = None

    async def get_embedding(self, text: str) -> np.ndarray:
        """Generate documentation-optimized embedding."""
        if self.model is None:
            return np.random.rand(self.dimension).astype(np.float32)

        # Preprocess documentation for better semantic capture
        processed_text = self._preprocess_documentation(text)
        embedding = self.model.encode(processed_text)
        return embedding.astype(np.float32)

    def _preprocess_documentation(self, text: str) -> str:
        """Preprocess documentation for better semantic understanding."""
        # Remove markdown artifacts that might confuse the model
        import re

        # Remove markdown headers symbols but keep the text
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)

        # Remove code block markers but keep content
        text = re.sub(r'```\w*\n', '', text)
        text = re.sub(r'```', '', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    async def get_embeddings_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple documentation chunks."""
        if self.model is None:
            return [await self.get_embedding(text) for text in texts]

        processed_texts = [self._preprocess_documentation(text) for text in texts]
        embeddings = self.model.encode(processed_texts)
        return [emb.astype(np.float32) for emb in embeddings]

async def multi_provider_setup():
    """Demonstrate multi-provider embedding setup with specialized models."""

    # Create test content with different types
    test_project = Path("multi_provider_test")
    test_project.mkdir(exist_ok=True)

    # Create different content types
    content = {
        "code/auth.py": '''"""Authentication module with JWT tokens."""

import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

class AuthenticationService:
    """Handles user authentication and JWT token management."""

    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_expiry = timedelta(hours=24)

    async def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user credentials and return user data."""
        user = await self._get_user_by_username(username)
        if user and self._verify_password(password, user['password_hash']):
            return user
        return None

    def generate_token(self, user_data: Dict[str, Any]) -> str:
        """Generate JWT token for authenticated user."""
        payload = {
            'user_id': user_data['id'],
            'username': user_data['username'],
            'exp': datetime.utcnow() + self.token_expiry,
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def _verify_password(self, password: str, hash: str) -> bool:
        """Verify password against bcrypt hash."""
        return bcrypt.checkpw(password.encode('utf-8'), hash)
''',

        "code/database.py": '''"""Database connection and query utilities."""

import asyncpg
import asyncio
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

class DatabaseManager:
    """Manages PostgreSQL database connections and queries."""

    def __init__(self, connection_string: str, pool_size: int = 10):
        self.connection_string = connection_string
        self.pool_size = pool_size
        self.pool: Optional[asyncpg.Pool] = None

    async def initialize(self):
        """Initialize database connection pool."""
        self.pool = await asyncpg.create_pool(
            self.connection_string,
            min_size=1,
            max_size=self.pool_size
        )

    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool."""
        if not self.pool:
            raise RuntimeError("Database pool not initialized")

        async with self.pool.acquire() as connection:
            yield connection

    async def execute_query(self, query: str, *args) -> List[Dict[str, Any]]:
        """Execute SELECT query and return results."""
        async with self.get_connection() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]

    async def execute_command(self, command: str, *args) -> str:
        """Execute INSERT/UPDATE/DELETE command."""
        async with self.get_connection() as conn:
            return await conn.execute(command, *args)
''',

        "docs/authentication.md": '''# Authentication System Documentation

## Overview

The authentication system provides secure user authentication using JWT tokens and bcrypt password hashing. It includes user registration, login, logout, and token validation capabilities.

## Key Features

### Secure Password Handling
- Passwords are hashed using bcrypt with salt
- No plain text passwords are stored
- Configurable hash complexity

### JWT Token Management
- Stateless authentication using JSON Web Tokens
- Configurable token expiration times
- Support for token refresh workflows
- Secure token validation

### User Session Management
- Session tracking and management
- Automatic session cleanup
- Configurable session timeouts

## API Endpoints

### POST /auth/register
Register a new user account.

**Request Body:**
```json
{
  "username": "string",
  "email": "string",
  "password": "string"
}
```

**Response:**

```json
{
  "user_id": "uuid",
  "username": "string",
  "token": "jwt_string"
}
```

### POST /auth/login

Authenticate existing user and return token.

**Request Body:**

```json
{
  "username": "string",
  "password": "string"
}
```

**Response:**

```json
{
  "user_id": "uuid",
  "username": "string",
  "token": "jwt_string",
  "expires": "datetime"
}
```

## Security Considerations

### Password Security

- Minimum password length: 8 characters
- Password complexity requirements configurable
- Rate limiting on authentication attempts
- Account lockout after failed attempts

### Token Security

- JWT tokens signed with secure secret key
- Short token expiration times recommended
- Token validation on every request
- Proper token storage on client side

## Configuration

### Environment Variables

- `JWT_SECRET_KEY`: Secret key for JWT signing
- `JWT_EXPIRY_HOURS`: Token expiration time
- `BCRYPT_ROUNDS`: Password hash complexity
- `MAX_LOGIN_ATTEMPTS`: Failed login threshold

### Database Schema

The authentication system requires these database tables:

- `users`: User account information
- `sessions`: Active user sessions
- `login_attempts`: Failed login tracking
''',

        "docs/database.md": '''# Database Configuration and Management

## Database Setup

The application uses PostgreSQL as the primary database with asyncio-based connection pooling for optimal performance.

## Connection Configuration

### Basic Configuration

Configure the database connection using environment variables or configuration files:

```yaml
database:
  host: localhost
  port: 5432
  name: myapp
  user: dbuser
  password: ${DATABASE_PASSWORD}
  ssl_mode: prefer
```

### Connection Pooling

The database manager uses asyncpg connection pooling to handle multiple concurrent connections efficiently:

- **Minimum pool size**: 1 connection
- **Maximum pool size**: Configurable (default: 10)
- **Connection timeout**: 30 seconds
- **Command timeout**: 60 seconds

## Schema Design

### Core Tables

#### users

Stores user account information:

- `id` (UUID, Primary Key)
- `username` (String, Unique)
- `email` (String, Unique)
- `password_hash` (String)
- `created_at` (Timestamp)
- `updated_at` (Timestamp)

#### sessions

Manages user sessions:

- `id` (UUID, Primary Key)
- `user_id` (UUID, Foreign Key → users.id)
- `token_hash` (String)
- `expires_at` (Timestamp)
- `created_at` (Timestamp)

## Query Patterns

### User Management Queries

Common database operations for user management:

```sql
-- Create new user
INSERT INTO users (id, username, email, password_hash, created_at)
VALUES ($1, $2, $3, $4, NOW());

-- Find user by username
SELECT id, username, email, password_hash
FROM users
WHERE username = $1;

-- Update user profile
UPDATE users
SET email = $1, updated_at = NOW()
WHERE id = $2;
```

### Session Management Queries

Database operations for session handling:

```sql
-- Create session
INSERT INTO sessions (id, user_id, token_hash, expires_at, created_at)
VALUES ($1, $2, $3, $4, NOW());

-- Validate session
SELECT user_id, expires_at
FROM sessions
WHERE token_hash = $1 AND expires_at > NOW();

-- Clean expired sessions
DELETE FROM sessions WHERE expires_at < NOW();
```

## Performance Optimization

### Indexing Strategy

Key indexes for optimal query performance:

```sql
-- User lookup indexes
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);

-- Session lookup indexes
CREATE INDEX idx_sessions_token_hash ON sessions(token_hash);
CREATE INDEX idx_sessions_expires_at ON sessions(expires_at);
CREATE INDEX idx_sessions_user_id ON sessions(user_id);
```

### Connection Pool Tuning

Optimize connection pool settings based on your application load:

- **Low traffic**: 1-5 connections
- **Medium traffic**: 5-15 connections
- **High traffic**: 15-50 connections

Monitor connection usage and adjust pool size accordingly.

## Backup and Maintenance

### Regular Backups

Implement automated backups with retention policies:

- Daily full backups retained for 30 days
- Weekly backups retained for 12 weeks
- Monthly backups retained for 12 months

### Maintenance Tasks

Regular maintenance to keep the database healthy:

- `VACUUM` and `ANALYZE` on high-activity tables
- Index maintenance and rebuilding
- Connection pool monitoring
- Query performance analysis
''',

        "config/settings.yaml": '''# Application Configuration

# Database Configuration

database:

# Connection settings

  host: ${DB_HOST:-localhost}
  port: ${DB_PORT:-5432}
  name: ${DB_NAME:-myapp}
  user: ${DB_USER:-postgres}
  password: ${DB_PASSWORD}

# Connection pooling

  pool:
    min_size: 1
    max_size: 10
    timeout: 30
    command_timeout: 60

# SSL settings

  ssl:
    enabled: ${DB_SSL_ENABLED:-true}
    cert_file: ${DB_SSL_CERT}
    key_file: ${DB_SSL_KEY}

# Authentication Configuration

authentication:

# JWT settings

  jwt:
    secret_key: ${JWT_SECRET_KEY}
    algorithm: HS256
    expiry_hours: 24
    refresh_enabled: true

# Password settings

  password:
    min_length: 8
    require_uppercase: true
    require_numbers: true
    require_symbols: false
    bcrypt_rounds: 12

# Rate limiting

  rate_limit:
    max_attempts: 5
    lockout_duration_minutes: 15
    window_minutes: 60

# Redis Configuration

redis:

# Connection

  host: ${REDIS_HOST:-localhost}
  port: ${REDIS_PORT:-6379}
  password: ${REDIS_PASSWORD}
  db: ${REDIS_DB:-0}

# Pool settings

  pool:
    max_connections: 20
    retry_on_timeout: true
    socket_timeout: 5

# Logging Configuration

logging:
  level: ${LOG_LEVEL:-INFO}
  format: json
  file: logs/app.log
  max_size_mb: 100
  backup_count: 5

# Structured logging

  structured: true
  include_timestamp: true
  include_level: true
  include_logger_name: true

# Application Settings

app:
  name: MyApplication
  version: 1.0.0
  debug: ${DEBUG:-false}

# Server settings

  host: ${HOST:-0.0.0.0}
  port: ${PORT:-8000}
  workers: ${WORKERS:-4}

# CORS settings

  cors:
    enabled: true
    origins:
      - <https://myapp.com>
      - <https://admin.myapp.com>
    methods: [GET, POST, PUT, DELETE]
    headers: [Content-Type, Authorization]
'''
    }

    # Write test files
    for file_path, file_content in content.items():
        full_path = test_project / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(file_content)

    # Configure multi-provider setup
    config = {
        "embedding": {
            "multi_provider": True,
            "providers": {
                "code": {
                    "provider_class": "SpecializedCodeEmbedding",
                    "model": "code-specialized-v1",
                    "batch_size": 16
                },
                "documentation": {
                    "provider_class": "DocumentationEmbedding",
                    "model": "docs-optimized-v1",
                    "batch_size": 8
                },
                "config": {
                    "provider": "sentence_transformers",
                    "model": "all-MiniLM-L6-v2",
                    "batch_size": 32
                }
            },
            "routing_rules": {
                "*.py": "code",
                "*.js": "code",
                "*.ts": "code",
                "*.md": "documentation",
                "*.rst": "documentation",
                "*.txt": "documentation",
                "*.yaml": "config",
                "*.yml": "config",
                "*.json": "config"
            }
        }
    }

    server = EOLRAGContextServer(config=config)

    try:
        print("🔧 Multi-Provider Embedding Setup Example\n")

        # Register custom providers
        print("📝 Registering custom embedding providers...")
        server.register_embedding_provider('SpecializedCodeEmbedding', SpecializedCodeEmbedding)
        server.register_embedding_provider('DocumentationEmbedding', DocumentationEmbedding)

        await server.initialize()

        print("✅ Server initialized with multi-provider setup")

        # Index content with provider routing
        print("\n📚 Indexing content with specialized providers...")
        result = await server.index_directory(
            directory_path=str(test_project),
            recursive=True,
            use_provider_routing=True
        )

        print(f"   ✅ Indexed {result['indexed_files']} files")
        print(f"   🧩 Created {result['total_chunks']} chunks")

        # Show provider usage breakdown
        provider_stats = result.get('provider_usage', {})
        if provider_stats:
            print(f"\n🎯 Provider Usage Breakdown:")
            for provider, stats in provider_stats.items():
                print(f"   {provider}: {stats['files_processed']} files, {stats['chunks_created']} chunks")

        # Test search quality with different providers
        print(f"\n🔍 Testing Search Quality by Content Type:")

        # Test code-specific queries
        code_queries = [
            "JWT token generation implementation",
            "database connection pooling setup",
            "password hashing with bcrypt",
            "async database query execution"
        ]

        print(f"\n   🐍 Code Queries (using specialized code embeddings):")
        for query in code_queries:
            result = await server.search_context({
                'query': query,
                'max_results': 2,
                'filters': {'file_types': ['.py']}
            }, None)

            print(f"      '{query}':")
            for r in result['results']:
                print(f"         📄 {Path(r['source_path']).name} (score: {r['similarity']:.3f})")
                print(f"             {r['content'][:80]}...")

        # Test documentation queries
        doc_queries = [
            "authentication system overview",
            "database configuration guide",
            "JWT security considerations",
            "connection pool performance tuning"
        ]

        print(f"\n   📚 Documentation Queries (using specialized doc embeddings):")
        for query in doc_queries:
            result = await server.search_context({
                'query': query,
                'max_results': 2,
                'filters': {'file_types': ['.md']}
            }, None)

            print(f"      '{query}':")
            for r in result['results']:
                print(f"         📄 {Path(r['source_path']).name} (score: {r['similarity']:.3f})")
                print(f"             {r['content'][:80]}...")

        # Test configuration queries
        config_queries = [
            "database connection settings",
            "JWT configuration options",
            "logging configuration setup",
            "Redis connection parameters"
        ]

        print(f"\n   ⚙️ Configuration Queries (using general embeddings):")
        for query in config_queries:
            result = await server.search_context({
                'query': query,
                'max_results': 2,
                'filters': {'file_types': ['.yaml', '.yml']}
            }, None)

            print(f"      '{query}':")
            for r in result['results']:
                print(f"         📄 {Path(r['source_path']).name} (score: {r['similarity']:.3f})")
                print(f"             {r['content'][:80]}...")

        # Compare cross-provider search results
        print(f"\n⚖️ Cross-Provider Comparison:")

        test_query = "user authentication implementation"

        # Search all content types
        all_results = await server.search_context({
            'query': test_query,
            'max_results': 6
        }, None)

        print(f"   Query: '{test_query}'")
        print(f"   Results across all providers:")

        for r in all_results['results']:
            file_path = Path(r['source_path'])
            provider_type = "code" if file_path.suffix == '.py' else "docs" if file_path.suffix == '.md' else "config"
            print(f"      📄 {file_path.name} [{provider_type}] (score: {r['similarity']:.3f})")

        # Performance comparison
        print(f"\n⚡ Provider Performance Comparison:")

        import time

        # Test each provider's performance
        provider_performance = {}

        for provider_name in ['code', 'documentation', 'config']:
            # Get appropriate file extension for this provider
            ext_map = {'code': '.py', 'documentation': '.md', 'config': '.yaml'}
            file_ext = ext_map[provider_name]

            start_time = time.time()

            result = await server.search_context({
                'query': f'{provider_name} example query',
                'max_results': 3,
                'filters': {'file_types': [file_ext]}
            }, None)

            end_time = time.time()

            provider_performance[provider_name] = {
                'time_ms': (end_time - start_time) * 1000,
                'results': len(result['results'])
            }

        for provider, perf in provider_performance.items():
            print(f"   {provider}: {perf['time_ms']:.1f}ms ({perf['results']} results)")

    finally:
        # Cleanup
        import shutil
        if test_project.exists():
            shutil.rmtree(test_project)

        await server.close()

# Run the example

asyncio.run(multi_provider_setup())

```

## Production Scaling and Monitoring

### High-Availability Setup with Comprehensive Monitoring

Configure EOL RAG Context for production deployment:

```python
import asyncio
import json
import time
import psutil
from datetime import datetime, timedelta
from pathlib import Path
from eol.rag_context import EOLRAGContextServer

async def production_scaling_example():
    """Demonstrate production-ready scaling and monitoring setup."""

    # Production configuration
    production_config = {
        "redis": {
            # Redis cluster configuration
            "cluster_mode": True,
            "nodes": [
                {"host": "redis-1", "port": 6379},
                {"host": "redis-2", "port": 6379},
                {"host": "redis-3", "port": 6379}
            ],
            "max_connections": 100,
            "connection_pool_size": 20,
            "retry_on_timeout": True,
            "health_check_interval": 30,
            "socket_timeout": 5,
            "socket_connect_timeout": 5
        },

        "embedding": {
            "provider": "openai",  # Production-grade provider
            "model": "text-embedding-ada-002",
            "batch_size": 100,
            "parallel_workers": 8,
            "connection_pooling": True,
            "rate_limiting": {
                "requests_per_minute": 3000,
                "tokens_per_minute": 1000000
            },
            "retry_strategy": {
                "max_retries": 3,
                "backoff_factor": 2,
                "max_backoff": 60
            }
        },

        "indexing": {
            "parallel_indexing": True,
            "worker_count": 16,
            "batch_size": 100,
            "memory_limit_mb": 8192,
            "checkpoint_interval": 1000,
            "error_handling": "log_and_continue",
            "performance_monitoring": True
        },

        "caching": {
            "enabled": True,
            "distributed": True,
            "replication_factor": 2,
            "max_cache_size": 50000,
            "target_hit_rate": 0.31,
            "adaptive_threshold": True,
            "eviction_policy": "adaptive_lru",
            "quality_scoring": True,
            "analytics": True
        },

        "monitoring": {
            "enabled": True,
            "detailed_metrics": True,
            "export_prometheus": True,
            "export_port": 9090,
            "log_level": "INFO",
            "log_format": "json",
            "structured_logging": True,
            "alert_webhooks": [
                "http://alertmanager:9093/api/v1/alerts"
            ],
            "health_check": {
                "enabled": True,
                "interval": 30,
                "timeout": 10,
                "endpoints": ["/health", "/metrics", "/ready"]
            }
        },

        "performance": {
            "max_memory_mb": 16384,
            "gc_threshold": 0.8,
            "gc_interval": 300,
            "query_timeout": 30,
            "batch_timeout": 300,
            "connection_timeout": 10
        },

        "security": {
            "enable_auth": True,
            "api_keys": {
                "admin": "${ADMIN_API_KEY}",
                "readonly": "${READONLY_API_KEY}"
            },
            "rate_limiting": {
                "requests_per_minute": 1000,
                "burst_size": 100
            },
            "cors": {
                "enabled": True,
                "origins": ["https://myapp.com", "https://admin.myapp.com"],
                "methods": ["GET", "POST", "PUT"],
                "headers": ["Authorization", "Content-Type"]
            }
        }
    }

    # Note: For this example, we'll use a simplified config that works locally
    local_config = {
        "redis": {
            "url": "redis://localhost:6379",
            "max_connections": 50,
            "connection_pool_size": 10
        },
        "embedding": {
            "provider": "sentence_transformers",
            "model": "all-MiniLM-L6-v2",
            "batch_size": 32
        },
        "indexing": {
            "parallel_indexing": True,
            "worker_count": 4,
            "batch_size": 50
        },
        "monitoring": {
            "enabled": True,
            "detailed_metrics": True,
            "performance_tracking": True
        }
    }

    server = EOLRAGContextServer(config=local_config)

    try:
        print("🏭 Production Scaling and Monitoring Example\n")

        await server.initialize()

        # Setup monitoring and alerting
        print("📊 Setting up comprehensive monitoring...")

        await server.enable_monitoring(
            track_performance=True,
            track_memory=True,
            track_errors=True,
            export_metrics=True
        )

        await server.setup_health_checks(
            check_redis=True,
            check_embeddings=True,
            check_memory=True,
            check_disk_space=True
        )

        print("✅ Monitoring and health checks configured")

        # Simulate production workload
        print("\n🔄 Simulating production workload...")

        # Create large-scale test content
        test_content = await create_large_test_dataset()

        # Performance benchmarking
        print("\n⚡ Performance Benchmarking:")

        # Indexing performance
        print("   Indexing Performance Test:")

        indexing_start = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Simulate batch indexing
        batch_results = []
        for i in range(5):  # 5 batches
            batch_start = time.time()

            # Create batch of test files
            batch_files = await create_test_batch(f"batch_{i}", 20)  # 20 files per batch

            result = await server.index_directory(
                directory_path=str(batch_files),
                recursive=True,
                batch_size=10,
                parallel_workers=2
            )

            batch_end = time.time()
            batch_time = batch_end - batch_start

            batch_results.append({
                'batch': i,
                'files': result['indexed_files'],
                'chunks': result['total_chunks'],
                'time': batch_time,
                'files_per_second': result['indexed_files'] / batch_time
            })

            print(f"      Batch {i}: {result['indexed_files']} files, {batch_time:.1f}s ({result['indexed_files']/batch_time:.1f} files/sec)")

            # Cleanup batch
            import shutil
            shutil.rmtree(batch_files)

        indexing_end = time.time()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024

        total_files = sum(b['files'] for b in batch_results)
        total_chunks = sum(b['chunks'] for b in batch_results)
        total_time = indexing_end - indexing_start
        memory_increase = final_memory - initial_memory

        print(f"\n   📊 Indexing Summary:")
        print(f"      Total files: {total_files}")
        print(f"      Total chunks: {total_chunks}")
        print(f"      Total time: {total_time:.1f}s")
        print(f"      Average throughput: {total_files/total_time:.1f} files/sec")
        print(f"      Memory increase: {memory_increase:.1f} MB")

        # Search performance testing
        print("\n   🔍 Search Performance Test:")

        search_queries = [
            "user authentication system",
            "database configuration",
            "API endpoint documentation",
            "error handling best practices",
            "deployment and scaling",
            "security and authorization",
            "performance optimization",
            "logging and monitoring"
        ]

        search_times = []
        cache_hits = 0

        # Run multiple rounds to test caching
        for round_num in range(3):
            print(f"      Round {round_num + 1}:")

            round_times = []

            for query in search_queries:
                start_time = time.time()

                result = await server.search_context({
                    'query': query,
                    'max_results': 5,
                    'similarity_threshold': 0.7
                }, None)

                end_time = time.time()
                search_time = (end_time - start_time) * 1000  # Convert to ms

                round_times.append(search_time)
                search_times.append(search_time)

                if result.get('cache_hit', False):
                    cache_hits += 1

            avg_round_time = sum(round_times) / len(round_times)
            print(f"         Average: {avg_round_time:.1f}ms")

        total_queries = len(search_queries) * 3
        cache_hit_rate = cache_hits / total_queries
        avg_search_time = sum(search_times) / len(search_times)

        print(f"\n   📈 Search Summary:")
        print(f"      Total queries: {total_queries}")
        print(f"      Average time: {avg_search_time:.1f}ms")
        print(f"      Cache hit rate: {cache_hit_rate:.1%}")
        print(f"      95th percentile: {sorted(search_times)[int(0.95 * len(search_times))]:.1f}ms")

        # System health monitoring
        print("\n🏥 System Health Monitoring:")

        health_status = await server.get_health_status()

        print(f"   Overall Status: {health_status.get('status', 'unknown').upper()}")

        health_checks = health_status.get('checks', {})
        for check_name, check_result in health_checks.items():
            status_icon = "✅" if check_result.get('healthy', False) else "❌"
            print(f"   {status_icon} {check_name}: {check_result.get('message', 'N/A')}")

        # Performance metrics
        print("\n📊 Performance Metrics:")

        metrics = await server.get_performance_metrics()

        print(f"   System Metrics:")
        print(f"      Memory usage: {metrics.get('memory_usage_mb', 0):.0f} MB")
        print(f"      CPU usage: {metrics.get('cpu_percent', 0):.1f}%")
        print(f"      Active connections: {metrics.get('active_connections', 0)}")

        print(f"   Application Metrics:")
        print(f"      Queries per second: {metrics.get('queries_per_second', 0):.1f}")
        print(f"      Average response time: {metrics.get('avg_response_time_ms', 0):.1f}ms")
        print(f"      Error rate: {metrics.get('error_rate', 0):.2%}")
        print(f"      Cache hit rate: {metrics.get('cache_hit_rate', 0):.1%}")

        # Resource utilization alerts
        print("\n⚠️ Resource Utilization Alerts:")

        alerts = await server.get_performance_alerts()

        if alerts:
            for alert in alerts:
                severity_icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(alert['severity'], "❓")
                print(f"   {severity_icon} {alert['message']}")
        else:
            print("   ✅ No alerts - system operating within normal parameters")

        # Scaling recommendations
        print("\n📈 Scaling Recommendations:")

        recommendations = await server.get_scaling_recommendations()

        for rec in recommendations:
            priority_icon = {"low": "🔵", "medium": "🟡", "high": "🔴"}.get(rec['priority'], "⚪")
            print(f"   {priority_icon} {rec['component']}: {rec['recommendation']}")
            if 'current_value' in rec and 'recommended_value' in rec:
                print(f"      Current: {rec['current_value']}, Recommended: {rec['recommended_value']}")

        # Export metrics for external monitoring
        print("\n📤 Exporting Metrics:")

        metrics_data = await server.export_prometheus_metrics()

        if metrics_data:
            metrics_file = Path("production_metrics.txt")
            metrics_file.write_text(metrics_data)
            print(f"   ✅ Prometheus metrics exported to: {metrics_file}")

        # Generate performance report
        print("\n📋 Performance Report:")

        report = {
            "timestamp": datetime.now().isoformat(),
            "indexing": {
                "total_files": total_files,
                "total_chunks": total_chunks,
                "throughput_files_per_sec": total_files / total_time,
                "memory_usage_mb": memory_increase
            },
            "search": {
                "total_queries": total_queries,
                "avg_response_time_ms": avg_search_time,
                "cache_hit_rate": cache_hit_rate,
                "p95_response_time_ms": sorted(search_times)[int(0.95 * len(search_times))]
            },
            "system": {
                "memory_mb": metrics.get('memory_usage_mb', 0),
                "cpu_percent": metrics.get('cpu_percent', 0),
                "health_status": health_status.get('status', 'unknown')
            }
        }

        report_file = Path("performance_report.json")
        report_file.write_text(json.dumps(report, indent=2))
        print(f"   📄 Detailed report saved to: {report_file}")

    finally:
        await server.close()

async def create_large_test_dataset():
    """Create a large test dataset for performance testing."""
    # In a real implementation, this would create substantial test content
    # For the example, we'll just return a placeholder
    return {"message": "Large test dataset would be created here"}

async def create_test_batch(batch_name: str, file_count: int) -> Path:
    """Create a batch of test files for indexing performance testing."""
    batch_dir = Path(f"test_batch_{batch_name}")
    batch_dir.mkdir(exist_ok=True)

    # Create test files with varied content
    for i in range(file_count):
        file_content = f'''# Test Document {i} for {batch_name}

This is a test document created for performance benchmarking.

## Overview
This document contains sample content to test indexing performance
and search functionality.

## Content
Content for file {i} in batch {batch_name}:
- Feature description
- Implementation details
- Usage examples
- Configuration options

## Code Example
```python
def test_function_{i}():
    """Test function for file {i}."""
    return f"Hello from file {i} in {batch_name}"
```

## Additional Content

{' '.join([f'Word{j}' for j in range(50)])}  # 50 words of content
'''

        file_path = batch_dir / f"test_doc_{i}.md"
        file_path.write_text(file_content)

    return batch_dir

# Run the example

asyncio.run(production_scaling_example())

```

## Next Steps

After exploring these advanced examples:

### Integration Patterns
→ **[Integration Examples](integration-examples.md)** - Real-world deployment patterns and MCP integrations

### Problem Solving
→ **[Troubleshooting Examples](troubleshooting.md)** - Debug complex issues with proven solutions

### Custom Development
- Build custom embedding providers for your domain
- Implement specialized chunking strategies
- Create custom knowledge graph extractors
- Develop monitoring dashboards

### Production Deployment
- Implement proper logging and monitoring
- Set up Redis clustering for scale
- Configure load balancing and failover
- Establish backup and recovery procedures

These advanced examples demonstrate the full capabilities of EOL RAG Context for sophisticated production use cases. Adapt them to your specific requirements and scale as needed.
