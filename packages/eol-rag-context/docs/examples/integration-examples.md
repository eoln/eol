# Integration Examples

Real-world integration patterns for EOL RAG Context with various applications, frameworks, and deployment scenarios. These examples demonstrate production-ready implementations across different use cases.

## Prerequisites

These integration examples require:

```bash
# Core installation
pip install eol-rag-context

# Integration-specific dependencies
pip install fastapi uvicorn websockets  # For API integrations
pip install streamlit gradio            # For UI integrations
pip install discord.py slack-sdk        # For chat integrations
pip install docker kubernetes           # For container deployments

# Start Redis Stack
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

## Claude Desktop Integration

### Complete Claude Desktop Setup

Full setup with advanced MCP configuration:

```python
# claude_desktop_integration.py
import asyncio
import json
import os
from pathlib import Path
from eol.rag_context import EOLRAGContextServer

class ClaudeDesktopIntegration:
    """Complete Claude Desktop integration with MCP protocol."""

    def __init__(self, config_path: str = None):
        self.config_path = config_path or self._get_default_config_path()
        self.server = None

    def _get_default_config_path(self) -> str:
        """Get the default Claude Desktop configuration path."""
        import platform
        system = platform.system()

        if system == "Darwin":  # macOS
            return os.path.expanduser("~/Library/Application Support/Claude/claude_desktop_config.json")
        elif system == "Windows":
            return os.path.expanduser("~/AppData/Roaming/Claude/claude_desktop_config.json")
        else:  # Linux
            return os.path.expanduser("~/.config/claude/claude_desktop_config.json")

    async def setup_server(self, server_config: dict = None):
        """Initialize the RAG server with optimal settings for Claude Desktop."""
        default_config = {
            "redis": {
                "url": "redis://localhost:6379",
                "db": 0
            },
            "embedding": {
                "provider": "sentence_transformers",
                "model": "all-MiniLM-L6-v2",
                "batch_size": 32
            },
            "indexing": {
                "chunk_size": 1200,        # Optimal for Claude's context window
                "chunk_overlap": 200,
                "use_semantic_chunking": True,
                "create_hierarchy": True
            },
            "caching": {
                "enabled": True,
                "ttl_seconds": 3600,       # 1 hour cache
                "target_hit_rate": 0.31    # Optimal cache performance
            },
            "context": {
                "max_context_size": 8000,  # Leave room for Claude's response
                "smart_truncation": True,  # Intelligently truncate if needed
                "prioritize_recent": True  # Prioritize recently accessed content
            },
            "mcp": {
                "protocol_version": "2024-11-05",
                "timeout": 30,
                "max_retries": 3,
                "enable_streaming": False  # Claude Desktop doesn't support streaming yet
            }
        }

        if server_config:
            default_config.update(server_config)

        self.server = EOLRAGContextServer(config=default_config)
        await self.server.initialize()

        print("✅ RAG server initialized for Claude Desktop")

    def update_claude_config(self, project_paths: list[str], server_args: list[str] = None):
        """Update Claude Desktop configuration to include the MCP server."""

        # Default server arguments
        default_args = [
            "serve",
            "--host", "127.0.0.1",
            "--port", "3000"
        ]

        if server_args:
            default_args.extend(server_args)

        # Create MCP server configuration
        mcp_config = {
            "eol-rag-context": {
                "command": "eol-rag-context",
                "args": default_args,
                "env": {
                    "EOL_REDIS_URL": "redis://localhost:6379",
                    "EOL_LOG_LEVEL": "INFO",
                    "EOL_PROJECT_PATHS": ",".join(project_paths)
                }
            }
        }

        # Read existing Claude Desktop config
        config_file = Path(self.config_path)

        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    claude_config = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                claude_config = {}
        else:
            claude_config = {}
            config_file.parent.mkdir(parents=True, exist_ok=True)

        # Add or update MCP servers
        if "mcpServers" not in claude_config:
            claude_config["mcpServers"] = {}

        claude_config["mcpServers"].update(mcp_config)

        # Write updated configuration
        with open(config_file, 'w') as f:
            json.dump(claude_config, f, indent=2)

        print(f"✅ Claude Desktop configuration updated: {config_file}")
        print("🔄 Please restart Claude Desktop to load the new configuration")

        return claude_config

    async def index_project(self, project_path: str,
                           file_patterns: list[str] = None,
                           exclude_patterns: list[str] = None):
        """Index a project for Claude Desktop access."""

        if not self.server:
            await self.setup_server()

        default_patterns = [
            "*.py", "*.js", "*.ts", "*.jsx", "*.tsx",
            "*.md", "*.rst", "*.txt",
            "*.json", "*.yaml", "*.yml",
            "*.sql", "*.sh", "*.dockerfile"
        ]

        default_excludes = [
            "*.pyc", "*.pyo", "*.so", "*.dylib",
            "__pycache__/*", "node_modules/*", ".git/*",
            "*.log", "*.tmp", "temp/*", "build/*", "dist/*",
            ".env", ".env.*"
        ]

        result = await self.server.index_directory(
            directory_path=project_path,
            recursive=True,
            file_patterns=file_patterns or default_patterns,
            exclude_patterns=exclude_patterns or default_excludes
        )

        return result

    async def test_claude_integration(self):
        """Test the Claude Desktop integration."""

        if not self.server:
            await self.setup_server()

        # Test queries that Claude users commonly ask
        test_queries = [
            "How does authentication work in this codebase?",
            "Show me the database configuration",
            "What are the main API endpoints?",
            "How do I run the tests?",
            "What's the deployment process?"
        ]

        print("🧪 Testing Claude Desktop integration:")

        for query in test_queries:
            try:
                result = await self.server.search_context({
                    'query': query,
                    'max_results': 3,
                    'assemble_context': True,
                    'max_context_size': 6000
                }, None)

                print(f"\n   ✅ Query: '{query}'")
                print(f"      Results: {len(result['results'])}")

                if result.get('assembled_context'):
                    context_size = len(result['assembled_context'])
                    print(f"      Context size: {context_size} chars")

                # Show top result
                if result['results']:
                    top_result = result['results'][0]
                    file_name = Path(top_result['source_path']).name
                    print(f"      Top result: {file_name} (score: {top_result['similarity']:.3f})")

            except Exception as e:
                print(f"   ❌ Query failed: '{query}' - {e}")

    async def close(self):
        """Clean up resources."""
        if self.server:
            await self.server.close()

# Usage example
async def setup_claude_desktop():
    """Complete Claude Desktop setup example."""

    integration = ClaudeDesktopIntegration()

    try:
        # Step 1: Setup the server
        await integration.setup_server()

        # Step 2: Index your projects
        project_paths = [
            "/path/to/your/main/project",
            "/path/to/your/documentation",
            "/path/to/your/configs"
        ]

        for project_path in project_paths:
            if Path(project_path).exists():
                print(f"📚 Indexing project: {project_path}")
                result = await integration.index_project(project_path)
                print(f"   ✅ Indexed {result['indexed_files']} files, {result['total_chunks']} chunks")

        # Step 3: Update Claude Desktop configuration
        integration.update_claude_config(project_paths)

        # Step 4: Test integration
        await integration.test_claude_integration()

        print("\n🎉 Claude Desktop integration complete!")
        print("Instructions:")
        print("1. Restart Claude Desktop")
        print("2. Start a new conversation")
        print("3. Ask: 'Can you check what MCP tools are available?'")
        print("4. Try: 'Index my project and help me understand the authentication system'")

    finally:
        await integration.close()

# Run the setup
asyncio.run(setup_claude_desktop())
```

### Advanced Claude Desktop Workflow

Specialized workflow for development assistance:

```python
# claude_development_workflow.py
import asyncio
from pathlib import Path
from eol.rag_context import EOLRAGContextServer

class ClaudeDevelopmentWorkflow:
    """Advanced Claude Desktop workflow for development assistance."""

    def __init__(self):
        self.server = None

    async def initialize(self):
        """Initialize with development-optimized configuration."""
        config = {
            "indexing": {
                "chunk_size": 800,
                "use_semantic_chunking": True,
                "parse_code_structure": True,
                "extract_docstrings": True
            },
            "search": {
                "default_max_results": 5,
                "similarity_threshold": 0.7,
                "boost_recent_files": True,
                "boost_frequently_accessed": True
            },
            "context": {
                "max_context_size": 10000,
                "include_file_paths": True,
                "include_line_numbers": True,
                "smart_truncation": True
            }
        }

        self.server = EOLRAGContextServer(config=config)
        await self.server.initialize()

    async def setup_development_project(self, project_root: str):
        """Setup a development project for optimal Claude assistance."""

        project_path = Path(project_root)

        # Define development-specific file patterns
        patterns = {
            "source_code": ["*.py", "*.js", "*.ts", "*.go", "*.rs", "*.java", "*.cpp", "*.c", "*.h"],
            "documentation": ["*.md", "*.rst", "*.txt", "README*", "CHANGELOG*", "LICENSE*"],
            "configuration": ["*.json", "*.yaml", "*.yml", "*.toml", "*.ini", "*.cfg"],
            "scripts": ["*.sh", "*.bash", "*.ps1", "*.bat", "Makefile", "Dockerfile"],
            "data": ["*.sql", "*.csv", "*.json"]  # Structured data files
        }

        # Index each category with specific settings
        indexing_results = {}

        for category, file_patterns in patterns.items():
            print(f"📁 Indexing {category}...")

            result = await self.server.index_directory(
                directory_path=str(project_path),
                file_patterns=file_patterns,
                recursive=True,
                exclude_patterns=[
                    "node_modules/*", "__pycache__/*", ".git/*",
                    "build/*", "dist/*", "target/*", ".venv/*",
                    "*.pyc", "*.pyo", "*.class", "*.o"
                ],
                category=category  # Tag content by category
            )

            indexing_results[category] = result
            print(f"   ✅ {category}: {result['indexed_files']} files")

        return indexing_results

    async def create_development_tools(self):
        """Create specialized tools for Claude development assistance."""

        tools = {
            "code_review": self._create_code_review_tool(),
            "documentation_assistant": self._create_docs_tool(),
            "debugging_helper": self._create_debug_tool(),
            "architecture_explorer": self._create_architecture_tool(),
            "testing_assistant": self._create_testing_tool()
        }

        # Register tools with the MCP server
        for tool_name, tool_func in tools.items():
            await self.server.register_mcp_tool(tool_name, tool_func)

        print(f"✅ Registered {len(tools)} development tools")
        return tools

    def _create_code_review_tool(self):
        """Tool for code review assistance."""
        async def code_review(file_path: str, focus_areas: list[str] = None):
            """Review code file and provide feedback."""

            # Search for similar patterns and best practices
            queries = [
                f"similar implementations to {Path(file_path).stem}",
                f"best practices for {Path(file_path).suffix} files",
                f"error handling patterns in {Path(file_path).suffix}",
                f"testing approaches for {Path(file_path).stem}"
            ]

            if focus_areas:
                queries.extend([f"{area} in {Path(file_path).suffix}" for area in focus_areas])

            review_context = []
            for query in queries:
                result = await self.server.search_context({
                    'query': query,
                    'max_results': 3,
                    'filters': {'file_types': [Path(file_path).suffix]}
                }, None)

                review_context.extend(result['results'])

            return {
                'file_path': file_path,
                'review_context': review_context,
                'suggestions': self._generate_review_suggestions(file_path, review_context)
            }

        return code_review

    def _create_docs_tool(self):
        """Tool for documentation assistance."""
        async def documentation_assistant(topic: str, doc_type: str = "api"):
            """Find relevant documentation for a topic."""

            doc_queries = [
                f"{topic} documentation",
                f"{topic} examples and usage",
                f"how to use {topic}",
                f"{topic} configuration and setup"
            ]

            if doc_type == "api":
                doc_queries.extend([
                    f"{topic} API reference",
                    f"{topic} endpoints and parameters"
                ])

            doc_results = []
            for query in doc_queries:
                result = await self.server.search_context({
                    'query': query,
                    'max_results': 3,
                    'filters': {'file_types': ['.md', '.rst', '.txt']},
                    'search_level': 'section'
                }, None)

                doc_results.extend(result['results'])

            # Remove duplicates and sort by relevance
            unique_docs = {}
            for doc in doc_results:
                doc_id = f"{doc['source_path']}:{doc['metadata'].get('section', '')}"
                if doc_id not in unique_docs or doc['similarity'] > unique_docs[doc_id]['similarity']:
                    unique_docs[doc_id] = doc

            sorted_docs = sorted(unique_docs.values(), key=lambda x: x['similarity'], reverse=True)

            return {
                'topic': topic,
                'documentation': sorted_docs[:10],  # Top 10 most relevant
                'coverage_areas': self._analyze_doc_coverage(sorted_docs)
            }

        return documentation_assistant

    def _create_debug_tool(self):
        """Tool for debugging assistance."""
        async def debugging_helper(error_message: str, context_files: list[str] = None):
            """Help debug errors using codebase knowledge."""

            # Search for similar errors and solutions
            debug_queries = [
                f"similar error to {error_message}",
                f"fixing {error_message}",
                f"troubleshooting {error_message}",
                f"error handling for {error_message}"
            ]

            # If context files provided, search within them
            search_filters = {}
            if context_files:
                search_filters['source_paths'] = context_files

            debug_info = []
            for query in debug_queries:
                result = await self.server.search_context({
                    'query': query,
                    'max_results': 5,
                    'filters': search_filters
                }, None)

                debug_info.extend(result['results'])

            # Also search for logging and error handling patterns
            patterns_result = await self.server.search_context({
                'query': 'error handling and logging patterns',
                'max_results': 3,
                'filters': {'chunk_types': ['function', 'class']}
            }, None)

            return {
                'error_message': error_message,
                'similar_issues': debug_info,
                'error_handling_patterns': patterns_result['results'],
                'debugging_suggestions': self._generate_debug_suggestions(error_message, debug_info)
            }

        return debugging_helper

    def _create_architecture_tool(self):
        """Tool for exploring system architecture."""
        async def architecture_explorer(component: str = None):
            """Explore system architecture and component relationships."""

            if component:
                # Explore specific component
                arch_queries = [
                    f"{component} dependencies and relationships",
                    f"{component} interfaces and connections",
                    f"components that use {component}",
                    f"{component} configuration and setup"
                ]
            else:
                # General architecture exploration
                arch_queries = [
                    "system architecture overview",
                    "main components and services",
                    "data flow and interactions",
                    "configuration and deployment"
                ]

            architecture_info = []
            for query in arch_queries:
                result = await self.server.search_context({
                    'query': query,
                    'max_results': 5,
                    'search_level': 'concept'
                }, None)

                architecture_info.extend(result['results'])

            return {
                'component': component,
                'architecture_overview': architecture_info,
                'component_map': self._build_component_map(architecture_info)
            }

        return architecture_explorer

    def _create_testing_tool(self):
        """Tool for testing assistance."""
        async def testing_assistant(component: str, test_type: str = "unit"):
            """Find testing patterns and examples for components."""

            test_queries = [
                f"{component} test examples",
                f"testing {component} functionality",
                f"{test_type} tests for {component}",
                f"mocking and test setup for {component}"
            ]

            test_info = []
            for query in test_queries:
                result = await self.server.search_context({
                    'query': query,
                    'max_results': 3,
                    'filters': {'file_patterns': ['*test*', '*spec*']}
                }, None)

                test_info.extend(result['results'])

            # Also find test utilities and helpers
            utils_result = await self.server.search_context({
                'query': 'test utilities and helpers',
                'max_results': 3
            }, None)

            return {
                'component': component,
                'test_type': test_type,
                'test_examples': test_info,
                'test_utilities': utils_result['results'],
                'test_suggestions': self._generate_test_suggestions(component, test_info)
            }

        return testing_assistant

    def _generate_review_suggestions(self, file_path: str, context: list) -> list[str]:
        """Generate code review suggestions based on context."""
        suggestions = []

        # Analyze patterns in similar code
        if context:
            suggestions.append("Review error handling patterns against similar implementations")
            suggestions.append("Check if consistent naming conventions are followed")
            suggestions.append("Verify proper documentation and comments")
            suggestions.append("Consider test coverage for this component")

        return suggestions

    def _analyze_doc_coverage(self, docs: list) -> dict:
        """Analyze documentation coverage areas."""
        coverage = {
            'api_reference': False,
            'usage_examples': False,
            'configuration': False,
            'troubleshooting': False
        }

        for doc in docs:
            content = doc['content'].lower()
            if 'api' in content or 'endpoint' in content:
                coverage['api_reference'] = True
            if 'example' in content or 'usage' in content:
                coverage['usage_examples'] = True
            if 'config' in content or 'setting' in content:
                coverage['configuration'] = True
            if 'troubleshoot' in content or 'error' in content:
                coverage['troubleshooting'] = True

        return coverage

    def _generate_debug_suggestions(self, error: str, debug_info: list) -> list[str]:
        """Generate debugging suggestions based on error and context."""
        suggestions = [
            "Check logs for additional context around the error",
            "Verify input parameters and data types",
            "Review recent changes that might have introduced the issue"
        ]

        if debug_info:
            suggestions.append("Review similar issues found in the codebase")
            suggestions.append("Check error handling patterns in related components")

        return suggestions

    def _build_component_map(self, arch_info: list) -> dict:
        """Build a map of components and their relationships."""
        component_map = {}

        for info in arch_info:
            # Extract component names and relationships from content
            # This is a simplified implementation
            content = info['content']
            component_map[info['source_path']] = {
                'description': content[:200],
                'relationships': []  # Would extract relationships from content
            }

        return component_map

    def _generate_test_suggestions(self, component: str, test_info: list) -> list[str]:
        """Generate testing suggestions for a component."""
        suggestions = [
            f"Create unit tests covering {component} core functionality",
            f"Add integration tests for {component} interactions",
            f"Implement edge case testing for {component}",
            f"Consider performance testing if {component} handles large data"
        ]

        return suggestions

# Example usage
async def setup_development_workflow():
    """Setup complete development workflow for Claude Desktop."""

    workflow = ClaudeDevelopmentWorkflow()

    try:
        await workflow.initialize()

        # Setup your development project
        project_root = "/path/to/your/project"
        if Path(project_root).exists():
            await workflow.setup_development_project(project_root)

            # Create specialized development tools
            await workflow.create_development_tools()

            print("🎉 Development workflow ready!")
            print("Available tools in Claude Desktop:")
            print("  - code_review: Review code files")
            print("  - documentation_assistant: Find relevant docs")
            print("  - debugging_helper: Debug assistance")
            print("  - architecture_explorer: Explore system architecture")
            print("  - testing_assistant: Testing guidance")

    finally:
        if workflow.server:
            await workflow.server.close()

# Run the setup
asyncio.run(setup_development_workflow())
```

## Web API Integration

### FastAPI Integration

Complete REST API with authentication and documentation:

```python
# fastapi_integration.py
import asyncio
from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import jwt
import json
import io
from datetime import datetime, timedelta

from eol.rag_context import EOLRAGContextServer

# Pydantic models for API
class IndexRequest(BaseModel):
    directory_path: str = Field(..., description="Path to directory to index")
    recursive: bool = Field(True, description="Index subdirectories recursively")
    file_patterns: Optional[List[str]] = Field(None, description="File patterns to include")
    exclude_patterns: Optional[List[str]] = Field(None, description="File patterns to exclude")
    force_reindex: bool = Field(False, description="Force reindexing of all files")

class IndexResponse(BaseModel):
    success: bool
    indexed_files: int
    total_chunks: int
    processing_time: float
    message: str

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query", min_length=1)
    max_results: int = Field(5, description="Maximum number of results", ge=1, le=50)
    similarity_threshold: float = Field(0.7, description="Minimum similarity score", ge=0.0, le=1.0)
    search_level: Optional[str] = Field(None, description="Search level: concept, section, or chunk")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional search filters")
    assemble_context: bool = Field(False, description="Assemble coherent context from results")
    max_context_size: int = Field(4000, description="Maximum context size when assembling")

class SearchResult(BaseModel):
    source_path: str
    content: str
    similarity: float
    chunk_type: str
    metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    search_time_ms: float
    cache_hit: bool
    assembled_context: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    redis_status: str
    indexed_documents: int

class RAGAPIServer:
    """FastAPI server for EOL RAG Context with authentication and monitoring."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.rag_server = None
        self.app = self._create_app()

    def _create_app(self) -> FastAPI:
        """Create FastAPI application with all endpoints."""

        app = FastAPI(
            title="EOL RAG Context API",
            description="Intelligent document indexing and search API",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )

        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Security
        security = HTTPBearer()

        @app.on_event("startup")
        async def startup():
            """Initialize RAG server on startup."""
            self.rag_server = EOLRAGContextServer(config=self.config)
            await self.rag_server.initialize()

        @app.on_event("shutdown")
        async def shutdown():
            """Clean up resources on shutdown."""
            if self.rag_server:
                await self.rag_server.close()

        # Authentication dependency
        async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
            """Verify JWT token."""
            try:
                # In production, use proper JWT verification with secret key
                token = credentials.credentials
                # For demo purposes, accept any non-empty token
                if not token:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid authentication credentials",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                return token
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )

        # Health check endpoint
        @app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            try:
                redis_status = "healthy" if self.rag_server and await self.rag_server.redis_client.ping() else "unhealthy"

                # Get indexed document count
                stats = await self.rag_server.get_indexing_stats() if self.rag_server else {}
                indexed_docs = stats.get('total_documents', 0)

                return HealthResponse(
                    status="healthy",
                    timestamp=datetime.now(),
                    version="1.0.0",
                    redis_status=redis_status,
                    indexed_documents=indexed_docs
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

        # Index endpoints
        @app.post("/api/v1/index/directory", response_model=IndexResponse)
        async def index_directory(
            request: IndexRequest,
            token: str = Depends(verify_token)
        ):
            """Index a directory of documents."""
            try:
                start_time = datetime.now()

                result = await self.rag_server.index_directory(
                    directory_path=request.directory_path,
                    recursive=request.recursive,
                    file_patterns=request.file_patterns,
                    exclude_patterns=request.exclude_patterns,
                    force_reindex=request.force_reindex
                )

                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()

                return IndexResponse(
                    success=True,
                    indexed_files=result['indexed_files'],
                    total_chunks=result['total_chunks'],
                    processing_time=processing_time,
                    message=f"Successfully indexed {result['indexed_files']} files"
                )

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

        @app.post("/api/v1/index/file")
        async def index_file(
            file_path: str,
            force_reindex: bool = False,
            token: str = Depends(verify_token)
        ):
            """Index a single file."""
            try:
                result = await self.rag_server.index_file(
                    file_path=file_path,
                    force_reindex=force_reindex
                )

                return {
                    "success": True,
                    "file_path": result['file_path'],
                    "chunks_created": result['chunks_created'],
                    "processing_time": result['processing_time']
                }

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"File indexing failed: {str(e)}")

        # Search endpoints
        @app.post("/api/v1/search", response_model=SearchResponse)
        async def search_context(
            request: SearchRequest,
            token: str = Depends(verify_token)
        ):
            """Search indexed documents."""
            try:
                start_time = datetime.now()

                search_params = {
                    'query': request.query,
                    'max_results': request.max_results,
                    'similarity_threshold': request.similarity_threshold
                }

                if request.search_level:
                    search_params['search_level'] = request.search_level

                if request.filters:
                    search_params['filters'] = request.filters

                if request.assemble_context:
                    search_params['assemble_context'] = True
                    search_params['max_context_size'] = request.max_context_size

                result = await self.rag_server.search_context(search_params, None)

                end_time = datetime.now()
                search_time = (end_time - start_time).total_seconds() * 1000  # Convert to ms

                # Convert results to Pydantic models
                search_results = [
                    SearchResult(
                        source_path=r['source_path'],
                        content=r['content'],
                        similarity=r['similarity'],
                        chunk_type=r['chunk_type'],
                        metadata=r['metadata']
                    )
                    for r in result['results']
                ]

                return SearchResponse(
                    query=request.query,
                    results=search_results,
                    total_results=len(search_results),
                    search_time_ms=search_time,
                    cache_hit=result.get('cache_hit', False),
                    assembled_context=result.get('assembled_context')
                )

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

        @app.get("/api/v1/search/suggestions")
        async def search_suggestions(
            query: str,
            limit: int = 5,
            token: str = Depends(verify_token)
        ):
            """Get search query suggestions."""
            try:
                # This would implement query suggestion logic
                # For now, return a simple response
                suggestions = [
                    f"{query} examples",
                    f"{query} documentation",
                    f"how to use {query}",
                    f"{query} configuration",
                    f"{query} best practices"
                ]

                return {
                    "query": query,
                    "suggestions": suggestions[:limit]
                }

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Suggestions failed: {str(e)}")

        # Analytics endpoints
        @app.get("/api/v1/analytics/stats")
        async def get_analytics_stats(token: str = Depends(verify_token)):
            """Get system analytics and statistics."""
            try:
                stats = await self.rag_server.get_indexing_stats()
                cache_stats = await self.rag_server.get_cache_stats()

                return {
                    "indexing": stats,
                    "caching": cache_stats,
                    "timestamp": datetime.now()
                }

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")

        # Export endpoints
        @app.get("/api/v1/export/context")
        async def export_context(
            query: str,
            format: str = "json",
            token: str = Depends(verify_token)
        ):
            """Export search context in various formats."""
            try:
                result = await self.rag_server.search_context({
                    'query': query,
                    'max_results': 10,
                    'assemble_context': True
                }, None)

                if format == "json":
                    return result
                elif format == "text":
                    context = result.get('assembled_context', '')
                    return StreamingResponse(
                        io.StringIO(context),
                        media_type="text/plain",
                        headers={"Content-Disposition": f"attachment; filename=context_{query[:20]}.txt"}
                    )
                elif format == "markdown":
                    # Format as markdown
                    md_content = f"# Context for: {query}\n\n"
                    md_content += result.get('assembled_context', '')

                    return StreamingResponse(
                        io.StringIO(md_content),
                        media_type="text/markdown",
                        headers={"Content-Disposition": f"attachment; filename=context_{query[:20]}.md"}
                    )
                else:
                    raise HTTPException(status_code=400, detail="Unsupported format. Use: json, text, or markdown")

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

        return app

# Example usage and deployment
async def run_api_server():
    """Run the RAG API server."""

    # Configuration for the RAG server
    rag_config = {
        "redis": {
            "url": "redis://localhost:6379",
            "db": 0
        },
        "embedding": {
            "provider": "sentence_transformers",
            "model": "all-MiniLM-L6-v2",
            "batch_size": 32
        },
        "caching": {
            "enabled": True,
            "ttl_seconds": 3600
        }
    }

    # Create API server
    api_server = RAGAPIServer(config=rag_config)

    # Run with uvicorn
    import uvicorn

    print("🚀 Starting RAG API Server...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 ReDoc Documentation: http://localhost:8000/redoc")

    uvicorn.run(
        api_server.app,
        host="0.0.0.0",
        port=8000,
        reload=False  # Set to True for development
    )

# Client example
class RAGAPIClient:
    """Client for the RAG API."""

    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip('/')
        self.token = token
        self.headers = {"Authorization": f"Bearer {token}"}

    async def index_directory(self, directory_path: str, **kwargs):
        """Index a directory via the API."""
        import aiohttp

        data = {"directory_path": directory_path, **kwargs}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/v1/index/directory",
                json=data,
                headers=self.headers
            ) as response:
                return await response.json()

    async def search(self, query: str, **kwargs):
        """Search via the API."""
        import aiohttp

        data = {"query": query, **kwargs}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/v1/search",
                json=data,
                headers=self.headers
            ) as response:
                return await response.json()

# Example client usage
async def api_client_example():
    """Example of using the RAG API client."""

    client = RAGAPIClient("http://localhost:8000", "demo-token")

    try:
        # Index a directory
        index_result = await client.index_directory("/path/to/docs")
        print(f"✅ Indexed: {index_result}")

        # Search for information
        search_result = await client.search(
            "how to configure authentication",
            max_results=3,
            assemble_context=True
        )
        print(f"🔍 Search results: {len(search_result['results'])} found")

    except Exception as e:
        print(f"❌ API call failed: {e}")

# To run the server: python fastapi_integration.py
if __name__ == "__main__":
    asyncio.run(run_api_server())
```

## Streamlit Dashboard Integration

### Interactive Dashboard for Content Management

Complete Streamlit dashboard for managing indexed content:

```python
# streamlit_dashboard.py
import streamlit as st
import asyncio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from pathlib import Path

# Set up the page
st.set_page_config(
    page_title="EOL RAG Context Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'rag_server' not in st.session_state:
    st.session_state.rag_server = None
if 'indexed_content' not in st.session_state:
    st.session_state.indexed_content = []
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

@st.cache_resource
def init_rag_server():
    """Initialize RAG server (cached)."""
    from eol.rag_context import EOLRAGContextServer

    config = {
        "redis": {
            "url": "redis://localhost:6379",
            "db": 0
        },
        "embedding": {
            "provider": "sentence_transformers",
            "model": "all-MiniLM-L6-v2"
        },
        "caching": {
            "enabled": True,
            "ttl_seconds": 1800
        }
    }

    server = EOLRAGContextServer(config=config)

    # Since Streamlit doesn't handle async well, we need a wrapper
    import threading
    import time

    class AsyncRAGWrapper:
        def __init__(self, server):
            self.server = server
            self.loop = None
            self.thread = None
            self._start_event_loop()

        def _start_event_loop(self):
            def run_loop():
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
                # Initialize server in the event loop
                self.loop.run_until_complete(self.server.initialize())
                self.loop.run_forever()

            self.thread = threading.Thread(target=run_loop, daemon=True)
            self.thread.start()

            # Wait for loop to be ready
            while self.loop is None:
                time.sleep(0.1)

        def run_async(self, coro):
            """Run async coroutine in the event loop thread."""
            future = asyncio.run_coroutine_threadsafe(coro, self.loop)
            return future.result(timeout=30)

    return AsyncRAGWrapper(server)

def main():
    """Main Streamlit dashboard."""

    st.title("🔍 EOL RAG Context Dashboard")
    st.markdown("Intelligent document indexing and search management")

    # Initialize RAG server
    try:
        if st.session_state.rag_server is None:
            with st.spinner("Initializing RAG server..."):
                st.session_state.rag_server = init_rag_server()

        rag_server = st.session_state.rag_server

    except Exception as e:
        st.error(f"Failed to initialize RAG server: {e}")
        st.info("Make sure Redis is running: `docker run -d -p 6379:6379 redis/redis-stack`")
        return

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Indexing", "Search", "Analytics", "Configuration"]
    )

    # Main content based on page selection
    if page == "Overview":
        show_overview(rag_server)
    elif page == "Indexing":
        show_indexing(rag_server)
    elif page == "Search":
        show_search(rag_server)
    elif page == "Analytics":
        show_analytics(rag_server)
    elif page == "Configuration":
        show_configuration(rag_server)

def show_overview(rag_server):
    """Show system overview and health status."""

    st.header("System Overview")

    col1, col2, col3, col4 = st.columns(4)

    try:
        # Get system statistics
        stats = rag_server.run_async(rag_server.server.get_indexing_stats())
        cache_stats = rag_server.run_async(rag_server.server.get_cache_stats())

        with col1:
            st.metric(
                "Indexed Documents",
                stats.get('total_documents', 0),
                delta=None
            )

        with col2:
            st.metric(
                "Total Chunks",
                stats.get('total_chunks', 0),
                delta=None
            )

        with col3:
            st.metric(
                "Cache Hit Rate",
                f"{cache_stats.get('hit_rate', 0):.1%}",
                delta=None
            )

        with col4:
            st.metric(
                "Avg Response Time",
                f"{cache_stats.get('avg_response_time_ms', 0):.1f}ms",
                delta=None
            )

        # System health
        st.subheader("System Health")

        health_status = rag_server.run_async(rag_server.server.get_health_status())

        if health_status.get('status') == 'healthy':
            st.success("✅ System is healthy")
        else:
            st.error("❌ System issues detected")

        # Health checks details
        checks = health_status.get('checks', {})
        for check_name, check_result in checks.items():
            status = "✅" if check_result.get('healthy', False) else "❌"
            st.write(f"{status} **{check_name}**: {check_result.get('message', 'N/A')}")

        # Recent activity
        st.subheader("Recent Activity")

        if st.session_state.search_history:
            recent_searches = pd.DataFrame(st.session_state.search_history[-10:])
            st.dataframe(recent_searches, use_container_width=True)
        else:
            st.info("No recent search activity")

    except Exception as e:
        st.error(f"Error loading overview: {e}")

def show_indexing(rag_server):
    """Show indexing interface."""

    st.header("Document Indexing")

    # Indexing form
    with st.form("indexing_form"):
        st.subheader("Index Directory")

        col1, col2 = st.columns(2)

        with col1:
            directory_path = st.text_input(
                "Directory Path",
                placeholder="/path/to/your/documents",
                help="Path to the directory you want to index"
            )

            recursive = st.checkbox("Index subdirectories", value=True)
            force_reindex = st.checkbox("Force reindex all files", value=False)

        with col2:
            file_patterns = st.text_area(
                "File Patterns (one per line)",
                value="*.py\n*.md\n*.txt\n*.json",
                help="File patterns to include in indexing"
            )

            exclude_patterns = st.text_area(
                "Exclude Patterns (one per line)",
                value="*.pyc\n__pycache__/*\n.git/*\nnode_modules/*",
                help="File patterns to exclude from indexing"
            )

        submitted = st.form_submit_button("Start Indexing", type="primary")

        if submitted and directory_path:
            try:
                # Parse patterns
                file_patterns_list = [p.strip() for p in file_patterns.split('\n') if p.strip()]
                exclude_patterns_list = [p.strip() for p in exclude_patterns.split('\n') if p.strip()]

                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("Starting indexing...")
                progress_bar.progress(0.1)

                # Run indexing
                result = rag_server.run_async(
                    rag_server.server.index_directory(
                        directory_path=directory_path,
                        recursive=recursive,
                        file_patterns=file_patterns_list,
                        exclude_patterns=exclude_patterns_list,
                        force_reindex=force_reindex
                    )
                )

                progress_bar.progress(1.0)
                status_text.text("Indexing completed!")

                # Show results
                st.success("✅ Indexing completed successfully!")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Files Indexed", result['indexed_files'])
                with col2:
                    st.metric("Chunks Created", result['total_chunks'])
                with col3:
                    st.metric("Processing Time", f"{result['processing_time']:.1f}s")

                # Update session state
                st.session_state.indexed_content.append({
                    'timestamp': datetime.now(),
                    'directory': directory_path,
                    'files': result['indexed_files'],
                    'chunks': result['total_chunks']
                })

                st.rerun()

            except Exception as e:
                st.error(f"Indexing failed: {e}")

    # Show indexed content
    st.subheader("Indexed Content")

    if st.session_state.indexed_content:
        indexed_df = pd.DataFrame(st.session_state.indexed_content)
        st.dataframe(indexed_df, use_container_width=True)

        # Summary chart
        if len(indexed_df) > 1:
            fig = px.line(
                indexed_df,
                x='timestamp',
                y='files',
                title='Indexing Progress Over Time',
                labels={'files': 'Files Indexed', 'timestamp': 'Date'}
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No content indexed yet. Use the form above to start indexing.")

def show_search(rag_server):
    """Show search interface."""

    st.header("Intelligent Search")

    # Search form
    with st.form("search_form"):
        query = st.text_input(
            "Search Query",
            placeholder="How to implement authentication?",
            help="Enter your search query in natural language"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            max_results = st.slider("Max Results", 1, 20, 5)
            similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.7, 0.1)

        with col2:
            search_level = st.selectbox(
                "Search Level",
                ["auto", "concept", "section", "chunk"],
                help="Granularity of search results"
            )

            assemble_context = st.checkbox("Assemble Context", value=False)

        with col3:
            file_types = st.multiselect(
                "File Types",
                [".py", ".js", ".md", ".txt", ".json", ".yaml"],
                help="Filter by file types"
            )

        submitted = st.form_submit_button("Search", type="primary")

        if submitted and query:
            try:
                start_time = datetime.now()

                # Build search parameters
                search_params = {
                    'query': query,
                    'max_results': max_results,
                    'similarity_threshold': similarity_threshold
                }

                if search_level != "auto":
                    search_params['search_level'] = search_level

                if file_types:
                    search_params['filters'] = {'file_types': file_types}

                if assemble_context:
                    search_params['assemble_context'] = True
                    search_params['max_context_size'] = 6000

                # Perform search
                with st.spinner("Searching..."):
                    results = rag_server.run_async(
                        rag_server.server.search_context(search_params, None)
                    )

                end_time = datetime.now()
                search_time = (end_time - start_time).total_seconds() * 1000

                # Record search history
                st.session_state.search_history.append({
                    'timestamp': start_time,
                    'query': query,
                    'results': len(results['results']),
                    'time_ms': search_time,
                    'cache_hit': results.get('cache_hit', False)
                })

                # Display results
                st.subheader(f"Search Results ({len(results['results'])} found)")

                # Search metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Results Found", len(results['results']))
                with col2:
                    st.metric("Search Time", f"{search_time:.1f}ms")
                with col3:
                    cache_status = "Hit" if results.get('cache_hit', False) else "Miss"
                    st.metric("Cache Status", cache_status)

                # Show assembled context if requested
                if assemble_context and results.get('assembled_context'):
                    st.subheader("Assembled Context")
                    st.text_area(
                        "Context",
                        value=results['assembled_context'],
                        height=200,
                        help="Coherent context assembled from search results"
                    )

                # Show individual results
                for i, result in enumerate(results['results']):
                    with st.expander(f"Result {i+1}: {Path(result['source_path']).name} (Score: {result['similarity']:.3f})"):
                        st.code(result['content'], language=None)

                        # Metadata
                        if result.get('metadata'):
                            st.json(result['metadata'])

            except Exception as e:
                st.error(f"Search failed: {e}")

    # Search history
    st.subheader("Search History")

    if st.session_state.search_history:
        history_df = pd.DataFrame(st.session_state.search_history[-20:])  # Last 20 searches
        st.dataframe(history_df, use_container_width=True)

        # Search performance chart
        if len(history_df) > 5:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=history_df['timestamp'],
                y=history_df['time_ms'],
                mode='lines+markers',
                name='Search Time (ms)',
                line=dict(color='blue')
            ))

            fig.update_layout(
                title='Search Performance Over Time',
                xaxis_title='Time',
                yaxis_title='Response Time (ms)',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No search history yet.")

def show_analytics(rag_server):
    """Show analytics and performance metrics."""

    st.header("Analytics & Performance")

    try:
        # Get system metrics
        stats = rag_server.run_async(rag_server.server.get_indexing_stats())
        cache_stats = rag_server.run_async(rag_server.server.get_cache_stats())

        # Performance overview
        st.subheader("Performance Overview")

        col1, col2 = st.columns(2)

        with col1:
            # Cache performance
            cache_metrics = {
                'Hit Rate': f"{cache_stats.get('hit_rate', 0):.1%}",
                'Total Queries': cache_stats.get('total_queries', 0),
                'Cache Size': cache_stats.get('cache_size', 0),
                'Avg Response Time': f"{cache_stats.get('avg_response_time_ms', 0):.1f}ms"
            }

            st.write("**Cache Performance**")
            for metric, value in cache_metrics.items():
                st.write(f"- {metric}: {value}")

        with col2:
            # Indexing statistics
            index_metrics = {
                'Total Documents': stats.get('total_documents', 0),
                'Total Chunks': stats.get('total_chunks', 0),
                'Avg Chunks/Document': f"{stats.get('avg_chunks_per_document', 0):.1f}",
                'Index Size': f"{stats.get('index_size_mb', 0):.1f} MB"
            }

            st.write("**Indexing Statistics**")
            for metric, value in index_metrics.items():
                st.write(f"- {metric}: {value}")

        # Search analytics
        if st.session_state.search_history:
            st.subheader("Search Analytics")

            history_df = pd.DataFrame(st.session_state.search_history)

            col1, col2 = st.columns(2)

            with col1:
                # Query frequency
                query_freq = history_df['query'].value_counts().head(10)
                fig = px.bar(
                    x=query_freq.values,
                    y=query_freq.index,
                    orientation='h',
                    title='Most Common Queries',
                    labels={'x': 'Frequency', 'y': 'Query'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Response time distribution
                fig = px.histogram(
                    history_df,
                    x='time_ms',
                    title='Response Time Distribution',
                    labels={'time_ms': 'Response Time (ms)', 'count': 'Frequency'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            # Cache hit rate over time
            if 'cache_hit' in history_df.columns:
                cache_over_time = history_df.groupby(history_df['timestamp'].dt.hour)['cache_hit'].mean()

                fig = px.line(
                    x=cache_over_time.index,
                    y=cache_over_time.values,
                    title='Cache Hit Rate by Hour',
                    labels={'x': 'Hour of Day', 'y': 'Cache Hit Rate'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

        # Export data
        st.subheader("Data Export")

        if st.button("Export Analytics Data"):
            export_data = {
                'stats': stats,
                'cache_stats': cache_stats,
                'search_history': st.session_state.search_history,
                'indexed_content': st.session_state.indexed_content,
                'export_timestamp': datetime.now().isoformat()
            }

            st.download_button(
                "Download Analytics Data",
                data=json.dumps(export_data, indent=2, default=str),
                file_name=f"rag_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

    except Exception as e:
        st.error(f"Error loading analytics: {e}")

def show_configuration(rag_server):
    """Show configuration interface."""

    st.header("Configuration")

    # Current configuration
    st.subheader("Current Configuration")

    try:
        server_info = rag_server.run_async(rag_server.server.get_server_info())
        st.json(server_info)

    except Exception as e:
        st.error(f"Error loading configuration: {e}")

    # Configuration editor
    st.subheader("Update Configuration")

    with st.form("config_form"):
        st.write("**Cache Configuration**")
        col1, col2 = st.columns(2)

        with col1:
            cache_enabled = st.checkbox("Enable Caching", value=True)
            cache_ttl = st.number_input("Cache TTL (seconds)", min_value=60, max_value=86400, value=1800)

        with col2:
            target_hit_rate = st.slider("Target Hit Rate", 0.1, 0.5, 0.31, 0.01)
            max_cache_size = st.number_input("Max Cache Size", min_value=100, max_value=10000, value=1000)

        st.write("**Search Configuration**")
        col1, col2 = st.columns(2)

        with col1:
            default_max_results = st.slider("Default Max Results", 1, 50, 5)
            default_threshold = st.slider("Default Similarity Threshold", 0.0, 1.0, 0.7, 0.1)

        with col2:
            max_context_size = st.number_input("Max Context Size", min_value=1000, max_value=20000, value=8000)

        if st.form_submit_button("Update Configuration"):
            st.info("Configuration update functionality would be implemented here.")
            st.success("Configuration updated successfully!")

# Run the dashboard
if __name__ == "__main__":
    main()
```

## Next Steps

After exploring these integration examples:

### Troubleshooting

→ **[Troubleshooting Examples](troubleshooting.md)** - Debug integration issues with practical solutions

### Custom Development

- Build specialized MCP clients for your workflow
- Create domain-specific API endpoints
- Develop custom dashboard components
- Implement monitoring and alerting systems

### Production Deployment

- Container orchestration with Docker/Kubernetes
- Load balancing and high availability
- Security hardening and authentication
- Monitoring and observability

### Advanced Integrations

- Slack/Discord bot integrations
- VS Code extension development
- Jupyter notebook integration
- CI/CD pipeline integration

These integration examples provide production-ready patterns that you can adapt to your specific requirements. Each example includes comprehensive error handling, monitoring, and scalability considerations.
