# EOL System Architecture

## Overview
EOL is an AI framework for building modern LLM applications with a two-phase development model supporting rapid prototyping and progressive implementation.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        EOL CLI                              │
│                   (User Interface Layer)                    │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│                    EOL Core Engine                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ .eol Parser  │  │Phase Manager │  │Context Manager│     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│                   Execution Layer                           │
│  ┌──────────────────────┐  ┌──────────────────────┐       │
│  │ Prototyping Engine    │  │ Implementation Engine│       │
│  │ (LLM-based)          │  │ (Deterministic)      │       │
│  └──────────────────────┘  └──────────────────────┘       │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│                    Service Layer                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │ MCP Services│  │Redis Vector │  │Code Generator│       │
│  │ (FastMCP)   │  │   (RedisVL) │  │             │       │
│  └─────────────┘  └─────────────┘  └─────────────┘       │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│                 Infrastructure Layer                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │   Redis v8  │  │   Docker    │  │External APIs│       │
│  │             │  │  Containers │  │             │       │
│  └─────────────┘  └─────────────┘  └─────────────┘       │
└──────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. User Interface Layer

#### EOL CLI
- **Purpose**: Primary interface for developers
- **Technologies**: Python, Typer, Rich
- **Responsibilities**:
  - Command parsing and execution
  - Feature file management
  - Phase switching
  - Progress visualization

```python
# CLI command structure
eol run <feature.eol> [--phase prototyping|implementation|hybrid]
eol test <feature.test.eol> [--coverage]
eol generate <feature.eol> --output src/
eol switch <feature.eol> --to implementation
```

### 2. Core Engine

#### .eol Parser
- **Purpose**: Parse and validate .eol files
- **Technologies**: Python, YAML, Markdown
- **Responsibilities**:
  - Frontmatter extraction
  - Markdown section parsing
  - Code block extraction
  - Reference resolution

#### Phase Manager
- **Purpose**: Control execution phase transitions
- **Technologies**: Python state machine
- **Responsibilities**:
  - Phase detection
  - Runtime switching
  - Hybrid mode orchestration
  - Performance tracking

#### Context Manager
- **Purpose**: Manage LLM context window
- **Technologies**: RedisVL, vector embeddings
- **Responsibilities**:
  - Context selection
  - Auto-compression
  - Memory management
  - Reference tracking

### 3. Execution Layer

#### Prototyping Engine
- **Purpose**: Execute natural language specifications
- **Technologies**: LLM APIs, MCP
- **Components**:
  ```python
  class PrototypingEngine:
      def __init__(self):
          self.llm_client = LLMClient()
          self.mcp_client = MCPClient()
          self.redis_mcp = RedisMCPServer()
      
      async def execute(self, spec: str):
          # Natural language to action
          context = await self.context_manager.get_context()
          response = await self.llm_client.process(spec, context)
          
          # Execute via MCP
          if response.needs_data_operation:
              result = await self.redis_mcp.execute(response.operation)
          
          return result
  ```

#### Implementation Engine
- **Purpose**: Execute deterministic code
- **Technologies**: Python runtime, sandboxing
- **Components**:
  ```python
  class ImplementationEngine:
      def __init__(self):
          self.runtime = PythonRuntime()
          self.validator = CodeValidator()
          self.monitor = PerformanceMonitor()
      
      async def execute(self, code: str):
          # Validate code
          self.validator.check(code)
          
          # Execute in sandbox
          result = await self.runtime.execute_sandboxed(code)
          
          # Track metrics
          self.monitor.record(result.metrics)
          
          return result
  ```

### 4. Service Layer

#### MCP Services (FastMCP)
- **Purpose**: Provide Model Context Protocol interface
- **Implementation**:
  ```python
  from fastmcp import FastMCP
  
  app = FastMCP("eol-mcp-server")
  
  @app.resource()
  async def get_feature_context(feature_id: str):
      """Expose feature context to LLMs"""
      return await redis.get_context(feature_id)
  
  @app.tool()
  async def execute_operation(operation: str):
      """Execute feature operation"""
      return await operation_manager.execute(operation)
  ```

#### Redis Vector Service (RedisVL)
- **Purpose**: Vector storage and similarity search
- **Implementation**:
  ```python
  from redisvl import SearchIndex, EmbeddingsCache
  
  class VectorService:
      def __init__(self):
          self.index = SearchIndex.from_yaml("config/vector_index.yaml")
          self.cache = EmbeddingsCache(ttl=3600)
      
      async def store_embedding(self, text: str, metadata: dict):
          embedding = await self.embed(text)
          await self.index.add(embedding, metadata)
      
      async def search(self, query: str, k: int = 10):
          query_embedding = await self.embed(query)
          return await self.index.search(query_embedding, k)
  ```

#### Code Generator
- **Purpose**: Generate implementation from prototypes
- **Technologies**: LLM, AST, templates
- **Implementation**:
  ```python
  class CodeGenerator:
      def __init__(self):
          self.llm = LLMClient()
          self.templates = TemplateEngine()
          self.ast_builder = ASTBuilder()
      
      async def generate(self, prototype: str, context: dict):
          # Generate code via LLM
          code = await self.llm.generate_code(prototype, context)
          
          # Parse and validate AST
          ast_tree = self.ast_builder.parse(code)
          
          # Apply templates
          final_code = self.templates.apply(ast_tree)
          
          return final_code
  ```

### 5. Infrastructure Layer

#### Redis v8 Configuration
```yaml
# redis-stack.yaml
redis:
  image: redis/redis-stack:latest
  modules:
    - RediSearch
    - RedisJSON
    - RedisGraph
    - RedisTimeSeries
  
  vector_config:
    index_type: HNSW
    dim: 1536
    distance_metric: COSINE
    initial_cap: 10000
    m: 16
    ef_construction: 200
```

#### Docker Services
```yaml
# docker-compose.yml
version: '3.8'

services:
  redis:
    image: redis/redis-stack:latest
    ports:
      - "6379:6379"
      - "8001:8001"  # RedisInsight
    volumes:
      - redis_data:/data
    
  redis-mcp:
    build: ./services/redis-mcp
    environment:
      REDIS_URL: redis://redis:6379
    depends_on:
      - redis
    
  eol-api:
    build: ./services/api
    ports:
      - "8000:8000"
    environment:
      REDIS_URL: redis://redis:6379
      MCP_SERVER_URL: http://redis-mcp:8080
    depends_on:
      - redis
      - redis-mcp
    
  eol-worker:
    build: ./services/worker
    environment:
      REDIS_URL: redis://redis:6379
    depends_on:
      - redis
    scale: 3  # Multiple workers

volumes:
  redis_data:
```

## Data Flow

### Prototyping Phase Flow
```
1. User writes .eol file with natural language
2. CLI parses file and extracts specifications
3. Phase Manager routes to Prototyping Engine
4. Context Manager retrieves relevant context
5. LLM processes specification with context
6. MCP services execute operations (e.g., Redis)
7. Results returned to user
```

### Implementation Phase Flow
```
1. User writes/generates Python code in .eol
2. CLI parses and validates code
3. Phase Manager routes to Implementation Engine
4. Code executed in sandboxed environment
5. Direct Redis operations via RedisVL
6. Performance metrics collected
7. Results returned to user
```

### Hybrid Phase Flow
```
1. Operations marked with specific phase
2. Phase Manager orchestrates execution
3. Prototyping operations → LLM + MCP
4. Implementation operations → Direct execution
5. Results combined and returned
```

## Monorepo Structure

```
eol/
├── .claude/                      # Claude context management
│   ├── context/                  # Architecture documentation
│   └── commands/                 # Custom AI commands
│
├── packages/                     # Core packages (uv workspace)
│   ├── eol-core/                # Core engine
│   │   ├── src/
│   │   │   ├── parser/          # .eol file parser
│   │   │   ├── phase/           # Phase management
│   │   │   └── context/         # Context management
│   │   ├── tests/
│   │   └── pyproject.toml
│   │
│   ├── eol-proto/               # Prototyping engine
│   │   ├── src/
│   │   │   ├── llm/             # LLM interfaces
│   │   │   ├── mcp/             # MCP client
│   │   │   └── executor/        # Natural language executor
│   │   └── pyproject.toml
│   │
│   ├── eol-impl/                # Implementation engine
│   │   ├── src/
│   │   │   ├── runtime/         # Python runtime
│   │   │   ├── generator/       # Code generation
│   │   │   └── validator/       # Code validation
│   │   └── pyproject.toml
│   │
│   ├── eol-redis/               # Redis integration
│   │   ├── src/
│   │   │   ├── vector/          # Vector operations
│   │   │   ├── cache/           # Semantic caching
│   │   │   └── rag/             # RAG implementation
│   │   └── pyproject.toml
│   │
│   ├── eol-mcp/                 # MCP services
│   │   ├── src/
│   │   │   ├── server/          # FastMCP server
│   │   │   ├── resources/       # MCP resources
│   │   │   └── tools/           # MCP tools
│   │   └── pyproject.toml
│   │
│   └── eol-cli/                 # CLI interface
│       ├── src/
│       │   ├── commands/        # CLI commands
│       │   ├── config/          # Configuration
│       │   └── ui/              # Terminal UI
│       └── pyproject.toml
│
├── services/                     # Microservices
│   ├── redis-mcp/               # Redis MCP server
│   ├── api/                     # REST API
│   └── worker/                  # Background workers
│
├── features/                     # .eol specifications
│   ├── examples/
│   ├── prototypes/
│   └── implementations/
│
├── runtime/                      # Runtime environments
│   ├── docker/
│   └── k8s/
│
├── tests/                        # Test suites
├── scripts/                      # Development scripts
└── docs/                         # Additional documentation
```

## Deployment Architecture

### Development Environment
```yaml
# Local development with hot reload
services:
  eol-dev:
    build: 
      context: .
      dockerfile: runtime/docker/dev.Dockerfile
    volumes:
      - .:/workspace
      - ~/.cache/uv:/root/.cache/uv
    environment:
      - DEVELOPMENT=true
      - AUTO_RELOAD=true
    command: uv run eol dev --watch
```

### Production Environment
```yaml
# Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: eol-api
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: eol-api
        image: eol/api:latest
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
```

## Security Architecture

### Sandboxing
```python
class SecuritySandbox:
    def __init__(self):
        self.allowed_imports = [
            'redis', 'json', 'datetime', 
            'typing', 'asyncio'
        ]
        self.resource_limits = {
            'memory': '512MB',
            'cpu_time': 30,  # seconds
            'network': 'restricted'
        }
    
    def execute_safe(self, code: str):
        # Validate imports
        # Apply resource limits
        # Execute in isolated environment
        pass
```

### Authentication & Authorization
```python
class AuthManager:
    def __init__(self):
        self.redis = Redis()
    
    async def authenticate(self, token: str):
        # Validate token
        # Check permissions
        # Return user context
        pass
    
    async def authorize(self, user, resource, action):
        # Check RBAC rules
        # Audit log
        pass
```

## Performance Optimization

### Caching Strategy
1. **Semantic Cache**: 31% hit rate for similar queries
2. **Result Cache**: Deterministic function results
3. **Context Cache**: Frequently accessed context
4. **Embedding Cache**: Pre-computed embeddings

### Scaling Strategy
1. **Horizontal Scaling**: Multiple worker instances
2. **Load Balancing**: Round-robin for API requests
3. **Queue Management**: Redis Streams for async tasks
4. **Connection Pooling**: Reuse Redis connections

## Monitoring & Observability

### Metrics Collection
```python
class MetricsCollector:
    def __init__(self):
        self.prometheus = PrometheusClient()
    
    def track_execution(self, operation, phase, duration):
        self.prometheus.histogram(
            'eol_execution_duration',
            duration,
            labels={'operation': operation, 'phase': phase}
        )
    
    def track_cache_hit(self, cache_type, hit):
        self.prometheus.counter(
            'eol_cache_hits',
            labels={'type': cache_type, 'hit': hit}
        )
```

### Logging Strategy
```python
import structlog

logger = structlog.get_logger()

logger.info(
    "operation_executed",
    operation="create_session",
    phase="prototyping",
    duration_ms=45,
    user_id="user-123"
)
```

## Integration Points

### LLM Providers
- OpenAI API
- Anthropic Claude API
- Local models (Ollama)
- Custom model endpoints

### External Services
- GitHub for version control
- S3 for artifact storage
- Webhook endpoints
- Custom APIs

## Development Workflow

### Feature Development
1. Write `.eol` specification
2. Write `.test.eol` tests
3. Run in prototyping mode
4. Validate with tests
5. Generate implementation
6. Switch to implementation mode
7. Deploy to production

### Testing Strategy
```bash
# Unit tests
uv run pytest packages/*/tests/

# Integration tests
uv run pytest tests/integration/

# E2E tests
uv run pytest tests/e2e/

# Coverage report
uv run pytest --cov=packages --cov-report=html
```

## Best Practices

### Architecture Principles
1. **Separation of Concerns**: Clear layer boundaries
2. **Dependency Injection**: Configurable components
3. **Interface Segregation**: Minimal interfaces
4. **Single Responsibility**: Focused modules

### Code Organization
1. **Package Independence**: Minimal inter-package dependencies
2. **Shared Types**: Common type definitions
3. **Configuration Management**: Environment-based config
4. **Error Handling**: Consistent error patterns

### Performance Guidelines
1. **Async by Default**: Use async/await
2. **Connection Pooling**: Reuse connections
3. **Batch Operations**: Group similar operations
4. **Lazy Loading**: Load resources on demand

## Future Enhancements

### Planned Features
1. **Multi-language Support**: JavaScript, Go implementations
2. **Distributed Execution**: Cross-node orchestration
3. **Advanced RAG**: GraphRAG integration
4. **Real-time Collaboration**: Multi-user editing
5. **Visual Designer**: GUI for .eol files

### Research Areas
1. **Autonomous Agents**: Self-improving features
2. **Federated Learning**: Distributed model training
3. **Quantum Computing**: Quantum algorithm integration
4. **Neural Architecture Search**: Auto-optimize models