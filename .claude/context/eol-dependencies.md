# EOL Dependency System

## Overview
EOL's dependency system enables system composition through declarative dependency management across multiple layers: features, MCP servers, services, packages, containers, and LLM models.

## Dependency Types

### 1. Feature Dependencies
Dependencies on other `.eol.md` files for feature composition.

```yaml
dependencies:
  features:
    - path: auth/authentication.eol.md
      version: "^2.0.0"
      phase: implementation
      inject:
        - authenticate_user
        - validate_token
    
    - path: core/rate-limiting.eol.md
      version: "~1.5.0"
      phase: all
      config:
        max_requests: 100
        window_seconds: 60
```

#### Feature Composition Pattern
```python
class FeatureComposer:
    """Compose features through dependency injection"""
    
    def __init__(self):
        self.features = {}
        self.dependency_graph = {}
    
    async def compose(self, feature_path: str):
        """Compose feature with its dependencies"""
        
        # Parse feature file
        feature = await self.parse_feature(feature_path)
        
        # Resolve dependencies
        deps = await self.resolve_dependencies(feature.dependencies)
        
        # Inject dependencies
        composed = await self.inject_dependencies(feature, deps)
        
        return composed
    
    async def resolve_dependencies(self, deps: Dict):
        """Resolve feature dependencies recursively"""
        
        resolved = {}
        
        for dep_type, dep_list in deps.items():
            if dep_type == 'features':
                for feature_dep in dep_list:
                    # Load dependent feature
                    dep_feature = await self.load_feature(feature_dep['path'])
                    
                    # Check version compatibility
                    if not self.check_version(dep_feature.version, feature_dep['version']):
                        raise VersionError(f"Incompatible version for {feature_dep['path']}")
                    
                    # Recursively resolve its dependencies
                    dep_feature.dependencies = await self.resolve_dependencies(
                        dep_feature.dependencies
                    )
                    
                    resolved[feature_dep['path']] = dep_feature
        
        return resolved
```

### 2. MCP Server Dependencies
Model Context Protocol servers for LLM-based operations.

```yaml
dependencies:
  mcp_servers:
    - name: redis-mcp
      version: ">=1.0.0"
      transport: stdio
      config:
        url: ${REDIS_URL:-redis://localhost:6379}
        max_connections: 10
      health_check:
        endpoint: /health
        interval: 30
      fallback: redis-direct
      required: true
    
    - name: postgres-mcp
      version: "2.x"
      transport: sse
      endpoint: http://localhost:8001
      auth:
        type: api_key
        key: ${POSTGRES_MCP_KEY}
      required: false
```

#### MCP Integration
```python
class MCPDependencyManager:
    """Manage MCP server dependencies"""
    
    def __init__(self):
        self.servers = {}
        self.health_checker = HealthChecker()
    
    async def connect_mcp_servers(self, dependencies):
        """Connect to required MCP servers"""
        
        for server_dep in dependencies.get('mcp_servers', []):
            try:
                # Create MCP client
                client = await self.create_mcp_client(
                    name=server_dep['name'],
                    version=server_dep['version'],
                    config=server_dep.get('config', {})
                )
                
                # Test connection
                if await self.health_checker.check(client):
                    self.servers[server_dep['name']] = client
                else:
                    # Try fallback
                    if 'fallback' in server_dep:
                        client = await self.connect_fallback(server_dep['fallback'])
                        self.servers[server_dep['name']] = client
                    elif server_dep.get('required', True):
                        raise ConnectionError(f"Failed to connect to {server_dep['name']}")
            
            except Exception as e:
                if server_dep.get('required', True):
                    raise
                logger.warning(f"Optional MCP server {server_dep['name']} unavailable: {e}")
    
    async def execute_with_mcp(self, server_name: str, operation: str, params: dict):
        """Execute operation via MCP server"""
        
        if server_name not in self.servers:
            raise ValueError(f"MCP server {server_name} not available")
        
        client = self.servers[server_name]
        
        # Use circuit breaker pattern
        async with CircuitBreaker(server_name):
            result = await client.execute(operation, params)
        
        return result
```

### 3. Service Dependencies
External HTTP APIs and microservices.

```yaml
dependencies:
  services:
    - name: stripe-api
      type: rest
      url: ${STRIPE_API_URL:-https://api.stripe.com}
      version: "2024-01-01"
      auth:
        type: bearer
        token: ${STRIPE_API_KEY}
      retry:
        max_attempts: 3
        backoff: exponential
      circuit_breaker:
        failure_threshold: 5
        timeout: 60
      phase: implementation
    
    - name: internal-auth-service
      type: grpc
      endpoint: auth-service:50051
      proto: protos/auth.proto
      phase: implementation
```

#### Service Integration
```python
class ServiceDependencyManager:
    """Manage external service dependencies"""
    
    def __init__(self):
        self.services = {}
        self.circuit_breakers = {}
    
    async def initialize_services(self, dependencies):
        """Initialize service connections"""
        
        for service_dep in dependencies.get('services', []):
            if service_dep['type'] == 'rest':
                client = await self.create_rest_client(service_dep)
            elif service_dep['type'] == 'grpc':
                client = await self.create_grpc_client(service_dep)
            elif service_dep['type'] == 'graphql':
                client = await self.create_graphql_client(service_dep)
            
            # Add circuit breaker
            if 'circuit_breaker' in service_dep:
                self.circuit_breakers[service_dep['name']] = CircuitBreaker(
                    **service_dep['circuit_breaker']
                )
            
            self.services[service_dep['name']] = client
    
    async def call_service(self, name: str, method: str, **kwargs):
        """Call external service with resilience patterns"""
        
        client = self.services.get(name)
        if not client:
            raise ValueError(f"Service {name} not initialized")
        
        # Apply circuit breaker if configured
        if name in self.circuit_breakers:
            async with self.circuit_breakers[name]:
                return await client.call(method, **kwargs)
        else:
            return await client.call(method, **kwargs)
```

### 4. Package Dependencies
Python packages from PyPI or private repositories.

```yaml
dependencies:
  packages:
    - name: redis[vector]
      version: ">=5.0.0"
      phase: all
      extras: [vector, search]
    
    - name: fastapi
      version: ">=0.100.0,<1.0.0"
      phase: implementation
      optional: false
    
    - name: internal-sdk
      version: "^2.0.0"
      source:
        type: git
        url: git@github.com:company/internal-sdk.git
      phase: implementation
```

#### Package Management
```python
class PackageDependencyManager:
    """Manage Python package dependencies"""
    
    def __init__(self):
        self.installed = {}
        self.uv = UVPackageManager()
    
    async def install_packages(self, dependencies):
        """Install required packages"""
        
        for package in dependencies.get('packages', []):
            # Check phase compatibility
            if not self.is_phase_compatible(package.get('phase', 'all')):
                continue
            
            # Install package
            if 'source' in package:
                await self.install_from_source(package)
            else:
                await self.uv.install(
                    name=package['name'],
                    version=package['version'],
                    extras=package.get('extras', [])
                )
            
            self.installed[package['name']] = package['version']
    
    def generate_requirements(self, phase: str):
        """Generate requirements.txt for specific phase"""
        
        requirements = []
        for package in self.dependencies.get('packages', []):
            if self.is_phase_compatible(package.get('phase', 'all'), phase):
                requirements.append(f"{package['name']}{package['version']}")
        
        return '\n'.join(requirements)
```

### 5. Container Dependencies
Docker containers for services and databases.

```yaml
dependencies:
  containers:
    - name: redis
      image: redis/redis-stack:latest
      ports:
        - "6379:6379"
        - "8001:8001"  # RedisInsight
      volumes:
        - redis_data:/data
      environment:
        REDIS_PASSWORD: ${REDIS_PASSWORD}
      health_check:
        test: ["CMD", "redis-cli", "ping"]
        interval: 5s
        retries: 5
      phase: all
    
    - name: postgres
      image: postgres:15-alpine
      ports:
        - "5432:5432"
      environment:
        POSTGRES_PASSWORD: ${DB_PASSWORD}
        POSTGRES_DB: ${DB_NAME:-eol}
      phase: implementation
```

#### Container Orchestration
```python
class ContainerDependencyManager:
    """Manage Docker container dependencies"""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.containers = {}
    
    async def start_containers(self, dependencies):
        """Start required containers"""
        
        for container_dep in dependencies.get('containers', []):
            # Check if container already running
            existing = self.find_container(container_dep['name'])
            
            if existing and existing.status == 'running':
                self.containers[container_dep['name']] = existing
                continue
            
            # Start new container
            container = self.docker_client.containers.run(
                image=container_dep['image'],
                name=f"eol_{container_dep['name']}",
                ports=self.parse_ports(container_dep.get('ports', [])),
                environment=container_dep.get('environment', {}),
                volumes=container_dep.get('volumes', []),
                detach=True,
                auto_remove=False
            )
            
            # Wait for health check
            if 'health_check' in container_dep:
                await self.wait_for_healthy(container, container_dep['health_check'])
            
            self.containers[container_dep['name']] = container
    
    async def stop_containers(self):
        """Stop all managed containers"""
        
        for name, container in self.containers.items():
            try:
                container.stop(timeout=10)
                container.remove()
            except Exception as e:
                logger.warning(f"Failed to stop container {name}: {e}")
```

### 6. LLM Model Dependencies
AI models for different purposes and phases.

```yaml
dependencies:
  models:
    - name: claude-3-opus
      provider: anthropic
      version: "20240229"
      purpose: complex-reasoning
      config:
        temperature: 0.7
        max_tokens: 4096
        top_p: 0.95
      cost:
        input: 0.015  # per 1K tokens
        output: 0.075  # per 1K tokens
      phase: prototyping
      fallback: claude-3-sonnet
    
    - name: gpt-4-turbo
      provider: openai
      version: "gpt-4-turbo-preview"
      purpose: code-generation
      config:
        temperature: 0.3
        max_tokens: 8192
        response_format: {"type": "json_object"}
      phase: all
    
    - name: llama-3.1-70b
      provider: local
      endpoint: ${LOCAL_LLM_ENDPOINT:-http://localhost:11434}
      purpose: fast-inference
      config:
        temperature: 0.5
        num_predict: 2048
      requirements:
        gpu_memory: "48GB"
        compute_capability: "7.0"
      phase: prototyping
    
    - name: text-embedding-3-large
      provider: openai
      purpose: embeddings
      dimensions: 3072
      config:
        batch_size: 100
      phase: all
```

#### LLM Model Management
```python
class LLMDependencyManager:
    """Manage LLM model dependencies"""
    
    def __init__(self):
        self.models = {}
        self.providers = {
            'anthropic': AnthropicProvider(),
            'openai': OpenAIProvider(),
            'local': LocalLLMProvider(),
            'huggingface': HuggingFaceProvider()
        }
        self.usage_tracker = UsageTracker()
    
    async def initialize_models(self, dependencies):
        """Initialize LLM models based on dependencies"""
        
        for model_dep in dependencies.get('models', []):
            provider = self.providers.get(model_dep['provider'])
            if not provider:
                raise ValueError(f"Unknown provider: {model_dep['provider']}")
            
            # Check local requirements
            if model_dep['provider'] == 'local':
                if not self.check_requirements(model_dep.get('requirements', {})):
                    # Try fallback
                    if 'fallback' in model_dep:
                        model_dep = self.get_fallback_model(model_dep['fallback'])
                    else:
                        raise RuntimeError(f"Local requirements not met for {model_dep['name']}")
            
            # Initialize model
            model = await provider.initialize(
                name=model_dep['name'],
                version=model_dep.get('version'),
                config=model_dep.get('config', {})
            )
            
            # Wrap with usage tracking
            model = UsageTrackingWrapper(model, self.usage_tracker)
            
            # Store by purpose
            purpose = model_dep['purpose']
            if purpose not in self.models:
                self.models[purpose] = []
            self.models[purpose].append({
                'model': model,
                'config': model_dep,
                'provider': provider
            })
    
    async def generate(self, purpose: str, prompt: str, **kwargs):
        """Generate using appropriate model for purpose"""
        
        if purpose not in self.models:
            raise ValueError(f"No model configured for purpose: {purpose}")
        
        # Get primary model for purpose
        model_info = self.models[purpose][0]
        model = model_info['model']
        config = model_info['config']
        
        # Merge configs
        generation_config = {**config.get('config', {}), **kwargs}
        
        try:
            # Generate with tracking
            result = await model.generate(prompt, **generation_config)
            
            # Track costs if configured
            if 'cost' in config:
                self.usage_tracker.track_cost(
                    model=config['name'],
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    cost_config=config['cost']
                )
            
            return result
            
        except Exception as e:
            # Try fallback if available
            if 'fallback' in config:
                return await self.generate_with_fallback(
                    config['fallback'], 
                    prompt, 
                    **kwargs
                )
            raise
    
    async def embed(self, text: str, purpose: str = 'embeddings'):
        """Generate embeddings using configured model"""
        
        embedding_models = self.models.get(purpose, [])
        if not embedding_models:
            raise ValueError("No embedding model configured")
        
        model = embedding_models[0]['model']
        return await model.embed(text)
```

## Dependency Resolution

### Resolution Strategy
```python
class DependencyResolver:
    """Resolve and validate all dependencies"""
    
    def __init__(self):
        self.managers = {
            'features': FeatureComposer(),
            'mcp_servers': MCPDependencyManager(),
            'services': ServiceDependencyManager(),
            'packages': PackageDependencyManager(),
            'containers': ContainerDependencyManager(),
            'models': LLMDependencyManager()
        }
    
    async def resolve(self, feature_path: str, phase: str = 'hybrid'):
        """Resolve all dependencies for a feature"""
        
        # Parse feature file
        feature = await self.parse_feature(feature_path)
        
        # Build dependency graph
        graph = await self.build_dependency_graph(feature)
        
        # Check for circular dependencies
        if self.has_circular_dependency(graph):
            raise CircularDependencyError("Circular dependency detected")
        
        # Resolve in topological order
        resolution_order = self.topological_sort(graph)
        
        resolved = {}
        for dep_type in resolution_order:
            if dep_type in feature.dependencies:
                manager = self.managers.get(dep_type)
                if manager:
                    resolved[dep_type] = await manager.resolve(
                        feature.dependencies[dep_type],
                        phase=phase
                    )
        
        return resolved
    
    def validate_versions(self, dependencies):
        """Validate version compatibility"""
        
        for dep_type, deps in dependencies.items():
            for dep in deps:
                if 'version' in dep:
                    if not self.is_version_compatible(dep['version']):
                        raise VersionError(f"Incompatible version: {dep}")
```

## Dependency Injection

### Injection Patterns
```python
class DependencyInjector:
    """Inject resolved dependencies into features"""
    
    def __init__(self):
        self.injection_strategies = {
            'constructor': self.inject_constructor,
            'property': self.inject_property,
            'method': self.inject_method,
            'context': self.inject_context
        }
    
    async def inject(self, feature, dependencies, strategy='context'):
        """Inject dependencies using specified strategy"""
        
        injector = self.injection_strategies.get(strategy)
        if not injector:
            raise ValueError(f"Unknown injection strategy: {strategy}")
        
        return await injector(feature, dependencies)
    
    async def inject_context(self, feature, dependencies):
        """Inject dependencies into execution context"""
        
        context = ExecutionContext()
        
        # Add resolved dependencies to context
        context.features = dependencies.get('features', {})
        context.mcp_servers = dependencies.get('mcp_servers', {})
        context.services = dependencies.get('services', {})
        context.models = dependencies.get('models', {})
        
        # Create dependency proxy
        proxy = DependencyProxy(context)
        
        # Inject into feature
        feature.deps = proxy
        
        return feature
```

## Dependency Profiles

### Environment-Specific Dependencies
```yaml
# development.deps.yaml
profiles:
  development:
    mcp_servers:
      - name: redis-mcp-dev
        endpoint: localhost:8000
    models:
      - name: gpt-3.5-turbo  # Cheaper for development
        provider: openai
        purpose: all
    containers:
      - name: redis
        image: redis:alpine  # Lighter image
  
  production:
    mcp_servers:
      - name: redis-mcp-prod
        endpoint: redis-mcp.prod.internal:8000
    models:
      - name: claude-3-opus
        provider: anthropic
        purpose: complex-reasoning
    containers:
      - name: redis
        image: redis/redis-stack:latest
        cluster: true
```

### Profile Selection
```python
class DependencyProfileManager:
    """Manage environment-specific dependency profiles"""
    
    def __init__(self):
        self.profiles = {}
        self.active_profile = None
    
    def load_profile(self, profile_name: str):
        """Load dependency profile"""
        
        profile_path = f"profiles/{profile_name}.deps.yaml"
        with open(profile_path) as f:
            profile = yaml.safe_load(f)
        
        self.profiles[profile_name] = profile
        self.active_profile = profile_name
    
    def merge_dependencies(self, base_deps, profile_name=None):
        """Merge base dependencies with profile overrides"""
        
        if not profile_name:
            profile_name = self.active_profile or os.getenv('EOL_PROFILE', 'development')
        
        if profile_name not in self.profiles:
            self.load_profile(profile_name)
        
        profile = self.profiles[profile_name]
        
        # Deep merge dependencies
        merged = deepcopy(base_deps)
        for dep_type, deps in profile.get('profiles', {}).get(profile_name, {}).items():
            if dep_type in merged:
                merged[dep_type] = self.merge_dependency_list(
                    merged[dep_type], 
                    deps
                )
            else:
                merged[dep_type] = deps
        
        return merged
```

## Health Monitoring

### Dependency Health Checks
```python
class DependencyHealthMonitor:
    """Monitor health of all dependencies"""
    
    def __init__(self):
        self.checks = {}
        self.status = {}
        self.alerts = AlertManager()
    
    async def monitor(self, dependencies):
        """Continuously monitor dependency health"""
        
        while True:
            for dep_type, deps in dependencies.items():
                for dep in deps:
                    health = await self.check_health(dep_type, dep)
                    
                    # Update status
                    dep_id = f"{dep_type}:{dep.get('name', dep.get('path'))}"
                    self.status[dep_id] = health
                    
                    # Alert on status change
                    if health['status'] != 'healthy':
                        await self.alerts.send(
                            level='warning' if not dep.get('required') else 'error',
                            message=f"Dependency {dep_id} is {health['status']}",
                            details=health
                        )
            
            await asyncio.sleep(30)  # Check every 30 seconds
    
    async def check_health(self, dep_type: str, dep: dict):
        """Check health of specific dependency"""
        
        if dep_type == 'mcp_servers':
            return await self.check_mcp_health(dep)
        elif dep_type == 'services':
            return await self.check_service_health(dep)
        elif dep_type == 'containers':
            return await self.check_container_health(dep)
        elif dep_type == 'models':
            return await self.check_model_health(dep)
        else:
            return {'status': 'unknown', 'message': 'No health check available'}
```

## Best Practices

### 1. Version Management
- Use semantic versioning for features
- Pin major versions in production
- Use ranges for development dependencies

### 2. Fallback Strategies
- Always define fallbacks for critical dependencies
- Test fallback paths regularly
- Monitor fallback usage

### 3. Phase-Specific Dependencies
- Minimize prototyping dependencies
- Isolate implementation dependencies
- Share common dependencies across phases

### 4. Cost Optimization
- Use cheaper models for development
- Implement caching for expensive operations
- Track and monitor usage costs

### 5. Security
- Never hardcode credentials
- Use environment variables for secrets
- Implement least-privilege access

### 6. Performance
- Lazy-load optional dependencies
- Cache resolved dependencies
- Implement connection pooling

## CLI Integration

### Dependency Commands
```bash
# Install all dependencies
eol deps install

# Install for specific phase
eol deps install --phase prototyping

# Check dependency health
eol deps health

# List all dependencies
eol deps list

# Update dependencies
eol deps update

# Generate dependency graph
eol deps graph --output deps.svg

# Validate dependencies
eol deps validate feature.eol.md

# Show dependency costs
eol deps cost --period month
```

### Dependency Resolution in Execution
```python
@cli.command()
async def run(
    feature: str,
    phase: str = "hybrid",
    profile: str = None,
    skip_deps: bool = False
):
    """Run feature with dependency resolution"""
    
    if not skip_deps:
        # Resolve dependencies
        resolver = DependencyResolver()
        deps = await resolver.resolve(feature, phase)
        
        # Initialize dependencies
        await initialize_dependencies(deps)
        
        # Inject into execution context
        context = create_context_with_deps(deps)
    else:
        context = create_minimal_context()
    
    # Execute feature
    result = await execute_feature(feature, context, phase)
    
    return result
```

## Conclusion
EOL's comprehensive dependency system enables complex system composition through declarative dependency management. By supporting multiple dependency types including LLM models, the framework provides flexibility for both prototyping and production deployments while maintaining clear separation of concerns and phase-specific optimizations.