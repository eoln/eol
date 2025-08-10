# EOL Dependency Resolution and Injection Implementation

## Overview
Complete implementation guide for EOL's dependency resolution and injection mechanism, providing concrete examples and integration patterns.

## Core Implementation

### Dependency Resolver Engine
```python
# eol/core/dependencies/resolver.py
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import asyncio
import networkx as nx
from pathlib import Path

class DependencyPhase(Enum):
    """Execution phases for dependencies"""
    PROTOTYPING = "prototyping"
    IMPLEMENTATION = "implementation"
    HYBRID = "hybrid"
    ALL = "all"

@dataclass
class DependencyNode:
    """Represents a dependency in the graph"""
    type: str  # features, mcp_servers, services, packages, containers, models
    name: str
    version: Optional[str]
    phase: DependencyPhase
    config: Dict[str, Any]
    resolved: bool = False
    instance: Optional[Any] = None

class DependencyResolver:
    """Main dependency resolution engine"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.graph = nx.DiGraph()
        self.resolved = {}
        self.managers = self._initialize_managers()
        
    def _initialize_managers(self):
        """Initialize dependency type managers"""
        return {
            'features': FeatureDependencyManager(self.project_root),
            'mcp_servers': MCPServerManager(),
            'services': ServiceManager(),
            'packages': PackageManager(),
            'containers': ContainerManager(),
            'models': ModelManager()
        }
    
    async def resolve_feature(self, feature_path: str, phase: str = "hybrid"):
        """Resolve all dependencies for a feature"""
        
        # Parse feature file
        feature = await self._parse_feature(feature_path)
        
        # Build dependency graph
        await self._build_graph(feature, phase)
        
        # Check for circular dependencies
        if not nx.is_directed_acyclic_graph(self.graph):
            cycles = list(nx.simple_cycles(self.graph))
            raise CircularDependencyError(f"Circular dependencies detected: {cycles}")
        
        # Resolve in topological order
        resolution_order = list(nx.topological_sort(self.graph))
        
        # Resolve each dependency
        for node_id in resolution_order:
            node = self.graph.nodes[node_id]['data']
            if not node.resolved:
                await self._resolve_node(node)
        
        return self.resolved
    
    async def _build_graph(self, feature, phase):
        """Build dependency graph"""
        
        feature_node = DependencyNode(
            type='feature',
            name=feature.name,
            version=feature.version,
            phase=DependencyPhase(phase),
            config=feature.config
        )
        
        self.graph.add_node(feature.name, data=feature_node)
        
        # Add dependencies to graph
        for dep_type, deps in feature.dependencies.items():
            for dep in deps:
                await self._add_dependency(feature.name, dep_type, dep, phase)
    
    async def _add_dependency(self, parent: str, dep_type: str, dep: dict, phase: str):
        """Add dependency to graph"""
        
        # Create unique node ID
        node_id = f"{dep_type}:{dep.get('name', dep.get('path'))}"
        
        # Check phase compatibility
        dep_phase = dep.get('phase', 'all')
        if not self._is_phase_compatible(dep_phase, phase):
            return
        
        # Create dependency node
        node = DependencyNode(
            type=dep_type,
            name=dep.get('name', dep.get('path')),
            version=dep.get('version'),
            phase=DependencyPhase(dep_phase),
            config=dep
        )
        
        # Add to graph
        self.graph.add_node(node_id, data=node)
        self.graph.add_edge(parent, node_id)
        
        # Recursively add sub-dependencies
        if dep_type == 'features':
            sub_feature = await self._parse_feature(dep['path'])
            for sub_type, sub_deps in sub_feature.dependencies.items():
                for sub_dep in sub_deps:
                    await self._add_dependency(node_id, sub_type, sub_dep, phase)
    
    async def _resolve_node(self, node: DependencyNode):
        """Resolve individual dependency node"""
        
        manager = self.managers.get(node.type)
        if not manager:
            raise ValueError(f"No manager for dependency type: {node.type}")
        
        try:
            # Resolve dependency
            instance = await manager.resolve(node)
            node.instance = instance
            node.resolved = True
            
            # Store resolved instance
            key = f"{node.type}:{node.name}"
            self.resolved[key] = instance
            
        except Exception as e:
            # Try fallback if available
            if 'fallback' in node.config:
                await self._resolve_fallback(node)
            else:
                raise DependencyResolutionError(f"Failed to resolve {node.name}: {e}")
    
    def _is_phase_compatible(self, dep_phase: str, current_phase: str) -> bool:
        """Check if dependency is compatible with current phase"""
        
        if dep_phase == 'all' or current_phase == 'all':
            return True
        if current_phase == 'hybrid':
            return True
        return dep_phase == current_phase
```

### Dependency Injection Framework
```python
# eol/core/dependencies/injection.py
from typing import Any, Dict, Optional, Callable
from functools import wraps
import inspect

class DependencyInjector:
    """Dependency injection framework"""
    
    def __init__(self, resolved_dependencies: Dict[str, Any]):
        self.dependencies = resolved_dependencies
        self.container = DependencyContainer(resolved_dependencies)
    
    def inject(self, func: Callable) -> Callable:
        """Decorator for dependency injection"""
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get function signature
            sig = inspect.signature(func)
            
            # Inject dependencies
            for param_name, param in sig.parameters.items():
                if param.annotation and param_name not in kwargs:
                    # Try to resolve dependency by type
                    dep = self._resolve_by_type(param.annotation)
                    if dep:
                        kwargs[param_name] = dep
                    # Try to resolve by name
                    elif param_name in self.dependencies:
                        kwargs[param_name] = self.dependencies[param_name]
            
            # Call function with injected dependencies
            if inspect.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        return wrapper
    
    def _resolve_by_type(self, annotation):
        """Resolve dependency by type annotation"""
        
        # Map annotations to dependency types
        type_map = {
            'RedisClient': 'mcp_servers:redis-mcp',
            'AuthService': 'services:auth-service',
            'LLMModel': 'models:claude-3-opus',
            'Database': 'containers:postgres'
        }
        
        type_name = annotation.__name__ if hasattr(annotation, '__name__') else str(annotation)
        dep_key = type_map.get(type_name)
        
        if dep_key and dep_key in self.dependencies:
            return self.dependencies[dep_key]
        
        return None

class DependencyContainer:
    """Container for managing dependency lifecycle"""
    
    def __init__(self, dependencies: Dict[str, Any]):
        self.dependencies = dependencies
        self.singletons = {}
        self.factories = {}
    
    def register_singleton(self, key: str, instance: Any):
        """Register a singleton dependency"""
        self.singletons[key] = instance
    
    def register_factory(self, key: str, factory: Callable):
        """Register a factory for creating dependencies"""
        self.factories[key] = factory
    
    def get(self, key: str) -> Any:
        """Get dependency by key"""
        
        # Check singletons first
        if key in self.singletons:
            return self.singletons[key]
        
        # Check factories
        if key in self.factories:
            instance = self.factories[key]()
            self.singletons[key] = instance  # Cache as singleton
            return instance
        
        # Check resolved dependencies
        if key in self.dependencies:
            return self.dependencies[key]
        
        raise KeyError(f"Dependency not found: {key}")
    
    async def get_async(self, key: str) -> Any:
        """Get dependency asynchronously"""
        
        if key in self.factories:
            factory = self.factories[key]
            if inspect.iscoroutinefunction(factory):
                instance = await factory()
                self.singletons[key] = instance
                return instance
        
        return self.get(key)
```

### Feature-Specific Managers

#### Feature Dependency Manager
```python
# eol/core/dependencies/managers/features.py
class FeatureDependencyManager:
    """Manages feature dependencies"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.cache = {}
        self.parser = EOLParser()
    
    async def resolve(self, node: DependencyNode):
        """Resolve feature dependency"""
        
        feature_path = self.project_root / node.config.get('path', f"{node.name}.eol.md")
        
        # Check cache
        if str(feature_path) in self.cache:
            return self.cache[str(feature_path)]
        
        # Parse feature
        feature = await self.parser.parse(feature_path)
        
        # Validate version
        if node.version:
            if not self._check_version_compatibility(feature.version, node.version):
                raise VersionError(f"Version mismatch: {feature.version} vs {node.version}")
        
        # Create feature instance
        instance = FeatureInstance(
            feature=feature,
            config=node.config,
            phase=node.phase
        )
        
        # Extract injectable functions
        if 'inject' in node.config:
            instance.injectable = self._extract_injectable(feature, node.config['inject'])
        
        # Cache and return
        self.cache[str(feature_path)] = instance
        return instance
    
    def _extract_injectable(self, feature, inject_list):
        """Extract functions to be injected"""
        
        injectable = {}
        for func_name in inject_list:
            if func_name in feature.operations:
                injectable[func_name] = feature.operations[func_name]
        return injectable
```

#### MCP Server Manager
```python
# eol/core/dependencies/managers/mcp.py
from fastmcp import FastMCP
import httpx

class MCPServerManager:
    """Manages MCP server dependencies"""
    
    def __init__(self):
        self.clients = {}
        self.health_checker = HealthChecker()
    
    async def resolve(self, node: DependencyNode):
        """Resolve MCP server dependency"""
        
        config = node.config
        transport = config.get('transport', 'stdio')
        
        if transport == 'stdio':
            client = await self._create_stdio_client(config)
        elif transport == 'sse':
            client = await self._create_sse_client(config)
        elif transport == 'websocket':
            client = await self._create_websocket_client(config)
        else:
            raise ValueError(f"Unknown transport: {transport}")
        
        # Test connection
        if not await self.health_checker.check(client):
            if 'fallback' in config:
                return await self._resolve_fallback(config['fallback'])
            raise ConnectionError(f"Failed to connect to {node.name}")
        
        # Wrap with circuit breaker
        client = CircuitBreakerWrapper(client, node.name)
        
        self.clients[node.name] = client
        return client
    
    async def _create_stdio_client(self, config):
        """Create stdio MCP client"""
        
        from mcp import StdioClient
        
        return StdioClient(
            command=config.get('command', ['python', '-m', config['name']]),
            env=config.get('env', {})
        )
    
    async def _create_sse_client(self, config):
        """Create SSE MCP client"""
        
        from mcp import SSEClient
        
        return SSEClient(
            url=config.get('endpoint', f"http://localhost:8000"),
            headers=self._build_auth_headers(config.get('auth'))
        )
```

#### LLM Model Manager
```python
# eol/core/dependencies/managers/models.py
from typing import Dict, List, Optional
import litellm

class ModelManager:
    """Manages LLM model dependencies"""
    
    def __init__(self):
        self.models = {}
        self.usage_tracker = UsageTracker()
        self.cost_calculator = CostCalculator()
    
    async def resolve(self, node: DependencyNode):
        """Resolve LLM model dependency"""
        
        config = node.config
        provider = config['provider']
        
        # Create model client based on provider
        if provider == 'anthropic':
            client = await self._create_anthropic_client(config)
        elif provider == 'openai':
            client = await self._create_openai_client(config)
        elif provider == 'local':
            client = await self._create_local_client(config)
        else:
            # Use litellm for universal support
            client = await self._create_litellm_client(config)
        
        # Wrap with usage tracking
        client = UsageTrackingWrapper(
            client=client,
            tracker=self.usage_tracker,
            cost_config=config.get('cost')
        )
        
        # Apply purpose-specific configuration
        client = PurposeConfiguredClient(
            client=client,
            purpose=config['purpose'],
            default_config=config.get('config', {})
        )
        
        self.models[node.name] = client
        return client
    
    async def _create_anthropic_client(self, config):
        """Create Anthropic client"""
        
        from anthropic import AsyncAnthropic
        
        client = AsyncAnthropic(
            api_key=os.getenv('ANTHROPIC_API_KEY'),
            max_retries=3
        )
        
        return AnthropicModelWrapper(
            client=client,
            model=config.get('version', 'claude-3-opus-20240229')
        )
    
    async def _create_local_client(self, config):
        """Create local model client"""
        
        endpoint = config.get('endpoint', 'http://localhost:11434')
        
        # Check if endpoint is accessible
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{endpoint}/api/tags")
                if response.status_code != 200:
                    raise ConnectionError(f"Local model endpoint not accessible: {endpoint}")
            except Exception as e:
                if 'fallback' in config:
                    return await self._resolve_fallback(config['fallback'])
                raise
        
        return LocalModelClient(
            endpoint=endpoint,
            model=config['name']
        )

class UsageTrackingWrapper:
    """Wrapper to track LLM usage and costs"""
    
    def __init__(self, client, tracker, cost_config=None):
        self.client = client
        self.tracker = tracker
        self.cost_config = cost_config or {}
    
    async def generate(self, prompt: str, **kwargs):
        """Generate with usage tracking"""
        
        start_time = time.time()
        
        # Call underlying model
        result = await self.client.generate(prompt, **kwargs)
        
        # Track usage
        usage = {
            'model': self.client.model_name,
            'input_tokens': result.usage.get('input_tokens', 0),
            'output_tokens': result.usage.get('output_tokens', 0),
            'total_tokens': result.usage.get('total_tokens', 0),
            'latency': time.time() - start_time
        }
        
        # Calculate cost if configured
        if self.cost_config:
            usage['cost'] = self._calculate_cost(usage)
        
        await self.tracker.track(usage)
        
        return result
    
    def _calculate_cost(self, usage):
        """Calculate usage cost"""
        
        input_cost = usage['input_tokens'] * self.cost_config.get('input', 0) / 1000
        output_cost = usage['output_tokens'] * self.cost_config.get('output', 0) / 1000
        
        return {
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': input_cost + output_cost,
            'currency': self.cost_config.get('currency', 'USD')
        }
```

### Execution Context with Dependencies
```python
# eol/core/execution/context.py
class ExecutionContext:
    """Execution context with injected dependencies"""
    
    def __init__(self, dependencies: Dict[str, Any]):
        self.dependencies = dependencies
        self.injector = DependencyInjector(dependencies)
        
        # Organize by type
        self.features = self._filter_by_type('features')
        self.mcp_servers = self._filter_by_type('mcp_servers')
        self.services = self._filter_by_type('services')
        self.models = self._filter_by_type('models')
        self.containers = self._filter_by_type('containers')
    
    def _filter_by_type(self, dep_type: str):
        """Filter dependencies by type"""
        
        filtered = {}
        prefix = f"{dep_type}:"
        
        for key, value in self.dependencies.items():
            if key.startswith(prefix):
                name = key[len(prefix):]
                filtered[name] = value
        
        return filtered
    
    async def execute_with_deps(self, operation: str, **kwargs):
        """Execute operation with dependency injection"""
        
        # Find operation in features
        for feature_name, feature in self.features.items():
            if hasattr(feature, 'operations') and operation in feature.operations:
                func = feature.operations[operation]
                
                # Inject dependencies
                injected_func = self.injector.inject(func)
                
                # Execute
                return await injected_func(**kwargs)
        
        raise ValueError(f"Operation not found: {operation}")
    
    async def get_model(self, purpose: str):
        """Get model by purpose"""
        
        for model_name, model in self.models.items():
            if model.purpose == purpose:
                return model
        
        raise ValueError(f"No model found for purpose: {purpose}")
    
    async def call_service(self, service_name: str, method: str, **kwargs):
        """Call external service"""
        
        if service_name not in self.services:
            raise ValueError(f"Service not found: {service_name}")
        
        service = self.services[service_name]
        return await service.call(method, **kwargs)
```

### Integration Example
```python
# eol/cli/commands/run.py
import typer
from pathlib import Path

app = typer.Typer()

@app.command()
async def run(
    feature: str = typer.Argument(..., help="Path to .eol.md file"),
    phase: str = typer.Option("hybrid", help="Execution phase"),
    profile: str = typer.Option(None, help="Dependency profile"),
    operation: str = typer.Option(None, help="Specific operation to run"),
    skip_deps: bool = typer.Option(False, help="Skip dependency resolution")
):
    """Run EOL feature with dependency injection"""
    
    try:
        # Initialize resolver
        resolver = DependencyResolver(Path.cwd())
        
        # Load profile if specified
        if profile:
            profile_manager = DependencyProfileManager()
            profile_manager.load_profile(profile)
        
        # Resolve dependencies
        if not skip_deps:
            typer.echo("Resolving dependencies...")
            dependencies = await resolver.resolve_feature(feature, phase)
            typer.echo(f"✓ Resolved {len(dependencies)} dependencies")
        else:
            dependencies = {}
        
        # Create execution context
        context = ExecutionContext(dependencies)
        
        # Parse feature
        parser = EOLParser()
        feature_spec = await parser.parse(feature)
        
        # Execute specific operation or all
        if operation:
            result = await context.execute_with_deps(operation)
        else:
            # Execute all operations in order
            results = []
            for op in feature_spec.operations:
                if _is_phase_compatible(op.get('phase', 'all'), phase):
                    result = await context.execute_with_deps(op['name'])
                    results.append(result)
            result = results
        
        # Display results
        typer.echo(f"✓ Execution completed successfully")
        if result:
            typer.echo(json.dumps(result, indent=2))
        
    except CircularDependencyError as e:
        typer.echo(f"✗ Circular dependency detected: {e}", err=True)
        raise typer.Exit(1)
    
    except DependencyResolutionError as e:
        typer.echo(f"✗ Failed to resolve dependencies: {e}", err=True)
        raise typer.Exit(1)
    
    except Exception as e:
        typer.echo(f"✗ Execution failed: {e}", err=True)
        raise typer.Exit(1)

@app.command()
async def deps():
    """Manage dependencies"""
    pass

@deps.command()
async def install(
    feature: str = typer.Argument(..., help="Feature file"),
    phase: str = typer.Option("all", help="Phase to install for")
):
    """Install dependencies for a feature"""
    
    resolver = DependencyResolver(Path.cwd())
    dependencies = await resolver.resolve_feature(feature, phase)
    
    # Install packages
    package_manager = PackageManager()
    for key, dep in dependencies.items():
        if key.startswith("packages:"):
            await package_manager.install(dep)
    
    typer.echo("✓ Dependencies installed")

@deps.command()
async def health(
    feature: str = typer.Argument(..., help="Feature file")
):
    """Check health of dependencies"""
    
    resolver = DependencyResolver(Path.cwd())
    dependencies = await resolver.resolve_feature(feature)
    
    monitor = DependencyHealthMonitor()
    
    for key, dep in dependencies.items():
        health = await monitor.check_health_for_key(key, dep)
        
        status_icon = "✓" if health['status'] == 'healthy' else "✗"
        typer.echo(f"{status_icon} {key}: {health['status']}")
        
        if health.get('message'):
            typer.echo(f"  {health['message']}")

@deps.command()
async def graph(
    feature: str = typer.Argument(..., help="Feature file"),
    output: str = typer.Option("deps.svg", help="Output file")
):
    """Generate dependency graph visualization"""
    
    resolver = DependencyResolver(Path.cwd())
    await resolver.resolve_feature(feature)
    
    # Generate visualization
    import matplotlib.pyplot as plt
    import networkx as nx
    
    pos = nx.spring_layout(resolver.graph)
    
    # Color nodes by type
    node_colors = []
    for node in resolver.graph.nodes():
        data = resolver.graph.nodes[node]['data']
        color_map = {
            'feature': 'blue',
            'mcp_servers': 'green',
            'services': 'orange',
            'packages': 'purple',
            'containers': 'red',
            'models': 'cyan'
        }
        node_colors.append(color_map.get(data.type, 'gray'))
    
    plt.figure(figsize=(12, 8))
    nx.draw(resolver.graph, pos, node_color=node_colors, with_labels=True,
            node_size=1500, font_size=10, font_weight='bold',
            arrows=True, edge_color='gray', arrowsize=20)
    
    plt.title("Dependency Graph")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output, format='svg', dpi=150)
    
    typer.echo(f"✓ Graph saved to {output}")
```

## Testing Dependency Resolution

### Unit Tests
```python
# tests/test_dependency_resolution.py
import pytest
from eol.core.dependencies import DependencyResolver

@pytest.mark.asyncio
async def test_circular_dependency_detection():
    """Test that circular dependencies are detected"""
    
    # Create feature with circular dependency
    feature_a = """
    ---
    name: feature-a
    version: 1.0.0
    phase: hybrid
    dependencies:
      features:
        - path: feature-b.eol.md
    ---
    """
    
    feature_b = """
    ---
    name: feature-b
    version: 1.0.0
    phase: hybrid
    dependencies:
      features:
        - path: feature-a.eol.md
    ---
    """
    
    resolver = DependencyResolver(Path("/tmp"))
    
    with pytest.raises(CircularDependencyError):
        await resolver.resolve_feature("feature-a.eol.md")

@pytest.mark.asyncio
async def test_phase_filtering():
    """Test that dependencies are filtered by phase"""
    
    feature = """
    ---
    name: test-feature
    version: 1.0.0
    phase: implementation
    dependencies:
      models:
        - name: claude-3
          provider: anthropic
          phase: prototyping
          purpose: reasoning
        - name: gpt-4
          provider: openai
          phase: implementation
          purpose: generation
    ---
    """
    
    resolver = DependencyResolver(Path("/tmp"))
    deps = await resolver.resolve_feature("test-feature.eol.md", phase="implementation")
    
    # Only implementation phase model should be resolved
    assert "models:gpt-4" in deps
    assert "models:claude-3" not in deps

@pytest.mark.asyncio
async def test_fallback_resolution():
    """Test fallback mechanism"""
    
    feature = """
    ---
    name: test-feature
    version: 1.0.0
    phase: prototyping
    dependencies:
      models:
        - name: expensive-model
          provider: anthropic
          purpose: reasoning
          fallback: cheap-model
        - name: cheap-model
          provider: openai
          purpose: reasoning
    ---
    """
    
    resolver = DependencyResolver(Path("/tmp"))
    
    # Mock expensive model to fail
    with patch('eol.core.dependencies.managers.models.ModelManager._create_anthropic_client') as mock:
        mock.side_effect = ConnectionError("API limit exceeded")
        
        deps = await resolver.resolve_feature("test-feature.eol.md")
        
        # Should fall back to cheap model
        assert deps["models:expensive-model"].name == "cheap-model"
```

## Conclusion

The dependency resolution and injection mechanism provides:

1. **Comprehensive Resolution**: Handles all dependency types with proper ordering
2. **Circular Detection**: Prevents circular dependencies through graph analysis
3. **Phase Awareness**: Resolves only phase-appropriate dependencies
4. **Fallback Support**: Automatic fallback to alternatives when primary fails
5. **Injection Patterns**: Multiple injection strategies for flexibility
6. **Health Monitoring**: Continuous monitoring of dependency health
7. **Cost Tracking**: Track and optimize LLM usage costs
8. **Testing Support**: Comprehensive testing utilities

This implementation ensures reliable, efficient dependency management throughout the EOL framework lifecycle.