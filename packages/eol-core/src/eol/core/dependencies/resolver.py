"""Dependency Resolver - Resolves and manages feature dependencies"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import asyncio
import networkx as nx
from ..parser.eol_parser import EOLParser, EOLFeature, ExecutionPhase


class DependencyType(str, Enum):
    """Types of dependencies"""
    FEATURES = "features"
    MCP_SERVERS = "mcp_servers"
    SERVICES = "services"
    PACKAGES = "packages"
    CONTAINERS = "containers"
    MODELS = "models"


@dataclass
class DependencyNode:
    """Represents a dependency in the resolution graph"""
    type: DependencyType
    name: str
    version: Optional[str] = None
    phase: ExecutionPhase = ExecutionPhase.ALL
    config: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    instance: Optional[Any] = None
    fallback: Optional[str] = None
    required: bool = True


class CircularDependencyError(Exception):
    """Raised when circular dependencies are detected"""
    pass


class DependencyResolutionError(Exception):
    """Raised when dependency resolution fails"""
    pass


class VersionError(Exception):
    """Raised when version requirements are not met"""
    pass


class DependencyResolver:
    """Main dependency resolution engine"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.graph = nx.DiGraph()
        self.resolved: Dict[str, Any] = {}
        self.parser = EOLParser()
        self.managers: Dict[DependencyType, Any] = {}
        self._initialize_managers()
    
    def _initialize_managers(self):
        """Initialize dependency type managers"""
        # Managers will be initialized as needed
        # This avoids circular imports
        pass
    
    async def resolve_feature(self, feature_path: str, phase: str = "hybrid") -> Dict[str, Any]:
        """Resolve all dependencies for a feature"""
        
        # Parse feature file
        feature = self.parser.parse_feature(Path(feature_path))
        
        # Build dependency graph
        await self._build_graph(feature, ExecutionPhase(phase))
        
        # Check for circular dependencies
        if not nx.is_directed_acyclic_graph(self.graph):
            cycles = list(nx.simple_cycles(self.graph))
            raise CircularDependencyError(f"Circular dependencies detected: {cycles}")
        
        # Resolve in topological order
        resolution_order = list(nx.topological_sort(self.graph))
        
        # Resolve each dependency
        for node_id in resolution_order:
            node_data = self.graph.nodes[node_id]
            if 'data' in node_data:
                node = node_data['data']
                if not node.resolved:
                    await self._resolve_node(node)
        
        return self.resolved
    
    async def _build_graph(self, feature: EOLFeature, phase: ExecutionPhase):
        """Build dependency graph from feature"""
        
        # Create root node for feature
        feature_node = DependencyNode(
            type=DependencyType.FEATURES,
            name=feature.name,
            version=feature.version,
            phase=phase,
            config={'feature': feature}
        )
        
        self.graph.add_node(feature.name, data=feature_node)
        
        # Add dependencies to graph
        for dep_type_str, deps in feature.dependencies.items():
            dep_type = DependencyType(dep_type_str)
            for dep in deps:
                await self._add_dependency(feature.name, dep_type, dep, phase)
    
    async def _add_dependency(self, parent: str, dep_type: DependencyType, 
                            dep: Dict[str, Any], phase: ExecutionPhase):
        """Add dependency to graph"""
        
        # Create unique node ID
        dep_name = dep.get('name', dep.get('path', 'unknown'))
        node_id = f"{dep_type.value}:{dep_name}"
        
        # Check phase compatibility
        dep_phase_str = dep.get('phase', 'all')
        dep_phase = ExecutionPhase(dep_phase_str)
        
        if not self._is_phase_compatible(dep_phase, phase):
            return  # Skip incompatible dependencies
        
        # Create dependency node
        node = DependencyNode(
            type=dep_type,
            name=dep_name,
            version=dep.get('version'),
            phase=dep_phase,
            config=dep,
            fallback=dep.get('fallback'),
            required=dep.get('required', True)
        )
        
        # Add to graph if not already present
        if node_id not in self.graph:
            self.graph.add_node(node_id, data=node)
        
        # Add edge from parent to dependency
        self.graph.add_edge(parent, node_id)
        
        # Recursively add sub-dependencies for features
        if dep_type == DependencyType.FEATURES:
            feature_path = self.project_root / dep.get('path', f"{dep_name}.eol.md")
            if feature_path.exists():
                try:
                    sub_feature = self.parser.parse_feature(feature_path)
                    for sub_type_str, sub_deps in sub_feature.dependencies.items():
                        sub_type = DependencyType(sub_type_str)
                        for sub_dep in sub_deps:
                            await self._add_dependency(node_id, sub_type, sub_dep, phase)
                except Exception as e:
                    if node.required:
                        raise DependencyResolutionError(f"Failed to parse feature {feature_path}: {e}")
    
    def _is_phase_compatible(self, dep_phase: ExecutionPhase, current_phase: ExecutionPhase) -> bool:
        """Check if dependency is compatible with current phase"""
        
        if dep_phase == ExecutionPhase.ALL or current_phase == ExecutionPhase.ALL:
            return True
        if current_phase == ExecutionPhase.HYBRID:
            return True  # Hybrid supports all phases
        return dep_phase == current_phase
    
    async def _resolve_node(self, node: DependencyNode):
        """Resolve individual dependency node"""
        
        try:
            # Get appropriate manager for dependency type
            manager = await self._get_manager(node.type)
            
            # Resolve dependency
            instance = await manager.resolve(node)
            
            # Mark as resolved
            node.instance = instance
            node.resolved = True
            
            # Store resolved instance
            key = f"{node.type.value}:{node.name}"
            self.resolved[key] = instance
            
        except Exception as e:
            # Try fallback if available
            if node.fallback:
                await self._resolve_fallback(node)
            elif node.required:
                raise DependencyResolutionError(f"Failed to resolve {node.name}: {e}")
            else:
                # Optional dependency - log warning but continue
                print(f"Warning: Optional dependency {node.name} could not be resolved: {e}")
    
    async def _resolve_fallback(self, node: DependencyNode):
        """Resolve fallback dependency"""
        
        fallback_dep = {
            'name': node.fallback,
            'phase': node.phase.value,
            'required': node.required
        }
        
        # Create fallback node
        fallback_node = DependencyNode(
            type=node.type,
            name=node.fallback,
            phase=node.phase,
            config=fallback_dep,
            required=node.required
        )
        
        # Try to resolve fallback
        await self._resolve_node(fallback_node)
        
        # If successful, use fallback as the resolution for original
        if fallback_node.resolved:
            key = f"{node.type.value}:{node.name}"
            self.resolved[key] = fallback_node.instance
            node.resolved = True
            node.instance = fallback_node.instance
    
    async def _get_manager(self, dep_type: DependencyType):
        """Get or create manager for dependency type"""
        
        if dep_type not in self.managers:
            if dep_type == DependencyType.FEATURES:
                from .managers.features import FeatureDependencyManager
                self.managers[dep_type] = FeatureDependencyManager(self.project_root)
            elif dep_type == DependencyType.MCP_SERVERS:
                from .managers.mcp import MCPServerManager
                self.managers[dep_type] = MCPServerManager()
            elif dep_type == DependencyType.SERVICES:
                from .managers.services import ServiceManager
                self.managers[dep_type] = ServiceManager()
            elif dep_type == DependencyType.PACKAGES:
                from .managers.packages import PackageManager
                self.managers[dep_type] = PackageManager()
            elif dep_type == DependencyType.CONTAINERS:
                from .managers.containers import ContainerManager
                self.managers[dep_type] = ContainerManager()
            elif dep_type == DependencyType.MODELS:
                from .managers.models import ModelManager
                self.managers[dep_type] = ModelManager()
            else:
                raise ValueError(f"Unknown dependency type: {dep_type}")
        
        return self.managers[dep_type]
    
    def get_dependency_graph(self) -> nx.DiGraph:
        """Get the dependency graph"""
        return self.graph
    
    def get_resolution_order(self) -> List[str]:
        """Get the order in which dependencies were/will be resolved"""
        
        if nx.is_directed_acyclic_graph(self.graph):
            return list(nx.topological_sort(self.graph))
        else:
            return []
    
    def validate_versions(self) -> List[str]:
        """Validate version compatibility of all dependencies"""
        
        errors = []
        
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            if 'data' in node_data:
                node = node_data['data']
                if node.version:
                    # Version validation logic here
                    # This would check semantic versioning compatibility
                    pass
        
        return errors
    
    def get_unresolved_dependencies(self) -> List[str]:
        """Get list of unresolved dependencies"""
        
        unresolved = []
        
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            if 'data' in node_data:
                node = node_data['data']
                if not node.resolved and node.required:
                    unresolved.append(f"{node.type.value}:{node.name}")
        
        return unresolved
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all resolved dependencies"""
        
        health_status = {}
        
        for key, instance in self.resolved.items():
            # Check if instance has health check method
            if hasattr(instance, 'health_check'):
                try:
                    status = await instance.health_check()
                    health_status[key] = {'status': 'healthy', 'details': status}
                except Exception as e:
                    health_status[key] = {'status': 'unhealthy', 'error': str(e)}
            else:
                health_status[key] = {'status': 'unknown', 'message': 'No health check available'}
        
        return health_status