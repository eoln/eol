"""EOL Core Engine - Parser, Phase Manager, and Context Management"""

from .parser.eol_parser import EOLParser, EOLFeature, EOLTest
from .phase.manager import PhaseManager, ExecutionPhase
from .context.manager import ContextManager
from .dependencies.resolver import DependencyResolver, DependencyNode

__version__ = "0.1.0"

__all__ = [
    "EOLParser",
    "EOLFeature",
    "EOLTest",
    "PhaseManager",
    "ExecutionPhase",
    "ContextManager",
    "DependencyResolver",
    "DependencyNode",
]