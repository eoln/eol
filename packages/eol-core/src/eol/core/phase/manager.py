"""Phase Manager - Controls execution phase transitions and hybrid mode"""

from typing import Dict, List, Optional, Any, Set
from enum import Enum
from dataclasses import dataclass, field
import asyncio
from pathlib import Path
import json


class ExecutionPhase(str, Enum):
    """Execution phases for EOL features"""
    PROTOTYPING = "prototyping"
    IMPLEMENTATION = "implementation"
    HYBRID = "hybrid"
    ALL = "all"


@dataclass
class PhaseTransition:
    """Represents a phase transition event"""
    feature: str
    from_phase: ExecutionPhase
    to_phase: ExecutionPhase
    operations: Optional[List[str]] = None
    timestamp: Optional[float] = None
    reason: Optional[str] = None


@dataclass
class PhaseMetrics:
    """Metrics for phase execution"""
    phase: ExecutionPhase
    execution_count: int = 0
    total_duration: float = 0.0
    average_duration: float = 0.0
    success_rate: float = 1.0
    last_execution: Optional[float] = None


class PhaseManager:
    """Manages execution phases for EOL features"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path.cwd() / "eol.config.yaml"
        self.phase_config: Dict[str, ExecutionPhase] = {}
        self.operation_phases: Dict[str, Dict[str, ExecutionPhase]] = {}
        self.metrics: Dict[str, PhaseMetrics] = {}
        self.transitions: List[PhaseTransition] = []
        self._load_config()
    
    def _load_config(self):
        """Load phase configuration from file"""
        
        if self.config_path.exists():
            import yaml
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                
                if 'features' in config:
                    for feature, settings in config['features'].items():
                        if 'phase' in settings:
                            self.phase_config[feature] = ExecutionPhase(settings['phase'])
                        
                        if 'operations' in settings:
                            self.operation_phases[feature] = {}
                            for op_name, op_phase in settings['operations'].items():
                                self.operation_phases[feature][op_name] = ExecutionPhase(op_phase)
    
    def get_phase(self, feature: str) -> ExecutionPhase:
        """Get the current phase for a feature"""
        
        return self.phase_config.get(feature, ExecutionPhase.HYBRID)
    
    def get_operation_phase(self, feature: str, operation: str) -> ExecutionPhase:
        """Get the phase for a specific operation"""
        
        # Check operation-specific phase
        if feature in self.operation_phases:
            if operation in self.operation_phases[feature]:
                return self.operation_phases[feature][operation]
        
        # Fall back to feature phase
        return self.get_phase(feature)
    
    def set_phase(self, feature: str, phase: ExecutionPhase, operations: Optional[List[str]] = None):
        """Set the phase for a feature or specific operations"""
        
        old_phase = self.get_phase(feature)
        
        if operations:
            # Set phase for specific operations
            if feature not in self.operation_phases:
                self.operation_phases[feature] = {}
            
            for op in operations:
                self.operation_phases[feature][op] = phase
        else:
            # Set phase for entire feature
            self.phase_config[feature] = phase
        
        # Record transition
        transition = PhaseTransition(
            feature=feature,
            from_phase=old_phase,
            to_phase=phase,
            operations=operations,
            timestamp=asyncio.get_event_loop().time()
        )
        self.transitions.append(transition)
        
        # Save configuration
        self._save_config()
    
    def switch_phase(self, feature: str, to_phase: ExecutionPhase, 
                    operations: Optional[List[str]] = None,
                    reason: Optional[str] = None) -> Dict[str, Any]:
        """Switch feature execution phase with validation"""
        
        current_phase = self.get_phase(feature)
        
        # Validate transition
        if not self._is_valid_transition(current_phase, to_phase):
            raise ValueError(f"Invalid phase transition from {current_phase} to {to_phase}")
        
        # Perform transition
        self.set_phase(feature, to_phase, operations)
        
        # Record transition with reason
        if self.transitions:
            self.transitions[-1].reason = reason
        
        return {
            "feature": feature,
            "from_phase": current_phase.value,
            "to_phase": to_phase.value,
            "operations": operations,
            "reason": reason,
            "timestamp": asyncio.get_event_loop().time()
        }
    
    def _is_valid_transition(self, from_phase: ExecutionPhase, to_phase: ExecutionPhase) -> bool:
        """Check if a phase transition is valid"""
        
        # All transitions are valid for now
        # Could add restrictions later (e.g., no direct prototyping -> production)
        return True
    
    async def execute_with_phase(self, feature: str, operation: str, 
                                 prototyping_func, implementation_func, **kwargs):
        """Execute operation based on current phase"""
        
        phase = self.get_operation_phase(feature, operation)
        
        # Track metrics
        metric_key = f"{feature}:{operation}:{phase.value}"
        if metric_key not in self.metrics:
            self.metrics[metric_key] = PhaseMetrics(phase=phase)
        
        metric = self.metrics[metric_key]
        metric.execution_count += 1
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            if phase == ExecutionPhase.PROTOTYPING:
                result = await prototyping_func(**kwargs)
            elif phase == ExecutionPhase.IMPLEMENTATION:
                result = await implementation_func(**kwargs)
            elif phase == ExecutionPhase.HYBRID:
                # Try implementation first, fall back to prototyping
                try:
                    result = await implementation_func(**kwargs)
                except NotImplementedError:
                    result = await prototyping_func(**kwargs)
            else:  # ALL
                # Execute both and return combined result
                proto_result = await prototyping_func(**kwargs)
                impl_result = await implementation_func(**kwargs)
                result = {
                    "prototyping": proto_result,
                    "implementation": impl_result
                }
            
            # Update metrics on success
            duration = asyncio.get_event_loop().time() - start_time
            metric.total_duration += duration
            metric.average_duration = metric.total_duration / metric.execution_count
            metric.last_execution = asyncio.get_event_loop().time()
            
            return result
            
        except Exception as e:
            # Update failure metrics
            metric.success_rate = (metric.execution_count - 1) / metric.execution_count
            raise
    
    def get_metrics(self, feature: Optional[str] = None) -> Dict[str, Any]:
        """Get execution metrics for features"""
        
        if feature:
            # Get metrics for specific feature
            feature_metrics = {}
            for key, metric in self.metrics.items():
                if key.startswith(f"{feature}:"):
                    feature_metrics[key] = {
                        "phase": metric.phase.value,
                        "execution_count": metric.execution_count,
                        "average_duration": metric.average_duration,
                        "success_rate": metric.success_rate,
                        "last_execution": metric.last_execution
                    }
            return feature_metrics
        else:
            # Get all metrics
            all_metrics = {}
            for key, metric in self.metrics.items():
                all_metrics[key] = {
                    "phase": metric.phase.value,
                    "execution_count": metric.execution_count,
                    "average_duration": metric.average_duration,
                    "success_rate": metric.success_rate,
                    "last_execution": metric.last_execution
                }
            return all_metrics
    
    def analyze_readiness(self, feature: str) -> Dict[str, Any]:
        """Analyze feature readiness for phase transition"""
        
        analysis = {
            "feature": feature,
            "current_phase": self.get_phase(feature).value,
            "metrics": self.get_metrics(feature),
            "recommendations": []
        }
        
        # Analyze metrics for recommendations
        for key, metric_data in analysis["metrics"].items():
            phase = metric_data["phase"]
            
            if phase == "prototyping":
                # Check if ready for implementation
                if metric_data["execution_count"] > 10 and metric_data["success_rate"] > 0.95:
                    analysis["recommendations"].append({
                        "operation": key.split(":")[1],
                        "action": "ready_for_implementation",
                        "reason": "High success rate and sufficient testing"
                    })
            
            elif phase == "implementation":
                # Check for performance issues
                if metric_data["average_duration"] > 5.0:
                    analysis["recommendations"].append({
                        "operation": key.split(":")[1],
                        "action": "optimize_performance",
                        "reason": f"High average duration: {metric_data['average_duration']:.2f}s"
                    })
        
        return analysis
    
    def _save_config(self):
        """Save phase configuration to file"""
        
        import yaml
        
        config = {
            "features": {}
        }
        
        # Save feature phases
        for feature, phase in self.phase_config.items():
            config["features"][feature] = {"phase": phase.value}
        
        # Save operation phases
        for feature, operations in self.operation_phases.items():
            if feature not in config["features"]:
                config["features"][feature] = {}
            
            config["features"][feature]["operations"] = {}
            for op_name, op_phase in operations.items():
                config["features"][feature]["operations"][op_name] = op_phase.value
        
        # Write to file
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def get_transition_history(self, feature: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get phase transition history"""
        
        history = []
        
        for transition in self.transitions:
            if feature and transition.feature != feature:
                continue
            
            history.append({
                "feature": transition.feature,
                "from_phase": transition.from_phase.value,
                "to_phase": transition.to_phase.value,
                "operations": transition.operations,
                "timestamp": transition.timestamp,
                "reason": transition.reason
            })
        
        return history
    
    async def optimize_phases(self, feature: str) -> Dict[str, Any]:
        """Automatically optimize phase assignments based on metrics"""
        
        optimizations = {
            "feature": feature,
            "changes": []
        }
        
        # Get all metrics for the feature
        metrics = self.get_metrics(feature)
        
        for key, metric_data in metrics.items():
            parts = key.split(":")
            if len(parts) < 2:
                continue
            
            operation = parts[1]
            current_phase = metric_data["phase"]
            
            # Decision logic for optimization
            if current_phase == "prototyping":
                # Consider moving to implementation if stable
                if (metric_data["execution_count"] > 20 and 
                    metric_data["success_rate"] > 0.98 and
                    metric_data["average_duration"] < 2.0):
                    
                    self.set_phase(feature, ExecutionPhase.IMPLEMENTATION, [operation])
                    optimizations["changes"].append({
                        "operation": operation,
                        "from": "prototyping",
                        "to": "implementation",
                        "reason": "Stable performance metrics"
                    })
            
            elif current_phase == "implementation":
                # Consider reverting to prototyping if failing
                if metric_data["success_rate"] < 0.8:
                    self.set_phase(feature, ExecutionPhase.PROTOTYPING, [operation])
                    optimizations["changes"].append({
                        "operation": operation,
                        "from": "implementation",
                        "to": "prototyping",
                        "reason": f"Low success rate: {metric_data['success_rate']:.2%}"
                    })
        
        return optimizations