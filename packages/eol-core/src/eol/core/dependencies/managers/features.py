"""Feature Dependency Manager - Manages dependencies on other EOL features"""

from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import semver

from ...parser.eol_parser import EOLParser, EOLFeature
from ..resolver import DependencyNode


@dataclass
class FeatureInstance:
    """Represents a resolved feature dependency"""
    feature: EOLFeature
    config: Dict[str, Any]
    phase: str
    injectable: Optional[Dict[str, Any]] = None


class VersionError(Exception):
    """Raised when version requirements are not met"""
    pass


class FeatureDependencyManager:
    """Manages feature dependencies"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.cache: Dict[str, FeatureInstance] = {}
        self.parser = EOLParser()
    
    async def resolve(self, node: DependencyNode) -> FeatureInstance:
        """Resolve feature dependency"""
        
        # Determine feature path
        feature_path = self._get_feature_path(node)
        
        # Check cache
        cache_key = str(feature_path)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Validate file exists
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_path}")
        
        # Parse feature
        feature = self.parser.parse_feature(feature_path)
        
        # Validate version if specified
        if node.version:
            if not self._check_version_compatibility(feature.version, node.version):
                raise VersionError(
                    f"Version mismatch for {node.name}: "
                    f"required {node.version}, found {feature.version}"
                )
        
        # Create feature instance
        instance = FeatureInstance(
            feature=feature,
            config=node.config,
            phase=node.phase.value
        )
        
        # Extract injectable functions if specified
        if 'inject' in node.config:
            instance.injectable = self._extract_injectable(feature, node.config['inject'])
        
        # Cache and return
        self.cache[cache_key] = instance
        return instance
    
    def _get_feature_path(self, node: DependencyNode) -> Path:
        """Get the path to a feature file"""
        
        if 'path' in node.config:
            # Explicit path provided
            path = Path(node.config['path'])
            if not path.is_absolute():
                path = self.project_root / path
        else:
            # Derive from name
            path = self.project_root / "features" / f"{node.name}.eol.md"
        
        return path
    
    def _check_version_compatibility(self, actual: str, required: str) -> bool:
        """Check if actual version satisfies required version constraint"""
        
        try:
            # Parse actual version
            actual_ver = semver.VersionInfo.parse(actual)
            
            # Handle different constraint formats
            if required.startswith("^"):
                # Caret: compatible with version
                min_ver = semver.VersionInfo.parse(required[1:])
                max_ver = semver.VersionInfo(
                    major=min_ver.major + 1,
                    minor=0,
                    patch=0
                )
                return min_ver <= actual_ver < max_ver
            
            elif required.startswith("~"):
                # Tilde: approximately equivalent
                min_ver = semver.VersionInfo.parse(required[1:])
                max_ver = semver.VersionInfo(
                    major=min_ver.major,
                    minor=min_ver.minor + 1,
                    patch=0
                )
                return min_ver <= actual_ver < max_ver
            
            elif ">=" in required:
                # Greater than or equal
                min_ver = semver.VersionInfo.parse(required[2:])
                return actual_ver >= min_ver
            
            elif ">" in required:
                # Greater than
                min_ver = semver.VersionInfo.parse(required[1:])
                return actual_ver > min_ver
            
            elif "<=" in required:
                # Less than or equal
                max_ver = semver.VersionInfo.parse(required[2:])
                return actual_ver <= max_ver
            
            elif "<" in required:
                # Less than
                max_ver = semver.VersionInfo.parse(required[1:])
                return actual_ver < max_ver
            
            elif "," in required:
                # Range: e.g., ">=1.0.0,<2.0.0"
                constraints = required.split(",")
                for constraint in constraints:
                    if not self._check_version_compatibility(actual, constraint.strip()):
                        return False
                return True
            
            else:
                # Exact match
                required_ver = semver.VersionInfo.parse(required)
                return actual_ver == required_ver
                
        except ValueError:
            # If parsing fails, do string comparison
            return actual == required
    
    def _extract_injectable(self, feature: EOLFeature, inject_list: List[str]) -> Dict[str, Any]:
        """Extract functions/operations to be injected"""
        
        injectable = {}
        
        for item in inject_list:
            # Check in operations
            for op in feature.operations:
                if op.get('name') == item:
                    injectable[item] = op
                    break
            
            # Check in implementation
            if feature.implementation and item not in injectable:
                # Look for function in implementation code
                for lang, code in feature.implementation.items():
                    if lang != 'description' and f"def {item}" in code:
                        injectable[item] = {
                            'type': 'function',
                            'language': lang,
                            'code': self._extract_function(code, item)
                        }
                        break
        
        return injectable
    
    def _extract_function(self, code: str, function_name: str) -> str:
        """Extract a specific function from code"""
        
        lines = code.split('\n')
        function_lines = []
        in_function = False
        indent_level = 0
        
        for line in lines:
            if f"def {function_name}" in line:
                in_function = True
                indent_level = len(line) - len(line.lstrip())
                function_lines.append(line)
            elif in_function:
                current_indent = len(line) - len(line.lstrip())
                if line.strip() and current_indent <= indent_level:
                    # End of function
                    break
                function_lines.append(line)
        
        return '\n'.join(function_lines)
    
    async def health_check(self, instance: FeatureInstance) -> Dict[str, Any]:
        """Check health of a feature instance"""
        
        return {
            'name': instance.feature.name,
            'version': instance.feature.version,
            'phase': instance.phase,
            'operations': len(instance.feature.operations),
            'injectable': len(instance.injectable) if instance.injectable else 0,
            'status': 'healthy'
        }