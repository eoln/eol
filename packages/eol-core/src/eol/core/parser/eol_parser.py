"""EOL File Parser - Parses .eol.md and .test.eol.md files"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml
import re
import markdown
from enum import Enum


class ExecutionPhase(str, Enum):
    """Execution phases for EOL features"""
    PROTOTYPING = "prototyping"
    IMPLEMENTATION = "implementation"
    HYBRID = "hybrid"
    ALL = "all"


@dataclass
class EOLFeature:
    """Represents a parsed .eol.md feature file"""
    name: str
    version: str
    phase: ExecutionPhase
    tags: List[str] = field(default_factory=list)
    description: str = ""
    requirements: List[str] = field(default_factory=list)
    context: List[str] = field(default_factory=list)
    prototyping: Optional[Dict[str, Any]] = None
    implementation: Optional[Dict[str, Any]] = None
    operations: List[Dict[str, Any]] = field(default_factory=list)
    configuration: Optional[Dict[str, Any]] = None
    dependencies: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    monitoring: Optional[Dict[str, Any]] = None
    tests: Optional[str] = None
    examples: Optional[List[str]] = None
    file_path: Optional[Path] = None


@dataclass
class EOLTest:
    """Represents a parsed .test.eol.md test file"""
    name: str
    feature: str
    version: str
    phase: ExecutionPhase
    coverage: Optional[Dict[str, Any]] = None
    setup: Optional[Dict[str, Any]] = None
    test_cases: List[Dict[str, Any]] = field(default_factory=list)
    fixtures: Optional[Dict[str, Any]] = None
    configuration: Optional[Dict[str, Any]] = None
    file_path: Optional[Path] = None


class EOLParser:
    """Parser for EOL feature and test files"""
    
    def __init__(self):
        self.markdown_parser = markdown.Markdown(
            extensions=['meta', 'fenced_code', 'tables']
        )
    
    def parse_feature(self, file_path: Path) -> EOLFeature:
        """Parse an .eol.md feature file"""
        
        # Validate file extension
        if not str(file_path).endswith('.eol.md'):
            raise ValueError(f"Invalid file extension. Expected .eol.md, got {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return self.parse_feature_content(content, file_path)
    
    def parse_feature_content(self, content: str, file_path: Optional[Path] = None) -> EOLFeature:
        """Parse EOL feature content"""
        
        # Split frontmatter and content
        parts = content.split('---', 2)
        if len(parts) < 3:
            raise ValueError("Invalid EOL file format: missing frontmatter")
        
        # Parse YAML frontmatter
        frontmatter = yaml.safe_load(parts[1])
        if not frontmatter:
            raise ValueError("Empty frontmatter")
        
        # Parse markdown content
        markdown_content = parts[2]
        sections = self._parse_markdown_sections(markdown_content)
        
        # Extract code blocks by language
        code_blocks = self._extract_code_blocks(markdown_content)
        
        # Build feature model
        feature = EOLFeature(
            name=frontmatter['name'],
            version=frontmatter['version'],
            phase=ExecutionPhase(frontmatter['phase']),
            tags=frontmatter.get('tags', []),
            description=sections.get('description', ''),
            requirements=self._parse_list_section(sections.get('requirements', '')),
            context=self._parse_context_references(sections.get('context', '')),
            prototyping=self._parse_prototyping_section(sections, code_blocks),
            implementation=self._parse_implementation_section(sections, code_blocks),
            operations=self._parse_operations(sections),
            configuration=self._parse_yaml_section(sections.get('configuration', '')),
            dependencies=self._parse_dependencies(frontmatter.get('dependencies', {})),
            monitoring=self._parse_yaml_section(sections.get('monitoring', '')),
            tests=frontmatter.get('tests'),
            examples=self._parse_code_examples(sections.get('examples', '')),
            file_path=file_path
        )
        
        # Validate feature
        self._validate_feature(feature)
        
        return feature
    
    def parse_test(self, file_path: Path) -> EOLTest:
        """Parse a .test.eol.md test file"""
        
        # Validate file extension
        if not str(file_path).endswith('.test.eol.md'):
            raise ValueError(f"Invalid file extension. Expected .test.eol.md, got {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return self.parse_test_content(content, file_path)
    
    def parse_test_content(self, content: str, file_path: Optional[Path] = None) -> EOLTest:
        """Parse EOL test content"""
        
        # Split frontmatter and content
        parts = content.split('---', 2)
        if len(parts) < 3:
            raise ValueError("Invalid test file format: missing frontmatter")
        
        # Parse YAML frontmatter
        frontmatter = yaml.safe_load(parts[1])
        
        # Parse markdown content
        markdown_content = parts[2]
        sections = self._parse_markdown_sections(markdown_content)
        code_blocks = self._extract_code_blocks(markdown_content)
        
        # Build test model
        test = EOLTest(
            name=frontmatter['name'],
            feature=frontmatter['feature'],
            version=frontmatter['version'],
            phase=ExecutionPhase(frontmatter.get('phase', 'hybrid')),
            coverage=frontmatter.get('coverage'),
            setup=self._parse_setup_section(sections, code_blocks),
            test_cases=self._parse_test_cases(sections, code_blocks),
            fixtures=self._parse_yaml_section(sections.get('test data', '')),
            configuration=self._parse_yaml_section(sections.get('test configuration', '')),
            file_path=file_path
        )
        
        return test
    
    def _parse_markdown_sections(self, content: str) -> Dict[str, str]:
        """Parse markdown content into sections by headers"""
        
        sections = {}
        current_section = None
        current_content = []
        
        lines = content.split('\n')
        for line in lines:
            # Check for headers
            if line.startswith('## '):
                # Save previous section
                if current_section:
                    sections[current_section.lower()] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = line[3:].strip()
                current_content = []
            elif current_section:
                current_content.append(line)
        
        # Save last section
        if current_section:
            sections[current_section.lower()] = '\n'.join(current_content).strip()
        
        return sections
    
    def _extract_code_blocks(self, content: str) -> Dict[str, List[str]]:
        """Extract code blocks by language identifier"""
        
        code_blocks = {}
        pattern = r'```(\w+)\n(.*?)```'
        
        matches = re.findall(pattern, content, re.DOTALL)
        for lang, code in matches:
            if lang not in code_blocks:
                code_blocks[lang] = []
            code_blocks[lang].append(code.strip())
        
        return code_blocks
    
    def _parse_list_section(self, content: str) -> List[str]:
        """Parse a section containing a list"""
        
        if not content:
            return []
        
        items = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('- '):
                items.append(line[2:])
            elif line.startswith('* '):
                items.append(line[2:])
            elif re.match(r'^\d+\.\s', line):
                items.append(re.sub(r'^\d+\.\s', '', line))
        
        return items
    
    def _parse_context_references(self, content: str) -> List[str]:
        """Parse context references (@ references)"""
        
        if not content:
            return []
        
        references = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('- @') or line.startswith('* @'):
                ref = line.split('@', 1)[1] if '@' in line else ''
                if ref:
                    references.append(ref.strip())
        
        return references
    
    def _parse_prototyping_section(self, sections: Dict, code_blocks: Dict) -> Optional[Dict]:
        """Parse prototyping section"""
        
        if 'prototyping' not in sections and 'natural' not in code_blocks:
            return None
        
        result = {}
        
        # Get natural language specifications
        if 'natural' in code_blocks:
            result['specification'] = '\n\n'.join(code_blocks['natural'])
        
        # Get prototyping description from section
        if 'prototyping' in sections:
            result['description'] = sections['prototyping']
        
        return result if result else None
    
    def _parse_implementation_section(self, sections: Dict, code_blocks: Dict) -> Optional[Dict]:
        """Parse implementation section"""
        
        if 'implementation' not in sections and not any(
            lang in code_blocks for lang in ['python', 'javascript', 'typescript', 'go', 'rust']
        ):
            return None
        
        result = {}
        
        # Get code implementations by language
        for lang in ['python', 'javascript', 'typescript', 'go', 'rust']:
            if lang in code_blocks:
                result[lang] = '\n\n'.join(code_blocks[lang])
        
        # Get implementation description
        if 'implementation' in sections:
            result['description'] = sections['implementation']
        
        return result if result else None
    
    def _parse_operations(self, sections: Dict) -> List[Dict]:
        """Parse operations section"""
        
        if 'operations' not in sections:
            return []
        
        try:
            # Try to parse as YAML
            operations_yaml = yaml.safe_load(sections['operations'])
            if isinstance(operations_yaml, dict) and 'operations' in operations_yaml:
                return operations_yaml['operations']
            elif isinstance(operations_yaml, list):
                return operations_yaml
        except:
            # Fall back to text parsing
            pass
        
        return []
    
    def _parse_yaml_section(self, content: str) -> Optional[Dict]:
        """Parse a YAML code block section"""
        
        if not content:
            return None
        
        # Extract YAML from code block if present
        yaml_match = re.search(r'```yaml\n(.*?)```', content, re.DOTALL)
        if yaml_match:
            content = yaml_match.group(1)
        
        try:
            return yaml.safe_load(content)
        except:
            return None
    
    def _parse_dependencies(self, deps: Dict) -> Dict[str, List[Dict]]:
        """Parse and validate dependencies"""
        
        if not deps:
            return {}
        
        parsed = {
            'features': [],
            'mcp_servers': [],
            'services': [],
            'packages': [],
            'containers': [],
            'models': []
        }
        
        for dep_type in parsed.keys():
            if dep_type in deps:
                items = deps[dep_type]
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, str):
                            # Simple string dependency
                            parsed[dep_type].append({'name': item})
                        elif isinstance(item, dict):
                            parsed[dep_type].append(item)
        
        return parsed
    
    def _parse_code_examples(self, content: str) -> Optional[List[str]]:
        """Parse example code blocks"""
        
        if not content:
            return None
        
        examples = []
        
        # Extract all code blocks from examples section
        code_blocks = re.findall(r'```\w*\n(.*?)```', content, re.DOTALL)
        for code in code_blocks:
            examples.append(code.strip())
        
        return examples if examples else None
    
    def _parse_setup_section(self, sections: Dict, code_blocks: Dict) -> Optional[Dict]:
        """Parse test setup section"""
        
        if 'setup' not in sections:
            return None
        
        setup = {}
        
        # Check for Gherkin background
        if 'gherkin' in code_blocks:
            setup['gherkin'] = '\n\n'.join(code_blocks['gherkin'])
        
        # Check for Python fixtures
        if 'python' in code_blocks:
            for block in code_blocks['python']:
                if '@pytest.fixture' in block or 'def setup' in block:
                    setup['python'] = block
                    break
        
        return setup if setup else None
    
    def _parse_test_cases(self, sections: Dict, code_blocks: Dict) -> List[Dict]:
        """Parse test cases from sections"""
        
        test_cases = []
        
        # Find test case sections
        for section_name, content in sections.items():
            if 'test:' in section_name.lower() or 'scenario:' in section_name.lower():
                test_case = {
                    'name': section_name,
                    'description': content
                }
                
                # Extract Gherkin scenarios
                gherkin_match = re.search(r'```gherkin\n(.*?)```', content, re.DOTALL)
                if gherkin_match:
                    test_case['gherkin'] = gherkin_match.group(1)
                
                # Extract Python implementation
                python_match = re.search(r'```python\n(.*?)```', content, re.DOTALL)
                if python_match:
                    test_case['python'] = python_match.group(1)
                
                test_cases.append(test_case)
        
        return test_cases
    
    def _validate_feature(self, feature: EOLFeature):
        """Validate feature specification"""
        
        # Check phase requirements
        if feature.phase in [ExecutionPhase.PROTOTYPING, ExecutionPhase.HYBRID]:
            if not feature.prototyping:
                raise ValueError(f"Feature '{feature.name}' requires prototyping section for phase '{feature.phase}'")
        
        if feature.phase in [ExecutionPhase.IMPLEMENTATION, ExecutionPhase.HYBRID]:
            if not feature.implementation:
                raise ValueError(f"Feature '{feature.name}' requires implementation section for phase '{feature.phase}'")
        
        # Validate dependencies
        for dep_type, deps in feature.dependencies.items():
            for dep in deps:
                self._validate_dependency(dep_type, dep)
    
    def _validate_dependency(self, dep_type: str, dep: Dict):
        """Validate individual dependency"""
        
        if dep_type == 'models':
            required_fields = ['name', 'provider', 'purpose']
            for field in required_fields:
                if field not in dep:
                    raise ValueError(f"Model dependency missing required field '{field}': {dep}")
        
        elif dep_type == 'features':
            if 'path' not in dep and 'name' not in dep:
                raise ValueError(f"Feature dependency must have 'path' or 'name': {dep}")
        
        elif dep_type == 'mcp_servers':
            if 'name' not in dep:
                raise ValueError(f"MCP server dependency missing 'name': {dep}")
        
        elif dep_type == 'services':
            if 'name' not in dep:
                raise ValueError(f"Service dependency missing 'name': {dep}")