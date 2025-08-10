"""Context Manager - Manages LLM context window and context selection"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import json
import asyncio
import aiofiles
import yaml


class ContextType(str, Enum):
    """Types of context information"""
    EPISODIC = "episodic"      # Examples of desired behavior
    PROCEDURAL = "procedural"  # Instructions to steer behavior
    SEMANTIC = "semantic"      # Task-relevant facts
    REFERENCE = "reference"    # Documentation and guides
    CODE = "code"             # Source code context


@dataclass
class ContextItem:
    """Represents a single context item"""
    type: ContextType
    content: str
    source: Optional[str] = None
    relevance_score: float = 1.0
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextWindow:
    """Represents the current context window"""
    items: List[ContextItem] = field(default_factory=list)
    total_tokens: int = 0
    max_tokens: int = 100000
    usage_percent: float = 0.0


class ContextManager:
    """Manages context for LLM operations"""
    
    def __init__(self, project_root: Optional[Path] = None, max_tokens: int = 100000):
        self.project_root = project_root or Path.cwd()
        self.context_dir = self.project_root / ".claude" / "context"
        self.claude_md_path = self.project_root / "CLAUDE.md"
        self.max_tokens = max_tokens
        self.window = ContextWindow(max_tokens=max_tokens)
        self.context_cache: Dict[str, ContextItem] = {}
        
    async def initialize(self):
        """Initialize context manager and load base context"""
        
        # Create context directory if it doesn't exist
        self.context_dir.mkdir(parents=True, exist_ok=True)
        
        # Load CLAUDE.md if it exists
        if self.claude_md_path.exists():
            await self.load_claude_md()
        
        # Load context files
        await self.load_context_files()
    
    async def load_claude_md(self):
        """Load CLAUDE.md project rules"""
        
        async with aiofiles.open(self.claude_md_path, 'r') as f:
            content = await f.read()
        
        item = ContextItem(
            type=ContextType.PROCEDURAL,
            content=content,
            source=str(self.claude_md_path),
            relevance_score=1.0,  # Always highly relevant
            token_count=self._estimate_tokens(content),
            metadata={'permanent': True}
        )
        
        self.context_cache['CLAUDE.md'] = item
        await self.add_to_window(item)
    
    async def load_context_files(self):
        """Load all context files from .claude/context/"""
        
        if not self.context_dir.exists():
            return
        
        # Scan for markdown files
        for file_path in self.context_dir.rglob("*.md"):
            relative_path = file_path.relative_to(self.context_dir)
            
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
            
            # Determine context type based on path
            context_type = self._determine_context_type(relative_path)
            
            item = ContextItem(
                type=context_type,
                content=content,
                source=str(file_path),
                relevance_score=0.8,
                token_count=self._estimate_tokens(content),
                metadata={'path': str(relative_path)}
            )
            
            self.context_cache[str(relative_path)] = item
    
    def _determine_context_type(self, path: Path) -> ContextType:
        """Determine context type from file path"""
        
        path_str = str(path).lower()
        
        if 'example' in path_str:
            return ContextType.EPISODIC
        elif 'pattern' in path_str or 'guide' in path_str:
            return ContextType.PROCEDURAL
        elif 'architecture' in path_str or 'spec' in path_str:
            return ContextType.SEMANTIC
        elif 'reference' in path_str or 'doc' in path_str:
            return ContextType.REFERENCE
        else:
            return ContextType.SEMANTIC
    
    async def select_context(self, task: str, required_types: Optional[List[ContextType]] = None) -> List[ContextItem]:
        """Select relevant context for a task"""
        
        selected = []
        
        # Default to all types if not specified
        if not required_types:
            required_types = list(ContextType)
        
        # Select from each required type
        for context_type in required_types:
            type_items = [
                item for item in self.context_cache.values()
                if item.type == context_type
            ]
            
            # Sort by relevance and select top items
            type_items.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Add items that fit in window
            for item in type_items:
                if self._can_fit_in_window(item):
                    selected.append(item)
                    await self.add_to_window(item)
        
        return selected
    
    async def add_to_window(self, item: ContextItem) -> bool:
        """Add context item to window"""
        
        if not self._can_fit_in_window(item):
            # Try to compress window first
            await self.compress_window()
            
            # Check again
            if not self._can_fit_in_window(item):
                return False
        
        self.window.items.append(item)
        self.window.total_tokens += item.token_count
        self.window.usage_percent = (self.window.total_tokens / self.window.max_tokens) * 100
        
        return True
    
    def _can_fit_in_window(self, item: ContextItem) -> bool:
        """Check if item can fit in current window"""
        
        return (self.window.total_tokens + item.token_count) <= self.window.max_tokens
    
    async def compress_window(self):
        """Compress context window when approaching limits"""
        
        if self.window.usage_percent < 80:
            return  # No compression needed
        
        # Remove low-relevance items first
        self.window.items.sort(key=lambda x: (x.metadata.get('permanent', False), x.relevance_score))
        
        while self.window.usage_percent > 75 and self.window.items:
            # Don't remove permanent items
            if self.window.items[0].metadata.get('permanent'):
                break
            
            removed = self.window.items.pop(0)
            self.window.total_tokens -= removed.token_count
            self.window.usage_percent = (self.window.total_tokens / self.window.max_tokens) * 100
        
        # If still too full, try summarization
        if self.window.usage_percent > 90:
            await self._summarize_context()
    
    async def _summarize_context(self):
        """Summarize context items to reduce token count"""
        
        # Group similar items
        grouped = {}
        for item in self.window.items:
            if item.metadata.get('permanent'):
                continue  # Don't summarize permanent items
            
            key = f"{item.type}:{item.source}"
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(item)
        
        # Summarize groups with multiple items
        for key, items in grouped.items():
            if len(items) > 2:
                # Combine content
                combined_content = "\n\n".join([item.content for item in items])
                
                # Create summary (in real implementation, would use LLM)
                summary = f"[Summary of {len(items)} items from {key}]\n{combined_content[:1000]}..."
                
                # Replace items with summary
                summary_item = ContextItem(
                    type=items[0].type,
                    content=summary,
                    source=f"summary:{key}",
                    relevance_score=max(item.relevance_score for item in items),
                    token_count=self._estimate_tokens(summary),
                    metadata={'summary': True, 'original_count': len(items)}
                )
                
                # Remove original items
                for item in items:
                    self.window.items.remove(item)
                    self.window.total_tokens -= item.token_count
                
                # Add summary
                self.window.items.append(summary_item)
                self.window.total_tokens += summary_item.token_count
        
        self.window.usage_percent = (self.window.total_tokens / self.window.max_tokens) * 100
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        
        # Simple estimation: ~4 characters per token
        # In production, use tiktoken or similar
        return len(text) // 4
    
    async def search_context(self, query: str, limit: int = 10) -> List[ContextItem]:
        """Search for relevant context items"""
        
        results = []
        query_lower = query.lower()
        
        for item in self.context_cache.values():
            # Simple keyword matching (in production, use embeddings)
            if query_lower in item.content.lower():
                item.relevance_score = self._calculate_relevance(query, item.content)
                results.append(item)
        
        # Sort by relevance
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return results[:limit]
    
    def _calculate_relevance(self, query: str, content: str) -> float:
        """Calculate relevance score for content"""
        
        # Simple implementation - count keyword matches
        query_words = query.lower().split()
        content_lower = content.lower()
        
        matches = sum(1 for word in query_words if word in content_lower)
        
        return min(matches / len(query_words), 1.0)
    
    async def add_code_context(self, file_paths: List[Path]):
        """Add source code files to context"""
        
        for file_path in file_paths:
            if not file_path.exists():
                continue
            
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
            
            item = ContextItem(
                type=ContextType.CODE,
                content=f"```{file_path.suffix[1:]}\n# {file_path}\n{content}\n```",
                source=str(file_path),
                relevance_score=0.9,
                token_count=self._estimate_tokens(content),
                metadata={'file_type': file_path.suffix}
            )
            
            await self.add_to_window(item)
    
    def get_window_status(self) -> Dict[str, Any]:
        """Get current window status"""
        
        return {
            'total_tokens': self.window.total_tokens,
            'max_tokens': self.window.max_tokens,
            'usage_percent': self.window.usage_percent,
            'item_count': len(self.window.items),
            'items_by_type': self._count_by_type(),
            'permanent_items': sum(1 for item in self.window.items if item.metadata.get('permanent'))
        }
    
    def _count_by_type(self) -> Dict[str, int]:
        """Count items by context type"""
        
        counts = {}
        for item in self.window.items:
            counts[item.type.value] = counts.get(item.type.value, 0) + 1
        return counts
    
    def export_context(self) -> str:
        """Export current context window as formatted text"""
        
        sections = {context_type: [] for context_type in ContextType}
        
        # Group items by type
        for item in self.window.items:
            sections[item.type].append(item)
        
        # Format output
        output = []
        output.append("=" * 80)
        output.append("CONTEXT WINDOW")
        output.append(f"Usage: {self.window.usage_percent:.1f}% ({self.window.total_tokens}/{self.window.max_tokens} tokens)")
        output.append("=" * 80)
        
        for context_type in ContextType:
            items = sections[context_type]
            if items:
                output.append(f"\n## {context_type.value.upper()}")
                output.append("-" * 40)
                
                for item in items:
                    if item.source:
                        output.append(f"### Source: {item.source}")
                    output.append(item.content)
                    output.append("")
        
        return "\n".join(output)
    
    async def save_context(self, path: Path):
        """Save current context to file"""
        
        context_data = {
            'window': {
                'total_tokens': self.window.total_tokens,
                'max_tokens': self.window.max_tokens,
                'usage_percent': self.window.usage_percent
            },
            'items': [
                {
                    'type': item.type.value,
                    'content': item.content,
                    'source': item.source,
                    'relevance_score': item.relevance_score,
                    'token_count': item.token_count,
                    'metadata': item.metadata
                }
                for item in self.window.items
            ]
        }
        
        async with aiofiles.open(path, 'w') as f:
            await f.write(json.dumps(context_data, indent=2))
    
    async def load_context(self, path: Path):
        """Load context from file"""
        
        async with aiofiles.open(path, 'r') as f:
            content = await f.read()
            context_data = json.loads(content)
        
        # Clear current window
        self.window.items.clear()
        self.window.total_tokens = 0
        
        # Load items
        for item_data in context_data['items']:
            item = ContextItem(
                type=ContextType(item_data['type']),
                content=item_data['content'],
                source=item_data.get('source'),
                relevance_score=item_data.get('relevance_score', 1.0),
                token_count=item_data.get('token_count', 0),
                metadata=item_data.get('metadata', {})
            )
            self.window.items.append(item)
            self.window.total_tokens += item.token_count
        
        self.window.usage_percent = (self.window.total_tokens / self.window.max_tokens) * 100