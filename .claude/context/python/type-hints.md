# Type Hints and Annotations

## Basic Type Hints

### Function Signatures
```python
from typing import List, Dict, Optional, Union, Tuple, Any
from typing import Callable, Awaitable, TypeVar, Generic

async def process_text(
    text: str,
    max_length: int = 1000,
    metadata: Optional[Dict[str, Any]] = None
) -> Tuple[str, Dict[str, Any]]:
    """Process text with optional metadata"""
    processed = text[:max_length]
    meta = metadata or {}
    return processed, meta
```

## Advanced Typing

### TypedDict for Structured Data
```python
from typing import TypedDict, NotRequired

class DocumentMetadata(TypedDict):
    source: str
    file_type: str
    created_at: str
    updated_at: NotRequired[str]
    tags: NotRequired[List[str]]

def process_document(metadata: DocumentMetadata) -> None:
    # Type checker knows the structure
    source = metadata["source"]  # OK
    tags = metadata.get("tags", [])  # OK
```

### Protocol for Duck Typing
```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Embeddable(Protocol):
    """Protocol for objects that can be embedded"""
    
    async def to_embedding(self) -> List[float]:
        ...
    
    def get_content(self) -> str:
        ...

async def embed_item(item: Embeddable) -> List[float]:
    """Works with any object following Embeddable protocol"""
    return await item.to_embedding()
```

## Generics

### Generic Classes
```python
T = TypeVar('T')

class Cache(Generic[T]):
    def __init__(self) -> None:
        self._items: Dict[str, T] = {}
    
    async def get(self, key: str) -> Optional[T]:
        return self._items.get(key)
    
    async def set(self, key: str, value: T) -> None:
        self._items[key] = value

# Usage with specific types
string_cache: Cache[str] = Cache()
doc_cache: Cache[Document] = Cache()
```

## Type Aliases

### Complex Type Definitions
```python
from typing import Alias

# Type aliases for clarity
EmbeddingVector = List[float]
DocumentID = str
Score = float
SearchResult = Tuple[DocumentID, Score, Dict[str, Any]]

# New syntax (Python 3.12+)
type QueryResult = List[SearchResult]

async def search(
    query: str,
    k: int = 5
) -> QueryResult:
    """Search with clear type hints"""
    pass
```

## Literal Types and Unions

```python
from typing import Literal, Union

ChunkStrategy = Literal["semantic", "fixed", "ast"]
ModelProvider = Literal["openai", "anthropic", "local"]

class Config:
    chunk_strategy: ChunkStrategy
    model: Union[ModelProvider, str]  # Known providers or custom
    
    def __init__(
        self,
        chunk_strategy: ChunkStrategy = "semantic",
        model: ModelProvider = "openai"
    ):
        self.chunk_strategy = chunk_strategy
        self.model = model
```

## Mypy Configuration

### pyproject.toml
```toml
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_any_unimported = false
no_implicit_optional = true
check_untyped_defs = true
strict_optional = true
strict_equality = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false
```

## Common Patterns

### Optional vs Union[T, None]
```python
# Preferred: use Optional for clarity
def get_value(key: str) -> Optional[str]:
    ...

# Equivalent but less clear
def get_value(key: str) -> Union[str, None]:
    ...
```

### Callable Types
```python
from typing import Callable, Awaitable

# Sync callback
ProcessCallback = Callable[[Document], bool]

# Async callback
AsyncProcessCallback = Callable[[Document], Awaitable[bool]]

async def process_with_callback(
    doc: Document,
    callback: AsyncProcessCallback
) -> bool:
    return await callback(doc)
```

## Best Practices
1. Use type hints for all public APIs
2. Use `Optional` for nullable values
3. Prefer `TypedDict` over raw dicts
4. Use `Protocol` for duck typing
5. Create type aliases for complex types
6. Run mypy in strict mode
7. Document generic type parameters