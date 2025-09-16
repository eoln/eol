# Document Processing Patterns

## Chunking Strategies

### Content-Aware Chunking

```python
# For code files
def chunk_code(content: str, language: str) -> List[Chunk]:
    """Use AST-based chunking for code files"""
    # Parse AST
    # Extract functions/classes as chunks
    # Preserve context with imports

# For text documents
def chunk_text(content: str) -> List[Chunk]:
    """Semantic paragraph chunking for text"""
    # Split by paragraphs
    # Group related paragraphs
    # Maintain semantic boundaries
```

### Metadata Extraction

- Extract file type, language, timestamps
- Capture document structure (headers, sections)
- Preserve relationships between chunks
- Include source information

## Hierarchical Indexing

```
Document
├── Concepts (high-level abstractions)
├── Sections (medium-level divisions)
└── Chunks (detailed content pieces)
```

## Best Practices

1. Never split mid-sentence or mid-function
2. Include overlap for context preservation
3. Size chunks based on embedding model limits
4. Add metadata for filtering and ranking
5. Preserve code syntax validity in chunks
