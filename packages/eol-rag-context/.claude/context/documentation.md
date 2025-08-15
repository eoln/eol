# Documentation Standards and Guidelines for EOL RAG Context

This document defines the documentation standards, tools, and practices for the EOL RAG Context MCP server project.

## Documentation Philosophy

### Core Principles

1. **Documentation as Code**: Documentation lives alongside source code and follows the same quality standards
2. **Type-First Documentation**: Leverage Python type hints as the foundation for API documentation
3. **Auto-Generation**: Minimize manual documentation maintenance through automated generation from docstrings and type hints
4. **Developer Experience**: Prioritize ease of writing, maintaining, and consuming documentation
5. **Living Documentation**: Documentation automatically stays in sync with code changes

### Target Audiences

- **Developers**: API reference, integration guides, contributing guidelines
- **Users**: Tutorials, examples, configuration guides
- **AI Assistants**: Context for understanding codebase structure and usage patterns

## Documentation Toolchain

### Primary Tools

**MkDocs + mkdocstrings** - Modern documentation generation

- **Why**: Clean, modern sites with excellent Python integration
- **Features**: Live preview, automatic API docs, Material theme
- **Used by**: FastAPI, Typer, Pydantic (similar projects)

**Google-style Docstrings** - Standardized documentation format

- **Why**: Readable, concise, excellent tool support
- **Integration**: Works seamlessly with mkdocstrings and IDEs
- **Benefits**: Less vertical space than NumPy style

### Supporting Tools

- **Type Hints**: Python 3.11+ type annotations for API documentation
- **Pydantic Models**: Schema documentation with automatic validation
- **GitHub Pages**: Automated deployment and hosting
- **Pre-commit Hooks**: Automated docstring validation
- **Black Code Formatter**: Consistent code formatting with docstring preservation

## Docstring Standards

### Format: Google Style

All docstrings MUST follow Google style format:

```python
def function_name(param1: Type1, param2: Type2 = default) -> ReturnType:
    """Brief description of the function.

    Longer description explaining the function's purpose, behavior,
    and any important implementation details.

    Args:
        param1: Description of the first parameter.
        param2: Description of the second parameter with default value.

    Returns:
        Description of the return value and its structure.

    Raises:
        SpecificError: When this specific error occurs.
        AnotherError: When this other error occurs.

    Example:
        >>> result = function_name("value", 42)
        >>> print(result.status)
        'success'
    """
```

### Required Elements

**All Public Functions/Methods MUST have:**

- Brief description (one line)
- Args section (if parameters exist)
- Returns section (if not None)
- Raises section (for expected exceptions)

**Classes MUST have:**

- Purpose and responsibility description
- Attributes section (for public attributes)
- Example usage

**Modules MUST have:**

- Module purpose and contents overview
- Key classes/functions summary

### Type Hints Integration

Combine comprehensive type hints with docstrings:

```python
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from pydantic import BaseModel, Field

class IndexConfig(BaseModel):
    """Configuration for document indexing operations.

    This configuration controls how documents are processed,
    chunked, and indexed in the vector database.

    Attributes:
        chunk_size: Maximum size of each content chunk.
        chunk_overlap: Characters to overlap between adjacent chunks.
        file_patterns: Glob patterns for files to include.
        exclude_patterns: Glob patterns for files to exclude.
    """
    chunk_size: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Maximum chunk size in characters"
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        description="Overlap between adjacent chunks"
    )
    file_patterns: List[str] = Field(
        default=["*.py", "*.md", "*.txt"],
        description="File patterns to include during indexing"
    )
    exclude_patterns: List[str] = Field(
        default=["*.pyc", "__pycache__/*"],
        description="File patterns to exclude during indexing"
    )

async def index_document(
    file_path: Union[str, Path],
    config: Optional[IndexConfig] = None,
    force_reindex: bool = False
) -> IndexResult:
    """Index a document with hierarchical chunking and metadata extraction.

    This function processes a document, extracts its content, and creates
    a hierarchical index with concepts, sections, and chunks. The resulting
    index is stored in Redis for vector similarity search.

    Args:
        file_path: Path to the document file. Supports various formats
            including Markdown, PDF, DOCX, and source code files.
        config: Optional indexing configuration. Uses defaults if not provided.
        force_reindex: Whether to reindex even if document hasn't changed.
            Useful for testing or when indexing logic has been updated.

    Returns:
        IndexResult containing the indexing outcome with statistics:
        - source_id: Unique identifier for the indexed document
        - total_chunks: Number of chunks created
        - hierarchy_levels: Number of hierarchy levels created
        - processing_time: Time taken to complete indexing

    Raises:
        FileNotFoundError: If the specified file doesn't exist.
        DocumentProcessingError: If the document format is unsupported
            or the file is corrupted.
        RedisConnectionError: If unable to connect to Redis database.

    Example:
        Basic usage with default configuration:

        >>> config = IndexConfig(chunk_size=500)
        >>> result = await index_document("docs/tutorial.md", config)
        >>> print(f"Created {result.total_chunks} chunks")
        Created 12 chunks

        Force reindexing an existing document:

        >>> result = await index_document("api.py", force_reindex=True)
        >>> print(f"Reindexed {result.source_id}")
        Reindexed api_py_20241211_143022
    """
```

## Documentation Structure

### Site Organization

```
docs/
├── index.md                 # Project overview and quick start
├── getting-started/
│   ├── installation.md      # Installation instructions
│   ├── configuration.md     # Configuration options
│   └── first-steps.md       # Basic usage tutorial
├── user-guide/
│   ├── indexing.md          # Document indexing guide
│   ├── searching.md         # Search and retrieval guide
│   ├── advanced-features.md # Knowledge graphs, caching, etc.
│   └── integrations.md      # MCP integration examples
├── api-reference/           # Auto-generated from docstrings
│   ├── server.md
│   ├── indexer.md
│   ├── embeddings.md
│   └── ...
├── development/
│   ├── contributing.md      # Development setup and guidelines
│   ├── architecture.md      # System design and architecture
│   ├── testing.md           # Testing procedures and standards
│   └── deployment.md        # Deployment and CI/CD
└── examples/
    ├── basic-usage.md       # Simple integration examples
    ├── advanced-usage.md    # Complex scenarios and patterns
    └── troubleshooting.md   # Common issues and solutions
```

### Content Guidelines

**Overview Pages (index.md, getting-started/)**

- Problem the project solves
- Key benefits and features
- Quick installation and setup
- Simple "hello world" example

**User Guide**

- Task-oriented documentation
- Step-by-step procedures
- Real-world examples
- Configuration options

**API Reference**

- Auto-generated from docstrings
- Comprehensive parameter documentation
- Return type specifications
- Exception handling

**Development Documentation**

- Architecture decisions and rationale
- Contributing guidelines
- Testing standards and procedures
- Deployment and release process

## Automation and Quality

### Pre-commit Hooks

Automated validation of documentation standards:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/PyCQA/docformatter
    rev: v1.7.5
    hooks:
      - id: docformatter
        args: [--in-place, --wrap-summaries=79, --wrap-descriptions=79]

  - repo: https://github.com/pycqa/pydocstyle
    rev: 6.3.0
    hooks:
      - id: pydocstyle
        args: [--convention=google]
```

### CI/CD Integration

Automated documentation building and deployment:

```yaml
# .github/workflows/docs.yml
name: Build and Deploy Documentation

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install mkdocs mkdocs-material mkdocstrings[python] black

      - name: Format code with Black
        run: black --check --diff src/

      - name: Build documentation
        run: mkdocs build

      - name: Deploy to GitHub Pages
        if: github.ref == 'refs/heads/main'
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./site
```

### Quality Metrics

Track documentation quality:

- **Coverage**: Percentage of public APIs with docstrings
- **Completeness**: Required sections (Args, Returns, Raises) present
- **Examples**: Percentage of functions with usage examples
- **Link Validation**: All internal/external links work
- **Freshness**: Documentation updated with code changes

## Migration Strategy

### Phase 1: Foundation (Week 1)

- Setup MkDocs configuration
- Create basic site structure
- Configure automated deployment

### Phase 2: Core Documentation (Week 2)

- Document server.py and main entry points
- Add comprehensive docstrings to public APIs
- Create basic user guide content

### Phase 3: Comprehensive Coverage (Week 3-4)

- Document all modules with Google-style docstrings
- Generate complete API reference
- Add advanced examples and guides

### Phase 4: Automation and Polish (Week 4)

- Setup pre-commit hooks
- Add CI/CD documentation builds
- Implement quality metrics tracking

## Best Practices

### Writing Guidelines

**Clarity Over Brevity**

- Explain the "why" not just the "what"
- Include context about when to use different options
- Provide realistic examples, not toy examples

**Code Examples**

- Always include practical, working examples
- Show both success and error handling cases
- Use consistent example data across documentation

**Maintenance**

- Keep examples up-to-date with API changes
- Test code examples in CI/CD
- Review documentation with each feature addition

### Common Patterns

**Configuration Classes**

```python
class ServiceConfig(BaseModel):
    """Configuration for the RAG service.

    This class centralizes all configuration options for the service,
    providing validation and documentation for each setting.
    """
    # Use Field() with descriptions for Pydantic models
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL"
    )
```

**Error Handling**

```python
def risky_operation() -> Result:
    """Perform operation that may fail.

    Returns:
        Result object with operation outcome.

    Raises:
        ValidationError: If input parameters are invalid.
        ConnectionError: If unable to connect to external service.
        ProcessingError: If operation fails due to data issues.
    """
```

**Async Functions**

```python
async def async_operation(data: InputData) -> OutputData:
    """Process data asynchronously.

    This function handles large datasets by processing them in batches
    to avoid memory issues and provide progress feedback.

    Args:
        data: Input data to process. Must be validated before calling.

    Returns:
        Processed data with additional metadata fields.

    Note:
        This function is CPU-intensive and should be called with
        appropriate concurrency limits to avoid overwhelming the system.
    """
```

## Code Formatting Integration (Black)

### Black Formatter Requirements

The project uses **Black** code formatter to ensure consistent code style across all Python files. This is enforced in the GitHub Actions quality gate and should be used during development.

**Key Black Configuration:**

- **Line length**: 88 characters (Black's default)
- **Target Python version**: 3.11+
- **Docstring preservation**: Black maintains Google-style docstring formatting
- **Multi-line docstring**: Proper indentation and line breaks preserved

### Black and Documentation Compatibility

Black is fully compatible with Google-style docstrings and enhances documentation quality:

**Docstring Formatting:**

```python
def example_function(param1: str, param2: int = 42) -> dict:
    """Example function with properly formatted docstring.

    Black preserves Google-style docstring structure while ensuring
    consistent indentation and spacing throughout the codebase.

    Args:
        param1: Description of the first parameter.
        param2: Description with default value.

    Returns:
        Dictionary containing processing results.

    Example:
        >>> result = example_function("test", 100)
        >>> print(result["status"])
        'success'
    """
    return {"status": "success", "param1": param1, "param2": param2}
```

### Development Workflow with Black

**Local Development:**

```bash
# Format all source files
black src/

# Check formatting without making changes
black --check --diff src/

# Format specific file
black src/eol/rag_context/document_processor.py
```

**IDE Integration:**

- **PyCharm**: Settings → Tools → External Tools → Add Black
- **VS Code**: Install "Black Formatter" extension
- **Vim/Neovim**: Use black plugin or ALE integration

### Quality Gate Integration

Black formatting is enforced in CI/CD pipeline:

```yaml
# GitHub Actions workflow includes:
- name: Format code with Black
  run: black --check --diff src/
```

**Enforcement Rules:**

- All pull requests must pass Black formatting check
- Code that doesn't conform to Black style will fail CI
- Developers should run Black locally before pushing
- Documentation builds only after formatting validation

### Impact on Documentation Generation

Black formatting improves documentation quality by:

**Consistency:**

- Uniform code style in docstring examples
- Consistent indentation for better readability
- Standardized spacing and line breaks

**Readability:**

- Code examples in docs are automatically well-formatted
- Type hints remain readable after formatting
- Multi-line parameter lists are properly aligned

**Maintenance:**

- Less time spent on style discussions during code review
- Automated formatting reduces manual formatting errors
- Consistent style improves AI assistant understanding

## Tools and Resources

### Development Tools

- **IDE Extensions**: Python docstring generators (PyCharm, VSCode)
- **Code Formatting**: Black for consistent Python code formatting
- **Linting**: pydocstyle for docstring validation
- **Formatting**: docformatter for consistent docstring formatting
- **Type Checking**: mypy for type hint validation

### Reference Materials

- [Google Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- [PEP 257 - Docstring Conventions](https://peps.python.org/pep-0257/)
- [mkdocstrings Documentation](https://mkdocstrings.github.io/)
- [MkDocs Material Theme](https://squidfunk.github.io/mkdocs-material/)

### Quality Checklists

**Before Merging Code:**

- [ ] Code is formatted with Black (88-character line limit)
- [ ] All public functions have Google-style docstrings
- [ ] Type hints are comprehensive and accurate
- [ ] Examples are tested and working
- [ ] Documentation builds without warnings
- [ ] Links are valid and working

**Before Release:**

- [ ] User guide reflects new features
- [ ] API reference is complete and accurate
- [ ] Migration guide updated for breaking changes
- [ ] Examples updated to use latest patterns
- [ ] Documentation deployed to production site

This documentation standard ensures that the EOL RAG Context project maintains high-quality, comprehensive documentation that serves both human developers and AI assistants effectively.
