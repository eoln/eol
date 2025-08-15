# LLM Data Patterns: Chunking, Indexing, and Retrieval

## Overview

Comprehensive chunking strategies for different content types, optimized for LLM processing and retrieval in RAG systems.

## Content-Specific Chunking Strategies

### 1. Source Code Chunking

#### AST-Based Chunking

```python
import ast
from tree_sitter import Language, Parser

class ASTChunker:
    """Abstract Syntax Tree based code chunking"""

    def __init__(self, language="python"):
        self.language = language
        self.parser = self.setup_parser(language)

    def chunk_code(self, code):
        """Parse code into logical units using AST"""
        if self.language == "python":
            return self.chunk_python(code)
        else:
            return self.chunk_with_treesitter(code)

    def chunk_python(self, code):
        """Python-specific AST chunking"""
        tree = ast.parse(code)
        chunks = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                chunk = {
                    "type": node.__class__.__name__,
                    "name": node.name,
                    "code": ast.unparse(node),
                    "line_start": node.lineno,
                    "docstring": ast.get_docstring(node),
                    "dependencies": self.extract_dependencies(node)
                }
                chunks.append(chunk)

        return chunks

    def chunk_with_treesitter(self, code):
        """Language-agnostic chunking with tree-sitter"""
        tree = self.parser.parse(bytes(code, "utf8"))
        chunks = []

        # Define queries for different languages
        queries = {
            "javascript": "(function_declaration) @func",
            "typescript": "(function_declaration) @func (class_declaration) @class",
            "java": "(method_declaration) @method (class_declaration) @class"
        }

        query = queries.get(self.language)
        captures = self.execute_query(tree, query)

        for node, name in captures:
            chunks.append({
                "type": name,
                "code": code[node.start_byte:node.end_byte],
                "start": node.start_point,
                "end": node.end_point
            })

        return chunks
```

#### Advantages of AST Chunking

- **Preserves logical boundaries**: Never splits functions/classes
- **Maintains context**: Includes imports and dependencies
- **Language-aware**: Respects syntax rules
- **Executable chunks**: Each chunk can potentially run independently

### 2. Document Structure Chunking

#### Markdown/Documentation

```python
class MarkdownChunker:
    def chunk_by_headers(self, markdown_text):
        """Split markdown by header hierarchy"""
        import re

        chunks = []
        current_chunk = {"headers": [], "content": []}

        for line in markdown_text.split('\n'):
            # Detect headers
            header_match = re.match(r'^(#+)\s+(.+)$', line)

            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2)

                # Start new chunk for major headers
                if level <= 2 and current_chunk["content"]:
                    chunks.append(self.finalize_chunk(current_chunk))
                    current_chunk = {"headers": [], "content": []}

                current_chunk["headers"].append({
                    "level": level,
                    "title": title
                })
            else:
                current_chunk["content"].append(line)

        if current_chunk["content"]:
            chunks.append(self.finalize_chunk(current_chunk))

        return chunks
```

#### HTML/XML Structured Documents

```python
from bs4 import BeautifulSoup

class HTMLChunker:
    def chunk_by_structure(self, html):
        """Chunk HTML preserving structure"""
        soup = BeautifulSoup(html, 'html.parser')
        chunks = []

        # Define semantic sections
        semantic_tags = ['article', 'section', 'div', 'main']

        for tag in semantic_tags:
            sections = soup.find_all(tag)

            for section in sections:
                # Check for meaningful content
                text_content = section.get_text(strip=True)
                if len(text_content) > 50:  # Min content threshold

                    chunk = {
                        "type": "html_section",
                        "tag": tag,
                        "content": text_content,
                        "html": str(section),
                        "metadata": {
                            "id": section.get('id'),
                            "class": section.get('class'),
                            "data_attrs": {
                                k: v for k, v in section.attrs.items()
                                if k.startswith('data-')
                            }
                        }
                    }
                    chunks.append(chunk)

        return chunks
```

#### JSON/Structured Data

```python
class JSONChunker:
    def chunk_nested_json(self, json_data, max_size=1000):
        """Chunk nested JSON structures"""
        chunks = []

        def traverse(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key

                    # Check if value is large enough to chunk
                    if self.estimate_size(value) > max_size:
                        traverse(value, new_path)
                    else:
                        chunks.append({
                            "path": new_path,
                            "data": value,
                            "type": type(value).__name__
                        })

            elif isinstance(obj, list):
                # Chunk large arrays
                if len(obj) > 100:
                    for i in range(0, len(obj), 100):
                        chunks.append({
                            "path": f"{path}[{i}:{i+100}]",
                            "data": obj[i:i+100],
                            "type": "array_slice"
                        })
                else:
                    chunks.append({
                        "path": path,
                        "data": obj,
                        "type": "array"
                    })

        traverse(json_data)
        return chunks
```

### 3. Table and CSV Chunking

```python
import pandas as pd

class TableChunker:
    def chunk_table(self, df, strategy="rows", chunk_size=100):
        """Chunk tables preserving structure"""
        chunks = []

        if strategy == "rows":
            # Row-based chunking
            for i in range(0, len(df), chunk_size):
                chunk_df = df.iloc[i:i+chunk_size]
                chunks.append({
                    "type": "table_rows",
                    "data": chunk_df,
                    "text": chunk_df.to_string(),
                    "metadata": {
                        "columns": list(df.columns),
                        "row_range": (i, min(i+chunk_size, len(df))),
                        "shape": chunk_df.shape
                    }
                })

        elif strategy == "columns":
            # Column-based chunking for wide tables
            col_groups = self.group_related_columns(df)
            for group_name, cols in col_groups.items():
                chunk_df = df[cols]
                chunks.append({
                    "type": "table_columns",
                    "data": chunk_df,
                    "text": self.format_column_chunk(chunk_df),
                    "metadata": {
                        "column_group": group_name,
                        "columns": cols
                    }
                })

        elif strategy == "semantic":
            # Group by semantic meaning
            chunks = self.semantic_table_chunking(df)

        return chunks

    def format_column_chunk(self, df):
        """Format column chunk for LLM understanding"""
        description = f"Table with columns: {', '.join(df.columns)}\n"
        description += f"Shape: {df.shape}\n"
        description += f"Sample data:\n{df.head(5).to_string()}\n"
        description += f"Statistics:\n{df.describe().to_string()}"
        return description
```

### 4. Multimodal Content Chunking

```python
class MultimodalChunker:
    def chunk_pdf_with_images(self, pdf_path):
        """Chunk PDF preserving text-image relationships"""
        import fitz  # PyMuPDF

        chunks = []
        doc = fitz.open(pdf_path)

        for page_num, page in enumerate(doc):
            # Extract text blocks
            text_blocks = page.get_text("blocks")

            # Extract images
            image_list = page.get_images()

            # Group related content
            page_chunks = self.group_page_content(
                text_blocks,
                image_list,
                page_num
            )

            for chunk in page_chunks:
                # Add visual context for images
                if chunk["type"] == "figure":
                    chunk["visual_description"] = self.describe_image(
                        chunk["image_data"]
                    )

                chunks.append(chunk)

        return chunks

    def chunk_video_transcript(self, transcript, timestamps):
        """Chunk video transcripts with temporal alignment"""
        chunks = []

        # Group by scene changes or speaker turns
        scenes = self.detect_scene_changes(timestamps)

        for scene in scenes:
            chunk = {
                "type": "video_segment",
                "start_time": scene["start"],
                "end_time": scene["end"],
                "transcript": self.extract_transcript_segment(
                    transcript,
                    scene["start"],
                    scene["end"]
                ),
                "metadata": {
                    "duration": scene["end"] - scene["start"],
                    "speakers": scene.get("speakers", []),
                    "topics": self.extract_topics(scene["transcript"])
                }
            }
            chunks.append(chunk)

        return chunks
```

### 5. Semantic Chunking

```python
class SemanticChunker:
    def __init__(self, embedder):
        self.embedder = embedder
        self.similarity_threshold = 0.7

    def chunk_by_semantic_similarity(self, text):
        """Split text based on semantic coherence"""
        sentences = self.split_sentences(text)
        chunks = []
        current_chunk = []
        current_embedding = None

        for sentence in sentences:
            sent_embedding = self.embedder.encode(sentence)

            if current_embedding is None:
                current_embedding = sent_embedding
                current_chunk = [sentence]
            else:
                # Calculate similarity
                similarity = self.cosine_similarity(
                    current_embedding,
                    sent_embedding
                )

                if similarity >= self.similarity_threshold:
                    # Add to current chunk
                    current_chunk.append(sentence)
                    # Update embedding (rolling average)
                    current_embedding = self.update_embedding(
                        current_embedding,
                        sent_embedding
                    )
                else:
                    # Start new chunk
                    chunks.append({
                        "text": " ".join(current_chunk),
                        "embedding": current_embedding,
                        "sentence_count": len(current_chunk)
                    })
                    current_chunk = [sentence]
                    current_embedding = sent_embedding

        # Add last chunk
        if current_chunk:
            chunks.append({
                "text": " ".join(current_chunk),
                "embedding": current_embedding,
                "sentence_count": len(current_chunk)
            })

        return chunks
```

## Advanced Chunking Patterns

### 1. Hybrid Chunking

```python
class HybridChunker:
    """Combines multiple chunking strategies"""

    def chunk_document(self, document):
        # First pass: Structure-based chunking
        structural_chunks = self.structural_chunk(document)

        # Second pass: Semantic refinement
        refined_chunks = []
        for chunk in structural_chunks:
            if self.is_too_large(chunk):
                # Apply semantic chunking to large chunks
                sub_chunks = self.semantic_chunk(chunk["content"])
                refined_chunks.extend(sub_chunks)
            elif self.is_too_small(chunk):
                # Merge with adjacent chunks
                refined_chunks = self.merge_small_chunks(
                    refined_chunks,
                    chunk
                )
            else:
                refined_chunks.append(chunk)

        # Third pass: Add overlap for context
        final_chunks = self.add_contextual_overlap(refined_chunks)

        return final_chunks
```

### 2. Recursive Chunking

```python
class RecursiveChunker:
    def __init__(self, separators=None):
        self.separators = separators or [
            "\n\n",  # Paragraphs
            "\n",    # Lines
            ". ",    # Sentences
            " ",     # Words
            ""       # Characters
        ]

    def chunk_recursively(self, text, max_size=1000):
        """Recursively split text using multiple separators"""
        chunks = []

        def split(text, sep_index=0):
            if len(text) <= max_size or sep_index >= len(self.separators):
                return [text]

            separator = self.separators[sep_index]
            parts = text.split(separator)

            current_chunk = []
            current_size = 0

            for part in parts:
                part_size = len(part)

                if current_size + part_size > max_size:
                    if current_chunk:
                        chunks.append(separator.join(current_chunk))
                    current_chunk = [part]
                    current_size = part_size
                else:
                    current_chunk.append(part)
                    current_size += part_size + len(separator)

            if current_chunk:
                final_text = separator.join(current_chunk)
                if len(final_text) > max_size:
                    # Recursively split with next separator
                    sub_chunks = split(final_text, sep_index + 1)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(final_text)

        split(text)
        return chunks
```

### 3. Context-Preserving Chunking

```python
class ContextPreservingChunker:
    def chunk_with_overlap(self, text, chunk_size=1000, overlap=200):
        """Add overlap between chunks for context preservation"""
        chunks = []
        sentences = self.split_sentences(text)

        current_chunk = []
        current_size = 0

        for i, sentence in enumerate(sentences):
            current_chunk.append(sentence)
            current_size += len(sentence)

            if current_size >= chunk_size:
                # Create chunk with metadata
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "index": len(chunks),
                    "sentence_range": (
                        i - len(current_chunk) + 1,
                        i + 1
                    )
                })

                # Keep overlap for next chunk
                overlap_sentences = self.calculate_overlap(
                    current_chunk,
                    overlap
                )
                current_chunk = overlap_sentences
                current_size = sum(len(s) for s in overlap_sentences)

        # Add final chunk
        if current_chunk:
            chunks.append({
                "text": " ".join(current_chunk),
                "index": len(chunks),
                "sentence_range": (
                    len(sentences) - len(current_chunk),
                    len(sentences)
                )
            })

        return chunks
```

## Redis Storage Patterns

```python
class ChunkStorage:
    def __init__(self, redis_client):
        self.redis = redis_client

    async def store_chunks(self, doc_id, chunks, chunk_type):
        """Store chunks with metadata in Redis"""

        for i, chunk in enumerate(chunks):
            chunk_key = f"chunk:{doc_id}:{chunk_type}:{i}"

            # Generate embedding
            if "embedding" not in chunk:
                chunk["embedding"] = await self.embed(chunk["text"])

            # Store chunk data
            await self.redis.hset(chunk_key, {
                "content": chunk.get("text", chunk.get("content")),
                "embedding": chunk["embedding"].tobytes(),
                "type": chunk_type,
                "doc_id": doc_id,
                "index": i,
                "metadata": json.dumps(chunk.get("metadata", {})),
                "created_at": time.time()
            })

            # Add to document's chunk list
            await self.redis.sadd(f"doc:{doc_id}:chunks", chunk_key)

    async def retrieve_chunks(self, query, chunk_types=None, k=10):
        """Retrieve relevant chunks by similarity"""

        query_embedding = await self.embed(query)

        # Build search query
        search_query = Query(
            "*=>[KNN {} @embedding $vec AS score]".format(k)
        ).dialect(2)

        if chunk_types:
            type_filter = " | ".join(
                f"@type:{t}" for t in chunk_types
            )
            search_query.add_filter(type_filter)

        results = await self.redis.ft().search(
            search_query,
            query_params={"vec": query_embedding.tobytes()}
        )

        return [self.parse_chunk(doc) for doc in results.docs]
```

## Best Practices

### 1. Choose Strategy by Content Type

- **Code**: AST-based chunking
- **Documentation**: Header-based hierarchical chunking
- **Data**: Structure-preserving chunking
- **Mixed content**: Hybrid approach

### 2. Optimize Chunk Size

```python
def determine_optimal_chunk_size(content_type, model_context_window=4096):
    """Determine optimal chunk size based on content and model"""

    sizes = {
        "code": min(1500, model_context_window // 4),
        "documentation": min(1000, model_context_window // 4),
        "data": min(500, model_context_window // 8),
        "conversation": min(2000, model_context_window // 2)
    }

    return sizes.get(content_type, 1000)
```

### 3. Maintain Context

- Add overlap between chunks (10-20%)
- Include metadata about position
- Preserve references and links
- Keep parent-child relationships

### 4. Quality Validation

```python
def validate_chunks(chunks):
    """Validate chunk quality"""

    issues = []

    for chunk in chunks:
        # Check size
        if len(chunk["text"]) < 50:
            issues.append(f"Chunk too small: {chunk['index']}")

        # Check coherence
        if not chunk["text"].strip():
            issues.append(f"Empty chunk: {chunk['index']}")

        # Check for broken elements
        if chunk["type"] == "code":
            if not is_valid_syntax(chunk["text"]):
                issues.append(f"Invalid code syntax: {chunk['index']}")

    return issues
```

## EOL Integration

```yaml
# chunking-config.eol
name: intelligent-chunking
phase: implementation

chunking:
  strategies:
    - type: ast
      languages: [python, javascript, typescript]
    - type: semantic
      threshold: 0.7
    - type: structural
      formats: [markdown, html, json]

  settings:
    max_chunk_size: 1000
    overlap: 200
    min_chunk_size: 100

  storage:
    backend: redis
    index: true
    embeddings: true
```
