"""
Document processor for multiple file formats.
"""

import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import magic
import aiofiles
from bs4 import BeautifulSoup
import markdown
import pypdf
from docx import Document as DocxDocument
import tree_sitter
from tree_sitter import Language, Parser
import logging

from .config import DocumentConfig, ChunkingConfig

logger = logging.getLogger(__name__)


@dataclass
class ProcessedDocument:
    """Processed document with metadata."""
    file_path: Path
    content: str
    doc_type: str  # markdown, code, pdf, docx, text, json
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    language: Optional[str] = None  # For code files


class DocumentProcessor:
    """Process various document formats for RAG indexing."""
    
    def __init__(self, doc_config: DocumentConfig, chunk_config: ChunkingConfig):
        self.doc_config = doc_config
        self.chunk_config = chunk_config
        self.mime = magic.Magic(mime=True)
        self.parsers = self._init_code_parsers()
    
    def _init_code_parsers(self) -> Dict[str, Parser]:
        """Initialize tree-sitter parsers for code."""
        parsers = {}
        
        # Map file extensions to languages
        lang_map = {
            ".py": "python",
            ".js": "javascript", 
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".rs": "rust",
            ".go": "go",
            ".java": "java",
        }
        
        # Load available parsers
        try:
            import tree_sitter_python
            import tree_sitter_javascript
            import tree_sitter_typescript
            import tree_sitter_rust
            import tree_sitter_go
            import tree_sitter_java
            
            languages = {
                "python": Language(tree_sitter_python.language(), "python"),
                "javascript": Language(tree_sitter_javascript.language(), "javascript"),
                "typescript": Language(tree_sitter_typescript.language_typescript(), "typescript"),
                "rust": Language(tree_sitter_rust.language(), "rust"),
                "go": Language(tree_sitter_go.language(), "go"),
                "java": Language(tree_sitter_java.language(), "java"),
            }
            
            for ext, lang in lang_map.items():
                if lang in languages:
                    parser = Parser()
                    parser.set_language(languages[lang])
                    parsers[ext] = parser
        except ImportError as e:
            logger.warning(f"Some tree-sitter languages not available: {e}")
        
        return parsers
    
    async def process_file(self, file_path: Path) -> Optional[ProcessedDocument]:
        """Process a single file based on its type."""
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.doc_config.max_file_size_mb:
            logger.warning(f"File too large ({file_size_mb:.2f}MB): {file_path}")
            return None
        
        # Detect file type
        mime_type = self.mime.from_file(str(file_path))
        suffix = file_path.suffix.lower()
        
        # Route to appropriate processor
        if suffix == ".md":
            return await self._process_markdown(file_path)
        elif suffix == ".pdf":
            return await self._process_pdf(file_path)
        elif suffix in [".docx", ".doc"]:
            return await self._process_docx(file_path)
        elif suffix in [".json", ".yaml", ".yml"]:
            return await self._process_structured(file_path)
        elif suffix in self.parsers:
            return await self._process_code(file_path)
        elif "text" in mime_type:
            return await self._process_text(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_path} ({mime_type})")
            return None
    
    async def _process_markdown(self, file_path: Path) -> ProcessedDocument:
        """Process Markdown files."""
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
        
        # Parse markdown
        html = markdown.markdown(content, extensions=['extra', 'codehilite', 'toc'])
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract structure
        doc = ProcessedDocument(
            file_path=file_path,
            content=content,
            doc_type="markdown",
            metadata={
                "format": "markdown",
                "size": len(content),
                "headers": self._extract_headers(soup),
            }
        )
        
        # Chunk by headers if configured
        if self.chunk_config.markdown_split_headers:
            doc.chunks = self._chunk_markdown_by_headers(content)
        else:
            doc.chunks = self._chunk_text(content)
        
        return doc
    
    def _extract_headers(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract header structure from HTML."""
        headers = []
        for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            headers.append({
                "level": int(h.name[1]),
                "text": h.get_text().strip(),
            })
        return headers
    
    def _chunk_markdown_by_headers(self, content: str) -> List[Dict[str, Any]]:
        """Chunk markdown by header structure."""
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        current_header = None
        
        for line in lines:
            # Check if line is a header
            header_match = re.match(r'^(#{1,6})\s+(.*)', line)
            
            if header_match:
                # Save previous chunk if exists
                if current_chunk:
                    chunks.append({
                        "content": '\n'.join(current_chunk),
                        "header": current_header,
                        "type": "section",
                    })
                
                # Start new chunk
                current_header = header_match.group(2)
                current_chunk = [line]
            else:
                current_chunk.append(line)
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                "content": '\n'.join(current_chunk),
                "header": current_header,
                "type": "section",
            })
        
        return chunks
    
    async def _process_pdf(self, file_path: Path) -> ProcessedDocument:
        """Process PDF files."""
        content_parts = []
        metadata = {"pages": 0, "format": "pdf"}
        
        with open(file_path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            metadata["pages"] = len(reader.pages)
            
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                content_parts.append(text)
                
                # Extract metadata from first page
                if i == 0 and reader.metadata:
                    metadata.update({
                        "title": reader.metadata.get('/Title', ''),
                        "author": reader.metadata.get('/Author', ''),
                        "subject": reader.metadata.get('/Subject', ''),
                    })
        
        content = '\n\n'.join(content_parts)
        
        doc = ProcessedDocument(
            file_path=file_path,
            content=content,
            doc_type="pdf",
            metadata=metadata
        )
        
        # Chunk by pages or semantic boundaries
        doc.chunks = self._chunk_pdf_content(content_parts)
        
        return doc
    
    def _chunk_pdf_content(self, pages: List[str]) -> List[Dict[str, Any]]:
        """Chunk PDF content intelligently."""
        chunks = []
        
        for i, page_content in enumerate(pages):
            # Try to detect paragraphs
            paragraphs = re.split(r'\n\s*\n', page_content)
            
            for para in paragraphs:
                if len(para.strip()) > 50:  # Skip very short paragraphs
                    chunks.append({
                        "content": para.strip(),
                        "page": i + 1,
                        "type": "paragraph",
                    })
        
        return chunks
    
    async def _process_docx(self, file_path: Path) -> ProcessedDocument:
        """Process Word documents."""
        doc = DocxDocument(str(file_path))
        
        content_parts = []
        metadata = {
            "format": "docx",
            "properties": {}
        }
        
        # Extract core properties
        if doc.core_properties:
            metadata["properties"] = {
                "title": doc.core_properties.title or "",
                "author": doc.core_properties.author or "",
                "created": str(doc.core_properties.created) if doc.core_properties.created else "",
            }
        
        # Extract paragraphs
        for para in doc.paragraphs:
            if para.text.strip():
                content_parts.append(para.text)
        
        # Extract tables
        for table in doc.tables:
            table_text = []
            for row in table.rows:
                row_text = [cell.text for cell in row.cells]
                table_text.append(" | ".join(row_text))
            content_parts.append("\n".join(table_text))
        
        content = '\n\n'.join(content_parts)
        
        doc_obj = ProcessedDocument(
            file_path=file_path,
            content=content,
            doc_type="docx",
            metadata=metadata
        )
        
        doc_obj.chunks = self._chunk_text(content)
        
        return doc_obj
    
    async def _process_code(self, file_path: Path) -> ProcessedDocument:
        """Process source code files with AST parsing."""
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
        
        suffix = file_path.suffix.lower()
        language = self._detect_language(suffix)
        
        doc = ProcessedDocument(
            file_path=file_path,
            content=content,
            doc_type="code",
            language=language,
            metadata={
                "format": "code",
                "language": language,
                "lines": len(content.splitlines()),
            }
        )
        
        # Parse with tree-sitter if available
        if suffix in self.parsers and self.chunk_config.code_chunk_by_function:
            doc.chunks = self._chunk_code_by_ast(content, self.parsers[suffix], language)
        else:
            doc.chunks = self._chunk_code_by_lines(content, language)
        
        return doc
    
    def _detect_language(self, suffix: str) -> str:
        """Detect programming language from file extension."""
        lang_map = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript", 
            ".tsx": "typescript",
            ".rs": "rust",
            ".go": "go",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".cs": "csharp",
            ".rb": "ruby",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
        }
        return lang_map.get(suffix, "unknown")
    
    def _chunk_code_by_ast(self, content: str, parser: Parser, language: str) -> List[Dict[str, Any]]:
        """Chunk code using AST to preserve function/class boundaries."""
        chunks = []
        tree = parser.parse(bytes(content, "utf8"))
        
        # Define node types to chunk by
        chunk_node_types = {
            "python": ["function_definition", "class_definition"],
            "javascript": ["function_declaration", "class_declaration", "arrow_function"],
            "typescript": ["function_declaration", "class_declaration", "arrow_function"],
            "rust": ["function_item", "impl_item"],
            "go": ["function_declaration", "method_declaration"],
            "java": ["method_declaration", "class_declaration"],
        }
        
        target_types = chunk_node_types.get(language, [])
        
        def extract_nodes(node, depth=0):
            """Recursively extract target nodes."""
            if node.type in target_types:
                start = node.start_byte
                end = node.end_byte
                chunk_content = content[start:end].decode('utf-8') if isinstance(content[start:end], bytes) else content[start:end]
                
                chunks.append({
                    "content": chunk_content,
                    "type": node.type,
                    "language": language,
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                })
            
            for child in node.children:
                extract_nodes(child, depth + 1)
        
        extract_nodes(tree.root_node)
        
        # If no functions found, fall back to line-based chunking
        if not chunks:
            return self._chunk_code_by_lines(content, language)
        
        return chunks
    
    def _chunk_code_by_lines(self, content: str, language: str) -> List[Dict[str, Any]]:
        """Chunk code by lines with overlap."""
        chunks = []
        lines = content.splitlines()
        
        chunk_size = self.chunk_config.code_max_lines
        overlap = min(10, chunk_size // 4)
        
        for i in range(0, len(lines), chunk_size - overlap):
            chunk_lines = lines[i:i + chunk_size]
            if chunk_lines:
                chunks.append({
                    "content": '\n'.join(chunk_lines),
                    "type": "lines",
                    "language": language,
                    "start_line": i + 1,
                    "end_line": min(i + chunk_size, len(lines)),
                })
        
        return chunks
    
    async def _process_structured(self, file_path: Path) -> ProcessedDocument:
        """Process JSON/YAML files."""
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
        
        suffix = file_path.suffix.lower()
        
        # Parse structured data
        if suffix == ".json":
            import json
            data = json.loads(content)
            doc_type = "json"
        else:
            import yaml
            data = yaml.safe_load(content)
            doc_type = "yaml"
        
        doc = ProcessedDocument(
            file_path=file_path,
            content=content,
            doc_type=doc_type,
            metadata={
                "format": doc_type,
                "keys": list(data.keys()) if isinstance(data, dict) else [],
                "type": type(data).__name__,
            }
        )
        
        # Chunk by top-level keys or array items
        doc.chunks = self._chunk_structured_data(data, doc_type)
        
        return doc
    
    def _chunk_structured_data(self, data: Any, format: str) -> List[Dict[str, Any]]:
        """Chunk structured data intelligently."""
        chunks = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                chunk_content = json.dumps({key: value}, indent=2) if format == "json" else str({key: value})
                chunks.append({
                    "content": chunk_content,
                    "key": key,
                    "type": "object_field",
                })
        elif isinstance(data, list):
            for i, item in enumerate(data):
                chunk_content = json.dumps(item, indent=2) if format == "json" else str(item)
                chunks.append({
                    "content": chunk_content,
                    "index": i,
                    "type": "array_item",
                })
        else:
            # Single value
            chunks.append({
                "content": str(data),
                "type": "value",
            })
        
        return chunks
    
    async def _process_text(self, file_path: Path) -> ProcessedDocument:
        """Process plain text files."""
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
        
        doc = ProcessedDocument(
            file_path=file_path,
            content=content,
            doc_type="text",
            metadata={
                "format": "text",
                "size": len(content),
                "lines": len(content.splitlines()),
            }
        )
        
        doc.chunks = self._chunk_text(content)
        
        return doc
    
    def _chunk_text(self, content: str) -> List[Dict[str, Any]]:
        """Chunk plain text with semantic boundaries."""
        chunks = []
        
        if self.chunk_config.use_semantic_chunking:
            # Try to split by paragraphs first
            paragraphs = re.split(r'\n\s*\n', content)
            
            current_chunk = []
            current_size = 0
            
            for para in paragraphs:
                para_size = len(para.split())
                
                if current_size + para_size > self.chunk_config.max_chunk_size:
                    if current_chunk:
                        chunks.append({
                            "content": '\n\n'.join(current_chunk),
                            "type": "semantic",
                            "tokens": current_size,
                        })
                    current_chunk = [para]
                    current_size = para_size
                else:
                    current_chunk.append(para)
                    current_size += para_size
            
            # Add final chunk
            if current_chunk:
                chunks.append({
                    "content": '\n\n'.join(current_chunk),
                    "type": "semantic",
                    "tokens": current_size,
                })
        else:
            # Simple token-based chunking
            words = content.split()
            chunk_size = self.chunk_config.max_chunk_size
            overlap = self.chunk_config.chunk_overlap
            
            for i in range(0, len(words), chunk_size - overlap):
                chunk_words = words[i:i + chunk_size]
                if chunk_words:
                    chunks.append({
                        "content": ' '.join(chunk_words),
                        "type": "token",
                        "tokens": len(chunk_words),
                    })
        
        return chunks
    
    async def process_directory(self, directory: Path) -> List[ProcessedDocument]:
        """Process all matching files in a directory."""
        documents = []
        
        for pattern in self.doc_config.file_patterns:
            for file_path in directory.rglob(pattern):
                if file_path.is_file():
                    try:
                        doc = await self.process_file(file_path)
                        if doc:
                            documents.append(doc)
                            logger.info(f"Processed {file_path}")
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")
        
        return documents