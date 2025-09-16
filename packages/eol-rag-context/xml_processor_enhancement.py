"""
Enhanced XML Processing for EOL RAG Context

This module shows how XML documents should be indexed to preserve structure
and enable semantic search on XML content.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional


class XMLProcessor:
    """Process XML documents for RAG indexing."""

    def process_xml(self, file_path: Path) -> dict:
        """
        Process XML files with structure preservation.

        Key features:
        1. Preserve XML hierarchy
        2. Extract attributes as metadata
        3. Create semantic chunks based on XML elements
        4. Support namespaces
        5. Handle nested structures
        """

        # Parse XML
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Extract document metadata
        metadata = {
            "format": "xml",
            "root_tag": root.tag,
            "namespaces": self._extract_namespaces(root),
            "attributes": root.attrib,
            "element_count": len(root.findall(".//*")),
        }

        # Process content
        content = self._extract_text_content(root)
        structured_data = self._element_to_dict(root)

        # Create chunks based on XML structure
        chunks = self._create_xml_chunks(root, str(file_path))

        return {
            "file_path": file_path,
            "content": content,  # Plain text representation
            "structured_data": structured_data,  # Preserved structure
            "doc_type": "xml",
            "metadata": metadata,
            "chunks": chunks,
        }

    def _extract_namespaces(self, root) -> dict:
        """Extract all namespaces from XML."""
        namespaces = {}
        for elem in root.iter():
            if elem.tag.startswith("{"):
                ns = elem.tag.split("}")[0][1:]
                prefix = elem.tag.split("}")[1]
                if ns not in namespaces:
                    namespaces[ns] = prefix
        return namespaces

    def _extract_text_content(self, element) -> str:
        """Extract all text content from XML element recursively."""
        text_parts = []

        if element.text:
            text_parts.append(element.text.strip())

        for child in element:
            child_text = self._extract_text_content(child)
            if child_text:
                text_parts.append(child_text)

        if element.tail:
            text_parts.append(element.tail.strip())

        return " ".join(filter(None, text_parts))

    def _element_to_dict(self, element) -> dict:
        """Convert XML element to dictionary preserving structure."""
        result = {
            "tag": element.tag,
            "attributes": element.attrib,
            "text": element.text.strip() if element.text else None,
            "children": [],
        }

        for child in element:
            result["children"].append(self._element_to_dict(child))

        return result

    def _create_xml_chunks(self, root, source_path: str) -> List[Dict[str, Any]]:
        """
        Create semantic chunks from XML structure.

        Chunking strategies:
        1. Element-based: Each significant element becomes a chunk
        2. Path-based: Include XPath for precise location
        3. Context-aware: Include parent context in metadata
        4. Size-aware: Combine small elements, split large ones
        """
        chunks = []

        def process_element(elem, path="", parent_context="", depth=0):
            # Build XPath
            current_path = f"{path}/{elem.tag}"

            # Extract element content
            elem_text = self._extract_text_content(elem)

            # Determine if element should be a chunk
            if self._should_chunk_element(elem, elem_text):
                chunk = {
                    "content": elem_text,
                    "type": "xml_element",
                    "chunk_index": len(chunks),
                    "source": source_path,
                    "metadata": {
                        "xpath": current_path,
                        "tag": elem.tag,
                        "attributes": elem.attrib,
                        "depth": depth,
                        "parent_context": parent_context[:200],  # Limited context
                        "has_children": len(elem) > 0,
                    },
                }
                chunks.append(chunk)

                # Update parent context for children
                parent_context = elem_text[:200] if elem_text else ""

            # Process children
            for child in elem:
                process_element(child, current_path, parent_context, depth + 1)

        # Start processing from root
        process_element(root)

        return chunks

    def _should_chunk_element(self, element, text_content: str) -> bool:
        """
        Determine if an XML element should become a chunk.

        Criteria:
        - Has meaningful text content (>50 chars)
        - Is a leaf node or has limited children
        - Represents a semantic unit (paragraph, section, record)
        """
        if not text_content or len(text_content.strip()) < 50:
            return False

        # Common semantic XML elements that should be chunks
        semantic_tags = {
            "paragraph",
            "p",
            "section",
            "article",
            "chapter",
            "description",
            "abstract",
            "summary",
            "content",
            "body",
            "text",
            "note",
            "comment",
            "entry",
            "record",
            "item",
            "row",
            "document",
        }

        tag_name = element.tag.split("}")[-1].lower()  # Handle namespaces

        if tag_name in semantic_tags:
            return True

        # Chunk if it's a leaf with substantial content
        if len(element) == 0 and len(text_content) > 100:
            return True

        # Chunk if it has few children but substantial content
        if len(element) < 5 and len(text_content) > 200:
            return True

        return False


class XMLIndexingStrategy:
    """
    Advanced XML indexing strategies for different XML types.
    """

    @staticmethod
    def index_rss_feed(xml_path: Path) -> List[Dict]:
        """Special handling for RSS/Atom feeds."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        chunks = []

        # Find all items/entries
        for item in root.findall(".//item") + root.findall(".//entry"):
            chunk = {"content": "", "type": "rss_item", "metadata": {}}

            # Extract standard RSS/Atom fields
            title = item.find("title")
            if title is not None:
                chunk["metadata"]["title"] = title.text
                chunk["content"] += f"Title: {title.text}\n"

            description = item.find("description") or item.find("summary")
            if description is not None:
                chunk["metadata"]["description"] = description.text
                chunk["content"] += f"Description: {description.text}\n"

            link = item.find("link")
            if link is not None:
                if link.text:
                    chunk["metadata"]["link"] = link.text
                else:
                    chunk["metadata"]["link"] = link.get("href", "")

            pubDate = item.find("pubDate") or item.find("published")
            if pubDate is not None:
                chunk["metadata"]["date"] = pubDate.text

            chunks.append(chunk)

        return chunks

    @staticmethod
    def index_svg(xml_path: Path) -> Dict:
        """Special handling for SVG files."""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        return {
            "content": f"SVG Image: {root.get('id', 'unnamed')}",
            "type": "svg",
            "metadata": {
                "width": root.get("width"),
                "height": root.get("height"),
                "viewBox": root.get("viewBox"),
                "element_count": len(root.findall(".//*")),
                "has_text": len(root.findall(".//text")) > 0,
            },
        }

    @staticmethod
    def index_config_xml(xml_path: Path) -> List[Dict]:
        """Special handling for configuration XML files."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        chunks = []

        def extract_config_items(elem, path=""):
            current_path = f"{path}/{elem.tag}" if path else elem.tag

            # If leaf node with value
            if not elem and elem.text:
                chunks.append(
                    {
                        "content": f"{current_path} = {elem.text}",
                        "type": "config_value",
                        "metadata": {
                            "path": current_path,
                            "value": elem.text,
                            "attributes": elem.attrib,
                        },
                    }
                )

            # Process children
            for child in elem:
                extract_config_items(child, current_path)

        extract_config_items(root)
        return chunks


# Example: How to integrate into DocumentProcessor
def add_xml_support_to_document_processor():
    """
    Example of how to add XML support to the existing DocumentProcessor.

    Add this to the process_file method in document_processor.py:
    """

    code_snippet = '''
    # In DocumentProcessor.process_file method, add:

    elif suffix in [".xml", ".rss", ".atom", ".svg"]:
        return await self._process_xml(file_path)

    # Add new method:
    async def _process_xml(self, file_path: Path) -> ProcessedDocument:
        """Process XML files with structure preservation."""

        processor = XMLProcessor()
        result = processor.process_xml(file_path)

        # Detect XML type for specialized processing
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for specific XML types
        if '<rss' in content or '<feed' in content:
            # RSS/Atom feed
            chunks = XMLIndexingStrategy.index_rss_feed(file_path)
        elif '<svg' in content:
            # SVG image
            svg_data = XMLIndexingStrategy.index_svg(file_path)
            chunks = [svg_data]
        elif any(tag in content for tag in ['<configuration>', '<config>', '<settings>']):
            # Configuration XML
            chunks = XMLIndexingStrategy.index_config_xml(file_path)
        else:
            # Generic XML
            chunks = result["chunks"]

        return ProcessedDocument(
            file_path=file_path,
            content=result["content"],
            doc_type="xml",
            metadata=result["metadata"],
            chunks=chunks
        )
    '''

    return code_snippet


# Example usage
if __name__ == "__main__":
    # Example XML content
    sample_xml = """
    <document>
        <metadata>
            <title>Sample Document</title>
            <author>John Doe</author>
            <date>2024-01-15</date>
        </metadata>
        <content>
            <section id="intro">
                <heading>Introduction</heading>
                <paragraph>
                    This is an introduction paragraph with substantial content
                    that should be indexed as a separate chunk for semantic search.
                </paragraph>
            </section>
            <section id="body">
                <heading>Main Content</heading>
                <paragraph>
                    Another paragraph with important information that needs
                    to be searchable independently.
                </paragraph>
                <list>
                    <item>First item</item>
                    <item>Second item</item>
                </list>
            </section>
        </content>
    </document>
    """

    # Save sample XML
    xml_path = Path("sample.xml")
    xml_path.write_text(sample_xml)

    # Process it
    processor = XMLProcessor()
    result = processor.process_xml(xml_path)

    print("Extracted Chunks:")
    for i, chunk in enumerate(result["chunks"]):
        print(f"\nChunk {i + 1}:")
        print(f"  Content: {chunk['content'][:100]}...")
        print(f"  XPath: {chunk['metadata']['xpath']}")
        print(f"  Tag: {chunk['metadata']['tag']}")
