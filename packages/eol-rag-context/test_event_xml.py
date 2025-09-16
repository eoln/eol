#!/usr/bin/env python
"""Test script to verify temporal context preservation in event XML processing."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from eol.rag_context.config import ChunkingConfig, DocumentConfig
from eol.rag_context.document_processor import DocumentProcessor


async def test_event_xml():
    """Test processing of event XML files with temporal context."""

    # Initialize processor
    doc_config = DocumentConfig()
    chunk_config = ChunkingConfig()
    processor = DocumentProcessor(doc_config, chunk_config)

    # Test with a sample event file
    event_file = Path("/Users/eoln/Devel/cjg-data/dolnoslaskie-2025-06/3dqf.xml")

    if not event_file.exists():
        print(f"Error: Test file not found: {event_file}")
        return

    print(f"Processing event XML: {event_file}")
    print("=" * 60)

    # Process the file
    doc = await processor.process_file(event_file)

    print(f"Document type: {doc.doc_type}")
    print(f"Number of chunks: {len(doc.chunks)}")
    print()

    # Display chunks with temporal information
    for i, chunk in enumerate(doc.chunks, 1):
        print(f"Chunk {i}:")
        print(f"  Type: {chunk.get('type', 'unknown')}")
        print(f"  Content length: {len(chunk.get('content', ''))}")

        # Check for temporal metadata
        metadata = chunk.get("metadata", {})
        if "date" in metadata:
            print(f"  ✅ Date preserved: {metadata['date']}")
        else:
            print(f"  ❌ No date found in metadata")

        if "location" in metadata:
            print(f"  ✅ Location preserved: {metadata['location']}")

        if "title" in metadata:
            print(f"  ✅ Title preserved: {metadata['title']}")

        # Show content preview
        content = chunk.get("content", "")
        if "Date/Time:" in content:
            print(f"  ✅ Temporal context in content")

        print(f"  Content preview:")
        print(f"    {content[:200]}...")
        print()

    # Test with another event file to verify consistency
    event_file2 = Path("/Users/eoln/Devel/cjg-data/dolnoslaskie-2025-06/6er9.xml")
    if event_file2.exists():
        print("\nTesting second event file...")
        doc2 = await processor.process_file(event_file2)

        if doc2.chunks:
            chunk = doc2.chunks[0]
            metadata = chunk.get("metadata", {})
            content = chunk.get("content", "")

            print(f"Event type: {doc2.doc_type}")
            print(f"Date in metadata: {metadata.get('date', 'NOT FOUND')}")
            print(f"Date in content: {'Date/Time:' in content}")
            print()


if __name__ == "__main__":
    print("Testing Event XML Processing with Temporal Context")
    print("=" * 60)
    asyncio.run(test_event_xml())
    print("\nTest complete!")
