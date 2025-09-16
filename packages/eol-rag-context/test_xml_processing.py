#!/usr/bin/env python
import sys

sys.path.insert(0, "/Users/eoln/Devel/eol/packages/eol-rag-context/src")

import asyncio
from pathlib import Path

from eol.rag_context.config import ChunkingConfig, DocumentConfig
from eol.rag_context.document_processor import DocumentProcessor


async def test():
    doc_config = DocumentConfig()
    chunk_config = ChunkingConfig()
    processor = DocumentProcessor(doc_config, chunk_config)

    # Test with a few XML files
    test_files = [
        "/Users/eoln/Devel/cjg-data/dolnoslaskie-2025-06/7wd7.xml",
        "/Users/eoln/Devel/cjg-data/dolnoslaskie-2025-06/3dqf.xml",
        "/Users/eoln/Devel/cjg-data/dolnoslaskie-2025-06/7p4f.xml",
    ]

    for file_str in test_files[:3]:
        file_path = Path(file_str)
        if file_path.exists():
            print(f"\n{'='*60}")
            print(f"Processing: {file_path.name}")
            print("=" * 60)

            result = await processor.process_file(file_path)
            print(f"File type: {result.doc_type}")
            print(f"Chunks created: {len(result.chunks)}")

            if result.chunks:
                for i, chunk in enumerate(result.chunks[:2]):
                    print(f"\nChunk {i}:")
                    print(f"  Content preview: {chunk.get('content', '')[:200]}...")


asyncio.run(test())
