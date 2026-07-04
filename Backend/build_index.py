"""Build or synchronize the persistent ChromaDB knowledge-base index."""

from __future__ import annotations

import sys

from rag.pipeline import PipelineIndexingError, RAGPipeline


def main() -> int:
    """Build the index and print a concise indexing report."""
    try:
        result = RAGPipeline().build_index()
    except PipelineIndexingError as exc:
        print(f"Index build failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Index build failed unexpectedly: {exc}", file=sys.stderr)
        return 1

    print("RAG index build complete")
    print(f"Loaded pages: {result.loaded_pages}")
    print(f"Generated chunks: {result.generated_chunks}")
    print(f"Stored vectors: {result.stored_chunks}")
    print(f"Collection name: {result.collection_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
