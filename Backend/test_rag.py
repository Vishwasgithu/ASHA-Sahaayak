"""Run one manual query against the configured RAG pipeline."""

from __future__ import annotations

import json
import sys

from rag.pipeline import PipelineQueryError, RAGPipeline


SAMPLE_QUERY = (
    "Patient is 8 months pregnant with severe headache, swelling and "
    "blurred vision."
)


def main() -> int:
    """Query the existing index and print the grounded response."""
    try:
        result = RAGPipeline().query(SAMPLE_QUERY)
    except PipelineQueryError as exc:
        print(f"RAG test failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"RAG test failed unexpectedly: {exc}", file=sys.stderr)
        return 1

    print(f"Query: {SAMPLE_QUERY}")
    print("\nAnswer:")
    print(result["answer"])
    print("\nRetrieved sources:")
    print(
        json.dumps(
            result["retrieved_sources"],
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )
    print("\nMetadata:")
    print(json.dumps(result["metadata"], indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
