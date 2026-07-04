"""
rag package initialization.
Exposes loader and chunker classes and helpers.
"""

from .loader import PDFDocumentLoader
from .chunker import DocumentChunker
from .retriever import ChromaRetriever, retrieve_documents
from .generator import (
    ClinicalResponseGenerator,
    INSUFFICIENT_EVIDENCE_RESPONSE,
    generate_recommendations,
)
from .pipeline import IndexingResult, RAGPipeline, run_pipeline

def load_kb_documents():
    """
    Convenience function to load all PDFs from the configured knowledge base directories.
    """
    loader = PDFDocumentLoader()
    return loader.load_all_documents()

def chunk_documents(documents):
    """
    Convenience function to chunk loaded documents.
    """
    chunker = DocumentChunker()
    return chunker.split_documents(documents)

__all__ = [
    "PDFDocumentLoader",
    "load_kb_documents",
    "DocumentChunker",
    "chunk_documents",
    "ChromaRetriever",
    "retrieve_documents",
    "ClinicalResponseGenerator",
    "INSUFFICIENT_EVIDENCE_RESPONSE",
    "generate_recommendations",
    "IndexingResult",
    "RAGPipeline",
    "run_pipeline",
]
