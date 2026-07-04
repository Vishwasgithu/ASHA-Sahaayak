"""
chunker.py — Document chunking module for ASHA Sahaayak RAG pipeline.
Splits large text pages into smaller overlapping chunks for vector search.
"""

from typing import List
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Resolve python paths to import config and utils correctly
import sys
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

import config
from utils.helpers import get_logger

logger = get_logger("RAG.Chunker")


class DocumentChunker:
    """
    Production-grade text chunker that splits LangChain Document pages
    into semantic chunks using RecursiveCharacterTextSplitter.
    """

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

        # Standard clean separators for plain text / Markdown/ PDF extractions
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len
        )
        logger.info(f"Initialized DocumentChunker (Chunk Size: {self.chunk_size}, Overlap: {self.chunk_overlap})")

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits a list of Document objects into smaller chunks.
        Ensures metadata (source, page, document_name) is correctly maintained and enriched.
        """
        if not documents:
            logger.warning("Empty list of documents provided for splitting.")
            return []

        logger.info(f"Splitting {len(documents)} page documents into chunks...")

        # Perform the split using RecursiveCharacterTextSplitter
        chunks = self.splitter.split_documents(documents)

        # Post-process metadata to ensure the exact fields requested are populated
        for chunk in chunks:
            # Get original metadata values
            source_path = chunk.metadata.get("source", "")
            page_num = chunk.metadata.get("page", 0)
            
            # Map existing file identifiers to "document_name"
            # Prefer 'filename' from loader, fallback to basename of source path, or empty string
            filename = chunk.metadata.get("filename")
            if not filename and source_path:
                filename = Path(source_path).name
            document_name = filename or "Unknown"

            # Reconstruct metadata to match user's explicit schema
            chunk.metadata.update({
                "source": source_path,
                "page": page_num,
                "document_name": document_name
            })

        logger.info(f"Successfully generated {len(chunks)} text chunks (Avg chunks per page: {len(chunks)/len(documents):.2f})")
        return chunks
