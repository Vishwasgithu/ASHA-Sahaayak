"""
loader.py — Document loading module for ASHA Sahaayak RAG pipeline.
Loads PDF documents from the centralized knowledge base directories.
"""

import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

# Resolve python paths to import config and utils correctly
import sys
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

import config
from utils.helpers import get_logger

logger = get_logger("RAG.Loader")


class PDFDocumentLoader:
    """
    Production-grade document loader to scan knowledge base directories
    and extract contents from PDF files.
    """

    def __init__(self, knowledge_base_dir: Path = None):
        self.knowledge_base_dir = knowledge_base_dir or config.KNOWLEDGE_BASE_DIR
        logger.info(f"Initialized PDFDocumentLoader with base dir: {self.knowledge_base_dir}")

    def load_single_pdf(self, file_path: Path, category: str) -> List[Document]:
        """
        Loads a single PDF file using LangChain's PyPDFLoader.
        Ignores the file if it is corrupted or unreadable.
        """
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return []

        try:
            logger.info(f"Loading document: {file_path.name} [Category: {category}]")
            
            # Using PyPDFLoader to load and split document by pages
            loader = PyPDFLoader(str(file_path))
            pages = loader.load()
            
            # Enrich metadata for each page Document
            for page in pages:
                # Retain existing metadata like source path and page number, then add custom ones
                page.metadata.update({
                    "category": category,
                    "filename": file_path.name,
                    "file_path": str(file_path.resolve())
                })
            
            logger.info(f"Successfully loaded {len(pages)} pages from {file_path.name}")
            return pages

        except Exception as e:
            # Handle corrupted or unreadable files gracefully
            logger.error(f"Skipping corrupted or unreadable PDF: {file_path.name}. Error details: {repr(e)}")
            logger.debug("Exception traceback details:", exc_info=True)
            return []

    def load_all_documents(self) -> List[Document]:
        """
        Scans all designated subdirectories inside the knowledge base
        and loads all PDF files.
        """
        all_documents: List[Document] = []
        
        # Subdirectories we want to scan (defined in config.py)
        categories = {
            "WHO": config.KB_WHO,
            "NHM_India": config.KB_NHM_INDIA,
            "ASHA": config.KB_ASHA,
            "Research_Papers": config.KB_RESEARCH_PAPERS
        }

        logger.info("Starting knowledge base scanning process...")

        for category, dir_path in categories.items():
            if not dir_path.exists():
                logger.warning(f"Category directory does not exist: {dir_path}. Creating it...")
                dir_path.mkdir(parents=True, exist_ok=True)
                continue

            # List and load all PDFs in this category
            pdf_files = list(dir_path.glob("*.pdf"))
            logger.info(f"Found {len(pdf_files)} PDF files in category '{category}'")

            for pdf_file in pdf_files:
                pages = self.load_single_pdf(pdf_file, category)
                all_documents.extend(pages)

        logger.info(f"Knowledge base load complete. Total documents/pages loaded: {len(all_documents)}")
        return all_documents
