"""
test_chunker.py — Verification script to test PDF Document Loading and Chunking.
"""

from pathlib import Path
import sys

# Add backend directory to python path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

from rag.loader import PDFDocumentLoader
from rag.chunker import DocumentChunker
from utils.helpers import setup_logging, get_logger

def main():
    setup_logging(log_to_file=True, log_file_name="test_chunker.log")
    logger = get_logger("TestChunker")

    logger.info("Initializing Loader and Chunker verification...")
    
    # 1. Load documents
    loader = PDFDocumentLoader()
    documents = loader.load_all_documents()
    logger.info(f"Loaded {len(documents)} pages.")
    
    if not documents:
        logger.error("No documents loaded to test chunker!")
        return

    # 2. Chunk documents
    chunker = DocumentChunker(chunk_size=800, chunk_overlap=150)
    chunks = chunker.split_documents(documents)
    logger.info(f"Generated {len(chunks)} chunks.")

    # 3. Verify chunk properties
    print("\n" + "="*50)
    print(" CHUNKING VERIFICATION REPORT")
    print("="*50)
    print(f"Total Pages Loaded: {len(documents)}")
    print(f"Total Chunks Generated: {len(chunks)}")
    print(f"Average Chunks per Page: {len(chunks)/len(documents):.2f}")
    
    # Check metadata validation for first few chunks
    print("\nChecking metadata schema of first 3 chunks:")
    for idx, chunk in enumerate(chunks[:3]):
        print(f"\nChunk #{idx+1}:")
        print(f"  - source: {chunk.metadata.get('source')}")
        print(f"  - page: {chunk.metadata.get('page')}")
        print(f"  - document_name: {chunk.metadata.get('document_name')}")
        print(f"  - chunk length: {len(chunk.page_content)} characters")

    # Sample chunk output
    print("\n" + "="*50)
    print(" SAMPLE CHUNK CONTENT")
    print("="*50)
    sample_chunk = chunks[0]
    print(f"Doc: {sample_chunk.metadata.get('document_name')} | Page: {sample_chunk.metadata.get('page') + 1}")
    print("-" * 50)
    print(sample_chunk.page_content)
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
