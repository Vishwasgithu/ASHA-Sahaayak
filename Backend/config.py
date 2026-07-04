"""
config.py — Centralized configuration for ASHA Sahaayak Backend
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ─────────────────────────────────────────────
# Base Paths
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

# Load local runtime configuration without overriding environment variables
# explicitly supplied by the shell or deployment platform.
load_dotenv(BASE_DIR / ".env", override=False)

KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
VECTOR_DB_DIR      = BASE_DIR / "vector_db"
PROMPTS_DIR        = BASE_DIR / "prompts"
MODELS_DIR         = BASE_DIR / "models"
REPORTS_DIR        = BASE_DIR / "reports"

# ─────────────────────────────────────────────
# Knowledge Base Sub-folders
# ─────────────────────────────────────────────
KB_WHO              = KNOWLEDGE_BASE_DIR / "WHO"
KB_NHM_INDIA        = KNOWLEDGE_BASE_DIR / "NHM_India"
KB_ASHA             = KNOWLEDGE_BASE_DIR / "ASHA"
KB_RESEARCH_PAPERS  = KNOWLEDGE_BASE_DIR / "Research_Papers"

# ─────────────────────────────────────────────
# ChromaDB Settings
# ─────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "asha_knowledge"
CHROMA_PERSIST_DIR     = str(VECTOR_DB_DIR)

# ─────────────────────────────────────────────
# Embedding Model
# ─────────────────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# ─────────────────────────────────────────────
# Text Chunking
# ─────────────────────────────────────────────
CHUNK_SIZE    = 800   # tokens / characters per chunk
CHUNK_OVERLAP = 150   # overlap between consecutive chunks

# ─────────────────────────────────────────────
# LLM Settings
# ─────────────────────────────────────────────
LLM_PROVIDER    = os.getenv("LLM_PROVIDER", "gemini")          # "gemini" | "openai" | "ollama"
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL_NAME  = os.getenv("LLM_MODEL_NAME", "gemini-2.0-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# ─────────────────────────────────────────────
# Retrieval Settings
# ─────────────────────────────────────────────
TOP_K_RESULTS = 5   # Number of chunks to retrieve per query

# ─────────────────────────────────────────────
# Flask Settings
# ─────────────────────────────────────────────
FLASK_HOST  = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT  = int(os.getenv("FLASK_PORT", "5000"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "true").lower() == "true"

# ─────────────────────────────────────────────
# Prompt Files
# ─────────────────────────────────────────────
CLINICAL_PROMPT_FILE = PROMPTS_DIR / "clinical_prompt.txt"
