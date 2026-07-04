"""
helpers.py — Core utility functions and logging configuration for ASHA Sahaayak
"""

import logging
import sys
from pathlib import Path

# Set up clean logging architecture
def setup_logging(level=logging.INFO, log_to_file=True, log_file_name="asha_backend.log"):
    """
    Configures logging for the backend application.
    Logs to stdout and optionally to a file in the reports/ directory.
    """
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # Root logger configuration
    root_logger = logging.getLogger()
    
    # Avoid duplicating handlers if already set up
    if root_logger.handlers:
        return root_logger

    root_logger.setLevel(level)

    # 1. Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)

    # 2. File Handler (if requested)
    if log_to_file:
        # Resolve backend root to save logs under reports/ or root
        backend_dir = Path(__file__).resolve().parent.parent
        log_dir = backend_dir / "reports"
        
        # Ensure log directory exists
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / log_file_name

        try:
            file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            root_logger.addHandler(file_handler)
            logging.info(f"Logging initialized. Writing log file to: {log_file_path}")
        except Exception as e:
            logging.error(f"Failed to initialize file logger: {e}. Logging to console only.")

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Utility function to obtain a configured logger.
    """
    # Auto-run setup if logger doesn't have handlers
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        setup_logging()
    return logging.getLogger(name)
