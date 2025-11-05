"""Centralized logging configuration for the benchmark suite."""

import logging
from pathlib import Path


def setup_logging(log_file: Path = None, level=logging.INFO):
    """
    Configure logging for the entire application.
    
    Should be called once at the application entry point.
    
    Parameters
    ----------
    log_file : Path, optional
        Path to log file. If None, uses 'benchmark.log' in current directory
    level : int
        Logging level (default: logging.INFO)
    """
    if log_file is None:
        log_file = Path("benchmark.log")
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    logging.info(f"Logging configured. Log file: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Parameters
    ----------
    name : str
        Module name (typically __name__)
        
    Returns
    -------
    logging.Logger
        Logger instance
    """
    return logging.getLogger(name)
