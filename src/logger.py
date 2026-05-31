"""
logger.py – Centralized logging setup for the OpenSetDGA pipeline.

Usage:
    from logger import get_logger
    log = get_logger(__name__)
    log.info("message")
"""
import logging
import sys
from pathlib import Path


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger with console handler using the project's standard format."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(module)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_run_logger(name: str, log_file: Path, level: int = logging.INFO) -> logging.Logger:
    """Return a logger that writes to both console and a file."""
    logger = get_logger(name, level)
    if any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        return logger

    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(module)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)
    return logger
