"""
logger.py

This module sets up and configures logging for the application.
"""

import logging
import sys
from typing import Optional

# Default log format
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Creating a Logger
logger = logging.getLogger("pptgen")


def setup_logger(level: str = "INFO", log_format: Optional[str] = None) -> logging.Logger:
    """
    Configures and returns an application logger

    Args:
        level: Log level, optional values: DEBUG, INFO, WARNING, ERROR, CRITICAL
        log_format: Log format, if None, use the default format

    Returns:
        logging.Logger: Configured logger
    """
    # Setting the log level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # If there is no processor, add a console processor
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(log_format or DEFAULT_LOG_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# Default Logger
setup_logger()


# Export common log functions for direct calling
def debug(msg: str, *args, **kwargs) -> None:
    """
    Record DEBUG level logs
    """
    logger.debug(msg, *args, **kwargs)


def info(msg: str, *args, **kwargs) -> None:
    """
    Record INFO level logs
    """
    logger.info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs) -> None:
    """
    Record WARNING level logs
    """
    logger.warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs) -> None:
    """
    Record ERROR level logs
    """
    logger.error(msg, *args, **kwargs)


def critical(msg: str, *args, **kwargs) -> None:
    """
    Record CRITICAL level logs
    """
    logger.critical(msg, *args, **kwargs)
