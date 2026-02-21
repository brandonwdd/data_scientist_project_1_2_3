"""Logging Configuration"""

import logging
import sys
from typing import Optional
from platform_sdk.common.config import Config

def setup_logging(
    name: str,
    level: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging for a module
    
    Args:
        name: Logger name
        level: Log level (defaults to Config.LOG_LEVEL)
        format_string: Custom format string
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    if level is None:
        level = Config.LOG_LEVEL
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))
    
    # Set format
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "[%(filename)s:%(lineno)d] - %(message)s"
        )
    
    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger
