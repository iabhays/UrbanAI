"""
UrbanAI AI — Multi-Agent Smart City Intelligence Platform

A production-grade, research-level multi-agent intelligence platform
for smart city operations.
"""

__version__ = "1.0.0"
__author__ = "UrbanAI AI Team"

# re-export lightweight utilities; heavy components are imported on demand
from .utils.logger import setup_logger
from .utils.config import get_config

__all__ = [
    "setup_logger",
    "get_config",
    "__version__",
    "__author__",
]
