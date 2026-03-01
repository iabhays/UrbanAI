"""Unified logger for UrbanAI modules.

Attempts to use loguru if available, otherwise falls back to the standard
`logging` module so that the codebase can operate without requiring
extra dependencies during lightweight testing or in constrained environments.
"""

try:
    # prefer loguru if installed for nicer formatting
    from loguru import logger  # type: ignore
except ImportError:  # pragma: no cover - fallback path for environments without loguru
    import logging

    logger = logging.getLogger("urbanai")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
