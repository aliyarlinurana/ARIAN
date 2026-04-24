"""
Consistent logging configuration for the whole pipeline.

Every module should do::

    from src.utils.logging_utils import get_logger
    logger = get_logger(__name__)

so that notebooks and CLI runs share the same format.
"""
from __future__ import annotations

import logging
import sys
from typing import Optional

_CONFIGURED: bool = False
_DEFAULT_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%H:%M:%S"


def _configure_root(level: int = logging.INFO) -> None:
    """Attach a single stderr handler to the root logger (idempotent)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers in notebooks that re-import
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_DEFAULT_FMT, datefmt=_DEFAULT_DATEFMT))
    root.addHandler(handler)

    # Silence noisy third-party chatter by default
    for noisy in ("urllib3", "fiona", "rasterio", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Return a module-scoped logger with the shared format applied."""
    _configure_root(level=level)
    return logging.getLogger(name if name else "wildfire")
