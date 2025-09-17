# -----------------------------------------------------------------------------
# ----- FILE: monitoring.py
# -----------------------------------------------------------------------------
"""monitoring

Monitoring utilities used by the pipeline engine and hooks.

This module contains a lightweight Monitor helper that can be extended to use
`psutil` for real metrics. It also provides a convenience function to setup
python logging with a sane default configuration.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class Monitor:
    """Lightweight runtime monitor.

    The Monitor exposes methods to sample simple process-level metrics. This
    is intentionally optional: the implementation tries to use `psutil` if
    available and falls back to coarse estimates otherwise.
    """

    # PRIVATE: we don't store mutable state here; the class is a thin wrapper.
    def sample(self) -> Dict[str, Any]:
        """Return a dictionary with sample metrics.

        The returned dict may include keys like `memory_rss`, `cpu_percent` and
        other environment specific information.
        """
        try:
            # Try to import psutil when available for accurate metrics
            import psutil  # type: ignore

            process = psutil.Process(os.getpid())
            mem = process.memory_info().rss
            cpu = process.cpu_percent(interval=0.0)
            return {"memory_rss": mem, "cpu_percent": cpu}

        except Exception:
            # Fallback: return environment variables or zeros when psutil is
            # unavailable. This is OK for low-fidelity monitoring.
            return {"memory_rss": 0, "cpu_percent": 0}


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with a simple format.

    Call this early in your application to ensure consistent log formatting
    across the engine and tasks.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(level)