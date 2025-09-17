# -----------------------------------------------------------------------------
# ----- FILE: state_manager.py
# -----------------------------------------------------------------------------
"""state_manager

Small utility to persist and restore pipeline state to disk.

This module centralizes pickling behaviour and file layout logic so the engine
and the execution context do not need to implement file IO directly.

Security note:
    This module uses `pickle` for simplicity. Pickled files are dangerous when
    loaded from untrusted sources; do not load pickles unless they originate
    from a trusted execution.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any, Dict

from .execution_context import ExecutionContext


class StateManager:
    """Handles simple on-disk persistence for the pipeline engine.

    Usage:
        StateManager.save(context, completed_tasks, failed_tasks, state_dir)
        context = StateManager.load(filename, state_dir)
    """

    @staticmethod
    def save(context: ExecutionContext, completed_tasks: set, failed_tasks: set, state_dir: str = "pipeline_state", filename: str = None) -> Path:
        """Persist the provided context and auxiliary sets to a pickle file.

        Args:
            context: ExecutionContext to persist.
            completed_tasks: set of completed tasks maintained by the engine.
            failed_tasks: set of failed tasks maintained by the engine.
            state_dir: Directory to store the state file.
            filename: Optional filename; if omitted a timestamped name is used.

        Returns:
            Path to the saved file.
        """
        state_directory = Path(state_dir)
        state_directory.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = f"pipeline_{context.pipeline_id}_{int(time.time())}.pkl"

        filepath = state_directory / filename
        with open(filepath, "wb") as f:
            pickle.dump({
                "context": context,
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
            }, f)

        return filepath

    @staticmethod
    def load(filename: str, state_dir: str = "pipeline_state") -> Dict[str, Any]:
        """Load pipeline state from disk and return the raw dictionary.

        The caller is responsible for type-checking and placing returned values
        back into the engine if necessary.
        """
        filepath = Path(state_dir) / filename
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        return state