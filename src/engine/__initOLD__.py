# Package: framework/engine
# This single document contains the recommended split across multiple files.
# Each file is annotated with a header comment "# ----- FILE: <name> -----" so you can copy them
# into the indicated file layout.

# -----------------------------------------------------------------------------
# ----- FILE: __initOLD__.py
# -----------------------------------------------------------------------------
"""framework.engine

Top-level package exports for the pipeline engine package.

This module re-exports the public API of the package so consumers can do::

    from framework.engine import PipelineEngine, ExecutionContext, Task

Note:
    This file contains only import/exports and documentation; implementation
    details live in submodules.
"""

# Public API: import key classes into package namespace for convenience.
from .pipeline_engine import PipelineEngine  # noqa: F401
from .execution_context import (  # noqa: F401
    ExecutionContext,
    TaskResult,
    PipelineMetrics,
    TaskStatus,
    PipelineStatus,
    ExecutionMode,
)
from .task_scheduler import (  # noqa: F401
    Task,
#    IngestionTask,
#    RetrievalTask,
    ProcessingTask,
)
from .state_manager import StateManager  # noqa: F401
from .hooks import PipelineHook, LoggingHook, MetricsHook, WebhookHook  # noqa: F401
from .monitoring import Monitor, setup_logging  # noqa: F401

__all__ = [
    "PipelineEngine",
    "ExecutionContext",
    "TaskResult",
    "PipelineMetrics",
    "TaskStatus",
    "PipelineStatus",
    "ExecutionMode",
    "Task",
#    "IngestionTask",
#    "RetrievalTask",
    "StateManager",
    "PipelineHook",
    "LoggingHook",
    "MetricsHook",
    "WebhookHook",
    "Monitor",
    "setup_logging",
]

