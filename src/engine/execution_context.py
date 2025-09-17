# -----------------------------------------------------------------------------
# ----- FILE: execution_context.py
# -----------------------------------------------------------------------------
"""execution_context

Execution context, result and metrics classes used across the pipeline engine.

This module defines small, focused data objects and the ExecutionContext which
is the mutable state object passed throughout pipeline execution.

Design notes:
* Keep this module dependency-light so it is safe to import from many places.
* ExecutionContext owns runtime state (shared data, task results, checkpoints)
  but does not implement persistence (StateManager handles on-disk operations).
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


# ------------------------------------------------------------------
# Enums: ExecutionMode, TaskStatus, PipelineStatus
# ------------------------------------------------------------------
class ExecutionMode(Enum):
    """Execution strategies supported by the engine.

    Values:
        SEQUENTIAL: Run tasks one after another in a single-threaded manner.
        PARALLEL: Run tasks in parallel up to `max_parallel_tasks`.
        ASYNC: Execute by dependency-level concurrently using asyncio.gather.
        DISTRIBUTED: Placeholder for executing across multiple machines.
    """

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ASYNC = "async"
    DISTRIBUTED = "distributed"


class TaskStatus(Enum):
    """Status values for a single TaskResult.

    These values allow callers and monitoring to infer what happened with a
    task. They are intentionally minimal so the status machine stays simple.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class PipelineStatus(Enum):
    """High-level pipeline lifecycle states."""

    INITIALIZED = "initialized"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


# ------------------------------------------------------------------
# Dataclasses: TaskResult, PipelineMetrics
# ------------------------------------------------------------------
@dataclass
class TaskResult:
    """Result container for a single task execution.

    Attributes:
        task_id: Unique identifier of the task.
        status: TaskStatus value reflecting outcome.
        data: Arbitrary return value or payload produced by the task.
        error: Textual error when the task failed (optional).
        execution_time: Wall-clock seconds the task took to run.
        memory_usage: Optional memory usage estimate (engine/monitor may fill this).
        metadata: Free-form metadata dictionary (counts, ids, diagnostics).
        timestamp: When the TaskResult object was created.
    """

    task_id: str
    status: TaskStatus
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    memory_usage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PipelineMetrics:
    """Aggregate metrics collected for a pipeline execution.

    This structure is lightweight and intended for both quick monitoring and
    being persisted alongside the execution context.
    """

    pipeline_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    execution_time: float = 0.0
    throughput: float = 0.0
    memory_peak: float = 0.0
    cpu_usage: float = 0.0


# ------------------------------------------------------------------
# ExecutionContext
# ------------------------------------------------------------------
class ExecutionContext:
    """Context object that carries mutable runtime state during execution.

    The ExecutionContext is the central place for tasks and engine code to
    read/write shared information. It intentionally keeps serialization simple
    (plain Python objects) and delegates persistence to StateManager.

    Example:

        ctx = ExecutionContext('pipeline-1', config={})
        ctx.set_shared_data('batch_id', 123)

    Thread-safety:
        This implementation is not thread-safe. If you run tasks across threads
        and they mutate context concurrently, protect access externally or add
        locks here.
    """

    def __init__(self, pipeline_id: str, config: Dict[str, Any]):
        """Create a new execution context.

        Args:
            pipeline_id: A human-readable id for the pipeline run.
            config: The pipeline configuration dictionary.
        """

        # Public identifiers
        self.pipeline_id = pipeline_id
        self.execution_id = str(uuid.uuid4())

        # Configuration and mutable shared state used by tasks
        self.config = config
        self.shared_state: Dict[str, Any] = {}

        # Mapping task_id -> TaskResult
        self.task_results: Dict[str, TaskResult] = {}

        # Metrics holder initialized with the current time as start
        self.metrics = PipelineMetrics(pipeline_id, datetime.now())

        # Convenience logger named per pipeline id
        self.logger = logging.getLogger(f"Context.{pipeline_id}")

        # Checkpoints: lightweight in-memory snapshot storage
        self.checkpoints: Dict[str, Any] = {}

        # Hooks attached to this execution (the engine also exposes hooks)
        self.hooks = []

    # ----------------------- shared state helpers -----------------------
    def set_shared_data(self, key: str, value: Any) -> None:
        """Store value under key in the shared execution state.

        This is the primary mechanism tasks use to pass data between each other.
        Keep values serializable if you intend to persist or checkpoint the
        context.
        """
        # PRIVATE: simple implementation; callers should ensure correctness.
        self.shared_state[key] = value

    def get_shared_data(self, key: str, default: Any = None) -> Any:
        """Return shared value or default if missing."""
        return self.shared_state.get(key, default)

    # ----------------------- task result helpers -----------------------
    def add_task_result(self, task_id: str, result: TaskResult) -> None:
        """Store the TaskResult for `task_id` and update aggregate metrics.

        Args:
            task_id: Identifier of the task whose result is being stored.
            result: TaskResult instance describing the outcome.
        """
        # PRIVATE: we keep a copy of the TaskResult reference and update
        # counters which monitoring and hooks expect to be immediate.
        self.task_results[task_id] = result

        if result.status == TaskStatus.COMPLETED:
            self.metrics.completed_tasks += 1
        elif result.status == TaskStatus.FAILED:
            self.metrics.failed_tasks += 1

    def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """Retrieve the TaskResult for a given task id, or None."""
        return self.task_results.get(task_id)

    # ----------------------- checkpoints -----------------------
    def checkpoint(self, name: str, data: Any) -> None:
        """Create a named checkpoint.

        The checkpoint includes the supplied `data`, a timestamp and a shallow
        copy of task results so that callers can easily inspect what finished
        at the checkpoint point.
        """
        self.checkpoints[name] = {
            "data": data,
            "timestamp": datetime.now(),
            "task_results": self.task_results.copy(),
        }
        self.logger.debug(f"Checkpoint created: {name}")

    def restore_checkpoint(self, name: str) -> bool:
        """Restore internal state from a checkpoint name.

        Returns True on success and False if checkpoint is unknown.
        Note that this restores only the task_results snapshot and does not
        automatically rewind any side-effects performed by tasks.
        """
        if name in self.checkpoints:
            checkpoint = self.checkpoints[name]
            self.task_results = checkpoint["task_results"].copy()
            self.logger.info(f"Restored checkpoint: {name}")
            return True
        return False