# -----------------------------------------------------------------------------
# ----- FILE: hooks.py
# -----------------------------------------------------------------------------
"""hooks

Hook abstractions for pipeline lifecycle events.

Hooks are synchronous handlers executed by the engine when specific events
occur (pipeline start/complete, task start/complete, etc.). They provide a
simple extension point for logging, metrics export or external notifications.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

from .execution_context import ExecutionContext, TaskResult


class PipelineHook(ABC):
    """Abstract base class for pipeline hooks.

    Implementations should not raise exceptions; the engine will catch and log
    errors from hooks to avoid interrupting pipeline execution.
    """

    @abstractmethod
    def handle_event(self, event: str, context: ExecutionContext, **kwargs) -> None:
        """Handle an event emitted by the PipelineEngine.

        Args:
            event: A string tag identifying the event (e.g. 'task_started').
            context: The ExecutionContext for the active pipeline run.
            kwargs: Event-specific keyword arguments (often 'task' or 'result').
        """
        raise NotImplementedError


class LoggingHook(PipelineHook):
    """Hook that writes human-friendly logs using the standard logging API.

    This is a low-cost default hook suitable for most development and testing
    scenarios.
    """

    def __init__(self):
        self.logger = logging.getLogger("PipelineHook.Logging")

    def handle_event(self, event: str, context: ExecutionContext, **kwargs) -> None:
        """Emit log lines for common pipeline events."""
        try:
            if event == "pipeline_started":
                self.logger.info(f"Pipeline {context.pipeline_id} started")
            elif event == "task_started":
                task = kwargs.get("task")
                self.logger.info(f"Task {task.task_id} started")
            elif event == "task_completed":
                task = kwargs.get("task")
                result: TaskResult = kwargs.get("result")
                self.logger.info(f"Task {task.task_id} completed in {result.execution_time:.2f}s")
            elif event == "pipeline_completed":
                metrics = context.metrics
                self.logger.info(f"Pipeline completed: {metrics.completed_tasks}/{metrics.total_tasks} tasks succeeded")
        except Exception:
            # Hooks must not propagate exceptions to the engine
            self.logger.exception("LoggingHook failed to handle event")


class MetricsHook(PipelineHook):
    """Collects per-task metrics in memory.

    This example hook keeps a simple in-memory record for the pipeline id.
    A production implementation would periodically flush to an external
    metrics backend.
    """

    def __init__(self):
        self.metrics = {}

    def handle_event(self, event: str, context: ExecutionContext, **kwargs) -> None:
        if event == "task_completed":
            task = kwargs.get("task")
            result: TaskResult = kwargs.get("result")

            if context.pipeline_id not in self.metrics:
                self.metrics[context.pipeline_id] = []

            self.metrics[context.pipeline_id].append({
                "task_id": task.task_id,
                "execution_time": result.execution_time,
                "memory_usage": result.memory_usage,
                "timestamp": result.timestamp.isoformat(),
            })


class WebhookHook(PipelineHook):
    """Hook that would send HTTP POSTs to an external webhook.

    This implementation logs what it *would* send. Replace with `requests`
    or `httpx` calls in real systems.
    """

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.logger = logging.getLogger("PipelineHook.Webhook")

    def handle_event(self, event: str, context: ExecutionContext, **kwargs) -> None:
        try:
            # PRIVATE: placeholder implementation that only logs. This avoids
            # adding a hard dependency on HTTP client libraries in the core.
            self.logger.info(f"Would send webhook to {self.webhook_url} for event: {event}")
        except Exception:
            self.logger.exception("WebhookHook failed")
