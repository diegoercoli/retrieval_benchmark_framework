# -----------------------------------------------------------------------------
# ----- FILE: task_scheduler.py
# -----------------------------------------------------------------------------
"""task_scheduler

Task base classes and common task implementations.

This module provides:
* `Task` (abstract base class) — dependency-aware, retrying async tasks.
* Concrete convenience tasks: `IngestionTask`, `RetrievalTask`, `ProcessingTask`.

Design principles:
* Task.execute() is `async` so the engine can await task execution or schedule
  it concurrently. Heavy CPU work should be run in a thread/process pool via
  ``asyncio.get_event_loop().run_in_executor``.
* Tasks are small, composable units of work. They should be side-effect aware
  and write to the ExecutionContext instead of global state.
"""
from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from .execution_context import ExecutionContext, TaskResult, TaskStatus


class Task(ABC):
    """Abstract base class for pipeline tasks.

    A Task represents a unit of work with optional dependencies. Concrete
    implementations must implement :meth:`execute` which performs the task and
    returns a TaskResult.

    Attributes (commonly set via `config` dict):
        dependencies: list of task ids that must complete before this task runs
        retry_count/max_retries: retry behaviour
        timeout: optional per-task timeout (seconds)
        priority: task scheduling priority (higher runs earlier if supported)
        parallel_safe: whether it's safe to run in parallel with others
        cache_result: whether the engine may cache the result
        condition: callable that receives ExecutionContext -> bool (run condition)
    """

    def __init__(self, task_id: str, config: Dict[str, Any]):
        self.task_id = task_id
        self.config = config or {}
        self.dependencies: List[str] = self.config.get("dependencies", [])
        self.retry_count: int = self.config.get("retry_count", 0)
        self.max_retries: int = self.config.get("max_retries", 3)
        self.timeout: Optional[float] = self.config.get("timeout", None)
        self.priority: int = self.config.get("priority", 0)
        self.parallel_safe: bool = self.config.get("parallel_safe", True)
        self.cache_result: bool = self.config.get("cache_result", False)
        self.condition = self.config.get("condition", None)
        self.logger = logging.getLogger(f"Task.{task_id}")

    @abstractmethod
    async def execute(self, context: ExecutionContext) -> TaskResult:
        """Run the task logic and return a TaskResult.

        Implementations should:
        * perform work
        * construct and return a TaskResult instance representing success/failure
        """
        raise NotImplementedError

    def should_execute(self, context: ExecutionContext) -> bool:
        """Optional conditional execution hook.

        If `self.condition` is provided it will be called with the ExecutionContext
        and must return a boolean indicating whether the task should run.
        """
        if self.condition:
            # PRIVATE: condition is responsible for not raising exceptions; if it
            # does raise we propagate the exception to the caller.
            return self.condition(context)
        return True

    def dependencies_met(self, context: ExecutionContext) -> bool:
        """Check if all dependencies are completed in the provided context."""
        for dep_id in self.dependencies:
            result = context.get_task_result(dep_id)
            if not result or result.status != TaskStatus.COMPLETED:
                return False
        return True

    async def execute_with_retry(self, context: ExecutionContext) -> TaskResult:
        """Execute the task with retry/backoff semantics.

        Returns TaskResult (successful or failed after retries).
        """
        # PRIVATE: start attempts at 0; attempt count includes the first run
        for attempt in range(self.max_retries + 1):
            start_time = time.time()
            try:
                if attempt > 0:
                    self.logger.info(f"Retrying task {self.task_id}, attempt {attempt}")

                # Respect per-task timeout, delegating to asyncio.wait_for
                if self.timeout:
                    result = await asyncio.wait_for(self.execute(context), timeout=self.timeout)
                else:
                    result = await self.execute(context)

                # Fill in execution timing
                result.execution_time = time.time() - start_time
                result.task_id = self.task_id
                return result

            except Exception as e:
                # PRIVATE: capture exception text for the task result
                self.logger.exception(f"Task {self.task_id} failed on attempt {attempt + 1}")
                if attempt == self.max_retries:
                    # Final failure: return TaskResult describing failure
                    return TaskResult(
                        task_id=self.task_id,
                        status=TaskStatus.FAILED,
                        error=str(e),
                        execution_time=time.time() - start_time,
                    )
                # Backoff before next retry (exponential)
                await asyncio.sleep(2 ** attempt)


class ProcessingTask(Task):
    """Generic CPU or IO-bound processing task wrapper.

    The processor_func is a callable that receives the values of input_keys
    unpacked in order and returns a result. The result is optionally stored
    under `output_key` in the execution context's shared data.
    """

    def __init__(self, task_id: str, config: Dict[str, Any], processor_func: Callable):
        super().__init__(task_id, config)
        self.processor_func = processor_func
        self.input_keys = config.get("input_keys", [])
        self.output_key = config.get("output_key")

    async def execute(self, context: ExecutionContext) -> TaskResult:
        """Collect inputs from context and run processor_func in threadpool."""
        try:
            # Gather inputs from the shared context according to the configured keys
            input_data = []
            for key in self.input_keys:
                data = context.get_shared_data(key)
                if data is not None:
                    input_data.append(data)

            # PRIVATE: execute potentially blocking business logic on a worker
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.processor_func, *input_data
            )

            # Optionally attach the result back into the shared context
            if self.output_key:
                context.set_shared_data(self.output_key, result)

            return TaskResult(task_id=self.task_id, status=TaskStatus.COMPLETED, data=result)

        except Exception as e:
            self.logger.exception("Processing task failed")
            return TaskResult(task_id=self.task_id, status=TaskStatus.FAILED, error=str(e))