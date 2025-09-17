import asyncio
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from queue import Queue, PriorityQueue
import logging
import json
import pickle
from pathlib import Path
import uuid

from src.engine import ExecutionMode, ExecutionContext, Task, PipelineMetrics, TaskResult, TaskStatus


class PipelineEngine:
    """Advanced pipeline execution engine"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.execution_mode = ExecutionMode(config.get('execution_mode', 'sequential'))
        self.max_parallel_tasks = config.get('max_parallel_tasks', 4)
        self.enable_checkpoints = config.get('enable_checkpoints', True)
        self.checkpoint_interval = config.get('checkpoint_interval', 10)  # Every 10 tasks
        self.enable_monitoring = config.get('enable_monitoring', True)

        # Execution components
        self.task_queue = PriorityQueue()
        self.completed_tasks = set()
        self.failed_tasks = set()
        self.running_tasks = set()

        # Threading/async components
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_parallel_tasks)
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_parallel_tasks)

        # Monitoring
        self.logger = logging.getLogger("PipelineEngine")
        self.hooks = []

        # State persistence
        self.state_dir = Path(config.get('state_dir', 'pipeline_state'))
        self.state_dir.mkdir(exist_ok=True)

    def add_hook(self, hook: 'PipelineHook'):
        """Add a pipeline hook for monitoring/callbacks"""
        self.hooks.append(hook)

    def _trigger_hook(self, event: str, context: ExecutionContext, **kwargs):
        """Trigger all registered hooks for an event"""
        for hook in self.hooks:
            try:
                hook.handle_event(event, context, **kwargs)
            except Exception as e:
                self.logger.error(f"Hook {hook.__class__.__name__} failed: {e}")

    async def execute_pipeline(self, tasks: List[Task], context: ExecutionContext) -> PipelineMetrics:
        """Execute a complete pipeline"""
        try:
            self.logger.info(f"Starting pipeline execution: {context.pipeline_id}")
            context.metrics.total_tasks = len(tasks)
            context.metrics.start_time = datetime.now()

            self._trigger_hook('pipeline_started', context)

            # Build task dependency graph
            task_graph = self._build_dependency_graph(tasks)

            # Execute based on mode
            if self.execution_mode == ExecutionMode.SEQUENTIAL:
                await self._execute_sequential(task_graph, context)
            elif self.execution_mode == ExecutionMode.PARALLEL:
                await self._execute_parallel(task_graph, context)
            elif self.execution_mode == ExecutionMode.ASYNC:
                await self._execute_async(task_graph, context)
            else:
                raise ValueError(f"Unsupported execution mode: {self.execution_mode}")

            context.metrics.end_time = datetime.now()
            context.metrics.execution_time = (
                    context.metrics.end_time - context.metrics.start_time
            ).total_seconds()

            # Calculate throughput
            if context.metrics.execution_time > 0:
                context.metrics.throughput = context.metrics.completed_tasks / context.metrics.execution_time

            self._trigger_hook('pipeline_completed', context)
            self.logger.info(f"Pipeline completed: {context.pipeline_id}")

            return context.metrics

        except Exception as e:
            context.metrics.end_time = datetime.now()
            self._trigger_hook('pipeline_failed', context, error=str(e))
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise

    def _build_dependency_graph(self, tasks: List[Task]) -> Dict[str, Task]:
        """Build task dependency graph"""
        task_graph = {task.task_id: task for task in tasks}

        # Validate dependencies exist
        for task in tasks:
            for dep_id in task.dependencies:
                if dep_id not in task_graph:
                    raise ValueError(f"Task {task.task_id} depends on unknown task {dep_id}")

        return task_graph

    async def _execute_sequential(self, task_graph: Dict[str, Task], context: ExecutionContext):
        """Execute tasks sequentially respecting dependencies"""
        executed = set()
        remaining = set(task_graph.keys())

        while remaining:
            # Find tasks with satisfied dependencies
            ready_tasks = []
            for task_id in remaining:
                task = task_graph[task_id]
                if all(dep_id in executed for dep_id in task.dependencies):
                    if task.should_execute(context):
                        ready_tasks.append(task)
                    else:
                        executed.add(task_id)
                        remaining.remove(task_id)
                        context.add_task_result(task_id, TaskResult(
                            task_id=task_id,
                            status=TaskStatus.SKIPPED
                        ))
                        break

            if not ready_tasks and remaining:
                raise RuntimeError("Circular dependency or unsatisfied dependencies detected")

            # Execute the first ready task
            if ready_tasks:
                task = ready_tasks[0]
                result = await self._execute_single_task(task, context)
                context.add_task_result(task.task_id, result)
                executed.add(task.task_id)
                remaining.remove(task.task_id)

                # Checkpoint if enabled
                if self.enable_checkpoints and len(executed) % self.checkpoint_interval == 0:
                    context.checkpoint(f"sequential_{len(executed)}", executed)

    async def _execute_parallel(self, task_graph: Dict[str, Task], context: ExecutionContext):
        """Execute tasks in parallel where possible"""
        executed = set()
        remaining = set(task_graph.keys())
        running = {}

        while remaining or running:
            # Start new tasks that have satisfied dependencies
            ready_tasks = []
            for task_id in list(remaining):
                task = task_graph[task_id]
                if all(dep_id in executed for dep_id in task.dependencies):
                    if task.should_execute(context):
                        ready_tasks.append(task)
                    else:
                        executed.add(task_id)
                        remaining.remove(task_id)
                        context.add_task_result(task_id, TaskResult(
                            task_id=task_id,
                            status=TaskStatus.SKIPPED
                        ))

            # Start tasks up to parallel limit
            slots_available = self.max_parallel_tasks - len(running)
            for task in ready_tasks[:slots_available]:
                if task.parallel_safe:
                    future = asyncio.create_task(self._execute_single_task(task, context))
                    running[task.task_id] = future
                    remaining.remove(task.task_id)

            # Wait for at least one task to complete
            if running:
                done_ids = []
                for task_id, future in running.items():
                    if future.done():
                        result = await future
                        context.add_task_result(task_id, result)
                        executed.add(task_id)
                        done_ids.append(task_id)

                for task_id in done_ids:
                    del running[task_id]

                # If nothing completed, wait a bit
                if not done_ids:
                    await asyncio.sleep(0.1)

    async def _execute_async(self, task_graph: Dict[str, Task], context: ExecutionContext):
        """Execute all independent tasks asynchronously"""
        # Group tasks by dependency level
        levels = self._get_dependency_levels(task_graph)

        for level_tasks in levels:
            if not level_tasks:
                continue

            # Execute all tasks at this level concurrently
            tasks_to_run = [
                task for task in level_tasks
                if task.should_execute(context)
            ]

            if tasks_to_run:
                futures = [
                    self._execute_single_task(task, context)
                    for task in tasks_to_run
                ]

                results = await asyncio.gather(*futures, return_exceptions=True)

                for task, result in zip(tasks_to_run, results):
                    if isinstance(result, Exception):
                        result = TaskResult(
                            task_id=task.task_id,
                            status=TaskStatus.FAILED,
                            error=str(result)
                        )
                    context.add_task_result(task.task_id, result)

    def _get_dependency_levels(self, task_graph: Dict[str, Task]) -> List[List[Task]]:
        """Group tasks by dependency levels for async execution"""
        levels = []
        remaining = set(task_graph.keys())

        while remaining:
            # Find tasks with no dependencies in remaining set
            current_level = []
            for task_id in list(remaining):
                task = task_graph[task_id]
                if all(dep_id not in remaining for dep_id in task.dependencies):
                    current_level.append(task)
                    remaining.remove(task_id)

            if not current_level:
                raise RuntimeError("Circular dependency detected")

            levels.append(current_level)

        return levels

    async def _execute_single_task(self, task: Task, context: ExecutionContext) -> TaskResult:
        """Execute a single task with monitoring"""
        self.logger.info(f"Executing task: {task.task_id}")
        self._trigger_hook('task_started', context, task=task)

        try:
            result = await task.execute_with_retry(context)
            self._trigger_hook('task_completed', context, task=task, result=result)
            return result
        except Exception as e:
            result = TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=str(e)
            )
            self._trigger_hook('task_failed', context, task=task, result=result)
            return result

    def save_state(self, context: ExecutionContext, filename: str = None):
        """Save pipeline state to disk"""
        if not filename:
            filename = f"pipeline_{context.pipeline_id}_{int(time.time())}.pkl"

        filepath = self.state_dir / filename
        with open(filepath, 'wb') as f:
            pickle.dump({
                'context': context,
                'completed_tasks': self.completed_tasks,
                'failed_tasks': self.failed_tasks
            }, f)

        self.logger.info(f"Pipeline state saved to {filepath}")

    def load_state(self, filename: str) -> ExecutionContext:
        """Load pipeline state from disk"""
        filepath = self.state_dir / filename
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        context = state['context']
        self.completed_tasks = state['completed_tasks']
        self.failed_tasks = state['failed_tasks']

        self.logger.info(f"Pipeline state loaded from {filepath}")
        return context