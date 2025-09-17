import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Protocol
from src.core.configuration import RAGConfiguration
from src.engine import Task, ExecutionContext, TaskResult, TaskStatus


# ===================== BUSINESS LOGIC PROTOCOLS =====================

class IngestionService(ABC):
    """Interface that clients must implement for ingestion business logic"""

    @abstractmethod
    def process(self, source: str, config: RAGConfiguration) -> int:
        """
        Process ingestion from source, return processed count.

        Args:
            source: Path or identifier for the data source
            config: RAG configuration containing chunk size, embedding model, etc.

        Returns:
            Number of items/chunks processed and stored

        Raises:
            Exception: If processing fails
        """
        pass


class RetrievalService(ABC):
    """Interface that clients must implement for retrieval business logic"""

    @abstractmethod
    def retieve_evaluate(self, query: str, config: RAGConfiguration) -> list:
        """
        Search and return results based on query.

        Args:
            query: Search query string
            config: RAG configuration containing search parameters

        Returns:
            List of search results

        Raises:
            Exception: If search fails
        """
        pass


class ProcessingService(ABC):
    """Interface that clients must implement for processing business logic"""

    @abstractmethod
    def transform(self, data: Any, config: RAGConfiguration) -> Any:
        """
        Transform data and return result.

        Args:
            data: Input data to transform
            config: RAG configuration containing transformation parameters

        Returns:
            Transformed data

        Raises:
            Exception: If transformation fails
        """
        pass


# ===================== ABSTRACT FRAMEWORK TASKS =====================

class AbstractIngestionTask(Task):
    """Abstract ingestion task that delegates to business logic service"""

    def __init__(self, task_id: str, config: Dict[str, Any],
                 service: IngestionService, rag_configuration: RAGConfiguration):
        super().__init__(task_id, config)
        self.service = service
        self.source = config.get('source')
        self.rag_configuration = rag_configuration

    async def execute(self, context: ExecutionContext) -> TaskResult:
        try:
            self.logger.info(f"Starting ingestion task {self.task_id}")

            # Delegate to external business logic service
            processed_count = await asyncio.get_event_loop().run_in_executor(
                None, self.service.process, self.source, self.rag_configuration
            )

            self.logger.info(f"Ingestion completed: {processed_count} items processed")

            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.COMPLETED,
                metadata={'processed_count': processed_count, 'source': self.source}
            )

        except Exception as e:
            self.logger.error(f"Ingestion task {self.task_id} failed: {str(e)}")
            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.FAILED,
                error=str(e)
            )


class AbstractRetrievalTask(Task):
    """Abstract retrieval task that delegates to business logic service"""

    def __init__(self, task_id: str, config: Dict[str, Any],
                 service: RetrievalService, rag_configuration: RAGConfiguration):
        super().__init__(task_id, config)
        self.service = service
        self.query = config.get('query')
        self.limit = config.get('limit', 10)
        self.rag_configuration = rag_configuration

    async def execute(self, context: ExecutionContext) -> TaskResult:
        try:
            self.logger.info(f"Starting retrieval task {self.task_id}")

            # Get query from context or config
            query = context.get_shared_data('current_query', self.query)

            # Delegate to external business logic service
            results = await asyncio.get_event_loop().run_in_executor(
                None, self.service.search, query, self.rag_configuration
            )

            self.logger.info(f"Retrieval completed: {len(results)} results found")

            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.COMPLETED,
                metadata={'result_count': len(results), 'query': query}
            )

        except Exception as e:
            self.logger.error(f"Retrieval task {self.task_id} failed: {str(e)}")
            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.FAILED,
                error=str(e)
            )


class AbstractProcessingTask(Task):
    """Abstract processing task that delegates to business logic service"""

    def __init__(self, task_id: str, config: Dict[str, Any],
                 service: ProcessingService, rag_configuration: RAGConfiguration):
        super().__init__(task_id, config)
        self.service = service
        self.input_source = config.get('input_source', 'database')
        self.rag_configuration = rag_configuration

    async def execute(self, context: ExecutionContext) -> TaskResult:
        try:
            self.logger.info(f"Starting processing task {self.task_id}")

            # Get input data (could be from context, database, etc.)
            input_data = self._get_input_data(context)

            # Delegate to external business logic service
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.service.transform, input_data, self.rag_configuration
            )

            self.logger.info(f"Processing completed successfully")

            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.COMPLETED,
                metadata={'input_source': self.input_source}
            )

        except Exception as e:
            self.logger.error(f"Processing task {self.task_id} failed: {str(e)}")
            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.FAILED,
                error=str(e)
            )

    def _get_input_data(self, context: ExecutionContext):
        """Get input data based on configuration"""
        if self.input_source == 'database':
            return None  # Service will read from database
        elif self.input_source == 'context':
            return context.get_shared_data('processing_input')
        else:
            return self.input_source