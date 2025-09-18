from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List
import time
from src.core.configuration import RAGConfiguration, SearchType, SearchConfiguration
from src.core.preprocess_dataset import process_dataset, GroundTruthDocumentRecord
from src.core.data_evaluation import EvaluationService
from src.vectordb.weaviate_db_manager import WeaviateDBManager


class RetrievalService(ABC):
    """Interface that clients must implement for retrieval business logic"""

    @abstractmethod
    def retrieve_evaluate(self, config: RAGConfiguration):
        """
        Search and return results based on query.

        Args:
            config: RAG configuration containing search parameters

        Raises:
            Exception: If search fails
        """
        pass


class DocumentRetrievalService(RetrievalService):

    def __init__(self, db_manager: WeaviateDBManager):
        self.db_manager = db_manager
        # Use the new dual-level evaluation service
        self.evaluation_service = EvaluationService()

    def retrieve_evaluate(self, config: RAGConfiguration):
        """Enhanced retrieve and evaluate method using dual-level evaluation service."""

        print(f"⏱️  Starting retrieval/evaluation for '{config.collection_name}'")
        #start_time = time.time()

        # Use the new evaluation service with injected retrieval function
        self.evaluation_service.evaluate_retrieval_results(
            config=config,
            retrieval_function=self._create_retrieval_function()
        )

        # Generate comprehensive reports
        self.evaluation_service.generate_reports(config)

        # Clear metrics for next collection (if processing multiple)
        # Note: Don't clear here if you want to aggregate across collections
        # self.evaluation_service.clear_metrics()

        # Return summary for programmatic access
        return self.evaluation_service.get_metrics_summary()

    def _create_retrieval_function(self):
        """
        Create a retrieval function that can be injected into the evaluation service.
        This follows dependency injection pattern for better testability.
        """

        def retrieval_function(query: str, collection_name: str, search_config: SearchConfiguration) -> List[dict]:
            """
            Retrieve function that always fetches 50 chunks for comprehensive evaluation.

            The evaluation service expects:
            - 50 chunks total for ranking evaluation (NDCG, MRR)
            - Will use top_k from search_config for precision/recall calculations
            """
            return self.retrieve_chunks(query, collection_name, search_config, fixed_k=50)

        return retrieval_function

    def retrieve_chunks(self, query: str, collection_name: str, search_config: SearchConfiguration,
                        fixed_k: int = 50) -> List[dict]:
        """
        Retrieve chunks using different search strategies.
        Always retrieves fixed_k=50 chunks for comprehensive evaluation.

        Args:
            query: Search query string
            collection_name: Name of the collection to search in
            search_config: Search configuration with type and parameters
            fixed_k: Fixed number of chunks to retrieve (default 50 for evaluation)

        Returns:
            List of matching chunk properties (always 50 chunks if available)
        """
        results = []

        # Always retrieve 50 chunks regardless of configured top_k
        # The evaluation service will use search_config.top_k for precision/recall
        # but will use all 50 for NDCG and MRR calculations
        k_to_retrieve = fixed_k

        match search_config.search_type:
            case SearchType.SEMANTIC_SEARCH:
                results = self.db_manager.semantic_search_retrieve(query, collection_name, k_to_retrieve)

            case SearchType.KEYWORD_SEARCH:
                results = self.db_manager.bm25_retrieve(query, collection_name, k_to_retrieve)

            case SearchType.HYBRID_SEARCH:
                results = self.db_manager.hybrid_search(query, collection_name, k_to_retrieve, search_config.alpha)

            case _:
                raise ValueError(f"Unsupported search type: {search_config.search_type}")

        return results

    def get_evaluation_insights(self) -> dict:
        """Get insights about document vs section level performance differences"""
        return self.evaluation_service.get_performance_insights()

    def get_current_metrics_summary(self) -> dict:
        """Get current metrics summary for programmatic access"""
        return self.evaluation_service.get_metrics_summary()

    def clear_evaluation_metrics(self):
        """Clear accumulated evaluation metrics"""
        self.evaluation_service.clear_metrics()