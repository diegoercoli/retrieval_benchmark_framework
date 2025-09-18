import time
from abc import ABC, abstractmethod
from typing import List

from src.core.configuration import RAGConfiguration, SearchType, SearchConfiguration
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
    """
    Service focused solely on document retrieval operations.
    Delegates evaluation to EvaluationService (SRP compliance).

    Implements 50-chunk retrieval strategy for comprehensive evaluation:
    - Always retrieves 50 chunks for ranking metrics (NDCG, MRR)
    - Uses configured top_k for precision/recall fairness
    """

    def __init__(self, db_manager: WeaviateDBManager):
        self.db_manager = db_manager
        self.evaluation_service = EvaluationService()

    def retrieve_evaluate(self, config: RAGConfiguration):
        """
        Execute retrieval and evaluation pipeline.
        Uses dependency injection to separate retrieval from evaluation.
        """
        print(f"\n🔍 Starting retrieval and evaluation for: {config.collection_name}")
        print(f"📊 Strategy: Retrieve 50 chunks per query, evaluate with configured top_k")
        start_time = time.time()

        # Delegate evaluation to specialized service, injecting retrieval function
        self.evaluation_service.evaluate_retrieval_results(
            config=config,
            retrieval_function=self.retrieve_chunks  # Inject our retrieval method
        )

        # Generate reports through evaluation service
        self.evaluation_service.generate_reports(config)
        # Calculate and display ingestion time
        retrieval_time = time.time() - start_time
        print(
            f"⏱️  Retrieval/Evaluation completed for '{config.collection_name}' in {retrieval_time:.2f} seconds.")

    def retrieve_chunks(self, query: str, collection_name: str, search_config: SearchConfiguration) -> List[dict]:
        """
        Core retrieval logic - always retrieves 50 chunks for comprehensive evaluation.

        Strategy:
        - Always retrieve 50 chunks from database for ranking metrics (NDCG, MRR)
        - Precision/Recall use only top_k chunks from search_config
        - This allows better ranking evaluation while respecting configuration limits

        Args:
            query: Search query string
            collection_name: Name of the collection to search in
            search_config: Search configuration containing type, top_k, and alpha

        Returns:
            List of 50 matching chunk properties (or fewer if not available)
        """
        # Always retrieve 50 chunks regardless of configured top_k
        RANKING_EVALUATION_SIZE = 50

        results = []

        match search_config.search_type:
            case SearchType.SEMANTIC_SEARCH:
                results = self.db_manager.semantic_search_retrieve(
                    query, collection_name, RANKING_EVALUATION_SIZE
                )

            case SearchType.KEYWORD_SEARCH:
                results = self.db_manager.bm25_retrieve(
                    query, collection_name, RANKING_EVALUATION_SIZE
                )

            case SearchType.HYBRID_SEARCH:
                results = self.db_manager.hybrid_search(
                    query, collection_name, RANKING_EVALUATION_SIZE, search_config.alpha
                )

            case _:
                raise ValueError(f"Unsupported search type: {search_config.search_type}")

        # Log retrieval info for debugging
        if len(results) < RANKING_EVALUATION_SIZE:
            print(f"⚠️  Retrieved only {len(results)} chunks (expected {RANKING_EVALUATION_SIZE})")

        return results

    # Delegate these methods to evaluation service for backward compatibility
    def clear_metrics(self):
        """Clear accumulated metrics"""
        self.evaluation_service.clear_metrics()

    def get_metrics_summary(self) -> dict:
        """Get metrics summary"""
        return self.evaluation_service.get_metrics_summary()

    def get_performance_insights(self) -> dict:
        """Get performance insights about dual-level evaluation"""
        return self.evaluation_service.get_performance_insights()