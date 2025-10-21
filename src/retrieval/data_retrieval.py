from abc import ABC, abstractmethod
from typing import List
from src.core.configuration import RAGConfiguration, SearchType, SearchConfiguration, EvaluationConfiguration
from src.retrieval.data_evaluation import EvaluationService
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

    def __init__(self, db_manager: WeaviateDBManager, evaluationConfiguration: EvaluationConfiguration):
        self.db_manager = db_manager
        # Use the new dual-level evaluation service
        self.evaluation_service = EvaluationService(evaluationConfiguration)

    def retrieve_evaluate(self, config: RAGConfiguration):
        """Enhanced retrieve and evaluate method using dual-level evaluation service."""

        print(f"⏱️  Starting retrieval/evaluation for '{config.collection_name}'")
        # start_time = time.time()

        # Use the new evaluation service with injected retrieval function
        self.evaluation_service.evaluate_retrieval_results(
            config=config,
            retrieval_function=self._create_retrieval_function()
        )

        # Generate comprehensive reports with configuration mapping
        #self.evaluation_service.generate_reports(config)

        # Clear metrics for next collection (if processing multiple)
        # Note: Don't clear here if you want to aggregate across collections
        # self.evaluation_service.clear_metrics()

        # Return summary for programmatic access
        return self.evaluation_service.get_metrics_summary()

    def retrieve_evaluate_multiple_configs(self, configs: List[RAGConfiguration]):
        """
        Enhanced method to evaluate multiple configurations and generate a consolidated report.

        Args:
            configs: List of RAGConfiguration objects to evaluate

        Returns:
            Summary of evaluation results across all configurations
        """
        print(f"⏱️  Starting retrieval/evaluation for {len(configs)} configurations")

        # Clear any previous metrics
        self.evaluation_service.clear_metrics()

        # Process each configuration
        for i, config in enumerate(configs, 1):
            print(f"\n--- Processing configuration {i}/{len(configs)}: {config.collection_name} ---")

            # Use the evaluation service with injected retrieval function
            self.evaluation_service.evaluate_retrieval_results(
                config=config,
                retrieval_function=self._create_retrieval_function()
            )

        # Generate consolidated reports for all configurations
        print(f"\n--- Generating consolidated reports for all {len(configs)} configurations ---")
        #self.evaluation_service.generate_reports_with_multiple_configs(configs)

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
            return self.retrieve_chunks(query, collection_name, search_config)

        return retrieval_function

    def retrieve_chunks(self, query: str, collection_name: str, search_config: SearchConfiguration) -> List[dict]:
        """
        Retrieve chunks using different search strategies.
        Always retrieves fixed_k=50 chunks for comprehensive evaluation.

        Args:
            query: Search query string
            collection_name: Name of the collection to search in
            search_config: Search configuration with type and parameters

        Returns:
            List of matching chunk properties (always 50 chunks if available)
        """
        results = []

        # Always retrieve 50 chunks regardless of configured top_k
        # The evaluation service will use search_config.top_k for precision/recall
        # but will use all 50 for NDCG and MRR calculations
        k_to_retrieve = search_config.pre_retrieval

        match search_config.search_type:
            case SearchType.SEMANTIC_SEARCH:
                results = self.db_manager.semantic_search_retrieve(query, collection_name, search_config.rerank_enabled, search_config.pre_retrieval, search_config.top_k)

            case SearchType.KEYWORD_SEARCH:
                results = self.db_manager.bm25_retrieve(query, collection_name, search_config.rerank_enabled, search_config.pre_retrieval, search_config.top_k)

            case SearchType.HYBRID_SEARCH:
                #call the method by specifying the attribute names
                results = self.db_manager.hybrid_search(
                    query=query,
                    collection_name=collection_name,
                    use_rerank=search_config.rerank_enabled,
                    pre_retrieval=search_config.pre_retrieval,
                    top_k=search_config.top_k,
                    alpha = search_config.alpha
                )

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

    def set_report_configuration_mappings(self, config_mappings: dict):
        """
        Set configuration mappings for enhanced reporting.

        Args:
            config_mappings: Dictionary mapping configuration_id to config details
        """
        self.evaluation_service.report_generator.set_configuration_mappings(config_mappings)