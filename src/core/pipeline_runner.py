"""
pipeline_runner.py

Extracted pipeline execution logic from BenchmarkFactory.
Handles running individual RAG configurations and reporting results.
"""

import time
from typing import List, Dict, Any
from src.core.configuration import RAGConfiguration
from src.core.data_ingestion import DocumentIngestionService
from src.retrieval.data_retrieval import DocumentRetrievalService
from src.vectordb.weaviate_db_manager import WeaviateDBManager


class PipelineRunner:
    """
    Handles execution of RAG benchmarking pipelines.

    Separated from BenchmarkFactory to follow Single Responsibility Principle.
    """

    def __init__(self, db_manager: WeaviateDBManager):
        """
        Initialize the pipeline runner.

        Args:
            db_manager: Weaviate database manager instance
        """
        self.db_manager = db_manager

    def run_individual_pipelines(self, rag_configs: List[RAGConfiguration]):
        """Run pipelines individually (original approach)"""
        import threading

        threads = []
        for i, rag_config in enumerate(rag_configs):
            print(f"Starting pipeline {i + 1}/{len(rag_configs)}: {rag_config.collection_name}")

            # Create thread for each configuration
            thread = threading.Thread(
                target=self.run_pipeline,
                args=(rag_config,),
                name=f"Pipeline-{rag_config.collection_name}"
            )
            threads.append(thread)
            thread.start()

            # Optional: Add small delay between thread starts to avoid resource contention
            time.sleep(1)

        # Wait for all threads to complete
        for thread in threads:
            thread.join()


    def run_pipelines_with_consolidated_reporting(self, rag_configs: List[RAGConfiguration]):
        """
        Run pipelines with consolidated reporting across all configurations.

        Handles database connection errors gracefully and ensures connection is closed.
        """
        report_path = rag_configs[0].evaluation_configuration.report_path_with_timestamp
        timestamp = rag_configs[0].evaluation_configuration.timestamp

        try:
            # Step 0: Open connection
            self.db_manager.connect()

            # Step 1: Run ingestion for all configurations
            print("\n=== INGESTION PHASE ===")
            ingestion_start = time.time()
            for i, rag_config in enumerate(rag_configs):
                print(f"Running ingestion {i + 1}/{len(rag_configs)}: {rag_config.collection_name}")
                ingestion_service = DocumentIngestionService(self.db_manager)
                processed_count = ingestion_service.process(rag_config, overwrite=True)
                print(f"Ingested {processed_count} chunks for {rag_config.collection_name}")
            ingestion_time = time.time() - ingestion_start

            # Step 2: Run evaluation for all configurations with consolidated reporting
            print(f"\n=== EVALUATION PHASE ===")
            evaluation_start = time.time()
            retrieval_service = DocumentRetrievalService(self.db_manager, rag_configs[0].evaluation_configuration)
            evaluation_summary = retrieval_service.retrieve_evaluate_multiple_configs(rag_configs)
            evaluation_time = time.time() - evaluation_start

            # Step 3: Print final summary
            self.print_final_summary(evaluation_summary, len(rag_configs))

            # Write execution time file
            exec_time_file = report_path / f"execution_time_{timestamp}.txt"
            with open(exec_time_file, "w") as f:
                f.write(f"Ingestion Time: {ingestion_time:.2f} seconds\n")
                f.write(f"Evaluation Time: {evaluation_time:.2f} seconds\n")

        except Exception as e:
            print(f"Error during pipeline execution: {e}")

        finally:
            # Step 4: Close connection
            self.db_manager.close()

    def print_final_summary(self, evaluation_summary: Dict[str, Any], num_configs: int):
        """Print final summary of all evaluations"""
        print(f"\n" + "=" * 100)
        print(f"🏁 BENCHMARK FACTORY COMPLETED - FINAL SUMMARY")
        print(f"=" * 100)
        print(f"📊 Configurations evaluated: {num_configs}")
        print(f"📋 Total queries processed: {evaluation_summary.get('total_queries', 0)}")

        if evaluation_summary.get('best_configurations'):
            print(f"\n🏆 OVERALL BEST PERFORMERS:")
            best_configs = evaluation_summary.get('best_configurations', {})

            # Show top document level performers
            print(f"\n📄 Document Level Champions:")
            for metric in ['document_precision', 'document_recall', 'document_f1_score', 'document_ndcg']:
                if metric in best_configs and best_configs[metric]:
                    config_id, value = best_configs[metric]
                    short_config = config_id[:25] + "..." if len(config_id) > 28 else config_id
                    metric_display = metric.replace('document_', '').replace('_', ' ').title()
                    print(f"  🥇 {metric_display}: {short_config} ({value:.4f})")

            # Show top section level performers
            print(f"\n📑 Section Level Champions:")
            for metric in ['section_precision', 'section_recall', 'section_f1_score', 'section_ndcg']:
                if metric in best_configs and best_configs[metric]:
                    config_id, value = best_configs[metric]
                    short_config = config_id[:25] + "..." if len(config_id) > 28 else config_id
                    metric_display = metric.replace('section_', '').replace('_', ' ').title()
                    print(f"  🥇 {metric_display}: {short_config} ({value:.4f})")

        print(f"\n📁 Reports generated with:")
        print(f"  • Configuration mapping tables showing embedder, search strategy, and chunking details")
        print(f"  • Dual-level metrics (document vs section level evaluation)")
        print(f"  • Enhanced Excel reports with specialized sheets")
        print(f"  • HTML reports with configuration tables")
        print(f"  • JSON reports with programmatic access to configuration details")
        print(f"=" * 100)


class PipelineExecutionStrategy:
    """
    Strategy pattern for different pipeline execution approaches.
    """

    @staticmethod
    def get_strategy(strategy_name: str):
        """
        Get execution strategy by name.

        Args:
            strategy_name: Either 'individual' or 'consolidated'

        Returns:
            Method reference to the appropriate execution strategy
        """
        strategies = {
            'individual': 'run_individual_pipelines',
            'consolidated': 'run_pipelines_with_consolidated_reporting'
        }

        if strategy_name not in strategies:
            raise ValueError(f"Unknown execution strategy: {strategy_name}. "
                             f"Available strategies: {list(strategies.keys())}")

        return strategies[strategy_name]