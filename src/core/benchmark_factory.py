import threading
import time
from pathlib import Path
from typing import List
import yaml
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

from src.core.configuration import RAGConfiguration, SearchConfiguration, SearchType, EvaluationConfiguration
from src.core.model_manager import ModelManager
from src.core.data_ingestion import DocumentIngestionService
from src.core.data_retrieval import DocumentRetrievalService
from src.utils.docling_utils import MDTableSerializerProvider
from src.vectordb.weaviate_db_manager import WeaviateDBManager
from src.chunking.CustomChunker import CustomChunker
from src.vectordb.embedding_service import start_server


class BenchmarkFactory:
    """
    Main factory class that orchestrates the RAG benchmarking pipeline with enhanced configuration tracking.

    This class:
    1. Loads configuration from YAML
    2. Starts embedding/reranking servers
    3. Creates different RAG configurations for benchmarking
    4. Runs ingestion and evaluation pipelines
    5. Generates enhanced reports with configuration mapping
    """

    def __init__(self, yaml_file: Path):
        self.config = self._load_config(yaml_file)
        self.embedding_server_thread = None
        self.rag_configurations = []  # Store all created configurations for reporting

    def _load_config(self, yaml_file: Path) -> dict:
        """Load and validate YAML configuration"""
        if not yaml_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_file}")

        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)

        # Basic validation
        required_sections = ['embedding', 'chunking', 'vector_database', 'evaluation']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")

        return config

    def start(self):
        """Main entry point to start the benchmarking process with enhanced reporting"""
        print("Starting RAG benchmark factory...")

        # Step 1: Bootstrap embedding server (commented out as per original)
        # if not self._bootstrap_embedding_server():
        #    raise RuntimeError("Failed to start embedding server")

        # Step 2: Create database manager
        db_manager = self._create_db_manager()

        # Step 3: Create RAG configurations for different combinations
        rag_configs = self._create_RAG_configurations()
        self.rag_configurations = rag_configs  # Store for later use in reporting

        print(f"Created {len(rag_configs)} RAG configurations to benchmark")

        # Step 4: Option A - Run pipelines individually (original approach)
        # self._run_individual_pipelines(rag_configs, db_manager)

        # Step 4: Option B - Run pipelines with consolidated reporting (enhanced approach)
        self._run_pipelines_with_consolidated_reporting(rag_configs, db_manager)

        print("All benchmark pipelines completed with enhanced reporting!")

    def _run_individual_pipelines(self, rag_configs: List[RAGConfiguration], db_manager: WeaviateDBManager):
        """Run pipelines individually (original approach)"""
        threads = []
        for i, rag_config in enumerate(rag_configs):
            print(f"Starting pipeline {i + 1}/{len(rag_configs)}: {rag_config.collection_name}")

            # Create thread for each configuration
            thread = threading.Thread(
                target=self._run_pipeline,
                args=(rag_config, db_manager),
                name=f"Pipeline-{rag_config.collection_name}"
            )
            threads.append(thread)
            thread.start()

            # Optional: Add small delay between thread starts to avoid resource contention
            time.sleep(1)

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

    def _run_pipelines_with_consolidated_reporting(self, rag_configs: List[RAGConfiguration],
                                                   db_manager: WeaviateDBManager):
        """Run pipelines with consolidated reporting across all configurations"""

        # Step 1: Run ingestion for all configurations
        print("\n=== INGESTION PHASE ===")
        for i, rag_config in enumerate(rag_configs):
            print(f"Running ingestion {i + 1}/{len(rag_configs)}: {rag_config.collection_name}")
            ingestion_service = DocumentIngestionService(db_manager)
            processed_count = ingestion_service.process(rag_config)
            print(f"Ingested {processed_count} chunks for {rag_config.collection_name}")

        # Step 2: Run evaluation for all configurations with consolidated reporting
        print(f"\n=== EVALUATION PHASE ===")
        retrieval_service = DocumentRetrievalService(db_manager)
        evaluation_summary = retrieval_service.retrieve_evaluate_multiple_configs(rag_configs)

        # Step 3: Print final summary
        self._print_final_summary(evaluation_summary, len(rag_configs))

    def _print_final_summary(self, evaluation_summary: dict, num_configs: int):
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

    def _create_db_manager(self) -> WeaviateDBManager:
        """Create and configure Weaviate database manager"""
        vector_db_config = self.config['vector_database']

        # Extract ports (use defaults if not specified)
        port = vector_db_config.get('port', 8081)
        grpc_port = vector_db_config.get('grpc_port', 50051)

        # Get inference URL from embedding config
        inference_url = self.config['embedding']['transformers_inference_api']

        return WeaviateDBManager(
            port=port,
            grpc_port=grpc_port,
            inference_url=inference_url
        )

    def _create_RAG_configurations(self) -> List[RAGConfiguration]:
        """
        Create different RAG configurations by combining:
        - Different chunking strategies
        - Different search strategies
        - Same embedding model (from config)
        """
        configurations = []

        # Get base paths
        main_folder = Path(self.config['loading']['output_folder']).resolve()

        embedding_model_name = self.config['embedding']['model']

        # Create model manager
        model_manager = ModelManager(
            embedding_model_name=embedding_model_name,
            reranker_model_name=self.config['reranking']['model']
        )

        # Create evaluation configuration
        eval_config = EvaluationConfiguration(
            dataset_path=Path(self.config['evaluation']['dataset_path']).resolve(),
            report_path=Path(self.config['evaluation']['report_path']).resolve()
        )

        # Get chunking strategies
        chunking_strategies = self.config['chunking']['strategies']
        search_strategies = self.config['vector_database']['search_strategies']

        # Create search configurations
        search_configs = self._create_search_configurations(search_strategies)

        # Generate configurations for each combination
        for chunk_strategy in chunking_strategies:
            # Create chunker based on strategy
            chunker = self._create_chunker(chunk_strategy, embedding_model_name)

            # Generate unique collection name
            strategy_name = chunk_strategy['name']
            collection_base_name = self.config['vector_database']['collection_name_suffix']

            # Create a cleaner collection name for the dual-level system
            embedding_short_name = embedding_model_name.split('/')[-1].replace('-', '').replace('_', '')
            collection_name = f"{strategy_name}chunking{embedding_short_name}"

            # Create RAG configuration
            rag_config = RAGConfiguration(
                chunking=chunker,
                model_manager=model_manager,
                collection_name=collection_name,
                search_configs=search_configs,
                main_folder=main_folder,
                evaluation_configuration=eval_config
            )

            configurations.append(rag_config)

        return configurations

    def _create_chunker(self, chunk_strategy: dict, embedding_model: str) -> CustomChunker:
        """Create chunker based on strategy configuration"""
        strategy_name = chunk_strategy['name']
        blacklist_chapters = self.config['chunking'].get('blacklist_chapters', [])

        if strategy_name == 'hierarchical':
            tokenizer = HuggingFaceTokenizer(
                tokenizer=AutoTokenizer.from_pretrained(embedding_model),
                max_tokens=chunk_strategy['num_tokens'],  # optional, by default derived from `tokenizer` for HF case
            )
            # Wrap in custom chunker
            # Create CustomChunker with parameters that match the constructor signature
            return CustomChunker(
                name=f"{strategy_name}_chunking",  # Add name parameter
                blacklist_chapters=blacklist_chapters,  # Pass as constructor parameter
                tokenizer=tokenizer,
                serializer_provider=MDTableSerializerProvider(),
                merge_peers=True,
            )
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy_name}")

        ''' 
        elif strategy_name == 'fixed_size':
            return CustomChunker(
                blacklist_chapters=blacklist_chapters,
                tokenizer="tiktoken",
                max_tokens=chunk_strategy.get('min_size', 135)
            )

        elif strategy_name == 'overlapping':
            # For overlapping, you might need to implement overlap logic
            return CustomChunker(
                blacklist_chapters=blacklist_chapters,
                tokenizer="tiktoken",
                max_tokens=chunk_strategy.get('min_size', 135)
                # Note: overlap logic would need to be implemented in CustomChunker
            )
        '''

    def _create_search_configurations(self, search_strategies: list) -> list:
        """Convert search strategy configs to SearchConfiguration objects"""
        search_configs = []  # Changed from set to list

        for strategy in search_strategies:
            if 'semantic' in strategy:
                search_configs.append(SearchConfiguration(
                    search_type=SearchType.SEMANTIC_SEARCH,
                    top_k=int(strategy['semantic']['top_k']),
                    alpha=0.0  # Not used for semantic search
                ))

            elif 'keyword' in strategy:
                search_configs.append(SearchConfiguration(
                    search_type=SearchType.KEYWORD_SEARCH,
                    top_k=strategy['keyword']['top_k'],
                    alpha=0.0  # Not used for keyword search
                ))

            elif 'hybrid' in strategy:
                search_configs.append(SearchConfiguration(
                    search_type=SearchType.HYBRID_SEARCH,
                    top_k=strategy['hybrid']['top_k'],
                    alpha=strategy['hybrid']['alpha']
                ))

        return search_configs

    def _bootstrap_embedding_server(self) -> bool:
        """Start the embedding and reranking server"""
        try:
            # Create model manager for the server
            embedding_model = self.config['embedding']['model']
            reranking_model = self.config['reranking']['model']

            server_model_manager = ModelManager(
                embedding_model_name=embedding_model,
                reranker_model_name=reranking_model
            )

            # Extract server configuration
            server_url = self.config['embedding']['transformers_inference_api']
            # Parse host and port from URL (e.g., 'http://172.20.64.1:5000')
            if '://' in server_url:
                host_port = server_url.split('://')[-1]
            else:
                host_port = server_url

            if ':' in host_port:
                host, port_str = host_port.split(':')
                port = int(port_str)
            else:
                host = host_port
                port = 5000

            # Start server in thread
            print(f"Starting embedding server at {host}:{port}")
            self.embedding_server_thread = start_server(
                configured_model_manager=server_model_manager,
                host=host,
                port=port,
                use_thread=True
            )

            # Wait a moment for server to start
            time.sleep(3)

            # Basic health check (you could implement a more robust check)
            print("Embedding server started successfully")
            return True

        except Exception as e:
            print(f"Failed to start embedding server: {e}")
            return False

    def _run_pipeline(self, rag_config: RAGConfiguration, db_manager: WeaviateDBManager):
        """
        Execute the complete pipeline for a single RAG configuration:
        1. Document ingestion (chunking and vector storage)
        2. Document retrieval and evaluation with dual-level metrics
        """
        try:
            print(f"Starting pipeline for {rag_config.collection_name}")

            # Step 1: Execute Document Ingestion Service
            print(f"Starting document ingestion for {rag_config.collection_name}")
            ingestion_start = time.time()
            ingestion_service = DocumentIngestionService(db_manager)

            # Process documents and store in vector database
            processed_count = ingestion_service.process(rag_config)
            ingestion_time = time.time() - ingestion_start
            print(f"Ingested {processed_count} chunks for {rag_config.collection_name} in {ingestion_time:.2f} seconds")

            # Step 2: Execute Document Retrieval Service with Dual-Level Evaluation
            print(f"Starting dual-level evaluation for {rag_config.collection_name}")
            retrieval_start = time.time()
            retrieval_service = DocumentRetrievalService(db_manager)

            # Run dual-level evaluation against ground truth
            # This now includes both document-level and section-level metrics
            evaluation_summary = retrieval_service.retrieve_evaluate(rag_config)
            retrieval_time = time.time() - retrieval_start
            print(f"Dual-level evaluation completed for {rag_config.collection_name} in {retrieval_time:.2f} seconds")

            # Print summary of evaluation results
            if evaluation_summary:
                print(f"Evaluation Summary for {rag_config.collection_name}:")
                print(f"  Total queries evaluated: {evaluation_summary.get('total_queries', 0)}")

                best_configs = evaluation_summary.get('best_configurations', {})
                if best_configs:
                    print("  Best performers:")
                    for level in ['document', 'section']:
                        print(f"    {level.title()} Level:")
                        for metric in ['precision', 'recall', 'f1_score', 'ndcg']:
                            key = f'{level}_{metric}'
                            if key in best_configs and best_configs[key]:
                                config_id, value = best_configs[key]
                                print(f"      Best {metric}: {value:.4f}")

            print(f"Pipeline completed for {rag_config.collection_name}")

        except Exception as e:
            print(f"Pipeline failed for {rag_config.collection_name}: {e}")
            import traceback
            traceback.print_exc()
            raise

    def get_configuration_summary(self) -> dict:
        """
        Get a summary of all created configurations for reporting purposes.

        Returns:
            Dictionary with configuration details for reporting
        """
        summary = {
            'total_configurations': len(self.rag_configurations),
            'embedding_model': self.config['embedding']['model'],
            'reranking_model': self.config['reranking']['model'],
            'configurations': []
        }

        for config in self.rag_configurations:
            config_info = {
                'collection_name': config.collection_name,
                'embedder': config.model_manager.config.get('embedding_model_name', 'Unknown'),
                'chunking_strategy': getattr(config.chunking, 'name', getattr(config.chunking, '_name', 'Unknown')),
                'search_strategies': [sc.search_type.name for sc in config.search_configs]
            }
            summary['configurations'].append(config_info)

        return summary