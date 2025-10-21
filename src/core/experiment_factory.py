"""
experiment_factory.py

Refactored BenchmarkFactory with extracted pipeline execution logic.
"""
import time
from pathlib import Path
from typing import List
import yaml
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
from src.core.configuration import RAGConfiguration, SearchConfiguration, SearchType, EvaluationConfiguration, PreprocessConfiguration
from src.core.model_manager import ModelManager
from src.preprocessing.docling_utils import MDTableSerializerProvider
from src.utils.proxy_helper import set_proxy_authentication
from src.vectordb.weaviate_db_manager import WeaviateDBManager
from src.chunking.CustomChunker import CustomChunker
from src.vectordb.embedding_service import start_server
from src.core.pipeline_runner import PipelineRunner, PipelineExecutionStrategy


class BenchmarkFactory:
    """
    Main factory class that orchestrates the RAG benchmarking pipeline with enhanced configuration tracking.

    This class:
    1. Loads configuration from YAML
    2. Starts embedding/reranking servers
    3. Creates different RAG configurations for benchmarking
    4. Delegates pipeline execution to PipelineRunner
    5. Generates enhanced reports with configuration mapping

    Responsibilities:
    - Configuration management
    - RAG configuration creation
    - Server bootstrapping
    - Pipeline coordination (delegated to PipelineRunner)
    """

    def __init__(self, yaml_file: Path):
        self.config = self._load_config(yaml_file)
        self.embedding_server_thread = None
        self.rag_configurations = []  # Store all created configurations for reporting
        self.pipeline_runner = None  # Will be initialized after DB manager creation

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

    def start(self, execution_strategy: str = 'consolidated'):
        """
        Main entry point to start the benchmarking process with enhanced reporting.

        Args:
            execution_strategy: Either 'individual' or 'consolidated' (default)
        """
        print("Starting RAG benchmark factory...")

        # Step 1: Bootstrap embedding server (commented out as per original)
        # if not self._bootstrap_embedding_server():
        #    raise RuntimeError("Failed to start embedding server")

        # Step 2: Create database manager and pipeline runner
        db_manager = self._create_db_manager()
        self.pipeline_runner = PipelineRunner(db_manager)

        # Step 3: Create RAG configurations for different combinations
        rag_configs = self._create_RAG_configurations()
        self.rag_configurations = rag_configs  # Store for later use in reporting

        print(f"Created {len(rag_configs)} RAG configurations to benchmark")

        # Step 4: Execute pipelines using the specified strategy
        self._execute_pipelines(rag_configs, execution_strategy)

        print("All benchmark pipelines completed with enhanced reporting!")

    def _execute_pipelines(self, rag_configs: List[RAGConfiguration], strategy: str):
        """
        Execute pipelines using the specified strategy.

        Args:
            rag_configs: List of RAG configurations to benchmark
            strategy: Execution strategy ('individual' or 'consolidated')
        """
        # Get the strategy method name
        strategy_method_name = PipelineExecutionStrategy.get_strategy(strategy)

        # Get the actual method from the pipeline runner
        strategy_method = getattr(self.pipeline_runner, strategy_method_name)

        print(f"Executing pipelines using '{strategy}' strategy...")
        strategy_method(rag_configs)

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
            report_path=Path(self.config['evaluation'].get('report_path', '../report')).resolve(),
            allowed_query_complexities=self.config['evaluation'].get('allowed_query_complexities', ['text'])
        )

        # Get chunking strategies
        chunking_strategies = self.config['chunking']['strategies']
        search_strategies = self.config['vector_database']['search_strategies']

        # Create search configurations
        pre_retrieval = self.config['vector_database'].get('pre_retrieval', 10)
        top_k = self.config['vector_database'].get('top_k', 10)
        search_configs = self._create_search_configurations(search_strategies, pre_retrieval=pre_retrieval, top_k=top_k)

        # Create PreprocessConfiguration with YAML values or defaults
        preprocess_config = PreprocessConfiguration(
            lowercase = self.config['preprocessing'].get('lowercase', True),
            #strip_whitespace = self.config['preprocessing'].get('strip_whitespace', True),
            #remove_punctuation = self.config['preprocessing'].get('remove_punctuation', False
        )

        # Generate configurations for each combination
        for chunk_strategy in chunking_strategies:
            # Create chunker based on strategy
            chunker = self._create_chunker(chunk_strategy, embedding_model_name, preprocess_config)

            # Generate unique collection name
            #strategy_name = chunk_strategy['name']
            #collection_base_name = self.config['vector_database']['collection_name_suffix']

            # Create a cleaner collection name for the dual-level system
            #embedding_short_name = embedding_model_name.split('/')[-1].replace('-', '').replace('_', '')
           # collection_name = f"{strategy_name}chunking{embedding_short_name}"

            # Create RAG configuration
            rag_config = RAGConfiguration(
                chunking=chunker,
                #collection_name = collection_name,
                model_manager=model_manager,
                search_configs=search_configs,
                main_folder=main_folder,
                evaluation_configuration=eval_config,
                preprocess_configuration=preprocess_config
            )

            configurations.append(rag_config)

        return configurations

    def _create_chunker(self, chunk_strategy: dict, embedding_model: str, preprocessing_configuration: PreprocessConfiguration) -> CustomChunker:
        """Create chunker based on strategy configuration"""
        strategy_name = chunk_strategy['name']
        blacklist_chapters = self.config['chunking'].get('blacklist_chapters', [])

        if strategy_name == 'hierarchical':
            set_proxy_authentication()
            tokenizer = HuggingFaceTokenizer(
                tokenizer=AutoTokenizer.from_pretrained(embedding_model),
                max_tokens=chunk_strategy['num_tokens'],  # optional, by default derived from `tokenizer` for HF case
            )
            # Wrap in custom chunker
            # Create CustomChunker with parameters that match the constructor signature
            return CustomChunker(
                name=f"{strategy_name}_chunking",  # Add name parameter
                preprocess_configuration=preprocessing_configuration,  # <-- fixed name
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

    def _create_search_configurations(self, search_strategies: list, pre_retrieval: int,top_k:int) -> list:
        """Convert search strategy configs to SearchConfiguration objects"""
        search_configs = []  # Changed from set to list

        for strategy in search_strategies:
            if 'semantic' in strategy:
                search_configs.append(SearchConfiguration(
                    search_type=SearchType.SEMANTIC_SEARCH,
                    top_k=top_k,
                    rerank_enabled= strategy['semantic'].get('rerank_enabled', True),
                    pre_retrieval=pre_retrieval,
                    alpha=0.0  # Not used for semantic search
                ))

            elif 'keyword' in strategy:
                search_configs.append(SearchConfiguration(
                    search_type=SearchType.KEYWORD_SEARCH,
                    top_k=top_k,
                    rerank_enabled=strategy['keyword'].get('rerank_enabled', True),
                    pre_retrieval=pre_retrieval,
                    alpha=0.0  # Not used for keyword search
                ))

            elif 'hybrid' in strategy:
                search_configs.append(SearchConfiguration(
                    search_type=SearchType.HYBRID_SEARCH,
                    top_k=top_k,
                    rerank_enabled=strategy['hybrid'].get('rerank_enabled', True),
                    pre_retrieval=pre_retrieval,
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

    def get_pipeline_runner(self) -> PipelineRunner:
        """
        Get the pipeline runner instance.

        Returns:
            PipelineRunner instance

        Raises:
            RuntimeError: If called before start() method
        """
        if self.pipeline_runner is None:
            raise RuntimeError("Pipeline runner not initialized. Call start() method first.")
        return self.pipeline_runner