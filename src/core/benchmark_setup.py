from pathlib import Path

import yaml

from src.rest_api.dataset import DatasetAPI


class BenchmarkInitializer:
    """
    Handles the initialization and setup phase of RAG benchmarking experiments.
    Prepares all necessary resources before running experiments.
    """

    def __init__(self, config_path: str):
        """
        Initialize the benchmark setup with configuration file.

        Args:
            config_path: Path to the benchmark configuration YAML file
        """
        self.config = None
        self.config_path = config_path
        self.kb_data = None
        self.dataset = None
        self.vector_db = None
        self.configurations = []

    def __read_config(self) -> dict:
        """
        Read and parse the benchmark configuration file.

        Returns:
            dict: Parsed configuration data
        """
        config_path = Path("../../config/benchmark_config.yaml").resolve()
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)


    def __fetch_knowledge_base(self) -> None:
        """
        Fetch or load the knowledge base from backend service.
        Stores KB data for later use.
        """
        pass

    def __read_dataset(self) -> int:
        """
        Read the evaluation dataset (queries, ground truth, etc.).
        """
        client = DatasetAPI(base_url="http://localhost:8000", timeout=30)
        dataset =
        response = client.create_dataset(complex_dataset)

    def __setup_vector_database(self) -> None:
        """
        Initialize and configure the vector database.
        Creates collections, sets up schemas, etc.
        """
        pass

    def __generate_configurations(self) -> list:
        """
        Generate all possible experiment configurations based on config parameters.
        Creates combinations of search strategies, embeddings, reranking, etc.

        Returns:
            list: List of experiment configuration dictionaries
        """
        pass

    def initialize(self) -> bool:
        """
        Main orchestration method that runs all initialization steps in sequence.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.__read_config()
            self.__fetch_knowledge_base()
            self.__read_dataset()
            self.__setup_vector_database()
            self.__generate_configurations()
            return True
        except Exception as e:
            print(f"Initialization failed: {e}")
            return False