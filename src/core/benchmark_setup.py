from pathlib import Path

import yaml

from src.preprocessing.dataset_loader import load_dataset
from src.rest_api.dataset import DatasetAPI


class BenchmarkSetup:
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
        self.config_path = config_path
        self.config = None
        self.kb_data = None
        self.dataset = None
        self.vector_db = None
        self.configurations = []

    def __read_config(self) :
        """
        Read and parse the benchmark configuration file.

        Returns:
            dict: Parsed configuration data
        """
        with open(self.config_path, "r") as file:
            self.config = yaml.safe_load(file)


    def __fetch_knowledge_base(self) -> None:
        """
        Fetch or load the knowledge base from backend service.
        Stores KB data for later use.
        """
        return
        # Create parser instance and process documents
        word_parser = WordParser(
            input_folder_path=config["loading"]["input_folder"],
            output_folder_path=config["loading"]["output_folder"],
        )
        # Parse all files in the input folder
        skipped_files = word_parser.parse_all_files()
        if skipped_files:
            print(f"Skipped {len(skipped_files)} files: {skipped_files}")
        else:
            print("All documents processed successfully")


    def __read_dataset(self) -> int:
        """
        Read the evaluation dataset (queries, ground truth, etc.).
        """
        client = DatasetAPI(base_url=self.config['backend']['url'], timeout=30)
        dataset_path = Path(self.config['evaluation']['dataset_path'])

        # Extract dataset name: use stem if it's a file (has suffix), otherwise use folder name
        dataset_name = dataset_path.stem if dataset_path.suffix else dataset_path.name
        dataset_input = load_dataset(dataset_path, dataset_name)
        response = client.create_dataset(dataset_input)
        return response.id

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
            id_dataset = self.__read_dataset()
            self.__setup_vector_database()
            self.__generate_configurations()
            return True
        except Exception as e:
            print(f"Initialization failed: {e}")
            return False