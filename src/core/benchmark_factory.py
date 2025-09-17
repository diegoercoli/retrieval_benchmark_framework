from pathlib import Path
from typing import List

from src.core.configuration import RAGConfiguration
from src.vectordb.weaviate_db_manager import WeaviateDBManager


class BenchmarkFactory:

    def __init__(self, yaml_file: Path):
        self.yaml_file = yaml_file

    def start(self):
        #__bootstrap_embedding_server

        #__create_RAG_configurations

        #for each configuration run a pipeline on a dedicated thread
        pass

    def __create_db_manager(self) -> WeaviateDBManager:
        pass

    def __create_RAG_configuration(self) -> List[RAGConfiguration]:
        pass

    def __bootstrap_embedding_server(self) -> bool:
        pass

    def __run_pipeline(self, RAG_config: RAGConfiguration):
        # execute DocumentIngestionService process

        # execute DocumentRetrievalService retrieve_evaluate
        pass

