from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import List

from pandas.core.interchange.dataframe_protocol import DataFrame
from src.core.configuration import RAGConfiguration, SearchType, SearchConfiguration
from src.core.preprocess_dataset import process_dataset, GroundTruthDocumentRecord
from src.engine.custom_tasks import RetrievalService
from src.utils.dataset_loader import load_benchmark_dataset
from src.utils.hashing_utils import generate_id_from_strings
from src.vectordb.weaviate_db_manager import WeaviateDBManager
import pandas as pd

@dataclass
class QueryEvaluationMetrics:
    query_id: str
    precision: float
    recall: float
    f1_score: float
    ndcg: float  # Normalized Discounted Cumulative Gain


class DocumentRetrievalService(RetrievalService):

    def __init__(self, db_manager: WeaviateDBManager):
        self.db_manager = db_manager

    def retrieve_evaluate(self, config: RAGConfiguration ):
        target_collection = config.collection_name
        df_dataset = process_dataset(Path(config.evaluation_configuration.dataset_path))
        for search in config.search_configs:
            # Switch-case for different search types
            # Loop through each row in the dataset
            for index, row in df_dataset.iterrows():
                question = row['question']
                ground_truth = row['ground_truth']  # List[GroundTruthDocumentRecord]
                complexity = row['complexity']
                chunks =  self.retrieve_chunks(question, config.collection_name, search)

    def __evaluate(self,
                   chunks: List[dict],
                   ground_truth: List[GroundTruthDocumentRecord],
                   search_type: SearchType,
                   config: RAGConfiguration) -> QueryEvaluationMetrics:

        #Generate unique_id based on these parameters:
        id = generate_id_from_strings(search_type, config.model_manager.embedding_model, )
        #you must compute precision, recall f1_score and  NormalizedDiscountedCumulativeGain
        return None





    def retrieve_chunks(self, query: str, collection_name: str, searchConfig: SearchConfiguration) -> list:
        """
        Retrieve chunks using different search strategies.

        Args:
            query: Search query string
            collection_name: Name of the collection to search in
            search_type: Type of search to perform
            top_k: Number of results to return
            alpha: Weight for hybrid search (0.0 = semantic, 1.0 = BM25)

        Returns:
            List of matching chunk properties
        """
        results = []

        match searchConfig.search_type:
            case SearchType.SEMANTIC_SEARCH:
                results = self.db_manager.semantic_search_retrieve(query, collection_name, searchConfig.top_k)

            case SearchType.KEYWORD_SEARCH:
                results = self.db_manager.bm25_retrieve(query, collection_name,  searchConfig.top_k)

            case SearchType.HYBRID_SEARCH:
                results = self.db_manager.hybrid_search(query, collection_name, searchConfig.top_k, searchConfig.alpha)

            case _:
                raise ValueError(f"Unsupported search type: {searchConfig.search_type}")

        return results