from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Set, List

from docling_core.transforms.chunker import BaseChunker

from src.core.model_manager import ModelManager


class SearchType(Enum):
    KEYWORD_SEARCH = 1
    SEMANTIC_SEARCH = 2
    HYBRID_SEARCH = 3

@dataclass
class SearchConfiguration:
    search_type: SearchType
    top_k: int
    alpha: float

from datetime import datetime

class EvaluationConfiguration:
    """
    Configuration for evaluation, including dataset and report paths.

    Attributes:
        dataset_path (Path): Path to the dataset.
        report_path (Path): Path to the report directory.
        timestamp (str): Local timestamp set at instantiation.
    """
    def __init__(self, dataset_path: Path, report_path: Path):
        self.dataset_path = dataset_path
        self.report_path = report_path
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    @property
    def report_path_with_timestamp(self) -> Path:
        """
        Returns the report path joined with the timestamp.

        Returns:
            Path: The report path with the timestamp appended.
        """
        return self.report_path / self.timestamp

@dataclass
class RAGConfiguration:
    r"""Configuration of RAG."""
    chunking : BaseChunker
    model_manager : ModelManager
    collection_name : str
    search_configs : List[SearchConfiguration]
    main_folder : Path
    evaluation_configuration: EvaluationConfiguration
