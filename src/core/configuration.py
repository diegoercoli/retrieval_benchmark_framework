from dataclasses import dataclass
from pathlib import Path
from typing import Set

from docling_core.transforms.chunker import BaseChunker

from src.core.model_manager import ModelManager


@dataclass
class SearchType:
    KEYWORD_SEARCH = 1
    SEMANTIC_SEARCH = 2
    HYBRID_SEARCH = 3

@dataclass
class SearchConfiguration:
    search_type: SearchType
    top_k: int
    alpha: float

@dataclass
class EvaluationConfiguration:
    dataset_path: Path

@dataclass
class RAGConfiguration:
    r"""Configuration of RAG."""
    chunking : BaseChunker
    model_manager : ModelManager
    collection_name : str
    search_configs : Set[SearchConfiguration]
    main_folder : Path
    evaluation_configuration: EvaluationConfiguration