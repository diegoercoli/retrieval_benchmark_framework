from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Set, List
from docling_core.transforms.chunker import BaseChunker
from src.core.model_manager import ModelManager
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import List
from dataclasses import dataclass

from src.utils.hashing_utils import generate_id_from_strings
from src.utils.name_cleaner import normalize_embedding_name


class SearchType(Enum):
    KEYWORD_SEARCH = 1
    SEMANTIC_SEARCH = 2
    HYBRID_SEARCH = 3

@dataclass
class SearchConfiguration:
    search_type: SearchType
    rerank_enabled: bool
    pre_retrieval: int
    top_k: int
    alpha: float


@dataclass
class PreprocessConfiguration:
    """
    Configuration for text preprocessing.

    Attributes:
        lowercase (bool): Convert all text to lowercase.
        strip_whitespace (bool): Remove leading and trailing whitespace from text.
        remove_punctuation (bool): Remove punctuation from text.
    """
    lowercase: bool = True
    #strip_whitespace: bool = True
    #remove_punctuation: bool = False


@dataclass
class EvaluationConfiguration:
    """
    Configuration for evaluation, including dataset and report paths.

    Attributes:
        dataset_path (Path): Path to the dataset.
        report_path (Path): Path to the report directory.
        allowed_query_complexities (List[str]): Allowed query complexity levels.
        timestamp (str): Local timestamp set at instantiation.
    """
    dataset_path: Path
    report_path: Path
    allowed_query_complexities: List[str]
    timestamp: str = field(init=False)

    def __post_init__(self):
        # Set the timestamp to the current local time in the specified format
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
    preprocess_configuration: PreprocessConfiguration
    chunking : BaseChunker
    model_manager : ModelManager
    search_configs : List[SearchConfiguration]
    main_folder : Path
    evaluation_configuration: EvaluationConfiguration

    def configuration_id(self,search_type: SearchType):
        """Generate unique ID based on configuration parameters"""
        return generate_id_from_strings(
            self.collection_name,
            str(search_type)
        )

    #collection_name as readonly property derived from main_folder name
    @property
    def collection_name(self) -> str:
        str_lowercase = "lowercase" if self.preprocess_configuration.lowercase else "nolowercase"
        embedding_model_name =  self.model_manager.config["embedding_model_name"]
        embedding_short_name =  normalize_embedding_name(embedding_model_name)#embedding_model_name.split('/')[-1].replace('-', '').replace('_', '')
        strategy_name = self.chunking.name
        return f"{strategy_name}_{embedding_short_name}_{str_lowercase}"