import os
import threading
import time
from typing import Dict, Any, Optional

import torch
from sentence_transformers import SentenceTransformer

from src.utils.proxy_helper import set_proxy_authentication


class FlagReranker:
    """
    A class representing a reranker model for scoring query-document pairs.

    Attributes:
        model_name (str): The name of the reranker model.
        cache_dir (str): The directory where the model is cached. Defaults to '.models/'.
        use_fp16 (bool): Whether to use FP16 precision for the model. Defaults to True.
    """
    def __init__(self, model_name: str, cache_dir: str = '.models/', use_fp16: bool = True):
        """
        Initialize the FlagReranker with the specified model name, cache directory, and precision setting.

        Args:
            model_name (str): The name of the reranker model.
            cache_dir (str, optional): The directory where the model is cached. Defaults to '.models/'.
            use_fp16 (bool, optional): Whether to use FP16 precision for the model. Defaults to True.
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.use_fp16 = use_fp16

    def compute_score(self, pairs):
        """
        Compute scores for a list of query-document pairs.

        Args:
            pairs (list): A list of tuples, where each tuple contains a query and a document.

        Returns:
            list: A list of scores, one for each query-document pair. Currently returns a placeholder score of 0.5.
        """
        # Placeholder implementation
        return [0.5] * len(pairs)


class ModelManager:
    """Centralized model management with lazy loading and error handling"""

    def __init__(self,
                 embedding_model_name: str = "BAAI/bge-base-en-v1.5",
                 reranker_model_name: str = "BAAI/bge-reranker-base",
                 config: Dict[str, Any] = None):
        self._embedding_model: Optional[SentenceTransformer] = None
        self._reranker: Optional[FlagReranker] = None
        self._model_lock = threading.Lock()
        self._initialization_status = {
            'embedding': False,
            'reranker': False
        }
        self._initialization_errors = {}

        # Initialize configuration
        self._init_config(embedding_model_name, reranker_model_name, config)

    def _init_config(self, embedding_model_name: str, reranker_model_name: str, custom_config: Dict[str, Any] = None):
        """Initialize configuration with model names"""

        # Default configuration
        self.config = {
            'embedding_model_name': embedding_model_name,
            'reranker_model_name': reranker_model_name,
            'cache_dir': os.getenv('MODEL_CACHE_DIR', '.models/'),
            'use_fp16': os.getenv('USE_FP16', 'true').lower() == 'true',
            'device': os.getenv('DEVICE', 'auto'),  # 'auto', 'cpu', 'cuda'
            'max_seq_length': int(os.getenv('MAX_SEQ_LENGTH', '512')),
            'batch_size': int(os.getenv('BATCH_SIZE', '32')),
        }

        # Apply custom configuration overrides
        if custom_config:
            self.config.update(custom_config)

    def _determine_device(self):
        """Determine the best device to use for models"""
        if self.config['device'] == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"CUDA available, using GPU: {torch.cuda.get_device_name()}")
            else:
                device = 'cpu'
                print("CUDA not available, using CPU")
        else:
            device = self.config['device']
            print(f"Using specified device: {device}")
        return device

    def _initialize_embedding_model(self):
        """Initialize the embedding model with error handling"""
        if self._embedding_model is not None:
            return self._embedding_model

        try:
            print(f"Initializing embedding model: {self.config['embedding_model_name']}")
            start_time = time.time()

            device = self._determine_device()

            # enable proxy authentication
            set_proxy_authentication()

            self._embedding_model = SentenceTransformer(
                self.config['embedding_model_name'],
                cache_folder=self.config['cache_dir'],
                device=device
            )

            # Warm up the model with a dummy input
            dummy_text = ["This is a test sentence for model warmup."]
            _ = self._embedding_model.encode(dummy_text)

            load_time = time.time() - start_time
            print(f"Embedding model loaded successfully in {load_time:.2f}s")
            self._initialization_status['embedding'] = True

            return self._embedding_model

        except Exception as e:
            error_msg = f"Failed to initialize embedding model: {str(e)}"
            print(error_msg)
            self._initialization_errors['embedding'] = error_msg
            raise RuntimeError(error_msg) from e

    def _initialize_reranker(self):
        """Initialize the reranker model with error handling"""
        if self._reranker is not None:
            return self._reranker

        try:
            print(f"Initializing reranker model: {self.config['reranker_model_name']}")
            start_time = time.time()
            set_proxy_authentication()
            self._reranker = FlagReranker(
                self.config['reranker_model_name'],
                cache_dir=self.config['cache_dir'],
                use_fp16=self.config['use_fp16']
            )

            # Warm up the reranker with dummy input
            dummy_pairs = [("test query", "test document")]
            _ = self._reranker.compute_score(dummy_pairs)

            load_time = time.time() - start_time
            print(f"Reranker model loaded successfully in {load_time:.2f}s")
            self._initialization_status['reranker'] = True

            return self._reranker

        except Exception as e:
            error_msg = f"Failed to initialize reranker model: {str(e)}"
            print(error_msg)
            self._initialization_errors['reranker'] = error_msg
            raise RuntimeError(error_msg) from e

    @property
    def embedding_model(self) -> SentenceTransformer:
        """Get embedding model with lazy initialization and thread safety"""
        if self._embedding_model is None:
            with self._model_lock:
                if self._embedding_model is None:  # Double-check locking
                    self._initialize_embedding_model()
        return self._embedding_model

    @property
    def reranker(self) -> FlagReranker:
        """Get reranker model with lazy initialization and thread safety"""
        if self._reranker is None:
            with self._model_lock:
                if self._reranker is None:  # Double-check locking
                    self._initialize_reranker()
        return self._reranker

    def preload_models(self, models: list = None):
        """Preload specified models or all models"""
        if models is None:
            models = ['embedding', 'reranker']

        for model_name in models:
            try:
                if model_name == 'embedding':
                    _ = self.embedding_model
                elif model_name == 'reranker':
                    _ = self.reranker
                else:
                    print(f"Unknown model name: {model_name}")
            except Exception as e:
                print(f"Failed to preload {model_name} model: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get initialization status of all models"""
        return {
            'initialization_status': self._initialization_status.copy(),
            'initialization_errors': self._initialization_errors.copy(),
            'config': self.config.copy()
        }

    def set_models(self, embedding_model: str = None, reranker_model: str = None):
        """
        Dynamically change models (requires reinitialization)

        Args:
            embedding_model: Full name for embedding model
            reranker_model: Full name for reranker model
        """
        with self._model_lock:
            if embedding_model:
                self.config['embedding_model_name'] = embedding_model
                self._embedding_model = None
                self._initialization_status['embedding'] = False
                if 'embedding' in self._initialization_errors:
                    del self._initialization_errors['embedding']
                print(f"Set new embedding model: {self.config['embedding_model_name']}")

            if reranker_model:
                self.config['reranker_model_name'] = reranker_model
                self._reranker = None
                self._initialization_status['reranker'] = False
                if 'reranker' in self._initialization_errors:
                    del self._initialization_errors['reranker']
                print(f"Set new reranker model: {self.config['reranker_model_name']}")

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on models"""
        status = {
            'embedding_model': 'not_initialized',
            'reranker': 'not_initialized',
            'overall_status': 'healthy'
        }

        # Check embedding model
        try:
            if self._embedding_model is not None:
                test_embedding = self._embedding_model.encode(["health check"])
                if test_embedding is not None and len(test_embedding) > 0:
                    status['embedding_model'] = 'healthy'
                else:
                    status['embedding_model'] = 'unhealthy'
                    status['overall_status'] = 'unhealthy'
        except Exception as e:
            status['embedding_model'] = f'error: {str(e)}'
            status['overall_status'] = 'unhealthy'

        # Check reranker
        try:
            if self._reranker is not None:
                test_score = self._reranker.compute_score([("test", "test")])
                if test_score is not None:
                    status['reranker'] = 'healthy'
                else:
                    status['reranker'] = 'unhealthy'
                    status['overall_status'] = 'unhealthy'
        except Exception as e:
            status['reranker'] = f'error: {str(e)}'
            status['overall_status'] = 'unhealthy'

        return status


""""
USAGE:
# Example usage
if __name__ == "__main__":
    # Initialize with default models
    manager = ModelManager()
    
    # Initialize with custom models
    manager_custom = ModelManager(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        reranker_model_name="BAAI/bge-reranker-large"
    )
    
    # Initialize with custom config
    config = {
        'device': 'cuda',
        'use_fp16': True,
        'cache_dir': './custom_models/'
    }
    manager_with_config = ModelManager(
        embedding_model_name="BAAI/bge-large-en-v1.5",
        reranker_model_name="BAAI/bge-reranker-base",
        config=config
    )
    
    # Preload models
    manager.preload_models()
    
    # Check status
    status = manager.get_status()
    print(status)
"""