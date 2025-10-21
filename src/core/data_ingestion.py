from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from docling_core.transforms.chunker import BaseChunk
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

from src.chunking.CustomChunk import CustomChunk
from src.core.configuration import RAGConfiguration
from src.preprocessing.docling_utils import rebuilt_docling_doc_from_json
from src.utils.file_manager import scan_folder
from src.vectordb.weaviate_db_manager import WeaviateDBManager

class IngestionService(ABC):
    """Interface that clients must implement for ingestion business logic"""

    @abstractmethod
    def process(self,  config: RAGConfiguration) -> int:
        """
        Process ingestion from source, return processed count.

        Args:
            #source: Path or identifier for the data source
            config: RAG configuration containing chunk size, embedding model, etc.

        Returns:
            Number of items/chunks processed and stored

        Raises:
            Exception: If processing fails
        """
        pass

class DocumentIngestionService(IngestionService):
    """Your business logic implementation - completely separate from framework"""

    def __init__(self, db_manager: WeaviateDBManager):
        self.db_manager = db_manager

    # This is not required, but makes the intent clear

    def process(self, config: RAGConfiguration, overwrite=False) -> int:
        """
        Core business logic for document ingestion.

        Args:
            config (RAGConfiguration): RAG configuration containing chunk size, embedding model, etc.
            overwrite (bool): If True, overwrite existing collection; if False, skip if collection exists.

        Returns:
            int: Number of items/chunks processed and stored.

        Raises:
            Exception: If processing fails.
        """

        if self.db_manager.collection_exists(config.collection_name):
            if not overwrite:
                print(f"Collection '{config.collection_name}' already exists. Skipping ingestion.")
                return 0
            else:
                print(f"Collection '{config.collection_name}' already exists. Overwriting.")
                self.db_manager.delete_collection(config.collection_name)

        # Your business logic here:
        # 1. Read files from filesystem based on source
        # Start timer
        #start_time = time.time()
        json_docxs = scan_folder(config.main_folder, ".json")
        chunker = config.chunking
        all_chunks = []
        for json_docx in json_docxs:
            #rebuilt docling docx
            docling_doc = rebuilt_docling_doc_from_json(Path(json_docx))
            chunks = list(chunker.chunk(docling_doc))
            enriched_chunks = []
            for chunk in chunks:
                chunk.text = chunker.contextualize(chunk=chunk)
                self.__validate_chunks_size( chunker.tokenizer, chunk)
                enriched_chunks.append(chunk)
                all_chunks.append(chunk)
            #store chunks
            file_path = config.main_folder / "chunks" / f"{docling_doc.name}.md"
            self.store_chunks(enriched_chunks, file_path)
        if all_chunks:
            self.db_manager.create_collection(config.collection_name, all_chunks, CustomChunk.get_fields(), "text",
                                              overwrite=False)
        count = self.db_manager.get_collection_count(config.collection_name)
        print(f"Document count: {count}")
        # ingestion_time = time.time() - start_time
        # print(            f"⏱️  Ingestion completed for '{config.collection_name}' in {ingestion_time:.2f} seconds ({len(all_chunks)} chunks)")
        return count


    def __validate_chunks_size(self, tokenizer: HuggingFaceTokenizer, chunk: BaseChunk):
        max_tokens = tokenizer.tokenizer.model_max_length
        chunk_size = tokenizer.count_tokens(chunk.text)
        # trigger exception if chunk size exceeds max tokens
        if chunk_size > max_tokens:
            raise ValueError(
                f"Chunk effective size is {chunk_size}, without context we estimated a priori the size to be {tokenizer.max_tokens}."
                f"The problem is that exceeds the embedding model max input length: {max_tokens}. "
                f"Consider decreasing a priori chunk size: {tokenizer.max_tokens} or using a model with larger context window."
            )

    def store_chunks(self, chunks: List[CustomChunk] ,filepath: Path):
        with open(filepath, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                enriched_text = chunk.text
                f.write(enriched_text + "\n\n")
                f.write("-" * 200 + "\n\n")
                #print(f"=== Chunk_{i}_written ===")
        print(f"Chunks written to {filepath}")