from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from src.chunking.CustomChunk import CustomChunk
from src.core.configuration import RAGConfiguration
from src.utils.docling_utils import rebuilt_docling_doc_from_json
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

    def process(self, config: RAGConfiguration) -> int:
        """Your core business logic for document ingestion"""

        # Your business logic here:
        # 1. Read files from filesystem based on source
        json_docxs = scan_folder(config.main_folder / ".json")
        chunker = config.chunking
        all_chunks = []
        for json_docx in json_docxs:
            #rebuilt docling docx
            docling_doc = rebuilt_docling_doc_from_json(Path(json_docx))
            chunks = list(chunker.chunk(docling_doc))
            enriched_chunks = []
            for chunk in chunks:
                chunk.text = chunker.contextualize(chunk=chunk)
                enriched_chunks.append(chunk)
                all_chunks.append(chunk)
            #store chunks
            file_path = config.main_folder / "chunks" / f"{docling_doc.name}.md"
            self.store_chunks(enriched_chunks, file_path)
        # Create collection with all chunks at once
        if all_chunks:
            self.db_manager.create_collection(config.collection_name, all_chunks, overwrite=True)
        return len(all_chunks)

    def store_chunks(self, chunks: List[CustomChunk] ,filepath: Path):
        with open(filepath, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                enriched_text = chunk.text
                f.write(enriched_text + "\n")
                f.write("-" * 200 + "\n")
                print(f"=== Chunk_{i}_written ===")
        print(f"Chunks written to {filepath}")


