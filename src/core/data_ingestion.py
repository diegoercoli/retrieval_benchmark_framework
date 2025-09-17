from pathlib import Path

from src.core.configuration import RAGConfiguration
from src.engine.custom_tasks import IngestionService
from src.utils.file_manager import scan_folder, rebuilt_docling_doc_from_json
from src.vectordb.weaviate_db_manager import WeaviateDBManager


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
            chunks = chunker.chunk(docling_doc)
            enriched_chunks = []
            for chunk in chunks:
                chunk.text = chunker.contextualize(chunk=chunk)
                enriched_chunks.append(chunk)
                all_chunks.append(chunk)
            self.db_manager.create_collection(config.collection_name, enriched_chunks)

            all_chunks.extend(chunks)
        return len(all_chunks)


