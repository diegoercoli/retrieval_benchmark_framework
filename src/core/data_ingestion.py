from pathlib import Path

from src.core.configuration import RAGConfiguration
from src.engine.custom_tasks import IngestionService
from src.utils.file_manager import scan_folder, rebuilt_docling_doc_from_json


class DocumentIngestionService(IngestionService):
    """Your business logic implementation - completely separate from framework"""

    def __init__(self, ):
        self.db_connection = None

    # This is not required, but makes the intent clear

    def process(self, source: str, config: RAGConfiguration) -> int:
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
            enriched_chunks = [ chunker.contextualize(chunk=chunk) for chunk in chunks]


            all_chunks.extend(chunks)
        return len(all_chunks)

   def create_collection(self, collection_name: str) -> bool:
       return False
