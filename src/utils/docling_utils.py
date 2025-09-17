from pathlib import Path

from docling_core.transforms.chunker.hierarchical_chunker import ChunkingSerializerProvider, ChunkingDocSerializer
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
from docling_core.types import DoclingDocument


def rebuilt_docling_doc_from_json(json_path: Path) -> DoclingDocument:
    with open(json_path, "r", encoding='utf-8') as f:
        json_string = f.read()
        rebuilt_doc = DoclingDocument.model_validate_json(json_string)
        return rebuilt_doc


class MDTableSerializerProvider(ChunkingSerializerProvider):
    def get_serializer(self, doc):
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),  # configuring a different table serializer
        )