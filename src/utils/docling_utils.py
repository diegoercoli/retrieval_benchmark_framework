from pathlib import Path

from docling_core.types import DoclingDocument


def rebuilt_docling_doc_from_json(json_path: Path) -> DoclingDocument:
    with open(json_path, "r", encoding='utf-8') as f:
        json_string = f.read()
        rebuilt_doc = DoclingDocument.model_validate_json(json_string)
        return rebuilt_doc