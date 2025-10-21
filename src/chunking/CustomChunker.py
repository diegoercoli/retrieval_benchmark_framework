from typing import Iterator, Any, List, Optional

from contourpy import chunk
from pydantic import Field
from docling_core.transforms.chunker import BaseChunk, DocChunk
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.types import DoclingDocument
from src.chunking.CustomChunk import CustomChunk
from src.core.configuration import PreprocessConfiguration


class CustomChunker(HybridChunker):
    # Define these as Pydantic fields since we inherit from BaseModel
    name: str = Field(default="hierarchical_chunking", alias="name")
    blacklist_chapters: List[str] = Field(default_factory=list)
    processed_chunks: List[CustomChunk] = Field(default_factory=list, init=False)
    preprocess_configuration: PreprocessConfiguration

    def __init__(self, name: str = "", blacklist_chapters: Optional[List[str]] = None,  preprocess_configuration: PreprocessConfiguration = None, **args):
        """
        Initialize the custom hierarchical chunker.

        Args:
            name: Name of the chunker
            blacklist_chapters: List of chapter titles to ignore during chunking.
            **args: Additional arguments passed to the parent HybridChunker.
        """
        # Pass the fields to the parent constructor
        super().__init__(
            name=name or "hierarchical_chunking",
            blacklist_chapters=blacklist_chapters or [],
            preprocess_configuration=preprocess_configuration,
            **args
        )

    @property
    def processed_chunks(self) -> List[CustomChunk]:
        return self._processed_chunks

    @property
    def name(self) -> str:
        return self._name

    def __filter_chapter(self, chunk: CustomChunk) -> bool:
        """Return True if the chunk should be filtered out based on blacklist."""
        if not chunk.meta.chapter:
            return True
        chapter_title = chunk.meta.chapter.title
        chapter_id = int(chunk.meta.chapter.id) if str(chunk.meta.chapter.id).isdigit() else 0
        if chapter_title is None or chapter_title == "" or chapter_title in self.blacklist_chapters or chapter_id <= 0:
            return True
        normalized_title = chapter_title.replace(" ", "").lower()
        normalized_blacklist = [item.replace(" ", "").lower() for item in self.blacklist_chapters]
        return normalized_title in normalized_blacklist

    ######## BASE_CHUNK INTERFACE ############

    def chunk(self, dl_doc: DoclingDocument, **kwargs: Any) -> Iterator[BaseChunk]:
        """Chunk the document hierarchically."""
        self._processed_chunks = []
        docling_chunks = super().chunk(dl_doc, **kwargs)
        current_header = ""
        division = 0
        for docling_chunk in docling_chunks:
            # Convert base chunk to CustomChunk
            docling_chunk: DocChunk  # type hint
            custom_chunk = CustomChunk.from_doc_chunk(docling_chunk)
            if custom_chunk.meta.__str__() == current_header:
                division += 1
            else:
                current_header = custom_chunk.meta.__str__()
                division = 0
            custom_chunk.meta.division = f"Division_{division}"
            # Filter if needed
            if self.__filter_chapter(custom_chunk):
                continue
            self._processed_chunks.append(custom_chunk)
            # produce the iterator
            yield custom_chunk

    def contextualize(self, chunk: BaseChunk) -> str:
        """Contextualize the given chunk. This implementation is embedding-targeted.

        Args:
            chunk: chunk to serialize

        Returns:
            str: the serialized form of the chunk
        """
        items = []
        if isinstance(chunk, CustomChunk):
            items.append(chunk.get_context_string())
        items.append(chunk.text)
        if self.preprocess_configuration is not None and self.preprocess_configuration.lowercase:
            #make all the text to be returned in lowercase
            items = [item.lower() for item in items if item]
        return self.delim.join(items)