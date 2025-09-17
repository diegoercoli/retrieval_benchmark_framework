from typing import Iterator, Any, List, Optional
from docling_core.transforms.chunker import BaseChunk, DocChunk
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.types import DoclingDocument

from src.chunking.CustomChunk import CustomChunk


class CustomChunker(HybridChunker):

    def __init__(self, blacklist_chapters: Optional[List[str]] = None, **args):
        """
        Initialize the custom hierarchical chunker.

        Args:
            blacklist_chapters: List of chapter titles to ignore during chunking.
            **args: Additional arguments passed to the parent HybridChunker.
        """
        super().__init__(**args)
        self._processed_chunks: List[CustomChunk] = []
        self.blacklist_chapters: List[str] = blacklist_chapters or []

    @property
    def processed_chunks(self) -> List[CustomChunk]:
        return self._processed_chunks

    def __filter_chapter(self, chunk: CustomChunk) -> bool:
        """Return True if the chunk should be filtered out based on blacklist."""
        chapter_title = chunk.meta.chapter.title
        return chapter_title is None or  chapter_title in self.blacklist_chapters


    def chunk(self, dl_doc: DoclingDocument, **kwargs: Any) -> Iterator[BaseChunk]:
        """Chunk the document hierarchically."""
        self._processed_chunks = []

        docling_chunks = super().chunk(dl_doc, **kwargs)

        for docling_chunk in docling_chunks:
            # Convert base chunk to CustomChunk
            docling_chunk: DocChunk  # type hint
            custom_chunk = CustomChunk.from_doc_chunk(docling_chunk)
            # Filter if needed
            if self.__filter_chapter(custom_chunk):
                continue
            self._processed_chunks.append(custom_chunk)
            #produce the iterator
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
        return self.delim.join(items)

