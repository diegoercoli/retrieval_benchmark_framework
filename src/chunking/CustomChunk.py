import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from pydantic import Field, ConfigDict
from docling_core.transforms.chunker.hierarchical_chunker import DocChunk, DocMeta

@dataclass
class Header:
    """
    Represents a document or section header with an ID and a title.

    Attributes:
        id (str): The identifier of the header, e.g., "6.1" or "3.2.1".
        title (str): The textual title of the header.
    """

    id: str
    title: str

    @classmethod
    def from_string(cls, text: str) -> "Header":
        """
        Create a Header instance from a single string.

        The string should start with the ID followed by a space and the title.

        Args:
            text (str): The header string, e.g., "6.1 Reset cards counters on operator panel".

        Returns:
            Header: A new Header instance with parsed id and title.

        Raises:
            ValueError: If the string does not match the expected format.
        """
        match = re.match(r"^(\d+(?:\.\d+)*)\s+(.*)$", text)
        if not match:
            raise ValueError(f"Invalid header format: {text}")
        return cls(id=match.group(1), title=match.group(2))



class CustomMeta(DocMeta):
    """
    Extended document metadata class that allows extra fields and structured hierarchy.

    This class provides readonly properties that automatically extract:

        filename (str): Name of the file containing the document.
        chapter (Header): The chapter header, e.g., "16 Troubleshooting e Test delle Periferiche".
        section (Header): The section header, e.g., "16.1 Monitor TFT/Touch utente ed operatore".
        subsection (Header): The subsection header, e.g., "16.1.1 Ripristino schermate e/o risoluzioni".
        division (Optional[str]): Optional division identifier, e.g., "Division_1".
    """

    model_config = ConfigDict(extra='allow')
    ''''
    File (or Document)
    └─ Chapter
        └─ Section
            └─ Subsection
                └─ Division    
    '''

    @property
    def filename(self) -> str:
        """Extract filename from origin metadata."""
        if self.origin and self.origin.filename:
            return self.origin.filename
        return "unknown"

    @property
    def chapter(self) -> Optional[Header]:
        """Extract chapter from first heading."""
        if self.headings and len(self.headings) > 0:
            try:
                return Header.from_string(self.headings[0])
            except ValueError:
                # If parsing fails, create a simple header
                return Header(id="", title=self.headings[0])
        return None

    @property
    def section(self) -> Optional[Header]:
        """Extract section from second heading."""
        if self.headings and len(self.headings) > 1:
            try:
                return Header.from_string(self.headings[1])
            except ValueError:
                # If parsing fails, create a simple header
                return Header(id="", title=self.headings[1])
        return None

    @property
    def subsection(self) -> Optional[Header]:
        """Extract subsection from third heading."""
        if self.headings and len(self.headings) > 2:
            try:
                return Header.from_string(self.headings[2])
            except ValueError:
                # If parsing fails, create a simple header
                return Header(id="", title=self.headings[2])
        return None

    division: Optional[str] = Field(None, description="Optional division identifier, e.g., 'Division_1'.")

    @classmethod
    def get_fields(self) -> List[str]:
        return  ["chapter", "section", "subsection", "division"]

    def __str__(self) -> str:
        """
            Return a flat, hierarchical path suitable for embeddings.
            For embedding models, it is recommended to use a flat, concise, normalized string rather than a Markdown tree.
            Embedding models work best with short, readable sequences where hierarchy is implied by separators like >
        """
        parts = [self.filename]

        if self.chapter:
            parts.append(f"{self.chapter.id} {self.chapter.title}")
        if self.section:
            parts.append(f"{self.section.id} {self.section.title}")
        if self.subsection:
            parts.append(f"{self.sudockebsection.id} {self.subsection.title}")
        if self.division:
            parts.append(self.division)

        # Join with a clear separator; " > " is common for embeddings
        return " > ".join(parts)

    def to_dict(self) -> dict:
        """Return metadata as a plain dictionary, including computed fields."""
        return {
            "filename": self.filename,
            "chapter": self.chapter.title if self.chapter else None,
            "section": self.section.title if self.section else None,
            "subsection": self.subsection.title if self.subsection else None,
            "division": self.division}


class CustomChunk(DocChunk):
    """Extended chunk class with custom metadata support."""

    model_config = ConfigDict(extra='allow')

    # Override the meta field to use CustomMeta
    # instance-level field (per-object) — Pydantic handles initialization, validation, and serialization for each instance.
    meta: CustomMeta

    # Add any additional chunk-level fields if needed
    chunk_id: Optional[str] = Field(default=None, description="Unique identifier for this chunk")

    @classmethod
    def from_doc_chunk(cls, doc_chunk: DocChunk, **additional_fields) -> 'CustomChunk':
        """Create a CustomChunk from an existing DocChunk."""

        # Extract meta extras for custom fields
        meta_extras = additional_fields.pop('meta_extras', {})

        # Method 1: Copy all fields from the original DocMeta to create CustomMeta
        # This preserves all the original DocMeta data and validation
        custom_meta_data = {
            'doc_items': doc_chunk.meta.doc_items,
            'headings': doc_chunk.meta.headings,
            'origin': doc_chunk.meta.origin,
            'division': meta_extras.get('division', None)
        }

        # Only include captions if it exists and is not None to avoid deprecation warning
       # if hasattr(doc_chunk.meta, 'captions') and doc_chunk.meta.captions is not None:
        #    try:
                # Try to access captions, but catch any deprecation issues
        #        captions = doc_chunk.meta.captions
        #        custom_meta_data['captions'] = captions
        #    except:
                # If there's any issue accessing captions, skip it
        #        pass

        # Add any additional custom fields from meta_extras
        for key, value in meta_extras.items():
            if key not in custom_meta_data:
                custom_meta_data[key] = value

        # Create the custom meta
        custom_meta = CustomMeta(**custom_meta_data)

        # Create the custom chunk
        return cls(
            text=doc_chunk.text,
            meta=custom_meta,
            **additional_fields
        )

    @classmethod
    def from_doc_chunk_v2(cls, doc_chunk: DocChunk, **additional_fields) -> 'CustomChunk':
        """
        Alternative method: Create CustomChunk by copying the original meta
        and then creating a new CustomMeta with the same data.
        """

        # Extract meta extras for custom fields
        meta_extras = additional_fields.pop('meta_extras', {})

        # Create a new CustomMeta by copying from the original DocMeta
        # and adding custom fields
        custom_meta = CustomMeta(
            doc_items=doc_chunk.meta.doc_items,
            headings=doc_chunk.meta.headings,
            captions=doc_chunk.meta.captions,
            origin=doc_chunk.meta.origin,
            division=meta_extras.get('division', None),
            **{k: v for k, v in meta_extras.items() if k != 'division'}
        )

        # Create the custom chunk
        return cls(
            text=doc_chunk.text,
            meta=custom_meta,
            **additional_fields
        )

    def is_same_section(self, other: 'CustomChunk') -> bool:
        """Check if this chunk belongs to the same section as another chunk."""
        return (
                self.meta.filename == other.meta.filename and
                self.meta.chapter == other.meta.chapter and
                self.meta.section == other.meta.section and
                self.meta.subsection == other.meta.subsection
        )

    def get_context_string(self) -> str:
        """Get a readable context string for this chunk."""
        return str(self.meta)

    def get_section_id(self) -> Optional[str]:
        """Get the most specific section ID available."""
        if self.meta.subsection:
            return self.meta.subsection.id
        elif self.meta.section:
            return self.meta.section.id
        elif self.meta.chapter:
            return self.meta.chapter.id
        return None

    @classmethod
    def get_fields(self) -> List[str]:
        return CustomMeta.get_fields() + ["text"]

    def to_dict(self) -> dict:
        """Return chunk and metadata as a plain dictionary."""
        base = { "text": self.text}
        if self.meta:
            base.update(self.meta.to_dict())
        return base