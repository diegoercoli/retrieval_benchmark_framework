from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum


class ConfidenceLevel(str, Enum):
    """Confidence level enum matching backend"""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class HierarchicalMetadata(BaseModel):
    """Hierarchical metadata for document structure"""
    id_section: Optional[str] = Field(None, description="Section identifier")
    section_title: Optional[str] = Field(None, description="Section title")
    depth: Optional[int] = Field(None, ge=0, description="Depth level in hierarchy")

    class Config:
        json_schema_extra = {
            "example": {
                "id_section": "3.2.1",
                "section_title": "Installation Guide",
                "depth": 3
            }
        }


class GroundTruth(BaseModel):
    """Ground truth reference for query evaluation"""
    filename: str = Field(..., description="Name of the reference document")
    hierarchical_metadata: Optional[HierarchicalMetadata] = Field(
        None,
        description="Hierarchical metadata for the ground truth location"
    )
    confidence: ConfidenceLevel = Field(..., description="Confidence level of the ground truth")

    class Config:
        json_schema_extra = {
            "example": {
                "filename": "user_manual.pdf",
                "hierarchical_metadata": {
                    "id_section": "3.2.1",
                    "section_title": "Installation",
                    "depth": 3
                },
                "confidence": "High"
            }
        }