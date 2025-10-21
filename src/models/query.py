from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum
from src.models.ground_truth import GroundTruth


class ComplexityQuery(str, Enum):
    """Query complexity types"""
    TEXTUAL_DESCRIPTION = "Textual_Description"
    IMAGE_ANALYSIS = "Image_Analysis"
    TABLE_ANALYSIS = "Table_Analysis"
    REASONING = "Reasoning"


class Query(BaseModel):
    """Query with associated ground truths"""
    position_id: int = Field(..., ge=1, description="Position of query in dataset (1, 2, 3, ...)")
    prompt: str = Field(..., min_length=1, description="Query text/prompt")
    device: Optional[str] = Field(None, description="Device context for the query")
    customer: Optional[str] = Field(None, description="Customer context for the query")
    complexity: ComplexityQuery = Field(..., description="Complexity type of the query")
    ground_truths: List[GroundTruth] = Field(
        default_factory=list,
        description="List of ground truth references"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "position_id": 1,
                "prompt": "How do I install the software?",
                "device": "iPhone 14",
                "customer": "premium_user",
                "complexity": "Textual_Description",
                "ground_truths": [
                    {
                        "filename": "installation_guide.pdf",
                        "hierarchical_metadata": {
                            "id_section": "2.1",
                            "section_title": "Quick Start",
                            "depth": 2
                        },
                        "confidence": "High"
                    }
                ]
            }
        }