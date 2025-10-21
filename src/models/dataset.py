from typing import List, Optional
from datetime import date
from pydantic import BaseModel, Field
from src.models.query import Query


class DatasetInput(BaseModel):
    """Input model for creating/updating a dataset"""
    dataset_name: str = Field(..., min_length=1, max_length=100, description="Unique dataset name")
    queries: List[Query] = Field(
        default_factory=list,
        description="List of queries with ground truths"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "dataset_name": "product_support_qa",
                "queries": [
                    {
                        "position_id": 1,
                        "prompt": "How do I reset my password?",
                        "device": "Web Browser",
                        "customer": "standard_user",
                        "complexity": "Textual_Description",
                        "ground_truths": [
                            {
                                "filename": "user_guide.pdf",
                                "hierarchical_metadata": {
                                    "id_section": "4.1",
                                    "section_title": "Account Management",
                                    "depth": 2
                                },
                                "confidence": "High"
                            }
                        ]
                    }
                ]
            }
        }


class DatasetResponse(BaseModel):
    """Response model from dataset creation/update"""
    id: int = Field(..., description="Dataset ID")
    dataset_name: str = Field(..., description="Dataset name")
    data_creation: date = Field(..., description="Date when dataset was created")
    data_update: Optional[date] = Field(None, description="Date of last update")
    queries_added: int = Field(0, description="Number of queries added")
    queries_updated: int = Field(0, description="Number of queries updated")
    queries_marked_obsolete: int = Field(0, description="Number of queries marked as obsolete")
    ground_truths_added: int = Field(0, description="Number of ground truths added")

    class Config:
        from_attributes = True