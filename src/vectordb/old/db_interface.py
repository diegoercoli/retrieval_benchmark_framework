from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass
import uuid
from datetime import datetime

from src.core.configuration import SearchType


class DataType(Enum):
    TEXT = "text"
    NUMBER = "number"
    DATE = "date"
    BOOLEAN = "boolean"
    ARRAY = "array"


@dataclass
class Property:
    name: str
    data_type: DataType
    vectorize: bool = False
    description: Optional[str] = None


@dataclass
class SearchResult:
    id: str
    data: Dict[str, Any]
    score: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CollectionConfig:
    name: str
    properties: List[Property]
    vectorizer_model: Optional[str] = None
    description: Optional[str] = None


class VectorDatabaseInterface(ABC):
    """Abstract interface for vector database operations"""

    @abstractmethod
    def create_collection(self, config: CollectionConfig) -> bool:
        """Create a new collection with specified configuration"""
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        pass

    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists"""
        pass

    @abstractmethod
    def insert_document(self, collection_name: str, document: Dict[str, Any], doc_id: Optional[str] = None) -> str:
        """Insert a single document and return its ID"""
        pass

    @abstractmethod
    def insert_documents(self, collection_name: str, documents: List[Dict[str, Any]],
                         doc_ids: Optional[List[str]] = None) -> List[str]:
        """Insert multiple documents and return their IDs"""
        pass

    @abstractmethod
    def search(self, collection_name: str, query: str, search_type: SearchType = SearchType.VECTOR,
               limit: int = 10, alpha: Optional[float] = None, filters: Optional[Dict[str, Any]] = None) -> List[
        SearchResult]:
        """Search documents in collection"""
        pass

    @abstractmethod
    def get_document(self, collection_name: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID"""
        pass

    @abstractmethod
    def update_document(self, collection_name: str, doc_id: str, updates: Dict[str, Any]) -> bool:
        """Update document by ID"""
        pass

    @abstractmethod
    def delete_document(self, collection_name: str, doc_id: str) -> bool:
        """Delete document by ID"""
        pass
