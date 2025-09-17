from typing import List, Dict, Any, Optional
import uuid

from weaviate.collections.classes.config import Property

from src.core.configuration import SearchType
from src.vectordb.old.db_interface import VectorDatabaseInterface, DataType, SearchResult, CollectionConfig


class WeaviateAdapter(VectorDatabaseInterface):
    """Weaviate implementation of the vector database interface"""

    def __init__(self, client, vectorizer_config=None):
        self.client = client
        self.vectorizer_config = vectorizer_config

    def _convert_data_type(self, data_type: DataType):
        """Convert generic DataType to Weaviate-specific DataType"""
        from weaviate.classes.config import DataType as WeaviateDataType

        mapping = {
            DataType.TEXT: WeaviateDataType.TEXT,
            DataType.NUMBER: WeaviateDataType.NUMBER,
            DataType.DATE: WeaviateDataType.DATE,
            #DataType.BOOLEAN: WeaviateDataType.BOOLEAN,
            DataType.ARRAY: WeaviateDataType.TEXT_ARRAY
        }
        return mapping.get(data_type, WeaviateDataType.TEXT)

    def _convert_properties(self, properties: List[Property]):
        """Convert generic properties to Weaviate properties"""
        from weaviate.classes.config import Property as WeaviateProperty

        weaviate_props = []
        for prop in properties:
            weaviate_props.append(
                WeaviateProperty(
                    name=prop.name,
                    data_type=self._convert_data_type(prop.data_type),
                    vectorize_property_name=prop.vectorize
                )
            )
        return weaviate_props

    def create_collection(self, config: CollectionConfig) -> bool:
        try:
            collection = self.client.collections.create(
                name=config.name,
                vectorizer_config=self.vectorizer_config,
                properties=self._convert_properties(config.properties)
            )
            return True
        except Exception as e:
            print(f"Error creating collection: {e}")
            return False

    def delete_collection(self, collection_name: str) -> bool:
        try:
            self.client.collections.delete(collection_name)
            return True
        except Exception as e:
            print(f"Error deleting collection: {e}")
            return False

    def collection_exists(self, collection_name: str) -> bool:
        try:
            collection = self.client.collections.get(collection_name)
            return collection is not None
        except:
            return False

    def insert_document(self, collection_name: str, document: Dict[str, Any], doc_id: Optional[str] = None) -> str:
        collection = self.client.collections.get(collection_name)
        if doc_id is None:
            doc_id = str(uuid.uuid4())

        collection.data.insert(
            properties=document,
            uuid=doc_id
        )
        return doc_id

    def insert_documents(self, collection_name: str, documents: List[Dict[str, Any]],
                         doc_ids: Optional[List[str]] = None) -> List[str]:
        collection = self.client.collections.get(collection_name)

        if doc_ids is None:
            doc_ids = [str(uuid.uuid4()) for _ in documents]

        with collection.batch.fixed_size(batch_size=100, concurrent_requests=4) as batch:
            for i, document in enumerate(documents):
                batch.add_object(
                    properties=document,
                    uuid=doc_ids[i]
                )

        return doc_ids

    def search(self, collection_name: str, query: str, search_type: SearchType = SearchType.VECTOR,
               limit: int = 10, alpha: Optional[float] = None, filters: Optional[Dict[str, Any]] = None) -> List[
        SearchResult]:
        collection = self.client.collections.get(collection_name)

        try:
            if search_type == SearchType.SEMANTIC_SEARCH:
                result = collection.query.near_text(query=query, limit=limit)
            elif search_type == SearchType.KEYWORD_SEACH:
                result = collection.query.bm25(query=query, limit=limit)
            elif search_type == SearchType.HYBRID_SEARCH:
                alpha_val = alpha if alpha is not None else 0.5
                result = collection.query.hybrid(query=query, alpha=alpha_val, limit=limit)
            else:
                raise ValueError(f"Unsupported search type: {search_type}")

            search_results = []
            for obj in result.objects:
                search_results.append(SearchResult(
                    id=str(obj.uuid),
                    data=obj.properties,
                    score=getattr(obj.metadata, 'score', 0.0) if hasattr(obj, 'metadata') else 0.0
                ))

            return search_results

        except Exception as e:
            print(f"Error during search: {e}")
            return []

    def get_document(self, collection_name: str, doc_id: str) -> Optional[Dict[str, Any]]:
        try:
            collection = self.client.collections.get(collection_name)
            result = collection.query.fetch_object_by_id(doc_id)
            return result.properties if result else None
        except Exception as e:
            print(f"Error getting document: {e}")
            return None

    def update_document(self, collection_name: str, doc_id: str, updates: Dict[str, Any]) -> bool:
        try:
            collection = self.client.collections.get(collection_name)
            collection.data.update(
                uuid=doc_id,
                properties=updates
            )
            return True
        except Exception as e:
            print(f"Error updating document: {e}")
            return False

    def delete_document(self, collection_name: str, doc_id: str) -> bool:
        try:
            collection = self.client.collections.get(collection_name)
            collection.data.delete_by_id(doc_id)
            return True
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False

