import os
import logging
from typing import List, Optional
from tqdm import tqdm
import weaviate
from weaviate.collections.classes.config import Configure, Property, DataType
from weaviate.classes.query import Rerank, Filter
from weaviate.util import generate_uuid5

from src.utils.proxy_helper import set_no_proxy_localhost

# Suppress Weaviate/httpx INFO logs about authentication checks
logging.getLogger("weaviate").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


class WeaviateDBManager:
    """
    Weaviate Database Manager with context manager support for efficient connection handling.

    Usage:
        # As context manager (recommended)
        with WeaviateDBManager(port=8081, grpc_port=50051, inference_url="http://...") as db:
            db.create_collection(...)
            db.semantic_search_retrieve(...)

        # Manual connection management
        db = WeaviateDBManager(port=8081, grpc_port=50051, inference_url="http://...")
        db.connect()
        try:
            db.create_collection(...)
        finally:
            db.close()
    """

    def __init__(self, port: int, grpc_port: int, inference_url: str, auto_connect: bool = False):
        """
        Initialize WeaviateDBManager.

        Args:
            port: Weaviate HTTP port
            grpc_port: Weaviate gRPC port
            inference_url: URL for inference service
            auto_connect: If True, connect immediately on initialization
        """
        print(f"Initializing WeaviateDBManager with inference_url: {inference_url}")
        self.port = port
        self.grpc_port = grpc_port
        self.inference_url = inference_url
        self._client: Optional[weaviate.WeaviateClient] = None
        self._is_connected = False

        if auto_connect:
            self.connect()

    def __enter__(self):
        """Context manager entry - establish connection"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close connection"""
        self.close()
        return False  # Don't suppress exceptions

    def connect(self):
        """Establish connection to Weaviate"""
        if self._is_connected and self._client is not None:
            print("Already connected to Weaviate")
            return self._client

        try:
            print(f"Connecting to Weaviate DB using port={self.port}, grpc_port={self.grpc_port}")
            set_no_proxy_localhost()
            # Connect with skip_init_checks to avoid authentication warnings
            self._client = weaviate.connect_to_local(
                port=self.port,
                grpc_port=self.grpc_port,
                skip_init_checks=True  # Skip OIDC and other init checks
            )
            self._is_connected = True
            print("✓ Successfully connected to Weaviate")
            return self._client
        except Exception as e:
            print(f"✗ Failed to connect to Weaviate: {e}")
            self._is_connected = False
            raise

    def close(self):
        """Close connection to Weaviate"""
        if self._client is not None and self._is_connected:
            try:
                self._client.close()
                print("✓ Weaviate connection closed")
            except Exception as e:
                print(f"⚠ Error closing connection: {e}")
            finally:
                self._client = None
                self._is_connected = False

    def __del__(self):
        """Destructor - ensure connection is closed"""
        self.close()

    @property
    def client(self) -> weaviate.WeaviateClient:
        """Get the Weaviate client, ensuring connection is established"""
        if not self._is_connected or self._client is None:
            raise RuntimeError("Not connected to Weaviate. Call connect() or use context manager.")
        return self._client

    def is_connected(self) -> bool:
        """Check if connected to Weaviate"""
        return self._is_connected and self._client is not None

    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists"""
        return self.client.collections.exists(collection_name)

    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        if not self.collection_exists(collection_name):
            print(f"Collection '{collection_name}' does not exist")
            return False
        self.client.collections.delete(collection_name)
        print(f"✓ Deleted collection '{collection_name}'")
        return True



    def create_collection(self,
                          collection_name: str,
                          chunks: List,
                          all_fields: list,
                          embedding_field: str,
                          overwrite: bool = False) -> bool:
        """
        Create a collection and insert chunks.

        Args:
            collection_name: Name of the collection to create
            chunks: List of chunk objects with to_dict() method
            all_fields: List of all field names
            embedding_field: Field to use for embeddings
            overwrite: If True, delete existing collection before creating

        Returns:
            bool: True if successful, False otherwise
        """
        if all_fields is None:
            all_fields = []

        try:
            exist = self.collection_exists(collection_name)
            if not overwrite and exist:
                print(f"Collection '{collection_name}' already exists and overwrite=False")
                return False

            if exist:
                print(f"Deleting existing collection '{collection_name}'...")
                self.delete_collection(collection_name)

            # Determine which fields should be used for embedding
            embedding_fields = [embedding_field]

            vectorizer_config = [Configure.NamedVectors.text2vec_transformers(
                name="vector",
                source_properties=embedding_fields,
                vectorize_collection_name=False,
                inference_url=self.inference_url,
            )]

            # Create properties with proper vectorize_property_name configuration
            properties = []
            for field in all_fields:
                if field in embedding_fields:
                    properties.append(
                        Property(name=field, vectorize_property_name=True, data_type=DataType.TEXT)
                    )
                else:
                    properties.append(
                        Property(name=field, vectorize_property_name=False, data_type=DataType.TEXT)
                    )

            print(f"Creating collection '{collection_name}'...")
            collection = self.client.collections.create(
                name=collection_name,
                vectorizer_config=vectorizer_config,
                reranker_config=Configure.Reranker.transformers(),
                properties=properties
            )

            # Batch insert chunks
            print(f"Inserting {len(chunks)} chunks...")
            with collection.batch.fixed_size(batch_size=50, concurrent_requests=2) as batch:
                for chunk in tqdm(chunks, desc="Inserting chunks"):
                    chunk_dict = chunk.to_dict()
                    uuid = generate_uuid5(chunk_dict)
                    batch.add_object(properties=chunk_dict, uuid=uuid)

            print(f"✓ Successfully created collection '{collection_name}' with {len(chunks)} chunks")
            return True

        except Exception as e:
            print(f"✗ Error creating collection: {e}")
            raise

    def filter_by_metadata(self,
                           metadata_property: str,
                           values: list[str],
                           collection_name: str,
                           limit: int = 5) -> list:
        """
        Retrieve objects from a collection based on metadata filtering.

        Args:
            metadata_property: The metadata property to filter on
            values: List of values to match against the property
            collection_name: Name of the collection to query
            limit: Maximum number of objects to retrieve

        Returns:
            List of object properties matching the filter criteria
        """
        if not self.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")

        collection = self.client.collections.get(collection_name)

        response = collection.query.fetch_objects(
            limit=limit,
            filters=Filter.by_property(metadata_property).contains_any(values)
        )

        return [x.properties for x in response.objects]

    def __get_reranker(self, query:str, prop:str ="text"):
        return Rerank(query=query, prop=prop)

    def semantic_search_retrieve(self,
                                 query: str,
                                 collection_name: str,
                                 use_rerank: bool = True,
                                 pre_retrieval: int = 50,
                                 top_k: int = 5) -> list:
        """
        Perform semantic search on a collection.

        Args:
            query: Search query for semantic matching
            collection_name: Name of the collection to search
            use_rerank: Whether to apply reranking
            pre_retrieval: Number of candidates for reranking
            top_k: Number of top results to retrieve

        Returns:
            List of object properties most relevant to the query
        """
        if not self.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")

        collection = self.client.collections.get(collection_name)

        if use_rerank:
            response = collection.query.near_text(
                query=query,
                limit=pre_retrieval,
                rerank=self.__get_reranker(query=query)
            )
            return [x.properties for x in response.objects[:top_k]]
        else:
            response = collection.query.near_text(query=query, limit=top_k)
            return [x.properties for x in response.objects]

    def bm25_retrieve(self,
                      query: str,
                      collection_name: str,
                      use_rerank: bool = True,
                      pre_retrieval: int = 50,
                      top_k: int = 5) -> list:
        """
        Perform BM25 search on a collection, optionally reranking results.

        Args:
            query: Search query for keyword matching
            collection_name: Name of the collection to search
            top_k: Number of top results to retrieve
            use_rerank: Whether to rerank results using a reranker
            pre_retrieval: Number of candidates for reranking

        Returns:
            List of object properties most relevant to the query
        """
        if not self.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")

        collection = self.client.collections.get(collection_name)

        if use_rerank:
            response = collection.query.bm25(
                query=query,
                limit=pre_retrieval,
                rerank=self.__get_reranker(query=query)
            )
            return [x.properties for x in response.objects[:top_k]]
        else:
            response = collection.query.bm25(query=query, limit=top_k)
            return [x.properties for x in response.objects]

    def hybrid_search(self,
                      query: str,
                      collection_name: str,
                      use_rerank: bool = True,
                      pre_retrieval: int = 50,
                      top_k: int = 5,
                      alpha: float = 0.5
                      ) -> list:
        """
        Perform hybrid search combining semantic search with BM25, optionally reranking results.

        Args:
            query: Search query for matching
            collection_name: Name of the collection to search
            top_k: Number of results to return
            alpha: Weight between semantic (0.0) and BM25 (1.0). Default 0.5
            use_rerank: Whether to rerank results using a reranker
            pre_retrieval: Number of candidates for reranking

        Returns:
            List of matching object properties
        """
        if not self.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")

        collection = self.client.collections.get(collection_name)

        if use_rerank:
            response = collection.query.hybrid(
                query=query,
                alpha=alpha,
                limit=pre_retrieval,
                rerank=self.__get_reranker(query=query)
            )
            return [x.properties for x in response.objects[:top_k]]
        else:
            response = collection.query.hybrid(query=query, alpha=alpha, limit=top_k)
            return [x.properties for x in response.objects]

    def get_collection_count(self, collection_name: str) -> int:
        """
        Get the total number of objects in a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Number of objects in the collection
        """
        if not self.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")

        collection = self.client.collections.get(collection_name)
        # Use aggregate to get count
        result = collection.aggregate.over_all(total_count=True)
        return result.total_count