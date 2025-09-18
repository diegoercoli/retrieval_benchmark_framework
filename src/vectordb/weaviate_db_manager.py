from typing import List
from tqdm import tqdm
from weaviate.classes.config import Configure, Property, DataType
import weaviate
from weaviate.util import generate_uuid5
from weaviate.classes.query import (
    Filter,
    Rerank
)
from src.chunking.CustomChunk import CustomChunk


class WeaviateDBManager:

    def __init__(self, port: int, grpc_port: int, inference_url: str):
        print(f"##############INFERENCE_URL: {inference_url}")
        self.port = port
        self.grpc_port = grpc_port
        self.inference_url = inference_url

    def __connect(self):
        return weaviate.connect_to_local(port=self.port, grpc_port=self.grpc_port)

    def __close(self, client):
        """Close the Weaviate client connection."""
        client.close()

    def __collection_exists(self, client, collection_name: str) -> bool:
        return client.collections.exists(collection_name)

    def __delete_collection(self, client, collection_name: str):
        return client.collections.delete(collection_name)

    def collection_exists(self, collection_name: str) -> bool:
        """Public method to check if collection exists"""
        client = self.__connect()
        try:
            return self.__collection_exists(client, collection_name)
        finally:
            self.__close(client)

    def delete_collection(self, collection_name: str) -> bool:
        """Public method to delete collection"""
        client = self.__connect()
        try:
            return self.__delete_collection(client, collection_name)
        finally:
            self.__close(client)

    def create_collection(self, collection_name: str, chunks: List[CustomChunk], overwrite=False) -> bool:
        client = self.__connect()
        try:
            exist = self.__collection_exists(client, collection_name)
            if not overwrite and exist:
                return False
            if exist:
                self.__delete_collection(client, collection_name)

            # Get all fields from CustomChunk
            all_fields = CustomChunk.get_fields()

            # Determine which fields should be used for embedding
            # Assuming 'text' is the main content field for embedding
            # and other fields are metadata
            embedding_fields = ["text"]  # Modify this based on your needs

            vectorizer_config = [Configure.NamedVectors.text2vec_transformers(
                name="vector",
                source_properties=embedding_fields,  # Only text field used for embedding
                vectorize_collection_name=False,
                inference_url=self.inference_url,
            )]

            # Create properties with proper vectorize_property_name configuration
            properties = []
            for field in all_fields:
                if field in embedding_fields:
                    # Fields used for embedding
                    properties.append(
                        Property(name=field, vectorize_property_name=True, data_type=DataType.TEXT)
                    )
                else:
                    # Metadata fields (not used for embedding)
                    properties.append(
                        Property(name=field, vectorize_property_name=False, data_type=DataType.TEXT)
                    )

            collection = client.collections.create(
                name=collection_name,
                vectorizer_config=vectorizer_config,
                reranker_config=Configure.Reranker.transformers(),
                properties=properties
            )

            # Set up a batch process with specified fixed size and concurrency
            with collection.batch.fixed_size(batch_size=50, concurrent_requests=2) as batch:
                # Iterate over a subset of the dataset
                for chunk in tqdm(chunks):
                    # Generate a UUID based on the chunk content
                    chunk_dict = chunk.to_dict()
                    uuid = generate_uuid5(chunk_dict)

                    # Add the object to the batch with properties and UUID.
                    batch.add_object(
                        properties=chunk_dict,
                        uuid=uuid,
                    )
            return True
        finally:
            self.__close(client)

    def filter_by_metadata(self, metadata_property: str,
                           values: list[str],
                           collection_name: str,
                           limit: int = 5) -> list:
        """
        Retrieves objects from a specified collection based on metadata filtering criteria.

        This function queries a collection within the specified client to fetch objects that match
        certain metadata criteria. It uses a filter to find objects whose specified 'property' contains
        any of the given 'values'. The number of objects retrieved is limited by the 'limit' parameter.

        Args:
        metadata_property (str): The name of the metadata property to filter on.
        values (List[str]): A list of values to be matched against the specified property.
        collection_name (str): The name of the collection to query.
        limit (int, optional): The maximum number of objects to retrieve. Defaults to 5.

        Returns:
        List[dict]: A list of object properties from the collection that match the filtering criteria.
        """
        client = self.__connect()
        try:
            if not self.__collection_exists(client, collection_name):
                raise Exception(f"Collection '{collection_name}' does not exist.")

            collection = client.collections.get(collection_name)

            # Retrieve using collection.query.fetch_objects
            response = collection.query.fetch_objects(
                limit=limit,
                filters=Filter.by_property(metadata_property).contains_any(values)
            )

            response_objects = [x.properties for x in response.objects]
            return response_objects
        finally:
            self.__close(client)

    def semantic_search_retrieve(self, query: str,
                                 collection_name: str,
                                 top_k: int = 5) -> list:
        """
        Performs a semantic search on a collection and retrieves the top relevant chunks.

        This function executes a semantic search query on a specified collection to find text chunks
        that are most relevant to the input 'query'. The search retrieves a limited number of top
        matching objects, as specified by 'top_k'. The function returns the properties of
        each of the top matching objects.

        Args:
        query (str): The search query used to find relevant text chunks.
        collection_name (str): The name of the collection in which the semantic search is performed.
        top_k (int, optional): The number of top relevant objects to retrieve. Defaults to 5.

        Returns:
        List[dict]: A list of object properties that are most relevant to the given query.
        """
        client = self.__connect()
        try:
            if not self.__collection_exists(client, collection_name):
                raise Exception(f"Collection '{collection_name}' does not exist.")

            collection = client.collections.get(collection_name)

            # Retrieve using collection.query.near_text
            response = collection.query.near_text(query=query, limit=top_k)

            response_objects = [x.properties for x in response.objects]
            return response_objects
        finally:
            self.__close(client)

    def bm25_retrieve(self, query: str,
                      collection_name: str,
                      top_k: int = 5) -> list:
        """
        Performs a BM25 search on a collection and retrieves the top relevant chunks.

        This function executes a BM25 search query on a specified collection to find text chunks
        that are most relevant to the input 'query'. The search retrieves a limited number of top
        matching objects, as specified by 'top_k'. The function returns the properties of
        each of the top matching objects.

        Args:
        query (str): The search query used to find relevant text chunks.
        collection_name (str): The name of the collection in which the BM25 search is performed.
        top_k (int, optional): The number of top relevant objects to retrieve. Defaults to 5.

        Returns:
        List[dict]: A list of object properties that are most relevant to the given query.
        """
        client = self.__connect()
        try:
            if not self.__collection_exists(client, collection_name):
                raise Exception(f"Collection '{collection_name}' does not exist.")

            collection = client.collections.get(collection_name)

            # Retrieve using collection.query.bm25
            response = collection.query.bm25(query=query, limit=top_k)

            response_objects = [x.properties for x in response.objects]
            return response_objects
        finally:
            self.__close(client)

    def hybrid_search(self, query: str,
                      collection_name: str,
                      top_k: int = 5,
                      alpha: float = 0.5)-> list:
        """
        Performs a hybrid search combining semantic search with BM25.

        Args:
        query (str): The search query for semantic matching.
        collection_name (str): The name of the collection to search.
        alpha (float): The weight between semantic (0.0) and BM25 (1.0) search. Default 0.5.
        top_k (int): Number of results to return.

        Returns:
        List[dict]: List of matching object properties.
        """
        client = self.__connect()
        try:
            if not self.__collection_exists(client, collection_name):
                raise Exception(f"Collection '{collection_name}' does not exist.")

            collection = client.collections.get(collection_name)

            response = collection.query.hybrid(query=query, alpha=alpha, limit=top_k)

            response_objects = [x.properties for x in response.objects]
            return response_objects
        finally:
            self.__close(client)