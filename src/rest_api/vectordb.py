from typing import Optional
from pydantic import BaseModel
from src.rest_api.base import BaseRestClient, APIException


class VectorDBProviderBase(BaseModel):
    """Base model for vector database provider"""
    name: str
    port_number: int


class VectorDBProviderCreate(VectorDBProviderBase):
    """Model for creating a vector database provider"""
    pass


class VectorDBProviderResponse(VectorDBProviderBase):
    """Response model from vector database provider creation"""
    id: int

    class Config:
        from_attributes = True


class VectorDBAPI(BaseRestClient):
    """Client for vector database-related REST API endpoints"""

    def __init__(self, base_url: str, timeout: int = 30):
        """
        Initialize VectorDB API client

        Args:
            base_url: Base URL of the API (e.g., 'http://localhost:8000')
            timeout: Request timeout in seconds
        """
        super().__init__(base_url, timeout)
        self.endpoint_base = '/api/v1/vector-db/providers'

    def create_provider(self, provider: VectorDBProviderCreate) -> VectorDBProviderResponse:
        """
        Create or register a vector database provider

        Args:
            provider: VectorDBProviderCreate object containing provider information

        Returns:
            VectorDBProviderResponse with provider_id

        Raises:
            BadRequestError: If input validation fails
            ServerError: If server encounters an error
            APIException: For other errors

        Example:
            >>> from src.rest_api.vectordb import VectorDBAPI, VectorDBProviderCreate
            >>>
            >>> provider = VectorDBProviderCreate(
            ...     name="weaviate",
            ...     port_number=8081
            ... )
            >>>
            >>> client = VectorDBAPI(base_url="http://localhost:8000")
            >>> response = client.create_provider(provider)
            >>> print(f"Provider ID: {response.id}")
        """
        try:
            # Convert Pydantic model to dict, excluding None values
            payload = provider.model_dump(mode='json', exclude_none=True)

            # Make POST request
            response_data = self.post(self.endpoint_base, json=payload)

            # Parse response into VectorDBProviderResponse model
            return VectorDBProviderResponse(**response_data)

        except APIException:
            # Re-raise API exceptions as-is
            raise
        except Exception as e:
            # Wrap unexpected exceptions
            raise APIException(f"Failed to create vector database provider: {str(e)}") from e

    def get_provider(self, provider_id: int) -> VectorDBProviderResponse:
        """
        Get vector database provider by ID

        Args:
            provider_id: ID of the provider

        Returns:
            VectorDBProviderResponse object

        Raises:
            NotFoundError: If provider not found
            APIException: For other errors
        """
        try:
            response_data = self.get(f"{self.endpoint_base}/{provider_id}")
            return VectorDBProviderResponse(**response_data)
        except APIException:
            raise
        except Exception as e:
            raise APIException(f"Failed to get provider {provider_id}: {str(e)}") from e

    def list_providers(self) -> list[VectorDBProviderResponse]:
        """
        List all vector database providers

        Returns:
            List of VectorDBProviderResponse objects

        Raises:
            APIException: For errors
        """
        try:
            response_data = self.get(self.endpoint_base)
            return [VectorDBProviderResponse(**item) for item in response_data]
        except APIException:
            raise
        except Exception as e:
            raise APIException(f"Failed to list providers: {str(e)}") from e

    def delete_provider(self, provider_id: int) -> None:
        """
        Delete vector database provider by ID

        Args:
            provider_id: ID of the provider to delete

        Raises:
            NotFoundError: If provider not found
            APIException: For other errors
        """
        try:
            self.delete(f"{self.endpoint_base}/{provider_id}")
        except APIException:
            raise
        except Exception as e:
            raise APIException(f"Failed to delete provider {provider_id}: {str(e)}") from e