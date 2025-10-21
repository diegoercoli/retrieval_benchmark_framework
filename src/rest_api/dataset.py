from typing import List, Optional
from src.rest_api.base import BaseRestClient, APIException
from src.models.dataset import DatasetInput, DatasetResponse


class DatasetAPI(BaseRestClient):
    """Client for dataset-related REST API endpoints"""

    def __init__(self, base_url: str, timeout: int = 30):
        """
        Initialize Dataset API client

        Args:
            base_url: Base URL of the API (e.g., 'http://localhost:8000')
            timeout: Request timeout in seconds
        """
        super().__init__(base_url, timeout)
        self.endpoint_base = '/api/v1/datasets'

    def create_dataset(self, dataset: DatasetInput) -> DatasetResponse:
        """
        Create or update a dataset with queries and ground truths

        Args:
            dataset: DatasetInput object containing dataset information

        Returns:
            DatasetResponse with dataset_id and statistics

        Raises:
            BadRequestError: If input validation fails
            ServerError: If server encounters an error
            APIException: For other errors

        Example:
            >>> from src.models import DatasetInput, Query, GroundTruth
            >>>
            >>> dataset = DatasetInput(
            ...     dataset_name="my_dataset",
            ...     queries=[
            ...         Query(
            ...             position_id=1,
            ...             prompt="How to install?",
            ...             complexity="Textual_Description",
            ...             ground_truths=[
            ...                 GroundTruth(
            ...                     filename="manual.pdf",
            ...                     confidence="High"
            ...                 )
            ...             ]
            ...         )
            ...     ]
            ... )
            >>>
            >>> client = DatasetAPI(base_url="http://localhost:8000")
            >>> response = client.create_dataset(dataset)
            >>> print(f"Dataset ID: {response.id}")
        """
        try:
            # Convert Pydantic model to dict, excluding None values
            payload = dataset.model_dump(mode='json', exclude_none=True)

            # Make POST request
            response_data = self.post(self.endpoint_base, json=payload)

            # Parse response into DatasetResponse model
            return DatasetResponse(**response_data)

        except APIException:
            # Re-raise API exceptions as-is
            raise
        except Exception as e:
            # Wrap unexpected exceptions
            raise APIException(f"Failed to create dataset: {str(e)}") from e

    def get_dataset(self, dataset_id: int) -> DatasetResponse:
        """
        Get dataset by ID

        Args:
            dataset_id: ID of the dataset

        Returns:
            DatasetResponse object

        Raises:
            NotFoundError: If dataset not found
            APIException: For other errors
        """
        try:
            response_data = self.get(f"{self.endpoint_base}/{dataset_id}")
            return DatasetResponse(**response_data)
        except APIException:
            raise
        except Exception as e:
            raise APIException(f"Failed to get dataset {dataset_id}: {str(e)}") from e

    def list_datasets(
            self,
            skip: int = 0,
            limit: int = 100
    ) -> List[DatasetResponse]:
        """
        List all datasets with pagination

        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of DatasetResponse objects

        Raises:
            APIException: For errors
        """
        try:
            params = {'skip': skip, 'limit': limit}
            response_data = self.get(self.endpoint_base, params=params)
            return [DatasetResponse(**item) for item in response_data]
        except APIException:
            raise
        except Exception as e:
            raise APIException(f"Failed to list datasets: {str(e)}") from e

    def delete_dataset(self, dataset_id: int) -> None:
        """
        Delete dataset by ID

        Args:
            dataset_id: ID of the dataset to delete

        Raises:
            NotFoundError: If dataset not found
            APIException: For other errors
        """
        try:
            self.delete(f"{self.endpoint_base}/{dataset_id}")
        except APIException:
            raise
        except Exception as e:
            raise APIException(f"Failed to delete dataset {dataset_id}: {str(e)}") from e

    def get_dataset_queries(
            self,
            dataset_id: int,
            obsolete: Optional[bool] = None
    ) -> List[dict]:
        """
        Get all queries for a dataset

        Args:
            dataset_id: ID of the dataset
            obsolete: Filter by obsolete status (None = all queries)

        Returns:
            List of query dictionaries

        Raises:
            NotFoundError: If dataset not found
            APIException: For other errors
        """
        try:
            params = {}
            if obsolete is not None:
                params['obsolete'] = obsolete

            response_data = self.get(
                f"{self.endpoint_base}/{dataset_id}/queries",
                params=params
            )
            return response_data
        except APIException:
            raise
        except Exception as e:
            raise APIException(
                f"Failed to get queries for dataset {dataset_id}: {str(e)}"
            ) from e