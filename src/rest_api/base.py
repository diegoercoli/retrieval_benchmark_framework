import requests
from typing import Optional, Dict, Any
from requests.exceptions import RequestException, Timeout, HTTPError


class APIException(Exception):
    """Base exception for API errors"""

    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[Dict] = None):
        self.message = message
        self.status_code = status_code
        self.response = response
        super().__init__(self.message)


class BadRequestError(APIException):
    """400 Bad Request"""
    pass


class NotFoundError(APIException):
    """404 Not Found"""
    pass


class ServerError(APIException):
    """5xx Server Error"""
    pass


class BaseRestClient:
    """Base REST API client with common HTTP operations"""

    def __init__(self, base_url: str, timeout: int = 30):
        """
        Initialize the REST client

        Args:
            base_url: Base URL of the API (e.g., 'http://localhost:8000')
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })

    def _build_url(self, endpoint: str) -> str:
        """Build full URL from endpoint"""
        return f"{self.base_url}{endpoint}"

    def _handle_response(self, response: requests.Response) -> Dict[Any, Any]:
        """
        Handle API response and raise appropriate exceptions

        Args:
            response: Response object from requests

        Returns:
            Parsed JSON response

        Raises:
            BadRequestError: For 400 errors
            NotFoundError: For 404 errors
            ServerError: For 5xx errors
            APIException: For other HTTP errors
        """
        try:
            response.raise_for_status()
            return response.json()
        except HTTPError as e:
            # Try to extract error detail from response
            error_detail = "Unknown error"
            try:
                error_data = response.json()
                error_detail = error_data.get('detail', str(error_data))
            except Exception:
                error_detail = response.text or str(e)

            # Raise specific exceptions based on status code
            if response.status_code == 400:
                raise BadRequestError(
                    f"Bad Request: {error_detail}",
                    status_code=response.status_code,
                    response=error_data if 'error_data' in locals() else None
                )
            elif response.status_code == 404:
                raise NotFoundError(
                    f"Not Found: {error_detail}",
                    status_code=response.status_code,
                    response=error_data if 'error_data' in locals() else None
                )
            elif 500 <= response.status_code < 600:
                raise ServerError(
                    f"Server Error: {error_detail}",
                    status_code=response.status_code,
                    response=error_data if 'error_data' in locals() else None
                )
            else:
                raise APIException(
                    f"HTTP {response.status_code}: {error_detail}",
                    status_code=response.status_code,
                    response=error_data if 'error_data' in locals() else None
                )

    def _request(
            self,
            method: str,
            endpoint: str,
            **kwargs
    ) -> Dict[Any, Any]:
        """
        Make HTTP request with error handling

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            **kwargs: Additional arguments for requests

        Returns:
            Parsed JSON response

        Raises:
            APIException: For various API errors
            Timeout: For timeout errors
            RequestException: For connection errors
        """
        url = self._build_url(endpoint)

        try:
            response = self.session.request(
                method=method,
                url=url,
                timeout=self.timeout,
                **kwargs
            )
            return self._handle_response(response)

        except Timeout as e:
            raise APIException(f"Request timeout after {self.timeout}s: {url}") from e
        except RequestException as e:
            raise APIException(f"Request failed: {str(e)}") from e

    def get(self, endpoint: str, params: Optional[Dict] = None, **kwargs) -> Dict[Any, Any]:
        """Make GET request"""
        return self._request('GET', endpoint, params=params, **kwargs)

    def post(self, endpoint: str, json: Optional[Dict] = None, **kwargs) -> Dict[Any, Any]:
        """Make POST request"""
        return self._request('POST', endpoint, json=json, **kwargs)

    def put(self, endpoint: str, json: Optional[Dict] = None, **kwargs) -> Dict[Any, Any]:
        """Make PUT request"""
        return self._request('PUT', endpoint, json=json, **kwargs)

    def patch(self, endpoint: str, json: Optional[Dict] = None, **kwargs) -> Dict[Any, Any]:
        """Make PATCH request"""
        return self._request('PATCH', endpoint, json=json, **kwargs)

    def delete(self, endpoint: str, **kwargs) -> Dict[Any, Any]:
        """Make DELETE request"""
        return self._request('DELETE', endpoint, **kwargs)