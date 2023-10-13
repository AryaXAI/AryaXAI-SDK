import requests
from pydantic import BaseModel

class APIClient(BaseModel):
    """API client to interact with Arya XAI services
    """
    base_url: str = ''
    auth_token: str = ''
    headers: dict = {
        'Content-Type': 'application/json'
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_auth_token(self, auth_token):
        """sets jwt auth token value

        :param auth_token: jwt auth token
        """
        self.auth_token = auth_token

    def update_headers(self, auth_token):
        """sets jwt auth token and updates headers for all requests
        """
        self.set_auth_token(auth_token)
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.auth_token}'
        }

    def get(self, uri):
        """makes get request to xai base service

        :param uri: api uri
        :raises Exception: Request exception
        :return: JSON response
        """
        url = f'{self.base_url}/{uri}'

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"GET request failed: {e}")

    def post(self, uri, payload={}):
        """makes post request to xai base service

        :param uri: api uri
        :param payload: api payload, defaults to {}
        :raises Exception: Request exception
        :return: JSON response
        """
        url = f'{self.base_url}/{uri}'

        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"POST request failed: {e}")
