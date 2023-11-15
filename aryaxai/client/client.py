import requests
from aryaxai.common.xai_uris import LOGIN_URI
import jwt
from pydantic import BaseModel
import json


class APIClient(BaseModel):
    """API client to interact with Arya XAI services"""

    debug: bool = False
    base_url: str = ""
    access_token: str = ""
    auth_token: str = ""
    headers: dict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_auth_token(self) -> str:
        """get jwt auth token value

        Returns:
            str: jwt auth token
        """
        return self.auth_token

    def set_auth_token(self, auth_token):
        """sets jwt auth token value

        :param auth_token: jwt auth token
        """
        self.auth_token = auth_token

    def set_access_token(self, access_token):
        """sets access token value

        :param auth_token: jwt auth token
        """
        self.access_token = access_token

    def get_url(self, uri) -> str:
        """get url by appending uri to base url

        :param uri: uri of endpoint
        :return: url
        """
        return f"{self.base_url}/{uri}"

    def update_headers(self, auth_token):
        """sets jwt auth token and updates headers for all requests"""
        self.set_auth_token(auth_token)
        self.headers = {
            "Authorization": f"Bearer {self.auth_token}",
        }

    def refresh_bearer_token(self):
        try:
            if self.auth_token:
                jwt.decode(
                    self.auth_token,
                    options={"verify_signature": False, "verify_exp": True},
                )
        except jwt.exceptions.ExpiredSignatureError as e:
            response = self.base_request(
                "POST", LOGIN_URI, {"access_token": self.access_token}
            ).json()
            self.update_headers(response["access_token"])

    def base_request(self, method, uri, payload={}, files=None, stream=False):
        """makes request to xai base service

        :param uri: api uri
        :param method: GET, POST, PUT, DELETE
        :raises Exception: Request exception
        :return: JSON response
        """
        url = f"{self.base_url}/{uri}"
        try:
            response = requests.request(
                method,
                url,
                headers=self.headers,
                json=payload,
                files=files,
                stream=stream,
            )
            if 400 <= response.status_code < 500:
                raise Exception(response.json())
            elif 500 <= response.status_code < 600:
                raise Exception(response.json())
            else:
                return response
        except Exception as e:
            raise e

    def request(self, method, uri, payload):
        self.refresh_bearer_token()
        response = self.base_request(method, uri, payload)
        return response

    def get(self, uri):
        """makes get request to xai base service

        :param uri: api uri
        :raises Exception: Request exception
        :return: JSON response
        """

        self.refresh_bearer_token()
        response = self.base_request("GET", uri)
        return response.json()

    def post(self, uri, payload={}):
        """makes post request to xai base service

        :param uri: api uri
        :param payload: api payload, defaults to {}
        :raises Exception: Request exception
        :return: JSON response
        """

        self.refresh_bearer_token()
        response = self.base_request("POST", uri, payload)

        return response.json()

    def stream(self, uri):
        """makes streaming request to xai base service

        :param uri: api uri
        :param payload: api payload, defaults to {}
        :raises Exception: Request exception
        :return: JSON response
        """

        self.refresh_bearer_token()
        response = self.base_request("GET", uri, stream=True)
        for res in response.iter_lines():
            yield json.loads(res.decode("utf-8"))

    def file(self, uri, files):
        """makes multipart request to send files

        :param uri: api uri
        :param file_path: file path
        :return: JSON response
        """
        self.refresh_bearer_token()
        response = self.base_request("POST", uri, files=files)
        return response.json()
