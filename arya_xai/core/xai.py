import os
from pydantic import BaseModel

from arya_xai.client.client import APIClient
from arya_xai.common.environment import Environment
from arya_xai.common.xai_uris import LOGIN_URI, GET_WORKSPACES_URI
import getpass

class XAI(BaseModel):
    """Base class to connect with AryaXAI platform
    """
    env: Environment = Environment()
    api_client: APIClient = APIClient(base_url=env.get_base_url())

    def login(self):
        """login to AryaXAI platform

        :param api_key: API key, defaults to ARYAXAI_API_KEY environment variable
        """
        access_token = os.environ.get('XAI_ACCESS_TOKEN', None) or getpass.getpass("Enter your Arya XAI Access Token: ")
        
        if not access_token:
            raise ValueError("Either set XAI_ACCESS_TOKEN or pass the Access token")

        res = self.api_client.post(LOGIN_URI, payload={'access_token': access_token})
        self.api_client.update_headers(res['access_token'])
        
        print('Authenticated successfully.')
        
    def get_workspaces(self):
        """get user workspaces

        :return: list of workspace
        """
        workspaces = self.api_client.get(GET_WORKSPACES_URI)
        return workspaces['details']