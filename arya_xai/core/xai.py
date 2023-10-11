import os
from pydantic import BaseModel

from arya_xai.client.client import APIClient
from arya_xai.common.environment import Environment
from arya_xai.common.xai_uris import LOGIN_URI, GET_WORKSPACES_URI

class xai(BaseModel):
    """Base class to connect with AryaXAI platform
    """
    env: Environment = Environment()
    api_client: APIClient = APIClient(base_url=env.get_base_url())

    def login(self, api_key=None):
        """login to AryaXAI platform

        :param api_key: API key, defaults to ARYAXAI_API_KEY environment variable
        """
        api_key = api_key or os.environ.get('ARYAXAI_API_KEY', None)
        
        if not api_key:
            raise ValueError("Either set ARYAXAI_API_KEY or pass the API key in XAIBase class.")

        res = self.api_client.post(LOGIN_URI, payload={'api_key': api_key})
        self.api_client.update_headers(res.auth_token)
        
        print('Authenticated successfully.')
        
    def get_workspaces(self):
        """get user workspaces

        :return: list of workspace
        """
        return self.api_client.get(GET_WORKSPACES_URI)