import os
from pydantic import BaseModel
from arya_xai.client.client import APIClient
from arya_xai.common.environment import Environment

from arya_xai.common.xai_uris import CREATE_WORKSPACE_URI, LOGIN_URI, GET_WORKSPACES_URI
import getpass
from typing import List

from arya_xai.core.workspace import Workspace


class XAI(BaseModel):
    """Base class to connect with AryaXAI platform"""

    env: Environment = Environment()
    api_client: APIClient = APIClient(base_url=env.get_base_url())

    def login(self):
        """login to AryaXAI platform

        :param api_key: API key, defaults to XAI_ACCESS_TOKEN environment variable
        """
        access_token = os.environ.get("XAI_ACCESS_TOKEN", None) or getpass.getpass(
            "Enter your Arya XAI Access Token: "
        )

        if not access_token:
            raise ValueError("Either set XAI_ACCESS_TOKEN or pass the Access token")

        res = self.api_client.post(LOGIN_URI, payload={"access_token": access_token})
        self.api_client.update_headers(res["access_token"])
        self.api_client.set_access_token(access_token)

        print("Authenticated successfully.")

    def get_workspaces(self):
        """get user workspaces

        :return: list of workspace
        """
        user_workspaces = []

        workspaces = self.api_client.get(GET_WORKSPACES_URI)
        user_workspaces = [
            Workspace(api_client=self.api_client, **workspace)
            for workspace in workspaces["details"]
        ]

        return user_workspaces

    def create_workspace(self, workspace_name: str) -> Workspace:
        """create user workspace

        :param workspace_name: name for the workspace
        :return: response
        """

        res = self.api_client.post(
            CREATE_WORKSPACE_URI, {"workspace_name": workspace_name}
        )
        return res.get("details")
