import os
from pydantic import BaseModel
from aryaxai.client.client import APIClient
from aryaxai.common.environment import Environment

from aryaxai.common.xai_uris import CREATE_WORKSPACE_URI, LOGIN_URI, GET_WORKSPACES_URI
import getpass
from typing import List

from aryaxai.core.workspace import Workspace


class XAI(BaseModel):
    """Base class to connect with AryaXAI platform"""

    env: Environment = Environment()
    __api_client: APIClient

    def __init__(self):
        super().__init__()

        self.env = Environment()
        self.__api_client = APIClient(
            base_url=self.env.get_base_url()
        )

    def login(self):
        """login to AryaXAI platform

        :param api_key: API key, defaults to XAI_ACCESS_TOKEN environment variable
        """
        access_token = os.environ.get("XAI_ACCESS_TOKEN", None) or getpass.getpass(
            "Enter your Arya XAI Access Token: "
        )

        if not access_token:
            raise ValueError("Either set XAI_ACCESS_TOKEN or pass the Access token")

        res = self.__api_client.post(LOGIN_URI, payload={"access_token": access_token})
        self.__api_client.update_headers(res["access_token"])
        self.__api_client.set_access_token(access_token)

        print("Authenticated successfully.")

    def workspaces(self) -> List[Workspace]:
        """get user workspaces

        :return: list of workspace
        """
        user_workspaces = []

        workspaces = self.__api_client.get(GET_WORKSPACES_URI)
        user_workspaces = [ 
            Workspace(api_client=self.__api_client, **workspace)
            for workspace in workspaces["details"]
        ]

        return user_workspaces

    def workspace(self, workspace_name) -> Workspace:
        """select specific workspace

        :param workspace_name: Name of the workspace to be used
        :return: Workspace
        """
        workspaces = self.workspaces()

        workspace = next(
            filter(
                lambda workspace: workspace.user_workspace_name == workspace_name,
                workspaces,
            ),
            None,
        )

        if workspace is None:
            raise Exception("Workspace Not Found")

        return workspace

    def create_workspace(self, workspace_name: str) -> Workspace:
        """create user workspace

        :param workspace_name: name for the workspace
        :return: response
        """

        res = self.__api_client.post(
            CREATE_WORKSPACE_URI, {"workspace_name": workspace_name}
        )
        if not res["success"]:
            raise Exception(res.get("details"))

        workspace = Workspace(api_client=self.__api_client, **res["workspace_details"])

        return workspace
