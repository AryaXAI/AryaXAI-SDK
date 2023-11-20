import os
import pandas as pd
from pydantic import BaseModel
from aryaxai.client.client import APIClient
from aryaxai.common.environment import Environment
from aryaxai.core.workspace import Workspace
from aryaxai.common.xai_uris import (
    CLEAR_NOTIFICATIONS_URI,
    CREATE_WORKSPACE_URI,
    GET_NOTIFICATIONS_URI,
    LOGIN_URI,
    GET_WORKSPACES_URI,
)
import getpass


class XAI(BaseModel):
    """Base class to connect with AryaXAI platform"""

    __env: Environment = Environment()
    __api_client: APIClient

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        debug = self.__env.get_debug()
        base_url = self.__env.get_base_url()

        self.__api_client = APIClient(debug=debug, base_url=base_url)

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

    def workspaces(self) -> pd.DataFrame:
        """get user workspaces

        :return: workspace details dataframe
        """
        workspaces = self.__api_client.get(GET_WORKSPACES_URI)

        workspace_df = pd.DataFrame(
            workspaces["details"],
            columns=[
                "user_workspace_name",
                "access_type",
                "created_by",
                "created_at",
                "updated_at",
            ],
        )

        return workspace_df

    def workspace(self, workspace_name) -> Workspace:
        """select specific workspace

        :param workspace_name: Name of the workspace to be used
        :return: Workspace
        """
        workspaces = self.__api_client.get(GET_WORKSPACES_URI)
        user_workspaces = [
            Workspace(api_client=self.__api_client, **workspace)
            for workspace in workspaces["details"]
        ]

        workspace = next(
            filter(
                lambda workspace: workspace.user_workspace_name == workspace_name,
                user_workspaces,
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

    def get_notifications(self) -> pd.DataFrame:
        """get user notifications

        :return: DataFrame
        """
        res = self.__api_client.get(GET_NOTIFICATIONS_URI)

        if not res["success"]:
            raise Exception("Error while getting user notifications.")

        notifications = res["details"]

        if not notifications:
            return "No notifications found."

        return pd.DataFrame(notifications).reindex(
            columns=["project_name", "message", "time"]
        )

    def clear_notifications(self) -> str:
        """clear user notifications

        :raises Exception: _description_
        :return: str
        """
        res = self.__api_client.post(CLEAR_NOTIFICATIONS_URI)

        if not res["success"]:
            raise Exception("Error while clearing user notifications.")

        return res["details"]
