import pandas as pd
from pydantic import BaseModel
from typing import Optional
from aryaxai.client.client import APIClient
from aryaxai.common.validation import Validate
from aryaxai.common.xai_uris import (
    AVAILABLE_CUSTOM_SERVERS_URI,
    CREATE_WORKSPACE_URI,
    GET_WORKSPACES_URI,
    INVITE_USER_ORGANIZATION_URI,
    ORGANIZATION_MEMBERS_URI,
    REMOVE_USER_ORGANIZATION_URI,
)
from aryaxai.core.workspace import Workspace


class Organization(BaseModel):
    """Class to work with AryaXAI organizations"""

    organization_id: Optional[str] = None
    name: str
    created_by: str
    created_at: Optional[str] = None

    api_client: APIClient

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_client = kwargs.get("api_client")

    def add_user_to_organization(self, user_email: str) -> str:
        """Add user to Organization

        :param user_email: Email of user to be added to organization.
        :return: response
        """
        payload = {
            "email": user_email,
            "organization_id": self.organization_id,
        }
        res = self.api_client.post(INVITE_USER_ORGANIZATION_URI, payload)

        if not res["success"]:
            raise Exception(res.get("details", "Failed to add user to organization"))

        return res.get("details", "User added successfully")

    def remove_user_from_organization(self, user_email: str) -> str:
        """Remove user from Organization

        :param user_email: Email of user to be removed from organization.
        :return: response
        """
        payload = {
            "organization_user_email": user_email,
            "organization_id": self.organization_id,
        }
        res = self.api_client.post(REMOVE_USER_ORGANIZATION_URI, payload)

        if not res["success"]:
            raise Exception(
                res.get("details", "Failed to remove user from organization")
            )

        return res.get("details", "User removed successfully")

    def member_details(self) -> pd.DataFrame:
        """Organization Member details

        :return: member details dataframe
        """
        res = self.api_client.get(
            f"{ORGANIZATION_MEMBERS_URI}?organization_id={self.organization_id}"
        )

        if not res["success"]:
            raise Exception(
                res.get("details", "Failed to get organization member details")
            )

        member_details_df = pd.DataFrame(
            res.get("details").get("users"),
            columns=[
                "full_name",
                "email",
                "organization_owner",
                "organization_admin",
                "created_at",
            ],
        )

        return member_details_df

    def workspaces(self) -> pd.DataFrame:
        """get user workspaces

        :return: workspace details dataframe
        """

        url = GET_WORKSPACES_URI
        if self.organization_id:
            url = url + f"?organization_id={self.organization_id}"
        workspaces = self.api_client.get(url)

        workspace_df = pd.DataFrame(
            workspaces["details"],
            columns=[
                "user_workspace_name",
                "access_type",
                "created_by",
                "created_at",
                "updated_at",
                "instance_type",
                "instance_status",
            ],
        )

        return workspace_df

    def workspace(self, workspace_name: str) -> Workspace:
        """select specific workspace

        :param workspace_name: Name of the workspace to be used
        :return: Workspace
        """

        url = GET_WORKSPACES_URI
        if self.organization_id:
            url = url + f"?organization_id={self.organization_id}"
        workspaces = self.api_client.get(url)
        user_workspaces = [
            Workspace(api_client=self.api_client, **workspace)
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

    def create_workspace(
        self, workspace_name: str, server_type: Optional[str] = None
    ) -> Workspace:
        """create user workspace

        :param workspace_name: name for the workspace
        :param server_type: dedicated instance to run workloads
            for all available instances check xai.available_custom_servers()
            defaults to shared
        :return: response
        """
        payload = {"workspace_name": workspace_name}

        if server_type:
            custom_servers = self.api_client.get(AVAILABLE_CUSTOM_SERVERS_URI)
            Validate.value_against_list(
                "server_type",
                server_type,
                [server["name"] for server in custom_servers],
            )

            payload["instance_type"] = server_type

        if self.organization_id:
            payload["organization_id"] = self.organization_id

        res = self.api_client.post(CREATE_WORKSPACE_URI, payload)

        if not res["success"]:
            raise Exception(res.get("details"))

        workspace = Workspace(api_client=self.api_client, **res["workspace_details"])

        return workspace

    def __print__(self) -> str:
        return f"Organization(name='{self.name}', created_by='{self.created_by}', created_at='{self.created_at}')"

    def __str__(self) -> str:
        return self.__print__()

    def __repr__(self) -> str:
        return self.__print__()
