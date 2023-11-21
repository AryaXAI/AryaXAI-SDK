import pandas as pd
from pydantic import BaseModel
from typing import List
from aryaxai.client.client import APIClient
from aryaxai.common.enums import UserRole
from aryaxai.common.xai_uris import (
    CREATE_PROJECT_URI,
    GET_WORKSPACES_URI,
    UPDATE_WORKSPACE_URI,
    GET_NOTIFICATIONS_URI,
    CLEAR_NOTIFICATIONS_URI,
)
from aryaxai.core.project import Project


class Workspace(BaseModel):
    """Class to work with AryaXAI workspaces"""

    created_by: str
    user_workspace_name: str
    workspace_name: str
    created_at: str

    __api_client: APIClient

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__api_client = kwargs.get("api_client")

    def rename_workspace(self, new_workspace_name: str) -> str:
        """rename the current workspace to new name

        :param new_workspace_name: name for the workspace to be renamed to
        :return: response
        """
        payload = {
            "workspace_name": self.workspace_name,
            "modify_req": {
                "rename_workspace": new_workspace_name,
            },
        }
        res = self.__api_client.post(UPDATE_WORKSPACE_URI, payload)
        self.user_workspace_name = new_workspace_name
        return res.get("details")

    def delete_workspace(self) -> str:
        """deletes the current workspace
        :return: response
        """
        payload = {
            "workspace_name": self.workspace_name,
            "modify_req": {"delete_workspace": self.user_workspace_name},
        }
        res = self.__api_client.post(UPDATE_WORKSPACE_URI, payload)
        return res.get("details")

    def add_user_to_workspace(self, email: str, role: UserRole) -> str:
        """adds user to current workspace

        :param email: user email
        :param role: user role ["admin", "user"]
        :return: response
        """
        payload = {
            "workspace_name": self.workspace_name,
            "modify_req": {
                "add_user_workspace": {
                    "email": email,
                    "role": role,
                },
            },
        }
        res = self.__api_client.post(UPDATE_WORKSPACE_URI, payload)
        return res.get("details")

    def remove_user_from_workspace(self, email: str) -> str:
        """removes user from the current workspace

        :param email: user email
        :return: response
        """
        payload = {
            "workspace_name": self.workspace_name,
            "modify_req": {
                "remove_user_workspace": email,
            },
        }
        res = self.__api_client.post(UPDATE_WORKSPACE_URI, payload)
        return res.get("details")

    def update_user_access_for_workspace(self, email: str, role: UserRole) -> str:
        """update the user access for the workspace

        :param email: user email
        :param role: new user role ["admin", "user"]
        :return: _description_
        """
        payload = {
            "workspace_name": self.workspace_name,
            "modify_req": {
                "update_user_workspace": {
                    "email": email,
                    "role": role,
                }
            },
        }
        res = self.__api_client.post(UPDATE_WORKSPACE_URI, payload)
        return res.get("details")

    def projects(self) -> pd.DataFrame:
        """get user projects for this Workspace

        :return: Projects details dataframe
        """
        workspaces = self.__api_client.get(GET_WORKSPACES_URI)
        current_workspace = next(
            filter(
                lambda workspace: workspace["workspace_name"] == self.workspace_name,
                workspaces["details"],
            )
        )
        projects_df = pd.DataFrame(
            current_workspace["projects"],
            columns=[
                "user_project_name",
                "access_type",
                "created_by",
                "created_at",
                "updated_at",
            ],
        )
        return projects_df

    def project(self, project_name: str) -> Project:
        """Select specific project

        :param project_name: Name of the project
        :return: Project
        """
        workspaces = self.__api_client.get(GET_WORKSPACES_URI)
        current_workspace = next(
            filter(
                lambda workspace: workspace["workspace_name"] == self.workspace_name,
                workspaces["details"],
            )
        )
        projects = [
            Project(api_client=self.__api_client, **project)
            for project in current_workspace["projects"]
        ]

        project = next(
            filter(lambda project: project.user_project_name == project_name, projects),
            None,
        )

        if project is None:
            raise Exception("Project Not Found")

        return project

    def create_project(self, project_name: str) -> Project:
        """creates new project in the current workspace

        :param project_name: name for the project
        :return: response
        """
        payload = {
            "project_name": project_name,
            "workspace_name": self.workspace_name,
        }
        res = self.__api_client.post(CREATE_PROJECT_URI, payload)

        if not res["success"]:
            raise Exception(res["details"])

        project = Project(api_client=self.__api_client, **res["details"])

        return project

    def get_notifications(self) -> pd.DataFrame:
        """get user workspace notifications

        :return: DataFrame
        """
        url = f"{GET_NOTIFICATIONS_URI}?workspace_name={self.workspace_name}"

        res = self.__api_client.get(url)

        if not res["success"]:
            raise Exception("Error while getting workspace notifications.")

        notifications = res["details"]

        if not notifications:
            return "No notifications found."

        return pd.DataFrame(notifications).reindex(
            columns=["project_name", "message", "time"]
        )

    def clear_notifications(self) -> str:
        """clear user workspace notifications

        :raises Exception: _description_
        :return: str
        """
        url = f"{CLEAR_NOTIFICATIONS_URI}?workspace_name={self.workspace_name}"

        res = self.__api_client.post(url)

        if not res["success"]:
            raise Exception("Error while clearing workspace notifications.")

        return res["details"]

    def __print__(self) -> str:
        return f"Workspace(user_workspace_name='{self.user_workspace_name}', created_by='{self.created_by}', created_at='{self.created_at}')"

    def __str__(self) -> str:
        return self.__print__()

    def __repr__(self) -> str:
        return self.__print__()
