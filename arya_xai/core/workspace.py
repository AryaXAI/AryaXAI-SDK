from pydantic import BaseModel
from typing import List
from arya_xai.client.client import APIClient
from arya_xai.common.enums import UserRole
from arya_xai.common.xai_uris import CREATE_PROJECT_URI, UPDATE_WORKSPACE_URI

from arya_xai.core.usage_control import UsageControl
from arya_xai.core.project import Project


class Workspace(BaseModel):
    """Class to work with AryaXAI workspaces"""

    created_by: str
    user_workspace_name: str
    workspace_name: str
    projects: List[Project]
    user_access: List[str]
    created_at: str
    updated_at: str
    enterprise: bool
    usage_control: UsageControl
    access_type: str
    api_client: APIClient

    def __init__(self, api_client, **kwargs):
        super().__init__(api_client=api_client, **kwargs)

    def get_projects(self):
        """get user projects for this Workspace

        :return: list of Projects
        """
        return self.projects

    def rename_workspace(self, new_workspace_name: str) -> str:
        """rename the current workspace to new name

        :param new_workspace_name: name for the workspace to be renamed to
        :return: response
        """
        payload = {
            "workspace_name": self.user_workspace_name,
            "modify_req": {"rename_workspace": new_workspace_name},
        }
        res = self.api_client.post(UPDATE_WORKSPACE_URI, payload)
        self.user_workspace_name = new_workspace_name
        return res.get("details")

    def delete_workspace(self) -> str:
        """deletes the current workspace
        :return: response
        """
        payload = {
            "workspace_name": self.user_workspace_name,
            "modify_req": {"delete_workspace": self.user_workspace_name},
        }
        res = self.api_client.post(UPDATE_WORKSPACE_URI, payload)
        return res.get("details")

    def add_user_to_workspace(self, email: str, role: UserRole) -> str:
        """adds user to current workspace

        :param email: user email
        :param role: user role ["admin", "user"]
        :return: response
        """
        payload = {
            "workspace_name": self.user_workspace_name,
            "modify_req": {"add_user_workspace": {"email": email, "role": role}},
        }
        res = self.api_client.post(UPDATE_WORKSPACE_URI, payload)
        return res.get("details")

    def remove_user_from_workspace(self, email: str) -> str:
        """removes user from the current workspace

        :param email: user email
        :return: response
        """
        payload = {
            "workspace_name": self.user_workspace_name,
            "modify_req": {"remove_user_workspace": email},
        }
        res = self.api_client.post(UPDATE_WORKSPACE_URI, payload)
        return res.get("details")

    def update_user_access_for_workspace(self, email: str, role: UserRole) -> str:
        """update the user access for the workspace

        :param email: user email
        :param role: new user role ["admin", "user"]
        :return: _description_
        """
        payload = {
            "workspace_name": self.user_workspace_name,
            "modify_req": {"update_user_workspace": {"email": email, "role": role}},
        }
        res = self.api_client.post(UPDATE_WORKSPACE_URI, payload)
        return res.get("details")

    def create_project(self, project_name: str) -> Project:
        """creates new project in the current workspace

        :param project_name: name for the project
        :return: response
        """
        payload = {"project_name": project_name, "workspace_name": self.workspace_name}
        res = self.api_client.post(CREATE_PROJECT_URI, payload)
        project = Project(**res["details"])
        return project

    def __print__(self) -> str:
        return f"Workspace(user_workspace_name='{self.user_workspace_name}', created_by='{self.created_by}', created_at='{self.created_at}')"

    def __str__(self) -> str:
        return self.__print__()

    def __repr__(self) -> str:
        return self.__print__()
