import json
import os
from typing import List, Optional
import pandas as pd
from pydantic import BaseModel
import requests
from aryaxai.client.client import APIClient
from aryaxai.common.environment import Environment
from aryaxai.core.organization import Organization
from aryaxai.common.xai_uris import (
    AVAILABLE_BATCH_SERVERS_URI,
    AVAILABLE_CUSTOM_SERVERS_URI,
    AVAILABLE_SYNTHETIC_CUSTOM_SERVERS_URI,
    CLEAR_NOTIFICATIONS_URI,
    CREATE_ORGANIZATION_URI,
    GET_CASE_PROFILE_URI,
    GET_NOTIFICATIONS_URI,
    LOGIN_URI,
    UPLOAD_DATA_PROJECT_URI,
    USER_ORGANIZATION_URI,
)
import getpass


class XAI(BaseModel):
    """Base class to connect with AryaXAI platform"""

    env: Environment = Environment()
    api_client: APIClient = APIClient()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        debug = self.env.get_debug()
        base_url = self.env.get_base_url()

        self.api_client = APIClient(debug=debug, base_url=base_url)

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

    def organizations(self) -> pd.DataFrame:
        """Get all organizations associated with user

        :return: Organization details dataframe
        """

        res = self.api_client.get(USER_ORGANIZATION_URI)

        if not res["success"]:
            raise Exception(res.get("details", "Failed to get organizations"))

        res["details"].insert(
            0,
            {
                "name": "Personal",
                "organization_owner": True,
                "organization_admin": True,
                "current_users": 1,
                "created_by": "you",
            },
        )

        organization_df = pd.DataFrame(
            res["details"],
            columns=[
                "name",
                "organization_owner",
                "organization_admin",
                "current_users",
                "created_by",
                "created_at",
            ],
        )

        return organization_df

    def organization(self, organization_name: str) -> Organization:
        """Select specific organization

        :param organization_name: Name of the organization to be used
        :return: Organization object
        """
        if organization_name == "personal":
            return Organization(
                api_client=self.api_client,
                **{
                    "name": "Personal",
                    "organization_owner": True,
                    "organization_admin": True,
                    "current_users": 1,
                    "created_by": "you",
                }
            )

        organizations = self.api_client.get(USER_ORGANIZATION_URI)

        if not organizations["success"]:
            raise Exception(organizations.get("details", "Failed to get organizations"))

        user_organization = [
            Organization(api_client=self.api_client, **organization)
            for organization in organizations["details"]
        ]

        organization = next(
            filter(
                lambda organization: organization.name == organization_name,
                user_organization,
            ),
            None,
        )

        if organization is None:
            raise Exception("Organization Not Found")

        return organization

    def create_organization(self, organization_name: str) -> Organization:
        """Create New Organization

        :param organization_name: Name of the new organization
        :return: Organization object
        """
        payload = {"organization_name": organization_name}
        res = self.api_client.post(CREATE_ORGANIZATION_URI, payload)

        if not res["success"]:
            raise Exception(res.get("details", "Failed to create organization"))

        return Organization(api_client=self.api_client, **res["organization_details"])

    def get_notifications(self) -> pd.DataFrame:
        """get user notifications

        :return: notification details dataFrame
        """
        res = self.api_client.get(GET_NOTIFICATIONS_URI)

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

        :return: response
        """
        res = self.api_client.post(CLEAR_NOTIFICATIONS_URI)

        if not res["success"]:
            raise Exception("Error while clearing user notifications.")

        return res["details"]

    def available_batch_servers(self) -> dict:
        """available custom batch servers

        :return: response
        """
        res = self.api_client.get(AVAILABLE_BATCH_SERVERS_URI)
        return res["details"]

    def available_custom_servers(self) -> dict:
        """available custom servers

        :return: response
        """
        res = self.api_client.get(AVAILABLE_CUSTOM_SERVERS_URI)
        return res

    def available_synthetic_custom_servers(self) -> dict:
        """available synthetic custom servers

        :return: response
        """
        res = self.api_client.get(AVAILABLE_SYNTHETIC_CUSTOM_SERVERS_URI)
        return res["details"]
    
    def register_case(
        self,
        token: str,
        client_id: str,
        unique_identifier: Optional[str] = None,
        project_name: str = None,
        tag: Optional[str] = None,
        data: Optional[str] = None,
        processed_data: Optional[bool] = False,
        merge: Optional[bool] = False,
        image_class: Optional[str] = None,
        prompt: Optional[str] = None,
        serverless_instance_type: Optional[str] = None,
        explainability_method: Optional[str] = None,
        explain_model: Optional[bool] = False,
        session_id: Optional[str] = None,
        xai: Optional[str] = None,
        file_path: Optional[str] = None
    ):
        form_data = {
            "client_id": client_id,
            "project_name": project_name,
            "unique_identifier": unique_identifier,
            "tag": tag,
            "data": json.dumps(data) if isinstance(data, list) else data,
            "processed_data": str(processed_data).lower(),
            "merge": str(merge).lower(),
            "image_class": image_class,
            "prompt": prompt,
            "serverless_instance_type": serverless_instance_type,
            "explainability_method": explainability_method,
            "explain_model": str(explain_model).lower(),
            "session_id": str(session_id).lower(),
            "xai": xai
        }
        headers = {
            "x-api-token": token
        }
        form_data = {k: v for k, v in form_data.items() if v is not None}
        files = {}
        if file_path:
            files["in_file"] = open(file_path, "rb")
        response = requests.post(
            self.env.get_base_url() + "/" + UPLOAD_DATA_PROJECT_URI,
            data=form_data,
            files=files if files else None,
            headers=headers
        ).json()
        if files:
            files["in_file"].close()
        return response

    def case_profile(
        self,
        token: str,
        client_id: str,
        unique_identifier: Optional[str] = None,
        project_name: str = None,
        tag: str = None,
        xai: Optional[List[str]] = None,
        refresh: Optional[bool] = None
    ):
        headers = {
            "x-api-token": token
        }
        payload = {
            "client_id": client_id,
            "project_name": project_name,
            "unique_identifier": unique_identifier,
            "tag": tag,
            "xai": xai,
            "refresh": refresh
        }
        res = requests.post(
            self.env.get_base_url() + "/" + GET_CASE_PROFILE_URI,
            headers=headers,
            json=payload
        ).json()
        return res["details"]
