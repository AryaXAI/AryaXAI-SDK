from pydantic import BaseModel
from typing import List
from arya_xai.common.xai_uris import (
    GET_DATA_DIAGNOSIS_URI,
    GET_DATA_SUMMARY,
    GET_DATA_DRIFT_DIAGNOSIS_REPORT_URI,
)


class Project(BaseModel):
    created_by: str
    project_name: str
    user_project_name: str
    user_workspace_name: str
    workspace_name: str
    project_data_dir_path: str
    collections_name: List[str]
    user_access: List[str]
    created_at: str
    updated_at: str
    metadata: dict
    access_type: str

    # def get_data_summary(self):
    #     res = self.api_client.post(f"{GET_DATA_SUMMARY}/?project_name={self.user_project_name}&refresh=true'")
    #     return res

    # def get_data_diagnosis(self):
    #     res = self.api_client.get(f"{GET_DATA_DIAGNOSIS_URI}/?project_name={self.user_project_name}")
    #     return res

    # def get_data_diagnosis_drift_report(self):
    #     res = self.api_client.get(f"{GET_DATA_DRIFT_DIAGNOSIS_REPORT_URI}/?project_name={self.user_project_name}")
    #     return res

    def __print__(self) -> str:
        return f"Project(user_project_name='{self.user_project_name}', created_by='{self.created_by}', created_at='{self.created_at}')"

    def __str__(self) -> str:
        return self.__print__()

    def __repr__(self) -> str:
        return self.__print__()
