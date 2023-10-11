from pydantic import BaseModel
from typing import List

from arya_xai.core.usage_control import UsageControl
from arya_xai.core.project import Project

class Workspace(BaseModel):
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

    def get_projects(self):
        return self.projects
    
    def __str__(self) -> str:
        return self.user_workspace_name