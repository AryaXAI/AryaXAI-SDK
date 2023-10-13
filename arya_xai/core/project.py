from pydantic import BaseModel
from typing import List

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