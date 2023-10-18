from pydantic import BaseModel


class UsageControl(BaseModel):
    allowed_users_to_haveaccess: int
    allowed_projects_to_create: int
    data_allowance: dict
    feature_allowance: dict
