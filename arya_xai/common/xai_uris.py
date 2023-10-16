# API version
API_VERSION = 'v1'

# URIs of XAI base service starts here
# Auth
LOGIN_URI = f'{API_VERSION}/access-token/authorize'

# User
GET_WORKSPACES_URI = f'{API_VERSION}/users/workspaces'
CREATE_WORKSPACE_URI = f'{API_VERSION}/users/create_workspace'

# Workspace
UPDATE_WORKSPACE_URI = f'{API_VERSION}/users/workspace_config_update'
CREATE_PROJECT_URI = f'{API_VERSION}/users/create_project'

# Project
UPLOAD_DATA_FILE_URI = f'{API_VERSION}/users/uploadfile'
GET_DATA_SUMMARY_URI = f'{API_VERSION}/project/data_summary'
GET_DATA_DIAGNOSIS_URI = f'{API_VERSION}/project/get_data_diagnosis'
GET_DATA_DRIFT_DIAGNOSIS_REPORT_URI = f'{API_VERSION}/project/get_data_diagnosis_drift_report'
