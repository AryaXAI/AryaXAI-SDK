# API version
API_VERSION = "v1"

# URIs of XAI base service starts here
# Auth
LOGIN_URI = f"{API_VERSION}/access-token/authorize"

# User
GET_WORKSPACES_URI = f"{API_VERSION}/users/workspaces"
CREATE_WORKSPACE_URI = f"{API_VERSION}/users/create_workspace"

# Workspace
UPDATE_WORKSPACE_URI = f"{API_VERSION}/users/workspace_config_update"
CREATE_PROJECT_URI = f"{API_VERSION}/users/create_project"

# Project
UPDATE_PROJECT_URI = f"{API_VERSION}/users/project_config_update"
UPLOAD_DATA_FILE_URI = f"{API_VERSION}/project/uploadfile_with_info"
UPLOAD_DATA_FILE_INFO_URI = f"{API_VERSION}/project/get_Uploaded_file_info"
DELETE_DATA_FILE_URI = f"{API_VERSION}/project/delete_data"
ALL_DATA_FILE_URI = f"{API_VERSION}/project/get_all_uploaded_files"
UPLOAD_DATA_URI = f"{API_VERSION}/project/upload_data"
UPLOAD_DATA_WITH_CHECK_URI = f"{API_VERSION}/project/upload_data_with_check"
GET_DATA_SUMMARY_URI = f"{API_VERSION}/project/data_summary"
GET_DATA_DIAGNOSIS_URI = f"{API_VERSION}/project/get_data_diagnosis"
DATA_DRFIT_DIAGNOSIS_URI = f"{API_VERSION}/project/run_data_drift_diagnosis"
GET_PROJECT_CONFIG = f"{API_VERSION}/users/get_xai_config"
AVAILABLE_TAGS_URI = f"{API_VERSION}/project/get_all_available_tags_info"

# Monitoring
DATA_DRIFT_DASHBOARD_URI = f"{API_VERSION}/dashboard/data_drift_dashboard"
TARGET_DRIFT_DASHBOARD_URI = f"{API_VERSION}/dashboard/target_drift_dashboard"
BIAS_MONITORING_DASHBOARD_URI = f"{API_VERSION}/dashboard/bias_monitoring_dashboard"
MODEL_PERFORMANCE_DASHBOARD_URI = f"{API_VERSION}/dashboard/model_performance_dashboard"

GET_LABELS_URI = f"{API_VERSION}/triggers/get_label_classes"

# Auto ML
MODEL_PARAMETERS_URI = f"{API_VERSION}/users/get_xai_model_parameters"
TRAIN_MODEL_URI = f"{API_VERSION}/users/xai_config_update"
GET_MODELS_URI = f"{API_VERSION}/ai-models/get_all_models"
UPDATE_ACTIVE_MODEL_URI = f"{API_VERSION}/ai-models/update_active_model"
REMOVE_MODEL_URI = f"{API_VERSION}/ai-models/remove_model"
RUN_MODEL_ON_DATA_URI = f"{API_VERSION}/ai-models/run_model_on_data"
DOWNLOAD_TAG_DATA_URI = f"{API_VERSION}/ai-models/download_tag_data"
MODEL_SUMMARY_URI = f"{API_VERSION}/project/get_model_perfermance"
MODEL_SVG_URI = f"{API_VERSION}/project/get_model_svg_plot"
GET_MODEL_PERFORMANCE_URI = f"{API_VERSION}/dashboard/model_performance"

# Explainability
GET_CASES_URI = f"{API_VERSION}/ai-models/get_cases"
SEARCH_CASE_URI = f"{API_VERSION}/ai-models/search_case"
CASE_INFO_URI = f"{API_VERSION}/ai-models/get_case_info"
DELETE_CASE_URI = f"{API_VERSION}/project/delete_data_with_filter"

# Alerts
GET_TRIGGERS_URI = f"{API_VERSION}/triggers/get_triggers"
CREATE_TRIGGER_URI = f"{API_VERSION}/triggers/update_triggers"
DELETE_TRIGGER_URI = f"{API_VERSION}/triggers/update_triggers"
EXECUTED_TRIGGER_URI = f"{API_VERSION}/triggers/get_executed_triggers"
