import os

# API version
API_VERSION = os.getenv("XAI_API_VERSION", "v1")
API_VERSION_V2 = "v2"

# APP
XAI_APP_URI = "https://beta.aryaxai.com"

# URIs of XAI base service starts here
# Auth
LOGIN_URI = f"{API_VERSION_V2}/access-token/authorize"

# User
GET_WORKSPACES_URI = f"{API_VERSION_V2}/users/workspaces"
GET_WORKSPACES_DETAILS_URI = f"{API_VERSION_V2}/project/details/workspace"
CREATE_WORKSPACE_URI = f"{API_VERSION_V2}/users/create_workspace"

# Custom Server
AVAILABLE_CUSTOM_SERVERS_URI = f"{API_VERSION_V2}/project/server_types"
START_CUSTOM_SERVER_URI = f"{API_VERSION_V2}/project/start_server"
STOP_CUSTOM_SERVER_URI = f"{API_VERSION_V2}/project/stop_server"

# Batch Server
AVAILABLE_BATCH_SERVERS_URI = f"{API_VERSION_V2}/users/automl_custom_servers"

# Notifications
GET_NOTIFICATIONS_URI = f"{API_VERSION_V2}/notifications/fetch"
CLEAR_NOTIFICATIONS_URI = f"{API_VERSION_V2}/notifications/clear"

# Workspace
UPDATE_WORKSPACE_URI = f"{API_VERSION_V2}/users/workspace_config_update"
CREATE_PROJECT_URI = f"{API_VERSION_V2}/users/create_project"

# Project
UPDATE_PROJECT_URI = f"{API_VERSION_V2}/users/project_config_update"
UPLOAD_DATA_FILE_URI = f"{API_VERSION_V2}/project/uploadfile_with_info"
UPLOAD_DATA_FILE_INFO_URI = f"{API_VERSION_V2}/project/get_Uploaded_file_info"
DELETE_DATA_FILE_URI = f"{API_VERSION_V2}/project/delete_data"
ALL_DATA_FILE_URI = f"{API_VERSION_V2}/project/get_all_uploaded_files"
UPLOAD_DATA_URI = f"{API_VERSION_V2}/project/upload_data"
UPLOAD_DATA_WITH_CHECK_URI = f"{API_VERSION_V2}/project/upload_data_with_check"
UPLOAD_MODEL_URI = f"{API_VERSION_V2}/project/upload_model"
GET_MODEL_TYPES_URI = f"{API_VERSION_V2}/project/get_model_types"
GET_DATA_SUMMARY_URI = f"{API_VERSION_V2}/project/data_summary"
GET_DATA_DIAGNOSIS_URI = f"{API_VERSION_V2}/project/get_data_diagnosis"
RUN_DATA_DRIFT_DIAGNOSIS_URI = f"{API_VERSION_V2}/project/run_data_drift_diagnosis"
GET_DATA_DRIFT_DIAGNOSIS_URI = f"{API_VERSION_V2}/project/get_data_drift_diagnosis"
GET_PROJECT_CONFIG = f"{API_VERSION_V2}/users/get_xai_config"
AVAILABLE_TAGS_URI = f"{API_VERSION_V2}/project/get_all_available_tags_info"
TAG_DATA_URI = f"{API_VERSION_V2}/project/tag_data"

# Monitoring
GENERATE_DASHBOARD_URI = f"{API_VERSION_V2}/dashboards/generate_dashboard"
DASHBOARD_CONFIG_URI = f"{API_VERSION_V2}/dashboards/config"
MODEL_PERFORMANCE_DASHBOARD_URI = (
    f"{API_VERSION_V2}/dashboard/model_performance_dashboard"
)
DASHBOARD_LOGS_URI = f"{API_VERSION_V2}/dashboards/get_dashboard_logs"
GET_DASHBOARD_URI = f"{API_VERSION_V2}/dashboards/get_dashboard"
DOWNLOAD_DASHBOARD_LOGS_URI = f"{API_VERSION_V2}/dashboards/download_dashboard_logs"

# Auto ML
MODEL_PARAMETERS_URI = f"{API_VERSION_V2}/users/get_xai_model_parameters"
TRAIN_MODEL_URI = f"{API_VERSION_V2}/users/xai_config_update"
GET_MODELS_URI = f"{API_VERSION_V2}/ai-models/get_all_models"
UPDATE_ACTIVE_MODEL_URI = f"{API_VERSION_V2}/ai-models/update_active_model"
REMOVE_MODEL_URI = f"{API_VERSION_V2}/ai-models/remove_model"
RUN_MODEL_ON_DATA_URI = f"{API_VERSION_V2}/ai-models/run_model_on_data"
DOWNLOAD_TAG_DATA_URI = f"{API_VERSION_V2}/ai-models/download_tag_data"
MODEL_SUMMARY_URI = f"{API_VERSION_V2}/project/get_model_perfermance"
MODEL_SVG_URI = f"{API_VERSION_V2}/project/get_model_svg_plot"
MODEL_INFERENCES_URI = f"{API_VERSION_V2}/ai-models/get_all_tags_for_models"

# Explainability
GET_CASES_URI = f"{API_VERSION_V2}/ai-models/get_cases"
SEARCH_CASE_URI = f"{API_VERSION_V2}/ai-models/search_case"
CASE_INFO_URI = f"{API_VERSION_V2}/ai-models/get_case_info"
CASE_DTREE_URI = f"{API_VERSION_V2}/ai-models/get_case_dtree"
DELETE_CASE_URI = f"{API_VERSION_V2}/project/delete_data_with_filter"
CASE_LOGS_URI = f"{API_VERSION_V2}/ai-models/explainability_logs"
GET_VIEWED_CASE_URI = f"{API_VERSION_V2}/ai-models/get_viewed_case"

# Observations
GET_OBSERVATIONS_URI = f"{API_VERSION_V2}/observations/get_observations"
GET_OBSERVATION_PARAMS_URI = f"{API_VERSION_V2}/observations/get_observation_params"
CREATE_OBSERVATION_URI = f"{API_VERSION_V2}/observations/create_observation"
UPDATE_OBSERVATION_URI = f"{API_VERSION_V2}/observations/observation_config_update"

# Policies
GET_POLICIES_URI = f"{API_VERSION_V2}/policies/get_policies"
GET_POLICY_PARAMS_URI = f"{API_VERSION_V2}/policies/get_policy_params"
CREATE_POLICY_URI = f"{API_VERSION_V2}/policies/create_policy"
UPDATE_POLICY_URI = f"{API_VERSION_V2}/policies/policy_config_update"

# Alerts
GET_TRIGGERS_URI = f"{API_VERSION_V2}/triggers/get_triggers"
CREATE_TRIGGER_URI = f"{API_VERSION_V2}/triggers/update_triggers"
DELETE_TRIGGER_URI = f"{API_VERSION_V2}/triggers/update_triggers"
EXECUTED_TRIGGER_URI = f"{API_VERSION_V2}/triggers/get_executed_triggers"
GET_EXECUTED_TRIGGER_INFO = f"{API_VERSION_V2}/triggers/get_trigger_details"
GET_LABELS_URI = f"{API_VERSION_V2}/triggers/get_label_classes"

# Synthetic AI
AVAILABLE_SYNTHETIC_CUSTOM_SERVERS_URI = f"{API_VERSION_V2}/synthetics/custom_servers"
GET_SYNTHETIC_MODEL_PARAMS_URI = (
    f"{API_VERSION_V2}/synthetics/get_synthetic_model_parameters"
)
TRAIN_SYNTHETIC_MODEL_URI = f"{API_VERSION_V2}/synthetics/train_synthetic_model"
GET_SYNTHETIC_MODELS_URI = f"{API_VERSION_V2}/synthetics/get_synthetics_models"
DELETE_SYNTHETIC_MODEL_URI = f"{API_VERSION_V2}/synthetics/delete_synthetic_model"
GET_SYNTHETIC_MODEL_DETAILS_URI = (
    f"{API_VERSION_V2}/synthetics/get_synthetic_model_details"
)
GENERATE_SYNTHETIC_DATA_URI = f"{API_VERSION_V2}/synthetics/generate_synthetic_data"
GENERATE_ANONYMITY_SCORE_URI = f"{API_VERSION_V2}/synthetics/generate_anonimity_score"
GET_ANONYMITY_SCORE_URI = f"{API_VERSION_V2}/synthetics/get_anonimity_score"

GET_SYNTHETIC_DATA_TAGS_URI = f"{API_VERSION_V2}/synthetics/get_synthetic_data_tags"
DOWNLOAD_SYNTHETIC_DATA_URI = f"{API_VERSION_V2}/synthetics/download_synthetic_data"
DELETE_SYNTHETIC_TAG_URI = f"{API_VERSION_V2}/project/delete_data_with_filter"

CREATE_SYNTHETIC_PROMPT_URI = f"{API_VERSION_V2}/synthetics/create_synthetic_prompts"
UPDATE_SYNTHETIC_PROMPT_URI = f"{API_VERSION_V2}/synthetics/synthetic_prompts_update"
GET_SYNTHETIC_PROMPT_URI = f"{API_VERSION_V2}/synthetics/get_synthetics_promts"

# Events
POLL_EVENTS = f"{API_VERSION_V2}/events/poll"
FETCH_EVENTS = f"{API_VERSION_V2}/events/fetch"

# Organization
USER_ORGANIZATION_URI = f"{API_VERSION_V2}/organization/user_organization"
CREATE_ORGANIZATION_URI = f"{API_VERSION_V2}/organization/create_organization"
INVITE_USER_ORGANIZATION_URI = f"{API_VERSION_V2}/organization/invite_user"
REMOVE_USER_ORGANIZATION_URI = f"{API_VERSION_V2}/organization/organization_user_delete"
ORGANIZATION_MEMBERS_URI = f"{API_VERSION_V2}/organization/organization_users"
