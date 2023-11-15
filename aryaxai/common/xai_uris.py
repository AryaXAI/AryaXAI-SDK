import os

# API version
API_VERSION = os.getenv("XAI_API_VERSION", "v1")

# URIs of XAI base service starts here
# Auth
LOGIN_URI = f"{API_VERSION}/access-token/authorize"

# User
GET_WORKSPACES_URI = f"{API_VERSION}/users/workspaces"
CREATE_WORKSPACE_URI = f"{API_VERSION}/users/create_workspace"
GET_NOTIFICATIONS_URI = f"{API_VERSION}/users/notifications"
CLEAR_NOTIFICATIONS_URI = f"{API_VERSION}/users/clear-notifications"

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
UPLOAD_MODEL_URI = f"{API_VERSION}/project/upload_model"
GET_MODEL_TYPES_URI = f"{API_VERSION}/project/get_model_types"
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

# Observations
GET_OBSERVATIONS_URI = f"{API_VERSION}/observations/get_observations"
GET_OBSERVATION_PARAMS_URI = f"{API_VERSION}/observations/get_observation_params"
CREATE_OBSERVATION_URI = f"{API_VERSION}/observations/create_observation"
UPDATE_OBSERVATION_URI = f"{API_VERSION}/observations/observation_config_update"

# Policies
GET_POLICIES_URI = f"{API_VERSION}/policies/get_policies"
GET_POLICY_PARAMS_URI = f"{API_VERSION}/policies/get_policy_params"
CREATE_POLICY_URI = f"{API_VERSION}/policies/create_policy"
UPDATE_POLICY_URI = f"{API_VERSION}/policies/policy_config_update"

# Alerts
GET_TRIGGERS_URI = f"{API_VERSION}/triggers/get_triggers"
CREATE_TRIGGER_URI = f"{API_VERSION}/triggers/update_triggers"
DELETE_TRIGGER_URI = f"{API_VERSION}/triggers/update_triggers"
EXECUTED_TRIGGER_URI = f"{API_VERSION}/triggers/get_executed_triggers"
GET_EXECUTED_TRIGGER_INFO = f"{API_VERSION}/triggers/get_trigger_details"

# Synthetic AI
GET_SYNTHETIC_MODEL_PARAMS_URI = (
    f"{API_VERSION}/synthetics/get_synthetic_model_parameters"
)
TRAIN_SYNTHETIC_MODEL_URI = f"{API_VERSION}/synthetics/train_synthetic_model"
GET_SYNTHETIC_MODELS_URI = f"{API_VERSION}/synthetics/get_synthetics_models"
DELETE_SYNTHETIC_MODEL_URI = f"{API_VERSION}/synthetics/delete_synthetic_model"
GET_SYNTHETIC_MODEL_DETAILS_URI = (
    f"{API_VERSION}/synthetics/get_synthetic_model_details"
)
GET_SYNTHETIC_TRAINING_LOGS_URI = f"{API_VERSION}/synthetics/get_training_logs"
GENERATE_SYNTHETIC_DATA_URI = f"{API_VERSION}/synthetics/generate_synthetic_data"
GENERATE_ANONYMITY_SCORE_URI = f"{API_VERSION}/synthetics/generate_anonimity_score"
GET_ANONYMITY_SCORE_URI = f"{API_VERSION}/synthetics/get_anonimity_score"

GET_SYNTHETIC_DATA_TAGS_URI = f"{API_VERSION}/synthetics/get_synthetic_data_tags"
DOWNLOAD_SYNTHETIC_DATA_URI = f"{API_VERSION}/synthetics/download_synthetic_data"
DELETE_SYNTHETIC_TAG_URI = f"{API_VERSION}/project/delete_data_with_filter"

CREATE_SYNTHETIC_PROMPT_URI = f"{API_VERSION}/synthetics/create_synthetic_prompts"
UPDATE_SYNTHETIC_PROMPT_URI = f"{API_VERSION}/synthetics/synthetic_prompts_update"
GET_SYNTHETIC_PROMPT_URI = f"{API_VERSION}/synthetics/get_synthetics_promts"

# Events
POLL_EVENTS = f"{API_VERSION}/events/poll"
