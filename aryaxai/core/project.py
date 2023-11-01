from pydantic import BaseModel
from typing import List, Optional
from aryaxai.client.client import APIClient
from aryaxai.common.constants import (
    DATA_DRIFT_TRIGGER_REQUIRED_FIELDS,
    MAIL_FREQUENCIES,
    MODEL_PERF_METRICS_CLASSIFICATION,
    MODEL_PERF_METRICS_REGRESSION,
    MODEL_PERF_TRIGGER_REQUIRED_FIELDS,
    MODEL_TYPES,
    DATA_DRIFT_DASHBOARD_REQUIRED_FIELDS,
    DATA_DRIFT_STAT_TESTS,
    TARGET_DRIFT_DASHBOARD_REQUIRED_FIELDS,
    TARGET_DRIFT_MODEL_TYPES,
    TARGET_DRIFT_STAT_TESTS,
    BIAS_MONITORING_DASHBOARD_REQUIRED_FIELDS,
    MODEL_PERF_DASHBOARD_REQUIRED_FIELDS,
    TARGET_DRIFT_STAT_TESTS_CLASSIFICATION,
    TARGET_DRIFT_STAT_TESTS_REGRESSION,
    TARGET_DRIFT_TRIGGER_REQUIRED_FIELDS,
)
from aryaxai.common.types import DataConfig, ProjectConfig
from aryaxai.common.validation import Validate
from aryaxai.common.monitoring import (
    BiasMonitoringPayload,
    DataDriftPayload,
    ModelPerformancePayload,
    TargetDriftPayload,
)

import pandas as pd

from aryaxai.common.xai_uris import (
    ALL_DATA_FILE_URI,
    AVAILABLE_TAGS_URI,
    CASE_INFO_URI,
    CREATE_TRIGGER_URI,
    DATA_DRFIT_DIAGNOSIS_URI,
    DELETE_CASE_URI,
    DELETE_DATA_FILE_URI,
    DOWNLOAD_TAG_DATA_URI,
    DELETE_TRIGGER_URI,
    EXECUTED_TRIGGER_URI,
    GET_CASES_URI,
    GET_DATA_DIAGNOSIS_URI,
    GET_DATA_SUMMARY_URI,
    GET_EXECUTED_TRIGGER_INFO,
    GET_LABELS_URI,
    GET_MODEL_PERFORMANCE_URI,
    GET_MODELS_URI,
    GET_PROJECT_CONFIG,
    MODEL_PARAMETERS_URI,
    MODEL_SUMMARY_URI,
    REMOVE_MODEL_URI,
    RUN_MODEL_ON_DATA_URI,
    SEARCH_CASE_URI,
    TRAIN_MODEL_URI,
    UPDATE_ACTIVE_MODEL_URI,
    GET_TRIGGERS_URI,
    UPDATE_PROJECT_URI,
    UPLOAD_DATA_FILE_INFO_URI,
    UPLOAD_DATA_FILE_URI,
    UPLOAD_DATA_URI,
    UPLOAD_DATA_WITH_CHECK_URI,
    DATA_DRIFT_DASHBOARD_URI,
    TARGET_DRIFT_DASHBOARD_URI,
    BIAS_MONITORING_DASHBOARD_URI,
    MODEL_PERFORMANCE_DASHBOARD_URI,
)
import json
import io
from aryaxai.core.alert import Alert

from aryaxai.core.case import Case
from aryaxai.core.model_summary import ModelSummary

from aryaxai.core.dashboard import Dashboard


class Project(BaseModel):
    created_by: str
    project_name: str
    user_project_name: str
    user_workspace_name: str
    workspace_name: str

    __api_client: APIClient

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__api_client = kwargs.get("api_client")

    def rename_project(self, new_project_name: str) -> str:
        """Renames current project

        :param new_project_name: new name for the project
        :return: response
        """
        payload = {
            "project_name": self.project_name,
            "modify_req": {
                "rename_project": new_project_name,
            },
        }
        res = self.__api_client.post(UPDATE_PROJECT_URI, payload)
        self.user_project_name = new_project_name
        return res.get("details")

    def delete_project(self) -> str:
        """Deletes current project

        :return: response
        """
        payload = {
            "project_name": self.project_name,
            "modify_req": {
                "delete_project": self.user_project_name,
            },
        }
        res = self.__api_client.post(UPDATE_PROJECT_URI, payload)
        return res.get("details")

    def add_user_to_project(self, email: str, role: str) -> str:
        """Adds new user to project

        :param email: user email
        :param role: user role ["admin", "user"]
        :return: response
        """
        payload = {
            "project_name": self.project_name,
            "modify_req": {
                "add_user_project": {
                    "email": email,
                    "role": role,
                },
            },
        }
        res = self.__api_client.post(UPDATE_PROJECT_URI, payload)
        return res.get("details")

    def remove_user_from_project(self, email: str) -> str:
        """Removes user from project

        :param email: _description_
        :return: _description_
        """
        payload = {
            "project_name": self.project_name,
            "modify_req": {"remove_user_project": email},
        }
        res = self.__api_client.post(UPDATE_PROJECT_URI, payload)
        return res.get("details")

    def update_user_access_for_project(self, email: str, role: str) -> str:
        """Updates user access for project

        :param email: user email
        :param role: user role
        :return: response
        """
        payload = {
            "project_name": self.project_name,
            "modify_req": {
                "update_user_project": {
                    "email": email,
                    "role": role,
                },
            },
        }
        res = self.__api_client.post(UPDATE_PROJECT_URI, payload)
        return res.get("details")

    def config(self) -> str:
        """returns config for the project

        :return: response
        """
        res = self.__api_client.get(
            f"{GET_PROJECT_CONFIG}?project_name={self.project_name}"
        )

        return res.get("details")

    def available_tags(self) -> str:
        """get available tags for the project

        :return: response
        """
        res = self.__api_client.get(
            f"{AVAILABLE_TAGS_URI}?project_name={self.project_name}"
        )

        if not res["success"]:
            error_details = res.get("details", "Failed to get available tags.")
            raise Exception(error_details)

        return res["details"]

    def get_labels(self, feature_name: str) -> List[str]:
        """get unique value of feature name

        :param feature_name: feature name
        :return: unique values of feature
        """
        res = self.__api_client.get(
            f"{GET_LABELS_URI}?project_name={self.project_name}&feature_name={feature_name}"
        )

        if not res["success"]:
            error_details = res.get(
                "details", f"Failed to get labels for {feature_name}"
            )
            raise Exception(error_details)

        return res["labels"]

    def files(self) -> pd.DataFrame:
        """Lists all files uploaded by user

        :return: user uploaded files dataframe
        """
        files = self.__api_client.get(
            f"{ALL_DATA_FILE_URI}?project_name={self.project_name}"
        )

        if not files.get("details"):
            raise Exception("Please upload files first")

        files_df = (
            pd.DataFrame(files["details"])
            .drop(["metadata", "project_name", "version"], axis=1)
            .rename(columns={"filepath": "file_name"})
        )

        files_df["file_name"] = files_df["file_name"].apply(
            lambda file_path: file_path.split("/")[-1]
        )

        return files_df

    def file_summary(self, file_name: str) -> pd.DataFrame:
        """File Summary

        :param file_name: user uploaded file name
        :return: file summary dataframe
        """
        files = self.__api_client.get(
            f"{ALL_DATA_FILE_URI}?project_name={self.project_name}"
        )

        if not files.get("details"):
            raise Exception("Please upload files first")

        file_data = next(
            filter(
                lambda file: file["filepath"].split("/")[-1] == file_name,
                files["details"],
            ),
            None,
        )

        if not file_data:
            raise Exception("File Not Found, please pass valid file name")

        file_metadata = {
            "file_size_mb": file_data["metadata"]["file_size_mb"],
            "columns": file_data["metadata"]["columns"],
            "rows": file_data["metadata"]["rows"],
        }

        print(file_metadata)

        file_summary_df = pd.DataFrame(file_data["metadata"]["details"])

        return file_summary_df

    def delete_file(self, file_name: str) -> str:
        """deletes file for the project

        :param file_name: uploaded file name
        :return: response
        """
        files = self.__api_client.get(
            f"{ALL_DATA_FILE_URI}?project_name={self.project_name}"
        )

        if not files.get("details"):
            raise Exception("Please upload files first")

        file_data = next(
            filter(
                lambda file: file["filepath"] == file_name
                or file["filepath"].split("/")[-1] == file_name,
                files["details"],
            ),
            None,
        )

        if not file_data:
            raise Exception("File Not Found, please pass valid file name")

        payload = {
            "project_name": self.project_name,
            "workspace_name": self.workspace_name,
            "path": file_data["filepath"],
        }

        res = self.__api_client.post(DELETE_DATA_FILE_URI, payload)
        return res.get("details")

    def upload_file(
        self, file_path: str, tag: str, config: Optional[ProjectConfig] = None
    ) -> str:
        """Uploads file for the current project
        :param file_path: file path to be uploaded
        :param config: project config
                {
                    "project_type": "",
                    "unique_identifier": "",
                    "true_label": "",
                    "pred_label": "",
                    "feature_exclude": [],
                },
                defaults to None
        :return: response
        """

        def upload_file_and_return_path() -> str:
            res = self.__api_client.file(
                f"{UPLOAD_DATA_FILE_URI}?project_name={self.project_name}&data_type=data&tag={tag}",
                file_path,
            )

            if not res["success"]:
                raise Exception(res.get("details"))

            uploaded_path = res.get("metadata").get("filepath")

            return uploaded_path

        project_config = self.config()

        if project_config == "Not Found":
            if not config:
                config = {
                    "project_type": "",
                    "unique_identifier": "",
                    "true_label": "",
                    "pred_label": "",
                    "feature_exclude": [],
                }
                raise Exception(
                    f"Project Config is required, since no config is set for project \n {json.dumps(config)}"
                )

            Validate.check_for_missing_keys(
                config, ["project_type", "unique_identifier", "true_label"]
            )

            valid_project_type = ["classification", "regression"]
            if not config["project_type"] in valid_project_type:
                raise Exception(
                    f"{config['project_type']} is not a valid project_type, select from {valid_project_type}"
                )

            uploaded_path = upload_file_and_return_path()

            file_info = self.__api_client.post(
                UPLOAD_DATA_FILE_INFO_URI, {"path": uploaded_path}
            )

            column_names = file_info.get("details").get("column_names")

            if not config["unique_identifier"] in column_names:
                self.delete_file(uploaded_path)
                raise Exception(
                    f"{config['unique_identifier']} is not a valid unique_identifier, select from {column_names}"
                )

            if config.get("feature_exclude"):
                if not all(
                    feature in column_names for feature in config["feature_exclude"]
                ):
                    self.delete_file(uploaded_path)
                    raise Exception(
                        f"feature_exclude is not valid, select valid values from {column_names}"
                    )

            feature_exclude = [
                config["unique_identifier"],
                config["true_label"],
                *config.get("feature_exclude", []),
            ]

            feature_include = [
                feature for feature in column_names if feature not in feature_exclude
            ]

            payload = {
                "project_name": self.project_name,
                "project_type": config["project_type"],
                "unique_identifier": config["unique_identifier"],
                "true_label": config["true_label"],
                "pred_label": config.get("pred_label"),
                "metadata": {
                    "path": uploaded_path,
                    "tag": tag,
                    "tags": [tag],
                    "drop_duplicate_uid": False,
                    "feature_exclude": feature_exclude,
                    "feature_include": feature_include,
                    "feature_encodings": {},
                    "feature_actual_used": [],
                },
            }

            res = self.__api_client.post(UPLOAD_DATA_WITH_CHECK_URI, payload)

            if not res["success"]:
                self.delete_file(uploaded_path)
                raise Exception(res.get("details"))

            return res.get("details")

        uploaded_path = upload_file_and_return_path()

        payload = {
            "path": uploaded_path,
            "tag": tag,
            "type": "data",
            "project_name": self.project_name,
        }
        res = self.__api_client.post(UPLOAD_DATA_URI, payload)

        if not res["success"]:
            self.delete_file(uploaded_path)
            raise Exception(res.get("details"))

        return res.get("details")

    def data_summary(self, tag: str) -> pd.DataFrame:
        """Data Summary for the project

        :param tag: tag for data ["Training", "Testing", "Validation", "Custom"]
        :return: data summary dataframe
        """
        res = self.__api_client.post(
            f"{GET_DATA_SUMMARY_URI}?project_name={self.project_name}&refresh=true"
        )
        valid_tags = res["data"]["data"].keys()

        if not valid_tags:
            raise Exception("Data summary not available, please upload data first.")

        if tag not in valid_tags:
            raise Exception(f"Not a vaild tag. Pick a valid tag from {valid_tags}")

        data = {
            "Total Data Volume": res["data"]["overview"]["Total Data Volumn"],
            "Unique Features": res["data"]["overview"]["Unique Features"],
        }

        print(data)
        summary = pd.DataFrame(res["data"]["data"][tag]).drop(
            ["data_description"], axis=1
        )
        return summary

    def data_diagnosis(self, tag: str) -> pd.DataFrame:
        """Data Diagnosis for the project

        :param tag: tag for data ["Training", "Testing", "Validation", "Custom"]
        :return: data diagnosis dataframe
        """
        res = self.__api_client.get(
            f"{GET_DATA_DIAGNOSIS_URI}?project_name={self.project_name}"
        )
        valid_tags = res["details"].keys()

        if not valid_tags:
            raise Exception("Data diagnosis not available, please upload data first.")

        if tag not in valid_tags:
            raise Exception(f"Not a vaild tag. Pick a valid tag from {valid_tags}")

        data_diagnosis = pd.DataFrame(res["details"][tag]["alerts"])
        data_diagnosis[["Tag", "Description"]] = data_diagnosis[0].str.extract(
            r"\['(.*?)'] (.+?) #"
        )
        data_diagnosis["Description"] = data_diagnosis["Description"].str.replace(
            r"[^\w\s]", "", regex=True
        )
        data_diagnosis = data_diagnosis[["Description", "Tag"]]

        data = {"Warnings": len(data_diagnosis)}
        print(data)

        return data_diagnosis

    def data_drift_diagnosis(
        self, baseline_tags: List[str], current_tags: List[str]
    ) -> pd.DataFrame:
        """Data Drift Diagnosis for the project

        :param tag: tag for data ["Training", "Testing", "Validation", "Custom"]
        :return: data drift diagnosis dataframe
        """
        payload = {
            "project_name": self.project_name,
            "baseline_tags": baseline_tags,
            "current_tags": current_tags,
        }
        res = self.__api_client.post(DATA_DRFIT_DIAGNOSIS_URI, payload)

        if not res["success"]:
            raise Exception(res.get("details").get("reason"))

        data_drift_diagnosis = pd.DataFrame(
            res["details"]["results"]["detailed_report"]
        ).drop(["current_small_hist", "ref_small_hist"], axis=1)

        return data_drift_diagnosis

    def get_default_dashboard_config(self, uri: str) -> dict:
        """get default config value of given dashboard

        :param uri: uri of the dashboard
        :return: dict of dashboard config
        """
        config = {"project_name": self.project_name}

        try:
            res = self.__api_client.post(uri, config)

            if not res["success"]:
                # take default config when not passed
                config = res["config"]
        except:
            pass

        return config

    def get_data_drift_dashboard(self, payload: DataDriftPayload = {}) -> Dashboard:
        """get data drift dashboard

        :param payload: data drift payload
            {
                "base_line_tag": "",
                "current_tag": "",
                "stat_test_name": "",
                "stat_test_threshold": "",
                "features_to_use": []
                "date_feature": "",
                "baseline_date": { "start_date": "", "end_date": ""},
                "current_date": { "start_date": "", "end_date": ""},
            }
            defaults to None
            key values for payload:
                stat_test_name=
                    chisquare (Chi-Square test):
                        default for categorical features if the number of labels for feature > 2
                        only for categorical features
                        returns p_value
                        default threshold 0.05
                        drift detected when p_value < threshold
                    jensenshannon (Jensen-Shannon distance):
                        for numerical and categorical features
                        returns distance
                        default threshold 0.05
                        drift detected when distance >= threshold
                    ks (Kolmogorov–Smirnov (K-S) test):
                        default for numerical features
                        only for numerical features
                        returns p_value
                        default threshold 0.05
                        drift detected when p_value < threshold
                    kl_div (Kullback-Leibler divergence):
                        for numerical and categorical features
                        returns divergence
                        default threshold 0.05
                        drift detected when divergence >= threshold,
                    psi (Population Stability Index):
                        for numerical and categorical features
                        returns psi_value
                        default_threshold=0.1
                        drift detected when psi_value >= threshold
                    wasserstein (Wasserstein distance (normed)):
                        only for numerical features
                        returns distance
                        default threshold 0.05
                        drift detected when distance >= threshold
                    z (Ztest):
                        default for categorical features if the number of labels for feature <= 2
                        only for categorical features
                        returns p_value
                        default threshold 0.05
                        drift detected when p_value < threshold
        :raises Exception: _description_
        :raises Exception: _description_
        :return: Dashboard
        """
        if not payload:
            payload = self.get_default_dashboard_config(DATA_DRIFT_DASHBOARD_URI)

        payload["project_name"] = self.project_name

        # validate required fields
        Validate.check_for_missing_keys(payload, DATA_DRIFT_DASHBOARD_REQUIRED_FIELDS)

        # validate tags and labels
        tags_info = self.available_tags()
        all_tags = tags_info["alltags"]

        Validate.validate_tags(payload["base_line_tag"], all_tags)
        Validate.validate_tags(payload["current_tag"], all_tags)

        Validate.validate_date_feature_val(payload, tags_info["alldatetimefeatures"])

        Validate.validate_features(
            payload.get("features_to_use", []), tags_info["alluniquefeatures"]
        )

        if payload["stat_test_name"] not in DATA_DRIFT_STAT_TESTS:
            raise Exception(
                f"{payload['stat_test_name']} is not a valid stat_test_name. Pick a valid value from {DATA_DRIFT_STAT_TESTS}."
            )

        res = self.__api_client.post(DATA_DRIFT_DASHBOARD_URI, payload)

        if not res["success"]:
            error_details = res.get("details", "Failed to get dashboard url")
            raise Exception(error_details)

        dashboard_url = res.get("hosted_path", None)
        auth_token = self.__api_client.get_auth_token()
        query_params = f"?id={auth_token}"

        return Dashboard(config=res["config"], url=f"{dashboard_url}{query_params}")

    def get_target_drift_dashboard(self, payload: TargetDriftPayload = {}) -> Dashboard:
        """get target drift dashboard

        :param payload: target drift payload
                {
                    "base_line_tag": "",
                    "current_tag": "",
                    "stat_test_name": "",
                    "stat_test_threshold": "",
                    "date_feature": "",
                    "baseline_date": { "start_date": "", "end_date": ""},
                    "current_date": { "start_date": "", "end_date": ""},
                    "model_type": "",
                    "baseline_true_label": "",
                    "current_true_label": ""
                }
                defaults to None
                key values for payload:
                    stat_test_name=
                        chisquare (Chi-Square test):
                        default for categorical features if the number of labels for feature > 2
                        only for categorical features
                        returns p_value
                        default threshold 0.05
                        drift detected when p_value < threshold
                    jensenshannon (Jensen-Shannon distance):
                        for numerical and categorical features
                        returns distance
                        default threshold 0.05
                        drift detected when distance >= threshold
                    kl_div (Kullback-Leibler divergence):
                        for numerical and categorical features
                        returns divergence
                        default threshold 0.05
                        drift detected when divergence >= threshold,
                    psi (Population Stability Index):
                        for numerical and categorical features
                        returns psi_value
                        default_threshold=0.1
                        drift detected when psi_value >= threshold
                    z (Ztest):
                        default for categorical features if the number of labels for feature <= 2
                        only for categorical features
                        returns p_value
                        default threshold 0.05
                        drift detected when p_value < threshold
        :raises Exception: _description_
        :raises Exception: _description_
        :raises Exception: _description_
        :return: Dashboard
        """
        if not payload:
            payload = self.get_default_dashboard_config(TARGET_DRIFT_DASHBOARD_URI)

        payload["project_name"] = self.project_name

        # validate required fields
        Validate.check_for_missing_keys(payload, TARGET_DRIFT_DASHBOARD_REQUIRED_FIELDS)

        # validate tags and labels
        tags_info = self.available_tags()
        all_tags = tags_info["alltags"]

        Validate.validate_tags(payload["base_line_tag"], all_tags)
        Validate.validate_tags(payload["current_tag"], all_tags)

        Validate.validate_date_feature_val(payload, tags_info["alldatetimefeatures"])

        if payload["model_type"] not in TARGET_DRIFT_MODEL_TYPES:
            raise Exception(
                f"{payload['model_type']} is not a valid model_type. Pick a valid value from {TARGET_DRIFT_MODEL_TYPES}."
            )

        if payload["stat_test_name"] not in TARGET_DRIFT_STAT_TESTS:
            raise Exception(
                f"{payload['stat_test_name']} is not a valid stat_test_name. Pick a valid value from {DATA_DRIFT_STAT_TESTS}."
            )

        Validate.validate_features(
            [payload["baseline_true_label"]], tags_info["alluniquefeatures"]
        )

        Validate.validate_features(
            [payload["current_true_label"]], tags_info["alluniquefeatures"]
        )

        res = self.__api_client.post(TARGET_DRIFT_DASHBOARD_URI, payload)

        if not res["success"]:
            error_details = res.get("details", "Failed to get dashboard url")
            raise Exception(error_details)

        dashboard_url = res.get("hosted_path", None)
        auth_token = self.__api_client.get_auth_token()
        query_params = f"?id={auth_token}"

        return Dashboard(config=res["config"], url=f"{dashboard_url}{query_params}")

    def get_bias_monitoring_dashboard(
        self, payload: BiasMonitoringPayload = {}
    ) -> Dashboard:
        """get bias monitoring dashboard

        :param payload: bias monitoring payload
                {
                    "base_line_tag": "",
                    "date_feature": "",
                    "baseline_date": { "start_date": "", "end_date": ""},
                    "current_date": { "start_date": "", "end_date": ""},
                    "model_type": "",
                    "baseline_true_label": "",
                    "baseline_pred_label": "",
                    features_to_use: []
                }
                defaults to None
        :raises Exception: _description_
        :raises Exception: _description_
        :return: Dashboard
        """
        if not payload:
            payload = self.get_default_dashboard_config(BIAS_MONITORING_DASHBOARD_URI)

        payload["project_name"] = self.project_name

        # validate required fields
        Validate.check_for_missing_keys(
            payload, BIAS_MONITORING_DASHBOARD_REQUIRED_FIELDS
        )

        # validate tags and labels
        tags_info = self.available_tags()
        all_tags = tags_info["alltags"]

        Validate.validate_tags(payload["base_line_tag"], all_tags)

        Validate.validate_date_feature_val(payload, tags_info["alldatetimefeatures"])

        if payload["model_type"] not in MODEL_TYPES:
            raise Exception(
                f"{payload['model_type']} is not a valid model_type. Pick a valid value from {TARGET_DRIFT_MODEL_TYPES}."
            )

        Validate.validate_features(
            [payload["baseline_true_label"]], tags_info["alluniquefeatures"]
        )

        Validate.validate_features(
            [payload["baseline_pred_label"]], tags_info["alluniquefeatures"]
        )

        Validate.validate_features(
            payload.get("features_to_use", []), tags_info["alluniquefeatures"]
        )

        res = self.__api_client.post(BIAS_MONITORING_DASHBOARD_URI, payload)

        if not res["success"]:
            error_details = res.get("details", "Failed to get dashboard url")
            raise Exception(error_details)

        dashboard_url = res.get("hosted_path", None)
        auth_token = self.__api_client.get_auth_token()
        query_params = f"?id={auth_token}"

        return Dashboard(config=res["config"], url=f"{dashboard_url}{query_params}")

    def get_model_performance_dashboard(
        self, payload: ModelPerformancePayload = {}
    ) -> Dashboard:
        """get model performance dashboard

        :param payload: model performance payload
                {
                    "base_line_tag": "",
                    "current_tag": "",
                    "date_feature": "",
                    "baseline_date": { "start_date": "", "end_date": ""},
                    "current_date": { "start_date": "", "end_date": ""},
                    "model_type": "",
                    "baseline_true_label": "",
                    "baseline_pred_label": "",
                    "current_true_label": "",
                    "current_pred_label": ""
                }
                defaults to None
        :raises Exception: _description_
        :raises Exception: _description_
        :return: Dashboard
        """
        if not payload:
            payload = self.get_default_dashboard_config(MODEL_PERFORMANCE_DASHBOARD_URI)

        payload["project_name"] = self.project_name

        # validate required fields
        Validate.check_for_missing_keys(payload, MODEL_PERF_DASHBOARD_REQUIRED_FIELDS)

        # validate tags and labels
        tags_info = self.available_tags()
        all_tags = tags_info["alltags"]

        Validate.validate_tags(payload["base_line_tag"], all_tags)
        Validate.validate_tags(payload["current_tag"], all_tags)

        Validate.validate_date_feature_val(payload, tags_info["alldatetimefeatures"])

        if payload["model_type"] not in MODEL_TYPES:
            raise Exception(
                f"{payload['model_type']} is not a valid model_type. Pick a valid value from {TARGET_DRIFT_MODEL_TYPES}."
            )

        Validate.validate_features(
            [payload["baseline_true_label"]], tags_info["alluniquefeatures"]
        )

        Validate.validate_features(
            [payload["baseline_pred_label"]], tags_info["alluniquefeatures"]
        )

        Validate.validate_features(
            [payload["current_true_label"]], tags_info["alluniquefeatures"]
        )

        Validate.validate_features(
            [payload["current_pred_label"]], tags_info["alluniquefeatures"]
        )

        res = self.__api_client.post(MODEL_PERFORMANCE_DASHBOARD_URI, payload)

        if not res["success"]:
            error_details = res.get("details", "Failed to get dashboard url")
            raise Exception(error_details)

        dashboard_url = res.get("hosted_path", None)
        auth_token = self.__api_client.get_auth_token()
        query_params = f"?id={auth_token}"

        return Dashboard(config=res["config"], url=f"{dashboard_url}{query_params}")

    def monitoring_triggers(self) -> dict:
        """get monitoring triggers of project

        :return: DataFrame
        """
        url = f"{GET_TRIGGERS_URI}?project_name={self.project_name}"
        res = self.__api_client.get(url)

        if not res["success"]:
            return Exception(res.get("details", "Failed to get triggers"))

        monitoring_triggers = res.get("details", [])

        if not monitoring_triggers:
            return "No monitoring triggers found."

        monitoring_triggers = pd.DataFrame(monitoring_triggers)
        monitoring_triggers = monitoring_triggers.drop("project_name", axis=1)

        return monitoring_triggers

    def create_monitoring_trigger(self, type: str, payload: dict) -> str:
        """create monitoring trigger for project

        :param type: trigger type ["Data Drift", "Target Drift", "Model Performance"]
        :param payload: Data Drift Trigger Payload
                {
                    "trigger_name": "",
                    "mail_list": [],
                    "frequency": "",
                    "stat_test_name": "",
                    "stat_test_threshold": 0,
                    "datadrift_features_per": 7,
                    "features_to_use": [],
                    "date_feature": "",
                    "baseline_date": { "start_date": "", "end_date": ""},
                    "current_date": { "start_date": "", "end_date": ""},
                    "base_line_tag": "",
                    "current_tag": ""
                } OR Target Drift Trigger Payload
                {
                    "trigger_name": "",
                    "mail_list": [],
                    "frequency": "",
                    "model_type": "",
                    "stat_test_name": ""
                    "stat_test_threshold": 0,
                    "date_feature": "",
                    "baseline_date": { "start_date": "", "end_date": ""},
                    "current_date": { "start_date": "", "end_date": ""},
                    "base_line_tag": "",
                    "current_tag": "",
                    "baseline_true_label": "",
                    "current_true_label": ""
                } OR Model Performance Trigger Payload
                {
                    "trigger_name": "",
                    "mail_list": [],
                    "frequency": "",
                    "model_type": "",
                    "model_performance_metric": "",
                    "model_performance_threshold": "",
                    "date_feature": "",
                    "baseline_date": { "start_date": "", "end_date": ""},
                    "current_date": { "start_date": "", "end_date": ""},
                    "base_line_tag": "",
                    "baseline_true_label": "",
                    "baseline_pred_label": ""
                }
                defaults to None
        :raises Exception: _description_
        :raises Exception: _description_
        :raises Exception: _description_
        :raises Exception: _description_
        :return: _description_
        """
        payload["project_name"] = self.project_name
        payload["trigger_type"] = type

        # validate tags and labels
        tags_info = self.available_tags()
        all_tags = tags_info["alltags"]

        if type == "Data Drift":
            Validate.check_for_missing_keys(payload, DATA_DRIFT_TRIGGER_REQUIRED_FIELDS)

            Validate.validate_tags(payload["base_line_tag"], all_tags)
            Validate.validate_tags(payload["current_tag"], all_tags)

            if payload["stat_test_name"] not in DATA_DRIFT_STAT_TESTS:
                raise Exception(
                    f"{payload['stat_test_name']} is not a valid stat_test_name. Pick a valid value from {DATA_DRIFT_STAT_TESTS}."
                )

            Validate.validate_features(
                payload.get("features_to_use", []), tags_info["alluniquefeatures"]
            )
        elif type == "Target Drift":
            Validate.check_for_missing_keys(
                payload, TARGET_DRIFT_TRIGGER_REQUIRED_FIELDS
            )

            Validate.validate_tags(payload["base_line_tag"], all_tags)
            Validate.validate_tags(payload["current_tag"], all_tags)

            if payload["model_type"] not in MODEL_TYPES:
                raise Exception(
                    f"{payload['model_type']} is not a valid model_type. Pick a valid type from {MODEL_TYPES}"
                )

            if (
                payload["model_type"] == "classification"
                and payload["stat_test_name"]
                not in TARGET_DRIFT_STAT_TESTS_CLASSIFICATION
            ):
                raise Exception(
                    f"{payload['stat_test_name']} is not a valid stat_test_name. Pick a valid value from {TARGET_DRIFT_STAT_TESTS_CLASSIFICATION}."
                )

            if (
                payload["model_type"] == "regression"
                and payload["stat_test_name"] not in TARGET_DRIFT_STAT_TESTS_REGRESSION
            ):
                raise Exception(
                    f"{payload['stat_test_name']} is not a valid stat_test_name. Pick a valid value from {TARGET_DRIFT_STAT_TESTS_REGRESSION}."
                )

            Validate.validate_features(
                [payload["baseline_true_label"]], tags_info["alluniquefeatures"]
            )

            Validate.validate_features(
                [payload["current_true_label"]], tags_info["alluniquefeatures"]
            )
        elif type == "Model Performance":
            Validate.check_for_missing_keys(payload, MODEL_PERF_TRIGGER_REQUIRED_FIELDS)

            Validate.validate_tags(payload["base_line_tag"], all_tags)

            if payload["model_type"] not in MODEL_TYPES:
                raise Exception(
                    f"{payload['model_type']} is not a valid model type. Pick a valid type from {MODEL_TYPES}"
                )

            Validate.validate_features(
                [payload["baseline_true_label"]], tags_info["alluniquefeatures"]
            )

            Validate.validate_features(
                [payload["baseline_pred_label"]], tags_info["alluniquefeatures"]
            )

            if payload["model_type"] == "classification":
                if not payload["class_label"]:
                    raise Exception(
                        "class_label is required for classification model type."
                    )

                all_class_label = self.get_labels(payload["baseline_true_label"])

                if payload["class_label"] not in all_class_label:
                    raise Exception(
                        f"{payload['class_label']} is not a valid class_label. Pick a valid value from {all_class_label}."
                    )

                if (
                    payload["model_performance_metric"]
                    not in MODEL_PERF_METRICS_CLASSIFICATION
                ):
                    raise Exception(
                        f"{payload['model_performance_metric']} is not a valid model_performance_metric. Pick a valid value from {MODEL_PERF_METRICS_CLASSIFICATION}."
                    )

            if (
                payload["model_type"] == "regression"
                and payload["model_performance_metric"]
                not in MODEL_PERF_METRICS_REGRESSION
            ):
                raise Exception(
                    f"{payload['model_performance_metric']} is not a valid model_performance_metric. Pick a valid value from {MODEL_PERF_METRICS_REGRESSION}."
                )
        else:
            raise Exception(
                'Invalid trigger type. Please use one of ["Data Drift", "Target Drift", "Model Performance"]'
            )

        if payload["frequency"] not in MAIL_FREQUENCIES:
            raise Exception(
                f"Invalid frequency value. Please use one of {MAIL_FREQUENCIES}"
            )

        Validate.validate_date_feature_val(payload, tags_info["alldatetimefeatures"])

        payload = {
            "project_name": self.project_name,
            "modify_req": {
                "create_trigger": payload,
            },
        }
        res = self.__api_client.post(CREATE_TRIGGER_URI, payload)

        if not res["success"]:
            return Exception(res.get("details", "Failed to create trigger"))

        return "Trigger created successfully."

    def delete_monitoring_trigger(self, name: str) -> str:
        """delete monitoring trigger for project

        :param name: trigger name
        :return: str
        """
        payload = {
            "project_name": self.project_name,
            "modify_req": {
                "delete_trigger": name,
            },
        }

        res = self.__api_client.post(DELETE_TRIGGER_URI, payload)

        if not res["success"]:
            return Exception(res.get("details", "Failed to delete trigger"))

        return "Monitoring trigger deleted successfully."

    def alerts(self, page_num: int = 1) -> dict:
        """get monitoring alerts of project

        :param page_num: page num, defaults to 1
        :return: DataFrame
        """
        payload = {"page_num": page_num, "project_name": self.project_name}

        res = self.__api_client.post(EXECUTED_TRIGGER_URI, payload)

        if not res["success"]:
            return Exception(res.get("details", "Failed to get alerts"))

        monitoring_alerts = res.get("details", [])

        if not monitoring_alerts:
            return "No monitoring alerts found."

        return pd.DataFrame(monitoring_alerts)
    
    def get_alert_details(self, id: str) -> dict:
        """get alert details by id

        :param id: alert or trigger id
        :return: DataFrame
        """
        payload = {
            "project_name": self.project_name,
            "id": id,
        }
        res = self.__api_client.post(GET_EXECUTED_TRIGGER_INFO, payload)

        if not res["success"]:
            return Exception(res.get("details", "Failed to get trigger details"))

        trigger_info =  res['details'][0]        

        if not trigger_info['successful']:
            return Alert(
                info={},
                detailed_report=[],
                not_used_features=[]
            )

        trigger_info = trigger_info['details']
        
        detailed_report = trigger_info['detailed_report']
        not_used_features = trigger_info['Not_Used_Features']
        
        del trigger_info['detailed_report']
        del trigger_info['Not_Used_Features']
        
        return Alert(
            info=trigger_info,
            detailed_report=detailed_report,
            not_used_features=not_used_features
        )

    def get_model_performance(self, model_name: str = None) -> Dashboard:
        """
        get model performance dashboard
        """
        url = self.__api_client.get_url(GET_MODEL_PERFORMANCE_URI)

        # append params
        auth_token = self.__api_client.get_auth_token()
        url = f"{url}/{self.project_name}?id={auth_token}"

        if model_name:
            url = f"{url}&model_name={model_name}"

        return Dashboard(config={}, url=url)

    def train_model(
        self,
        model_type: str,
        data_config: Optional[DataConfig] = None,
        model_config: Optional[dict] = None,
    ) -> str:
        """Train new model

        :param model_type: type of model
        :param data_config: config for the data, defaults to None
        :param model_config: config with hyper parameters for the model, defaults to None
        :return: response
        """
        project_config = self.config()

        if project_config == "Not Found":
            raise Exception("Upload files first")

        available_models = self.available_models()

        if model_type not in available_models:
            raise Exception(
                f"{model_type} is not a valid model_type, select from \n{available_models}"
            )

        if data_config:
            if data_config.get("feature_exclude"):
                if not all(
                    feature in project_config["metadata"]["feature_include"]
                    for feature in data_config["feature_exclude"]
                ):
                    raise Exception(
                        f"feature_exclude is not valid,\nalready excluded features : {project_config['metadata']['feature_exclude']} \nselect valid values from : {project_config['metadata']['feature_include'] }"
                    )

            if data_config.get("tags"):
                available_tags = self.tags()
                if not all(tag in available_tags for tag in data_config["tags"]):
                    raise Exception(
                        f"tags is not valid,select valid values from :\n{available_tags}"
                    )

        if model_config:
            model_params = self.__api_client.get(MODEL_PARAMETERS_URI)
            model_name = f"{model_type}_{project_config['project_type']}".lower()
            model_parameters = model_params.get(model_name)

            if model_parameters:
                for model_config_param in model_config.keys():
                    model_param = model_parameters.get(model_config_param)
                    model_config_param_value = model_config[model_config_param]

                    if not model_param:
                        raise Exception(
                            f"Invalid model config for {model_type} \n {json.dumps(model_parameters)}"
                        )

                    if model_param["type"] == "select":
                        if model_config_param_value not in model_param["value"]:
                            raise Exception(
                                f"Invalid value for {model_config_param}, select from {model_param['value']}"
                            )
                    elif model_param["type"] == "input":
                        if model_config_param_value > model_param["max"]:
                            raise Exception(
                                f"{model_config_param} value cannot be greater than {model_param['max']}"
                            )
                        if model_config_param_value < model_param["min"]:
                            raise Exception(
                                f"{model_config_param} value cannot be less than {model_param['min']}"
                            )

        data_conf = data_config or {}

        feature_exclude = [
            *project_config["metadata"]["feature_exclude"],
            *data_conf.get("feature_exclude", []),
        ]

        feature_include = [
            feature
            for feature in project_config["metadata"]["feature_include"]
            if feature not in feature_exclude
        ]

        feature_encodings = (
            data_conf.get("feature_encodings")
            or project_config["metadata"]["feature_encodings"]
        )

        drop_duplicate_uid = (
            data_conf.get("drop_duplicate_uid")
            or project_config["metadata"]["drop_duplicate_uid"]
        )

        tags = data_conf.get("tags") or project_config["metadata"]["tags"]

        payload = {
            "project_name": self.project_name,
            "project_type": project_config["project_type"],
            "unique_identifier": project_config["unique_identifier"],
            "true_label": project_config["true_label"],
            "metadata": {
                "model_name": model_type,
                "model_parameters": model_config,
                "feature_include": feature_include,
                "feature_exclude": feature_exclude,
                "feature_encodings": feature_encodings,
                "drop_duplicate_uid": drop_duplicate_uid,
                "tags": tags,
            },
        }

        res = self.__api_client.post(TRAIN_MODEL_URI, payload)

        if not res["success"]:
            raise Exception(res["details"])

        return "Model Trained Successfully"

    def models(self) -> pd.DataFrame:
        """Models trained for the project

        :return: Dataframe with details of all models
        """
        res = self.__api_client.get(
            f"{GET_MODELS_URI}?project_name={self.project_name}"
        )

        if not res["success"]:
            raise Exception(res["details"])

        staged_models = res["details"]["staged"]

        staged_models_df = pd.DataFrame(staged_models)

        return staged_models_df

    def available_models(self) -> List[str]:
        """Returns all models which can be trained on platform

        :return: list of all models
        """
        res = self.__api_client.get(
            f"{GET_MODELS_URI}?project_name={self.project_name}"
        )

        if not res["success"]:
            raise Exception(res["details"])

        available_models = list(
            map(lambda data: data["model_name"], res["details"]["available"])
        )

        return available_models

    def activate_model(self, model_name: str):
        """Sets the model to active for the project

        :param model_name: name of the model
        :return: response
        """
        payload = {
            "project_name": self.project_name,
            "model_name": model_name,
        }
        res = self.__api_client.post(UPDATE_ACTIVE_MODEL_URI, payload)

        if not res["success"]:
            raise Exception(res["details"])

        return res.get("details")

    def remove_model(self, model_name: str):
        """Removes the trained model for the project

        :param model_name: name of the model
        :return: response
        """
        payload = {
            "project_name": self.project_name,
            "model_name": model_name,
        }
        res = self.__api_client.post(REMOVE_MODEL_URI, payload)

        if not res["success"]:
            raise Exception(res["details"])

        return res.get("details")

    def model_inference(self, tag: str, model_name: Optional[str] = None):
        """Run model inference on data

        :param tag: data tag for running inference
        :param model_name: name of the model, defaults to active model for the project
        :return: model inference dataframe
        """
        available_tags = self.tags()
        if tag not in available_tags:
            raise Exception(
                f"{tag} tag is not valid, select valid tag from :\n{available_tags}"
            )

        models = self.models()

        available_models = models["model_name"].to_list()

        if model_name and model_name not in available_models:
            raise Exception(
                f"{model_name} model is not valid,select valid model from :\n{available_models}"
            )

        model = (
            model_name
            or models.loc[models["status"] == "active"]["model_name"].values[0]
        )

        run_model_payload = {
            "project_name": self.project_name,
            "model_name": model,
            "tags": tag,
        }

        run_model_res = self.__api_client.post(RUN_MODEL_ON_DATA_URI, run_model_payload)

        if not run_model_res["success"]:
            raise Exception(run_model_res["details"])

        download_tag_payload = {
            "project_name": self.project_name,
            "tag": f"{tag}_{model}_Inference",
        }

        tag_data = self.__api_client.request(
            "POST", DOWNLOAD_TAG_DATA_URI, download_tag_payload
        )

        tag_data_df = pd.read_csv(io.StringIO(tag_data.text))

        return tag_data_df

    def model_summary(self, model_name: Optional[str] = None) -> dict:
        """Model Summary

        :param model_name: name of the model, defaults to active model for project
        :return: model summary
        """
        res = self.__api_client.get(
            f"{MODEL_SUMMARY_URI}?project_name={self.project_name}"
            + (f"&model_name={model_name}" if model_name else "")
        )

        if not res["success"]:
            raise Exception(res["details"])

        return ModelSummary(api_client=self.__api_client, **res.get("details"))

    def tags(self) -> List[str]:
        """Available User Tags for Project

        :return: list of tags
        """
        available_tags = self.available_tags()

        tags = available_tags.get("user_tags")

        return tags

    def all_tags(self) -> List[str]:
        """Available All Tags for Project

        :return: list of tags
        """
        available_tags = self.available_tags()

        tags = available_tags.get("alltags")

        return tags

    def cases(
        self,
        unique_identifier: Optional[str] = None,
        tag: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        """Cases for the Project

        :param unique_identifier: unique identifer of the case for filtering, defaults to None
        :param tag: data tag for filtering, defaults to None
        :param start_date: start date for filtering, defaults to None
        :param end_date: end data for filtering, defaults to None
        """

        def get_cases():
            payload = {
                "project_name": self.project_name,
                "page_num": 1,
            }
            res = self.__api_client.post(GET_CASES_URI, payload)
            return res

        def search_cases():
            payload = {
                "project_name": self.project_name,
                "unique_identifier": unique_identifier,
                "start_date": start_date,
                "end_date": end_date,
                "tag": tag,
                "page_num": 1,
            }
            res = self.__api_client.post(SEARCH_CASE_URI, payload)
            return res

        cases = (
            search_cases()
            if unique_identifier or tag or start_date or end_date
            else get_cases()
        )

        if not cases["success"]:
            raise Exception("No cases found")

        cases_df = pd.DataFrame(cases.get("details"))

        return cases_df

    def case_info(
        self,
        unique_identifer: str,
        case_id: Optional[str] = None,
        tag: Optional[str] = None,
    ):
        """Case Info

        :param unique_identifer: unique identifer of case
        :param case_id: case id, defaults to None
        :param tag: case tag, defaults to None
        :return: Case object with details
        """
        payload = {
            "project_name": self.project_name,
            "case_id": case_id,
            "unique_identifier": unique_identifer,
            "tag": tag,
        }
        res = self.__api_client.post(CASE_INFO_URI, payload)

        if not res["success"]:
            raise Exception(res["details"])

        case = Case(**res["details"])

        return case

    def delete_cases(
        self,
        unique_identifier: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        tag: Optional[str] = None,
    ):
        if tag:
            all_tags = self.all_tags()
            if tag not in all_tags:
                raise Exception(
                    f"Invalid {tag} tag, select valid tag from \n{all_tags}"
                )

        paylod = {
            "project_name": self.project_name,
            "unique_identifier": [unique_identifier],
            "start_date": start_date,
            "end_date": end_date,
            "tag": tag,
        }

        res = self.__api_client.post(DELETE_CASE_URI, paylod)

        if not res["success"]:
            raise Exception(res["details"])

        return res["details"]

    def __print__(self) -> str:
        return f"Project(user_project_name='{self.user_project_name}', created_by='{self.created_by}')"

    def __str__(self) -> str:
        return self.__print__()

    def __repr__(self) -> str:
        return self.__print__()
