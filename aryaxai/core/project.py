from __future__ import annotations
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
    SYNTHETIC_MODELS_DEFAULT_HYPER_PARAMS,
    TARGET_DRIFT_DASHBOARD_REQUIRED_FIELDS,
    TARGET_DRIFT_MODEL_TYPES,
    TARGET_DRIFT_STAT_TESTS,
    BIAS_MONITORING_DASHBOARD_REQUIRED_FIELDS,
    MODEL_PERF_DASHBOARD_REQUIRED_FIELDS,
    TARGET_DRIFT_STAT_TESTS_CLASSIFICATION,
    TARGET_DRIFT_STAT_TESTS_REGRESSION,
    TARGET_DRIFT_TRIGGER_REQUIRED_FIELDS,
)
from aryaxai.common.types import DataConfig, ProjectConfig, SyntheticDataConfig, SyntheticModelHyperParams
from aryaxai.common.utils import parse_datetime, parse_float, poll_events
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
    CLEAR_NOTIFICATIONS_URI,
    CREATE_OBSERVATION_URI,
    CREATE_POLICY_URI,
    CREATE_SYNTHETIC_PROMPT_URI,
    CREATE_TRIGGER_URI,
    DATA_DRFIT_DIAGNOSIS_URI,
    DELETE_CASE_URI,
    DELETE_DATA_FILE_URI,
    DELETE_SYNTHETIC_MODEL_URI,
    DELETE_SYNTHETIC_TAG_URI,
    DOWNLOAD_SYNTHETIC_DATA_URI,
    DOWNLOAD_TAG_DATA_URI,
    DELETE_TRIGGER_URI,
    EXECUTED_TRIGGER_URI,
    GET_CASES_URI,
    GET_DATA_DIAGNOSIS_URI,
    GET_DATA_SUMMARY_URI,
    GET_EXECUTED_TRIGGER_INFO,
    GET_LABELS_URI,
    GET_MODEL_PERFORMANCE_URI,
    GET_MODEL_TYPES_URI,
    GET_MODELS_URI,
    GET_NOTIFICATIONS_URI,
    GET_OBSERVATION_PARAMS_URI,
    GET_OBSERVATIONS_URI,
    GET_POLICIES_URI,
    GET_POLICY_PARAMS_URI,
    GET_PROJECT_CONFIG,
    GET_SYNTHETIC_DATA_TAGS_URI,
    GET_SYNTHETIC_MODEL_DETAILS_URI,
    GET_SYNTHETIC_MODEL_PARAMS_URI,
    GET_SYNTHETIC_MODELS_URI,
    GET_SYNTHETIC_PROMPT_URI,
    MODEL_PARAMETERS_URI,
    MODEL_SUMMARY_URI,
    REMOVE_MODEL_URI,
    RUN_MODEL_ON_DATA_URI,
    SEARCH_CASE_URI,
    TRAIN_MODEL_URI,
    TRAIN_SYNTHETIC_MODEL_URI,
    UPDATE_ACTIVE_MODEL_URI,
    GET_TRIGGERS_URI,
    UPDATE_OBSERVATION_URI,
    UPDATE_POLICY_URI,
    UPDATE_PROJECT_URI,
    UPDATE_SYNTHETIC_PROMPT_URI,
    UPLOAD_DATA_FILE_INFO_URI,
    UPLOAD_DATA_FILE_URI,
    UPLOAD_DATA_URI,
    UPLOAD_DATA_WITH_CHECK_URI,
    DATA_DRIFT_DASHBOARD_URI,
    TARGET_DRIFT_DASHBOARD_URI,
    BIAS_MONITORING_DASHBOARD_URI,
    MODEL_PERFORMANCE_DASHBOARD_URI,
    UPLOAD_MODEL_URI,
)
import json
import io
from aryaxai.core.alert import Alert

from aryaxai.core.case import Case
from aryaxai.core.model_summary import ModelSummary

from aryaxai.core.dashboard import Dashboard
from datetime import datetime
import re

from aryaxai.core.synthetic import SyntheticDataTag, SyntheticModel, SyntheticPrompt


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

    def upload_data(
        self, data: str | pd.DataFrame, tag: str, config: Optional[ProjectConfig] = None
    ) -> str:
        """Uploads data for the current project
        :param file_path: file path | dataframe to be uploaded
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

        def build_upload_data():
            if isinstance(data, str):
                file = open(data, "rb")
                return file
            elif isinstance(data, pd.DataFrame):
                csv_buffer = io.BytesIO()
                data.to_csv(csv_buffer, index=False, encoding="utf-8")
                csv_buffer.seek(0)
                file_name = f"{tag}_sdk_{datetime.now().replace(microsecond=0)}.csv"
                file = (file_name, csv_buffer.getvalue())
                return file
            else:
                raise Exception("Invalid Data Type")

        def upload_file_and_return_path() -> str:
            files = {"in_file": build_upload_data()}
            res = self.__api_client.file(
                f"{UPLOAD_DATA_FILE_URI}?project_name={self.project_name}&data_type=data&tag={tag}",
                files,
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

            Validate.value_against_list(
                "project_type", config, ["classification", "regression"]
            )

            uploaded_path = upload_file_and_return_path()

            file_info = self.__api_client.post(
                UPLOAD_DATA_FILE_INFO_URI, {"path": uploaded_path}
            )

            column_names = file_info.get("details").get("column_names")

            Validate.value_against_list(
                "unique_identifier",
                config["unique_identifier"],
                column_names,
                lambda: self.delete_file(uploaded_path),
            )

            if config.get("feature_exclude"):
                Validate.value_against_list(
                    "feature_exclude",
                    config["feature_exclude"],
                    column_names,
                    lambda: self.delete_file(uploaded_path),
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

            poll_events(self.__api_client, self.project_name, res["event_id"])

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

    def upload_data_description(self, data: str | pd.DataFrame):
        """uploads data description for the project

        :param data: response
        """

        def build_upload_data():
            if isinstance(data, str):
                file = open(data, "rb")
                return file
            elif isinstance(data, pd.DataFrame):
                csv_buffer = io.BytesIO()
                data.to_csv(csv_buffer, index=False, encoding="utf-8")
                csv_buffer.seek(0)
                file_name = (
                    f"data_description_sdk_{datetime.now().replace(microsecond=0)}.csv"
                )
                file = (file_name, csv_buffer.getvalue())
                return file
            else:
                raise Exception("Invalid Data Type")

        def upload_file_and_return_path() -> str:
            files = {"in_file": build_upload_data()}
            res = self.__api_client.file(
                f"{UPLOAD_DATA_FILE_URI}?project_name={self.project_name}&data_type=data_description",
                files,
            )

            if not res["success"]:
                raise Exception(res.get("details"))
            uploaded_path = res.get("metadata").get("filepath")

            return uploaded_path

        uploaded_path = upload_file_and_return_path()

        payload = {
            "path": uploaded_path,
            "type": "data_description",
            "project_name": self.project_name,
        }
        res = self.__api_client.post(UPLOAD_DATA_URI, payload)

        if not res["success"]:
            self.delete_file(uploaded_path)
            raise Exception(res.get("details"))

        return res.get("details", "Data description upload successful")

    def upload_model(
        self,
        model_path: str,
        model_architecture: str,
        model_type: str,
        model_name: str,
        model_data_tags: List[str],
    ):
        def upload_file_and_return_path() -> str:
            files = {"in_file": open(model_path, "rb")}
            res = self.__api_client.file(
                f"{UPLOAD_DATA_FILE_URI}?project_name={self.project_name}&data_type=data_description",
                files,
            )

            if not res["success"]:
                raise Exception(res.get("details"))
            uploaded_path = res.get("metadata").get("filepath")

            return uploaded_path

        model_types = self.__api_client.get(GET_MODEL_TYPES_URI)
        valid_model_architecture = model_types.get("model_architecture").keys()
        Validate.value_against_list(
            "model_achitecture", model_architecture, valid_model_architecture
        )

        valid_model_types = model_types.get("model_architecture")[model_architecture]
        Validate.value_against_list("model_type", model_type, valid_model_types)

        uploaded_path = upload_file_and_return_path()

        payload = {
            "project_name": self.project_name,
            "model_name": model_name,
            "model_architecture": model_architecture,
            "model_type": model_type,
            "model_path": uploaded_path,
            "model_data_tags": model_data_tags,
        }

        res = self.__api_client.post(UPLOAD_MODEL_URI, payload)

        if not res.get("success"):
            raise Exception(res.get("details"))

        poll_events(
            self.__api_client,
            self.project_name,
            res["event_id"],
            lambda: self.delete_file(uploaded_path),
        )

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
        summary = pd.DataFrame(res["data"]["data"][tag])
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

        Validate.value_against_list("tag", tag, valid_tags)

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

        Validate.value_against_list("base_line_tag", payload["base_line_tag"], all_tags)
        Validate.value_against_list("current_tag", payload["current_tag"], all_tags)

        Validate.validate_date_feature_val(payload, tags_info["alldatetimefeatures"])

        if payload.get("features_to_use"):
            Validate.value_against_list(
                "features_to_use",
                payload.get("features_to_use", []),
                tags_info["alluniquefeatures"],
            )

        Validate.value_against_list(
            "stat_test_name", payload["stat_test_name"], DATA_DRIFT_STAT_TESTS
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

        Validate.value_against_list("base_line_tag", payload["base_line_tag"], all_tags)
        Validate.value_against_list("current_tag", payload["current_tag"], all_tags)

        Validate.validate_date_feature_val(payload, tags_info["alldatetimefeatures"])

        Validate.value_against_list(
            "model_type", payload["model_type"], TARGET_DRIFT_MODEL_TYPES
        )

        Validate.value_against_list(
            "stat_test_name", payload["stat_test_name"], TARGET_DRIFT_STAT_TESTS
        )

        Validate.value_against_list(
            "baseline_true_label",
            [payload["baseline_true_label"]],
            tags_info["alluniquefeatures"],
        )

        Validate.value_against_list(
            "current_true_label",
            [payload["current_true_label"]],
            tags_info["alluniquefeatures"],
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

        Validate.value_against_list("base_line_tag", payload["base_line_tag"], all_tags)

        Validate.validate_date_feature_val(payload, tags_info["alldatetimefeatures"])

        Validate.value_against_list("model_type", payload["model_type"], MODEL_TYPES)

        Validate.value_against_list(
            "baseline_true_label",
            [payload["baseline_true_label"]],
            tags_info["alluniquefeatures"],
        )

        Validate.value_against_list(
            "baseline_pred_label",
            [payload["baseline_pred_label"]],
            tags_info["alluniquefeatures"],
        )

        if payload.get("features_to_use"):
            Validate.value_against_list(
                "features_to_use",
                payload.get("features_to_use", []),
                tags_info["alluniquefeatures"],
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

        Validate.value_against_list("base_line_tag", payload["base_line_tag"], all_tags)
        Validate.value_against_list("current_tag", payload["current_tag"], all_tags)

        Validate.validate_date_feature_val(payload, tags_info["alldatetimefeatures"])

        Validate.value_against_list("model_type", payload["model_type"], MODEL_TYPES)

        Validate.value_against_list(
            "baseline_true_label",
            [payload["baseline_true_label"]],
            tags_info["alluniquefeatures"],
        )

        Validate.value_against_list(
            "baseline_pred_label",
            [payload["baseline_pred_label"]],
            tags_info["alluniquefeatures"],
        )

        Validate.value_against_list(
            "current_true_label",
            [payload["current_true_label"]],
            tags_info["alluniquefeatures"],
        )

        Validate.value_against_list(
            "current_pred_label",
            [payload["current_pred_label"]],
            tags_info["alluniquefeatures"],
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

            Validate.value_against_list(
                "base_line_tag", payload["base_line_tag"], all_tags
            )
            Validate.value_against_list("current_tag", payload["current_tag"], all_tags)

            Validate.value_against_list(
                "stat_test_name", payload["stat_test_name"], DATA_DRIFT_STAT_TESTS
            )
            if payload.get("features_to_use"):
                Validate.value_against_list(
                    "features_to_use",
                    payload.get("features_to_use", []),
                    tags_info["alluniquefeatures"],
                )
        elif type == "Target Drift":
            Validate.check_for_missing_keys(
                payload, TARGET_DRIFT_TRIGGER_REQUIRED_FIELDS
            )

            Validate.value_against_list(
                "base_line_tag", payload["base_line_tag"], all_tags
            )
            Validate.value_against_list("current_tag", payload["current_tag"], all_tags)

            Validate.value_against_list(
                "model_type", payload["model_type"], MODEL_TYPES
            )

            if payload["model_type"] == "classification":
                Validate.value_against_list(
                    "stat_test_name",
                    payload["stat_test_name"],
                    TARGET_DRIFT_STAT_TESTS_CLASSIFICATION,
                )

            if payload["model_type"] == "regression":
                Validate.value_against_list(
                    "stat_test_name",
                    payload["stat_test_name"],
                    TARGET_DRIFT_STAT_TESTS_REGRESSION,
                )

            Validate.value_against_list(
                "baseline_true_label",
                [payload["baseline_true_label"]],
                tags_info["alluniquefeatures"],
            )

            Validate.value_against_list(
                "current_true_label"[payload["current_true_label"]],
                tags_info["alluniquefeatures"],
            )
        elif type == "Model Performance":
            Validate.check_for_missing_keys(payload, MODEL_PERF_TRIGGER_REQUIRED_FIELDS)

            Validate.value_against_list(
                "base_line_tag", payload["base_line_tag"], all_tags
            )

            Validate.value_against_list(
                "model_type", payload["model_type"], MODEL_TYPES
            )

            Validate.value_against_list(
                "baseline_true_label",
                [payload["baseline_true_label"]],
                tags_info["alluniquefeatures"],
            )

            Validate.value_against_list(
                "baseline_pred_label",
                [payload["baseline_pred_label"]],
                tags_info["alluniquefeatures"],
            )

            if payload["model_type"] == "classification":
                if not payload["class_label"]:
                    raise Exception(
                        "class_label is required for classification model type."
                    )

                all_class_label = self.get_labels(payload["baseline_true_label"])

                Validate.value_against_list(
                    "class_label", payload["class_label"], all_class_label
                )

                Validate.value_against_list(
                    "model_performance_metric",
                    payload["model_performance_metric"],
                    MODEL_PERF_METRICS_CLASSIFICATION,
                )

            if payload["model_type"] == "regression":
                Validate.value_against_list(
                    "model_performance_metric",
                    payload["model_performance_metric"],
                    MODEL_PERF_METRICS_REGRESSION,
                )

        else:
            raise Exception(
                'Invalid trigger type. Please use one of ["Data Drift", "Target Drift", "Model Performance"]'
            )

        Validate.value_against_list("frequency", payload["frequency"], MAIL_FREQUENCIES)

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

        trigger_info = res["details"][0]

        if not trigger_info["successful"]:
            return Alert(info={}, detailed_report=[], not_used_features=[])

        trigger_info = trigger_info["details"]

        detailed_report = trigger_info["detailed_report"]
        not_used_features = trigger_info["Not_Used_Features"]

        del trigger_info["detailed_report"]
        del trigger_info["Not_Used_Features"]

        return Alert(
            info=trigger_info,
            detailed_report=detailed_report,
            not_used_features=not_used_features,
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
        :param data_config: config for the data
                        {
                            "tags": List[str]
                            "feature_exclude": List[str]
                            "feature_encodings": List[str]
                            "drop_duplicate_uid": bool
                        },
                        defaults to None
        :param model_config: config with hyper parameters for the model, defaults to None
        :return: response
        """
        project_config = self.config()

        if project_config == "Not Found":
            raise Exception("Upload files first")

        available_models = self.available_models()

        Validate.value_against_list("model_type", model_type, available_models)

        if data_config:
            if data_config.get("feature_exclude"):
                Validate.value_against_list(
                    "feature_exclude",
                    data_config["feature_exclude"],
                    project_config["metadata"]["feature_include"],
                )

            if data_config.get("tags"):
                available_tags = self.tags()
                Validate.value_against_list("tags", data_config["tags"], available_tags)

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
                        Validate.value_against_list(
                            model_config_param,
                            model_config_param_value,
                            model_param["value"],
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

        poll_events(self.__api_client, self.project_name, res["event_id"])

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

        if model_name:
            Validate.value_against_list("model_name", model_name, available_models)

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
    ) -> pd.DataFrame:
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
    ) -> Case:
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
    ) -> str:
        """Delete Cases

        :param unique_identifier: unique identifier of case, defaults to None
        :param start_date: start date of case, defaults to None
        :param end_date: end date of case, defaults to None
        :param tag: tag of case, defaults to None
        :return: response
        """
        if tag:
            all_tags = self.all_tags()
            Validate.value_against_list("tag", tag, all_tags)

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

    def get_notifications(self) -> pd.DataFrame:
        """get user project notifications

        :return: DataFrame
        """
        url = f"{GET_NOTIFICATIONS_URI}?project_name={self.project_name}"

        res = self.__api_client.get(url)

        if not res["success"]:
            raise Exception("Error while getting project notifications.")

        notifications = [
            notification
            for notification in res["details"]
            if notification.get("project_name", None)
        ]

        if not notifications:
            return "No notifications found."

        return pd.DataFrame(notifications).reindex(columns=["message", "time"])

    def clear_notifications(self) -> str:
        """clear user project notifications

        :raises Exception: _description_
        :return: str
        """
        url = f"{CLEAR_NOTIFICATIONS_URI}?project_name={self.project_name}"

        res = self.__api_client.post(url)

        if not res["success"]:
            raise Exception("Error while clearing project notifications.")

        return res["details"]

    def observations(self) -> pd.DataFrame:
        """Observations

        :return: observation details dataframe
        """
        res = self.__api_client.get(
            f"{GET_OBSERVATIONS_URI}?project_name={self.project_name}"
        )

        if not res.get("details"):
            raise Exception("No observations found")

        observation_df = pd.DataFrame(res.get("details"))
        observation_df = observation_df[
            observation_df["status"].isin(["active", "inactive"])
        ]
        if observation_df.empty:
            raise Exception("No observations found")
        observation_df["expression"] = observation_df["metadata"].apply(
            lambda metadata: generate_expression(metadata["expression"])
        )

        observation_df = observation_df.drop(
            columns=[
                "project_name",
                "configuration",
                "metadata",
                "updated_by",
                "created_by",
                "updated_keys",
            ],
            errors="ignore",
        )
        observation_df = observation_df.reindex(
            [
                "observation_id",
                "observation_name",
                "status",
                "statement",
                "linked_features",
                "expression",
                "created_at",
                "updated_at",
            ],
            axis=1,
        )
        observation_df.reset_index(inplace=True, drop=True)
        return observation_df

    def observation_trail(self) -> pd.DataFrame:
        """Observation Trail

        :return: observation trail details dataframe
        """
        res = self.__api_client.get(
            f"{GET_OBSERVATIONS_URI}?project_name={self.project_name}"
        )

        if not res.get("details"):
            raise Exception("No observations found")

        observation_df = pd.DataFrame(res.get("details"))
        observation_df = observation_df[
            observation_df["status"].isin(["updated", "deleted"])
        ]
        if observation_df.empty:
            raise Exception("No observation trail found")
        observation_df = observation_df.rename(
            columns={
                "statement": "old_statement",
                "linked_features": "old_linked_features",
            }
        )
        observation_df["updated_keys"].replace(float("nan"), None, inplace=True)
        observation_df["updated_statement"] = observation_df["updated_keys"].apply(
            lambda data: data.get("statement") if data else None
        )

        observation_df["updated_linked_features"] = observation_df[
            "updated_keys"
        ].apply(lambda data: data.get("linked_features") if data else None)

        observation_df["old_expression"] = observation_df["metadata"].apply(
            lambda metadata: generate_expression(metadata["expression"])
        )

        observation_df["updated_expression"] = observation_df["updated_keys"].apply(
            lambda data: generate_expression(data.get("metadata", {}).get("expression"))
            if data
            else None
        )

        observation_df = observation_df.drop(
            columns=[
                "project_name",
                "configuration",
                "metadata",
                "updated_by",
                "created_by",
                "updated_keys",
            ],
            errors="ignore",
        )

        observation_df = observation_df.reindex(
            [
                "observation_id",
                "observation_name",
                "status",
                "old_statement",
                "updated_statement",
                "old_linked_features",
                "updated_linked_features",
                "old_expression",
                "updated_expression",
                "created_at",
                "updated_at",
            ],
            axis=1,
        )

        observation_df.reset_index(inplace=True, drop=True)
        return observation_df

    def create_observation(
        self,
        observation_name: str,
        expression: str,
        statement: str,
        linked_features: List[str],
    ) -> str:
        """Creates New Observation

        :param observation_name: name of observation
        :param expression: expression of observation
            Eg: BldgType !== Duplex and Neighborhood == OldTown
                Ensure that the left side of the conditional operator corresponds to a feature name,
                and the right side represents the comparison value for the feature.
                Valid conditional operators include "!==," "==," ">,", and "<."
                You can perform comparisons between two or more features using
                logical operators such as "and" or "or."
                Additionally, you have the option to use parentheses () to group and prioritize certain conditions.
        :param statement: statement of observation
            Eg: The building type is {BldgType}
                the content inside the curly brackets represents the feature name
        :param linked_features: linked features of observation
        :return: response
        """
        observation_params = self.__api_client.get(
            f"{GET_OBSERVATION_PARAMS_URI}?project_name={self.project_name}"
        )

        Validate.string("expression", expression)

        Validate.string("statement", statement)

        Validate.value_against_list(
            "linked_feature",
            linked_features,
            observation_params["details"]["eng_features"],
        )
        configuration, expression = build_expression(expression)

        validate_configuration(configuration, observation_params["details"])

        payload = {
            "project_name": self.project_name,
            "observation_name": observation_name,
            "status": "active",
            "configuration": configuration,
            "metadata": {"expression": expression},
            "statement": [statement],
            "linked_features": linked_features,
        }

        res = self.__api_client.post(CREATE_OBSERVATION_URI, payload)
        if not res["success"]:
            raise Exception(res.get("details"))

        return "Observation Created"

    def update_observation(
        self,
        observation_id: str,
        observation_name: str,
        status: Optional[str] = None,
        expression: Optional[str] = None,
        statement: Optional[str] = None,
        linked_features: Optional[List[str]] = None,
    ) -> str:
        """Updates Observation

        :param observation_id: id of observation
        :param observation_name: name of observation
        :param status: status of observation ["active","inactive"]
        :param expression: new expression for observation, defaults to None
            Eg: BldgType !== Duplex and Neighborhood == OldTown
                Ensure that the left side of the conditional operator corresponds to a feature name,
                and the right side represents the comparison value for the feature.
                Valid conditional operators include "!==," "==," ">,", and "<."
                You can perform comparisons between two or more features using
                logical operators such as "and" or "or."
                Additionally, you have the option to use parentheses () to group and prioritize certain conditions.
        :param statement: new statement for observation, defaults to None
            Eg: The building type is {BldgType}
                the content inside the curly brackets represents the feature name
        :param linked_features: new linked features for observation, defaults to None
        :return: response
        """
        if not status and not expression and not statement and not linked_features:
            raise Exception("update parameters for observation not passed")

        payload = {
            "project_name": self.project_name,
            "observation_id": observation_id,
            "observation_name": observation_name,
            "update_keys": {},
        }

        observation_params = self.__api_client.get(
            f"{GET_OBSERVATION_PARAMS_URI}?project_name={self.project_name}"
        )

        if expression:
            Validate.string("expression", expression)
            configuration, expression = build_expression(expression)
            validate_configuration(configuration, observation_params["details"])
            payload["update_keys"]["configuration"] = configuration
            payload["update_keys"]["metadata"] = {"expression": expression}

        if linked_features:
            Validate.value_against_list(
                "linked_feature",
                linked_features,
                observation_params["details"]["eng_features"],
            )
            payload["update_keys"]["linked_features"] = linked_features

        if statement:
            Validate.string("statement", statement)
            payload["update_keys"]["statement"] = [statement]

        if status:
            Validate.value_against_list("status", status, ["active", "inactive"])
            payload["update_keys"]["status"] = status

        res = self.__api_client.post(UPDATE_OBSERVATION_URI, payload)

        if not res["success"]:
            raise Exception(res.get("details"))

        return "Observation Updated"

    def delete_observation(
        self,
        observation_id: str,
        observation_name: str,
    ) -> str:
        """Deletes Observation

        :param observation_id: id of observation
        :param observation_name: name of observation
        :return: response
        """
        payload = {
            "project_name": self.project_name,
            "observation_id": observation_id,
            "observation_name": observation_name,
            "delete": True,
            "update_keys": {},
        }

        res = self.__api_client.post(UPDATE_OBSERVATION_URI, payload)

        if not res["success"]:
            raise Exception(res.get("details"))

        return "Observation Deleted"

    def policies(self) -> pd.DataFrame:
        """Policies

        :return: policy details dataframe
        """
        res = self.__api_client.get(
            f"{GET_POLICIES_URI}?project_name={self.project_name}"
        )

        if not res.get("details"):
            raise Exception("No policies found")

        policy_df = pd.DataFrame(res.get("details"))
        policy_df = policy_df[policy_df["status"].isin(["active", "inactive"])]
        if policy_df.empty:
            raise Exception("No policies found")
        policy_df["expression"] = policy_df["metadata"].apply(
            lambda metadata: generate_expression(metadata["expression"])
        )

        policy_df = policy_df.drop(
            columns=[
                "project_name",
                "configuration",
                "metadata",
                "linked_features",
                "updated_by",
                "created_by",
                "updated_keys",
            ],
            errors="ignore",
        )
        policy_df = policy_df.reindex(
            [
                "policy_id",
                "policy_name",
                "status",
                "statement",
                "decision",
                "expression",
                "created_at",
                "updated_at",
            ],
            axis=1,
        )
        policy_df.reset_index(inplace=True, drop=True)
        return policy_df

    def policy_trail(self) -> pd.DataFrame:
        """Policy Trail

        :return: observation details dataframe
        """
        res = self.__api_client.get(
            f"{GET_POLICIES_URI}?project_name={self.project_name}"
        )

        if not res.get("details"):
            raise Exception("No policies found")

        policy_df = pd.DataFrame(res.get("details"))
        policy_df = policy_df[policy_df["status"].isin(["updated", "deleted"])]
        if policy_df.empty:
            raise Exception("No policy trail found")
        policy_df = policy_df.rename(
            columns={
                "statement": "old_statement",
                "decision": "old_decision",
            }
        )

        policy_df["updated_keys"].replace(float("nan"), None, inplace=True)

        policy_df["updated_statement"] = policy_df["updated_keys"].apply(
            lambda data: data.get("statement") if data else None
        )

        policy_df["updated_decision"] = policy_df["updated_keys"].apply(
            lambda data: data.get("decision") if data else None
        )

        policy_df["old_expression"] = policy_df["metadata"].apply(
            lambda metadata: generate_expression(metadata["expression"])
        )

        policy_df["updated_expression"] = policy_df["updated_keys"].apply(
            lambda data: generate_expression(data.get("metadata", {}).get("expression"))
            if data
            else None
        )

        policy_df = policy_df.drop(
            columns=[
                "project_name",
                "configuration",
                "metadata",
                "updated_by",
                "created_by",
                "updated_keys",
                "linked_features",
            ],
            errors="ignore",
        )
        policy_df = policy_df.reindex(
            [
                "policy_id",
                "policy_name",
                "status",
                "old_statement",
                "updated_statement",
                "old_decision",
                "updated_decision",
                "old_expression",
                "updated_expression",
                "created_at",
                "updated_at",
            ],
            axis=1,
        )
        policy_df.reset_index(inplace=True, drop=True)

        return policy_df

    def create_policy(
        self,
        policy_name: str,
        expression: str,
        statement: str,
        decision: str,
        input: Optional[str] = None,
    ) -> str:
        """Creates New Policy

        :param policy_name: name of policy
        :param expression: expression of policy
            Eg: BldgType !== Duplex and Neighborhood == OldTown
                Ensure that the left side of the conditional operator corresponds to a feature name,
                and the right side represents the comparison value for the feature.
                Valid conditional operators include "!==," "==," ">,", and "<."
                You can perform comparisons between two or more features using
                logical operators such as "and" or "or."
                Additionally, you have the option to use parentheses () to group and prioritize certain conditions.
        :param statement: statement of policy
            Eg: The building type is {BldgType}
                the content inside the curly brackets represents the feature name
        :param decision: decision of policy
        :param input: custom input for the decision if input selected for decision of policy
        :return: response
        """
        configuration, expression = build_expression(expression)

        policy_params = self.__api_client.get(
            f"{GET_POLICY_PARAMS_URI}?project_name={self.project_name}"
        )

        validate_configuration(configuration, policy_params["details"])

        Validate.value_against_list(
            "decision", decision, list(policy_params["details"]["decision"].values())[0]
        )

        if decision == "input":
            Validate.string("Decision input", input)

        payload = {
            "project_name": self.project_name,
            "policy_name": policy_name,
            "status": "active",
            "configuration": configuration,
            "metadata": {"expression": expression},
            "statement": [statement],
            "decision": input if decision == "input" else decision,
        }

        res = self.__api_client.post(CREATE_POLICY_URI, payload)
        if not res["success"]:
            raise Exception(res.get("details"))

        return "Policy Created"

    def update_policy(
        self,
        policy_id: str,
        policy_name: str,
        status: Optional[str] = None,
        expression: Optional[str] = None,
        statement: Optional[str] = None,
        decision: Optional[str] = None,
        input: Optional[str] = None,
    ) -> str:
        """Updates Policy

        :param policy_id: id of policy
        :param policy_name: name of policy
        :param status: status of policy ["active","inactive"]
        :param expression: new expression for policy, defaults to None
            Eg: BldgType !== Duplex and Neighborhood == OldTown
                Ensure that the left side of the conditional operator corresponds to a feature name,
                and the right side represents the comparison value for the feature.
                Valid conditional operators include "!==," "==," ">,", and "<."
                You can perform comparisons between two or more features using
                logical operators such as "and" or "or."
                Additionally, you have the option to use parentheses () to group and prioritize certain conditions.
        :param statement: new statment for policy, defaults to None
            Eg: The building type is {BldgType}
                the content inside the curly brackets represents the feature name
        :param decision: new decision for policy, defaults to None
        :param input: custom input for the decision if input selected for decision of policy
        :return: response
        """
        if not status and not expression and not statement and not decision:
            raise Exception("update parameters for policy not passed")

        payload = {
            "project_name": self.project_name,
            "policy_id": policy_id,
            "policy_name": policy_name,
            "update_keys": {},
        }

        policy_params = self.__api_client.get(
            f"{GET_POLICY_PARAMS_URI}?project_name={self.project_name}"
        )

        if expression:
            Validate.string("expression", expression)
            configuration, expression = build_expression(expression)
            validate_configuration(configuration, policy_params["details"])
            payload["update_keys"]["configuration"] = configuration
            payload["update_keys"]["metadata"] = {"expression": expression}

        if statement:
            Validate.string("statement", statement)
            payload["update_keys"]["statement"] = [statement]

        if status:
            Validate.value_against_list("status", status, ["active", "inactive"])
            payload["update_keys"]["status"] = status

        if decision:
            Validate.value_against_list(
                "decision",
                decision,
                list(policy_params["details"]["decision"].values())[0],
            )
            if decision == "input":
                Validate.string("Decision input", input)
            payload["update_keys"]["decision"] = (
                input if decision == "input" else decision
            )

        res = self.__api_client.post(UPDATE_POLICY_URI, payload)

        if not res["success"]:
            raise Exception(res.get("details"))

        return "Policy Updated"

    def delete_policy(
        self,
        policy_id: str,
        policy_name: str,
    ) -> str:
        """Deletes Policy

        :param policy_id: id of policy
        :param policy_name: name of policy
        :return: response
        """
        payload = {
            "project_name": self.project_name,
            "policy_id": policy_id,
            "policy_name": policy_name,
            "delete": True,
            "update_keys": {},
        }

        res = self.__api_client.post(UPDATE_POLICY_URI, payload)

        if not res["success"]:
            raise Exception(res.get("details"))

        return "Policy Deleted"

    def get_synthetic_model_params(self) -> dict:
        """get hyper parameters of synthetic models

        :return: hyper params
        """
        return self.__api_client.get(GET_SYNTHETIC_MODEL_PARAMS_URI)

    def train_synthetic_model(
        self,
        model_name: str,
        data_config: Optional[SyntheticDataConfig] = {},
        hyper_params: Optional[SyntheticModelHyperParams] = {},
    ):
        """Train synthetic model

        :param model_name: model name ['GPT2', 'CTGAN']
        :param data_config: config for the data
            {
                "tags": List[str]
                "feature_exclude": List[str]
                "feature_include": List[str]
                "drop_duplicate_uid": bool
            },
            defaults to {}
        :param hyper_params: hyper parameters for the model. check param type and value range below,
            For GPT2 (Generative Pretrained Transformer) model - Works well on high dimensional tabular data,
            {
                "batch_size": int [1, 500] defaults to 100
                "early_stopping_patience": int [1, 100], defaults to 10
                "early_stopping_threshold": float [0.1, 100], defaults to 0.0001
                "epochs": int [1, 150], defaults to 100
                "model_type": "tabular",
                "random_state": int [1, 150], defaults to 1
                "tabular_config": "GPT2Config",
                "train_size": float [0, 0.9] defaults to 0.8
            }
            For CTGAN (Conditional Tabular GANs) model - Balances between training computation and dimensionality,
            {
                "epochs": int, [1, 150] defaults to 100
                "test_ratio": float [0, 1] defaults to 0.2
            }
            defaults to {}

        :return: response
        """
        project_config = self.config()

        if project_config == "Not Found":
            raise Exception("Upload files first")

        project_config = project_config["metadata"]

        all_models_param = self.get_synthetic_model_params()

        try:
            model_params = all_models_param[model_name]
        except KeyError as e:
            availabel_models = list(all_models_param.keys())
            Validate.value_against_list("model", [model_name], availabel_models)

        # validate and prepare data config
        data_config["model_name"] = model_name

        available_tags = self.tags()
        tags = data_config.get("tags", project_config["avaialble_tags"])

        Validate.value_against_list("tag", tags, available_tags)

        feature_exclude = data_config.get(
            "feature_exclude", project_config["feature_exclude"]
        )

        Validate.value_against_list(
            "feature_exclude", feature_exclude, project_config["avaialble_options"]
        )

        feature_include = data_config.get(
            "feature_include", project_config["feature_include"]
        )

        Validate.value_against_list(
            "feature_include",
            feature_include,
            project_config["avaialble_options"],
        )

        drop_duplicate_uid = data_config.get(
            "drop_duplicate_uid", project_config["drop_duplicate_uid"]
        )

        SYNTHETIC_MODELS_DEFAULT_HYPER_PARAMS[model_name].update(hyper_params)
        hyper_params = SYNTHETIC_MODELS_DEFAULT_HYPER_PARAMS[model_name]

        # validate model hyper parameters
        for key, value in hyper_params.items():
            model_param = model_params.get(key, None)

            if model_param:
                if model_param["type"] == "input":
                    if model_param["value"] == "int":
                        if not isinstance(value, int):
                            raise Exception(f"{key} value should be integer")
                    elif model_param["value"] == "float":
                        if not isinstance(value, float):
                            raise Exception(f"{key} value should be float")

                        if value < model_param["min"] or value > model_param["max"]:
                            raise Exception(
                                f"{key} value should be between {model_param['min']} and {model_param['max']}"
                            )
                    elif model_param["type"] == "select":
                        Validate.value_against_list(
                            "value", [value], model_param["value"]
                        )

        print(f"Using data config: {json.dumps(data_config, indent=4)}")
        print(f"Using hyper params: {json.dumps(hyper_params, indent=4)}")

        payload = {
            "project_name": self.project_name,
            "model_name": model_name,
            "metadata": {
                "model_name": model_name,
                "tags": tags,
                "feature_exclude": feature_exclude,
                "feature_include": feature_include,
                "feature_actual_used": [],
                "drop_duplicate_uid": drop_duplicate_uid,
                "model_parameters": hyper_params,
            },
        }

        res = self.__api_client.post(TRAIN_SYNTHETIC_MODEL_URI, payload)

        if not res["success"]:
            raise Exception(res["details"])

        print('Training initiated...')
        poll_events(self.__api_client, self.project_name, res["event_id"])

    def remove_synthetic_model(self, model_name: str) -> str:
        """deletes synthetic model

        :param model_name: model name
        :raises ValueError: _description_
        :raises Exception: _description_
        :return: response message
        """
        models_df = self.synthetic_models()
        valid_models = models_df['model_name'].tolist()

        if model_name not in valid_models:
            raise ValueError(f"{model_name} is not valid. Pick a valid value from {valid_models}")
        
        payload = {
            "project_name": self.project_name,
            "model_name": model_name
        }

        res = self.__api_client.post(DELETE_SYNTHETIC_MODEL_URI, payload)

        if not res['success']:
            raise Exception(res['details'])

        return f"{model_name} deleted successfully."

    def synthetic_models(self) -> pd.DataFrame:
        """get synthetic models for the project

        :return: synthetic models
        """
        url = f"{GET_SYNTHETIC_MODELS_URI}?project_name={self.project_name}"

        res = self.__api_client.get(url)

        if not res["success"]:
            raise Exception("Error while getting synthetics models.")

        models = res["details"]

        models_df = pd.DataFrame(models)

        return models_df

    def synthetic_model(self, model_name: str) -> dict:
        """get synthetic model details

        :param model_name: model name
        :raises Exception: _description_
        :return: _description_
        """
        models_df = self.synthetic_models()
        valid_models = models_df['model_name'].tolist()

        if model_name not in valid_models:
            raise ValueError(f"{model_name} is not valid. Pick a valid value from {valid_models}")

        url = f"{GET_SYNTHETIC_MODEL_DETAILS_URI}?project_name={self.project_name}&model_name={model_name}"

        res = self.__api_client.get(url)

        if not res["success"]:
            raise Exception(res["details"])

        model_details = res["details"][0]

        metadata = model_details["metadata"]
        data_quality = model_details["results"]

        del model_details["metadata"]
        del model_details["results"]

        synthetic_model = SyntheticModel(
            **model_details,
            **data_quality,
            metadata=metadata,
            api_client=self.__api_client,
            project=self,
        )

        return synthetic_model

    def synthetic_tags(self) -> pd.DataFrame:
        """get synthetic data tags of the model
        :raises Exception: _description_
        :return: list of tags
        """
        url = f"{GET_SYNTHETIC_DATA_TAGS_URI}?project_name={self.project_name}"

        res = self.__api_client.get(url)

        if not res["success"]:
            raise Exception("Error while getting synthetics data tags.")

        data_tags = res['details']

        for data_tag in data_tags:
            del data_tag["metadata"]
            del data_tag["plot_data"]

        return pd.DataFrame(data_tags)

    def synthetic_tag(self, tag: str) -> SyntheticDataTag:
        """get synthetic data tag by tag name
        :param tag: tag name
        :raises Exception: _description_
        :return: tag
        """
        url = f"{GET_SYNTHETIC_DATA_TAGS_URI}?project_name={self.project_name}"

        res = self.__api_client.get(url)

        if not res["success"]:
            raise Exception("Error while getting synthetics data tags.")

        data_tags = res['details']

        syn_data_tags = [SyntheticDataTag(
                    **data_tag,
                    api_client=self.__api_client,
                    project_name=self.project_name,
                    project=self
                ) for data_tag in data_tags]

        data_tag = next((syn_data_tag for syn_data_tag in syn_data_tags if syn_data_tag.tag == tag), None)

        if not data_tag:
            valid_tags = [syn_data_tag.tag for syn_data_tag in syn_data_tags]
            raise Exception(f'{tag} is invalid. Pick a valid value from {valid_tags}')

        return data_tag

    def synthetic_tag_datapoints(self, tag: str) -> pd.DataFrame:
        """get synthetic tag datapoints

        :param tag: tag name
        :raises Exception: _description_
        :return: datapoints
        """
        all_tags = self.all_tags()

        Validate.value_against_list(
            'tag',
            tag,
            all_tags,
        )

        payload = {
            "project_name": self.project_name,
            "tag": tag
        }

        res = self.__api_client.request(
            "POST",
            DOWNLOAD_SYNTHETIC_DATA_URI,
            payload
        )

        synthetic_data = pd.read_csv(io.StringIO(res.content.decode('utf-8')))

        return synthetic_data

    def remove_synthetic_tag(self, tag: str) -> str:
        """delete synthetic data tag

        :raises Exception: _description_
        :return: response messsage
        """
        all_tags = self.all_tags()

        Validate.value_against_list(
            'tag',
            tag,
            all_tags,
        )

        payload = {
            "project_name": self.project_name,
            "tag": tag,
        }

        res = self.__api_client.post(DELETE_SYNTHETIC_TAG_URI, payload)

        if not res["success"]:
            raise Exception(res["details"])

        return f"{tag} deleted successfully."

    def get_observation_params(self) -> dict:
        """get observation parameters for the project (used in validating synthetic prompt)"""
        url = f"{GET_OBSERVATION_PARAMS_URI}?project_name={self.project_name}"

        res = self.__api_client.get(url)

        return res["details"]

    def create_synthetic_prompt(self, name: str, expression: str) -> str:
        """create synthetic prompt for the project

        :param name: prompt name
        :param expression: expression of policy
            Eg: BldgType !== Duplex and Neighborhood == OldTown
                Ensure that the left side of the conditional operator corresponds to a feature name,
                and the right side represents the comparison value for the feature.
                Valid conditional operators include "!==," "==," ">,", and "<."
                You can perform comparisons between two or more features using
                logical operators such as "and" or "or."
                Additionally, you have the option to use parentheses () to group and prioritize certain conditions.
        :raises Exception: _description_
        :return: response message
        """
        name = name.strip()

        if not name:
            raise Exception("name is required")

        configuration, expression = build_expression(expression)

        prompt_params = self.get_observation_params()
        validate_configuration(configuration, prompt_params)

        payload = {
            "prompt_name": name,
            "project_name": self.project_name,
            "configuration": configuration,
            "metadata": {"expression": expression},
        }

        res = self.__api_client.post(CREATE_SYNTHETIC_PROMPT_URI, payload)

        if not res["success"]:
            raise Exception(res["details"])

        return "Synthetic prompt created successfully."

    def update_synthetic_prompt(self, prompt_id: str, status: str) -> str:
        """update synthetic prompt

        :param prompt_id: prompt id
        :param activate: True or False
        :raises Exception: _description_
        :raises Exception: _description_
        :return: response message
        """
        if status not in ["active", "inactive"]:
            raise Exception("Invalid status value. Pick a valid value from ['active', 'inactive'].")

        payload = {
            "delete": False,
            "project_name": self.project_name,
            "prompt_id": prompt_id,
            "update_keys": {
                "status": status
            }
        }

        res = self.__api_client.post(UPDATE_SYNTHETIC_PROMPT_URI, payload)

        if not res['success']:
            raise Exception(res['details'])

        return 'Synthetic prompt updated successfully.'

    def synthetic_prompts(self) -> pd.DataFrame:
        """get synthetic prompts for the project

        :raises Exception: _description_
        :return: _description_
        """
        url = f"{GET_SYNTHETIC_PROMPT_URI}?project_name={self.project_name}"

        res = self.__api_client.get(url)

        if not res["success"]:
            raise Exception(res["details"])

        prompts = res["details"]

        return pd.DataFrame(prompts).reindex(columns=["prompt_id", "prompt_name", "status", "created_at", "updated_at"])

    def synthetic_prompt(self, prompt_id: str) -> SyntheticPrompt:
        """get synthetic prompt by prompt id

        :raises Exception: _description_
        :return: _description_
        """
        url = f"{GET_SYNTHETIC_PROMPT_URI}?project_name={self.project_name}"

        res = self.__api_client.get(url)

        if not res["success"]:
            raise Exception(res["details"])

        prompts = res["details"]

        curr_prompt = next(
            (prompt for prompt in prompts if prompt['prompt_id'] == prompt_id),
            None
        )

        if not curr_prompt:
            raise Exception(f"Invalid prompt_id")

        return SyntheticPrompt(
            **curr_prompt,
            api_client=self.__api_client,
            project=self
        )

    def __print__(self) -> str:
        return f"Project(user_project_name='{self.user_project_name}', created_by='{self.created_by}')"

    def __str__(self) -> str:
        return self.__print__()

    def __repr__(self) -> str:
        return self.__print__()

def generate_expression(expression):
    if not expression:
        return None
    generated_expression = ""
    for item in expression:
        if isinstance(item, str):
            generated_expression += " " + item
        else:
            generated_expression += (
                f" {item['column']} {item['expression']} {item['value']}"
            )
    return generated_expression


def build_expression(expression_string):
    condition_operators = {
        "!==": "_NOTEQ",
        "==": "_ISEQ",
        ">": "_GRT",
        "<": "_LST",
    }
    logical_operators = {"and": "_AND", "or": "_OR"}

    metadata_expression = []
    configuration = []
    string_to_be_parsed = expression_string

    matches = re.findall(r"(\w+)\s*([!=<>]+)\s*(\w+)", expression_string)

    total_opening_parentheses = re.findall(r"\(", expression_string)
    total_closing_parentheses = re.findall(r"\)", expression_string)

    if len(total_opening_parentheses) != len(total_closing_parentheses):
        raise Exception("Invalid expression, check parentheses")

    for i, match in enumerate(matches):
        column, expression, value = match
        if expression not in condition_operators.keys():
            raise Exception(f"Not a valid condition operator in {match}")

        opening_parentheses = re.findall(r"\(", string_to_be_parsed.split(column, 1)[0])
        if opening_parentheses:
            metadata_expression.extend(opening_parentheses)
            configuration.extend(opening_parentheses)

        metadata_expression.append(
            {
                "column": column,
                "value": value,
                "expression": expression,
            }
        )
        configuration.append(
            {
                "column": column,
                "value": value,
                "expression": condition_operators[expression],
            }
        )

        string_to_be_parsed = string_to_be_parsed.split(value, 1)[1]
        between_conditions_split = string_to_be_parsed.split(
            matches[i + 1][0] if i < len(matches) - 1 else None, 1
        )
        closing_parentheses = re.findall(
            r"\)",
            between_conditions_split[0] if len(between_conditions_split) > 0 else "",
        )
        if closing_parentheses:
            metadata_expression.extend(closing_parentheses)
            configuration.extend(closing_parentheses)

        if i < len(matches) - 1:
            between_conditions = between_conditions_split[0].strip()
            between_conditions = between_conditions.replace(")", "").replace("(", "")
            logical_operator = re.search(r"and|or", between_conditions)
            if not logical_operator:
                raise Exception(f"{between_conditions} is not valid logical operator")
            log_operator = logical_operator.group()
            log_operator_split = list(
                filter(
                    lambda op: op != "" and op != " ",
                    between_conditions.split(log_operator, 1),
                )
            )
            if len(log_operator_split) > 0:
                raise Exception(f"{between_conditions} is not valid logical operator")
            metadata_expression.append(log_operator)
            configuration.append(logical_operators[log_operator])

    return configuration, metadata_expression


def validate_configuration(configuration, params):
    for expression in configuration:
        if isinstance(expression, str):
            if expression not in ["(", ")", *params.get("logical_operators")]:
                raise Exception(f"{expression} not a valid logical operator")

        if isinstance(expression, dict):
            # validate column name
            Validate.value_against_list(
                "feature", expression.get("column"), params.get("eng_features")
            )

            # validate operator
            Validate.value_against_list(
                "condition_operator",
                expression.get("expression"),
                params.get("condition_operators"),
            )

            # validate value(s)
            expression_value = expression.get("value")
            valid_feature_values = params.get("features").get(expression.get("column"))

            if isinstance(valid_feature_values, str):
                if valid_feature_values == "input" and not parse_float(
                    expression_value
                ):
                    raise Exception(
                        f"Invalid value comparison with {expression_value} for {expression.get('column')}"
                    )

                if valid_feature_values == "datetime" and not parse_datetime(
                    expression_value
                ):
                    raise Exception(
                        f"Invalid value comparison with {expression_value} for {expression.get('column')}"
                    )
