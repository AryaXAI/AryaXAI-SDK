from common.payload import MonitoringPayload
from pydantic import BaseModel
from typing import List, Optional
from aryaxai.client.client import APIClient
from aryaxai.common.types import ProjectConfig
from aryaxai.common.validation import Validate
from aryaxai.common.xai_uris import (
    DATA_DRFIT_DIAGNOSIS_URI,
    DELETE_DATA_FILE_URI,
    GET_DATA_DIAGNOSIS_URI,
    GET_DATA_SUMMARY_URI,
    GET_PROJECT_CONFIG,
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
import pandas as pd
import json


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

    def delete_file(self, path: str) -> str:
        """deletes file for the project

        :param path: uploaded file path
        :return: response
        """
        payload = {
            "project_name": self.project_name,
            "workspace_name": self.workspace_name,
            "path": path,
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

        print(res["data"]["overview"])
        summary = pd.DataFrame(res["data"]["data"][tag]).drop(
            ["Warnings", "data_description"], axis=1
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
    

    def get_data_drift_dashboard(self, config: MonitoringPayload) -> str:
        """get data drift dashboard url

        Args:
            config (MonitoringPayload): config for data drift dashboard

        Returns:
            str: data drift dashboard url
        """
        res = self.__api_client.post(DATA_DRIFT_DASHBOARD_URI, config)

        if not res['success']:
            error_details = res.get('details', 'Failed to get dashboard url')
            raise Exception(error_details)

        dashboard_url = res.get('hosted_path', None)
        auth_token = self.__api_client.get_auth_token()
        query_params = f'?id={auth_token}'

        return f"{dashboard_url}{query_params}"
    
    def get_target_drift_dashboard(self, config: MonitoringPayload) -> str:
        """get target drift dashboard url

        Args:
            config (MonitoringPayload): config for target drift dashboard

        Returns:
            str: target drift dashboard url
        """        
        res = self.__api_client.post(TARGET_DRIFT_DASHBOARD_URI, config)

        if not res['success']:
            error_details = res.get('details', 'Failed to get dashboard url')
            raise Exception(error_details)

        dashboard_url = res.get('hosted_path', None)
        auth_token = self.__api_client.get_auth_token()
        query_params = f'?id={auth_token}'

        return f"{dashboard_url}{query_params}"
    
    def get_bias_monitoring_dashboard(self, config: MonitoringPayload) -> str:
        """get bias monitoring dashboard url

        Args:
            config (MonitoringPayload): config for bias monitoring dashboard

        Returns:
            None: bias monitoring dashboard url
        """        
        res = self.__api_client.post(BIAS_MONITORING_DASHBOARD_URI, config)

        if not res['success']:
            error_details = res.get('details', 'Failed to get dashboard url')
            raise Exception(error_details)

        dashboard_url = res.get('hosted_path', None)
        auth_token = self.__api_client.get_auth_token()
        query_params = f'?id={auth_token}'

        return f"{dashboard_url}{query_params}"
    
    def get_model_performance_dashboard(self, config: MonitoringPayload) -> str:
        """get model performance dashboard url

        Args:
            config (MonitoringPayload): config for model performance dashboard

        Returns:
            str: model performance dashboard url
        """        
        res = self.__api_client.post(MODEL_PERFORMANCE_DASHBOARD_URI, config)

        if not res['success']:
            error_details = res.get('details', 'Failed to get dashboard url')
            raise Exception(error_details)

        dashboard_url = res.get('hosted_path', None)
        auth_token = self.__api_client.get_auth_token()
        query_params = f'?id={auth_token}'

        return f"{dashboard_url}{query_params}"


    def __print__(self) -> str:
        return f"Project(user_project_name='{self.user_project_name}', created_by='{self.created_by}')"

    def __str__(self) -> str:
        return self.__print__()

    def __repr__(self) -> str:
        return self.__print__()
