import io
from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict
import pandas as pd

import plotly.graph_objects as go

from aryaxai.client.client import APIClient
from aryaxai.common.validation import Validate
from aryaxai.common.xai_uris import DELETE_SYNTHETIC_MODEL_URI, DELETE_SYNTHETIC_TAG_URI, DOWNLOAD_SYNTHETIC_DATA_URI, GENERATE_ANONIMITY_SCORE_URI, GENERATE_SYNTHETIC_DATA_URI, GET_ANONIMITY_SCORE_URI, GET_SYNTHETIC_TRAINING_LOGS_URI, UPDATE_SYNTHETIC_PROMPT_URI

class SyntheticModel(BaseModel):
    """Synthetic Model Class

    :param BaseModel: _description_
    :return: _description_
    """
    __api_client: APIClient
    project_name: str
    project: Any

    model_name: str
    created_at: str
    created_by: str

    overall_quality_score: float
    column_shapes:float
    column_pair_trends: float

    metadata: Optional[dict] = {}
    plot_data: Optional[List[dict]] = []

    model_config = ConfigDict(protected_namespaces=())

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__api_client = kwargs.get("api_client")

    def get_model_type(self) -> str:
        """get model type

        :return: model type
        """
        return self.metadata['model_name']

    def get_data_quality(self) -> pd.DataFrame:
        """get data quality

        :return: data quality metrics
        """
        quality = {
            "overall_quality_score": self.overall_quality_score,
            "column_shapes": self.column_shapes,
            "column_pair_trends": self.column_pair_trends
        }

        df = pd.DataFrame(quality, index=[0])

        return df

    def plot_psi(self):
        """plot psi chart"""
        x_data = [item["Column"] for item in self.plot_data]
        y_data = [item["Quality Score"] for item in self.plot_data]
        metric_data = [item["Metric"] for item in self.plot_data]

        traces = []
        for metric in set(metric_data):
            indices = [i for i, val in enumerate(metric_data) if val == metric]
            traces.append(
                go.Bar(
                    x=[x_data[i] for i in indices],
                    y=[y_data[i] for i in indices],
                    name=metric
                )
            )

        fig = go.Figure(data=traces)

        fig.update_layout(
            barmode="relative",
            xaxis_title="Column Names",
            yaxis_title="Quality Score",
            height=500,
            bargap=0.01,
            legend_orientation="h",
            legend_x=0.1,
            legend_y=1.1,
        )

        fig.show(config={"displaylogo": False})

    def get_training_logs(self) -> str:
        """get model training logs

        :return: logs of string type
        """
        url = f"{GET_SYNTHETIC_TRAINING_LOGS_URI}?project_name={self.project_name}&model_name={self.model_name}"

        res = self.__api_client.get(url)

        if not res['success']:
            raise Exception('Error while getting training logs.')

        return res['details']

    def generate_datapoints(self, num_of_datapoints: int):
        """generate given number of synthetic datapoints

        :param num_of_datapoints: total datapoints to generate
        :raises Exception: _description_
        :return: None
        """
        payload = {
            "project_name": self.project_name,
            "model_name": self.model_name,
            "num_of_datapoints": num_of_datapoints
        }

        res = self.__api_client.post(GENERATE_SYNTHETIC_DATA_URI, payload)

        if not res['success']:
            raise Exception(res['details'])

        return res['details']

    def generate_anonimity_score(self, aux_columns: List[str], control_tag: str):
        """generate anonimity score

        :param aux_columns: list of features
        :param control_tag: tag
        :raises Exception: _description_
        :raises Exception: _description_
        :return: None
        """
        if len(aux_columns) < 2:
            raise Exception('aux_columns requires minimum 2 columns.')

        project_config = self.project.config()['metadata']

        Validate.raise_exception_on_invalid_value(
            aux_columns,
            project_config['feature_include'],
            'feature'
        )

        all_tags = self.project.all_tags()

        synthetic_tags = [tag for tag in all_tags if tag.endswith("SyntheticData")]

        Validate.raise_exception_on_invalid_value(
            [control_tag],
            synthetic_tags,
            field_name='tag'
        )

        payload = {
            "aux_columns": aux_columns,
            "control_tag": control_tag,
            "model_name": self.model_name,
            "project_name": self.project_name
        }

        res = self.__api_client.post(GENERATE_ANONIMITY_SCORE_URI, payload)

        if not res['success']:
            raise Exception(res['details'])

        return res['details']

    def get_anonimity_score(self):
        """get anonimity score

        :raises Exception: _description_
        :return: _description_
        """
        payload = {
            "project_name": self.project_name,
            "model_name": self.model_name,
        }

        res = self.__api_client.post(GET_ANONIMITY_SCORE_URI, payload)

        if not res['success']:
            raise Exception('Error while getting anonimity score.')

        return res['details']['scores']

    def delete(self):
        """
        deletes current model
        """
        payload = {
            "project_name": self.project_name,
            "model_name": self.model_name
        }

        res = self.__api_client.post(DELETE_SYNTHETIC_MODEL_URI, payload)

        if not res['success']:
            raise Exception(res['details'])

        return res['details']

    def __print__(self) -> str:
        return f"SyntheticModel(model_name={self.model_name}, created_at={self.created_at}, created_by={self.created_by})"

    def __str__(self) -> str:
        return self.__print__()

    def __repr__(self) -> str:
        return self.__print__()

class SyntheticDataTag(BaseModel):
    __api_client: APIClient
    project_name: str
    project: Any

    model_name: str
    tag: str
    created_at: str

    overall_quality_score: float
    column_shapes:float
    column_pair_trends: float

    metadata: Optional[dict] = {}
    plot_data: Optional[List[dict]] = []

    model_config = ConfigDict(protected_namespaces=())

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__api_client = kwargs.get("api_client")

    def get_model_name(self) -> str:
        """get model type

        :return: model type
        """
        return self.model_name

    def get_data_quality(self) -> pd.DataFrame:
        """get data quality

        :return: data quality metrics
        """
        quality = {
            "overall_quality_score": self.overall_quality_score,
            "column_shapes": self.column_shapes,
            "column_pair_trends": self.column_pair_trends
        }

        df = pd.DataFrame(quality, index=[0])

        return df

    def plot_psi(self):
        """plot psi chart"""
        x_data = [item["Column"] for item in self.plot_data]
        y_data = [item["Quality Score"] for item in self.plot_data]
        metric_data = [item["Metric"] for item in self.plot_data]

        traces = []
        for metric in set(metric_data):
            indices = [i for i, val in enumerate(metric_data) if val == metric]
            traces.append(
                go.Bar(
                    x=[x_data[i] for i in indices],
                    y=[y_data[i] for i in indices],
                    name=metric
                )
            )

        fig = go.Figure(data=traces)

        fig.update_layout(
            barmode="relative",
            xaxis_title="Column",
            yaxis_title="Quality Score",
            height=500,
            bargap=0.01,
            legend_orientation="h",
            legend_x=0.1,
            legend_y=1.1,
        )

        fig.show(config={"displaylogo": False})

    def get_metadata(self) -> pd.DataFrame:
        """view metadata of synthetic tag"""
        df = pd.DataFrame(self.metadata, index=[0])

        return df

    def get_datapoints(self) -> dict:
        """get synthetic datapoints

        :raises Exception: _description_
        :return: datapoints
        """
        all_tags = self.project.all_tags()
        synthetic_tags = [tag for tag in all_tags if tag.endswith("SyntheticData")]

        Validate.raise_exception_on_invalid_value(
            [self.tag],
            synthetic_tags,
            field_name='tag'
        )

        payload = {
            "project_name": self.project_name,
            "tag": self.tag
        }

        res = self.__api_client.request(
            'POST',
            DOWNLOAD_SYNTHETIC_DATA_URI,
            payload
        )

        synthetic_data = pd.read_csv(io.StringIO(res.content.decode('utf-8')))

        return synthetic_data

    def delete(self):
        """delete data tag

        :raises Exception: _description_
        :return: None
        """
        all_tags = self.project.all_tags()
        synthetic_tags = [tag for tag in all_tags if tag.endswith("SyntheticData")]

        Validate.raise_exception_on_invalid_value(
            [self.tag],
            synthetic_tags,
            field_name='tag'
        )

        payload = {
            "project_name": self.project_name,
            "tag": self.tag,
        }

        res = self.__api_client.post(DELETE_SYNTHETIC_TAG_URI, payload)

        if not res["success"]:
            raise Exception(res["details"])

        return res["details"]

    def __print__(self) -> str:
        return f"SyntheticDataTag(model_name={self.model_name}, tag={self.tag}, created_at={self.created_at})"

    def __str__(self) -> str:
        return self.__print__()

    def __repr__(self) -> str:
        return self.__print__()


class SyntheticPrompt(BaseModel):
    __api_client: APIClient
    project: Any

    prompt_name: str
    prompt_id: str
    project_name: str
    status: str
    configuration: List[dict]
    metadata: dict
    created_by: str
    updated_by: str
    created_at: str
    updated_at: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__api_client = kwargs.get("api_client")

    def get_expression(self) -> str:
        """construct prompt expression

        :return: prompt expression
        """
        expression_list = []

        if not self.metadata:
            raise Exception('Expression not found.')

        for item in self.metadata['expression']:
            if isinstance(item, dict):
                expression_list.append(f"{item['column']} {item['expression']} {item['value']}")
            else:
                expression_list.append(item)

        return ' '.join(expression_list)
    
    def get_config(self) -> List[dict]:
        """get prompt configuration

        :return: prompt configuration
        """
        return self.configuration
    
    def activate(self) -> str:
        """activate prompt

        :raises Exception: _description_
        :raises Exception: _description_
        :return: response message
        """
        if self.status == 'active':
            raise Exception('Prompt is already active.')

        payload = {
            "delete": False,
            "project_name": self.project_name,
            "prompt_id": self.prompt_id,
            "update_keys": {
                "status": "active"
            }
        }

        res = self.__api_client.post(UPDATE_SYNTHETIC_PROMPT_URI, payload)

        if not res['success']:
            raise Exception(res['details'])

        self.status = res['details'][0]['status']

        return 'Prompt activated successfully.'
        
    def deactivate(self) -> str:
        """deactive prompt

        :raises Exception: _description_
        :raises Exception: _description_
        :return: response message
        """
        if self.status == 'inactive':
            raise Exception('Prompt is already inactive.')

        payload = {
            "delete": False,
            "project_name": self.project_name,
            "prompt_id": self.prompt_id,
            "update_keys": {
                "status": "inactive"
            }
        }

        res = self.__api_client.post(UPDATE_SYNTHETIC_PROMPT_URI, payload)

        if not res['success']:
            raise Exception(res['details'])

        self.status = res['details'][0]['status']

        return 'Prompt deactivated successfully.'

    '''
    def delete(self) -> str:
        payload = {
            "delete": True,
            "project_name": self.project_name,
            "prompt_id": self.prompt_id,
            "update_keys": {}
        }

        res = self.__api_client.post(UPDATE_SYNTHETIC_PROMPT_URI, payload)

        if not res['success']:
            raise Exception(res['details'])
        
        return res['details']
    '''

    def __print__(self) -> str:
        return f"SyntheticPrompt(prompt_id={self.prompt_id}, prompt_name={self.prompt_name}, status={self.status}, created_by={self.created_by}, created_at={self.created_at}, updated_at={self.updated_at})"

    def __str__(self) -> str:
        return self.__print__()

    def __repr__(self) -> str:
        return self.__print__()