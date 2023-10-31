from aryaxai.client.client import APIClient
from aryaxai.common.xai_uris import MODEL_SVG_URI
from typing import Dict
from pydantic import BaseModel, ConfigDict
import plotly.graph_objects as go
from IPython.display import SVG, display


class ModelSummary(BaseModel):
    project_name: str
    project_type: str
    unique_identifier: str
    true_label: str
    pred_label: str
    metadata: Dict
    model_results: Dict
    is_automl: bool
    Source: str

    model_config = ConfigDict(protected_namespaces=())

    __api_client: APIClient

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__api_client = kwargs.get("api_client")

    def info(self) -> dict:
        """Model Info

        :return: model info dict
        """
        info = {
            "source": self.Source,
            "model_name": self.model_results.get("model_name"),
            "model_type": self.model_results.get("model_type"),
            "model_param": self.model_results.get("model_params"),
            "data_tags_used_for_modelling": self.model_results.get("data_used_tags"),
            "modelling_info": self.model_results.get("modelling_info"),
        }

        return info

    def feature_importance(self):
        """Global features plot"""
        global_features = self.model_results.get("GFI")
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                y=list(global_features.keys()),
                x=list(global_features.values()),
                orientation="h",
            )
        )

        fig.update_layout(
            title="Global Feaure",
            xaxis_title="Values",
            yaxis_title="Features",
            width=800,
            height=600,
            yaxis_autorange="reversed",
            bargap=0.01,
            legend_orientation="h",
            legend_x=0.1,
            legend_y=1.1,
        )

        fig.show(config={"displaylogo": False})

    def prediction_path(self):
        """Prediction path plot"""
        model_name = self.model_results.get("model_name")
        res = self.__api_client.get(
            f"{MODEL_SVG_URI}?project_name={self.project_name}&model_name={model_name}"
        )

        if not res["success"]:
            raise Exception(res.get("details"))

        svg = SVG(res.get("details"))
        display(svg)
