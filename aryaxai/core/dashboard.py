import os
from typing import Any
from pydantic import BaseModel
import json
from IPython.display import IFrame, display

from aryaxai.common.xai_uris import XAI_APP_URI


class Dashboard(BaseModel):
    config: dict
    query_params: str
    raw_data: dict | list

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.print_config()
        self.plot()

    def plot(self, width: int = "100%", height: int = 800):
        """plot the dashboard by remote url

        Args:
            width (int, optional): _description_. Defaults to 100%.
            height (int, optional): _description_. Defaults to 650.
        """
        uri = os.environ.get("XAI_APP_URL", XAI_APP_URI)
        url = f"{uri}/sdk/dashboard{self.query_params}"
        display(IFrame(src=f"{url}", width=width, height=height))

    def get_config(self) -> dict:
        """
        get the dashboard config
        """
        config_copy = {**self.config}
        config_copy.pop("metadata", None)
        return config_copy

    def get_raw_data(self) -> dict:
        """
        get the dashboard raw data
        """
        raw_data = {"created_at": self.config.get("created_at")}

        if self.config["type"] == "data_drift":
            data_drift_table = next(
                filter(
                    lambda data: data["metric"] == "DataDriftTable",
                    self.raw_data.get("metrics"),
                ),
                None,
            )
            if data_drift_table:
                for item in data_drift_table["result"].get("drift_by_columns"):
                    item.pop("current_small_distribution", None)
                    item.pop("reference_small_distribution", None)
                    item.pop("current_big_distribution", None)
                    item.pop("reference_big_distribution", None)
                    item.pop("current_mean", None)
                    item.pop("reference_std", None)
                raw_data.update(data_drift_table["result"])

        if self.config["type"] == "target_drift":
            column_drift_metric = next(
                filter(
                    lambda data: data["metric"] == "ColumnDriftMetric",
                    self.raw_data.get("metrics"),
                ),
                None,
            )
            if column_drift_metric:
                column_drift_metric["result"].pop("data", None)

                raw_data.update(column_drift_metric["result"])

        if self.config["type"] == "performance":
            classification_quality_metric = next(
                filter(
                    lambda data: data["metric"] == "ClassificationQualityMetric",
                    self.raw_data.get("metrics"),
                ),
                None,
            )
            if classification_quality_metric:
                for curr_ref in ["current", "reference"]:
                    classification_quality_metric["result"][curr_ref].pop(
                        "rate_plots_data", None
                    )
                    classification_quality_metric["result"][curr_ref].pop(
                        "plot_data", None
                    )
                raw_data.update(classification_quality_metric["result"])

        return raw_data

    def print_config(self):
        """
        pretty print the cdashboard config
        """
        config = {k: v for k, v in self.config.items() if v is not None}
        config.pop("metadata", None)
        print("Using config: ", end="")
        print(json.dumps(config, indent=4))

    def __print__(self) -> str:
        return f"Dashboard(config='{self.get_config()}')"

    def __str__(self) -> str:
        return self.__print__()

    def __repr__(self) -> str:
        return self.__print__()
