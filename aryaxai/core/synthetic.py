from typing import List
from pydantic import BaseModel
import pandas as pd

import plotly.graph_objects as go


class SyntheticDataTag(BaseModel):
    info: dict
    metadata: dict
    plot_data: List[dict]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

    def view_info(self) -> pd.DataFrame:
        """pretty print the synthetic data tag info

        :return: DataFrame
        """
        info = {k: v for k, v in self.info.items() if v is not None}

        return pd.DataFrame(info, index=[0])

    def view_metadata(self) -> pd.DataFrame:
        """pretty print the synthetic data metadata

        :return: DataFrame
        """
        metadata = {k: v for k, v in self.metadata.items() if v is not None}

        return pd.DataFrame(metadata, index=[0])

    def __print__(self) -> str:
        return f"SyntheticDataTag({self.info})"

    def __str__(self) -> str:
        return self.__print__()

    def __repr__(self) -> str:
        return self.__print__()


class SyntheticModel:
    pass
