from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict
import plotly.graph_objects as go
import pandas as pd
from IPython.display import SVG, display


class Case(BaseModel):
    status: str
    true_value: str | int
    pred_value: str | int
    pred_category: str | int
    observations: List
    shap_feature_importance: Optional[Dict] = {}
    lime_feature_importance: Optional[Dict] = {}
    similar_cases: List
    is_automl_prediction: Optional[bool] = False
    model_name: str
    case_prediction_path: Optional[str] = ""
    case_prediction_svg: Optional[str] = ""
    observation_checklist: Optional[List] = []
    policy_checklist: Optional[List] = []
    final_decision: Optional[str] = ""
    unique_identifier: Optional[str] = ""
    tag: Optional[str] = ""
    created_at: Optional[str] = ""
    data: Optional[Dict] = {}
    similar_cases_data: Optional[List] = []
    image_data: Optional[Dict] = {}

    model_config = ConfigDict(protected_namespaces=())

    def explainability_shap_feature_importance(self):
        """Plots Shap Feature Importance chart"""
        fig = go.Figure()

        if len(list(self.shap_feature_importance.values())) < 1:
            return "No Shap Feature Importance for the case"

        if isinstance(list(self.shap_feature_importance.values())[0], dict):
            for col in self.shap_feature_importance.keys():
                fig.add_trace(
                    go.Bar(
                        x=list(self.shap_feature_importance[col].values()),
                        y=list(self.shap_feature_importance[col].keys()),
                        orientation="h",
                        name=col,
                    )
                )
        else:
            fig.add_trace(
                go.Bar(
                    x=list(self.shap_feature_importance.values()),
                    y=list(self.shap_feature_importance.keys()),
                    orientation="h",
                )
            )
        fig.update_layout(
            barmode="relative",
            height=800,
            width=800,
            yaxis_autorange="reversed",
            bargap=0.01,
            legend_orientation="h",
            legend_x=0.1,
            legend_y=1.1,
        )
        fig.show(config={"displaylogo": False})

    def explainability_lime_feature_importance(self):
        """Plots Lime Feature Importance chart"""
        fig = go.Figure()

        if len(list(self.lime_feature_importance.values())) < 1:
            return "No Lime Feature Importance for the case"

        if isinstance(list(self.lime_feature_importance.values())[0], dict):
            for col in self.lime_feature_importance.keys():
                fig.add_trace(
                    go.Bar(
                        x=list(self.lime_feature_importance[col].values()),
                        y=list(self.lime_feature_importance[col].keys()),
                        orientation="h",
                        name=col,
                    )
                )
        else:
            fig.add_trace(
                go.Bar(
                    x=list(self.lime_feature_importance.values()),
                    y=list(self.lime_feature_importance.keys()),
                    orientation="h",
                )
            )
        fig.update_layout(
            barmode="relative",
            height=800,
            width=800,
            yaxis_autorange="reversed",
            bargap=0.01,
            legend_orientation="h",
            legend_x=0.1,
            legend_y=1.1,
        )
        fig.show(config={"displaylogo": False})

    def explainability_prediction_path(self):
        """Explainability Prediction Path"""
        svg = SVG(self.case_prediction_svg)
        display(svg)

    def explainability_raw_data(self) -> pd.DataFrame:
        """Explainability Raw Data

        :return: raw data dataframe
        """
        raw_data_df = (
            pd.DataFrame([self.data])
            .transpose()
            .reset_index()
            .rename(columns={"index": "Feature", 0: "Value"})
        )
        return raw_data_df

    def explainability_observations(self) -> pd.DataFrame:
        """Explainability Observations

        :return: observations dataframe
        """
        observations_df = pd.DataFrame(self.observation_checklist)

        return observations_df

    def explainability_policies(self) -> pd.DataFrame:
        """Explainability Policies

        :return: policies dataframe
        """
        policy_df = pd.DataFrame(self.policy_checklist)

        return policy_df

    def explainability_decision(self) -> pd.DataFrame:
        """Explainability Decision

        :return: decision dataframe
        """
        data = {
            "True Value": self.true_value,
            "Prediction Value": self.pred_value,
            "Prediction Category": self.pred_category,
            "Final Prediction": self.final_decision,
        }
        decision_df = pd.DataFrame([data])

        return decision_df

    def explainability_similar_cases(self) -> pd.DataFrame | str:
        """Similar Cases

        :return: similar cases dataframe
        """
        if not self.similar_cases_data:
            return "No similar cases found. Or add 'similar_cases' in components case_info()"

        similar_cases_df = pd.DataFrame(self.similar_cases_data)
        return similar_cases_df

    def explainability_gradcam(self):
        fig = go.Figure()

        fig.add_layout_image(
            dict(
                source=self.image_data.get("gradcam", {}).get("heatmap"),
                xref="x",
                yref="y",
                x=0,
                y=1,
                sizex=1,
                sizey=1,
                xanchor="left",
                yanchor="top",
                layer="below",
            )
        )

        fig.add_layout_image(
            dict(
                source=self.image_data.get("gradcam", {}).get("superimposed"),
                xref="x",
                yref="y",
                x=1.2,
                y=1,
                sizex=1,
                sizey=1,
                xanchor="left",
                yanchor="top",
                layer="below",
            )
        )

        fig.add_annotation(
            x=0.5,
            y=0.1,
            text="Attributions",
            showarrow=False,
            font=dict(size=16),
            xref="x",
            yref="y",
        )
        fig.add_annotation(
            x=1.7,
            y=0.1,
            text="Superimposed",
            showarrow=False,
            font=dict(size=16),
            xref="x",
            yref="y",
        )
        fig.update_layout(
            xaxis=dict(visible=False, range=[0, 2.5]),
            yaxis=dict(visible=False, range=[0, 1]),
            margin=dict(l=30, r=30, t=30, b=30),
        )

        fig.show(config={"displaylogo": False})

    def explainability_shap(self):
        fig = go.Figure()

        fig.add_layout_image(
            dict(
                source=self.image_data.get("shap", {}).get("plot"),
                xref="x",
                yref="y",
                x=0,
                y=1,
                sizex=1,
                sizey=1,
                xanchor="left",
                yanchor="top",
                layer="below",
            )
        )

        fig.update_layout(
            xaxis=dict(visible=False, range=[0, 2.5]),
            yaxis=dict(visible=False, range=[0, 1]),
            margin=dict(l=30, r=30, t=30, b=30),
        )

        fig.show(config={"displaylogo": False})

    def explainability_lime(self):
        fig = go.Figure()

        fig.add_layout_image(
            dict(
                source=self.image_data.get("lime", {}).get("plot"),
                xref="x",
                yref="y",
                x=0,
                y=1,
                sizex=1,
                sizey=1,
                xanchor="left",
                yanchor="top",
                layer="below",
            )
        )

        fig.update_layout(
            xaxis=dict(visible=False, range=[0, 2.5]),
            yaxis=dict(visible=False, range=[0, 1]),
            margin=dict(l=30, r=30, t=30, b=30),
        )

        fig.show(config={"displaylogo": False})

    def explainability_integrated_gradients(self):
        fig = go.Figure()

        fig.add_layout_image(
            dict(
                source=self.image_data.get("integrated_gradients", {}).get(
                    "attributions"
                ),
                xref="x",
                yref="y",
                x=0,
                y=1,
                sizex=1,
                sizey=1,
                xanchor="left",
                yanchor="top",
                layer="below",
            )
        )

        fig.add_layout_image(
            dict(
                source=self.image_data.get("integrated_gradients", {}).get(
                    "positive_attributions"
                ),
                xref="x",
                yref="y",
                x=1.2,
                y=1,
                sizex=1,
                sizey=1,
                xanchor="left",
                yanchor="top",
                layer="below",
            )
        )

        fig.add_layout_image(
            dict(
                source=self.image_data.get("integrated_gradients", {}).get(
                    "negative_attributions"
                ),
                xref="x",
                yref="y",
                x=2.4,
                y=1,
                sizex=1,
                sizey=1,
                xanchor="left",
                yanchor="top",
                layer="below",
            )
        )

        fig.add_annotation(
            x=0.5,
            y=0.1,
            text="Attributions",
            showarrow=False,
            font=dict(size=16),
            xref="x",
            yref="y",
        )
        fig.add_annotation(
            x=1.7,
            y=0.1,
            text="Positive Attributions",
            showarrow=False,
            font=dict(size=16),
            xref="x",
            yref="y",
        )
        fig.add_annotation(
            x=2.9,
            y=0.1,
            text="Negative Attributions",
            showarrow=False,
            font=dict(size=16),
            xref="x",
            yref="y",
        )
        fig.update_layout(
            xaxis=dict(visible=False, range=[0, 2.5]),
            yaxis=dict(visible=False, range=[0, 1]),
            margin=dict(l=30, r=30, t=30, b=30),
        )

        fig.show(config={"displaylogo": False})
