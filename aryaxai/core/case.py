from __future__ import annotations
from typing import Dict, List
from pydantic import BaseModel, ConfigDict
import plotly.graph_objects as go
import pandas as pd
from IPython.display import SVG, display


class Case(BaseModel):
    status: str
    true_value: str
    pred_value: str
    pred_category: str
    observations: List
    shap_feature_importance: Dict
    lime_feature_importance: Dict
    similar_cases: List
    is_automl_prediction: bool
    model_name: str
    case_prediction_path: str
    case_prediction_svg: str
    observation_checklist: List
    policy_checklist: List
    final_decision: str
    unique_identifier: str
    tag: str
    created_at: str
    data: Dict
    similar_cases_data: List

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
            "AutoML Prediction": self.pred_value,
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
