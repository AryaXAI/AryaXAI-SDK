from typing import List, Optional, TypedDict

class DataDriftPayload(TypedDict):
    project_name: Optional[str]
    base_line_tag: List[str]
    current_tag: List[str]

    date_feature: Optional[str]
    baseline_date: Optional[dict]
    current_date: Optional[dict]

    features_to_use: List[str]

    stat_test_name: str
    stat_test_threshold: str


class TargetDriftPayload(TypedDict):
    project_name: str
    base_line_tag: List[str]
    current_tag: List[str]

    date_feature: Optional[str]
    baseline_date: Optional[dict]
    current_date: Optional[dict]

    model_type: str

    baseline_true_label: str
    current_true_label: str

    stat_test_name: str
    stat_test_threshold: float

class BiasMonitoringPayload(TypedDict):
    project_name: str
    base_line_tag: List[str]

    date_feature: Optional[str]
    baseline_date: Optional[dict]
    current_date: Optional[dict]

    baseline_true_label: str
    baseline_pred_label: str

    features_to_use: List[str]
    model_type: str

class ModelPerformancePayload(TypedDict):
    project_name: str
    base_line_tag: List[str]
    current_tag: List[str]

    date_feature: Optional[str]
    baseline_date: Optional[dict]
    current_date: Optional[dict]

    baseline_true_label: str
    baseline_pred_label: str
    current_true_label: str
    current_pred_label: str

    model_type: str
