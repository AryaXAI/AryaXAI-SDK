from typing import List, Optional, TypedDict
    
class DataDriftTriggerPayload(TypedDict):
    project_name: str
    trigger_name: str
    trigger_type: str

    mail_list: List[str]
    frequency: str

    stat_test_name: str
    stat_test_threshold: Optional[float]

    datadrift_features_per: float

    features_to_use: List[str]

    date_feature: Optional[str]
    baseline_date: Optional[dict]
    current_date: Optional[dict]

    base_line_tag: List[str]
    current_tag: List[str]
    
    
class TargetDriftTriggerPayload(TypedDict):
    project_name: str
    trigger_name: str
    trigger_type: str

    mail_list: List[str]
    frequency: str

    model_type: str

    stat_test_name: str
    stat_test_threshold: Optional[float]

    baseline_true_label: str
    current_true_label: str

    date_feature: Optional[str]
    baseline_date: Optional[dict]
    current_date: Optional[dict]

    base_line_tag: List[str]
    current_tag: List[str]

    
class ModelPerfTriggerPayload(TypedDict):
    project_name: str
    trigger_name: str
    trigger_type: str

    mail_list: List[str]
    frequency: str

    model_type: str
    model_performance_metric: float
    model_performance_threshold: float

    baseline_true_label: str
    baseline_pred_label: str

    date_feature: Optional[str]
    baseline_date: Optional[dict]
    current_date: Optional[dict]

    base_line_tag: List[str]