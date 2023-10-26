from typing import List, Optional
from pydantic import BaseModel, Field, validator


class MonitoringPayload(BaseModel):
    project_name: Optional[str]
    base_line_tag: List[str] = Field(..., description="Base line tags are required.")
    
    date_feature: Optional[str] = ""
    
    # baseline_date: Optional[dict] = {}
    # current_date: Optional[dict] = {}

class DataDriftPayload(MonitoringPayload):
    current_tag: List[str] = Field(..., description="Current tags are required.")
    features_to_use: List[str] = []
    stat_test_name: str = "psi"
    stat_test_threshold: float = 0.2

    @validator('stat_test_threshold')
    def validate_stat_test_threshold(cls, v):
        if not isinstance(v, float):
            raise ValueError('Stat test threshold must be a float.')
        return v


class TargetDriftPayload(MonitoringPayload):
    baseline_true_label: str = Field(..., description="Baseline true label is required.")
    current_tag: List[str] = Field(..., description="Current tags are required.")
    current_true_label: Optional[str] = "Predicted_value_AutoML"
    model_type: str = Field(..., description="Model type is required.")
    stat_test_name: str = "psi"
    stat_test_threshold: float = 0.2

    @validator('model_type')
    def validate_model_type(cls, v):
        if v not in ['classification']:
            raise ValueError('Model type must be either "classification" or "regression".')
        return v
    
    @validator('stat_test_threshold')
    def validate_stat_test_threshold(cls, v):
        if not isinstance(v, float):
            raise ValueError('Stat test threshold must be a float.')
        return v


class BiasMonitoringPayload(MonitoringPayload):
    baseline_true_label: str = Field(..., description="Baseline true label is required.")
    baseline_pred_label: str = Field(..., description="Baseline predicted label is required.")
    features_to_use: List[str] = []
    model_type: str = Field(..., description="Model type is required.")

    @validator('model_type')
    def validate_model_type(cls, v):
        if v not in ['classification', 'regression']:
            raise ValueError('Model type must be either "classification" or "regression".')
        return v


class ModelPerformancePayload(MonitoringPayload):
    baseline_true_label: str = Field(..., description="Baseline true label is required.")
    baseline_pred_label: str = Field(..., description="Baseline predicted label is required.")
    current_tag: List[str] = Field(..., description="Current tags are required.")
    current_true_label: Optional[str] = "Predicted_value_AutoML"
    current_pred_label: str = Field(..., description="Current predicted label is required.")
    model_type: str = Field(..., description="Model type is required.")

    @validator('model_type')
    def validate_model_type(cls, v):
        if v not in ['classification', 'regression']:
            raise ValueError('Model type must be either "classification" or "regression".')
        return v
