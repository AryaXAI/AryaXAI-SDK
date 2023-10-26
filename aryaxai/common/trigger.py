from datetime import datetime
from typing import List, Dict
from pydantic import BaseModel, Field, validator


class TriggerPayload(BaseModel):
    project_name: str
    trigger_name: str
    trigger_type: str
    
    base_line_tag: List[str]
    current_tag: List[str]
    stat_test_name: str = "psi"
    stat_test_threshold: float = 0.2
    
    frequency: str
    mail_list: List[str] = []
    
    date_feature: str = ""
    
    # baseline_date: Dict[str, str]
    # model_type: str 
    # model_performance_metric: str
    # model_performance_threshold: str

    # baseline_true_label: str
    # current_true_label: str
    # curr_time_period_in_days: int
    
    @validator('frequency')
    def validate_model_type(cls, v):
        available_freq = ['daily', 'weekly', 'monthly', 'quarterly', 'yearly']
        if v not in available_freq:
            raise ValueError(f'Invalid frequency value. Please use one of {available_freq}')
        return v
    
    # @validator('model_type')
    # def validate_model_type(cls, v):
    #     if v not in ['classification', 'regression']:
    #         raise ValueError('Model type must be either "classification" or "regression".')
    #     return v
    
    # @validator('model_performance_metric')
    # def validate_model_performance_metric(cls, v):
    #     available_metrics = ['mean_abs_perc_error', 'mean_squared_error', 'r2_score', 'roc_auc_score', 'f1_score']
        
    #     if v not in available_metrics:
    #         raise ValueError(f'Invalid performance metric. Please use one of {available_metrics}')
    #     return v