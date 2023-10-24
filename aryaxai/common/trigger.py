from datetime import datetime
from typing import List, Dict
from pydantic import BaseModel


class TriggerPayload(BaseModel):
    project_name: str
    baseline_date: Dict[str, str]
    base_line_tag: List[str]
    current_tag: List[str]
    mail_list: List[str]
    trigger_name: str
    trigger_type: str
    model_type: str 
    stat_test_name: str
    stat_test_threshold: float
    baseline_true_label: str
    current_true_label: str
    date_feature: str
    frequency: str
    curr_time_period_in_days: int