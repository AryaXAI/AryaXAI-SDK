from pydantic import BaseModel
from typing import List, Optional

class MonitoringPayload(BaseModel):
    project_name: str
    
    base_line_tag: List[str]
    baseline_date: Optional[dict]
    baseline_true_label: Optional[str]
    baseline_pred_label: Optional[str]
    
    current_tag: List[str]
    current_date: Optional[dict]
    current_true_label: Optional[str]
    current_pred_label: Optional[str]
    
    date_feature: str
    features_to_use: List[str]
    model_type: Optional[str]
        
    stat_test_name: str
    stat_test_threshold: int