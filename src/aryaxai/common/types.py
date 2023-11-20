from typing import List, Optional, TypedDict


class ProjectConfig(TypedDict):
    project_type: str
    unique_identifier: str
    true_label: str
    tag: str
    pred_label: Optional[str]
    feature_exclude: Optional[List[str]]


class DataConfig(TypedDict):
    tags: List[str]
    feature_exclude: List[str]
    feature_encodings: List[str]
    drop_duplicate_uid: bool

class SyntheticDataConfig(TypedDict):
    model_name: str
    tags: List[str]
    feature_exclude: List[str]
    feature_include: List[str]
    feature_actual_used: List[str]
    drop_duplicate_uid: bool