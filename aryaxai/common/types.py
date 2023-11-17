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

class SyntheticModelHyperParams(TypedDict):
    # GPT2 hyper params
    batch_size: Optional[int]
    early_stopping_patience: Optional[int]
    early_stopping_threshold: Optional[float]
    epochs: Optional[int]
    model_type: Optional[str]
    random_state: Optional[int]
    tabular_config: Optional[str]
    train_size: Optional[float]

    #CTGAN hyper params
    epochs: Optional[int]
    test_ratio: Optional[float]