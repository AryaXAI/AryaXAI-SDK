from typing import List, Optional, TypedDict

from pydantic import BaseModel, ConfigDict


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

class SyntheticModelHyperParams(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    # GPT2 hyper params
    batch_size: Optional[int] = 250,
    early_stopping_patience: Optional[int] = 10
    early_stopping_threshold: Optional[float] = 0.0001
    epochs: Optional[int] = 100,
    model_type: Optional[str] = "tabular",
    random_state: Optional[int] = 1,
    tabular_config: Optional[str] = "GPT2Config",
    train_size: Optional[float] = 0.8

    #CTGAN hyper params
    epochs: Optional[int] = 100
    test_ratio: Optional[float] = 0.2