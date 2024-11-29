from typing import List, Optional, TypedDict, Dict


class ProjectConfig(TypedDict):
    project_type: str
    unique_identifier: str
    true_label: str
    tag: str
    pred_label: Optional[str]
    feature_exclude: Optional[List[str]]
    drop_duplicate_uid: Optional[bool]
    handle_errors: Optional[bool]
    feature_encodings: Optional[dict]

class DataConfig(TypedDict):
    tags: List[str]
    feature_exclude: List[str]
    feature_encodings: Dict[str, str]
    drop_duplicate_uid: bool
    sample_percentage: float
    explainability_sample_percentage: float
    lime_explainability_iterations: int
    explainability_method: List[str]

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

    # CTGAN hyper params
    epochs: Optional[int]
    test_ratio: Optional[float]

class GCSConfig(TypedDict):
    project_id: str
    gcp_project_name: str
    type: str
    private_key_id: str
    private_key: str
    client_email: str
    client_id: str
    auth_uri: str
    token_uri: str

class S3Config(TypedDict):
    region: Optional[str] = None
    access_key: str
    secret_key: str

class GDriveConfig(TypedDict):
    gdrive_file_id: str
    gdrive_file_name: str
