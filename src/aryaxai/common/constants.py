MODEL_TYPES = [
    'classification',
    'regression'
]

TARGET_DRIFT_MODEL_TYPES = [
    "classification"
]


DATA_DRIFT_DASHBOARD_REQUIRED_FIELDS = [
    "base_line_tag",
    "current_tag",
    "stat_test_name"
]

DATA_DRIFT_STAT_TESTS = [
    'chisquare',
    'jensenshannon',
    'ks',
    'kl_div',
    'psi',
    'wasserstein',
    'z'
]


TARGET_DRIFT_DASHBOARD_REQUIRED_FIELDS = [
    "base_line_tag",
    "current_tag",
    "baseline_true_label",
    "current_true_label",
    "model_type",
    "stat_test_name"
]

TARGET_DRIFT_STAT_TESTS = [
    'chisquare',
    'jensenshannon',
    'kl_div',
    'psi',
    'z'
]

TARGET_DRIFT_STAT_TESTS_CLASSIFICATION = [
    'chisquare',
    'jensenshannon',
    'kl_div',
    'psi',
]

TARGET_DRIFT_STAT_TESTS_REGRESSION = [
    'jensenshannon',
    'kl_div',
    'ks',
    'psi',
    'wasserstein',
]


BIAS_MONITORING_DASHBOARD_REQUIRED_FIELDS = [
    "base_line_tag",
    "baseline_true_label",
    "baseline_pred_label",
    "model_type",
]

MODEL_PERF_DASHBOARD_REQUIRED_FIELDS = [
    "base_line_tag",
    "current_tag",  
    "baseline_true_label",
    "baseline_pred_label",
    "current_true_label",
    "current_pred_label",
    "model_type"
]


DATA_DRIFT_TRIGGER_REQUIRED_FIELDS = [
    "trigger_name",
    "trigger_type",
    "mail_list",
    "frequency",
    "stat_test_name",
    "datadrift_features_per",
    "base_line_tag",
    "current_tag",
]

TARGET_DRIFT_TRIGGER_REQUIRED_FIELDS = [
    "trigger_name",
    "trigger_type",
    "mail_list",
    "frequency",
    "model_type",
    "stat_test_name",
    "baseline_true_label",
    "current_true_label",
    "base_line_tag",
    "current_tag",
]

MODEL_PERF_TRIGGER_REQUIRED_FIELDS = [
    "trigger_name",
    "trigger_type",
    "mail_list",
    "frequency",
    "model_type",
    "model_performance_metric",
    "model_performance_threshold",
    "baseline_true_label",
    "baseline_pred_label",  
    "base_line_tag"
]

MODEL_PERF_METRICS_CLASSIFICATION = [
    "accuracy",
    "f1",
    "false_negative_rate",
    "false_positive_rate",
    "precision",
    "recall",
    "true_negative_rate",
    "true_positive_rate"
]

MODEL_PERF_METRICS_REGRESSION = [
    "mean_abs_perc_error",
    "mean_abs_perc_error",
    "mean_squared_error",
    "r2_score"
]

MAIL_FREQUENCIES = [
    'daily',
    'weekly',
    'monthly',
    'quarterly',
    'yearly'
]
