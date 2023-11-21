from typing import List, Optional, Callable


class Validate:
    @staticmethod
    def string(field_name: str, value: str):
        if not (value and value.strip()):
            raise Exception(f"{field_name} is not valid, cannot be empty")

    @staticmethod
    def list(field_name: str, value: List):
        if not value:
            raise Exception(f"{field_name} is required, cannot be empty list")

    @staticmethod
    def value_against_list(
        field_name: str,
        value: str | List,
        valid_list: List,
        handle_error: Optional[Callable] = None,
    ):
        try:
            if isinstance(value, str):
                if value not in valid_list:
                    raise Exception(
                        f"{value} is not valid {field_name}, select from \n{valid_list}"
                    )
            if isinstance(value, List):
                Validate.list(field_name, value)
                for val in value:
                    if val not in valid_list:
                        raise Exception(
                            f"{val} is not a valid {field_name}, pick valid {field_name} from \n{valid_list}"
                        )
        except Exception as e:
            if handle_error:
                handle_error()
            raise e

    @staticmethod
    def check_for_missing_keys(value: dict, required_keys: list):
        missing_keys = [key for key in required_keys if key not in value]

        if missing_keys:
            raise Exception(f"Keys not present: {missing_keys}")

    @staticmethod
    def validate_date_feature_val(payload: dict, all_datetime_features: List[str]):
        if payload.get("date_feature", None):
            if payload["date_feature"] not in all_datetime_features:
                raise Exception(
                    f"{payload} is not a valid date_feature. Pick a valid payload from {all_datetime_features}."
                )

            if not payload["baseline_date"]:
                raise Exception(
                    "baseline_date is required when date_feature is passed."
                )
            if not payload["current_date"]:
                raise Exception("current_date is required when date_feature is passed.")

            if (
                payload["baseline_date"]["start_date"]
                > payload["baseline_date"]["end_date"]
            ):
                raise Exception(
                    "start_date of baseline_date should be less than end_date."
                )
            if (
                payload["current_date"]["start_date"]
                > payload["current_date"]["end_date"]
            ):
                raise Exception(
                    "start_date of current_date should be less than end_date."
                )
