from typing import List, Optional


class Validate:
    @staticmethod
    def check_for_missing_keys(value: dict, required_keys: list):
        missing_keys = [key for key in required_keys if key not in value]

        if missing_keys:
            raise Exception(f"Keys not present: {missing_keys}")
        
    @staticmethod
    def validate_tags(tags: List[str], all_tags: List[str]):
        if tags:
            for tag in tags:
                if tag not in all_tags:
                    raise Exception(f"{tag} is not a valid tag. Pick a valid value from {all_tags}.")
        

    @staticmethod
    def validate_date_feature_val(payload: dict, all_datetime_features: List[str]):
        if payload.get('date_feature', None):
            if payload['date_feature'] not in all_datetime_features:
                raise Exception(f"{payload} is not a valid date_feature. Pick a valid payload from {all_datetime_features}.")
                
            if not payload['baseline_date']:
                raise Exception("baseline_date is required when date_feature is passed.")
            if not payload['current_date']:
                raise Exception("current_date is required when date_feature is passed.")
            
            if payload['baseline_date']['start_date'] > payload['baseline_date']['end_date']:
                raise Exception("start_date of baseline_date should be less than end_date.")
            if payload['current_date']['start_date'] > payload['current_date']['end_date']:
                raise Exception("start_date of current_date should be less than end_date.")
            
    @staticmethod
    def validate_features(features: List[str], all_features: List[str]):
        if features:
            for feature in features:
                if feature not in all_features:
                    raise Exception(f"{feature} is not a valid feature. Pick a valid value from {all_features}.")

    @staticmethod
    def raise_exception_on_invalid_value(values: List[str], valid_values: List[str], field_name: str = 'value'):
        """raise exception if values are not among valid values

        :param value: values to be validated
        :param valid_values: list of valid values
        :param field_name: field name to be added in exception message, defaults to 'value'
        :raises Exception: Given value is invalid among valid values
        """
        if values and valid_values:
            for value in values:
                if value not in valid_values:
                    raise Exception(f"{value} is not valid {field_name}. Pick a valid {field_name} from {valid_values}.")
