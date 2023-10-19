class Validate:
    @staticmethod
    def check_for_missing_keys(value: dict, required_keys: list):
        missing_keys = [key for key in required_keys if key not in value]

        if missing_keys:
            raise Exception(f"Keys not present: {missing_keys}")
