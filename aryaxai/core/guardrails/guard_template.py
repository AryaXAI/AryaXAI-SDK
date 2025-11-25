from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict, Union 

class Guard:
    """Predefined guardrail configurations for common use cases"""
    
    @staticmethod
    def detect_pii(entities: List[str] = None) -> Dict[str, Any]:
        """Template for PII detection guardrail
        
        Args:
            entities: List of PII entity types to detect
        """
        if entities is None:
            entities = ["EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON", "ADDRESS"]
        
        return {
            "name": "Detect PII",
            "config": {
                "pii_entities": entities
            }
        }
    
    @staticmethod
    def nsfw_text(threshold: float = 0.8, validation_method: str = "sentence") -> Dict[str, Any]:
        """Template for NSFW text detection guardrail
        
        Args:
            threshold: Confidence threshold for detection (0.0-1.0)
            validation_method: "sentence", "paragraph", or "document"
        """
        return {
            "name": "NSFW Text",
            "config": {
                "threshold": threshold,
                "validation_method": validation_method
            }
        }
    @staticmethod
    def ban_list(banned_words: List[str]) -> Dict[str, Any]:
        """Template for banned words guardrail"""
        return {
            "name": "Ban List",
            "config": {
                "banned_words": banned_words
            }
        }
    @staticmethod
    def bias_check(threshold: float = 0.9) -> Dict[str, Any]:
        """Template for bias check guardrail"""
        return {
            "name": "Bias Check",
            "config": {
                "threshold": threshold
            }
        }
    
    @staticmethod
    def competitor_check(competitors: List[str]) -> Dict[str, Any]:
        """Template for competitor guardrail"""
        return {
            "name": "Competitor Check",
            "config": {
                "competitors": competitors
            }
        }
    
    @staticmethod
    def correct_language(expected_language_iso: str = "en", threshold: float = 0.75) -> Dict[str, Any]:
        """Template for correct language guardrail"""
        return {
            "name": "Correct Language",
            "config": {
                "expected_language_iso": expected_language_iso,
                "threshold": threshold
            }
        }
    
    @staticmethod
    def gibberish_text(threshold: float = 0.5, validation_method: str = "sentence") -> Dict[str, Any]:
        """Template for gibberish text guardrail"""
        return {
            "name": "Gibberish Text",
            "config": {
                "threshold": threshold,
                "validation_method": validation_method
            }
        }
    
    @staticmethod
    def profanity_free() -> Dict[str, Any]:
        """Template for profanity free guardrail"""
        return {
            "name": "Profanity Free",
            "config": {}
        }
    
    @staticmethod
    def secrets_present() -> Dict[str, Any]:
        """Template for secrets present guardrail"""
        return {
            "name": "Secrets Present",
            "config": {}
        }
    
    @staticmethod
    def toxic_language(threshold: float = 0.5, validation_method: str = "sentence") -> Dict[str, Any]:
        """Template for toxic language guardrail"""
        return {
            "name": "Toxic Language",
            "config": {
                "threshold": threshold,
                "validation_method": validation_method
            }
        }

    @staticmethod
    def contains_string(substring: str) -> Dict[str, Any]:
        """Template for contains string guardrail"""
        return {
            "name": "Contains String",
            "config": {
                "substring": substring
            }
        }

    @staticmethod
    def detect_jailbreak(threshold: float = 0.0) -> Dict[str, Any]:
        """Template for detect jailbreak guardrail"""
        return {
            "name": "Detect Jailbreak",
            "config": {
                "threshold": threshold
            }
        }

    @staticmethod
    def endpoint_is_reachable() -> Dict[str, Any]:
        """Template for endpoint is reachable guardrail"""
        return {
            "name": "Endpoint Is Reachable",
            "config": {}
        }
    
    @staticmethod
    def ends_with(end: str) -> Dict[str, Any]:
        """Template for ends with guardrail"""
        return {
            "name": "Ends With",
            "config": {
                "end": end
            }
        }

    @staticmethod
    def has_url() -> Dict[str, Any]:
        """Template for has url guardrail"""
        return {
            "name": "Has Url",
            "config": {}
        }

    @staticmethod
    def lower_case() -> Dict[str, Any]:
        """Template for lower case guardrail"""
        return {
            "name": "Lower Case",
            "config": {}
        }

    @staticmethod
    def mentions_drugs() -> Dict[str, Any]:
        """Template for mentions drugs guardrail"""
        return {
            "name": "Mentions Drugs",
            "config": {}
        }

    @staticmethod
    def one_line() -> Dict[str, Any]:
        """Template for one line guardrail"""
        return {
            "name": "One Line",
            "config": {}
        }

    @staticmethod
    def reading_time(reading_time: float) -> Dict[str, Any]:
        """Template for reading time guardrail"""
        return {
            "name": "Reading Time",
            "config": {
                "reading_time": reading_time
            }
        }

    @staticmethod
    def redundant_sentences(threshold: int = 70) -> Dict[str, Any]:
        """Template for redundant sentences guardrail"""
        return {
            "name": "Redundant Sentences",
            "config": {
                "threshold": threshold
            }
        }

    @staticmethod
    def regex_match(regex: str, match_type: str = "search") -> Dict[str, Any]:
        """Template for regex match guardrail"""
        return {
            "name": "Regex Match",
            "config": {
                "regex": regex,
                "match_type": match_type
            }
        }

    @staticmethod
    def sql_column_presence(cols: List[str]) -> Dict[str, Any]:
        """Template for SQL column presence guardrail"""
        return {
            "name": "Sql Column Presence",
            "config": {
                "cols": cols
            }
        }

    @staticmethod
    def two_words() -> Dict[str, Any]:
        """Template for two words guardrail"""
        return {
            "name": "Two Words",
            "config": {}
        }

    @staticmethod
    def upper_case() -> Dict[str, Any]:
        """Template for upper case guardrail"""
        return {
            "name": "Upper Case",
            "config": {}
        }

    @staticmethod
    def valid_choices(choices: List[str]) -> Dict[str, Any]:
        """Template for valid choices guardrail"""
        return {
            "name": "Valid Choices",
            "config": {
                "choices": choices
            }
        }

    @staticmethod
    def valid_json() -> Dict[str, Any]:
        """Template for valid json guardrail"""
        return {
            "name": "Valid Json",
            "config": {}
        }

    @staticmethod
    def valid_length(min: Optional[int] = None, max: Optional[int] = None) -> Dict[str, Any]:
        """Template for valid length guardrail"""
        config = {}
        if min is not None:
            config['min'] = min
        if max is not None:
            config['max'] = max
        return {
            "name": "Valid Length",
            "config": config
        }

    @staticmethod
    def valid_range(min: Optional[int] = None, max: Optional[int] = None) -> Dict[str, Any]:
        """Template for valid range guardrail"""
        config = {}
        if min is not None:
            config['min'] = min
        if max is not None:
            config['max'] = max
        return {
            "name": "Valid Range",
            "config": config
        }

    @staticmethod
    def valid_url() -> Dict[str, Any]:
        """Template for valid url guardrail"""
        return {
            "name": "Valid URL",
            "config": {}
        }
    
    @staticmethod
    def web_sanitization() -> Dict[str, Any]:
        """Template for web sanitization guardrail"""
        return {
            "name": "Web Sanitization",
            "config": {}
        }