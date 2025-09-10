from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict, Union
from aryaxai.common.xai_uris import (
    RUN_GUARDRAILS_URI,
)
from aryaxai.core.project import Project
from openinference.instrumentation.langchain import get_current_span
from opentelemetry import trace
import time


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

class GuardrailRunResult(TypedDict, total=False):
    success: bool
    details: Dict[str, Any]
    validated_output: Any
    validation_passed: bool
    sanitized_output: Any
    duration: float
    latency: str
    on_fail_action: str
    retry_count: int
    max_retries: int
    start_time: str
    end_time: str


class LangGraphGuardrail:
    """
    Decorator utility for applying Guardrails checks to LangGraph node inputs and outputs
    by calling the Guardrails HTTP APIs.

    Supports two modes:
    - "adhoc": calls /guardrails/run_guardrail per guard passed in the decorator
    - "configured": calls /guardrails/run to use project/model configured guardrails
    """

    def __init__(
        self,
        project: Optional[Project],
        default_apply_on: str = "input",
        llm: Optional[Any] = None,  # Add LLM parameter
    ) -> None:
        if project is not None:
            self.client = project.api_client
            self.project_name = project.project_name
            
        self.default_apply_on = default_apply_on

        self.logs: List[Dict[str, Any]] = []
        self.max_retries = 1
        self.retry_delay = 1.0 
        self.tracer = trace.get_tracer(__name__)
        self.llm = llm

    def guardrail(
        self,
        guards: Union[List[str], List[Dict[str, Any]], str, Dict[str, Any], None] = None,
        action: str = "block",
        apply_to: str = "both",
        apply_on: Optional[str] = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Decorator factory.

        - action: 'block' | 'retry' | 'warn'. If validation fails:
          - block: raise ValueError
          - retry: replace with sanitized output when available
          - warn: keep content, log only
        - apply_to: 'input' | 'output' | 'both'
        - apply_on (configured mode): 'input' | 'output'; defaults to self.default_apply_on
        """

        if isinstance(guards, (str, dict)):
            guards = [guards]  # type: ignore[assignment]

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                # In LangGraph, the first argument is the state dict
                state = args[0] if args else kwargs.get("state")
                node_name = getattr(func, "__name__", "unknown_node")

                # Pre-process input
                if state and apply_to in ("input", "both"):
                    # Check if state has messages (LangGraph format)
                    if "messages" in state:
                        state["messages"] = self._process_content(
                            content=state["messages"],
                            node_name=node_name,
                            content_type="input",
                            action=action,
                            guards=guards,
                            apply_on=(apply_on or self.default_apply_on),
                        )
                    # Also check for direct input field
                    elif "input" in state:
                        state["input"] = self._process_content(
                            content=state["input"],
                            node_name=node_name,
                            content_type="input",
                            action=action,
                            guards=guards,
                            apply_on=(apply_on or self.default_apply_on),
                        )

                result = func(*args, **kwargs)

                # Post-process output
                if apply_to in ("output", "both"):
                    if isinstance(result, dict) and "messages" in result:
                        result_output = result["messages"]
                        result["messages"] = self._process_content(
                            content=result_output,
                            node_name=node_name,
                            content_type="output",
                            action=action,
                            guards=guards,
                            apply_on=(apply_on or self.default_apply_on),
                        )
                    elif isinstance(result, str):
                        result = self._process_content(
                            content=result,
                            node_name=node_name,
                            content_type="output",
                            action=action,
                            guards=guards,
                            apply_on=(apply_on or self.default_apply_on),
                        )

                return result

            return wrapper

        return decorator

    def _process_content(
        self,
        content: Any,
        node_name: str,
        content_type: str,
        action: str,
        guards: Optional[List[Union[str, Dict[str, Any]]]],
        apply_on: str,
    ) -> Any:
        if not guards:
            return content

        is_list = isinstance(content, list)
        if is_list and content:
            content_to_process = content[-1].content
        elif isinstance(content, str):
            content_to_process = content
        else:
            # If not list or str, return as is
            return content

        current_content = content_to_process
        for guard in guards:
            if isinstance(guard, str):
                guard_spec: Dict[str, Any] = {"name": guard}
            else:
                guard_spec = dict(guard)

            current_content = self._apply_guardrail_with_retry(
                content=current_content,
                guard_spec=guard_spec,
                node_name=node_name,
                content_type=content_type,
                action=action,
            )

        if is_list:
            content[-1].content = current_content
            return content
        else:
            return current_content

    def _apply_guardrail_with_retry(
        self,
        content: Any,
        guard_spec: Dict[str, Any],
        node_name: str,
        content_type: str,
        action: str,
    ) -> Any:
        current_content = content
        retry_count = 0

        if action == "retry":
            while retry_count <= self.max_retries:
                run_result = self._call_run_guardrail(current_content, guard_spec, content_type)
                validation_passed = bool(run_result.get("validation_passed", True))
                detected_issue = not validation_passed or not run_result.get("success", True)

                if detected_issue and self.llm is not None and retry_count < self.max_retries:
                    prompt = self._build_sanitize_prompt(guard_spec.get("name", "unknown"), current_content, content_type)
                    try:
                        sanitized = self.llm.invoke(prompt)
                        if hasattr(sanitized, "content"):
                            sanitized = sanitized.content
                    except Exception:
                        sanitized = current_content
                    retry_action = f"retry_{retry_count+1}"
                    self._handle_action(
                        original=current_content,
                        run_result=run_result,
                        action=retry_action,
                        node_name=node_name,
                        content_type=content_type,
                        guard_name=guard_spec.get("name") or guard_spec.get("class", "unknown"),
                    )
                    current_content = sanitized
                    retry_count += 1
                    time.sleep(self.retry_delay)
                    continue
                else:
                    return self._handle_action(
                        original=current_content,
                        run_result=run_result,
                        action=f"retry_{retry_count}" if retry_count > 0 else action,
                        node_name=node_name,
                        content_type=content_type,
                        guard_name=guard_spec.get("name") or guard_spec.get("class", "unknown"),
                    )
            return current_content
        else:
            # Only one check for non-retry actions
            run_result = self._call_run_guardrail(current_content, guard_spec, content_type)
            return self._handle_action(
                original=current_content,
                run_result=run_result,
                action=action,
                node_name=node_name,
                content_type=content_type,
                guard_name=guard_spec.get("name") or guard_spec.get("class", "unknown"),
            )

    # --------- HTTP calls ---------
    def _call_run_guardrail(self, input_data: Any, guard: Dict[str, Any] , content_type :Any) -> GuardrailRunResult:
        # Try different possible endpoints
        
        uri=RUN_GUARDRAILS_URI
        
        input = input_data
        
        start_time = datetime.now()
        try:
            body = {"input_data": input, "guard": guard}

            data = self.client.post(uri, body)
            
            end_time = datetime.now()
            
            details = data.get("details", {}) if isinstance(data, dict) else {}
            result: GuardrailRunResult = {
                "success": bool(data.get("success", False)) if isinstance(data, dict) else False,
                "details": details if isinstance(details, dict) else {},
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
            }
            if "duration" not in details:
                result["duration"] = (end_time - start_time).total_seconds()
            if isinstance(details, dict):
                if "validated_output" in details:
                    result["validated_output"] = details["validated_output"]
                if "validation_passed" in details:
                    result["validation_passed"] = details["validation_passed"]
                if "sanitized_output" in details:
                    result["sanitized_output"] = details["sanitized_output"]
                if "duration" in details:
                    result["duration"] = details["duration"]
                if "latency" in details:
                    result["latency"] = details["latency"]
                
                result["retry_count"] = 0
                result["max_retries"] = self.max_retries

            result["response"] = data
            result["input"] = input

            return result
            
        except Exception as exc:
            end_time = datetime.now()  # Still capture end time on exception
            raise exc

    # --------- Action handling ---------
    def _handle_action(
        self,
        original: Any,
        run_result: GuardrailRunResult,
        action: str,
        node_name: str,
        content_type: str,
        guard_name: str,
    ) -> Any:
        validation_passed = bool(run_result.get("validation_passed", True))
        detected_issue = not validation_passed or not run_result.get("success", True)
        sanitized_output = run_result.get("sanitized_output")


        # Minimal event attributes (keep events clean)

        if detected_issue:
            on_fail_action = action
            try:
                parent_span = get_current_span()
            except Exception:
                parent_span = None

            if on_fail_action == "block":
                if parent_span is not None:
                    try:
                        ctx = trace.set_span_in_context(parent_span)
                        with self.tracer.start_as_current_span(f"guardrail: {guard_name}", context=ctx) as gr_span:
                            # Span-level attributes (timing, status, values)
                            gr_span.set_attribute("component", str(node_name))
                            gr_span.set_attribute("guard", str(guard_name))
                            gr_span.set_attribute("content_type", str(content_type))
                            gr_span.set_attribute("detected", True)
                            gr_span.set_attribute("action", "block")
                            gr_span.set_attribute("start_time", str(run_result.get("start_time", "")))
                            gr_span.set_attribute("end_time", str(run_result.get("end_time", "")))
                            gr_span.set_attribute("duration", float(run_result.get("duration", 0.0)))
                            gr_span.set_attribute("input.value", self._safe_str(run_result.get("input")))
                            gr_span.set_attribute("output.value", self._safe_str(run_result.get("response")))
                    except Exception:
                        pass
                # Block action: raise after recording the event
                raise ValueError(f"Guardrail '{guard_name}' detected an issue in {content_type}. Operation blocked.")

            elif "retry" in on_fail_action:
                if parent_span is not None:
                    try:
                        ctx = trace.set_span_in_context(parent_span)
                        with self.tracer.start_as_current_span(f"guardrail: {guard_name}", context=ctx) as gr_span:
                            gr_span.set_attribute("component", str(node_name))
                            gr_span.set_attribute("guard", str(guard_name))
                            gr_span.set_attribute("content_type", str(content_type))
                            gr_span.set_attribute("detected", True)
                            gr_span.set_attribute("action", "retry")
                            gr_span.set_attribute("start_time", str(run_result.get("start_time", "")))
                            gr_span.set_attribute("end_time", str(run_result.get("end_time", "")))
                            gr_span.set_attribute("duration", float(run_result.get("duration", 0.0)))
                            gr_span.set_attribute("input.value", self._safe_str(run_result.get("input")))
                            gr_span.set_attribute("output.value", self._safe_str(run_result.get("response")))
                    except Exception:
                        pass
                # Return sanitized content for retry
                return original

            else:  # default or warn: keep content, log only
                if parent_span is not None:
                    try:
                        ctx = trace.set_span_in_context(parent_span)
                        with self.tracer.start_as_current_span(f"guardrail: {guard_name}", context=ctx) as gr_span:
                            start_time = run_result.get("start_time", "")
                            end_time = run_result.get("end_time", "")
                            gr_span.set_attribute("component", str(node_name))
                            gr_span.set_attribute("guard", str(guard_name))
                            gr_span.set_attribute("content_type", str(content_type))
                            gr_span.set_attribute("detected", True)
                            gr_span.set_attribute("action", "warn")
                            gr_span.set_attribute("start_time", start_time)
                            gr_span.set_attribute("end_time", end_time)
                            gr_span.set_attribute("duration", float(run_result.get("duration", 0.0)))
                            gr_span.set_attribute("input.value", self._safe_str(run_result.get("input")))
                            gr_span.set_attribute("output.value", self._safe_str(run_result.get("response")))
                    except Exception:
                        pass
                return original

        try:
            parent_span = get_current_span()
        except Exception:
            parent_span = None
        if parent_span is not None:
            try:
                ctx = trace.set_span_in_context(parent_span)
                with self.tracer.start_as_current_span(f"guardrail: {guard_name}", context=ctx) as gr_span:
                    gr_span.set_attribute("component", str(node_name))
                    gr_span.set_attribute("guard", str(guard_name))
                    gr_span.set_attribute("content_type", str(content_type))
                    gr_span.set_attribute("detected", False)
                    gr_span.set_attribute("action", "passed")
                    gr_span.set_attribute("start_time", str(run_result.get("start_time", "")))
                    gr_span.set_attribute("end_time", str(run_result.get("end_time", "")))
                    gr_span.set_attribute("duration", float(run_result.get("duration", 0.0)))
                    gr_span.set_attribute("input.value", self._safe_str(run_result.get("input")))
                    gr_span.set_attribute("output.value", self._safe_str(run_result.get("response")))
            except Exception:
                pass
        return original

    def _build_sanitize_prompt(self, guard_name: str, content: Any, content_type: str) -> str:
        """
        Build a prompt for the LLM to sanitize the content according to the guardrail type.
        The prompt instructs the LLM to only sanitize, not change anything else.
        """
        instructions = {
            "Detect PII": "Sanitize the following text by removing or masking any personally identifiable information (PII). Do not change anything else.",
            "NSFW Text": "Sanitize the following text by removing or masking any not safe for work (NSFW) content. Do not change anything else.",
            "Ban List": "Sanitize the following text by removing or masking any banned words. Do not change anything else.",
            "Bias Check": "Sanitize the following text by removing or masking any biased language. Do not change anything else.",
            "Competitor Check": "Sanitize the following text by removing or masking any competitor names. Do not change anything else.",
            "Correct Language": "Sanitize the following text by correcting the language to the expected language. Do not change anything else.",
            "Gibberish Text": "Sanitize the following text by removing or correcting any gibberish. Do not change anything else.",
            "Profanity Free": "Sanitize the following text by removing or masking any profanity. Do not change anything else.",
            "Secrets Present": "Sanitize the following text by removing or masking any secrets. Do not change anything else.",
            "Toxic Language": "Sanitize the following text by removing or masking any toxic language. Do not change anything else.",
            "Contains String": "Sanitize the following text by removing or masking the specified substring. Do not change anything else.",
            "Detect Jailbreak": "Sanitize the following text by removing or masking any jailbreak attempts. Do not change anything else.",
            "Endpoint Is Reachable": "Sanitize the following text by ensuring any mentioned endpoints are reachable. Do not change anything else.",
            "Ends With": "Sanitize the following text by ensuring it ends with the specified string. Do not change anything else.",
            "Has Url": "Sanitize the following text by removing or masking any URLs. Do not change anything else.",
            "Lower Case": "Sanitize the following text by converting it to lower case. Do not change anything else.",
            "Mentions Drugs": "Sanitize the following text by removing or masking any mentions of drugs. Do not change anything else.",
            "One Line": "Sanitize the following text by ensuring it is a single line. Do not change anything else.",
            "Reading Time": "Sanitize the following text by ensuring its reading time matches the specified value. Do not change anything else.",
            "Redundant Sentences": "Sanitize the following text by removing redundant sentences. Do not change anything else.",
            "Regex Match": "Sanitize the following text by ensuring it matches the specified regex. Do not change anything else.",
            "Sql Column Presence": "Sanitize the following text by ensuring specified SQL columns are present. Do not change anything else.",
            "Two Words": "Sanitize the following text by ensuring it contains only two words. Do not change anything else.",
            "Upper Case": "Sanitize the following text by converting it to upper case. Do not change anything else.",
            "Valid Choices": "Sanitize the following text by ensuring it matches one of the valid choices. Do not change anything else.",
            "Valid Json": "Sanitize the following text by ensuring it is valid JSON. Do not change anything else.",
            "Valid Length": "Sanitize the following text by ensuring its length is valid. Do not change anything else.",
            "Valid Range": "Sanitize the following text by ensuring its value is within the valid range. Do not change anything else.",
            "Valid URL": "Sanitize the following text by ensuring it is a valid URL. Do not change anything else.",
            "Web Sanitization": "Sanitize the following text by removing any unsafe web content. Do not change anything else.",
        }
        instruction = instructions.get(guard_name, "Sanitize the following text according to the guardrail requirements. Do not change anything else.")
        prompt = f"{instruction}\n\nContent:\n{content}"
        return prompt

    @staticmethod
    def _safe_str(value: Any) -> str:
        try:
            if isinstance(value, (str, int, float, bool)) or value is None:
                s = str(value)
                return s
            if hasattr(value, "content"):
                s = str(getattr(value, "content", ""))
                return s
            
            if isinstance(value, (list, tuple)):
                parts = []
                for item in value:
                    parts.append(Guard._safe_str(item) if hasattr(Guard, "_safe_str") else str(item))
                s = ", ".join(parts)
                return s
            
            if isinstance(value, dict):
                safe_dict: Dict[str, Any] = {}
                for k, v in value.items():
                    key = str(k)
                    if isinstance(v, (str, int, float, bool)) or v is None:
                        safe_dict[key] = v
                    elif hasattr(v, "content"):
                        safe_dict[key] = str(getattr(v, "content", ""))
                    else:
                        safe_dict[key] = str(v)
                s = json.dumps(safe_dict, ensure_ascii=False)
                return s
            s = str(value)
            return s
        except Exception:
            return "<unserializable>"

# Convenience function for quick guardrail setup
def create_guardrail(project: Project) -> LangGraphGuardrail:
    """Quick factory function to create a guardrail instance with a project"""
    return LangGraphGuardrail(project=project)