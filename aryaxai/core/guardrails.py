from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict, Union

from aryaxai.client.client import APIClient
from aryaxai.common.environment import Environment
from aryaxai.common.xai_uris import (
    AVAILABLE_GUARDRAILS_URI,
    CONFIGURE_GUARDRAILS_URI,
    DELETE_GUARDRAILS_URI,
    GET_GUARDRAILS_URI,
    LOGIN_URI,
    UPDATE_GUARDRAILS_STATUS_URI,
    RUN_GUARDRAILS_URI,
)
from aryaxai.core.project import Project
from openinference.instrumentation.langchain import get_current_span
from opentelemetry import trace


class GuardrailTemplate:
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
        project: Optional[Project] = None,
        client: Optional[APIClient] = None,
        project_name: Optional[str] = None,
        default_apply_on: str = "input",
    ) -> None:
        if project is not None:
            self.client = project.api_client
            self.project_name = project.project_name
        else:
            self.client = client
            self.project_name = project_name

        self.default_apply_on = default_apply_on

        self.logs: List[Dict[str, Any]] = []
        self.max_retries = 1
        self.retry_delay = 1.0 
        self.tracer = trace.get_tracer(__name__)

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
        current_content = content

        if not guards:
            return current_content

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

        return current_content

    def _apply_guardrail_with_retry(
        self,
        content: Any,
        guard_spec: Dict[str, Any],
        node_name: str,
        content_type: str,
        action: str,
    ) -> Any:
        """Apply a single guardrail with retry logic based on guardrail response action"""
        current_content = content
        retry_count = 0
        
        while retry_count <= self.max_retries:
            run_result = self._call_run_guardrail(current_content, guard_spec , content_type)
            # Handle the action based on guardrail result
            current_content = self._handle_action(
                original=content,
                run_result=run_result,
                action=action,
                node_name=node_name,
                content_type=content_type,
                guard_name=guard_spec.get("name") or guard_spec.get("class", "unknown"),
            )
            
            # Check if we need to retry based on guardrail response
            if run_result.get("on_fail_action") == "retry" and retry_count < self.max_retries:
                retry_count += 1
                if retry_count <= self.max_retries:
                    import time
                    time.sleep(self.retry_delay)
                    continue
            
            # No retry needed or max retries reached
            break
        
        return current_content

    # --------- HTTP calls ---------
    def _call_run_guardrail(self, input_data: Any, guard: Dict[str, Any] , content_type :Any) -> GuardrailRunResult:
        # Try different possible endpoints
        
        uri=RUN_GUARDRAILS_URI
        
        if content_type == "output":
            input = input_data[1].content
        else:
            input = input_data[0].content
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
                raise ValueError("Guardrail '{guard_name}' detected an issue in {content_type}. Operation blocked.")

            elif on_fail_action == "sanitize":
                if parent_span is not None:
                    try:
                        ctx = trace.set_span_in_context(parent_span)
                        with self.tracer.start_as_current_span(f"guardrail: {guard_name}", context=ctx) as gr_span:
                            gr_span.set_attribute("component", str(node_name))
                            gr_span.set_attribute("guard", str(guard_name))
                            gr_span.set_attribute("content_type", str(content_type))
                            gr_span.set_attribute("detected", True)
                            gr_span.set_attribute("action", "sanitize")
                            gr_span.set_attribute("start_time", str(run_result.get("start_time", "")))
                            gr_span.set_attribute("end_time", str(run_result.get("end_time", "")))
                            gr_span.set_attribute("duration", float(run_result.get("duration", 0.0)))
                            gr_span.set_attribute("input.value", self._safe_str(run_result.get("input")))
                            gr_span.set_attribute("output.value", self._safe_str(run_result.get("response")))
                    except Exception:
                        pass
                return sanitized_output if sanitized_output is not None else original

            elif on_fail_action == "retry":
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
                # Return original for retry logic to handle
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
                    parts.append(GuardrailTemplate._safe_str(item) if hasattr(GuardrailTemplate, "_safe_str") else str(item))
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
