from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypedDict, Union
from aryaxai.common.xai_uris import RUN_GUARDRAILS_URI
from aryaxai.core.project import Project
from opentelemetry import trace
import time
from .guard_template import Guard
from dataclasses import dataclass

from agents import (
    Agent,
    RunContextWrapper,
    TResponseInputItem,
    input_guardrail,
    output_guardrail,
    ModelSettings,
    ModelTracing,
)

@dataclass
class GuardrailFunctionOutput:
    """The output of a guardrail function."""

    output_info: Any
    """
    Optional information about the guardrail's output. For example, the guardrail could include
    information about the checks it performed and granular results.
    """

    tripwire_triggered: bool
    """
    Whether the tripwire was triggered. If triggered, the agent's execution will be halted.
    """

    sanitized_content : str




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


class OpenAIAgentsGuardrail:
    """
    Utility for creating OpenAI Agents guardrails that call Guardrails HTTP APIs.
    
    Supports different actions:
    - "block": raises InputGuardrailTripwireTriggered/OutputGuardrailTripwireTriggered
    - "retry": attempts to sanitize content using LLM and passes sanitized content to handoff agent
    - "warn": logs the issue but allows content to pass through
    """

    def __init__(
        self,
        project: Optional[Project],
        model: Optional[Any] = None,
    ) -> None:
        if project is not None:
            self.client = project.api_client
            self.project_name = project.project_name
            
        self.logs: List[Dict[str, Any]] = []
        self.max_retries = 2
        self.retry_delay = 1.0 
        self.tracer = trace.get_tracer(__name__)
        self.model = model

    def create_input_guardrail(
        self,
        guards: Union[List[str], List[Dict[str, Any]], str, Dict[str, Any]],
        action: str = "block",
        name: str = "input_guardrail"
    ) -> Callable:
        """
        Create an input guardrail function for OpenAI Agents.
        
        Args:
            guards: List of guard specifications or single guard
            action: 'block' | 'retry' | 'warn'
            name: Name for the guardrail function
        """
        if isinstance(guards, (str, dict)):
            guards = [guards]

        @input_guardrail
        async def guardrail_function(
            ctx: RunContextWrapper[None], 
            agent: Agent, 
            input: str | list[TResponseInputItem]
        ) -> GuardrailFunctionOutput:
            # Convert input to string for processing
            if isinstance(input, list):
                # Handle list of input items (messages)
                input_text = ""
                for item in input:
                    if hasattr(item, 'content'):
                        input_text += str(item.content) + " "
                    else:
                        input_text += str(item) + " "
                input_text = input_text.strip()
            else:
                input_text = str(input)

            # Process through all guards
            current_content = input_text
            tripwire_triggered = False
            output_info = {}

            for guard in guards:
                if isinstance(guard, str):
                    guard_spec: Dict[str, Any] = {"name": guard}
                else:
                    guard_spec = dict(guard)

                try:
                    current_content, is_triggered, info = await self._apply_guardrail_with_retry(
                        content=current_content,
                        guard_spec=guard_spec,
                        guardrail_type="input",
                        action=action,
                        agent_name=agent.name,
                    )
                    
                    if is_triggered:
                        tripwire_triggered = True
                        output_info.update(info)
                        
                        if action == "block":
                            break  # Stop processing on first failure for block action
                            
                except Exception as e:
                    # Log error but don't fail the guardrail
                    output_info[f"error_{guard_spec.get('name', 'unknown')}"] = str(e)

            return GuardrailFunctionOutput(
                output_info=output_info,
                tripwire_triggered=tripwire_triggered,
                # Add sanitized content to be passed to handoff agent
                sanitized_content=current_content if action == "retry" else input_text
            )

        # Set function name for debugging
        guardrail_function.__name__ = name
        return guardrail_function

    def create_output_guardrail(
        self,
        guards: Union[List[str], List[Dict[str, Any]], str, Dict[str, Any]],
        action: str = "block",
        name: str = "output_guardrail"
    ) -> Callable:
        """
        Create an output guardrail function for OpenAI Agents.
        
        Args:
            guards: List of guard specifications or single guard
            action: 'block' | 'retry' | 'warn'
            name: Name for the guardrail function
        """
        if isinstance(guards, (str, dict)):
            guards = [guards]

        @output_guardrail
        async def guardrail_function(
            ctx: RunContextWrapper, 
            agent: Agent, 
            output: Any
        ) -> GuardrailFunctionOutput:
            # Extract text content from output
            if hasattr(output, 'response'):
                output_text = str(output.response)
            elif hasattr(output, 'content'):
                output_text = str(output.content)
            else:
                output_text = str(output)

            # Process through all guards
            current_content = output_text
            tripwire_triggered = False
            output_info = {}

            for guard in guards:
                if isinstance(guard, str):
                    guard_spec: Dict[str, Any] = {"name": guard}
                else:
                    guard_spec = dict(guard)

                try:
                    current_content, is_triggered, info = await self._apply_guardrail_with_retry(
                        content=current_content,
                        guard_spec=guard_spec,
                        guardrail_type="output",
                        action=action,
                        agent_name=agent.name,
                    )
                    
                    if is_triggered:
                        tripwire_triggered = True
                        output_info.update(info)
                        
                        if action == "block":
                            break  # Stop processing on first failure for block action
                            
                except Exception as e:
                    # Log error but don't fail the guardrail
                    output_info[f"error_{guard_spec.get('name', 'unknown')}"] = str(e)

            return GuardrailFunctionOutput(
                output_info=output_info,
                tripwire_triggered=tripwire_triggered,
                # Add sanitized content to be passed to handoff agent
                sanitized_content=current_content if action == "retry" else output_text
            )

        # Set function name for debugging
        guardrail_function.__name__ = name
        return guardrail_function

    async def _apply_guardrail_with_retry(
        self,
        content: Any,
        guard_spec: Dict[str, Any],
        guardrail_type: str,
        action: str,
        agent_name: str,
    ) -> tuple[Any, bool, Dict[str, Any]]:
        """
        Apply a single guardrail with retry logic.
        
        Returns:
            tuple: (processed_content, tripwire_triggered, output_info)
        """
        current_content = content
        retry_count = 0
        output_info = {}

        if action == "retry":
            while retry_count <= self.max_retries:
                run_result = await self._call_run_guardrail(current_content, guard_spec, guardrail_type)
                validation_passed = bool(run_result.get("validation_passed", True))
                detected_issue = not validation_passed or not run_result.get("success", True)
                if detected_issue and self.model is not None and retry_count < self.max_retries:
                    prompt = self._build_sanitize_prompt(
                        guard_spec.get("name", "unknown"), 
                        current_content, 
                        guardrail_type
                    )
                    try:
                        sanitized = await self._invoke_llm(prompt)
                        current_content = sanitized
                    except Exception:
                        pass  # Keep original content if sanitization fails
                    
                    self._log_guardrail_result(
                        run_result=run_result,
                        action=f"retry",
                        agent_name=agent_name,
                        guardrail_type=guardrail_type,
                        guard_name=guard_spec.get("name", "unknown"),
                    )
                    
                    retry_count += 1
                    await self._async_sleep(self.retry_delay)
                    continue
                else:
                    # Final attempt or successful validation
                    self._log_guardrail_result(
                        run_result=run_result,
                        action=f"retry" if retry_count > 0 else action,
                        agent_name=agent_name,
                        guardrail_type=guardrail_type,
                        guard_name=guard_spec.get("name", "unknown"),
                    )
                    
                    output_info.update({
                        "guard_name": guard_spec.get("name", "unknown"),
                        "detected_issue": detected_issue,
                        "retry_count": retry_count,
                        "final_content": current_content,
                    })
                    
                    return current_content, detected_issue, output_info
        else:
            # Single check for non-retry actions
            run_result = await self._call_run_guardrail(current_content, guard_spec, guardrail_type)
            validation_passed = bool(run_result.get("validation_passed", True))
            detected_issue = not validation_passed or not run_result.get("success", True)
            
            self._log_guardrail_result(
                run_result=run_result,
                action=action,
                agent_name=agent_name,
                guardrail_type=guardrail_type,
                guard_name=guard_spec.get("name", "unknown"),
            )
            
            output_info.update({
                "guard_name": guard_spec.get("name", "unknown"),
                "detected_issue": detected_issue,
                "action": action,
            })
            
            # For warn action, don't trigger tripwire even if issue detected
            if action == "warn":
                detected_issue = False
            
            return current_content, detected_issue, output_info

    async def _call_run_guardrail(
        self, 
        input_data: Any, 
        guard: Dict[str, Any], 
        guardrail_type: str
    ) -> GuardrailRunResult:
        """Call the guardrails HTTP API"""
        uri = RUN_GUARDRAILS_URI
        input_text = str(input_data)
        
        start_time = datetime.now()
        try:
            body = {"input_data": input_text, "guard": guard}

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
            result["input"] = input_text

            return result
            
        except Exception as exc:
            end_time = datetime.now()
            raise exc

    def _log_guardrail_result(
        self,
        run_result: GuardrailRunResult,
        action: str,
        agent_name: str,
        guardrail_type: str,
        guard_name: str,
    ) -> None:
        """Log guardrail results with tracing"""
        validation_passed = bool(run_result.get("validation_passed", True))
        detected_issue = not validation_passed or not run_result.get("success", True)
        try:
            parent_span = trace.get_current_span()
        except Exception:
            parent_span = None

        if parent_span is not None:
            try:
                ctx = trace.set_span_in_context(parent_span)
                with self.tracer.start_as_current_span(f"guardrail: {guard_name}", context=ctx) as gr_span:
                    gr_span.set_attribute("component", str(agent_name))
                    gr_span.set_attribute("guard", str(guard_name))
                    gr_span.set_attribute("guardrail_type", str(guardrail_type))
                    gr_span.set_attribute("detected", detected_issue)
                    gr_span.set_attribute("action", action)
                    gr_span.set_attribute("start_time", str(run_result.get("start_time", "")))
                    gr_span.set_attribute("end_time", str(run_result.get("end_time", "")))
                    gr_span.set_attribute("duration", float(run_result.get("duration", 0.0)))
                    gr_span.set_attribute("input.value", self._safe_str(run_result.get("input")))
                    gr_span.set_attribute("output.value", self._safe_str(run_result.get("response")))
            except Exception:
                pass

    def _build_sanitize_prompt(self, guard_name: str, content: Any, guardrail_type: str) -> str:
        """Build a prompt for the LLM to sanitize the content according to the guardrail type"""
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

    async def _invoke_llm(self, prompt: str) -> str:
        """Invoke the LLM for content sanitization"""
        if self.model is None:
            return prompt  # Return original if no LLM available
            
        try:
            data = await self.model.get_response(system_instructions="Based on the input you have to provide the best and accurate results",
                             input=prompt,
                             model_settings=ModelSettings(temperature=0.1),
                             tools=[],
                             output_schema=None,
                             handoffs=[],
                             tracing=ModelTracing.DISABLED,
                             previous_response_id=None)
            return str(data.output[0].content[0].text)
        except Exception:
            return prompt  # Return original on error

    async def _async_sleep(self, seconds: float) -> None:
        """Async sleep utility"""
        import asyncio
        await asyncio.sleep(seconds)

    @staticmethod
    def _safe_str(value: Any) -> str:
        """Safely convert any value to string for logging"""
        try:
            if isinstance(value, (str, int, float, bool)) or value is None:
                return str(value)
            if hasattr(value, "content"):
                return str(getattr(value, "content", ""))
            
            if isinstance(value, (list, tuple)):
                parts = []
                for item in value:
                    parts.append(OpenAIAgentsGuardrail._safe_str(item))
                return ", ".join(parts)
            
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
                return json.dumps(safe_dict, ensure_ascii=False)
            
            return str(value)
        except Exception:
            return "<unserializable>"


# Convenience function for quick guardrail setup
def create_guardrail(project: Project, model: Optional[Any] = None) -> OpenAIAgentsGuardrail:
    """Quick factory function to create a guardrail instance with a project"""
    return OpenAIAgentsGuardrail(project=project, model=model)