import json
from typing import Dict, Any, Optional, Callable, List, Union
from datetime import datetime
from autogen import ConversableAgent, UserProxyAgent
from autogen_agentchat.agents import AssistantAgent
import time
from aryaxai.core.project import Project
from aryaxai.common.xai_uris import (
    RUN_GUARDRAILS_URI,
)
from opentelemetry import trace, context
from .guard_template import Guard


class GuardrailRunResult(Dict[str, Any]):
    pass  # TypedDict not needed for runtime, but can add if desired

class GuardrailSupervisor:
    """Pluggable class to monitor and control agent behavior using API-based guardrails."""
    def __init__(self, 
                 guards: Union[List[Dict[str, Any]], Dict[str, Any], None] = None,
                 apply_to: str = 'both',
                 action: str = "block",
                 project: Optional[Project] = None,
                 llm: Optional[Any] = None, 
                 ):
        if apply_to not in ['input', 'output', 'both']:
            raise ValueError("apply_to must be one of 'input', 'output', 'both'")
        self.apply_to = apply_to
        if isinstance(guards, dict):
            guards = [guards]
        self.guards = guards or []
        if action not in ['block', 'retry', 'warn']:
            raise ValueError("action must be one of 'block', 'retry', 'warn'")
        self.action = action
        if project is not None:
            self.api_client = project.api_client
            self.project_name = project.project_name
        self.llm = llm
        self.max_retries = 1
        self.retry_delay = 1.0
        self.tracer = trace.get_tracer("autogen-app")  # Standardized tracer name


    def _process_content(
        self,
        content: str,
        agent_id: str,
        content_type: str,
        action: str,
        guards: List[Dict[str, Any]],
    ) -> str:
        if not guards:
            return content

        current_content = content
        for guard in guards:
            if isinstance(guard, str):
                guard_spec: Dict[str, Any] = {"name": guard}
            else:
                guard_spec = dict(guard)

            current_content = self._apply_guardrail_with_retry(
                content=current_content,
                guard_spec=guard_spec,
                agent_id=agent_id,
                content_type=content_type,
                action=action,
            )

        return current_content

    def _apply_guardrail_with_retry(
        self,
        content: str,
        guard_spec: Dict[str, Any],
        agent_id: str,
        content_type: str,
        action: str,
    ) -> str:
        current_content = content
        retry_count = 0

        if action == "retry":
            while retry_count <= self.max_retries:
                run_result = self._call_run_guardrail(current_content, guard_spec, content_type)
                validation_passed = bool(run_result.get("validation_passed", True))
                detected_issue = not validation_passed or not run_result.get("success", True)

                if detected_issue and self.llm is not None and retry_count < self.max_retries:
                    retry_count += 1
                    time.sleep(self.retry_delay)
                    continue
                else:
                    return self._handle_action(
                        original=current_content,
                        run_result=run_result,
                        action=f"retry_{retry_count}" if retry_count > 0 else action,
                        agent_id=agent_id,
                        content_type=content_type,
                        guard_name=guard_spec.get("name", "unknown"),
                    )
        else:
            run_result = self._call_run_guardrail(current_content, guard_spec, content_type)
            return self._handle_action(
                original=current_content,
                run_result=run_result,
                action=action,
                agent_id=agent_id,
                content_type=content_type,
                guard_name=guard_spec.get("name", "unknown"),
            )

    def _call_run_guardrail(self, input_data: str, guard: Dict[str, Any], content_type: str) -> GuardrailRunResult:
        start_time = datetime.now()
        try:
            body = {"input_data": input_data, "guard": guard}
            data = self.api_client.post(RUN_GUARDRAILS_URI, body)
            # print(data , "api ran")
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
                result.update({k: v for k, v in details.items() if k in [
                    "validated_output", "validation_passed", "sanitized_output", "duration", "latency"
                ]})
            result["retry_count"] = 0
            result["max_retries"] = self.max_retries
            result["response"] = data
            result["input"] = input_data
            return result
        except Exception as exc:
            end_time = datetime.now()
            result = {
                "success": False,
                "details": {},
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration": (end_time - start_time).total_seconds(),
                "error": str(exc)
            }
            return result

    def _handle_action(
        self,
        original: str,
        run_result: GuardrailRunResult,
        action: str,
        agent_id: str,
        content_type: str,
        guard_name: str,
    ) -> str:
        validation_passed = bool(run_result.get("validation_passed", True))
        detected_issue = not validation_passed or not run_result.get("success", True)
        sanitized_output = run_result.get("sanitized_output")

        status = "passed" if not detected_issue else "failed"
        self._log_event(
            agent_id=agent_id,
            stage=content_type,
            data=original,
            status=status,
            error=run_result.get("error", ""),
            details={
                "guard": guard_name,
                "action": action,
                "detected": detected_issue,
                "duration": run_result.get("duration", 0.0),
                "input": self._safe_str(run_result.get("input")),
                "output": self._safe_str(run_result.get("response")),
            }
        )

        if detected_issue:
            on_fail_action = action
            if on_fail_action == "block":
                raise ValueError(f"Guardrail '{guard_name}' detected an issue in {content_type}. Operation blocked.")
            elif "retry" in on_fail_action:
                return original  # Return last content (would be sanitized if implemented)
            else:  # warn
                return original
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

    def instrument_agents(self, agents: List[Union[ConversableAgent, AssistantAgent]]) -> List[Union[ConversableAgent, AssistantAgent]]:
        """
        Instruments a list of agents to apply guardrails.

        This method iterates through a list of agents and applies the appropriate
        instrumentation to intercept their message generation or run methods.
        It handles different agent types like `AssistantAgent` and `ConversableAgent`.

        Args:
            agents: A list of agents to be instrumented.
        
        Returns:
            The list of instrumented agents.
        """
        for agent in agents:
            # It's important to check for the more specific subclass first.
            if isinstance(agent, AssistantAgent):
                self.instrument_agent(agent)
        return agents


    async def _execute_guarded_run(self, agent, original_run, args, kwargs, current_context):
        """Execute the guarded run with proper context for guardrails."""
        # Extract task argument and ensure proper argument handling
        task = kwargs.get('task', None)
        if task is None and len(args) > 0:
            task = args[0]

        # Process input for guardrails - make them direct children of current agent span
        if self.apply_to in ['input', 'both'] and task:
            request_content = self._extract_task_content(task)
            if request_content:
                # Set input content attribute on the current agent span
                current_span = trace.get_current_span()
                if current_span.is_recording():
                    current_span.set_attribute("guardrail.input_content", self._safe_str(request_content))
                
                self._apply_input_guardrails(
                    request_content, agent.name, current_context
                )

        # Call original run method with proper argument handling
        try:
            reply = await original_run(*args, **kwargs)
                
        except Exception as e:
            raise ValueError(f"Error generating response: {str(e)}")

        # Process output through guardrails - make them direct children of current agent span
        if self.apply_to in ['output', 'both'] and reply:
            response_content = self._extract_response_content(reply)
            if response_content:
                # Set output content attribute on the current agent span
                self._apply_output_guardrails(
                    response_content, agent.name, current_context
                )

        # Ensure reply has required format for AutoGen consistency
        reply = self._format_reply(reply, agent)
        
        return reply

    def instrument_agent(self, agent) -> None:
        """Wrap AssistantAgent to intercept run method for guardrails (AutoGen 0.4+)."""
        original_run = agent.run
        # print(f"Guardrail running on {agent.__class__.__name__}")

        async def wrapped_run(*args, **kwargs):
            # print("Guardrail intercepted run method")
            
            # Get the current span and context
            current_span = trace.get_current_span()
            current_context = context.get_current()
            
            # If no parent span is active, start one for the agent run
            if not current_span.is_recording():
                with self.tracer.start_as_current_span(f"{agent.name}_run") as agent_span:
                    # Update current_context to include the new parent
                    current_context = context.get_current()
                    return await self._execute_guarded_run(agent, original_run, args, kwargs, current_context)
            else:
                return await self._execute_guarded_run(agent, original_run, args, kwargs, current_context)

        # Replace the run method with proper binding
        agent.run = wrapped_run
        return agent

    def _extract_task_content(self, task) -> str:
        """Extract content from task parameter for processing."""
        if isinstance(task, str):
            return task
        elif isinstance(task, list):
            # Handle list of messages/tasks
            content_parts = []
            for item in task:
                if isinstance(item, dict):
                    # Handle message dict format
                    if item.get('role') == 'user':
                        content_parts.append(item.get('content', ''))
                    elif 'content' in item:
                        content_parts.append(item.get('content', ''))
                    else:
                        content_parts.append(str(item))
                elif hasattr(item, 'content'):
                    content_parts.append(str(item.content))
                elif hasattr(item, 'role') and hasattr(item, 'content'):
                    if item.role == 'user':
                        content_parts.append(str(item.content))
                else:
                    content_parts.append(str(item))
            return ' '.join(filter(None, content_parts))
        elif isinstance(task, dict):
            # Handle single message dict
            return task.get('content', str(task))
        elif hasattr(task, 'content'):
            return str(task.content)
        else:
            return str(task)

    def _extract_response_content(self, reply) -> str:
        """Extract content from agent response for guardrail processing."""
        if isinstance(reply, str):
            return reply
        elif isinstance(reply, dict):
            return reply.get("content", "")
        elif hasattr(reply, "content"):
            return str(reply.content)
        elif hasattr(reply, "text"):
            return str(reply.text)
        else:
            return str(reply)

    def _apply_input_guardrails(self, content: str, agent_name: str, ctx) -> None:
        """Apply input guardrails with telemetry tracking as direct children of agent span."""
        for guard in self.guards:
            guard_name = guard.get("name", "unknown")
            
            # Create guardrail span as direct child of the current agent execution span
            with self.tracer.start_as_current_span(f"guardrail: {guard_name}", context=ctx) as guard_span:
                # Set comprehensive attributes linking this guardrail to the specific agent
                guard_span.set_attribute("component", agent_name)
                guard_span.set_attribute("guard", guard_name)
                guard_span.set_attribute("content_type", "input")
                
                try:
                    start_time = datetime.now()
                    run_result = self._call_run_guardrail(content, guard, "input")
                    end_time = datetime.now()
                    
                    # Set comprehensive telemetry attributes
                    guard_span.set_attribute("input.value", self._safe_str(content))
                    guard_span.set_attribute("output.value", 
                                        self._safe_str(run_result.get("response", "")))
                    guard_span.set_attribute("start_time", start_time.isoformat())
                    guard_span.set_attribute("end_time", end_time.isoformat())
                    guard_span.set_attribute("duration", 
                                        (end_time - start_time).total_seconds())
                    
                    # Check validation results
                    validation_passed = bool(run_result.get("validation_passed", True))
                    success = bool(run_result.get("success", True))
                    detected_issue = not validation_passed or not success
                    
                    guard_span.set_attribute("detected", detected_issue)
                    
                    # Handle guardrail violations based on your policy
                    if detected_issue:
                        guard_span.set_attribute("action", self.action)
                        error_msg = run_result.get("error_message", 
                                                f"Input guardrail '{guard_name}' detected an issue for agent '{agent_name}'")
                        guard_span.add_event("input_guardrail_violation", {
                            "agent": agent_name,
                            "guard": guard_name,
                            "error_message": error_msg
                        })
                        guard_span.set_attribute("violation.message", error_msg)
                        guard_span.set_attribute("violation.agent", agent_name)
                        # print(f"Input guardrail violation on {agent_name}: {error_msg}")
                        # Uncomment if you want to raise exceptions on violations:
                        # raise ValueError(error_msg)
                    else:
                        guard_span.set_attribute("action", "passed")
                        guard_span.add_event("input_guardrail_passed", {
                            "agent": agent_name,
                            "guard": guard_name
                        })
                        
                except Exception as e:
                    guard_span.record_exception(e)
                    guard_span.set_attribute("execution.error", True)
                    guard_span.set_attribute("error.message", str(e))
                    guard_span.set_attribute("error.agent", agent_name)
                    guard_span.add_event("input_guardrail_failed", {
                        "agent": agent_name,
                        "guard": guard_name,
                        "error": str(e)
                    })
                    # print(f"Error in input guardrail '{guard_name}' for agent '{agent_name}': {str(e)}")
                    # Re-raise if you want strict enforcement
                    # raise

    def _apply_output_guardrails(self, content: str, agent_name: str, parent_context) -> None:
        """Apply output guardrails with telemetry tracking as direct children of agent span."""
        for guard in self.guards:
            guard_name = guard.get("name", "unknown")
            
            # Create guardrail span as direct child of the current agent execution span
            with self.tracer.start_as_current_span(
                f"guardrail:{guard_name}",
                context=parent_context
            ) as guard_span:
                
                # Set comprehensive attributes linking this guardrail to the specific agent
                guard_span.set_attribute("component", agent_name)
                guard_span.set_attribute("guard", guard_name)
                guard_span.set_attribute("content_type", "output")
                
                try:
                    start_time = datetime.now()
                    guard_span.add_event("output_guardrail_started", {
                        "agent": agent_name,
                        "guard": guard_name
                    })
                    
                    run_result = self._call_run_guardrail(content, guard, "output")
                    
                    end_time = datetime.now()
                    guard_span.add_event("output_guardrail_completed", {
                        "agent": agent_name,
                        "guard": guard_name
                    })
                    
                    # Set comprehensive telemetry attributes
                    guard_span.set_attribute("input.value", self._safe_str(content))
                    guard_span.set_attribute("output.value", 
                                        self._safe_str(run_result.get("response", "")))
                    guard_span.set_attribute("start_time", start_time.isoformat())
                    guard_span.set_attribute("end_time", end_time.isoformat())
                    guard_span.set_attribute("duration", 
                                        (end_time - start_time).total_seconds())
                    
                    # Check validation results
                    validation_passed = bool(run_result.get("validation_passed", True))
                    success = bool(run_result.get("success", True))
                    detected_issue = not validation_passed or not success
                    
                    guard_span.set_attribute("detected", detected_issue)
                    
                    # Handle guardrail violations
                    if detected_issue:
                        guard_span.set_attribute("action", self.action)
                        error_msg = run_result.get("error_message", 
                                                f"Output guardrail '{guard_name}' detected an issue for agent '{agent_name}'")
                        guard_span.add_event("output_guardrail_violation", {
                            "agent": agent_name,
                            "guard": guard_name,
                            "error_message": error_msg
                        })
                        guard_span.set_attribute("violation.message", error_msg)
                        guard_span.set_attribute("violation.agent", agent_name)
                        # print(f"Output guardrail violation on {agent_name}: {error_msg}")
                        # Uncomment if you want to raise exceptions on violations:
                        # raise ValueError(error_msg)
                    else:
                        guard_span.set_attribute("action", "passed")
                        guard_span.add_event("output_guardrail_passed", {
                            "agent": agent_name,
                            "guard": guard_name
                        })
                        
                except Exception as e:
                    guard_span.record_exception(e)
                    guard_span.set_attribute("execution.error", True)
                    guard_span.set_attribute("error.message", str(e))
                    guard_span.set_attribute("error.agent", agent_name)
                    guard_span.add_event("output_guardrail_failed", {
                        "agent": agent_name,
                        "guard": guard_name,
                        "error": str(e)
                    })
                    # print(f"Error in output guardrail '{guard_name}' for agent '{agent_name}': {str(e)}")
                    # Re-raise if you want strict enforcement
                    # raise

    def _format_reply(self, reply, agent) -> Dict[str, Any]:
        """Ensure reply has consistent format for AutoGen compatibility."""
        agent_name = getattr(agent, 'name', 'assistant')
        
        if isinstance(reply, str):
            return {
                "role": "assistant", 
                "content": reply, 
                "name": agent_name
            }
        elif isinstance(reply, dict):
            # Ensure required fields exist
            formatted_reply = reply.copy()
            if "role" not in formatted_reply:
                formatted_reply["role"] = "assistant"
            if "name" not in formatted_reply:
                formatted_reply["name"] = agent_name
            if "content" not in formatted_reply and reply:
                # Try to extract content from the reply
                content = self._extract_response_content(reply)
                if content:
                    formatted_reply["content"] = content
            return formatted_reply
        else:
            # Handle other response types
            content = self._extract_response_content(reply)
            return {
                "role": "assistant",
                "content": content,
                "name": agent_name
            }

    def _safe_str(self, value, max_length: int = 1000) -> str:
        """Safely convert value to string with length limit for telemetry."""
        if value is None:
            return ""
        
        str_value = str(value)
        if len(str_value) > max_length:
            return str_value[:max_length] + "... [truncated]"
        return str_value