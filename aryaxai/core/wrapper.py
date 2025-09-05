import json
import time
import functools
from typing import Callable, Optional
import inspect
import uuid
from openai import OpenAI
from pydantic import BaseModel
import requests
from aryaxai.client.client import APIClient
from aryaxai.common.environment import Environment
from aryaxai.common.xai_uris import GENERATE_TEXT_CASE_URI

class Wrapper:
    def __init__(self, project_name, api_client):
        self.project_name = project_name
        self.api_client = api_client

    def add_message(self,session_id,trace_id,input_data, output_data,metadata, duration):
        payload = {
            "project_name":self.project_name,
            "session_id":session_id,
            "trace_id": trace_id,
            "input_data":input_data,
            "output_data": output_data,
            "metadata":metadata,
            "duration":duration,
        }
        res = self.api_client.post(
            f"sessions/add_session_message",
            payload=payload
        )

    def add_trace_details(self,session_id,trace_id,component,input_data,metadata,output_data=None,function_to_run=None):
        start_time = time.perf_counter()
        if function_to_run:
            result = function_to_run()
        duration = time.perf_counter() - start_time
        if not output_data:
            output_data= result
            if isinstance(result, BaseModel): output_data = result.model_dump()
        payload = {
            "project_name":self.project_name,
            "trace_id": trace_id,
            "session_id":session_id,
            "component":component,
            "input_data":input_data,
            "output_data":output_data,
            "metadata":metadata,
            "duration":duration,
        }
        res = self.api_client.post(
            f"traces/add_trace",
            payload=payload
        )
        if function_to_run:
            if component == "Input Guardrails" or component == "Output Guardrails":
                if not result.get("success"):
                    return result.get("details")
            return result
        return res

    def run_guardrails(self,input_data,trace_id,session_id,model_name, apply_on):
        res = self.api_client.post(
            f"v2/ai-models/run_guardrails",
            payload={
                "trace_id": trace_id,
                "session_id": session_id,
                "input_data":input_data,
                "model_name":model_name,
                "project_name":self.project_name,
                "apply_on": apply_on
            }
        )
        
        return res
    
    def _get_wrapper(self,original_method: Callable, method_name: str, project_name: str, session_id: Optional[str] = None) -> Callable:
        if inspect.iscoroutinefunction(original_method):
            @functools.wraps(original_method)
            async def async_wrapper(*args, **kwargs):
                result = await original_method(*args, **kwargs)
                return result
            return async_wrapper
        else:
            @functools.wraps(original_method)
            def wrapper(*args, **kwargs):
                # if original_method.__func__ == OpenAI.chat.completions.create:
                #     print("here openai")
                # if original_method.__func__ == AryaModels.generate_text_case:
                #     print("here arya models")

                if method_name == "client.chat.completions.create":
                    input_data = kwargs.get("messages")
                    model_name = kwargs.get("model")
                if method_name == "client.generate_text_case":
                    input_data = kwargs.get("prompt")
                    model_name = kwargs.get("model_name")
                total_start_time = time.perf_counter()
                trace_id = str(uuid.uuid4())
                trace_res = self.add_trace_details(
                    trace_id=trace_id,
                    session_id=session_id,
                    component="Input",
                    input_data=input_data,
                    output_data=input_data,
                    metadata={},
                )
                id_session = trace_res.get("details",{}).get("session_id")
                self.add_trace_details(
                    trace_id=trace_id,
                    session_id=id_session,
                    component="Input Guardrails",
                    input_data=input_data,
                    metadata={},
                    function_to_run=lambda: self.run_guardrails(
                        session_id=id_session,
                        trace_id=trace_id,
                        input_data=input_data,
                        model_name=model_name,
                        apply_on="input"
                    )
                )
                if method_name == "client.generate_text_case":
                    kwargs["session_id"] = id_session
                    kwargs["trace_id"] = trace_id
                result = self.add_trace_details(
                    trace_id=trace_id,
                    session_id=id_session,
                    component="LLM",
                    input_data=input_data,
                    metadata=kwargs,
                    function_to_run=lambda: original_method(*args, **kwargs)
                )
                if method_name == "client.chat.completions.create":
                    output_data = result.choices[0].message.content
                if method_name == "client.generate_text_case":
                    output_data = result.get("details",{}).get("result",{}).get("output")
                if output_data:
                    self.add_trace_details(
                        trace_id=trace_id,
                        session_id=id_session,
                        component="Output Guardrails",
                        input_data=output_data,
                        metadata={},
                        function_to_run=lambda: self.run_guardrails(
                            session_id=id_session,
                            trace_id=trace_id,
                            model_name=model_name,
                            input_data=output_data,
                            apply_on="output"
                        )
                    )
                    self.add_trace_details(
                        trace_id=trace_id,
                        session_id=id_session,
                        component="Output",
                        input_data=input_data,
                        output_data=output_data,
                        metadata={},
                    )
                metadata = {}
                if method_name == "client.generate_text_case":
                    metadata = {
                        "case_id":result.get("details",{}).get("case_id")
                    }
                self.add_message(
                    trace_id=trace_id,
                    session_id=id_session,
                    input_data=input_data,
                    output_data=output_data,
                    duration=time.perf_counter() - total_start_time,
                    metadata=metadata
                )
                return result
            return wrapper


class AryaModels:
    def __init__(self, project, api_client: APIClient):
        self.project = project
        self.api_client = api_client

    def generate_text_case(
        self,
        model_name: str,
        prompt: str,
        instance_type: str = "xsmall",
        serverless_instance_type: str = "gova-2",
        explainability_method: list = ["DLB"],
        explain_model: bool = False,
        trace_id: str = None,
        session_id: str = None,
        min_tokens: int = 100,
        max_tokens: int = 500,
        stream: bool = False,
    ):
        payload = {
            "session_id": session_id,
            "trace_id": trace_id,
            "project_name": self.project.project_name,
            "model_name": model_name,
            "input_text": prompt,
            "instance_type": instance_type,
            "serverless_instance_type": serverless_instance_type,
            "explainability_method": explainability_method,
            "explain_model": explain_model,
            "max_tokens": max_tokens,
            "min_tokens": min_tokens,
            "stream": stream,
        }
        
        if stream:
            env = Environment()
            url = env.get_base_url() + "/" + GENERATE_TEXT_CASE_URI
            with requests.post(
                url,
                headers={**self.api_client.headers, "Accept": "text/event-stream"},
                json=payload,
                stream=True,
            ) as response:
                response.raise_for_status()

                buffer = ""
                for line in response.iter_content(decode_unicode=True):
                    if not line or line.strip() == "[DONE]":
                        continue

                    if line.startswith("data:"):
                        line = line[len("data:"):].strip()
                    try:
                        event = json.loads(line)
                        text_piece = event.get("choices", [{}])[0].get("text", "")
                    except Exception as e:
                        text_piece = line
                    print(text_piece, end="", flush=True)
            return response
        else:
            res = self.api_client.post(GENERATE_TEXT_CASE_URI, payload)
            if not res.get("success"):
                raise Exception(res.get("details"))
            return res

def monitor(project, client, session_id=None):
    # print("client",client, type(client))
    # if not isinstance(client, OpenAI):
    #     raise Exception("Not a valid sdk to monitor")
    wrapper = Wrapper(project_name=project.project_name, api_client=project.api_client)
    if isinstance(client, OpenAI):
        models = project.models()["model_name"].to_list()
        if "OpenAI" not in models:
            raise Exception ("OpenAI Model Not Initialized")
        client.chat.completions.create = wrapper._get_wrapper(
            original_method=client.chat.completions.create,
            method_name="client.chat.completions.create",
            session_id=session_id,
            project_name=project.project_name
        )
    if isinstance(client, AryaModels):
        client.generate_text_case =wrapper._get_wrapper(
            original_method=client.generate_text_case,
            method_name="client.generate_text_case",
            session_id=session_id,
            project_name=project.project_name
        )
    return client
