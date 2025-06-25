from datetime import datetime
import time
import functools
from typing import Callable, Optional
import inspect
import pandas as pd
import uuid
from openai import OpenAI
from pydantic import BaseModel
import requests

from aryaxai.common.xai_uris import GENERATE_TEXT_CASE_URI

# monitoring_url = f"http://localhost:30050"
monitoring_url = f"http://3.108.15.217:30085"


def add_message(project_name,session_id,trace_id,input_data, output_data,metadata, duration):
    payload = {
        "project_name":project_name,
        "session_id":session_id,
        "trace_id": trace_id,
        "input_data":input_data,
        "output_data": output_data,
        "metadata":metadata,
        "duration":duration,
    }
    res = requests.post(
        f"{monitoring_url}/sessions/add_session_message",
        json=payload
    )

def add_trace(project_name,trace_id,input_data,output_data,metadata,duration):
    payload = {
        "project_name":project_name,
        "trace_id": trace_id,
        "input_data":input_data,
        "output_data":output_data,
        "metadata":metadata,
        "duration":duration,
    }
    res = requests.post(
        f"{monitoring_url}/traces/add_trace",
        json=payload
    )
    return res.json()

def add_trace_details(project_name,session_id,trace_id,component,input_data,metadata,output_data=None,function_to_run=None):
    start_time = time.perf_counter()
    if function_to_run:
        result = function_to_run()
    duration = time.perf_counter() - start_time
    if not output_data:
        output_data= result
        if isinstance(result, BaseModel): output_data = result.model_dump()
    payload = {
        "project_name":project_name,
        "trace_id": trace_id,
        "session_id":session_id,
        "component":component,
        "input_data":input_data,
        "output_data":output_data,
        "metadata":metadata,
        "duration":duration,
    }
    # print(payload)
    res = requests.post(
        f"{monitoring_url}/traces/add_trace",
        json=payload
    )
    # print("res",res.json())  
    if function_to_run:
        if component == "Input Guardrails" or component == "Output Guardrails":
            if not result.get("success"):
                return result.get("details")
        return result
    return res.json()

def run_guardrails(project_name,input_data,trace_id,session_id,model_name, apply_on):
    res = requests.post(
        f"{monitoring_url}/guardrails/run",
        json={
            "trace_id": trace_id,
            "session_id": session_id,
            "input_data":input_data,
            "model_name":model_name,
            "project_name":project_name,
            "apply_on": apply_on
        }
    )
    response = res.json()
    
    return response

def get_messages(project_name, session_id):
    res = requests.get(
        f"{monitoring_url}/sessions/get_session_messages?project_name={project_name}&session_id={session_id}",
    )
    response = res.json()
    
    return pd.DataFrame(response.get("details"))

def get_traces(project_name, trace_id):
    url = f"{monitoring_url}/traces/get_traces?project_name={project_name}"
    if trace_id: url += f"&trace_id={trace_id}"
    res = requests.get(url)
    response = res.json()
    
    return pd.DataFrame(response.get("details"))

def get_sessions(project_name):
    res = requests.get(
        f"{monitoring_url}/sessions/get_sessions?project_name={project_name}",
    )
    response = res.json()
    
    return pd.DataFrame(response.get("details"))

def get_active_guardrails(project_name):
    res = requests.get(
        f"{monitoring_url}/guardrails/active_guardrails?project_name={project_name}",
    )
    response = res.json()
    
    return pd.DataFrame(response.get("details"))

def available_guardrails():
    res = requests.get(
        f"{monitoring_url}/guardrails/all",
    )
    response = res.json()
    
    return pd.DataFrame(response.get("details"))

def configure_guardrail(project_name, guardrail_name, guardrail_config, model_name, apply_on):
    res = requests.post(
        f"{monitoring_url}/guardrails/configure",
        json={
            "name":guardrail_name,
            "config":guardrail_config,
            "model_name": model_name,
            "apply_on":apply_on,
            "project_name":project_name
        }
    )
    return res.json()

def _get_wrapper(original_method: Callable, method_name: str, project_name: str, session_id: Optional[str] = None) -> Callable:
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
            trace_res = add_trace_details(
                project_name=project_name,
                trace_id=trace_id,
                session_id=session_id,
                component="Input",
                input_data=input_data,
                output_data=input_data,
                metadata={},
            )
            id_session = trace_res.get("details",{}).get("session_id")
            add_trace_details(
                project_name=project_name,
                trace_id=trace_id,
                session_id=id_session,
                component="Input Guardrails",
                input_data=input_data,
                metadata={},
                function_to_run=lambda: run_guardrails(
                    project_name=project_name,
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
            result = add_trace_details(
                project_name=project_name,
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
            add_trace_details(
                project_name=project_name,
                trace_id=trace_id,
                session_id=id_session,
                component="Output Guardrails",
                input_data=output_data,
                metadata={},
                function_to_run=lambda: run_guardrails(
                    project_name=project_name,
                    session_id=id_session,
                    trace_id=trace_id,
                    model_name=model_name,
                    input_data=output_data,
                    apply_on="output"
                )
            )
            add_trace_details(
                project_name=project_name,
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
            add_message(
                project_name=project_name,
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
    def __init__(self, project):
        self.project = project

    def generate_text_case(
        self,
        model_name: str,
        prompt: str,
        instance_type: Optional[str] = "gova-2",
        serverless_instance_type: Optional[str] = "xsmall",
        explainability_method: Optional[list] = ["DLB"],
        explain_model: Optional[bool] = False,
        trace_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        if self.project.metadata.get("modality") == "text":
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
            }
            print("payload",payload)
            res = self.project.api_client.post(GENERATE_TEXT_CASE_URI, payload)
            if not res["success"]:
                raise Exception(res["details"])
            return res
        else:
            return "Text case generation is not supported for this modality type"
     


def monitor(project, client, session_id=None):
    # print("client",client, type(client))
    # if not isinstance(client, OpenAI):
    #     raise Exception("Not a valid sdk to monitor")
    
    if isinstance(client, OpenAI):
        client.chat.completions.create = _get_wrapper(
            original_method=client.chat.completions.create,
            method_name="client.chat.completions.create",
            session_id=session_id,
            project_name=project.project_name
        )
    if isinstance(client, AryaModels):
        client.generate_text_case = _get_wrapper(
            original_method=client.generate_text_case,
            method_name="client.generate_text_case",
            session_id=session_id,
            project_name=project.project_name
        )
    return client
