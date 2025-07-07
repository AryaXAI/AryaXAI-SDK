from typing import Optional
from aryaxai.common.utils import poll_events
from aryaxai.common.xai_uris import (
    AVAILABLE_GUARDRAILS_URI,
    CONFIGURE_GUARDRAILS_URI,
    DELETE_GUARDRAILS_URI,
    GET_AVAILABLE_TEXT_MODELS_URI,
    GET_GUARDRAILS_URI,
    INITIALIZE_TEXT_MODEL_URI,
    MESSAGES_URI,
    SESSIONS_URI,
    TRACES_URI,
    UPDATE_GUARDRAILS_STATUS_URI,
)
from aryaxai.core.project import Project
import pandas as pd

from aryaxai.core.wrapper import AryaModels, monitor


class TextProject(Project):
    """Project for text modality

    :return: TextProject
    """

    def llm_monitor(self, client, session_id=None):
        """llm monitoring for custom clients

        :param client: client to monitor like OpenAI
        :param session_id: id of the session
        :return: response
        """
        return monitor(project=self, client=client, session_id=session_id)

    def sessions(self) -> pd.DataFrame:
        """All sessions

        :return: response
        """
        res = self.api_client.get(f"{SESSIONS_URI}?project_name={self.project_name}")
        if not res["success"]:
            raise Exception(res.get("details"))

        return pd.DataFrame(res.get("details"))

    def messages(self, session_id: str) -> pd.DataFrame:
        """All messages for a session

        :param session_id: id of the session
        :return: response
        """
        res = self.api_client.get(
            f"{MESSAGES_URI}?project_name={self.project_name}&session_id={session_id}"
        )
        if not res["success"]:
            raise Exception(res.get("details"))

        return pd.DataFrame(res.get("details"))

    def traces(self, trace_id: str) -> pd.DataFrame:
        """Traces generated for trace_id

        :param trace_id: id of the trace
        :return: response
        """
        res = self.api_client.get(
            f"{TRACES_URI}?project_name={self.project_name}&trace_id={trace_id}"
        )
        if not res["success"]:
            raise Exception(res.get("details"))

        return pd.DataFrame(res.get("details"))

    def guardrails(self) -> pd.DataFrame:
        """Guardrails configured in project

        :return: response
        """
        res = self.api_client.get(
            f"{GET_GUARDRAILS_URI}?project_name={self.project_name}"
        )
        if not res["success"]:
            raise Exception(res.get("details"))

        return pd.DataFrame(res.get("details"))

    def update_guardrail_status(self, guardrail_id: str, status: bool) -> str:
        """Update Guardrail Status

        :param guardrail_id: id of the guardrail
        :param status: status to active/inactive
        :return: response
        """
        payload = {
            "project_name": self.project_name,
            "guardrail_id": guardrail_id,
            "status": status,
        }
        res = self.api_client.post(UPDATE_GUARDRAILS_STATUS_URI, payload=payload)
        if not res["success"]:
            raise Exception(res.get("details"))

        return res.get("details")

    def delete_guardrail(self, guardrail_id: str) -> str:
        """Deletes Guardrail

        :param guardrail_id: id of the guardrail
        :return: response
        """
        payload = {
            "project_name": self.project_name,
            "guardrail_id": guardrail_id,
        }
        res = self.api_client.post(DELETE_GUARDRAILS_URI, payload=payload)
        if not res["success"]:
            raise Exception(res.get("details"))

        return res.get("details")

    def available_guardrails(self) -> pd.DataFrame:
        """Available guardrails to configure

        :return: response
        """
        res = self.api_client.get(AVAILABLE_GUARDRAILS_URI)
        if not res["success"]:
            raise Exception(res.get("details"))

        return pd.DataFrame(res.get("details"))

    def configure_guardrail(
        self,
        guardrail_name: str,
        guardrail_config: dict,
        model_name: str,
        apply_on: str,
    ) -> str:
        """Configure guardrail for project

        :param guardrail_name: name of the guardrail
        :param guardrail_config: config for the guardrail
        :param model_name: name of the model
        :param apply_on: when to apply guardrails input/output
        :return: response
        """
        payload = {
            "name": guardrail_name,
            "config": guardrail_config,
            "model_name": model_name,
            "apply_on": apply_on,
            "project_name": self.project_name,
        }
        res = self.api_client.post(CONFIGURE_GUARDRAILS_URI, payload)
        if not res["success"]:
            raise Exception(res.get("details"))

        return res.get("details")

    def initialize_text_model(self, model_provider: str, model_name: str, model_task_type:str) -> str:
        """Initialize text model

        :param model_provider: model of provider
        :param model_name: name of the model to be initialized
        :param model_task_type: task type of model
        :return: response
        """
        payload = {
            "model_provider": model_provider,
            "model_name": model_name,
            "model_task_type": model_task_type,
            "project_name": self.project_name,
        }
        res = self.api_client.post(f"{INITIALIZE_TEXT_MODEL_URI}", payload)
        if not res["success"]:
            raise Exception(res.get("details", "Model Initialization Failed"))
        poll_events(self.api_client, self.project_name, res["event_id"])

    def generate_text_case(
        self,
        model_name: str,
        prompt: str,
        instance_type: Optional[str] = "gova-2",
        serverless_instance_type: Optional[str] = "xsmall",
        explainability_method: Optional[list] = ["DLB"],
        explain_model: Optional[bool] = False,
        session_id: Optional[str] = None,
    ) -> dict:
        """Generate Text Case

        :param model_name: name of the model
        :param model_type: type of the model
        :param input_text: input text for the case
        :param tag: tag for the case
        :param task_type: task type for the case, defaults to None
        :param instance_type: instance type for the case, defaults to None
        :param explainability_method: explainability method for the case, defaults to None
        :param explain_model: explain model for the case, defaults to False
        :return: response
        """
        llm = monitor(
            project=self, client=AryaModels(project=self), session_id=session_id
        )
        res = llm.generate_text_case(
            model_name=model_name,
            prompt=prompt,
            instance_type=instance_type,
            serverless_instance_type=serverless_instance_type,
            explainability_method=explainability_method,
            explain_model=explain_model,
        )
        return res

    def available_text_models(self) -> pd.DataFrame:
        """Get available text models

        :return: list of available text models
        """
        res = self.api_client.get(f"{GET_AVAILABLE_TEXT_MODELS_URI}")
        if not res["success"]:
            raise Exception(res.get("details"," Failed to fetch available text models"))
        return pd.DataFrame(res.get("details"))