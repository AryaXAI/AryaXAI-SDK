from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.sdk.resources import Resource
from openinference.instrumentation.langchain import LangChainInstrumentor
# from openinference.instrumentation.autogen_agent import AutogenInstrumentor
from openinference.instrumentation.autogen_agentchat import AutogenAgentChatInstrumentor
from openinference.instrumentation.crewai import CrewAIInstrumentor
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from openinference.instrumentation.dspy import DSPyInstrumentor
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
# from pydantic import BaseModel
import os


class Tracer:
    def __init__(self):
        self.base_url = os.getenv("XAI_API_URL", "https://apiv2.aryaxai.com")
        self.endpoint = f"{self.base_url}"
    def setup_langchain_tracing(self , project: object) -> None:
        """
        Sets up OpenTelemetry tracing for a given project with OTLP and console exporters.
        
        Args:
            project: An object containing project details, expected to have a 'name' attribute.
        """
        
        # Extract project name or use default
        
        project_name = getattr(project, 'project_name')
        # Create resource with service and project details
        resource = Resource(attributes={
            "service.name": "langgraph-app",
            "project_name": project_name,
        })
        
        # Initialize tracer provider
        tracer_provider = trace_sdk.TracerProvider(resource=resource)
        trace_api.set_tracer_provider(tracer_provider)
        
        # Add OTLP and console span processors
        tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(self.endpoint)))
        # Instrument LangChain
        LangChainInstrumentor().instrument()
    
    def setup_autogen_tracing(self ,project: object) -> None:
        """
        Sets up OpenTelemetry tracing for a given project with OTLP and console exporters.
        
        Args:
            project: An object containing project details, expected to have a 'name' attribute.
        """
        
        # Extract project name or use default
        
        project_name = getattr(project, 'project_name')
        # Create resource with service and project details
        resource = Resource(attributes={
            "service.name": "autogen-app",
            "project_name": project_name,
        })
        
        # Initialize tracer provider
        tracer_provider = trace_sdk.TracerProvider(resource=resource)
        trace_api.set_tracer_provider(tracer_provider)
        
        # Add OTLP and console span processors
        tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(self.endpoint)))
        # tracer_provider.add_span_processor(ConsoleSpanExporter())
        # Instrument Autogen
        AutogenAgentChatInstrumentor().instrument()

    def setup_crewai_tracing(self , project: object) -> None:
        """
        Sets up OpenTelemetry tracing for a given project with OTLP and console exporters.
        
        Args:
            project: An object containing project details, expected to have a 'name' attribute.
        """
        
        # Extract project name or use default
        
        project_name = getattr(project, 'project_name')
        # Create resource with service and project details
        resource = Resource(attributes={
            "service.name": "crewai-app",
            "project_name": project_name,
        })
        
        # Initialize tracer provider
        tracer_provider = trace_sdk.TracerProvider(resource=resource)
        trace_api.set_tracer_provider(tracer_provider)
        
        # Add OTLP and console span processors
        tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(self.endpoint)))
        # tracer_provider.add_span_processor(ConsoleSpanExporter())
        # Instrument CrewAI
        CrewAIInstrumentor().instrument()

    def setup_agents_tracing(self , project : object) -> None:
        """
        Sets up OpenTelemetry tracing for a given project with OTLP and console exporters.
        
        Args:
            project: An object containing project details, expected to have a 'name' attribute.
        """
        
        # Extract project name or use default
        
        project_name = getattr(project, 'project_name')
        # Create resource with service and project details
        resource = Resource(attributes={
            "service.name": "agents-app",
            "project_name": project_name,
        })
        
        # Initialize tracer provider
        tracer_provider = trace_sdk.TracerProvider(resource=resource)
        trace_api.set_tracer_provider(tracer_provider)
        
        # Add OTLP and console span processors
        tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(self.endpoint)))
        # tracer_provider.add_span_processor(ConsoleSpanExporter())
        # Instrument OpenAI
        OpenAIAgentsInstrumentor().instrument()

    def setup_dspy_tracing(self , project : object) -> None:
        """
        Sets up OpenTelemetry tracing for a given project with OTLP and console exporters.
        
        Args:
            project: An object containing project details, expected to have a 'name' attribute.
        """
        
        # Extract project name or use default
        
        project_name = getattr(project, 'project_name')
        # Create resource with service and project details
        resource = Resource(attributes={
            "service.name": "dspy-app",
            "project_name": project_name,
        })
        
        # Initialize tracer provider
        tracer_provider = trace_sdk.TracerProvider(resource=resource)
        trace_api.set_tracer_provider(tracer_provider)
        
        # Add OTLP and console span processors
        tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(self.endpoint)))
        # tracer_provider.add_span_processor(ConsoleSpanExporter())
        # Instrument DSPy
        DSPyInstrumentor().instrument()
    
    def setup_llamaindex_tracing(self , project : object) -> None:
        
        # Extract project name or use default
        
        project_name = getattr(project, 'project_name')
        # Create resource with service and project details
        resource = Resource(attributes={
            "service.name": "llamaindex-app",
            "project_name": project_name,
        })
        
        # Initialize tracer provider
        tracer_provider = trace_sdk.TracerProvider(resource=resource)
        trace_api.set_tracer_provider(tracer_provider)
        
        # Add OTLP and console span processors
        tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(self.endpoint)))
        # tracer_provider.add_span_processor(ConsoleSpanExporter())
        # Instrument llama
        LlamaIndexInstrumentor().instrument()

    def setup_smolagents_tracing(self , project : object) -> None:
        
        # Extract project name or use default
        
        project_name = getattr(project, 'project_name')
        # Create resource with service and project details
        resource = Resource(attributes={
            "service.name": "smolagents",
            "project_name": project_name,
        })
        
        # Initialize tracer provider
        tracer_provider = trace_sdk.TracerProvider(resource=resource)
        trace_api.set_tracer_provider(tracer_provider)
        
        # Add OTLP and console span processors
        tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(self.endpoint)))
        # tracer_provider.add_span_processor(ConsoleSpanExporter())
        # Instrument Smol
        SmolagentsInstrumentor().instrument()