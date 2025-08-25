from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.sdk.resources import Resource
from openinference.instrumentation.langchain import LangChainInstrumentor

class Tracer:

    def setup_langchain_tracing(project: object) -> None:
        """
        Sets up OpenTelemetry tracing for a given project with OTLP and console exporters.
        
        Args:
            project: An object containing project details, expected to have a 'name' attribute.
        """
        endpoint = "http://3.108.15.217:30075"
        
        # Extract project name or use default
        
        project_name = getattr(project, 'user_project_name')
        project_name=project_name.replace(" ", "_").strip()
        # Create resource with service and project details
        resource = Resource(attributes={
            "service.name": "langgraph-app",
            "project_name": project_name,
        })
        
        # Initialize tracer provider
        tracer_provider = trace_sdk.TracerProvider(resource=resource)
        trace_api.set_tracer_provider(tracer_provider)
        
        # Add OTLP and console span processors
        tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
        
        # Instrument LangChain
        LangChainInstrumentor().instrument()