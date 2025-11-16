from __future__ import annotations

import os
from typing import Any

from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter


def setup_telemetry() -> trace.Tracer:
    """
    Initialize OpenTelemetry with traces, logs, and metrics.

    Environment variables:
    - OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint (default: http://localhost:4317)
    - OTEL_SERVICE_NAME: Service name (default: pawliytix)
    - OTEL_DISABLED: Set to 'true' to disable telemetry
    """

    # Check if telemetry is disabled
    if os.getenv("OTEL_DISABLED", "").lower() in ("true", "1", "yes"):
        # Return a no-op tracer
        print("⚠️  OpenTelemetry disabled via OTEL_DISABLED")
        return trace.get_tracer(__name__)

    service_name = os.getenv("OTEL_SERVICE_NAME", "pawliytix")
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

    try:
        # Create resource with service name
        resource = Resource(attributes={SERVICE_NAME: service_name})

        # Setup Tracing
        tracer_provider = TracerProvider(resource=resource)
        span_exporter = OTLPSpanExporter(endpoint=endpoint)
        span_processor = BatchSpanProcessor(span_exporter)
        tracer_provider.add_span_processor(span_processor)
        trace.set_tracer_provider(tracer_provider)

        # Setup Logging for real-time streaming
        logger_provider = LoggerProvider(resource=resource)
        log_exporter = OTLPLogExporter(endpoint=endpoint)
        log_processor = BatchLogRecordProcessor(log_exporter)
        logger_provider.add_log_record_processor(log_processor)
        set_logger_provider(logger_provider)

        # Setup Metrics for analysis
        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=endpoint)
        )
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)

        print(f"✓ OpenTelemetry enabled: service={service_name}, endpoint={endpoint}")
        print(f"  Traces, logs, and metrics configured")

        # Return a tracer
        return trace.get_tracer(__name__)
    except Exception as e:
        print(f"⚠️  OpenTelemetry setup failed: {e}")
        print("   Continuing without telemetry...")
        return trace.get_tracer(__name__)


def get_tracer() -> trace.Tracer:
    """Get the configured tracer instance."""
    return trace.get_tracer(__name__)


def get_otel_logger() -> Any:
    """Get the configured OpenTelemetry logger for streaming events."""
    from opentelemetry._logs import get_logger_provider
    return get_logger_provider().get_logger(__name__)


def get_meter() -> Any:
    """Get the configured meter for metrics collection."""
    return metrics.get_meter(__name__)
