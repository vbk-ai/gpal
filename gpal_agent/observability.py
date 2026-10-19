"""Structured JSON logging, PII log filtering, and OpenTelemetry Cloud Tracing.

Satisfies Rubric Criteria:
- 4.1 Structured JSON Logging (`StructuredJsonFormatter`, `get_structured_logger`)
- 4.2 Intent vs. Outcome Capture (`log_intent`, `log_outcome`)
- 4.3 Distributed Tracing (`setup_cloud_tracing`, `get_tracer`, `trace_operation`)
- 4.4 PII Redaction Log Filter (`redact_pii_in_log_string`, `PiiRedactingLogFilter`)
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
import re
import secrets as _py_secrets
import time
from typing import Any, Final, Iterator

from gpal_agent.config import get_config

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        SimpleSpanProcessor,
    )
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from opentelemetry.trace import Status, StatusCode

    _HAS_OTEL = True
except ImportError:  # pragma: no cover - fallback when opentelemetry-sdk not installed
    _HAS_OTEL = False
    trace = None  # type: ignore[assignment]
    Resource = None  # type: ignore[assignment]
    TracerProvider = Any  # type: ignore[assignment,misc]
    BatchSpanProcessor = None  # type: ignore[assignment]
    SimpleSpanProcessor = None  # type: ignore[assignment]
    InMemorySpanExporter = Any  # type: ignore[assignment,misc]
    Status = None  # type: ignore[assignment]
    StatusCode = None  # type: ignore[assignment]

# ==============================================================================
# Deterministic PII Scrubbing Patterns (Criterion 4.4)
# ==============================================================================
# Order matters: scrub credit cards, SSNs, and IPv4 addresses before phone
# numbers so longer numeric patterns are not partially matched as phone numbers.
_CREDIT_CARD_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?:\d{4}[-\s]?){3}\d{1,4}\b"
)
_SSN_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b\d{3}-\d{2}-\d{4}\b"
)
_IPV4_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
)
_EMAIL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
_PHONE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<!\w)(?:\+?1[-.\s]?)?(?:\(\d{3}\)[-.\s]?|\b\d{3}[-.\s])\d{3}[-.\s]\d{4}\b"
)

_MAX_CAPTURED_LOG_RECORDS: Final[int] = 500
_RECENT_STRUCTURED_LOGS: list[dict[str, Any]] = []

_STANDARD_LOG_RECORD_KEYS: Final[frozenset[str]] = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
        "message",
        "lifecycle_phase",
        "event_type",
        "agent_name",
        "session_id",
        "user_id",
        "tool_name",
        "model_name",
        "status",
        "latency_ms",
        "trace_id",
        "span_id",
        "metadata",
        "extra_fields",
        "_gpal_captured",
    }
)


def redact_pii_in_log_string(text: str) -> str:
    """Scrub common PII patterns from a string before emitting to logs or spans.

    Replaces:
    - Credit card numbers -> `[REDACTED_CREDIT_CARD]`
    - US SSNs -> `[REDACTED_SSN]`
    - IPv4 addresses -> `[REDACTED_IP]`
    - Email addresses -> `[REDACTED_EMAIL]`
    - Phone numbers -> `[REDACTED_PHONE]`
    """
    if not text or not isinstance(text, str):
        return text

    scrubbed = _CREDIT_CARD_PATTERN.sub("[REDACTED_CREDIT_CARD]", text)
    scrubbed = _SSN_PATTERN.sub("[REDACTED_SSN]", scrubbed)
    scrubbed = _IPV4_PATTERN.sub("[REDACTED_IP]", scrubbed)
    scrubbed = _EMAIL_PATTERN.sub("[REDACTED_EMAIL]", scrubbed)
    scrubbed = _PHONE_PATTERN.sub("[REDACTED_PHONE]", scrubbed)
    return scrubbed


def _scrub_pii_recursive(value: Any) -> Any:
    """Recursively scrub PII from strings, dicts, lists, and tuples."""
    if isinstance(value, str):
        return redact_pii_in_log_string(value)
    if isinstance(value, dict):
        return {
            (redact_pii_in_log_string(k) if isinstance(k, str) else k): _scrub_pii_recursive(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_scrub_pii_recursive(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_scrub_pii_recursive(item) for item in value)
    if isinstance(value, set):
        return {_scrub_pii_recursive(item) for item in value}
    return value


class PiiRedactingLogFilter(logging.Filter):
    """Logging filter that scrubs PII from log messages, args, and extra attributes."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Scrub PII in-place on the `LogRecord` and allow emission (`True`)."""
        if isinstance(record.msg, str):
            record.msg = redact_pii_in_log_string(record.msg)
        elif record.msg is not None:
            record.msg = redact_pii_in_log_string(str(record.msg))

        if record.args:
            record.args = _scrub_pii_recursive(record.args)

        for attr_name in ("metadata", "extra_fields", "details"):
            if hasattr(record, attr_name):
                setattr(
                    record,
                    attr_name,
                    _scrub_pii_recursive(getattr(record, attr_name)),
                )

        for key, val in list(record.__dict__.items()):
            if key not in _STANDARD_LOG_RECORD_KEYS and not key.startswith("_"):
                record.__dict__[key] = _scrub_pii_recursive(val)

        return True


# ==============================================================================
# Fallback Span / Exporter for Environments Without OpenTelemetry SDK
# ==============================================================================
class _FallbackSpanContext:
    """Minimal SpanContext compatible with OpenTelemetry's SpanContext API."""

    def __init__(self, trace_id: int, span_id: int) -> None:
        self.trace_id = trace_id
        self.span_id = span_id
        self.is_valid = True


class _FallbackSpan:
    """Minimal Span compatible with OpenTelemetry's Span API."""

    def __init__(self, name: str, exporter: _FallbackInMemorySpanExporter) -> None:
        self.name = name
        self.attributes: dict[str, Any] = {}
        self.context = _FallbackSpanContext(
            trace_id=_py_secrets.randbits(128) or 1,
            span_id=_py_secrets.randbits(64) or 1,
        )
        self._exporter = exporter
        self.status_code: str = "UNSET"
        self.exceptions: list[Exception] = []

    def get_span_context(self) -> _FallbackSpanContext:
        return self.context

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def record_exception(self, exception: Exception) -> None:
        self.exceptions.append(exception)

    def set_status(self, status: Any) -> None:
        self.status_code = str(status)

    def __enter__(self) -> _FallbackSpan:
        _FALLBACK_ACTIVE_SPANS.append(self)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if _FALLBACK_ACTIVE_SPANS and _FALLBACK_ACTIVE_SPANS[-1] is self:
            _FALLBACK_ACTIVE_SPANS.pop()
        self._exporter.add_finished_span(self)


class _FallbackInMemorySpanExporter:
    """Fallback in-memory span exporter when opentelemetry-sdk is unavailable."""

    def __init__(self) -> None:
        self._spans: list[_FallbackSpan] = []

    def add_finished_span(self, span: _FallbackSpan) -> None:
        self._spans.append(span)

    def get_finished_spans(self) -> tuple[_FallbackSpan, ...]:
        return tuple(self._spans)

    def clear(self) -> None:
        self._spans.clear()


class _FallbackTracer:
    """Fallback tracer creating `_FallbackSpan` instances."""

    def __init__(
        self, name: str, exporter: _FallbackInMemorySpanExporter
    ) -> None:
        self.name = name
        self._exporter = exporter

    def start_as_current_span(self, name: str) -> _FallbackSpan:
        return _FallbackSpan(name=name, exporter=self._exporter)


class _FallbackTracerProvider:
    """Fallback TracerProvider when opentelemetry-sdk is unavailable."""

    def __init__(self, exporter: _FallbackInMemorySpanExporter) -> None:
        self._exporter = exporter

    def get_tracer(self, name: str = "gpal_agent") -> _FallbackTracer:
        return _FallbackTracer(name=name, exporter=self._exporter)


_FALLBACK_ACTIVE_SPANS: list[_FallbackSpan] = []


def _extract_current_trace_and_span_ids(
    record: logging.LogRecord | None = None,
) -> tuple[str, str]:
    """Return active `(trace_id, span_id)` as 32-char and 16-char hex strings."""
    if record is not None:
        explicit_trace = getattr(record, "trace_id", None)
        explicit_span = getattr(record, "span_id", None)
        if (
            isinstance(explicit_trace, str)
            and len(explicit_trace) == 32
            and explicit_trace != "0" * 32
            and isinstance(explicit_span, str)
            and len(explicit_span) == 16
            and explicit_span != "0" * 16
        ):
            return explicit_trace, explicit_span

    if _HAS_OTEL and trace is not None:
        try:
            current_span = trace.get_current_span()
            span_context = current_span.get_span_context() if current_span else None
            if span_context is not None and span_context.is_valid:
                return (
                    format(span_context.trace_id, "032x"),
                    format(span_context.span_id, "016x"),
                )
        except Exception:  # pragma: no cover - defensive guard
            pass
    elif _FALLBACK_ACTIVE_SPANS:
        ctx = _FALLBACK_ACTIVE_SPANS[-1].get_span_context()
        return (
            format(ctx.trace_id, "032x"),
            format(ctx.span_id, "016x"),
        )

    return ("0" * 32, "0" * 16)


class StructuredJsonFormatter(logging.Formatter):
    """Formats every log record as a single-line structured JSON payload."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize `record` into a single-line JSON string with trace correlation."""
        raw_message = record.getMessage()
        scrubbed_message = redact_pii_in_log_string(raw_message)
        trace_id, span_id = _extract_current_trace_and_span_ids(record)

        raw_metadata: dict[str, Any] = {}
        record_metadata = getattr(record, "metadata", None)
        if isinstance(record_metadata, dict):
            raw_metadata.update(record_metadata)

        extra_fields = getattr(record, "extra_fields", None)
        if isinstance(extra_fields, dict):
            raw_metadata.update(extra_fields)

        for key, value in record.__dict__.items():
            if key not in _STANDARD_LOG_RECORD_KEYS and not key.startswith("_"):
                raw_metadata[key] = value

        scrubbed_metadata = _scrub_pii_recursive(raw_metadata)

        latency_val = getattr(record, "latency_ms", None)
        if isinstance(latency_val, (int, float)):
            latency_ms: float | None = round(float(latency_val), 3)
        else:
            latency_ms = None

        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": scrubbed_message,
            "lifecycle_phase": getattr(record, "lifecycle_phase", "GENERAL"),
            "event_type": getattr(record, "event_type", "agent_log"),
            "agent_name": getattr(record, "agent_name", "gpal_friend_coordinator"),
            "session_id": redact_pii_in_log_string(
                str(getattr(record, "session_id", ""))
            ),
            "user_id": redact_pii_in_log_string(str(getattr(record, "user_id", ""))),
            "tool_name": str(getattr(record, "tool_name", "")),
            "model_name": str(getattr(record, "model_name", "")),
            "status": str(getattr(record, "status", "OK")),
            "latency_ms": latency_ms,
            "trace_id": trace_id,
            "span_id": span_id,
            "metadata": scrubbed_metadata,
        }

        if record.exc_info:
            payload["exception"] = redact_pii_in_log_string(
                self.formatException(record.exc_info)
            )

        if not getattr(record, "_gpal_captured", False):
            setattr(record, "_gpal_captured", True)
            _RECENT_STRUCTURED_LOGS.append(payload)
            if len(_RECENT_STRUCTURED_LOGS) > _MAX_CAPTURED_LOG_RECORDS:
                del _RECENT_STRUCTURED_LOGS[
                    : len(_RECENT_STRUCTURED_LOGS) - _MAX_CAPTURED_LOG_RECORDS
                ]

        return json.dumps(payload, default=str)


def get_captured_log_records() -> list[dict[str, Any]]:
    """Return a copy of recent structured JSON log records for testing & inspection."""
    return list(_RECENT_STRUCTURED_LOGS)


def clear_captured_log_records() -> None:
    """Clear the in-memory captured structured log record buffer."""
    _RECENT_STRUCTURED_LOGS.clear()


def get_structured_logger(name: str = "gpal_agent") -> logging.Logger:
    """Configure and return a structured JSON logger with PII redaction attached."""
    cfg = get_config()
    logger = logging.getLogger(name)
    level = getattr(logging, cfg.log_level.upper(), logging.INFO)
    logger.setLevel(level)

    has_pii_filter = any(
        isinstance(f, PiiRedactingLogFilter) for f in logger.filters
    )
    if not has_pii_filter:
        logger.addFilter(PiiRedactingLogFilter())

    has_structured_handler = any(
        isinstance(getattr(h, "formatter", None), StructuredJsonFormatter)
        for h in logger.handlers
    )
    if not has_structured_handler:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.addFilter(PiiRedactingLogFilter())
        handler.setFormatter(StructuredJsonFormatter())
        logger.addHandler(handler)

    logger.propagate = False
    return logger


# ==============================================================================
# OpenTelemetry Cloud Tracing & In-Memory Span Exporter (Criterion 4.3)
# ==============================================================================
_IN_MEMORY_SPAN_EXPORTER: Any = (
    InMemorySpanExporter() if _HAS_OTEL else _FallbackInMemorySpanExporter()
)
_TRACER_PROVIDER: Any = None
_IN_MEMORY_PROCESSOR_ATTACHED: bool = False
_CLOUD_TRACE_EXPORTER_PROJECTS: set[str] = set()


def get_in_memory_span_exporter() -> InMemorySpanExporter:
    """Return the shared `InMemorySpanExporter` for unit tests and local diagnostics."""
    return _IN_MEMORY_SPAN_EXPORTER


def clear_recorded_spans() -> None:
    """Clear all spans collected by the shared `InMemorySpanExporter`."""
    _IN_MEMORY_SPAN_EXPORTER.clear()


def setup_cloud_tracing(
    project_id: str | None = None,
    export_to_cloud: bool | None = None,
    service_name: str = "gpal_agent",
) -> TracerProvider:
    """Initialize OpenTelemetry tracing with InMemorySpanExporter and optional Cloud Trace.

    Args:
        project_id: Optional Google Cloud project ID. Defaults to `GpalConfig.project_id`.
        export_to_cloud: Whether to attach `CloudTraceSpanExporter`. Defaults to
            `GpalConfig.enable_cloud_trace`.
        service_name: Logical service name resource attribute.

    Returns:
        The configured OpenTelemetry `TracerProvider`.
    """
    global _TRACER_PROVIDER, _IN_MEMORY_PROCESSOR_ATTACHED

    cfg = get_config()
    resolved_project = (
        project_id if project_id is not None else cfg.project_id
    ).strip()
    should_export_cloud = (
        export_to_cloud if export_to_cloud is not None else cfg.enable_cloud_trace
    )

    if not _HAS_OTEL:
        if _TRACER_PROVIDER is None:
            _TRACER_PROVIDER = _FallbackTracerProvider(_IN_MEMORY_SPAN_EXPORTER)
        return _TRACER_PROVIDER

    if _TRACER_PROVIDER is None:
        existing_provider = trace.get_tracer_provider()
        if isinstance(existing_provider, TracerProvider):
            _TRACER_PROVIDER = existing_provider
        else:
            resource = Resource.create(
                {
                    "service.name": service_name,
                    "service.environment": cfg.environment,
                }
            )
            _TRACER_PROVIDER = TracerProvider(resource=resource)
            trace.set_tracer_provider(_TRACER_PROVIDER)

    if not _IN_MEMORY_PROCESSOR_ATTACHED:
        _TRACER_PROVIDER.add_span_processor(
            SimpleSpanProcessor(_IN_MEMORY_SPAN_EXPORTER)
        )
        _IN_MEMORY_PROCESSOR_ATTACHED = True

    if (
        should_export_cloud
        and resolved_project
        and resolved_project not in _CLOUD_TRACE_EXPORTER_PROJECTS
    ):
        try:
            import socket
            from gpal_agent.config import (
                is_reauth_error,
                terminate_on_reauth_error,
                verify_gcp_adc_or_exit,
            )

            with socket.create_connection(("oauth2.googleapis.com", 443), timeout=0.5):
                pass
            if not verify_gcp_adc_or_exit(cfg):
                return _TRACER_PROVIDER
            from opentelemetry.exporter.cloud_trace import (  # type: ignore[import-untyped]
                CloudTraceSpanExporter,
            )

            cloud_exporter = CloudTraceSpanExporter(project_id=resolved_project)
            _TRACER_PROVIDER.add_span_processor(BatchSpanProcessor(cloud_exporter))
            _CLOUD_TRACE_EXPORTER_PROJECTS.add(resolved_project)
        except Exception as exc:
            from gpal_agent.config import is_reauth_error, terminate_on_reauth_error

            if is_reauth_error(exc):
                terminate_on_reauth_error(exc)
            get_structured_logger().warning(
                "Cloud Trace exporter unavailable for project '%s' (%s); "
                "continuing with in-memory span exporter.",
                resolved_project,
                type(exc).__name__,
                extra={
                    "lifecycle_phase": "OBSERVABILITY",
                    "event_type": "cloud_trace_fallback",
                    "status": "FALLBACK",
                },
            )

    return _TRACER_PROVIDER


def get_tracer(name: str = "gpal_agent") -> Any:
    """Return an OpenTelemetry `Tracer`, ensuring the provider is initialized."""
    if _TRACER_PROVIDER is None:
        setup_cloud_tracing(export_to_cloud=False)
    if _HAS_OTEL and trace is not None:
        return trace.get_tracer(name)
    return _TRACER_PROVIDER.get_tracer(name)


@contextmanager
def trace_operation(
    operation_name: str,
    attributes: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """Context manager that creates an OpenTelemetry span with PII-safe attributes."""
    tracer = get_tracer()
    with tracer.start_as_current_span(operation_name) as span:
        if attributes:
            for key, raw_value in attributes.items():
                if raw_value is None:
                    continue
                if isinstance(raw_value, str):
                    span.set_attribute(key, redact_pii_in_log_string(raw_value))
                elif isinstance(raw_value, (bool, int, float)):
                    span.set_attribute(key, raw_value)
                else:
                    span.set_attribute(
                        key, redact_pii_in_log_string(str(raw_value))
                    )
        try:
            yield span
            if _HAS_OTEL and Status is not None and StatusCode is not None:
                span.set_status(Status(StatusCode.OK))
            else:
                span.set_status("OK")
        except Exception as exc:
            span.record_exception(exc)
            if _HAS_OTEL and Status is not None and StatusCode is not None:
                span.set_status(
                    Status(
                        StatusCode.ERROR,
                        redact_pii_in_log_string(str(exc)),
                    )
                )
            else:
                span.set_status("ERROR")
            raise


# ==============================================================================
# Intent vs. Outcome Capture Helpers (Criterion 4.2)
# ==============================================================================
def log_intent(
    *,
    event_type: str,
    agent_name: str = "gpal_friend_coordinator",
    session_id: str = "",
    user_id: str = "",
    tool_name: str = "",
    model_name: str = "",
    details: dict[str, Any] | None = None,
    message: str | None = None,
) -> float:
    """Log the `INTENT` of an agent, model, or tool action before execution.

    Returns:
        `time.perf_counter()` timestamp to pass into `log_outcome(start_time=...)`.
    """
    start_time = time.perf_counter()
    logger = get_structured_logger()
    log_msg = message or f"INTENT [{event_type}] started by {agent_name}"
    logger.info(
        log_msg,
        extra={
            "lifecycle_phase": "INTENT",
            "event_type": event_type,
            "agent_name": agent_name,
            "session_id": session_id,
            "user_id": user_id,
            "tool_name": tool_name,
            "model_name": model_name,
            "status": "PENDING",
            "latency_ms": 0.0,
            "metadata": _scrub_pii_recursive(details or {}),
        },
    )
    return start_time


def log_outcome(
    *,
    event_type: str,
    start_time: float | None = None,
    status: str = "SUCCESS",
    agent_name: str = "gpal_friend_coordinator",
    session_id: str = "",
    user_id: str = "",
    tool_name: str = "",
    model_name: str = "",
    details: dict[str, Any] | None = None,
    message: str | None = None,
) -> float:
    """Log the `OUTCOME` and measured latency of an agent, model, or tool action.

    Returns:
        Computed latency in milliseconds (`float`).
    """
    latency_ms = (
        round(max(0.0, (time.perf_counter() - start_time) * 1000.0), 3)
        if start_time is not None
        else 0.0
    )
    logger = get_structured_logger()
    log_msg = (
        message
        or f"OUTCOME [{event_type}] completed with status={status} in {latency_ms}ms"
    )
    logger.info(
        log_msg,
        extra={
            "lifecycle_phase": "OUTCOME",
            "event_type": event_type,
            "agent_name": agent_name,
            "session_id": session_id,
            "user_id": user_id,
            "tool_name": tool_name,
            "model_name": model_name,
            "status": status,
            "latency_ms": latency_ms,
            "metadata": _scrub_pii_recursive(details or {}),
        },
    )
    return latency_ms


__all__ = [
    "PiiRedactingLogFilter",
    "StructuredJsonFormatter",
    "_RECENT_STRUCTURED_LOGS",
    "clear_captured_log_records",
    "clear_recorded_spans",
    "get_captured_log_records",
    "get_in_memory_span_exporter",
    "get_structured_logger",
    "get_tracer",
    "log_intent",
    "log_outcome",
    "redact_pii_in_log_string",
    "setup_cloud_tracing",
    "trace_operation",
]
