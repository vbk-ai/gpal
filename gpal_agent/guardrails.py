"""Multi-layer security guardrails, PII redaction, and HITL callbacks for GPAL.

Implements:
- Layer 1: Google Cloud Model Armor prompt sanitization integration.
- Layer 2: Google Cloud DLP (Sensitive Data Protection) de-identification.
- Layer 3: Always-on deterministic regex scrubbing for SSN, credit cards, emails,
  phone numbers, and IPv4 addresses.
- Pre-storage PII scrubbing via ADK `BasePlugin.on_user_message_callback` so raw PII
  never reaches session persistence.
- Prompt injection & output policy self-evaluation guardrails.
- Human-in-the-Loop (HITL) `before_tool_callback` gating destructive memory wipes.
"""

from __future__ import annotations

from contextlib import contextmanager
import logging
import re
from typing import Any, Final, Iterator

from gpal_agent.config import (
    GpalConfig,
    HIGH_STAKES_TOOLS,
    STATE_TEMP_HITL_CONFIRMED_WIPE,
    STATE_TEMP_PENDING_WIPE_REQUEST,
    get_config,
)
from gpal_agent.schemas import StructuredToolError

try:
    from google.adk.plugins import BasePlugin as _AdkBasePlugin
except Exception:  # pragma: no cover - fallback when ADK plugins are unavailable

    class _AdkBasePlugin:  # type: ignore[no-redef]
        """Fallback base plugin class when google.adk.plugins is unavailable."""

        def __init__(self, name: str = "base_plugin") -> None:
            self.name = name


# ==============================================================================
# Observability Helpers (Graceful Runtime Resolution)
# ==============================================================================


def _get_logger() -> logging.Logger:
    """Resolve the structured logger from gpal_agent.observability with fallback."""
    try:
        from gpal_agent.observability import get_structured_logger

        return get_structured_logger("gpal_agent.guardrails")
    except Exception:
        return logging.getLogger("gpal_agent.guardrails")


def _log_intent(
    event_type: str,
    tool_name: str | None = None,
    details: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Record an INTENT structured log entry via gpal_agent.observability."""
    safe_details = details or {}
    try:
        from gpal_agent.observability import log_intent

        return log_intent(
            event_type=event_type,
            tool_name=tool_name,
            details=safe_details,
            **kwargs,
        )
    except TypeError:
        try:
            from gpal_agent.observability import log_intent

            return log_intent(event_type, tool_name=tool_name, details=safe_details)
        except Exception:
            pass
    except Exception:
        pass

    payload: dict[str, Any] = {
        "phase": "INTENT",
        "event_type": event_type,
        "tool_name": tool_name,
        "details": safe_details,
        **kwargs,
    }
    _get_logger().info("INTENT: %s", payload)
    return payload


def _log_outcome(
    event_type: str,
    status: str = "SUCCESS",
    tool_name: str | None = None,
    details: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Record an OUTCOME structured log entry via gpal_agent.observability."""
    safe_details = details or {}
    try:
        from gpal_agent.observability import log_outcome

        return log_outcome(
            event_type=event_type,
            status=status,
            tool_name=tool_name,
            details=safe_details,
            **kwargs,
        )
    except TypeError:
        try:
            from gpal_agent.observability import log_outcome

            return log_outcome(
                event_type,
                status=status,
                tool_name=tool_name,
                details=safe_details,
            )
        except Exception:
            pass
    except Exception:
        pass

    payload: dict[str, Any] = {
        "phase": "OUTCOME",
        "event_type": event_type,
        "status": status,
        "tool_name": tool_name,
        "details": safe_details,
        **kwargs,
    }
    _get_logger().info("OUTCOME: %s", payload)
    return payload


@contextmanager
def _start_pii_span(span_name: str = "gpal.pii_scrub") -> Iterator[Any]:
    """Start an OpenTelemetry span via gpal_agent.observability or opentelemetry."""
    tracer = None
    try:
        from gpal_agent.observability import get_tracer

        tracer = get_tracer("gpal_agent.guardrails")
    except Exception:
        try:
            from opentelemetry import trace

            tracer = trace.get_tracer("gpal_agent.guardrails")
        except Exception:
            tracer = None

    if tracer is not None and hasattr(tracer, "start_as_current_span"):
        with tracer.start_as_current_span(span_name) as span:
            yield span
    else:
        yield None


# ==============================================================================
# Deterministic PII Regex Definitions & Cloud DLP InfoTypes
# ==============================================================================

_DLP_INFO_TYPES: Final[tuple[str, ...]] = (
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SOCIAL_SECURITY_NUMBER",
    "CREDIT_CARD_NUMBER",
    "IP_ADDRESS",
)

_REGEX_SSN: Final[re.Pattern[str]] = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# Matches 13 to 16 digits separated by optional spaces or hyphens without swallowing trailing spaces
_REGEX_CREDIT_CARD: Final[re.Pattern[str]] = re.compile(
    r"(?<!\d)\b\d(?:[ -]?\d){12,15}\b(?!\d)"
)
_REGEX_EMAIL: Final[re.Pattern[str]] = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
_REGEX_IPV4: Final[re.Pattern[str]] = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_REGEX_PHONE: Final[re.Pattern[str]] = re.compile(
    r"(?:(?<!\w)\+?1[-.\s]?)?(?:\(\d{3}\)[-.\s]*|\b\d{3}[-.\s])\d{3}[-.\s]\d{4}\b"
    r"|\b\d{3}-\d{4}\b"
)

_PII_REGEX_RULES: Final[tuple[tuple[str, re.Pattern[str], str], ...]] = (
    ("SSN", _REGEX_SSN, "[REDACTED_SSN]"),
    ("CREDIT_CARD", _REGEX_CREDIT_CARD, "[REDACTED_CREDIT_CARD]"),
    ("EMAIL", _REGEX_EMAIL, "[REDACTED_EMAIL]"),
    ("IP_ADDRESS", _REGEX_IPV4, "[REDACTED_IP]"),
    ("PHONE", _REGEX_PHONE, "[REDACTED_PHONE]"),
)


_DLP_TOKEN_NORMALIZATION: Final[tuple[tuple[str, str], ...]] = (
    ("[EMAIL_ADDRESS]", "[REDACTED_EMAIL]"),
    ("[PHONE_NUMBER]", "[REDACTED_PHONE]"),
    ("[US_SOCIAL_SECURITY_NUMBER]", "[REDACTED_SSN]"),
    ("[CREDIT_CARD_NUMBER]", "[REDACTED_CREDIT_CARD]"),
    ("[IP_ADDRESS]", "[REDACTED_IP]"),
)

_NETWORK_REACHABLE_CACHE: bool | None = None


def _is_cloud_network_reachable() -> bool:
    """Fast 0.5s check to avoid 120s OAuth/gRPC hangs in offline or sandboxed environments."""
    global _NETWORK_REACHABLE_CACHE
    if _NETWORK_REACHABLE_CACHE is not None:
        return _NETWORK_REACHABLE_CACHE
    import socket

    try:
        with socket.create_connection(("oauth2.googleapis.com", 443), timeout=0.5):
            _NETWORK_REACHABLE_CACHE = True
    except OSError:
        _NETWORK_REACHABLE_CACHE = False
    return _NETWORK_REACHABLE_CACHE


def _normalize_dlp_tokens(text: str) -> str:
    """Normalize Cloud DLP / Model Armor info-type tokens to canonical `[REDACTED_*]` tokens."""
    normalized = text
    for dlp_token, canonical_token in _DLP_TOKEN_NORMALIZATION:
        if dlp_token in normalized:
            normalized = normalized.replace(dlp_token, canonical_token)
    return normalized


class PiiScrubber:
    """Multi-layer PII scrubber combining Model Armor, Cloud DLP, and regex rules."""

    def __init__(
        self,
        config: GpalConfig | None = None,
        enable_cloud_calls: bool = True,
    ) -> None:
        self.config: GpalConfig = config or get_config()
        self.enable_cloud_calls: bool = enable_cloud_calls
        self._model_armor_client: Any = None
        self._model_armor_available: bool = True
        self._dlp_client: Any = None
        self._dlp_available: bool = True

    def _scrub_via_model_armor(self, text: str) -> tuple[str, list[str]]:
        """Layer 1: Sanitize user prompt via Google Cloud Model Armor when configured."""
        if (
            not self.enable_cloud_calls
            or not self._model_armor_available
            or not self.config.model_armor_template_id
        ):
            return text, []

        if self._model_armor_client is None and not _is_cloud_network_reachable():
            self._model_armor_available = False
            return text, []

        try:
            from google.cloud import modelarmor_v1  # type: ignore[import-untyped]

            if self._model_armor_client is None:
                self._model_armor_client = modelarmor_v1.ModelArmorClient()
            request = modelarmor_v1.SanitizeUserPromptRequest(
                name=self.config.model_armor_template_id,
                user_prompt_data=modelarmor_v1.DataItem(text=text),
            )
            response = self._model_armor_client.sanitize_user_prompt(
                request=request, timeout=3.0
            )
            sanitization_result = getattr(response, "sanitization_result", None)
            if sanitization_result is not None:
                filter_results = getattr(sanitization_result, "filter_results", {}) or {}
                sdp_result = filter_results.get("sdp")
                deid_text = getattr(
                    getattr(sdp_result, "sdp_filter_result", None),
                    "deidentify_result",
                    None,
                )
                extracted = getattr(getattr(deid_text, "data", None), "text", None)
                if isinstance(extracted, str) and extracted and extracted != text:
                    return _normalize_dlp_tokens(extracted), ["MODEL_ARMOR_SDP"]
        except Exception as exc:
            from gpal_agent.config import is_reauth_error, terminate_on_reauth_error

            if is_reauth_error(exc):
                terminate_on_reauth_error(exc)
            self._model_armor_available = False
            _get_logger().debug(
                "Model Armor sanitization skipped or unavailable: %s", exc
            )

        return text, []

    def _scrub_via_cloud_dlp(self, text: str) -> tuple[str, list[str]]:
        """Layer 2: De-identify sensitive PII via Google Cloud DLP when enabled."""
        if (
            not self.enable_cloud_calls
            or not self._dlp_available
            or not self.config.enable_cloud_dlp
            or not self.config.project_id
        ):
            return text, []

        if self._dlp_client is None and not _is_cloud_network_reachable():
            self._dlp_available = False
            return text, []

        try:
            from google.cloud import dlp_v2  # type: ignore[import-untyped]

            if self._dlp_client is None:
                self._dlp_client = dlp_v2.DlpServiceClient()
            parent = f"projects/{self.config.project_id}/locations/global"
            inspect_config: dict[str, Any] = {
                "info_types": [{"name": info_type} for info_type in _DLP_INFO_TYPES]
            }
            deidentify_config: dict[str, Any] = {
                "info_type_transformations": {
                    "transformations": [
                        {
                            "primitive_transformation": {
                                "replace_with_info_type_config": {}
                            }
                        }
                    ]
                }
            }
            response = self._dlp_client.deidentify_content(
                request={
                    "parent": parent,
                    "inspect_config": inspect_config,
                    "deidentify_config": deidentify_config,
                    "item": {"value": text},
                },
                timeout=3.0,
            )
            deidentified_value = getattr(getattr(response, "item", None), "value", None)
            if (
                isinstance(deidentified_value, str)
                and deidentified_value
                and deidentified_value != text
            ):
                return _normalize_dlp_tokens(deidentified_value), ["CLOUD_DLP"]
        except Exception as exc:
            from gpal_agent.config import is_reauth_error, terminate_on_reauth_error

            if is_reauth_error(exc):
                terminate_on_reauth_error(exc)
            self._dlp_available = False
            _get_logger().debug(
                "Cloud DLP de-identification skipped or unavailable: %s", exc
            )

        return text, []

    def _scrub_via_regex(self, text: str) -> tuple[str, list[str]]:
        """Layer 3: Always-on deterministic regex scrubbing for common PII types."""
        if not text:
            return text, []

        sanitized = text
        redacted_types: list[str] = []

        for pii_type, pattern, replacement in _PII_REGEX_RULES:
            matches = pattern.findall(sanitized)
            if matches:
                sanitized = pattern.sub(replacement, sanitized)
                redacted_types.extend([pii_type] * len(matches))

        return sanitized, redacted_types

    def scrub_text(self, text: str) -> tuple[str, list[str]]:
        """Scrub PII across all active layers and record an OpenTelemetry span.

        Args:
            text: Input string that may contain sensitive personal data.

        Returns:
            A tuple of `(sanitized_text, list_of_redacted_types)`.
        """
        if not isinstance(text, str) or not text:
            return (text if isinstance(text, str) else ""), []

        with _start_pii_span("gpal.pii_scrub") as span:
            redacted_types: list[str] = []

            stage1_text, ma_types = self._scrub_via_model_armor(text)
            redacted_types.extend(ma_types)

            stage2_text, dlp_types = self._scrub_via_cloud_dlp(stage1_text)
            redacted_types.extend(dlp_types)

            final_text, regex_types = self._scrub_via_regex(stage2_text)
            redacted_types.extend(regex_types)

            if span is not None and hasattr(span, "set_attribute"):
                try:
                    span.set_attribute("gpal.pii_redacted_count", len(redacted_types))
                    if redacted_types:
                        span.set_attribute(
                            "gpal.pii_redacted_types",
                            ",".join(sorted(set(redacted_types))),
                        )
                except Exception:
                    pass

            return final_text, redacted_types

    def sanitize_string(self, text: str) -> str:
        """Return only the PII-scrubbed text string."""
        sanitized, _ = self.scrub_text(text)
        return sanitized

    def _scrub_any(self, value: Any) -> Any:
        """Recursively scrub strings inside nested dicts, lists, and tuples."""
        if isinstance(value, str):
            return self.sanitize_string(value)
        if isinstance(value, dict):
            return self.scrub_dict(value)
        if isinstance(value, list):
            for idx, item in enumerate(value):
                value[idx] = self._scrub_any(item)
            return value
        if isinstance(value, tuple):
            return tuple(self._scrub_any(item) for item in value)
        return value

    def scrub_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Recursively scrub all string values in a dictionary in-place and return it."""
        if not isinstance(data, dict):
            return data
        for key in list(data.keys()):
            data[key] = self._scrub_any(data[key])
        return data

    def scrub_content_in_place(self, content: Any) -> Any:
        """Mutate `google.genai.types.Content` (or duck-typed equivalent) in-place.

        Ensures `part.text` is replaced with `sanitize_string(part.text)` before
        ADK session persistence or LLM invocation.
        """
        if content is None:
            return None

        if isinstance(content, list):
            for item in content:
                self.scrub_content_in_place(item)
            return content

        if isinstance(content, dict):
            return self.scrub_dict(content)

        # If the object has a nested `.content` attribute (e.g., LlmResponse or Event)
        nested_content = getattr(content, "content", None)
        if nested_content is not None and nested_content is not content:
            self.scrub_content_in_place(nested_content)

        # If the object has `.parts` (standard google.genai.types.Content)
        parts = getattr(content, "parts", None)
        if parts:
            for part in parts:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text:
                    sanitized_text = self.sanitize_string(part_text)
                    if sanitized_text != part_text:
                        try:
                            part.text = sanitized_text
                        except Exception:
                            pass
                elif isinstance(part, dict) and isinstance(part.get("text"), str):
                    part["text"] = self.sanitize_string(part["text"])

        # Direct `.text` attribute on part/message objects
        direct_text = getattr(content, "text", None)
        if isinstance(direct_text, str) and direct_text and not parts:
            sanitized_direct = self.sanitize_string(direct_text)
            if sanitized_direct != direct_text:
                try:
                    content.text = sanitized_direct
                except Exception:
                    pass

        return content


default_pii_scrubber: Final[PiiScrubber] = PiiScrubber()


def scrub_pii_text(text: str) -> str:
    """Module-level convenience function to scrub PII from a text string."""
    return default_pii_scrubber.sanitize_string(text)


def scrub_content_in_place(content: Any) -> Any:
    """Module-level convenience function to scrub a Content object in-place."""
    return default_pii_scrubber.scrub_content_in_place(content)


# ==============================================================================
# Prompt Injection & Output Safety Self-Evaluation Guardrails
# ==============================================================================

_PROMPT_INJECTION_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    (
        "SYSTEM_PROMPT_EXFILTRATION",
        re.compile(
            r"(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|above)\s+"
            r"(?:instructions|prompts|rules|directions)"
            r"|(?:reveal|print|show|output|dump|repeat)\s+(?:your\s+|the\s+)?"
            r"(?:system\s+prompt|system\s+instructions|hidden\s+instructions|developer\s+prompt)",
            re.IGNORECASE,
        ),
    ),
    (
        "SAFETY_FILTER_BYPASS",
        re.compile(
            r"(?:bypass|disable|override|turn\s+off|ignore)\s+"
            r"(?:all\s+)?(?:safety\s+filters|guardrails|security\s+policies|content\s+filters|hitl)",
            re.IGNORECASE,
        ),
    ),
    (
        "JAILBREAK_PERSONA_OVERRIDE",
        re.compile(
            r"\b(?:you\s+are\s+now\s+dan|do\s+anything\s+now|developer\s+mode\s+enabled|"
            r"act\s+as\s+an\s+unrestricted|jailbreak\s+mode)\b",
            re.IGNORECASE,
        ),
    ),
)

_UNAPPROVED_OUTPUT_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("RAW_SSN_LEAKAGE", _REGEX_SSN),
    ("RAW_CREDIT_CARD_LEAKAGE", _REGEX_CREDIT_CARD),
    (
        "UNAPPROVED_DESTRUCTIVE_COMMAND",
        re.compile(r"\b(?:rm\s+-rf\s+/|DROP\s+TABLE\s+\w+)\b", re.IGNORECASE),
    ),
)


def evaluate_input_safety(text: str) -> dict[str, Any]:
    """Evaluate user input for prompt injection and jailbreak attempts.

    Args:
        text: User prompt text to inspect.

    Returns:
        Dictionary with keys `allowed` (bool), `reason` (str), and
        `policy_triggered` (str | None).
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "allowed": True,
            "reason": "Empty or non-text input is safe.",
            "policy_triggered": None,
        }

    for policy_name, pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern.search(text):
            return {
                "allowed": False,
                "reason": (
                    f"Input blocked by prompt security guardrail ({policy_name}): "
                    "Attempt to override system instructions or safety policies detected."
                ),
                "policy_triggered": policy_name,
            }

    return {
        "allowed": True,
        "reason": "Input passed prompt injection and safety policy checks.",
        "policy_triggered": None,
    }


def evaluate_output_safety(text: str) -> dict[str, Any]:
    """Self-evaluate model output to ensure no raw PII or unsafe actions leaked.

    Args:
        text: Model response text to inspect.

    Returns:
        Dictionary with keys `allowed` (bool), `reason` (str), and
        `policy_triggered` (str | None).
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "allowed": True,
            "reason": "Empty model output is safe.",
            "policy_triggered": None,
        }

    for policy_name, pattern in _UNAPPROVED_OUTPUT_PATTERNS:
        if pattern.search(text):
            return {
                "allowed": False,
                "reason": (
                    f"Model output flagged by self-evaluation guardrail ({policy_name})."
                ),
                "policy_triggered": policy_name,
            }

    return {
        "allowed": True,
        "reason": "Model output passed self-evaluation safety checks.",
        "policy_triggered": None,
    }


def get_default_safety_settings() -> list[Any]:
    """Return strict Gemini `SafetySetting` configurations for GPAL models."""
    categories = (
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    )
    try:
        from google.genai import types

        return [
            types.SafetySetting(
                category=getattr(types.HarmCategory, category, category),
                threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            )
            for category in categories
        ]
    except Exception:
        return [
            {"category": category, "threshold": "BLOCK_LOW_AND_ABOVE"}
            for category in categories
        ]


# ==============================================================================
# Agent-Level Model Callbacks
# ==============================================================================


def _extract_latest_user_text(llm_request: Any) -> str:
    """Extract concatenated text from the most recent user message in `llm_request`."""
    contents = getattr(llm_request, "contents", None)
    if not contents or not isinstance(contents, list):
        return ""

    for content in reversed(contents):
        role = getattr(content, "role", None)
        if role in ("user", None):
            parts = getattr(content, "parts", None) or []
            texts = [
                getattr(p, "text", "")
                for p in parts
                if isinstance(getattr(p, "text", None), str)
            ]
            if texts:
                return "\n".join(texts)
    return ""


def _build_guardrail_refusal_response(message: str) -> Any:
    """Construct an ADK `LlmResponse` (or fallback Content/dict) for blocked requests."""
    try:
        from google.adk.models.llm_response import LlmResponse
        from google.genai import types

        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=message)],
            )
        )
    except Exception:
        return {
            "role": "model",
            "content": {"role": "model", "parts": [{"text": message}]},
            "text": message,
        }


def guardrail_before_model_callback(
    callback_context: Any, llm_request: Any
) -> Any:
    """Scrub PII from `llm_request.contents` and enforce prompt injection safety.

    Args:
        callback_context: ADK `CallbackContext` for the current agent turn.
        llm_request: Outgoing `LlmRequest` about to be sent to Gemini.

    Returns:
        `None` when safe to proceed, or an `LlmResponse` declining unsafe input.
    """
    latest_raw_text = _extract_latest_user_text(llm_request)
    safety_verdict = evaluate_input_safety(latest_raw_text)

    contents = getattr(llm_request, "contents", None)
    if contents:
        default_pii_scrubber.scrub_content_in_place(contents)

    _log_intent(
        event_type="model_call",
        details={
            "agent_name": getattr(callback_context, "agent_name", "gpal_root_agent"),
            "input_allowed": safety_verdict["allowed"],
            "policy_triggered": safety_verdict["policy_triggered"],
        },
    )

    if not safety_verdict["allowed"]:
        _log_outcome(
            event_type="model_call",
            status="BLOCKED_BY_GUARDRAIL",
            details=safety_verdict,
        )
        refusal_text = (
            "I'm here as your supportive GPAL companion, so I can't bypass my safety "
            "guidelines or reveal internal system instructions. How else can I help you today?"
        )
        return _build_guardrail_refusal_response(refusal_text)

    return None


def guardrail_after_model_callback(
    callback_context: Any, llm_response: Any
) -> Any:
    """Scrub any residual PII in `llm_response` and self-evaluate output safety.

    Args:
        callback_context: ADK `CallbackContext` for the current agent turn.
        llm_response: Incoming `LlmResponse` returned by Gemini.

    Returns:
        `None` after mutating `llm_response` in-place to ensure zero PII leakage.
    """
    response_text = ""
    content = getattr(llm_response, "content", None)
    if content is not None:
        parts = getattr(content, "parts", None) or []
        response_text = "\n".join(
            getattr(p, "text", "")
            for p in parts
            if isinstance(getattr(p, "text", None), str)
        )

    output_verdict = evaluate_output_safety(response_text)
    default_pii_scrubber.scrub_content_in_place(llm_response)

    _log_outcome(
        event_type="model_call",
        status="SUCCESS",
        details={
            "agent_name": getattr(callback_context, "agent_name", "gpal_root_agent"),
            "output_allowed": output_verdict["allowed"],
            "was_sanitized": not output_verdict["allowed"],
            "policy_triggered": output_verdict["policy_triggered"],
        },
    )
    return None


# ==============================================================================
# Human-in-the-Loop (HITL) & Observability Tool Callbacks
# ==============================================================================


def _resolve_tool_name(tool: Any) -> str:
    """Extract the canonical tool name from an ADK BaseTool, callable, or string."""
    for attr in ("name", "__name__"):
        val = getattr(tool, attr, None)
        if isinstance(val, str) and val:
            return val
    return str(tool)


def _resolve_state_dict(tool_context: Any) -> Any:
    """Extract mutable state mapping from `tool_context`."""
    state = getattr(tool_context, "state", None)
    if state is not None:
        return state
    if isinstance(tool_context, dict) and "state" in tool_context:
        return tool_context["state"]
    return {}


def hitl_before_tool_callback(
    tool: Any,
    args: dict[str, Any],
    tool_context: Any,
) -> dict[str, Any] | None:
    """Enforce pre-tool PII scrubbing and Human-in-the-Loop confirmation for high-stakes tools.

    Args:
        tool: The ADK tool instance or function being invoked.
        args: Mutable dictionary of tool arguments supplied by the model.
        tool_context: ADK `ToolContext` providing access to session state.

    Returns:
        `None` to allow tool execution, or a `StructuredToolError` dict when blocked.
    """
    tool_name = _resolve_tool_name(tool)

    if isinstance(args, dict):
        default_pii_scrubber.scrub_dict(args)

    _log_intent(
        event_type="tool_call",
        tool_name=tool_name,
        details={"args": dict(args) if isinstance(args, dict) else {}},
    )

    if tool_name in HIGH_STAKES_TOOLS:
        state = _resolve_state_dict(tool_context)
        hitl_confirmed = False
        try:
            hitl_confirmed = state.get(STATE_TEMP_HITL_CONFIRMED_WIPE) is True
        except Exception:
            hitl_confirmed = False

        explicit_arg_confirmed = (
            isinstance(args, dict) and args.get("explicit_user_confirmation") is True
        )

        if not (hitl_confirmed and explicit_arg_confirmed):
            try:
                state[STATE_TEMP_PENDING_WIPE_REQUEST] = True
            except Exception:
                pass

            _log_outcome(
                event_type="tool_call",
                tool_name=tool_name,
                status="BLOCKED_BY_HITL",
                details={
                    "hitl_state_confirmed": hitl_confirmed,
                    "explicit_user_confirmation": explicit_arg_confirmed,
                },
            )

            return StructuredToolError(
                error_code="HITL_CONFIRMATION_REQUIRED",
                message=(
                    "High-stakes action 'wipe_all_user_memories' halted by "
                    "Human-in-the-Loop guardrail."
                ),
                recovery_instructions=(
                    "Ask the user to explicitly confirm that they want to permanently "
                    "delete all stored friend memories. Once the user replies yes, "
                    "call confirm_pending_memory_wipe first, and then call "
                    "wipe_all_user_memories(explicit_user_confirmation=True)."
                ),
            ).model_dump()

    return None


def observability_after_tool_callback(
    tool: Any,
    args: dict[str, Any],
    tool_context: Any,
    tool_response: Any,
) -> dict[str, Any] | None:
    """Scrub tool output and record structured OUTCOME telemetry after tool execution."""
    tool_name = _resolve_tool_name(tool)

    if isinstance(tool_response, dict):
        default_pii_scrubber.scrub_dict(tool_response)

    preview = default_pii_scrubber.sanitize_string(str(tool_response)[:200])
    _log_outcome(
        event_type="tool_call",
        tool_name=tool_name,
        status="SUCCESS",
        details={"response_preview": preview},
    )
    return None


# ==============================================================================
# ADK Runner Plugin: Pre-Storage PII Redaction & Model Armor Integration
# ==============================================================================


class GpalSafetyAndPiiPlugin(_AdkBasePlugin):
    """ADK Plugin that scrubs PII from user messages BEFORE session persistence."""

    def __init__(
        self,
        name: str = "gpal_safety_and_pii_plugin",
        scrubber: PiiScrubber | None = None,
    ) -> None:
        super().__init__(name=name)
        self.scrubber: PiiScrubber = scrubber or default_pii_scrubber

    async def on_user_message_callback(
        self,
        *,
        invocation_context: Any = None,
        user_message: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Scrub `user_message` in-place prior to `session_service.append_event`.

        This guarantees raw PII (SSNs, credit cards, emails, phone numbers, IPs)
        is redacted before the event is persisted to SQLite or Vertex SessionService.
        """
        _log_intent(
            event_type="pre_storage_user_message_scrub",
            details={"plugin": self.name},
        )
        sanitized_message = self.scrubber.scrub_content_in_place(user_message)
        _log_outcome(
            event_type="pre_storage_user_message_scrub",
            status="SUCCESS",
            details={"plugin": self.name},
        )
        return sanitized_message

    async def before_model_callback(
        self,
        *,
        callback_context: Any = None,
        llm_request: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Run pre-model PII scrubbing and prompt injection policy checks."""
        return guardrail_before_model_callback(
            callback_context=callback_context,
            llm_request=llm_request,
        )

    async def after_model_callback(
        self,
        *,
        callback_context: Any = None,
        llm_response: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Run post-model PII scrubbing and output self-evaluation checks."""
        return guardrail_after_model_callback(
            callback_context=callback_context,
            llm_response=llm_response,
        )


def get_configured_plugins(config: GpalConfig | None = None) -> list[Any]:
    """Return the configured ADK plugins including `GpalSafetyAndPiiPlugin` and optional `ModelArmorPlugin`."""
    cfg = config or get_config()
    plugins: list[Any] = [
        GpalSafetyAndPiiPlugin(
            name="gpal_safety_and_pii_plugin",
            scrubber=PiiScrubber(config=cfg),
        )
    ]

    if cfg.model_armor_template_id:
        for module_path in (
            "google.adk.integrations.model_armor",
            "google.adk.plugins.model_armor",
        ):
            try:
                import importlib

                mod = importlib.import_module(module_path)
                plugin_cls = getattr(mod, "ModelArmorPlugin", None)
                if plugin_cls is not None:
                    try:
                        plugins.append(
                            plugin_cls(template_id=cfg.model_armor_template_id)
                        )
                    except TypeError:
                        plugins.append(plugin_cls(cfg.model_armor_template_id))
                    break
            except Exception as exc:
                _get_logger().debug(
                    "Optional ADK ModelArmorPlugin (%s) not loaded: %s",
                    module_path,
                    exc,
                )

    return plugins


__all__ = [
    "GpalSafetyAndPiiPlugin",
    "PiiScrubber",
    "default_pii_scrubber",
    "evaluate_input_safety",
    "evaluate_output_safety",
    "get_configured_plugins",
    "get_default_safety_settings",
    "guardrail_after_model_callback",
    "guardrail_before_model_callback",
    "hitl_before_tool_callback",
    "observability_after_tool_callback",
    "scrub_content_in_place",
    "scrub_pii_text",
]
