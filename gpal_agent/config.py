"""Central configuration, model routing tiers, and voice metadata for GPAL."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# ==============================================================================
# Supported Prebuilt Gemini Voices & Persona Descriptions
# ==============================================================================
SUPPORTED_GEMINI_VOICES: Final[dict[str, str]] = {
    "Puck": "Upbeat, warm, approachable, and friendly companion (Default)",
    "Kore": "Calm, soothing, empathetic, and grounded listener",
    "Charon": "Deep, steady, informative, and composed conversationalist",
    "Aoede": "Breezy, melodic, expressive, and cheerful friend",
    "Fenrir": "Energetic, excitable, and dynamic speaker",
    "Leda": "Youthful, curious, and lighthearted companion",
    "Orus": "Firm, clear, confident, and encouraging voice",
    "Zephyr": "Bright, airy, uplifting, and optimistic tone",
}

DEFAULT_GEMINI_VOICE: Final[str] = os.getenv("DEFAULT_GEMINI_VOICE", "Puck")

# ==============================================================================
# Session State Keys (Following ADK `user:`, `app:`, and `temp:` Scopes)
# ==============================================================================
STATE_USER_PREFERRED_VOICE: Final[str] = "user:preferred_voice"
STATE_USER_VOICE_UPDATED_AT: Final[str] = "user:voice_updated_at"
STATE_USER_FRIEND_NOTES: Final[str] = "user:friend_notes"
STATE_USER_FRIEND_PROFILE_SUMMARY: Final[str] = "user:friend_profile_summary"
STATE_APP_COMPACTED_SUMMARY: Final[str] = "app:compacted_conversation_summary"
STATE_APP_COMPACTION_COUNT: Final[str] = "app:compaction_count"
STATE_TEMP_HITL_CONFIRMED_WIPE: Final[str] = "temp:hitl_confirmed_wipe"
STATE_TEMP_PENDING_WIPE_REQUEST: Final[str] = "temp:pending_wipe_request"
STATE_TEMP_LAST_AUDIO_ARTIFACT: Final[str] = "temp:last_audio_artifact_path"

# ==============================================================================
# Context Compaction Constants
# ==============================================================================
COMPACTION_TOKEN_THRESHOLD: Final[int] = int(
    os.getenv("COMPACTION_TOKEN_THRESHOLD", "5000")
)
EVENT_RETENTION_SIZE: Final[int] = int(os.getenv("EVENT_RETENTION_SIZE", "5"))
COMPACTION_INTERVAL: Final[int] = int(os.getenv("COMPACTION_INTERVAL", "10"))
COMPACTION_OVERLAP_SIZE: Final[int] = int(os.getenv("COMPACTION_OVERLAP_SIZE", "2"))

# ==============================================================================
# High-Stakes Tools Requiring Human-in-the-Loop (HITL) Confirmation
# ==============================================================================
HIGH_STAKES_TOOLS: Final[frozenset[str]] = frozenset({"wipe_all_user_memories"})


@dataclass(frozen=True)
class GpalConfig:
    """Immutable runtime configuration for the GPAL Agent application."""

    app_name: str = "gpal_agent"
    environment: str = field(
        default_factory=lambda: os.getenv("GPAL_ENVIRONMENT", "dev")
    )
    project_id: str = field(
        default_factory=lambda: os.getenv("GOOGLE_CLOUD_PROJECT", "")
    )
    location: str = field(
        default_factory=lambda: os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    )
    use_vertexai: bool = field(
        default_factory=lambda: os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "TRUE").upper()
        in ("TRUE", "1", "YES")
    )

    # Strategic Model Routing Tiers
    fast_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_FAST_MODEL", "gemini-2.5-flash")
    )
    pro_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_PRO_MODEL", "gemini-2.5-pro")
    )
    tts_primary_model: str = field(
        default_factory=lambda: os.getenv(
            "GEMINI_TTS_PRIMARY_MODEL", "gemini-3.8-flash-lite-tts"
        )
    )
    tts_expressive_model: str = field(
        default_factory=lambda: os.getenv(
            "GEMINI_TTS_EXPRESSIVE_MODEL", "gemini-3.8-flash-tts"
        )
    )
    tts_fallback_model: str = field(
        default_factory=lambda: os.getenv(
            "GEMINI_TTS_FALLBACK_MODEL", "gemini-2.5-flash-preview-tts"
        )
    )
    live_voice_model: str = field(
        default_factory=lambda: os.getenv(
            "GEMINI_LIVE_VOICE_MODEL", "gemini-live-2.5-flash-native-audio"
        )
    )

    # Voice & Audio Output Settings
    default_voice: str = field(default_factory=lambda: DEFAULT_GEMINI_VOICE)
    auto_synthesize_voice_reply: bool = field(
        default_factory=lambda: os.getenv(
            "AUTO_SYNTHESIZE_VOICE_REPLY", "TRUE"
        ).upper()
        in ("TRUE", "1", "YES")
    )
    audio_output_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("GPAL_AUDIO_OUTPUT_DIR", "artifacts/audio_replies")
        )
    )

    # Memory & Compaction Settings
    use_vertex_session_service: bool = field(
        default_factory=lambda: os.getenv(
            "USE_VERTEX_SESSION_SERVICE", "FALSE"
        ).upper()
        in ("TRUE", "1", "YES")
    )
    session_db_url: str = field(
        default_factory=lambda: os.getenv(
            "SESSION_DB_URL", "sqlite+aiosqlite:///./gpal_sessions.db"
        )
    )
    compaction_token_threshold: int = COMPACTION_TOKEN_THRESHOLD
    event_retention_size: int = EVENT_RETENTION_SIZE

    # Security, Model Armor & Cloud DLP
    model_armor_template_id: str = field(
        default_factory=lambda: os.getenv("MODEL_ARMOR_TEMPLATE_ID", "")
    )
    enable_cloud_dlp: bool = field(
        default_factory=lambda: os.getenv("ENABLE_CLOUD_DLP", "FALSE").upper()
        in ("TRUE", "1", "YES")
    )
    dlp_inspect_template_name: str = field(
        default_factory=lambda: os.getenv("DLP_INSPECT_TEMPLATE_NAME", "")
    )
    dlp_deidentify_template_name: str = field(
        default_factory=lambda: os.getenv("DLP_DEIDENTIFY_TEMPLATE_NAME", "")
    )
    secret_manager_secret_id: str = field(
        default_factory=lambda: os.getenv(
            "SECRET_MANAGER_SECRET_ID", "gpal-api-key-dev"
        )
    )

    # Observability & Tracing
    enable_cloud_trace: bool = field(
        default_factory=lambda: os.getenv("ENABLE_CLOUD_TRACE", "TRUE").upper()
        in ("TRUE", "1", "YES")
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper()
    )
    staging_bucket_uri: str = field(
        default_factory=lambda: os.getenv("STAGING_BUCKET_URI", "")
    )


def get_config() -> GpalConfig:
    """Return a fresh GpalConfig instance reflecting current environment variables."""
    return GpalConfig()


# Keywords indicating a query benefits from deep multi-step analytical reasoning (Pro tier)
_DEEP_REASONING_INDICATORS: Final[tuple[str, ...]] = (
    "compare and contrast",
    "pros and cons",
    "five-year plan",
    "5-year plan",
    "career strategy",
    "financial plan",
    "trade-offs",
    "tradeoffs",
    "step-by-step plan",
    "deep analysis",
    "architectural decision",
    "ethical dilemma",
    "long-term strategy",
)


def select_optimal_model_tier(
    user_query: str, config: GpalConfig | None = None
) -> dict[str, str]:
    """Strategically route a user request to the optimal Gemini model tier.

    Fast conversational turns route to `gemini-2.5-flash` for minimal latency,
    while complex planning or multi-step analytical queries route to `gemini-2.5-pro`.

    Args:
        user_query: The sanitized user input text.
        config: Optional GpalConfig instance.

    Returns:
        A dictionary containing `selected_model`, `tier`, and `routing_reason`.
    """
    cfg = config or get_config()
    normalized = (user_query or "").strip().lower()

    matched_indicator = next(
        (ind for ind in _DEEP_REASONING_INDICATORS if ind in normalized),
        None,
    )
    is_long_analytical_query = len(normalized.split()) >= 60 and (
        "analyze" in normalized or "plan" in normalized or "strategy" in normalized
    )

    if matched_indicator or is_long_analytical_query:
        reason = (
            f"Matched complexity indicator '{matched_indicator}'"
            if matched_indicator
            else "Long analytical planning query (>=60 words)"
        )
        return {
            "selected_model": cfg.pro_model,
            "tier": "pro_reasoning",
            "routing_reason": reason,
        }

    return {
        "selected_model": cfg.fast_model,
        "tier": "fast_conversational",
        "routing_reason": "Standard conversational turn optimized for low latency",
    }


# ==============================================================================
# Google Cloud ADC Reauthentication Preflight & Immediate Exit Guard
# ==============================================================================
def is_reauth_error(exc: BaseException | None) -> bool:
    """Return True if `exc` indicates expired ADC credentials requiring reauthentication."""
    if exc is None:
        return False
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        cls_name = type(current).__name__
        msg = str(current).lower()
        if (
            cls_name == "RefreshError"
            or "reauthentication is needed" in msg
            or "gcloud auth application-default login" in msg
            or "invalid_grant" in msg
            or "invalid_rapt" in msg
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def terminate_on_reauth_error(exc: BaseException) -> None:
    """Write an actionable reauthentication banner to stderr and exit immediately.

    Uses `os._exit(1)` outside of pytest so background gRPC / OpenTelemetry threads
    and uvicorn workers cannot trap `SystemExit` or block `Ctrl+C` (`SIGINT`).
    """
    import sys

    banner = (
        "\n"
        + "=" * 78
        + "\n[GPAL AUTH ERROR] Google Cloud Application Default Credentials (ADC) expired!\n"
        + f"Details: {exc}\n\n"
        + "Please reauthenticate in your terminal first:\n"
        + "  gcloud auth application-default login\n"
        + "and then re-run your command.\n"
        + "=" * 78
        + "\n"
    )
    sys.stderr.write(banner)
    sys.stderr.flush()

    in_pytest = "pytest" in sys.modules
    disable_hard_exit = os.getenv(
        "GPAL_DISABLE_HARD_EXIT_ON_REAUTH", "FALSE"
    ).upper() in ("TRUE", "1", "YES")

    if in_pytest or disable_hard_exit:
        raise SystemExit(1) from exc
    os._exit(1)


def verify_gcp_adc_or_exit(
    config: GpalConfig | None = None,
    *,
    force_check: bool = False,
) -> bool:
    """Verify Google Cloud ADC credentials can refresh without `RefreshError`, or exit immediately.

    Args:
        config: Optional `GpalConfig` instance.
        force_check: When True, runs the check even inside offline test contexts.

    Returns:
        `True` when credentials are valid (or cloud calls are disabled), `False` if offline.
    """
    import socket
    import sys

    cfg = config or get_config()

    if not force_check:
        if "pytest" in sys.modules:
            return True
        if os.getenv("GPAL_OFFLINE_TTS_FALLBACK", "FALSE").upper() in (
            "TRUE",
            "1",
            "YES",
        ):
            return True
        if os.getenv("GPAL_SKIP_AUTH_PREFLIGHT", "FALSE").upper() in (
            "TRUE",
            "1",
            "YES",
        ):
            return True

    needs_gcp_auth = (
        (cfg.use_vertexai and bool(cfg.project_id))
        or (cfg.enable_cloud_trace and bool(cfg.project_id))
        or (cfg.enable_cloud_dlp and bool(cfg.project_id))
        or bool(cfg.model_armor_template_id)
    )
    if not needs_gcp_auth and not force_check:
        return True

    if not force_check:
        try:
            with socket.create_connection(("oauth2.googleapis.com", 443), timeout=0.5):
                pass
        except OSError:
            return False

    try:
        import google.auth
        import google.auth.transport.requests

        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        if not getattr(credentials, "valid", False):
            request = google.auth.transport.requests.Request()
            credentials.refresh(request)
        return True
    except BaseException as exc:
        if is_reauth_error(exc):
            terminate_on_reauth_error(exc)
        return False


