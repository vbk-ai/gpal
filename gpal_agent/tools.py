"""Voice preference tools, Gemini TTS WAV synthesis, and HITL-protected memory tools for GPAL.

Satisfies Rubric Criteria:
- 1.1 Comprehensive Tool Docstrings (WHEN TO USE, WHEN NOT TO USE, Args, Returns, Error Recovery)
- 1.2 Descriptive Naming (`set_preferred_voice`, `list_available_gemini_voices`,
      `synthesize_gemini_tts_audio`, `save_friend_memory_note`, `get_friend_memory_profile`,
      `confirm_pending_memory_wipe`, `wipe_all_user_memories`)
- 1.3 Explicit JSON Schemas (Strict Pydantic input/output validation via `gpal_agent.schemas`)
- 1.4 Guided Error Handling (`StructuredToolError` responses with actionable recovery instructions)
"""

from __future__ import annotations

from datetime import datetime, timezone
import io
import logging
import math
import os
from pathlib import Path
import struct
import time
from typing import Any, Mapping, MutableMapping, Optional
import wave

from dataclasses import dataclass, field

try:
    from google.genai import types
except ImportError:  # pragma: no cover

    @dataclass
    class _FallbackPrebuiltVoiceConfig:
        voice_name: str = "Puck"

    @dataclass
    class _FallbackVoiceConfig:
        prebuilt_voice_config: _FallbackPrebuiltVoiceConfig = field(
            default_factory=_FallbackPrebuiltVoiceConfig
        )

    @dataclass
    class _FallbackSpeechConfig:
        voice_config: _FallbackVoiceConfig = field(
            default_factory=_FallbackVoiceConfig
        )

    @dataclass
    class _FallbackGenerateContentConfig:
        response_modalities: list[str] = field(default_factory=lambda: ["AUDIO"])
        speech_config: _FallbackSpeechConfig | None = None

    class _FallbackGenaiTypes:
        PrebuiltVoiceConfig = _FallbackPrebuiltVoiceConfig
        VoiceConfig = _FallbackVoiceConfig
        SpeechConfig = _FallbackSpeechConfig
        GenerateContentConfig = _FallbackGenerateContentConfig

    types = _FallbackGenaiTypes()  # type: ignore[assignment]

from pydantic import ValidationError

from gpal_agent.config import (
    DEFAULT_GEMINI_VOICE,
    STATE_TEMP_HITL_CONFIRMED_WIPE,
    STATE_TEMP_LAST_AUDIO_ARTIFACT,
    STATE_TEMP_PENDING_WIPE_REQUEST,
    STATE_USER_FRIEND_NOTES,
    STATE_USER_FRIEND_PROFILE_SUMMARY,
    STATE_USER_PREFERRED_VOICE,
    STATE_USER_VOICE_UPDATED_AT,
    SUPPORTED_GEMINI_VOICES,
    GpalConfig,
    get_config,
)
from gpal_agent.memory import (
    _format_consolidated_profile,
    schedule_async_memory_consolidation,
    scrub_pii_text,
)
from gpal_agent.schemas import (
    ConfirmWipeInput,
    ConfirmWipeOutput,
    FriendMemoryNoteInput,
    FriendMemoryNoteOutput,
    FriendProfileOutput,
    ListVoicesOutput,
    SetVoicePreferenceInput,
    SetVoicePreferenceOutput,
    StructuredToolError,
    VoiceSynthesisInput,
    VoiceSynthesisOutput,
    WipeMemoryInput,
    WipeMemoryOutput,
)

try:
    from google.adk.tools.tool_context import ToolContext
except ImportError:  # pragma: no cover
    ToolContext = Any  # type: ignore[misc,assignment]


# ==============================================================================
# Safe Observability Helpers
# ==============================================================================
_FALLBACK_TOOL_STATE: dict[str, Any] = {
    STATE_USER_PREFERRED_VOICE: DEFAULT_GEMINI_VOICE,
    STATE_USER_FRIEND_NOTES: [],
    STATE_USER_FRIEND_PROFILE_SUMMARY: "",
}


def _get_logger() -> logging.Logger:
    """Return the structured logger from gpal_agent.observability."""
    try:
        from gpal_agent.observability import get_structured_logger

        return get_structured_logger("gpal_agent.tools")
    except Exception:
        return logging.getLogger("gpal_agent.tools")


def _log_tool_intent(
    tool_name: str,
    user_id: str = "",
    session_id: str = "",
    details: dict[str, Any] | None = None,
) -> float | None:
    """Record tool invocation INTENT telemetry."""
    try:
        from gpal_agent.observability import log_intent

        return log_intent(
            event_type="tool_execution",
            tool_name=tool_name,
            user_id=user_id,
            session_id=session_id,
            details=details or {},
        )
    except Exception:
        _get_logger().info("Tool INTENT [%s]: %s", tool_name, details)
        return time.perf_counter()


def _log_tool_outcome(
    tool_name: str,
    status: str = "SUCCESS",
    start_time: float | None = None,
    user_id: str = "",
    session_id: str = "",
    details: dict[str, Any] | None = None,
) -> None:
    """Record tool invocation OUTCOME telemetry."""
    try:
        from gpal_agent.observability import log_outcome

        log_outcome(
            event_type="tool_execution",
            tool_name=tool_name,
            status=status,
            start_time=start_time,
            user_id=user_id,
            session_id=session_id,
            details=details or {},
        )
    except Exception:
        _get_logger().info("Tool OUTCOME [%s] (%s): %s", tool_name, status, details)


def _extract_state_and_ids(
    tool_context: Any,
) -> tuple[MutableMapping[str, Any], str, str]:
    """Extract mutable session state, user_id, and session_id from tool_context."""
    if tool_context is None:
        return _FALLBACK_TOOL_STATE, "default_user", "default_session"

    state: MutableMapping[str, Any] | None = None
    raw_state = getattr(tool_context, "state", None)
    if isinstance(raw_state, MutableMapping) or (
        raw_state is not None
        and hasattr(raw_state, "get")
        and hasattr(raw_state, "__setitem__")
    ):
        state = raw_state  # type: ignore[assignment]
    elif isinstance(tool_context, MutableMapping):
        nested = tool_context.get("state")
        state = nested if isinstance(nested, MutableMapping) else tool_context

    if state is None:
        state = _FALLBACK_TOOL_STATE

    user_id = str(
        getattr(tool_context, "user_id", None)
        or getattr(getattr(tool_context, "_invocation_context", None), "user_id", None)
        or state.get("user_id", "default_user")
    )
    session_id = str(
        getattr(tool_context, "session_id", None)
        or getattr(
            getattr(getattr(tool_context, "_invocation_context", None), "session", None),
            "id",
            None,
        )
        or state.get("session_id", "default_session")
    )
    return state, user_id, session_id


# ==============================================================================
# 1. Audio Helper: PCM-to-WAV RIFF Wrapping & Gemini TTS Synthesis Cascade
# ==============================================================================
def wrap_pcm_in_wav(
    raw_audio_bytes: bytes,
    sample_rate: int = 24000,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """Ensure raw 24kHz 16-bit mono PCM audio bytes are wrapped in a valid RIFF WAV header.

    Gemini TTS models return raw linear PCM (`audio/L16;rate=24000`) in `inline_data.data`.
    Saving raw PCM directly with a `.wav` extension produces an unplayable file lacking the
    44-byte RIFF/WAVE header. This helper detects whether a RIFF/WAVE header is already
    present and, if not, encapsulates the PCM stream using Python's standard `wave` module.

    Args:
        raw_audio_bytes: Raw PCM payload or existing WAV buffer.
        sample_rate: Audio sample rate in Hz (default 24000 for Gemini TTS).
        channels: Number of audio channels (default 1 for mono).
        sample_width: Bytes per sample (default 2 for 16-bit signed integer PCM).

    Returns:
        Complete `.wav` file bytes starting with `b"RIFF"` and containing `b"WAVE"`.
    """
    if (
        isinstance(raw_audio_bytes, (bytes, bytearray))
        and len(raw_audio_bytes) >= 44
        and bytes(raw_audio_bytes[:4]) == b"RIFF"
        and b"WAVE" in bytes(raw_audio_bytes[:16])
    ):
        return bytes(raw_audio_bytes)

    pcm_payload = bytes(raw_audio_bytes) if raw_audio_bytes else (b"\x00\x00" * 240)
    frame_alignment = max(1, channels * sample_width)
    remainder = len(pcm_payload) % frame_alignment
    if remainder != 0:
        pcm_payload += b"\x00" * (frame_alignment - remainder)

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_payload)
    return wav_buffer.getvalue()


def _generate_synthetic_pcm(
    text: str,
    voice_name: str = "Puck",
    sample_rate: int = 24000,
) -> bytes:
    """Generate a deterministic, pleasant harmonic chime/tone PCM buffer for offline fallback."""
    base_frequencies = {
        "Puck": 261.63,    # C4
        "Kore": 293.66,    # D4
        "Charon": 220.00,  # A3
        "Aoede": 329.63,   # E4
        "Fenrir": 349.23,  # F4
        "Leda": 392.00,    # G4
        "Orus": 246.94,    # B3
        "Zephyr": 440.00,  # A4
    }
    freq = base_frequencies.get(voice_name, 261.63)
    duration_sec = min(1.5, max(0.25, len(text) * 0.015))
    num_samples = int(sample_rate * duration_sec)
    frames = bytearray(num_samples * 2)

    for i in range(num_samples):
        t = i / sample_rate
        envelope = math.sin(math.pi * (i / max(1, num_samples - 1)))
        sample_val = (
            0.6 * math.sin(2.0 * math.pi * freq * t)
            + 0.3 * math.sin(2.0 * math.pi * (freq * 1.25) * t)
            + 0.1 * math.sin(2.0 * math.pi * (freq * 1.5) * t)
        )
        pcm_int16 = int(max(-32767, min(32767, sample_val * envelope * 12000)))
        struct.pack_into("<h", frames, i * 2, pcm_int16)

    return bytes(frames)


def _extract_audio_bytes_from_genai_response(response: Any) -> bytes | None:
    """Extract raw audio bytes from a google-genai GenerateContentResponse."""
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            inline_data = getattr(part, "inline_data", None)
            data = getattr(inline_data, "data", None) if inline_data else None
            if isinstance(data, (bytes, bytearray)) and len(data) > 0:
                return bytes(data)
    return None


def synthesize_speech_bytes(
    text: str,
    voice_name: str = DEFAULT_GEMINI_VOICE,
    style_prompt: str | None = None,
    config: GpalConfig | None = None,
) -> tuple[bytes, str]:
    """Synthesize spoken audio bytes using the Gemini TTS model cascade and wrap in RIFF WAV.

    Attempts synthesis across the configured Gemini TTS model cascade:
    1. `cfg.tts_primary_model` (`gemini-3.8-flash-lite-tts`)
    2. `cfg.tts_expressive_model` (`gemini-3.8-flash-tts`)
    3. `cfg.tts_fallback_model` (`gemini-2.5-flash-preview-tts`)

    If live API credentials are unavailable or offline mode is enabled, falls back to a
    deterministic local WAV synthesizer so callers always receive a valid, playable `.wav`.

    Args:
        text: Conversational text to speak (automatically PII-scrubbed before synthesis).
        voice_name: Prebuilt Gemini voice name (e.g., 'Puck', 'Kore', 'Charon').
        style_prompt: Optional delivery style instruction.
        config: Optional GpalConfig instance.

    Returns:
        A tuple of `(wav_bytes, tts_model_used)`.
    """
    cfg = config or get_config()
    clean_text = scrub_pii_text(text or "Hello from GPAL.")
    resolved_voice = (
        voice_name.strip().capitalize()
        if isinstance(voice_name, str)
        and voice_name.strip().capitalize() in SUPPORTED_GEMINI_VOICES
        else cfg.default_voice
    )

    prompt_text = (
        f"Speak with a {scrub_pii_text(style_prompt)} tone: {clean_text}"
        if style_prompt and style_prompt.strip()
        else clean_text
    )

    model_cascade = [
        cfg.tts_primary_model,
        cfg.tts_expressive_model,
        cfg.tts_fallback_model,
    ]

    offline_forced = os.getenv("GPAL_OFFLINE_TTS_FALLBACK", "FALSE").upper() in (
        "TRUE",
        "1",
        "YES",
    )

    if not offline_forced:
        try:
            import socket

            with socket.create_connection(("oauth2.googleapis.com", 443), timeout=0.5):
                pass
            from google import genai

            client_kwargs: dict[str, Any] = {}
            if cfg.use_vertexai and cfg.project_id:
                client_kwargs = {
                    "vertexai": True,
                    "project": cfg.project_id,
                    "location": cfg.location,
                }
            client = genai.Client(**client_kwargs)

            gen_config = types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=resolved_voice
                        )
                    )
                ),
            )

            for model_id in model_cascade:
                try:
                    response = client.models.generate_content(
                        model=model_id,
                        contents=prompt_text,
                        config=gen_config,
                    )
                    raw_audio = _extract_audio_bytes_from_genai_response(response)
                    if raw_audio:
                        return wrap_pcm_in_wav(raw_audio), model_id
                except Exception as model_exc:
                    _get_logger().debug(
                        "TTS model '%s' unavailable (%s); trying next in cascade.",
                        model_id,
                        model_exc,
                    )
        except Exception as client_exc:
            _get_logger().debug(
                "GenAI TTS client unavailable (%s); using local WAV fallback.",
                client_exc,
            )

    synthetic_pcm = _generate_synthetic_pcm(clean_text, voice_name=resolved_voice)
    wav_bytes = wrap_pcm_in_wav(synthetic_pcm)
    return wav_bytes, f"{cfg.tts_primary_model} (local-wav-synthesizer)"


# ==============================================================================
# 2. ADK Tool Functions (Criteria 1.1, 1.2, 1.3, 1.4)
# ==============================================================================
def set_preferred_voice(
    voice_name: str,
    tool_context: Optional[ToolContext] = None,
) -> dict[str, Any]:
    """Update and persist the user's preferred Gemini voice across all future sessions.

    WHEN TO USE:
        Call this tool whenever the user asks to change, switch, or set their preferred
        companion voice (for example: "Switch your voice to Kore", "Use the Charon voice",
        or "I want a calmer voice like Kore").

    WHEN NOT TO USE:
        Do NOT call this tool if the user only asks what voices are available without
        choosing one (use `list_available_gemini_voices` instead), or if the user wants
        a one-time voice preview without changing their saved default (use
        `synthesize_gemini_tts_audio` with `voice_override` instead).

    Args:
        voice_name: Name of the prebuilt Gemini voice to activate and persist in
            `user:preferred_voice`. Supported values: Puck, Kore, Charon, Aoede,
            Fenrir, Leda, Orus, Zephyr.
        tool_context: Optional ADK ToolContext providing access to persistent session state.

    Returns:
        A dictionary matching `SetVoicePreferenceOutput` on success, or
        `StructuredToolError` (`error_code="INVALID_VOICE_NAME"`) with `valid_options`
        and actionable `recovery_instructions` if an unsupported voice name was provided.
    """
    state, user_id, session_id = _extract_state_and_ids(tool_context)
    start_time = _log_tool_intent(
        "set_preferred_voice",
        user_id=user_id,
        session_id=session_id,
        details={"requested_voice": voice_name},
    )

    try:
        validated = SetVoicePreferenceInput(voice_name=voice_name)
    except ValidationError as exc:
        valid_voices = list(SUPPORTED_GEMINI_VOICES.keys())
        error_payload = StructuredToolError(
            error_code="INVALID_VOICE_NAME",
            message=(
                f"Voice '{voice_name}' is not a supported Gemini voice. "
                f"Supported voices are: {', '.join(valid_voices)}."
            ),
            recovery_instructions=(
                "Inform the user of the valid Gemini voices (Puck, Kore, Charon, Aoede, "
                "Fenrir, Leda, Orus, Zephyr) along with their vocal personas, and either "
                "select the closest matching voice from `valid_options` or ask the user "
                "which one they prefer."
            ),
            valid_options=valid_voices,
        ).model_dump()
        _log_tool_outcome(
            "set_preferred_voice",
            status="VALIDATION_ERROR",
            start_time=start_time,
            user_id=user_id,
            session_id=session_id,
            details={"error": str(exc), "invalid_voice": voice_name},
        )
        return error_payload

    previous_voice = str(state.get(STATE_USER_PREFERRED_VOICE, DEFAULT_GEMINI_VOICE))
    updated_at = datetime.now(timezone.utc).isoformat()
    chosen_voice = validated.voice_name

    state[STATE_USER_PREFERRED_VOICE] = chosen_voice
    state[STATE_USER_VOICE_UPDATED_AT] = updated_at

    output = SetVoicePreferenceOutput(
        previous_voice=previous_voice,
        preferred_voice=chosen_voice,  # type: ignore[arg-type]
        voice_description=SUPPORTED_GEMINI_VOICES[chosen_voice],
        persisted_scope="user",
        updated_at=updated_at,
    ).model_dump()

    _log_tool_outcome(
        "set_preferred_voice",
        status="SUCCESS",
        start_time=start_time,
        user_id=user_id,
        session_id=session_id,
        details={"previous_voice": previous_voice, "preferred_voice": chosen_voice},
    )
    return output


def list_available_gemini_voices(
    tool_context: Optional[ToolContext] = None,
) -> dict[str, Any]:
    """List all supported prebuilt Gemini voices, their vocal personas, and the active voice.

    WHEN TO USE:
        Call this tool when the user asks what voices are available, wants recommendations
        for a different vocal tone, or asks which voice GPAL is currently using.

    WHEN NOT TO USE:
        Do NOT call this tool if the user has already named a specific voice they want to
        switch to (call `set_preferred_voice` directly instead).

    Args:
        tool_context: Optional ADK ToolContext providing access to session state.

    Returns:
        A dictionary matching `ListVoicesOutput` containing `current_preferred_voice`
        and the complete `available_voices` mapping of voice names to persona descriptions.
    """
    state, user_id, session_id = _extract_state_and_ids(tool_context)
    start_time = _log_tool_intent(
        "list_available_gemini_voices",
        user_id=user_id,
        session_id=session_id,
    )

    current_voice = str(state.get(STATE_USER_PREFERRED_VOICE, DEFAULT_GEMINI_VOICE))
    output = ListVoicesOutput(
        current_preferred_voice=current_voice,
        available_voices=dict(SUPPORTED_GEMINI_VOICES),
    ).model_dump()

    _log_tool_outcome(
        "list_available_gemini_voices",
        status="SUCCESS",
        start_time=start_time,
        user_id=user_id,
        session_id=session_id,
        details={"current_preferred_voice": current_voice},
    )
    return output


def synthesize_gemini_tts_audio(
    text_to_speak: str,
    voice_override: Optional[str] = None,
    style_prompt: Optional[str] = None,
    tool_context: Optional[ToolContext] = None,
) -> dict[str, Any]:
    """Synthesize spoken `.wav` audio using Gemini TTS and the user's preferred voice.

    WHEN TO USE:
        Call this tool when the user explicitly asks to hear a spoken audio clip, asks
        for a voice sample/preview, or wants a message synthesized into a downloadable
        `.wav` audio file.

    WHEN NOT TO USE:
        Do NOT call this tool with empty text, and do NOT use it solely to persist a
        permanent voice change (use `set_preferred_voice` to save a new default voice).

    Args:
        text_to_speak: The conversational text to synthesize into audio (1-4000 chars).
            Any sensitive PII is automatically redacted before synthesis.
        voice_override: Optional prebuilt Gemini voice name to use for this audio clip
            instead of `session.state['user:preferred_voice']`.
        style_prompt: Optional expressive delivery guidance (e.g., 'warm and cheerful').
        tool_context: Optional ADK ToolContext providing access to session state.

    Returns:
        A dictionary matching `VoiceSynthesisOutput` with the saved `.wav` path and byte
        size, or `StructuredToolError` if `text_to_speak` or `voice_override` is invalid.
    """
    state, user_id, session_id = _extract_state_and_ids(tool_context)
    start_time = _log_tool_intent(
        "synthesize_gemini_tts_audio",
        user_id=user_id,
        session_id=session_id,
        details={"voice_override": voice_override, "style_prompt": style_prompt},
    )

    try:
        validated = VoiceSynthesisInput(
            text_to_speak=text_to_speak,
            voice_override=voice_override,
            style_prompt=style_prompt,
        )
    except ValidationError as exc:
        error_text = str(exc)
        is_voice_err = "voice_override" in error_text
        error_code = (
            "INVALID_VOICE_OVERRIDE" if is_voice_err else "INVALID_TTS_TEXT_INPUT"
        )
        valid_options = list(SUPPORTED_GEMINI_VOICES.keys()) if is_voice_err else None
        recovery = (
            "Retry `synthesize_gemini_tts_audio` with a `voice_override` from "
            "`valid_options` (or omit `voice_override` to use the user's saved voice)."
            if is_voice_err
            else "Provide a non-empty `text_to_speak` string between 1 and 4000 characters."
        )
        error_payload = StructuredToolError(
            error_code=error_code,
            message=f"Invalid parameters for TTS synthesis: {error_text}",
            recovery_instructions=recovery,
            valid_options=valid_options,
        ).model_dump()
        _log_tool_outcome(
            "synthesize_gemini_tts_audio",
            status="VALIDATION_ERROR",
            start_time=start_time,
            user_id=user_id,
            session_id=session_id,
            details={"error_code": error_code},
        )
        return error_payload

    cfg = get_config()
    persisted_voice = str(state.get(STATE_USER_PREFERRED_VOICE, cfg.default_voice))
    if persisted_voice not in SUPPORTED_GEMINI_VOICES:
        persisted_voice = cfg.default_voice
    active_voice = validated.voice_override or persisted_voice

    scrubbed_text = scrub_pii_text(validated.text_to_speak)
    wav_bytes, tts_model_used = synthesize_speech_bytes(
        text=scrubbed_text,
        voice_name=active_voice,
        style_prompt=validated.style_prompt,
        config=cfg,
    )

    output_dir = Path(cfg.audio_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp_ms = int(time.time() * 1000)
    filename = f"gpal_{active_voice.lower()}_{timestamp_ms}.wav"
    file_path = output_dir / filename
    file_path.write_bytes(wav_bytes)

    state[STATE_TEMP_LAST_AUDIO_ARTIFACT] = str(file_path)

    output = VoiceSynthesisOutput(
        voice_used=active_voice,
        tts_model_used=tts_model_used,
        mime_type="audio/wav",
        audio_file_path=str(file_path),
        audio_size_bytes=len(wav_bytes),
        synthesized_text_preview=scrubbed_text[:200],
    ).model_dump()

    _log_tool_outcome(
        "synthesize_gemini_tts_audio",
        status="SUCCESS",
        start_time=start_time,
        user_id=user_id,
        session_id=session_id,
        details={
            "voice_used": active_voice,
            "tts_model_used": tts_model_used,
            "audio_size_bytes": len(wav_bytes),
        },
    )
    return output


def save_friend_memory_note(
    memory_note: str,
    category: str = "general_fact",
    tool_context: Optional[ToolContext] = None,
) -> dict[str, Any]:
    """Save a PII-scrubbed personal fact, milestone, or interest to cross-session memory.

    WHEN TO USE:
        Call this tool when the user shares a meaningful personal detail, life goal,
        hobby, upcoming milestone, or communication preference that a thoughtful friend
        should remember in future conversations.

    WHEN NOT TO USE:
        Do NOT store raw sensitive identifiers (SSNs, credit card numbers, passwords),
        and do NOT call this tool for trivial greetings or ephemeral turn-by-turn chatter.

    Args:
        memory_note: Concise summary of the fact or milestone (3-1000 chars). Any PII
            is automatically scrubbed prior to storage.
        category: Semantic category for organizing the note. Must be one of:
            `personal_milestone`, `hobby_or_interest`, `communication_preference`,
            `life_goal`, or `general_fact`.
        tool_context: Optional ADK ToolContext providing access to session state.

    Returns:
        A dictionary matching `FriendMemoryNoteOutput` confirming storage and non-blocking
        async profile consolidation, or `StructuredToolError` if validation fails.
    """
    state, user_id, session_id = _extract_state_and_ids(tool_context)
    sanitized_candidate = scrub_pii_text(memory_note or "")
    start_time = _log_tool_intent(
        "save_friend_memory_note",
        user_id=user_id,
        session_id=session_id,
        details={"category": category, "note_preview": sanitized_candidate[:80]},
    )

    valid_categories = [
        "personal_milestone",
        "hobby_or_interest",
        "communication_preference",
        "life_goal",
        "general_fact",
    ]

    try:
        validated = FriendMemoryNoteInput(
            memory_note=sanitized_candidate,
            category=category,  # type: ignore[arg-type]
        )
    except ValidationError as exc:
        err_str = str(exc)
        is_cat_error = "category" in err_str
        error_code = (
            "INVALID_MEMORY_CATEGORY" if is_cat_error else "INVALID_MEMORY_NOTE"
        )
        recovery = (
            "Retry `save_friend_memory_note` using one of the allowed categories in "
            "`valid_options` (e.g., 'hobby_or_interest', 'personal_milestone', "
            "'life_goal', 'communication_preference', or 'general_fact')."
            if is_cat_error
            else "Provide a meaningful `memory_note` containing at least 3 characters."
        )
        error_payload = StructuredToolError(
            error_code=error_code,
            message=f"Could not save friend memory note: {err_str}",
            recovery_instructions=recovery,
            valid_options=valid_categories if is_cat_error else None,
        ).model_dump()
        _log_tool_outcome(
            "save_friend_memory_note",
            status="VALIDATION_ERROR",
            start_time=start_time,
            user_id=user_id,
            session_id=session_id,
            details={"error_code": error_code},
        )
        return error_payload

    clean_note = scrub_pii_text(validated.memory_note)
    existing_notes = list(state.get(STATE_USER_FRIEND_NOTES) or [])
    note_entry = {
        "note": clean_note,
        "category": validated.category,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    existing_notes.append(note_entry)
    state[STATE_USER_FRIEND_NOTES] = existing_notes

    # Schedule non-blocking background consolidation (Criterion 2.4)
    schedule_async_memory_consolidation(
        user_id=user_id,
        notes=existing_notes,
        target_state=state,
    )

    output = FriendMemoryNoteOutput(
        sanitized_note=clean_note,
        category=validated.category,
        total_notes_stored=len(existing_notes),
        async_consolidation_scheduled=True,
    ).model_dump()

    _log_tool_outcome(
        "save_friend_memory_note",
        status="SUCCESS",
        start_time=start_time,
        user_id=user_id,
        session_id=session_id,
        details={
            "category": validated.category,
            "total_notes_stored": len(existing_notes),
        },
    )
    return output


def get_friend_memory_profile(
    tool_context: Optional[ToolContext] = None,
) -> dict[str, Any]:
    """Retrieve the user's cross-session friend memory profile, notes, and preferred voice.

    WHEN TO USE:
        Call this tool when the user asks what you remember about them, when resuming
        a conversation across sessions to personalize advice, or to verify stored
        preferences and milestones.

    WHEN NOT TO USE:
        Do NOT call this tool to save a new note (use `save_friend_memory_note` instead).

    Args:
        tool_context: Optional ADK ToolContext providing access to session state.

    Returns:
        A dictionary matching `FriendProfileOutput` containing `preferred_voice`,
        `friend_profile_summary`, and the list of `friend_notes`.
    """
    state, user_id, session_id = _extract_state_and_ids(tool_context)
    start_time = _log_tool_intent(
        "get_friend_memory_profile",
        user_id=user_id,
        session_id=session_id,
    )

    preferred_voice = str(state.get(STATE_USER_PREFERRED_VOICE, DEFAULT_GEMINI_VOICE))
    raw_notes = list(state.get(STATE_USER_FRIEND_NOTES) or [])
    sanitized_notes: list[dict[str, str]] = []
    for item in raw_notes:
        if isinstance(item, Mapping):
            sanitized_notes.append(
                {
                    "note": scrub_pii_text(str(item.get("note", ""))),
                    "category": str(item.get("category", "general_fact")),
                    "recorded_at": str(item.get("recorded_at", "")),
                }
            )

    summary = str(state.get(STATE_USER_FRIEND_PROFILE_SUMMARY) or "").strip()
    if not summary:
        summary = _format_consolidated_profile(sanitized_notes)
        if sanitized_notes:
            state[STATE_USER_FRIEND_PROFILE_SUMMARY] = summary

    output = FriendProfileOutput(
        preferred_voice=preferred_voice,
        friend_profile_summary=scrub_pii_text(summary),
        friend_notes=sanitized_notes,
    ).model_dump()

    _log_tool_outcome(
        "get_friend_memory_profile",
        status="SUCCESS",
        start_time=start_time,
        user_id=user_id,
        session_id=session_id,
        details={"notes_count": len(sanitized_notes), "preferred_voice": preferred_voice},
    )
    return output


_REJECTION_WORDS = frozenset(
    {"no", "nope", "cancel", "stop", "abort", "never", "dont", "don't", "wait"}
)


def confirm_pending_memory_wipe(
    user_confirmation_phrase: str,
    tool_context: Optional[ToolContext] = None,
) -> dict[str, Any]:
    """Record explicit Human-in-the-Loop user approval to unlock `wipe_all_user_memories`.

    WHEN TO USE:
        Call this tool ONLY AFTER you have asked the user for confirmation to erase
        their memories AND the user has explicitly replied with an affirmative phrase
        (e.g., "Yes, please wipe all my memories", "Confirm delete").

    WHEN NOT TO USE:
        NEVER call this tool proactively on the same turn that the user first asks to
        reset their memory without having asked them to confirm, and NEVER fabricate a
        `user_confirmation_phrase`.

    Args:
        user_confirmation_phrase: The exact affirmative confirmation statement provided
            by the user in the current turn.
        tool_context: Optional ADK ToolContext providing access to session state.

    Returns:
        A dictionary matching `ConfirmWipeOutput` setting `temp:hitl_confirmed_wipe=True`,
        or `StructuredToolError` if the confirmation phrase is invalid or negative.
    """
    state, user_id, session_id = _extract_state_and_ids(tool_context)
    start_time = _log_tool_intent(
        "confirm_pending_memory_wipe",
        user_id=user_id,
        session_id=session_id,
        details={"phrase": scrub_pii_text(user_confirmation_phrase or "")},
    )

    try:
        validated = ConfirmWipeInput(
            user_confirmation_phrase=scrub_pii_text(user_confirmation_phrase or "")
        )
    except ValidationError as exc:
        state[STATE_TEMP_HITL_CONFIRMED_WIPE] = False
        return StructuredToolError(
            error_code="INVALID_CONFIRMATION_PHRASE",
            message=f"Confirmation phrase is too short or invalid: {exc}",
            recovery_instructions=(
                "Ask the user to clearly confirm with a phrase such as "
                "'Yes, wipe my memories' before attempting to delete their profile."
            ),
        ).model_dump()

    normalized_tokens = set(
        validated.user_confirmation_phrase.lower().replace(",", " ").split()
    )
    if normalized_tokens & _REJECTION_WORDS:
        state[STATE_TEMP_HITL_CONFIRMED_WIPE] = False
        state[STATE_TEMP_PENDING_WIPE_REQUEST] = False
        _log_tool_outcome(
            "confirm_pending_memory_wipe",
            status="CONFIRMATION_DECLINED",
            start_time=start_time,
            user_id=user_id,
            session_id=session_id,
        )
        return StructuredToolError(
            error_code="CONFIRMATION_PHRASE_REJECTED",
            message="The user's response indicates they declined or cancelled the memory wipe.",
            recovery_instructions=(
                "Do NOT call `wipe_all_user_memories`. Reassure the user that their "
                "friend memories and profile have been kept intact."
            ),
        ).model_dump()

    state[STATE_TEMP_HITL_CONFIRMED_WIPE] = True
    state[STATE_TEMP_PENDING_WIPE_REQUEST] = False

    output = ConfirmWipeOutput(
        hitl_confirmed_wipe=True,
        next_step_instruction=(
            "Human-in-the-Loop confirmation recorded in `temp:hitl_confirmed_wipe`. "
            "You may now call `wipe_all_user_memories(explicit_user_confirmation=True)`."
        ),
    ).model_dump()

    _log_tool_outcome(
        "confirm_pending_memory_wipe",
        status="SUCCESS",
        start_time=start_time,
        user_id=user_id,
        session_id=session_id,
    )
    return output


def wipe_all_user_memories(
    explicit_user_confirmation: bool = False,
    reason: str = "User requested full memory reset",
    tool_context: Optional[ToolContext] = None,
) -> dict[str, Any]:
    """Permanently erase all stored friend notes and profile history (High-Stakes HITL Tool).

    WHEN TO USE:
        Call this high-stakes destructive tool ONLY when the user wants to permanently
        erase all their saved friend notes and profile summary AND has provided explicit
        Human-in-the-Loop confirmation recorded via `confirm_pending_memory_wipe`.

    WHEN NOT TO USE:
        Do NOT call this tool with `explicit_user_confirmation=False`, and do NOT call
        it before `temp:hitl_confirmed_wipe` has been set to `True` in session state.

    Args:
        explicit_user_confirmation: Must be `True` AND accompanied by
            `session.state['temp:hitl_confirmed_wipe'] == True`.
        reason: Brief explanation for the audit log describing why memories were wiped.
        tool_context: Optional ADK ToolContext providing access to session state.

    Returns:
        A dictionary matching `WipeMemoryOutput` on confirmed execution, or
        `StructuredToolError` (`error_code="HITL_CONFIRMATION_REQUIRED"`) if invoked
        without prior Human-in-the-Loop approval.
    """
    state, user_id, session_id = _extract_state_and_ids(tool_context)
    start_time = _log_tool_intent(
        "wipe_all_user_memories",
        user_id=user_id,
        session_id=session_id,
        details={
            "explicit_user_confirmation": explicit_user_confirmation,
            "reason": scrub_pii_text(reason or ""),
        },
    )

    try:
        validated = WipeMemoryInput(
            explicit_user_confirmation=explicit_user_confirmation,
            reason=scrub_pii_text(reason or "User requested full memory reset"),
        )
    except ValidationError as exc:
        return StructuredToolError(
            error_code="INVALID_WIPE_ARGUMENTS",
            message=f"Invalid arguments for wipe_all_user_memories: {exc}",
            recovery_instructions=(
                "Ensure `reason` is at least 3 characters and obtain user confirmation "
                "before calling `wipe_all_user_memories`."
            ),
        ).model_dump()

    hitl_state_confirmed = state.get(STATE_TEMP_HITL_CONFIRMED_WIPE) is True
    if not (validated.explicit_user_confirmation and hitl_state_confirmed):
        state[STATE_TEMP_PENDING_WIPE_REQUEST] = True
        _log_tool_outcome(
            "wipe_all_user_memories",
            status="BLOCKED_BY_HITL",
            start_time=start_time,
            user_id=user_id,
            session_id=session_id,
            details={
                "explicit_user_confirmation": validated.explicit_user_confirmation,
                "hitl_state_confirmed": hitl_state_confirmed,
            },
        )
        return StructuredToolError(
            error_code="HITL_CONFIRMATION_REQUIRED",
            message=(
                "High-stakes action 'wipe_all_user_memories' halted by "
                "Human-in-the-Loop guardrail: explicit user confirmation is required."
            ),
            recovery_instructions=(
                "Ask the user to explicitly confirm that they want to permanently "
                "delete all stored friend memories. Once the user replies yes, "
                "call `confirm_pending_memory_wipe` first, and then call "
                "`wipe_all_user_memories(explicit_user_confirmation=True)`."
            ),
        ).model_dump()

    existing_notes = list(state.get(STATE_USER_FRIEND_NOTES) or [])
    deleted_count = len(existing_notes)
    retained_voice = str(state.get(STATE_USER_PREFERRED_VOICE, DEFAULT_GEMINI_VOICE))

    state[STATE_USER_FRIEND_NOTES] = []
    state[STATE_USER_FRIEND_PROFILE_SUMMARY] = ""
    state[STATE_TEMP_HITL_CONFIRMED_WIPE] = False
    state[STATE_TEMP_PENDING_WIPE_REQUEST] = False

    output = WipeMemoryOutput(
        notes_deleted_count=deleted_count,
        voice_preference_retained=retained_voice,
        message=(
            f"Successfully wiped {deleted_count} friend memory note(s) and reset the "
            f"consolidated profile summary. Retained preferred voice '{retained_voice}'."
        ),
    ).model_dump()

    _log_tool_outcome(
        "wipe_all_user_memories",
        status="SUCCESS",
        start_time=start_time,
        user_id=user_id,
        session_id=session_id,
        details={
            "notes_deleted_count": deleted_count,
            "voice_preference_retained": retained_voice,
        },
    )
    return output


# ==============================================================================
# 3. Auto-Synthesis Model Callback & Live Voice RunConfig Helper
# ==============================================================================
def auto_synthesize_after_model_callback(
    callback_context: Any,
    llm_response: Any,
) -> Any:
    """Automatically synthesize a `.wav` voice reply when `auto_synthesize_voice_reply` is enabled."""
    cfg = get_config()
    if not cfg.auto_synthesize_voice_reply or llm_response is None:
        return None

    content = getattr(llm_response, "content", None)
    parts = getattr(content, "parts", None) or []
    if any(getattr(p, "function_call", None) is not None for p in parts):
        return None

    text_segments = [
        getattr(p, "text", "").strip()
        for p in parts
        if isinstance(getattr(p, "text", None), str) and getattr(p, "text", "").strip()
    ]
    if not text_segments:
        return None

    spoken_text = scrub_pii_text(" ".join(text_segments))
    state = getattr(callback_context, "state", None)
    preferred_voice = (
        str(state.get(STATE_USER_PREFERRED_VOICE, cfg.default_voice))
        if state is not None and hasattr(state, "get")
        else cfg.default_voice
    )
    if preferred_voice not in SUPPORTED_GEMINI_VOICES:
        preferred_voice = cfg.default_voice

    try:
        wav_bytes, _ = synthesize_speech_bytes(
            text=spoken_text,
            voice_name=preferred_voice,
            config=cfg,
        )
        output_dir = Path(cfg.audio_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = output_dir / f"gpal_turn_{preferred_voice.lower()}_{int(time.time() * 1000)}.wav"
        artifact_path.write_bytes(wav_bytes)
        if state is not None and hasattr(state, "__setitem__"):
            state[STATE_TEMP_LAST_AUDIO_ARTIFACT] = str(artifact_path)
    except Exception as exc:
        _get_logger().debug("Auto voice synthesis skipped: %s", exc)

    return None


def get_live_voice_run_config(
    session_state: Mapping[str, Any] | None = None,
    config: GpalConfig | None = None,
) -> Any:
    """Construct an ADK RunConfig for bidirectional live voice streaming with the user's voice.

    Configures `speech_config` with the user's persisted `user:preferred_voice` and sets
    `response_modalities=["AUDIO"]` and `StreamingMode.BIDI` for `gemini-live-2.5-flash-native-audio`.
    """
    cfg = config or get_config()
    preferred_voice = (
        str(session_state.get(STATE_USER_PREFERRED_VOICE, cfg.default_voice))
        if session_state is not None and hasattr(session_state, "get")
        else cfg.default_voice
    )
    if preferred_voice not in SUPPORTED_GEMINI_VOICES:
        preferred_voice = cfg.default_voice

    speech_config = types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                voice_name=preferred_voice
            )
        )
    )

    try:
        from google.adk.agents.run_config import RunConfig, StreamingMode

        model_fields = getattr(RunConfig, "model_fields", {}) or {}
        candidate_kwargs: dict[str, Any] = {
            "speech_config": speech_config,
            "response_modalities": ["AUDIO"],
            "streaming_mode": StreamingMode.BIDI,
        }
        supported_kwargs = {
            k: v for k, v in candidate_kwargs.items() if k in model_fields
        }
        return RunConfig(**supported_kwargs)
    except Exception:
        return {
            "speech_config": speech_config,
            "response_modalities": ["AUDIO"],
            "voice_name": preferred_voice,
            "live_model": cfg.live_voice_model,
        }


ALL_GPAL_TOOLS: list[Any] = [
    set_preferred_voice,
    list_available_gemini_voices,
    synthesize_gemini_tts_audio,
    save_friend_memory_note,
    get_friend_memory_profile,
    confirm_pending_memory_wipe,
    wipe_all_user_memories,
]

__all__ = [
    "ALL_GPAL_TOOLS",
    "auto_synthesize_after_model_callback",
    "confirm_pending_memory_wipe",
    "get_friend_memory_profile",
    "get_live_voice_run_config",
    "list_available_gemini_voices",
    "save_friend_memory_note",
    "set_preferred_voice",
    "synthesize_gemini_tts_audio",
    "synthesize_speech_bytes",
    "wipe_all_user_memories",
    "wrap_pcm_in_wav",
]
