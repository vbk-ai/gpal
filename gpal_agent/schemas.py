"""Explicit Pydantic JSON schemas for GPAL tool inputs, outputs, and guided errors."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from gpal_agent.config import SUPPORTED_GEMINI_VOICES

GeminiVoiceLiteral = Literal[
    "Puck", "Kore", "Charon", "Aoede", "Fenrir", "Leda", "Orus", "Zephyr"
]

MemoryCategoryLiteral = Literal[
    "personal_milestone",
    "hobby_or_interest",
    "communication_preference",
    "life_goal",
    "general_fact",
]


class StructuredToolError(BaseModel):
    """Standardized guided error schema returned to the LLM when a tool call fails."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["error"] = Field(
        default="error",
        description="Indicates that the tool execution encountered a recoverable error.",
    )
    error_code: str = Field(
        ...,
        min_length=3,
        max_length=64,
        description="Machine-readable error identifier (e.g., INVALID_VOICE_NAME, HITL_CONFIRMATION_REQUIRED).",
    )
    message: str = Field(
        ...,
        min_length=5,
        description="Human-readable explanation of what went wrong.",
    )
    recovery_instructions: str = Field(
        ...,
        min_length=10,
        description="Actionable instructions guiding the LLM on how to self-correct or ask the user.",
    )
    valid_options: list[str] | None = Field(
        default=None,
        description="Optional list of valid parameter values when an invalid option was supplied.",
    )


class SetVoicePreferenceInput(BaseModel):
    """Input schema for updating the user's cross-session preferred Gemini voice."""

    model_config = ConfigDict(extra="forbid")

    voice_name: str = Field(
        ...,
        min_length=2,
        max_length=32,
        description=(
            "Name of the prebuilt Gemini voice to use across sessions. "
            "Must be one of: Puck, Kore, Charon, Aoede, Fenrir, Leda, Orus, Zephyr."
        ),
    )

    @field_validator("voice_name")
    @classmethod
    def validate_voice_name(cls, value: str) -> str:
        """Normalize capitalization and verify the voice is in SUPPORTED_GEMINI_VOICES."""
        cleaned = value.strip().capitalize()
        if cleaned not in SUPPORTED_GEMINI_VOICES:
            valid_list = ", ".join(SUPPORTED_GEMINI_VOICES.keys())
            raise ValueError(
                f"Unsupported voice '{value}'. Valid Gemini voices are: {valid_list}."
            )
        return cleaned


class SetVoicePreferenceOutput(BaseModel):
    """Output schema returned after persisting the user's preferred Gemini voice."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = "success"
    previous_voice: str = Field(
        ...,
        description="The voice that was active prior to this update.",
    )
    preferred_voice: GeminiVoiceLiteral = Field(
        ...,
        description="The newly persisted Gemini voice name stored in user:preferred_voice.",
    )
    voice_description: str = Field(
        ...,
        description="Persona and vocal tone description of the selected voice.",
    )
    persisted_scope: Literal["user"] = Field(
        default="user",
        description="Indicates that the preference is persisted across all sessions for this user.",
    )
    updated_at: str = Field(
        ...,
        description="ISO-8601 UTC timestamp of when the voice preference was saved.",
    )


class ListVoicesOutput(BaseModel):
    """Output schema enumerating all supported Gemini voices and the active user choice."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = "success"
    current_preferred_voice: str = Field(
        ...,
        description="The user's currently active Gemini voice.",
    )
    available_voices: dict[str, str] = Field(
        ...,
        description="Mapping of valid Gemini voice names to their vocal persona descriptions.",
    )


class VoiceSynthesisInput(BaseModel):
    """Input schema for synthesizing speech audio using a Gemini TTS model."""

    model_config = ConfigDict(extra="forbid")

    text_to_speak: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The conversational text to synthesize into spoken audio.",
    )
    voice_override: str | None = Field(
        default=None,
        description=(
            "Optional voice name override (e.g., for previewing a voice). "
            "If omitted, uses the user's persisted session.state['user:preferred_voice']."
        ),
    )
    style_prompt: str | None = Field(
        default=None,
        max_length=256,
        description="Optional vocal delivery style instruction (e.g., 'warm and encouraging friend').",
    )

    @field_validator("text_to_speak")
    @classmethod
    def validate_non_blank_text(cls, value: str) -> str:
        """Reject whitespace-only strings."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("text_to_speak cannot be empty or whitespace only.")
        return cleaned

    @field_validator("voice_override")
    @classmethod
    def validate_optional_voice(cls, value: str | None) -> str | None:
        """Validate voice_override when provided."""
        if value is None or not value.strip():
            return None
        cleaned = value.strip().capitalize()
        if cleaned not in SUPPORTED_GEMINI_VOICES:
            valid_list = ", ".join(SUPPORTED_GEMINI_VOICES.keys())
            raise ValueError(
                f"Unsupported voice_override '{value}'. Valid voices are: {valid_list}."
            )
        return cleaned


class VoiceSynthesisOutput(BaseModel):
    """Output schema returned after synthesizing speech audio with Gemini TTS."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = "success"
    voice_used: str = Field(
        ...,
        description="The Gemini prebuilt voice used for synthesis.",
    )
    tts_model_used: str = Field(
        ...,
        description="The Gemini TTS model identifier that generated the audio.",
    )
    mime_type: str = Field(
        default="audio/wav",
        description="MIME type of the generated audio container (always audio/wav with RIFF header).",
    )
    audio_file_path: str = Field(
        ...,
        description="Path to the saved .wav audio file on disk.",
    )
    audio_size_bytes: int = Field(
        ...,
        ge=44,
        description="Total byte length of the WAV file (including 44-byte RIFF header).",
    )
    synthesized_text_preview: str = Field(
        ...,
        description="PII-scrubbed preview of the spoken text.",
    )


class FriendMemoryNoteInput(BaseModel):
    """Input schema for storing a sanitized personal note in cross-session friend memory."""

    model_config = ConfigDict(extra="forbid")

    memory_note: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description=(
            "A concise, PII-free summary of a personal fact, interest, or milestone "
            "shared by the user (e.g., 'Training for a half-marathon in October')."
        ),
    )
    category: MemoryCategoryLiteral = Field(
        default="general_fact",
        description="Semantic category of the memory note.",
    )

    @field_validator("memory_note")
    @classmethod
    def validate_note(cls, value: str) -> str:
        """Ensure note is non-empty after stripping whitespace."""
        cleaned = value.strip()
        if len(cleaned) < 3:
            raise ValueError("memory_note must contain at least 3 non-whitespace characters.")
        return cleaned


class FriendMemoryNoteOutput(BaseModel):
    """Output schema returned when a friend memory note is saved and queued for async consolidation."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = "success"
    sanitized_note: str = Field(
        ...,
        description="The PII-scrubbed note saved to user:friend_notes.",
    )
    category: MemoryCategoryLiteral = Field(
        ...,
        description="Category under which the note was stored.",
    )
    total_notes_stored: int = Field(
        ...,
        ge=1,
        description="Total number of cross-session friend notes stored for this user.",
    )
    async_consolidation_scheduled: bool = Field(
        default=True,
        description="True when background async profile consolidation was scheduled without blocking.",
    )


class FriendProfileOutput(BaseModel):
    """Output schema for inspecting the user's cross-session memory profile and voice preference."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = "success"
    preferred_voice: str = Field(
        ...,
        description="The user's persisted preferred Gemini voice.",
    )
    friend_profile_summary: str = Field(
        ...,
        description="Consolidated long-term friend profile summary.",
    )
    friend_notes: list[dict[str, str]] = Field(
        default_factory=list,
        description="List of sanitized friend memory notes recorded across sessions.",
    )


class ConfirmWipeInput(BaseModel):
    """Input schema for recording explicit Human-in-the-Loop user approval to wipe memories."""

    model_config = ConfigDict(extra="forbid")

    user_confirmation_phrase: str = Field(
        ...,
        min_length=2,
        max_length=200,
        description="The exact affirmative phrase the user provided to confirm wiping their memory.",
    )


class ConfirmWipeOutput(BaseModel):
    """Output schema returned when Human-in-the-Loop memory wipe confirmation is recorded."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = "success"
    hitl_confirmed_wipe: bool = Field(
        default=True,
        description="Indicates temp:hitl_confirmed_wipe is now True for this session turn.",
    )
    next_step_instruction: str = Field(
        ...,
        description="Instruction telling the LLM it may now call wipe_all_user_memories.",
    )


class WipeMemoryInput(BaseModel):
    """Input schema for the high-stakes tool that erases all stored user notes and profile history."""

    model_config = ConfigDict(extra="forbid")

    explicit_user_confirmation: bool = Field(
        ...,
        description="Must be True and accompanied by temp:hitl_confirmed_wipe == True in session state.",
    )
    reason: str = Field(
        default="User requested full memory reset",
        min_length=3,
        max_length=256,
        description="Brief reason why the memory reset is being executed.",
    )


class WipeMemoryOutput(BaseModel):
    """Output schema returned after successfully wiping the user's cross-session friend memories."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = "success"
    notes_deleted_count: int = Field(
        ...,
        ge=0,
        description="Number of friend memory notes permanently removed.",
    )
    voice_preference_retained: str = Field(
        ...,
        description="The user's preferred voice (retained unless explicitly changed).",
    )
    message: str = Field(
        ...,
        description="Confirmation message summarizing the memory reset.",
    )

