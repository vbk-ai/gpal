#!/usr/bin/env python3
"""End-to-End Multi-Session Demonstration for GPAL ("Google Pal").

Demonstrates GPAL's core architectural guarantees locally without requiring live
cloud credentials:
  1. Pre-Storage PII Scrubbing: User input containing an email address and phone
     number is sanitized (`[REDACTED_EMAIL]`, `[REDACTED_PHONE]`) before being
     stored in `PiiSafeSessionService`.
  2. Dynamic Voice Selection & Gemini TTS Synthesis: User switches their voice
     from `Puck` to `Aoede` (`user:preferred_voice = "Aoede"`) and synthesizes a
     playable `.wav` audio artifact.
  3. Cross-Session Persistence (`user:` scope): A brand-new session (`session-002`)
     created for the same `user_id` automatically inherits `user:preferred_voice == "Aoede"`
     and the consolidated `user:friend_notes` / `user:friend_profile_summary`.

Usage:
  python scripts/run_local_session_demo.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Ensure project root is importable when executed as `python scripts/run_local_session_demo.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from google.adk.events import Event, EventActions  # noqa: E402
from google.genai import types  # noqa: E402

from gpal_agent.config import (  # noqa: E402
    STATE_USER_FRIEND_NOTES,
    STATE_USER_FRIEND_PROFILE_SUMMARY,
    STATE_USER_PREFERRED_VOICE,
)
from gpal_agent.guardrails import scrub_pii_text  # noqa: E402
from gpal_agent.memory import (  # noqa: E402
    create_session_service,
    wait_for_pending_memory_tasks,
)
from gpal_agent.tools import (  # noqa: E402
    get_friend_memory_profile,
    list_available_gemini_voices,
    save_friend_memory_note,
    set_preferred_voice,
    synthesize_gemini_tts_audio,
)


async def run_demo() -> dict[str, Any]:
    """Run the two-session GPAL demonstration and verify all state invariants."""
    app_name = "gpal_agent"
    user_id = "demo-user-alex"
    session_service = create_session_service(db_url="sqlite+aiosqlite:///:memory:")

    print("\n" + "=" * 78)
    print("GPAL ('Google Pal') — Multi-Session Voice, Memory & PII Scrubbing Demo")
    print("=" * 78)

    # ==========================================================================
    # SESSION 1: PII Scrubbing, Voice Switch ("Aoede"), & Friend Note Storage
    # ==========================================================================
    session_1 = await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id="session-001",
    )
    print(f"\n[Session 1] Created session_id='{session_1.id}' for user_id='{user_id}'")

    raw_user_message = (
        "Hi GPAL! I'm Alex. My email is alex.rivera@example.com and my phone is "
        "415-555-0199. Could you switch your voice to Aoede? Also, remember that "
        "I love weekend trail running in Marin County and training for a half-marathon."
    )
    print(f"\n[Session 1] Raw user input:\n  {raw_user_message}")

    # Append event through PiiSafeSessionService to verify pre-storage PII redaction
    user_event = Event(
        invocation_id=f"inv-{int(time.time() * 1000)}",
        author="user",
        content=types.Content(
            role="user",
            parts=[types.Part.from_text(text=raw_user_message)],
        ),
    )
    await session_service.append_event(session=session_1, event=user_event)
    stored_text = session_1.events[-1].content.parts[0].text or ""
    print(f"\n[Session 1] Stored event text (after PiiSafeSessionService):\n  {stored_text}")
    assert "alex.rivera@example.com" not in stored_text, "Raw email leaked!"
    assert "415-555-0199" not in stored_text, "Raw phone number leaked!"
    assert "[REDACTED_EMAIL]" in stored_text
    assert "[REDACTED_PHONE]" in stored_text

    s1_ctx = SimpleNamespace(
        state=session_1.state,
        user_id=user_id,
        session_id=session_1.id,
    )

    # 1. Switch voice preference to "Aoede"
    voice_res = set_preferred_voice(voice_name="Aoede", tool_context=s1_ctx)
    print(f"\n[Session 1] set_preferred_voice('Aoede') -> {voice_res}")

    # 2. Save PII-free friend memory note (also triggers async profile consolidation)
    note_res = save_friend_memory_note(
        memory_note="Loves weekend trail running in Marin County and is training for a half-marathon.",
        category="hobby_or_interest",
        tool_context=s1_ctx,
    )
    print(f"[Session 1] save_friend_memory_note(...) -> {note_res}")

    # Await background async consolidation task
    await wait_for_pending_memory_tasks()

    # 3. Synthesize spoken reply using the user's selected voice ("Aoede")
    tts_res = synthesize_gemini_tts_audio(
        text_to_speak=(
            "Awesome to meet you, Alex! I've switched my voice to Aoede and noted "
            "your half-marathon trail running goal!"
        ),
        tool_context=s1_ctx,
    )
    audio_path = Path(tts_res["audio_file_path"])
    print(
        f"[Session 1] synthesize_gemini_tts_audio(...) -> "
        f"voice='{tts_res['voice_used']}', model='{tts_res['tts_model_used']}', "
        f"file='{audio_path}' ({tts_res['audio_size_bytes']} bytes)"
    )
    assert audio_path.exists() and audio_path.stat().st_size > 44

    # Commit Session 1 state delta to PiiSafeSessionService
    state_event = Event(
        invocation_id=f"inv-{int(time.time() * 1000)}-state",
        author="gpal_friend_coordinator",
        actions=EventActions(state_delta=dict(session_1.state)),
    )
    await session_service.append_event(session=session_1, event=state_event)

    # ==========================================================================
    # SESSION 2: Brand-New Session ID, Same User ID — Verify Cross-Session State
    # ==========================================================================
    session_2 = await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id="session-002",
    )
    print(f"\n[Session 2] Created brand-new session_id='{session_2.id}' for user_id='{user_id}'")

    s2_ctx = SimpleNamespace(
        state=session_2.state,
        user_id=user_id,
        session_id=session_2.id,
    )
    persisted_voice_info = list_available_gemini_voices(tool_context=s2_ctx)
    profile_info = get_friend_memory_profile(tool_context=s2_ctx)

    persisted_voice = session_2.state.get(STATE_USER_PREFERRED_VOICE)
    persisted_notes = session_2.state.get(STATE_USER_FRIEND_NOTES, [])
    persisted_summary = session_2.state.get(STATE_USER_FRIEND_PROFILE_SUMMARY, "")

    print(f"[Session 2] Persisted `{STATE_USER_PREFERRED_VOICE}`: {persisted_voice!r}")
    print(f"[Session 2] Active voice from list_available_gemini_voices(): {persisted_voice_info['current_preferred_voice']!r}")
    print(f"[Session 2] Persisted `{STATE_USER_FRIEND_NOTES}` count: {len(persisted_notes)}")
    print(f"[Session 2] Persisted `{STATE_USER_FRIEND_PROFILE_SUMMARY}`: {persisted_summary!r}")

    assert persisted_voice == "Aoede", (
        f"Expected user:preferred_voice == 'Aoede' in Session 2, got {persisted_voice!r}"
    )
    assert len(persisted_notes) >= 1, "Expected friend notes to persist into Session 2"
    assert profile_info["preferred_voice"] == "Aoede"

    # Clean up demo audio artifact
    if audio_path.exists():
        audio_path.unlink()

    print("\n" + "=" * 78)
    print("SUCCESS: All PII scrubbing, Gemini TTS WAV synthesis, and cross-session")
    print("         `user:` state persistence checks passed!")
    print("=" * 78 + "\n")

    return {
        "session_1_id": session_1.id,
        "session_2_id": session_2.id,
        "sanitized_text": scrub_pii_text(raw_user_message),
        "persisted_voice": persisted_voice,
        "persisted_notes_count": len(persisted_notes),
    }


def main() -> int:
    """CLI entry point."""
    asyncio.run(run_demo())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
