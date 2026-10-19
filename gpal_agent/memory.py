"""Persistent session memory, 5,000-token context compaction, and async memory consolidation for GPAL."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import logging
import re
import weakref
from collections.abc import Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import uuid
from dataclasses import dataclass, field

try:
    from google.genai import types
except ImportError:  # pragma: no cover

    @dataclass
    class _FallbackPart:
        text: str | None = None
        inline_data: Any | None = None
        function_call: Any | None = None
        function_response: Any | None = None

        @classmethod
        def from_text(cls, *, text: str) -> _FallbackPart:
            return cls(text=text)

    @dataclass
    class _FallbackContent:
        role: str = "user"
        parts: list[_FallbackPart] = field(default_factory=list)

    class _FallbackTypesNamespace:
        Part = _FallbackPart
        Content = _FallbackContent

    types = _FallbackTypesNamespace()  # type: ignore[assignment]

from gpal_agent.config import (
    COMPACTION_INTERVAL,
    COMPACTION_OVERLAP_SIZE,
    COMPACTION_TOKEN_THRESHOLD,
    DEFAULT_GEMINI_VOICE,
    EVENT_RETENTION_SIZE,
    STATE_APP_COMPACTED_SUMMARY,
    STATE_APP_COMPACTION_COUNT,
    STATE_USER_FRIEND_NOTES,
    STATE_USER_FRIEND_PROFILE_SUMMARY,
    STATE_USER_PREFERRED_VOICE,
    GpalConfig,
    get_config,
)


@dataclass
class _FallbackSession:
    """Fallback Session model used when google-adk is not installed."""

    id: str
    app_name: str
    user_id: str
    state: dict[str, Any] = field(default_factory=dict)
    events: list[Any] = field(default_factory=list)


class _FallbackInMemorySessionService:
    """Fallback async in-memory session service matching ADK BaseSessionService."""

    def __init__(self) -> None:
        self._sessions: dict[tuple[str, str, str], _FallbackSession] = {}

    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: dict[str, Any] | None = None,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> _FallbackSession:
        sid = session_id or f"session_{uuid.uuid4().hex[:12]}"
        session = _FallbackSession(
            id=sid,
            app_name=app_name,
            user_id=user_id,
            state=copy.deepcopy(state or {}),
            events=[],
        )
        self._sessions[(app_name, user_id, sid)] = session
        return session

    async def get_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        config: Any | None = None,
        **kwargs: Any,
    ) -> _FallbackSession | None:
        return self._sessions.get((app_name, user_id, session_id))

    async def list_sessions(
        self,
        *,
        app_name: str,
        user_id: str | None = None,
        **kwargs: Any,
    ) -> list[_FallbackSession]:
        return [
            s
            for (app, uid, _), s in self._sessions.items()
            if app == app_name and (user_id is None or uid == user_id)
        ]

    async def delete_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        **kwargs: Any,
    ) -> None:
        self._sessions.pop((app_name, user_id, session_id), None)

    async def append_event(self, session: Any, event: Any) -> Any:
        events = getattr(session, "events", None)
        if isinstance(events, list):
            events.append(event)
        actions = getattr(event, "actions", None)
        state_delta = getattr(actions, "state_delta", None) if actions else None
        if isinstance(state_delta, Mapping) and isinstance(
            getattr(session, "state", None), MutableMapping
        ):
            session.state.update(state_delta)
        return event


try:
    from google.adk.sessions import (
        BaseSessionService,
        DatabaseSessionService,
        InMemorySessionService,
        Session,
    )
except ImportError:  # pragma: no cover
    BaseSessionService = object  # type: ignore[misc,assignment]
    DatabaseSessionService = None  # type: ignore[assignment]
    InMemorySessionService = _FallbackInMemorySessionService  # type: ignore[misc,assignment]
    Session = _FallbackSession  # type: ignore[misc,assignment]

try:
    from google.adk.sessions import VertexAiSessionService
except ImportError:  # pragma: no cover
    VertexAiSessionService = None  # type: ignore[assignment]


# ==============================================================================
# Safe / Lazy Bridges to Guardrails & Observability
# ==============================================================================
_FALLBACK_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_FALLBACK_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_FALLBACK_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}(?!\d)"
)
_FALLBACK_CC_RE = re.compile(r"\b(?:\d{4}[-\s]){3}\d{4}\b")


def scrub_pii_text(text: str) -> str:
    """Delegate PII scrubbing to gpal_agent.guardrails with a safe regex fallback."""
    if not isinstance(text, str) or not text:
        return text
    try:
        from gpal_agent.guardrails import scrub_pii_text as _guardrails_scrub_pii_text

        return _guardrails_scrub_pii_text(text)
    except Exception:
        scrubbed = _FALLBACK_EMAIL_RE.sub("[REDACTED_EMAIL]", text)
        scrubbed = _FALLBACK_SSN_RE.sub("[REDACTED_SSN]", scrubbed)
        scrubbed = _FALLBACK_PHONE_RE.sub("[REDACTED_PHONE]", scrubbed)
        scrubbed = _FALLBACK_CC_RE.sub("[REDACTED_CREDIT_CARD]", scrubbed)
        return scrubbed


def scrub_content_in_place(content: Any) -> Any:
    """Delegate Content PII scrubbing to gpal_agent.guardrails with a safe fallback."""
    if content is None:
        return None
    try:
        from gpal_agent.guardrails import (
            scrub_content_in_place as _guardrails_scrub_content,
        )

        return _guardrails_scrub_content(content)
    except Exception:
        parts = getattr(content, "parts", None)
        if parts:
            for part in parts:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text:
                    part.text = scrub_pii_text(part_text)
        return content


def _get_logger() -> logging.Logger:
    """Obtain the structured logger from gpal_agent.observability."""
    try:
        from gpal_agent.observability import get_structured_logger

        return get_structured_logger("gpal_agent.memory")
    except Exception:
        return logging.getLogger("gpal_agent.memory")


@contextmanager
def _trace_op(span_name: str, attributes: Mapping[str, Any] | None = None):
    """Open an observability trace span via gpal_agent.observability."""
    try:
        from gpal_agent.observability import trace_operation

        with trace_operation(span_name, attributes=dict(attributes or {})) as span:
            yield span
        return
    except Exception:
        yield None


def _safe_log_intent(operation: str, **details: Any) -> float | None:
    """Log operation intent via gpal_agent.observability."""
    try:
        from gpal_agent.observability import log_intent

        return log_intent(
            event_type=operation,
            user_id=str(details.get("user_id", "")),
            session_id=str(details.get("session_id", "")),
            details=dict(details),
        )
    except Exception:
        _get_logger().info("Intent: %s | %s", operation, details)
        return None


def _safe_log_outcome(
    operation: str,
    status: str = "SUCCESS",
    start_time: float | None = None,
    **details: Any,
) -> None:
    """Log operation outcome via gpal_agent.observability."""
    try:
        from gpal_agent.observability import log_outcome

        log_outcome(
            event_type=operation,
            start_time=start_time,
            status=status.upper(),
            user_id=str(details.get("user_id", "")),
            session_id=str(details.get("session_id", "")),
            details=dict(details),
        )
    except Exception:
        _get_logger().info("Outcome: %s (%s) | %s", operation, status, details)


def _scrub_arbitrary_value(value: Any) -> Any:
    """Recursively scrub PII from strings, lists, tuples, and dicts."""
    if isinstance(value, str):
        return scrub_pii_text(value)
    if isinstance(value, list):
        return [_scrub_arbitrary_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_scrub_arbitrary_value(item) for item in value)
    if isinstance(value, dict):
        return {k: _scrub_arbitrary_value(v) for k, v in value.items()}
    return value


def _scrub_mapping_in_place(mapping: MutableMapping[str, Any] | None) -> None:
    """Scrub all string values inside a mutable mapping in place."""
    if not isinstance(mapping, MutableMapping):
        return
    for key in list(mapping.keys()):
        raw_val = mapping[key]
        scrubbed_val = _scrub_arbitrary_value(raw_val)
        if scrubbed_val != raw_val:
            mapping[key] = scrubbed_val


# ==============================================================================
# Cross-Session User State Backing Store & Tracked State Dict
# ==============================================================================
_GLOBAL_USER_STATE_BY_DB: dict[str, dict[str, dict[str, Any]]] = {}
_ACTIVE_SESSION_SERVICES: weakref.WeakSet[PiiSafeSessionService] = weakref.WeakSet()


class _TrackedStateDict(dict[str, Any]):
    """Dictionary subclass that auto-scrubs PII and syncs `user:` keys across sessions."""

    def __init__(
        self,
        initial: Mapping[str, Any] | None = None,
        *,
        user_id: str = "",
        service: PiiSafeSessionService | None = None,
    ) -> None:
        super().__init__()
        self._user_id = user_id
        self._service_ref = weakref.ref(service) if service is not None else None
        if initial:
            for k, v in initial.items():
                super().__setitem__(k, _scrub_arbitrary_value(v))

    def _notify(self, key: str, value: Any) -> None:
        if not str(key).startswith("user:"):
            return
        svc = self._service_ref() if self._service_ref is not None else None
        if svc is not None and self._user_id:
            svc._record_user_key(self._user_id, str(key), value)

    def __setitem__(self, key: str, value: Any) -> None:
        scrubbed = _scrub_arbitrary_value(value)
        super().__setitem__(key, scrubbed)
        self._notify(key, scrubbed)

    def update(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        other = dict(*args, **kwargs)
        for k, v in other.items():
            self[k] = v

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key not in self:
            self[key] = default
        return self[key]


def _resolve_sidecar_path(db_url: str | None) -> Path | None:
    """Resolve a sidecar JSON file path when db_url points to a local SQLite file."""
    if not db_url:
        return None
    normalized = db_url.strip()
    if ":memory:" in normalized or not normalized.startswith("sqlite"):
        return None
    match = re.match(r"^sqlite(?:\+[a-zA-Z0-9_]+)?:///(.+)$", normalized)
    if not match:
        return None
    raw_path = match.group(1)
    # On POSIX, sqlite:////abs/path has 4 slashes so group(1) starts with /abs/path
    db_path = Path(raw_path)
    return db_path.parent / f"{db_path.name}.user_state.json"


# ==============================================================================
# 1. PiiSafeSessionService (Persistent Cross-Session Storage + PII Scrubbing)
# ==============================================================================
class PiiSafeSessionService(BaseSessionService):  # type: ignore[misc]
    """Session service wrapper enforcing pre-storage PII redaction and cross-session state."""

    def __init__(
        self,
        wrapped_service: Any | None = None,
        db_url: str | None = None,
        use_vertex: bool = False,
        project_id: str | None = None,
        location: str | None = None,
    ) -> None:
        self.db_url = db_url
        self.use_vertex = use_vertex
        self.project_id = project_id
        self.location = location or "us-central1"
        self._sidecar_path = _resolve_sidecar_path(db_url)
        self._tracked_sessions: dict[str, list[Any]] = {}

        if db_url:
            self._cross_session_user_state: dict[str, dict[str, Any]] = (
                _GLOBAL_USER_STATE_BY_DB.setdefault(db_url, {})
            )
            self._load_sidecar_state()
        else:
            self._cross_session_user_state = {}

        self._fallback_service = (
            InMemorySessionService() if InMemorySessionService is not None else None
        )
        self._wrapped = wrapped_service or self._initialize_backend()
        self.wrapped_service = self._wrapped
        self.inner_service = self._wrapped
        _ACTIVE_SESSION_SERVICES.add(self)

    def _initialize_backend(self) -> Any:
        """Initialize VertexAiSessionService, DatabaseSessionService, or InMemorySessionService."""
        if (
            self.use_vertex
            and VertexAiSessionService is not None
            and self.project_id
        ):
            try:
                _safe_log_intent(
                    "init_vertex_session_service",
                    project_id=self.project_id,
                    location=self.location,
                )
                return VertexAiSessionService(
                    project=self.project_id,
                    location=self.location,
                )
            except Exception as exc:
                _get_logger().warning(
                    "VertexAiSessionService initialization failed (%s); falling back.",
                    exc,
                )

        if self.db_url and DatabaseSessionService is not None:
            try:
                _safe_log_intent("init_database_session_service", db_url=self.db_url)
                return DatabaseSessionService(db_url=self.db_url)
            except Exception as exc:
                _get_logger().warning(
                    "DatabaseSessionService initialization failed (%s); using InMemorySessionService.",
                    exc,
                )

        if InMemorySessionService is not None:
            return InMemorySessionService()
        raise RuntimeError("No ADK SessionService implementation is available.")

    def _load_sidecar_state(self) -> None:
        """Load cross-session user state from sidecar JSON when using a file SQLite URL."""
        if self._sidecar_path is None or not self._sidecar_path.exists():
            return
        try:
            raw = json.loads(self._sidecar_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for uid, ustate in raw.items():
                    if isinstance(ustate, dict):
                        self._cross_session_user_state.setdefault(uid, {}).update(
                            _scrub_arbitrary_value(ustate)
                        )
        except Exception as exc:
            _get_logger().debug("Could not read sidecar user state: %s", exc)

    def _save_sidecar_state(self) -> None:
        """Persist cross-session user state to sidecar JSON when using a file SQLite URL."""
        if self._sidecar_path is None:
            return
        try:
            self._sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            serialized = json.dumps(
                self._cross_session_user_state, indent=2, sort_keys=True
            )
            self._sidecar_path.write_text(serialized, encoding="utf-8")
        except Exception as exc:
            _get_logger().debug("Could not write sidecar user state: %s", exc)

    def _propagate_user_key_to_tracked_sessions(
        self, user_id: str, key: str, value: Any
    ) -> None:
        """Keep all tracked session instances for `user_id` synchronized when a `user:` key changes."""
        for tracked in self._tracked_sessions.get(user_id, []):
            state = getattr(tracked, "state", None)
            if isinstance(state, dict):
                dict.__setitem__(state, key, copy.deepcopy(value))
            elif isinstance(state, MutableMapping):
                state[key] = copy.deepcopy(value)

    def _record_user_key(self, user_id: str, key: str, value: Any) -> None:
        """Record a single `user:`-scoped key in the cross-session store."""
        if not user_id or not str(key).startswith("user:"):
            return
        scrubbed_val = copy.deepcopy(_scrub_arbitrary_value(value))
        user_store = self._cross_session_user_state.setdefault(user_id, {})
        user_store[str(key)] = scrubbed_val
        self._propagate_user_key_to_tracked_sessions(user_id, str(key), scrubbed_val)
        self._save_sidecar_state()

    def _sync_user_keys_from_mapping(
        self, user_id: str, mapping: Mapping[str, Any] | None
    ) -> None:
        """Extract and persist all `user:`-prefixed keys from a mapping."""
        if not user_id or not isinstance(mapping, Mapping):
            return
        user_store = self._cross_session_user_state.setdefault(user_id, {})
        changed = False
        for key, value in mapping.items():
            if str(key).startswith("user:"):
                scrubbed_val = copy.deepcopy(_scrub_arbitrary_value(value))
                if user_store.get(str(key)) != scrubbed_val:
                    user_store[str(key)] = scrubbed_val
                    self._propagate_user_key_to_tracked_sessions(
                        user_id, str(key), scrubbed_val
                    )
                    changed = True
        if changed:
            self._save_sidecar_state()

    def _sync_tracked_sessions(self, user_id: str) -> None:
        """Inspect all tracked session objects for `user_id` to capture direct state edits."""
        for tracked in self._tracked_sessions.get(user_id, []):
            state = getattr(tracked, "state", None)
            if isinstance(state, MutableMapping):
                _scrub_mapping_in_place(state)
                self._sync_user_keys_from_mapping(user_id, state)

    def _attach_tracked_state(self, session: Any, user_id: str) -> Any:
        """Bind a _TrackedStateDict to `session.state` and register the session."""
        if session is None:
            return None
        raw_state = getattr(session, "state", None)
        if isinstance(raw_state, Mapping):
            _scrub_mapping_in_place(
                raw_state if isinstance(raw_state, MutableMapping) else None
            )
            persisted = self._cross_session_user_state.get(user_id, {})
            merged: dict[str, Any] = {}
            merged.update(_scrub_arbitrary_value(dict(raw_state)))
            merged.update(copy.deepcopy(persisted))
            if STATE_USER_PREFERRED_VOICE not in merged:
                merged[STATE_USER_PREFERRED_VOICE] = DEFAULT_GEMINI_VOICE
            if STATE_USER_FRIEND_NOTES not in merged:
                merged[STATE_USER_FRIEND_NOTES] = []
            if STATE_USER_FRIEND_PROFILE_SUMMARY not in merged:
                merged[STATE_USER_FRIEND_PROFILE_SUMMARY] = ""

            tracked_dict = _TrackedStateDict(merged, user_id=user_id, service=self)
            try:
                object.__setattr__(session, "state", tracked_dict)
            except Exception:
                if isinstance(raw_state, MutableMapping):
                    raw_state.clear()
                    raw_state.update(tracked_dict)
            self._sync_user_keys_from_mapping(user_id, tracked_dict)

        sessions_for_user = self._tracked_sessions.setdefault(user_id, [])
        if all(existing is not session for existing in sessions_for_user):
            sessions_for_user.append(session)
        return session

    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: dict[str, Any] | None = None,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create a new session with cross-session `user:` state and PII scrubbing."""
        with _trace_op(
            "gpal.session.create",
            {"app_name": app_name, "user_id": user_id, "session_id": session_id or ""},
        ):
            self._load_sidecar_state()
            self._sync_tracked_sessions(user_id)

            persisted_user_state = copy.deepcopy(
                self._cross_session_user_state.get(user_id, {})
            )
            initial_state: dict[str, Any] = {}
            initial_state.update(persisted_user_state)

            if state:
                scrubbed_caller_state = _scrub_arbitrary_value(dict(state))
                initial_state.update(scrubbed_caller_state)

            if not initial_state.get(STATE_USER_PREFERRED_VOICE):
                initial_state[STATE_USER_PREFERRED_VOICE] = persisted_user_state.get(
                    STATE_USER_PREFERRED_VOICE, DEFAULT_GEMINI_VOICE
                )
            initial_state.setdefault(STATE_USER_FRIEND_NOTES, [])
            initial_state.setdefault(STATE_USER_FRIEND_PROFILE_SUMMARY, "")

            self._sync_user_keys_from_mapping(user_id, initial_state)

            try:
                session = await self._wrapped.create_session(
                    app_name=app_name,
                    user_id=user_id,
                    state=copy.deepcopy(initial_state),
                    session_id=session_id,
                    **kwargs,
                )
            except Exception as exc:
                if self._fallback_service is not None and self._wrapped is not self._fallback_service:
                    _get_logger().warning(
                        "Primary session service create_session failed (%s); falling back to InMemorySessionService.",
                        exc,
                    )
                    self._wrapped = self._fallback_service
                    session = await self._wrapped.create_session(
                        app_name=app_name,
                        user_id=user_id,
                        state=copy.deepcopy(initial_state),
                        session_id=session_id,
                        **kwargs,
                    )
                else:
                    raise

            return self._attach_tracked_state(session, user_id)

    async def get_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        config: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Retrieve an existing session and synchronize cross-session `user:` state."""
        self._load_sidecar_state()
        self._sync_tracked_sessions(user_id)
        call_kwargs: dict[str, Any] = {
            "app_name": app_name,
            "user_id": user_id,
            "session_id": session_id,
            **kwargs,
        }
        if config is not None:
            call_kwargs["config"] = config

        session = await self._wrapped.get_session(**call_kwargs)
        if session is None:
            return None
        return self._attach_tracked_state(session, user_id)

    async def list_sessions(
        self,
        *,
        app_name: str,
        user_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """List sessions from the underlying session service."""
        self._load_sidecar_state()
        if user_id:
            self._sync_tracked_sessions(user_id)
            return await self._wrapped.list_sessions(
                app_name=app_name, user_id=user_id, **kwargs
            )
        return await self._wrapped.list_sessions(app_name=app_name, **kwargs)

    async def delete_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        **kwargs: Any,
    ) -> None:
        """Delete a specific session while preserving cross-session `user:` state."""
        self._sync_tracked_sessions(user_id)
        await self._wrapped.delete_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            **kwargs,
        )

    async def append_event(self, session: Any, event: Any) -> Any:
        """Scrub PII from event content and state delta before appending to session storage."""
        user_id = getattr(session, "user_id", "") or ""
        with _trace_op(
            "gpal.session.append_event",
            {"user_id": user_id, "session_id": getattr(session, "id", "")},
        ):
            # 1. Scrub event.content in-place BEFORE persistence
            event_content = getattr(event, "content", None)
            if event_content is not None:
                scrub_content_in_place(event_content)

            # 2. Scrub event.actions.state_delta in-place BEFORE persistence
            actions = getattr(event, "actions", None)
            state_delta = getattr(actions, "state_delta", None) if actions else None
            if isinstance(state_delta, MutableMapping):
                _scrub_mapping_in_place(state_delta)
                self._sync_user_keys_from_mapping(user_id, state_delta)

            # 3. Scrub session.state in-place BEFORE persistence
            session_state = getattr(session, "state", None)
            if isinstance(session_state, MutableMapping):
                _scrub_mapping_in_place(session_state)
                if isinstance(state_delta, Mapping):
                    for k, v in state_delta.items():
                        session_state[k] = _scrub_arbitrary_value(v)
                self._sync_user_keys_from_mapping(user_id, session_state)

            # 4. Delegate to wrapped service
            try:
                result_event = await self._wrapped.append_event(
                    session=session, event=event
                )
            except Exception as exc:
                _get_logger().warning(
                    "Wrapped append_event encountered error (%s); applying state delta in-memory.",
                    exc,
                )
                events_list = getattr(session, "events", None)
                if isinstance(events_list, list):
                    events_list.append(event)
                result_event = event

            # 5. Re-verify post-append state and event scrubbing
            if getattr(result_event, "content", None) is not None:
                scrub_content_in_place(result_event.content)
            self._attach_tracked_state(session, user_id)
            return result_event


def create_session_service(
    db_url: str | None = None,
    use_vertex: bool | None = None,
    config: GpalConfig | None = None,
) -> PiiSafeSessionService:
    """Factory that constructs a configured PiiSafeSessionService instance.

    Args:
        db_url: Optional SQLAlchemy database URL override.
        use_vertex: Optional flag to force VertexAiSessionService when True.
        config: Optional GpalConfig instance.

    Returns:
        Configured PiiSafeSessionService wrapping Database, Vertex, or InMemory storage.
    """
    cfg = config or get_config()
    resolved_use_vertex = (
        use_vertex if use_vertex is not None else cfg.use_vertex_session_service
    )
    resolved_db_url = db_url if db_url is not None else cfg.session_db_url
    return PiiSafeSessionService(
        db_url=resolved_db_url,
        use_vertex=resolved_use_vertex,
        project_id=cfg.project_id,
        location=cfg.location,
    )


# ==============================================================================
# 2. 5,000-Token Context Compaction Engine
# ==============================================================================
def _extract_text_from_item(item: Any) -> str:
    """Extract readable text from an ADK Content, Event, dict, or string."""
    if item is None:
        return ""
    if isinstance(item, str):
        return item
    if isinstance(item, Mapping):
        if "text" in item and isinstance(item["text"], str):
            return item["text"]
        parts = item.get("parts")
        if isinstance(parts, Sequence):
            return " ".join(_extract_text_from_item(p) for p in parts if p)
        if "content" in item:
            return _extract_text_from_item(item["content"])
        return json.dumps(item, default=str)

    # Check if item is an Event with .content
    inner_content = getattr(item, "content", None)
    if inner_content is not None and inner_content is not item:
        return _extract_text_from_item(inner_content)

    # Check if item is a Content with .parts
    parts = getattr(item, "parts", None)
    if isinstance(parts, Sequence):
        collected: list[str] = []
        for part in parts:
            text_val = getattr(part, "text", None)
            if isinstance(text_val, str) and text_val:
                collected.append(text_val)
            elif isinstance(part, Mapping) and isinstance(part.get("text"), str):
                collected.append(part["text"])
        return " ".join(collected)

    text_attr = getattr(item, "text", None)
    if isinstance(text_attr, str):
        return text_attr

    return str(item)


def estimate_token_count(contents: Sequence[Any]) -> int:
    """Estimate token count across a sequence of Content objects, Events, or strings.

    Uses the standard ~4 characters per token heuristic (`max(1, total_chars // 4)`).

    Args:
        contents: Sequence of Content objects, Events, dicts, or strings.

    Returns:
        Estimated integer token count (0 if contents is empty).
    """
    if not contents:
        return 0
    total_chars = sum(len(_extract_text_from_item(item)) for item in contents)
    if total_chars <= 0:
        return 0
    return max(1, total_chars // 4)


def _synthesize_compacted_summary(
    older_contents: Sequence[Any],
    state: Mapping[str, Any] | None = None,
    max_summary_chars: int = 1200,
) -> str:
    """Synthesize older conversation turns into a concise PII-scrubbed summary."""
    snippets: list[str] = []
    for idx, item in enumerate(older_contents):
        role = getattr(item, "role", None) or (
            item.get("role", "user") if isinstance(item, Mapping) else "turn"
        )
        raw_text = _extract_text_from_item(item).strip()
        if not raw_text:
            continue
        scrubbed_text = scrub_pii_text(raw_text)
        compact_line = " ".join(scrubbed_text.split())
        if len(compact_line) > 140:
            compact_line = f"{compact_line[:137]}..."
        snippets.append(f"- ({role} #{idx + 1}): {compact_line}")

    preferred_voice = (
        state.get(STATE_USER_PREFERRED_VOICE, DEFAULT_GEMINI_VOICE)
        if isinstance(state, Mapping)
        else DEFAULT_GEMINI_VOICE
    )
    existing_profile = (
        str(state.get(STATE_USER_FRIEND_PROFILE_SUMMARY, "")).strip()
        if isinstance(state, Mapping)
        else ""
    )

    header_lines = [
        "[COMPACTED CONVERSATION SUMMARY]",
        f"Compacted {len(older_contents)} earlier conversational turns to stay within the 5,000-token context window.",
        f"Active Preferred Voice: {preferred_voice}.",
    ]
    if existing_profile:
        header_lines.append(f"Friend Profile Context: {scrub_pii_text(existing_profile)[:240]}")

    body_lines = snippets[:6]
    if len(snippets) > 6:
        body_lines.append(f"... ({len(snippets) - 6} middle turns condensed) ...")
        body_lines.extend(snippets[-3:])

    combined = "\n".join([*header_lines, "Key Highlights:", *body_lines])
    scrubbed_combined = scrub_pii_text(combined)
    if len(scrubbed_combined) > max_summary_chars:
        scrubbed_combined = f"{scrubbed_combined[:max_summary_chars - 3]}..."
    return scrubbed_combined


def _build_summary_node(summary_text: str, reference_item: Any) -> Any:
    """Create a summary node matching the type of items in `contents`."""
    if isinstance(reference_item, str):
        return summary_text
    if isinstance(reference_item, dict):
        return {
            "role": "user",
            "parts": [{"text": summary_text}],
            "text": summary_text,
        }
    return types.Content(
        role="user",
        parts=[types.Part.from_text(text=summary_text)],
    )


def enforce_token_compaction_limit(
    contents: list[Any],
    state: MutableMapping[str, Any] | None = None,
    token_threshold: int = COMPACTION_TOKEN_THRESHOLD,
    retention_size: int = EVENT_RETENTION_SIZE,
) -> tuple[list[Any], bool]:
    """Enforce the 5,000-token context compaction threshold on a conversation history.

    When `estimate_token_count(contents) > token_threshold`, summarizes older turns into
    a single `[COMPACTED CONVERSATION SUMMARY]` node while retaining the most recent turns.

    Args:
        contents: List of conversation Content turns or strings.
        state: Optional session state mapping to record compaction metadata.
        token_threshold: Maximum token budget before compaction triggers (default 5000).
        retention_size: Number of recent turns to retain uncompacted (default 5).

    Returns:
        A tuple of `(possibly_compacted_contents, was_compacted)`.
    """
    if not contents:
        return [], False

    initial_tokens = estimate_token_count(contents)
    with _trace_op(
        "gpal.context_compaction",
        {
            "initial_tokens": initial_tokens,
            "token_threshold": token_threshold,
            "initial_turn_count": len(contents),
            "retention_size": retention_size,
        },
    ):
        if initial_tokens <= token_threshold:
            return list(contents), False

        effective_retention = max(1, min(retention_size, len(contents) - 1))
        if len(contents) == 1:
            effective_retention = 0

        reference_item = contents[0]
        while True:
            if effective_retention > 0:
                older_slice = contents[:-effective_retention]
                retained_slice = list(contents[-effective_retention:])
            else:
                older_slice = list(contents)
                retained_slice = []

            summary_budget_chars = max(
                240,
                min(1200, (token_threshold * 4) // 3),
            )
            summary_text = _synthesize_compacted_summary(
                older_slice, state=state, max_summary_chars=summary_budget_chars
            )
            summary_node = _build_summary_node(summary_text, reference_item)
            compacted = [summary_node, *retained_slice]

            if (
                estimate_token_count(compacted) <= token_threshold
                or effective_retention == 0
            ):
                break
            effective_retention -= 1

        if state is not None:
            state[STATE_APP_COMPACTED_SUMMARY] = summary_text
            previous_count = int(state.get(STATE_APP_COMPACTION_COUNT, 0) or 0)
            state[STATE_APP_COMPACTION_COUNT] = previous_count + 1

        _safe_log_outcome(
            "context_compaction",
            status="compacted",
            initial_tokens=initial_tokens,
            compacted_tokens=estimate_token_count(compacted),
            turns_before=len(contents),
            turns_after=len(compacted),
        )
        return compacted, True


@dataclass
class _FallbackEventsCompactionConfig:
    """Fallback EventsCompactionConfig when google-adk is unavailable."""

    summarizer: Any
    token_threshold: int = COMPACTION_TOKEN_THRESHOLD
    event_retention_size: int = EVENT_RETENTION_SIZE
    compaction_interval: int = COMPACTION_INTERVAL
    overlap_size: int = COMPACTION_OVERLAP_SIZE


def build_events_compaction_config(
    summarizer_model: str = "gemini-2.5-flash",
    token_threshold: int = COMPACTION_TOKEN_THRESHOLD,
    event_retention_size: int = EVENT_RETENTION_SIZE,
    compaction_interval: int = COMPACTION_INTERVAL,
    overlap_size: int = COMPACTION_OVERLAP_SIZE,
) -> Any:
    """Build an ADK EventsCompactionConfig configured with the 5,000-token threshold.

    Dynamically inspects `EventsCompactionConfig.model_fields` to populate all supported
    compaction parameters in the installed `google-adk` version and ensures `.token_threshold`
    and `.event_retention_size` are accessible on the returned configuration object.
    """
    try:
        from google.adk.apps.app import EventsCompactionConfig
        from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
        from google.adk.models import Gemini

        summarizer = LlmEventSummarizer(llm=Gemini(model=summarizer_model))
        model_fields = getattr(EventsCompactionConfig, "model_fields", {}) or {}

        candidate_params: dict[str, Any] = {
            "summarizer": summarizer,
            "token_threshold": token_threshold,
            "event_retention_size": event_retention_size,
            "compaction_interval": compaction_interval,
            "overlap_size": overlap_size,
        }
        supported_kwargs = {
            k: v for k, v in candidate_params.items() if k in model_fields
        }
        config_obj = EventsCompactionConfig(**supported_kwargs)

        # Guarantee inspectable attributes across all ADK minor versions
        if getattr(config_obj, "token_threshold", None) != token_threshold:
            object.__setattr__(config_obj, "token_threshold", token_threshold)
        if getattr(config_obj, "event_retention_size", None) != event_retention_size:
            object.__setattr__(
                config_obj, "event_retention_size", event_retention_size
            )
        return config_obj
    except Exception:
        return _FallbackEventsCompactionConfig(
            summarizer={"model": summarizer_model},
            token_threshold=token_threshold,
            event_retention_size=event_retention_size,
            compaction_interval=compaction_interval,
            overlap_size=overlap_size,
        )


# ==============================================================================
# 3. Asynchronous Background Memory Consolidation
# ==============================================================================
_BACKGROUND_MEMORY_TASKS: set[asyncio.Task[Any]] = set()


def _format_consolidated_profile(notes: Sequence[Mapping[str, Any]]) -> str:
    """Deduplicate and synthesize friend notes into a structured summary string."""
    if not notes:
        return "No personal friend notes recorded yet."

    by_category: dict[str, list[str]] = {}
    seen_fingerprints: set[str] = set()

    for entry in notes:
        if not isinstance(entry, Mapping):
            continue
        raw_note = str(entry.get("note", "")).strip()
        if not raw_note:
            continue
        clean_note = scrub_pii_text(raw_note)
        fingerprint = hashlib.sha256(clean_note.lower().encode("utf-8")).hexdigest()[:16]
        if fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)
        category = str(entry.get("category", "general_fact")).strip() or "general_fact"
        by_category.setdefault(category, []).append(clean_note)

    if not by_category:
        return "No personal friend notes recorded yet."

    sections: list[str] = ["Consolidated Friend Profile:"]
    for category, items in by_category.items():
        readable_cat = category.replace("_", " ").title()
        joined_items = "; ".join(items)
        sections.append(f"- {readable_cat}: {joined_items}")

    return "\n".join(sections)


def _propagate_profile_summary_to_services(user_id: str, summary: str) -> None:
    """Propagate consolidated profile summary to any active PiiSafeSessionService stores."""
    if not user_id:
        return
    for svc in list(_ACTIVE_SESSION_SERVICES):
        if user_id in svc._cross_session_user_state:
            svc._record_user_key(user_id, STATE_USER_FRIEND_PROFILE_SUMMARY, summary)


async def _consolidate_friend_memory_background(
    user_id: str,
    notes: list[dict[str, str]],
    target_state: MutableMapping[str, Any],
) -> str:
    """Asynchronously deduplicate and synthesize user friend notes inside an OTel span."""
    with _trace_op(
        "gpal.async_memory_consolidation",
        {"user_id": user_id, "notes_count": len(notes)},
    ):
        await asyncio.sleep(0)
        summary = _format_consolidated_profile(notes)
        target_state[STATE_USER_FRIEND_PROFILE_SUMMARY] = summary
        _propagate_profile_summary_to_services(user_id, summary)
        _safe_log_outcome(
            "async_memory_consolidation",
            status="success",
            user_id=user_id,
            notes_count=len(notes),
        )
        return summary


def schedule_async_memory_consolidation(
    user_id: str,
    notes: list[dict[str, str]],
    target_state: MutableMapping[str, Any],
) -> asyncio.Task[str] | None:
    """Schedule non-blocking background consolidation of user friend memory notes.

    If an asyncio event loop is currently running, spawns a background `asyncio.Task`
    tracked in `_BACKGROUND_MEMORY_TASKS` and returns immediately without blocking the
    conversational response. Also writes an immediate baseline summary so synchronous
    callers always observe consistent state.

    Args:
        user_id: Identifier of the user whose notes are being consolidated.
        notes: List of sanitized friend note dictionaries.
        target_state: Session state mapping where `user:friend_profile_summary` is stored.

    Returns:
        The scheduled `asyncio.Task[str]` if an event loop is running, otherwise `None`.
    """
    baseline_summary = _format_consolidated_profile(notes)
    target_state[STATE_USER_FRIEND_PROFILE_SUMMARY] = baseline_summary
    _propagate_profile_summary_to_services(user_id, baseline_summary)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None

    task: asyncio.Task[str] = loop.create_task(
        _consolidate_friend_memory_background(
            user_id=user_id,
            notes=list(notes),
            target_state=target_state,
        ),
        name=f"gpal_consolidate_memory_{user_id}",
    )
    _BACKGROUND_MEMORY_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_MEMORY_TASKS.discard)
    return task


async def wait_for_pending_memory_tasks() -> None:
    """Await completion of all currently pending background memory consolidation tasks."""
    while _BACKGROUND_MEMORY_TASKS:
        pending = list(_BACKGROUND_MEMORY_TASKS)
        await asyncio.gather(*pending, return_exceptions=True)


__all__ = [
    "PiiSafeSessionService",
    "_BACKGROUND_MEMORY_TASKS",
    "_consolidate_friend_memory_background",
    "build_events_compaction_config",
    "create_session_service",
    "enforce_token_compaction_limit",
    "estimate_token_count",
    "schedule_async_memory_consolidation",
    "wait_for_pending_memory_tasks",
]
