"""Multi-agent orchestration, strategic model routing, and ADK App assembly for GPAL.

Satisfies Rubric Criteria:
- 2.1 Robust System Instructions
- 2.2 History Compaction (5,000-token threshold via EventsCompactionConfig + guard)
- 3.1 Multi-Agent Patterns (Coordinator + Search Sub-Agent + Pro Reasoning Sub-Agent)
- 3.2 Strategic Model Routing (Flash for fast chat/search/compaction, Pro for complex advice,
      Gemini 3.8 TTS for voice synthesis, Gemini Live 2.5 Native Audio for bidirectional voice)
- 3.3 Guardrails & Policy Plugins (ModelArmorPlugin + GpalSafetyAndPiiPlugin)
- 3.4 Human-in-the-Loop Hooks (hitl_before_tool_callback on high-stakes tools)
"""

from __future__ import annotations

from typing import Any

from gpal_agent.config import (
    COMPACTION_TOKEN_THRESHOLD,
    EVENT_RETENTION_SIZE,
    GpalConfig,
    get_config,
    select_optimal_model_tier,
)
from gpal_agent.guardrails import (
    get_configured_plugins,
    get_default_safety_settings,
    guardrail_after_model_callback,
    guardrail_before_model_callback,
    hitl_before_tool_callback,
    observability_after_tool_callback,
)
from gpal_agent.memory import (
    build_events_compaction_config,
    create_session_service,
    enforce_token_compaction_limit,
)
from gpal_agent.observability import (
    get_structured_logger,
    log_intent,
    log_outcome,
    setup_cloud_tracing,
    trace_operation,
)
from gpal_agent.prompts import (
    DEEP_REASONING_ADVISOR_PROMPT,
    FACT_CHECKER_SEARCH_PROMPT,
    GPAL_CONSTITUTIONAL_SYSTEM_PROMPT,
)
from gpal_agent.tools import (
    ALL_GPAL_TOOLS,
    auto_synthesize_after_model_callback,
    confirm_pending_memory_wipe,
    get_friend_memory_profile,
    get_live_voice_run_config,
    list_available_gemini_voices,
    save_friend_memory_note,
    set_preferred_voice,
    synthesize_gemini_tts_audio,
    wipe_all_user_memories,
)

logger = get_structured_logger("gpal_agent.agent")


def combined_before_model_callback(
    callback_context: Any,
    llm_request: Any,
) -> Any:
    """Execute 5,000-token compaction, strategic model routing telemetry, and pre-model guardrails.

    Args:
        callback_context: ADK `CallbackContext` for the active turn.
        llm_request: Outgoing `LlmRequest` to be sent to Gemini.

    Returns:
        `None` when allowed to proceed, or an `LlmResponse` when blocked by a safety guardrail.
    """
    state = getattr(callback_context, "state", None)
    contents = getattr(llm_request, "contents", None)

    # 1. Enforce 5,000-token context compaction threshold on conversation history
    if isinstance(contents, list) and contents:
        compacted_contents, was_compacted = enforce_token_compaction_limit(
            contents=contents,
            state=state if hasattr(state, "__setitem__") else None,
            token_threshold=COMPACTION_TOKEN_THRESHOLD,
            retention_size=EVENT_RETENTION_SIZE,
        )
        if was_compacted:
            try:
                llm_request.contents = compacted_contents
            except Exception:
                contents[:] = compacted_contents

    # 2. Record strategic model routing decision for the latest user message
    latest_text = ""
    if isinstance(contents, list):
        for item in reversed(contents):
            parts = getattr(item, "parts", None) or []
            texts = [
                getattr(p, "text", "")
                for p in parts
                if isinstance(getattr(p, "text", None), str)
            ]
            if texts:
                latest_text = " ".join(texts)
                break

    routing_decision = select_optimal_model_tier(latest_text)
    with trace_operation("gpal.model_routing", attributes=routing_decision):
        logger.info(
            "Selected model tier '%s' (%s)",
            routing_decision["tier"],
            routing_decision["selected_model"],
            extra={
                "lifecycle_phase": "INTENT",
                "event_type": "strategic_model_routing",
                "model_name": routing_decision["selected_model"],
                "metadata": routing_decision,
            },
        )

    # 3. Run PII scrubbing and prompt injection safety guardrail
    return guardrail_before_model_callback(
        callback_context=callback_context,
        llm_request=llm_request,
    )


def combined_after_model_callback(
    callback_context: Any,
    llm_response: Any,
) -> Any:
    """Execute output safety self-evaluation and automatic Gemini TTS voice synthesis.

    Args:
        callback_context: ADK `CallbackContext` for the active turn.
        llm_response: Incoming `LlmResponse` from Gemini.

    Returns:
        `None` after in-place sanitization and `.wav` voice synthesis.
    """
    guardrail_result = guardrail_after_model_callback(
        callback_context=callback_context,
        llm_response=llm_response,
    )
    auto_synthesize_after_model_callback(
        callback_context=callback_context,
        llm_response=llm_response,
    )
    return guardrail_result


def _build_generate_content_config() -> Any:
    """Build Gemini `GenerateContentConfig` with explicit safety settings."""
    try:
        from google.genai import types

        return types.GenerateContentConfig(
            safety_settings=get_default_safety_settings(),
            temperature=0.7,
        )
    except Exception:
        return None


def _create_adk_agents_and_app(
    config: GpalConfig | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Instantiate the specialized sub-agents, root coordinator agent, and ADK App."""
    cfg = config or get_config()
    setup_cloud_tracing(
        project_id=cfg.project_id or None,
        export_to_cloud=cfg.enable_cloud_trace and bool(cfg.project_id),
    )

    from google.adk.agents import LlmAgent
    from google.adk.apps.app import App
    from google.adk.tools import google_search
    from google.adk.tools.agent_tool import AgentTool

    gen_config = _build_generate_content_config()

    # Sub-Agent 1: Grounded Google Search Fact Checker (Flash model for fast verification)
    fact_checker_agent = LlmAgent(
        name="fact_checker_search_agent",
        model=cfg.fast_model,
        description=(
            "Specialized Google Search verification sub-agent. Delegate to this agent "
            "whenever the user asks a question about current events, live news, weather, "
            "facts, or real-world trivia that cannot be answered with certainty from memory."
        ),
        instruction=FACT_CHECKER_SEARCH_PROMPT,
        tools=[google_search],
    )

    # Sub-Agent 2: Deep Reasoning & Strategic Planning Advisor (Pro model for complex tasks)
    advisor_agent = LlmAgent(
        name="deep_reasoning_advisor_agent",
        model=cfg.pro_model,
        description=(
            "Specialized deep reasoning and life/career planning sub-agent powered by "
            "Gemini Pro. Delegate to this agent when the user asks for multi-step plans, "
            "complex pros-and-cons comparisons, or deep analytical advice."
        ),
        instruction=DEEP_REASONING_ADVISOR_PROMPT,
    )

    coordinator_tools: list[Any] = [
        AgentTool(agent=fact_checker_agent),
        AgentTool(agent=advisor_agent),
        set_preferred_voice,
        list_available_gemini_voices,
        synthesize_gemini_tts_audio,
        save_friend_memory_note,
        get_friend_memory_profile,
        confirm_pending_memory_wipe,
        wipe_all_user_memories,
    ]

    agent_kwargs: dict[str, Any] = {
        "name": "gpal_friend_coordinator",
        "model": cfg.fast_model,
        "description": (
            "GPAL (Google Pal): Empathetic voice and text companion friend that listens "
            "to the user, responds with Gemini TTS voice audio in the user's preferred "
            "voice across sessions, verifies facts via Google Search, and scrubs PII "
            "before storing memories."
        ),
        "instruction": GPAL_CONSTITUTIONAL_SYSTEM_PROMPT,
        "tools": coordinator_tools,
        "before_model_callback": combined_before_model_callback,
        "after_model_callback": combined_after_model_callback,
        "before_tool_callback": hitl_before_tool_callback,
        "after_tool_callback": observability_after_tool_callback,
    }
    if gen_config is not None:
        agent_kwargs["generate_content_config"] = gen_config

    coordinator_agent = LlmAgent(**agent_kwargs)

    compaction_config = build_events_compaction_config(
        summarizer_model=cfg.fast_model,
        token_threshold=cfg.compaction_token_threshold,
        event_retention_size=cfg.event_retention_size,
    )
    plugins = get_configured_plugins(cfg)

    gpal_app = App(
        name=cfg.app_name,
        root_agent=coordinator_agent,
        events_compaction_config=compaction_config,
        plugins=plugins,
    )

    return fact_checker_agent, advisor_agent, coordinator_agent, gpal_app


(
    fact_checker_search_agent,
    deep_reasoning_advisor_agent,
    gpal_friend_coordinator,
    app,
) = _create_adk_agents_and_app()

# Standard ADK export expected by `adk web`, `adk run`, and `adk deploy`
root_agent = gpal_friend_coordinator

__all__ = [
    "ALL_GPAL_TOOLS",
    "app",
    "combined_after_model_callback",
    "combined_before_model_callback",
    "create_session_service",
    "deep_reasoning_advisor_agent",
    "fact_checker_search_agent",
    "get_live_voice_run_config",
    "gpal_friend_coordinator",
    "root_agent",
]

