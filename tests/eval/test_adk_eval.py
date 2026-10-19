"""ADK Golden Evaluation Suite verification for GPAL (`gpal_agent`)."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import pytest

from gpal_agent.agent import (
    fact_checker_search_agent,
    root_agent,
)
from gpal_agent.config import (
    STATE_USER_FRIEND_NOTES,
    STATE_USER_FRIEND_PROFILE_SUMMARY,
    STATE_USER_PREFERRED_VOICE,
)
from gpal_agent.guardrails import GpalSafetyAndPiiPlugin, PiiScrubber
from gpal_agent.memory import (
    PiiSafeSessionService,
    wait_for_pending_memory_tasks,
)
from gpal_agent.tools import (
    save_friend_memory_note,
    set_preferred_voice,
)

EVAL_DIR = Path(__file__).resolve().parent
EVAL_CONFIG_PATH = EVAL_DIR / "eval_config.json"
GOLDEN_DATASET_PATH = EVAL_DIR / "gpal_eval.test.json"


@dataclass
class _EvalToolContext:
    """ToolContext test double for executing golden trajectory tool calls."""

    state: dict[str, Any] = field(default_factory=dict)
    user_id: str = "eval_user"
    session_id: str = "eval_session"


@dataclass
class _EvalPart:
    text: str = ""


@dataclass
class _EvalContent:
    role: str = "user"
    parts: list[_EvalPart] = field(default_factory=list)


def _get_root_agent_tool_names() -> set[str]:
    """Return the set of all tool and AgentTool names registered on `root_agent`."""
    names: set[str] = set()
    for tool in getattr(root_agent, "tools", []) or []:
        if hasattr(tool, "name") and isinstance(tool.name, str):
            names.add(tool.name)
        elif hasattr(tool, "__name__") and isinstance(tool.__name__, str):
            names.add(tool.__name__)
        nested_agent = getattr(tool, "agent", None)
        if nested_agent is not None and hasattr(nested_agent, "name"):
            names.add(str(nested_agent.name))
    return names


def test_adk_eval_config_and_golden_dataset_schema() -> None:
    """Validate `eval_config.json` thresholds and `gpal_eval.test.json` structure."""
    assert EVAL_CONFIG_PATH.is_file()
    assert GOLDEN_DATASET_PATH.is_file()

    eval_config = json.loads(EVAL_CONFIG_PATH.read_text(encoding="utf-8"))
    criteria = eval_config.get("criteria", {})
    assert criteria.get("tool_trajectory_avg_score") == 1.0
    assert criteria.get("response_match_score") == 0.7

    dataset = json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    assert dataset.get("eval_set_id") == "gpal_golden_evaluation_set"
    eval_cases = dataset.get("eval_cases", [])
    assert len(eval_cases) == 3

    registered_tools = _get_root_agent_tool_names()
    for case in eval_cases:
        assert "eval_id" in case
        assert "session_input" in case
        conversation = case.get("conversation", [])
        assert len(conversation) >= 1
        for turn in conversation:
            assert "user_content" in turn
            assert "final_response" in turn
            tool_uses = turn.get("intermediate_data", {}).get("tool_uses", [])
            assert len(tool_uses) >= 1
            for tool_use in tool_uses:
                tool_name = tool_use["name"]
                assert tool_name in registered_tools, (
                    f"Golden trajectory tool '{tool_name}' is not registered on root_agent."
                )


@pytest.mark.asyncio
async def test_adk_golden_trajectories_execution(tmp_path: Path) -> None:
    """Execute the 3 golden evaluation trajectories through GPAL's guardrails and tools."""
    dataset = json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    cases_by_id = {c["eval_id"]: c for c in dataset["eval_cases"]}

    db_url = f"sqlite+aiosqlite:///{tmp_path / 'adk_eval.db'}"
    session_service = PiiSafeSessionService(db_url=db_url, use_vertex=False)
    plugin = GpalSafetyAndPiiPlugin(scrubber=PiiScrubber(enable_cloud_calls=False))

    # --- Case 1: Voice preference switch to Aoede ---
    case_1 = cases_by_id["case_1_switch_voice_to_aoede"]
    s1_input = case_1["session_input"]
    session_1 = await session_service.create_session(
        app_name=s1_input["app_name"],
        user_id=s1_input["user_id"],
        state=s1_input["state"],
        session_id="eval_sess_1",
    )
    turn_1 = case_1["conversation"][0]
    tool_call_1 = turn_1["intermediate_data"]["tool_uses"][0]
    assert tool_call_1["name"] == "set_preferred_voice"

    ctx_1 = _EvalToolContext(
        state=session_1.state,
        user_id=s1_input["user_id"],
        session_id=session_1.id,
    )
    out_1 = set_preferred_voice(**tool_call_1["args"], tool_context=ctx_1)
    assert out_1["status"] == "success"
    assert out_1["preferred_voice"] == "Aoede"
    assert session_1.state[STATE_USER_PREFERRED_VOICE] == "Aoede"

    # --- Case 2: Factual question delegated to fact_checker_search_agent ---
    case_2 = cases_by_id["case_2_factual_search_grounding"]
    turn_2 = case_2["conversation"][0]
    tool_call_2 = turn_2["intermediate_data"]["tool_uses"][0]
    assert tool_call_2["name"] == "fact_checker_search_agent"
    assert fact_checker_search_agent.name == "fact_checker_search_agent"
    assert len(getattr(fact_checker_search_agent, "tools", [])) >= 1

    # --- Case 3: Personal milestone with PII redaction ---
    case_3 = cases_by_id["case_3_personal_milestone_with_pii_redaction"]
    s3_input = case_3["session_input"]
    session_3 = await session_service.create_session(
        app_name=s3_input["app_name"],
        user_id=s3_input["user_id"],
        state=s3_input["state"],
        session_id="eval_sess_3",
    )
    turn_3 = case_3["conversation"][0]
    raw_user_text = turn_3["user_content"]["parts"][0]["text"]
    assert "runner.alex@example.com" in raw_user_text
    assert "415-555-0199" in raw_user_text

    # Run pre-storage PII plugin callback on user message
    user_msg = _EvalContent(role="user", parts=[_EvalPart(text=raw_user_text)])
    await plugin.on_user_message_callback(user_message=user_msg)
    assert "runner.alex@example.com" not in user_msg.parts[0].text
    assert "415-555-0199" not in user_msg.parts[0].text
    assert "[REDACTED_EMAIL]" in user_msg.parts[0].text
    assert "[REDACTED_PHONE]" in user_msg.parts[0].text

    tool_call_3 = turn_3["intermediate_data"]["tool_uses"][0]
    assert tool_call_3["name"] == "save_friend_memory_note"
    ctx_3 = _EvalToolContext(
        state=session_3.state,
        user_id=s3_input["user_id"],
        session_id=session_3.id,
    )
    out_3 = save_friend_memory_note(**tool_call_3["args"], tool_context=ctx_3)
    assert out_3["status"] == "success"
    assert out_3["category"] == "personal_milestone"

    await wait_for_pending_memory_tasks()
    assert len(session_3.state[STATE_USER_FRIEND_NOTES]) == 1
    assert "half-marathon" in session_3.state[STATE_USER_FRIEND_PROFILE_SUMMARY]
