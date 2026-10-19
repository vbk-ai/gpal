"""Pytest configuration ensuring unit and evaluation tests run fast and offline even when .env exists."""

from __future__ import annotations

import os
import pytest


# Force offline/deterministic settings at test collection time before gpal_agent imports .env
os.environ["ENABLE_CLOUD_TRACE"] = "FALSE"
os.environ["ENABLE_CLOUD_DLP"] = "FALSE"
os.environ["USE_SECRET_MANAGER"] = "FALSE"
os.environ["MODEL_ARMOR_TEMPLATE_ID"] = ""
os.environ["GPAL_OFFLINE_TTS_FALLBACK"] = "TRUE"


@pytest.fixture(autouse=True)
def _isolate_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure every test starts with deterministic offline cloud flags unless explicitly overridden."""
    monkeypatch.setenv("ENABLE_CLOUD_TRACE", "FALSE")
    monkeypatch.setenv("ENABLE_CLOUD_DLP", "FALSE")
    monkeypatch.setenv("USE_SECRET_MANAGER", "FALSE")
    monkeypatch.setenv("MODEL_ARMOR_TEMPLATE_ID", "")
    monkeypatch.setenv("GPAL_OFFLINE_TTS_FALLBACK", "TRUE")
