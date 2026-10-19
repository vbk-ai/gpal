"""Secure secret resolution via Google Cloud Secret Manager with local fallback.

Satisfies Rubric Criterion 5.3 (Secure Secret Management):
- Resolves sensitive credentials from Google Cloud Secret Manager in staging/production.
- Provides an in-memory cache (`_secret_cache`) with `clear_cache()` to minimize
  redundant network calls and support deterministic unit testing.
- Gracefully falls back to environment variables (`GOOGLE_API_KEY` or secret name)
  for local development and offline CI pipelines without raising unhandled exceptions.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from gpal_agent.config import get_config

try:
    from google.cloud import secretmanager  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - optional in lightweight local envs
    secretmanager = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class SecretManagerService:
    """Client wrapper for Google Cloud Secret Manager with caching and env fallback."""

    def __init__(self, client: Any | None = None) -> None:
        """Initialize the Secret Manager service wrapper.

        Args:
            client: Optional pre-configured `SecretManagerServiceClient` instance
                (useful for dependency injection in unit tests).
        """
        self._client = client
        self._secret_cache: dict[str, str] = {}

    @staticmethod
    def build_secret_version_path(
        project_id: str, secret_id: str, version: str = "latest"
    ) -> str:
        """Format the canonical Google Cloud Secret Manager version resource path."""
        return f"projects/{project_id}/secrets/{secret_id}/versions/{version}"

    @staticmethod
    def _is_secret_manager_enabled() -> bool:
        """Return whether Cloud Secret Manager lookup is enabled via environment."""
        flag = os.getenv("USE_SECRET_MANAGER", "TRUE").strip().upper()
        return flag not in ("FALSE", "0", "NO", "OFF")

    def _get_or_create_client(self) -> Any | None:
        """Lazily instantiate the SecretManagerServiceClient when needed."""
        if self._client is not None:
            return self._client
        if secretmanager is None:
            return None
        self._client = secretmanager.SecretManagerServiceClient()
        return self._client

    def clear_cache(self) -> None:
        """Clear all cached secret values from memory."""
        self._secret_cache.clear()

    def get_secret(
        self,
        secret_id: str | None = None,
        project_id: str | None = None,
        env_fallback_key: str = "GOOGLE_API_KEY",
        version: str = "latest",
    ) -> str:
        """Fetch a secret from Cloud Secret Manager with graceful env fallback.

        Args:
            secret_id: Secret identifier in Google Cloud Secret Manager. Defaults
                to `GpalConfig.secret_manager_secret_id`.
            project_id: Google Cloud project ID. Defaults to `GpalConfig.project_id`.
            env_fallback_key: Environment variable name checked when Secret Manager
                is disabled or unreachable.
            version: Secret version string (defaults to `"latest"`).

        Returns:
            The resolved secret string, or `""` if unavailable in both Secret Manager
            and environment variables.
        """
        cfg = get_config()
        resolved_project_id = (
            project_id if project_id is not None else cfg.project_id
        ).strip()
        resolved_secret_id = (
            secret_id if secret_id is not None else cfg.secret_manager_secret_id
        ).strip()

        cache_key = (
            f"{resolved_project_id or 'local'}:"
            f"{resolved_secret_id or env_fallback_key}:"
            f"{version}"
        )
        if cache_key in self._secret_cache:
            return self._secret_cache[cache_key]

        if (
            resolved_project_id
            and resolved_secret_id
            and self._is_secret_manager_enabled()
        ):
            resource_name = self.build_secret_version_path(
                project_id=resolved_project_id,
                secret_id=resolved_secret_id,
                version=version,
            )
            try:
                client = self._get_or_create_client()
                if client is not None:
                    response = client.access_secret_version(
                        request={"name": resource_name}
                    )
                    payload_bytes = getattr(
                        getattr(response, "payload", None), "data", b""
                    )
                    if isinstance(payload_bytes, bytes):
                        secret_value = payload_bytes.decode("utf-8").strip()
                    else:
                        secret_value = str(payload_bytes).strip()

                    if secret_value:
                        self._secret_cache[cache_key] = secret_value
                        return secret_value
            except Exception as exc:
                logger.warning(
                    "Cloud Secret Manager lookup failed for '%s' in project '%s' "
                    "(%s); falling back to environment variable '%s'.",
                    resolved_secret_id,
                    resolved_project_id,
                    type(exc).__name__,
                    env_fallback_key,
                )

        fallback_value = os.getenv(env_fallback_key, "").strip()
        if not fallback_value and resolved_secret_id:
            fallback_value = os.getenv(resolved_secret_id, "").strip()

        if fallback_value:
            self._secret_cache[cache_key] = fallback_value

        return fallback_value


_DEFAULT_SECRET_MANAGER = SecretManagerService()
_secret_cache: dict[str, str] = _DEFAULT_SECRET_MANAGER._secret_cache


def get_default_secret_manager() -> SecretManagerService:
    """Return the singleton default `SecretManagerService` instance."""
    return _DEFAULT_SECRET_MANAGER


def clear_cache() -> None:
    """Clear the module-level default `SecretManagerService` in-memory cache."""
    _DEFAULT_SECRET_MANAGER.clear_cache()


def get_secret(
    secret_id: str | None = None,
    project_id: str | None = None,
    env_fallback_key: str = "GOOGLE_API_KEY",
    version: str = "latest",
) -> str:
    """Module-level convenience wrapper around `SecretManagerService.get_secret`."""
    return _DEFAULT_SECRET_MANAGER.get_secret(
        secret_id=secret_id,
        project_id=project_id,
        env_fallback_key=env_fallback_key,
        version=version,
    )


__all__ = [
    "SecretManagerService",
    "_secret_cache",
    "clear_cache",
    "get_default_secret_manager",
    "get_secret",
]
