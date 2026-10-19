#!/usr/bin/env python3
"""Deploy the GPAL ADK Agent to Google Cloud Vertex AI Agent Engine.

This script bridges the Terraform IaC layer (`terraform/`) with the Python ADK
application (`gpal_agent/`). It automatically resolves deployment parameters from:
  1. Command-line flags (`--project-id`, `--location`, `--staging-bucket`),
  2. `terraform -chdir=terraform output -json` (if Terraform state is present), or
  3. `.env` / environment variables via `GpalConfig`.

Usage Examples:
  # Preview deployment configuration without making live GCP calls:
  python scripts/deploy_agent_engine.py --dry-run

  # Deploy using Terraform outputs automatically:
  python scripts/deploy_agent_engine.py --terraform-dir terraform

  # Deploy with explicit flags:
  python scripts/deploy_agent_engine.py \\
      --project-id your-gcp-project-id \\
      --location us-central1 \\
      --staging-bucket gs://your-gcp-project-id-gpal-agent-staging-dev
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

# Ensure project root is importable when executed as `python scripts/deploy_agent_engine.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpal_agent.config import get_config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("gpal.deploy")

AGENT_ENGINE_REQUIREMENTS: list[str] = [
    "google-adk>=1.0.0",
    "google-genai>=1.0.0",
    "pydantic>=2.7.0",
    "python-dotenv>=1.0.1",
    "opentelemetry-api>=1.25.0",
    "opentelemetry-sdk>=1.25.0",
    "opentelemetry-exporter-gcp-trace>=1.6.0",
    "google-cloud-secret-manager>=2.20.0",
    "google-cloud-dlp>=3.18.0",
    "google-cloud-logging>=3.10.0",
]


def load_terraform_outputs(terraform_dir: Path) -> dict[str, Any]:
    """Read `terraform output -json` from the specified Terraform directory if available."""
    if not terraform_dir.exists():
        logger.debug("Terraform directory %s does not exist; skipping.", terraform_dir)
        return {}

    try:
        result = subprocess.run(
            ["terraform", f"-chdir={terraform_dir}", "output", "-json"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return {}
        raw_outputs = json.loads(result.stdout)
        return {
            key: val.get("value")
            for key, val in raw_outputs.items()
            if isinstance(val, dict) and "value" in val
        }
    except (FileNotFoundError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        logger.debug("Could not read Terraform outputs: %s", exc)
        return {}


def build_deployment_plan(
    project_id: str,
    location: str,
    staging_bucket: str,
    environment: str,
    tf_outputs: dict[str, Any],
) -> dict[str, Any]:
    """Construct the Vertex AI Agent Engine deployment specification."""
    cfg = get_config()
    env_vars = {
        "GOOGLE_GENAI_USE_VERTEXAI": "TRUE",
        "GOOGLE_CLOUD_PROJECT": project_id,
        "GOOGLE_CLOUD_LOCATION": location,
        "GPAL_ENVIRONMENT": environment,
        "ENABLE_CLOUD_TRACE": "TRUE",
        "AUTO_SYNTHESIZE_VOICE_REPLY": "TRUE",
        "DEFAULT_GEMINI_VOICE": cfg.default_voice,
        "GEMINI_FAST_MODEL": cfg.fast_model,
        "GEMINI_PRO_MODEL": cfg.pro_model,
        "GEMINI_TTS_PRIMARY_MODEL": cfg.tts_primary_model,
        "GEMINI_TTS_EXPRESSIVE_MODEL": cfg.tts_expressive_model,
        "GEMINI_TTS_FALLBACK_MODEL": cfg.tts_fallback_model,
        "GEMINI_LIVE_VOICE_MODEL": cfg.live_voice_model,
        "MODEL_ARMOR_TEMPLATE_ID": (
            tf_outputs.get("model_armor_template_id") or cfg.model_armor_template_id
        ),
        "ENABLE_CLOUD_DLP": (
            "TRUE"
            if (tf_outputs.get("dlp_inspect_template_name") or cfg.enable_cloud_dlp)
            else "FALSE"
        ),
        "DLP_INSPECT_TEMPLATE_NAME": (
            tf_outputs.get("dlp_inspect_template_name") or cfg.dlp_inspect_template_name
        ),
        "DLP_DEIDENTIFY_TEMPLATE_NAME": (
            tf_outputs.get("dlp_deidentify_template_name")
            or cfg.dlp_deidentify_template_name
        ),
        "SECRET_MANAGER_SECRET_ID": (
            tf_outputs.get("secret_manager_secret_id") or cfg.secret_manager_secret_id
        ),
    }

    adk_cli_command = [
        "adk",
        "deploy",
        "agent_engine",
        f"--project={project_id}",
        f"--region={location}",
        f"--staging_bucket={staging_bucket}",
        f"--display_name=gpal-agent-{environment}",
        "--trace_to_cloud",
        "gpal_agent",
    ]

    return {
        "display_name": f"gpal-agent-{environment}",
        "description": (
            "GPAL (Google Pal): Empathetic voice and text companion agent with "
            "Gemini TTS, Google Search grounding, Model Armor PII scrubbing, and Cloud Trace."
        ),
        "project_id": project_id,
        "location": location,
        "staging_bucket": staging_bucket,
        "environment": environment,
        "trace_to_cloud": True,
        "requirements": AGENT_ENGINE_REQUIREMENTS,
        "env_vars": env_vars,
        "adk_cli_equivalent": " ".join(adk_cli_command),
    }


def deploy_to_vertex_agent_engine(plan: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    """Execute or simulate deployment to Google Cloud Vertex AI Agent Engine."""
    if dry_run:
        logger.info("DRY-RUN mode enabled. Validated Vertex AI Agent Engine deployment plan:")
        print(json.dumps(plan, indent=2))
        return {"status": "dry_run_validated", "plan": plan}

    missing = [
        name
        for name in ("project_id", "location", "staging_bucket")
        if not plan.get(name) or plan.get(name) == "your-gcp-project-id"
    ]
    if missing:
        raise ValueError(
            f"Missing required deployment settings: {', '.join(missing)}. "
            "Provide --project-id and --staging-bucket or run `terraform apply` first."
        )

    import vertexai
    from vertexai import agent_engines

    from gpal_agent.agent import app as gpal_adk_app

    logger.info(
        "Initializing Vertex AI (project=%s, location=%s, staging_bucket=%s)...",
        plan["project_id"],
        plan["location"],
        plan["staging_bucket"],
    )
    vertexai.init(
        project=plan["project_id"],
        location=plan["location"],
        staging_bucket=plan["staging_bucket"],
    )

    logger.info("Deploying %s to Vertex AI Agent Engine...", plan["display_name"])
    remote_agent = agent_engines.create(
        agent_engine=gpal_adk_app,
        requirements=plan["requirements"],
        extra_packages=["./gpal_agent"],
        display_name=plan["display_name"],
        description=plan["description"],
        env_vars=plan["env_vars"],
    )
    resource_name = getattr(remote_agent, "resource_name", str(remote_agent))
    logger.info("Successfully deployed GPAL to Vertex AI Agent Engine: %s", resource_name)
    return {"status": "deployed", "resource_name": resource_name, "plan": plan}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for Vertex AI Agent Engine deployment."""
    parser = argparse.ArgumentParser(
        description="Deploy the GPAL ADK Agent to Google Cloud Vertex AI Agent Engine."
    )
    parser.add_argument(
        "--project-id",
        default="",
        help="Google Cloud Project ID (overrides Terraform output and GOOGLE_CLOUD_PROJECT).",
    )
    parser.add_argument(
        "--location",
        default="",
        help="Google Cloud region (default: us-central1).",
    )
    parser.add_argument(
        "--staging-bucket",
        default="",
        help="GCS staging bucket URI (`gs://...`) for Agent Engine artifacts.",
    )
    parser.add_argument(
        "--environment",
        default="",
        choices=["", "dev", "staging", "prod"],
        help="Target environment tier (dev, staging, or prod).",
    )
    parser.add_argument(
        "--terraform-dir",
        default=str(PROJECT_ROOT / "terraform"),
        help="Path to the Terraform configuration directory (default: ./terraform).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved Agent Engine deployment plan as JSON without calling GCP.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the deployment CLI."""
    args = parse_args(argv)
    cfg = get_config()
    tf_outputs = load_terraform_outputs(Path(args.terraform_dir))

    project_id = (
        args.project_id
        or tf_outputs.get("project_id")
        or cfg.project_id
        or "your-gcp-project-id"
    )
    location = (
        args.location
        or tf_outputs.get("region")
        or cfg.location
        or "us-central1"
    )
    environment = (
        args.environment
        or tf_outputs.get("environment")
        or cfg.environment
        or "dev"
    )
    staging_bucket = (
        args.staging_bucket
        or tf_outputs.get("staging_bucket_uri")
        or cfg.staging_bucket_uri
        or f"gs://{project_id}-gpal-agent-staging-{environment}"
    )

    plan = build_deployment_plan(
        project_id=project_id,
        location=location,
        staging_bucket=staging_bucket,
        environment=environment,
        tf_outputs=tf_outputs,
    )

    try:
        deploy_to_vertex_agent_engine(plan, dry_run=args.dry_run)
        return 0
    except Exception as exc:
        logger.error("Deployment failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
