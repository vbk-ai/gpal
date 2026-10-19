#!/usr/bin/env python3
"""Export all public GPAL source files into a single plain-text Markdown bundle.

Generates `submission_bundle.md` (which is gitignored) containing a complete
table of contents and syntax-highlighted code blocks for every public file in
the repository, suitable for single-file assessment upload.

Usage:
  python scripts/export_submission_bundle.py
  python scripts/export_submission_bundle.py --output submission_bundle.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXCLUDED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".terraform",
        "artifacts",
        "gpal_agent.egg-info",
    }
)

EXCLUDED_FILENAMES: frozenset[str] = frozenset(
    {
        ".gdrive_token",
        ".env",
        "plan.md",
        "submission_bundle.md",
        "requirements.txt",
    }
)

EXCLUDED_SUFFIXES: tuple[str, ...] = (
    ".pyc",
    ".pyo",
    ".pyd",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".wav",
    ".tfstate",
    ".tfvars",
)

LANGUAGE_BY_SUFFIX: dict[str, str] = {
    ".py": "python",
    ".tf": "hcl",
    ".example": "hcl",
    ".md": "markdown",
    ".toml": "toml",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".json": "json",
}


def should_include_file(rel_path: Path) -> bool:
    """Return True if `rel_path` belongs in the public assessment bundle."""
    if any(part in EXCLUDED_DIRS for part in rel_path.parts):
        return False
    if rel_path.name in EXCLUDED_FILENAMES:
        return False
    if rel_path.name.startswith(".env.") and rel_path.name != ".env.example":
        return False
    if rel_path.name.endswith(EXCLUDED_SUFFIXES):
        return False
    return True


def collect_project_files(root: Path) -> list[Path]:
    """Return all eligible repository files in a deterministic priority order."""
    all_files = [
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and should_include_file(path.relative_to(root))
    ]

    priority_prefixes = (
        "README.md",
        "LICENSE",
        "pyproject.toml",
        ".env.example",
        ".gitignore",
        "gpal_agent/",
        "terraform/",
        "scripts/",
        "tests/",
        "docs/",
        ".github/",
    )

    def sort_key(rel: Path) -> tuple[int, str]:
        posix = rel.as_posix()
        for idx, prefix in enumerate(priority_prefixes):
            if posix == prefix or posix.startswith(prefix):
                return (idx, posix)
        return (len(priority_prefixes), posix)

    return sorted(all_files, key=sort_key)


def infer_fence_language(rel_path: Path) -> str:
    """Infer Markdown code-fence language identifier for a file path."""
    if rel_path.name == ".env.example":
        return "dotenv"
    if rel_path.name == "terraform.tfvars.example":
        return "hcl"
    return LANGUAGE_BY_SUFFIX.get(rel_path.suffix.lower(), "text")


def build_bundle_markdown(root: Path, files: list[Path]) -> str:
    """Assemble the complete plain-text Markdown submission bundle."""
    lines: list[str] = [
        "# GPAL (`gpal_agent`) — Complete Project Submission Bundle",
        "",
        "This single-file plain-text Markdown bundle contains the complete source code,",
        "Agent Constitution, multi-agent ADK architecture, Terraform Infrastructure-as-Code,",
        "unit & integration test suites, and technical documentation for **GPAL (Google Pal)**.",
        "",
        "## Included Files",
        "",
    ]

    for rel in files:
        lines.append(f"- `{rel.as_posix()}`")
    lines.append("")
    lines.append("---")
    lines.append("")

    for rel in files:
        full_path = root / rel
        content = full_path.read_text(encoding="utf-8", errors="replace").rstrip()
        lang = infer_fence_language(rel)
        fence = "````" if "```" in content else "```"
        lines.append(f"## File: `{rel.as_posix()}`")
        lines.append("")
        lines.append(f"{fence}{lang}")
        lines.append(content)
        lines.append(fence)
        lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for generating `submission_bundle.md`."""
    parser = argparse.ArgumentParser(
        description="Export all public GPAL source files into a single Markdown bundle."
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "submission_bundle.md"),
        help="Destination Markdown file path (default: ./submission_bundle.md).",
    )
    args = parser.parse_args(argv)

    output_path = Path(args.output).resolve()
    files = collect_project_files(PROJECT_ROOT)
    bundle_text = build_bundle_markdown(PROJECT_ROOT, files)
    output_path.write_text(bundle_text, encoding="utf-8")

    print(
        f"Exported {len(files)} files ({len(bundle_text):,} bytes) to {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
