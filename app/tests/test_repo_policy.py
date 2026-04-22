"""Repository policy guardrails for future agent instances."""

from __future__ import annotations

from pathlib import Path


def test_agents_md_keeps_critical_delivery_rule() -> None:
    agents_text = Path("AGENTS.md").read_text(encoding="utf-8")

    assert "## CRITICAL delivery rule" in agents_text
    assert "tests -> git status review -> commit -> push" in agents_text
    assert "same branch/commit reaches both GitHub and GitLab" in agents_text
    assert "Never silently stop after local edits or tests" in agents_text


def test_readme_keeps_repo_sync_policy_visible() -> None:
    readme_text = Path("README.md").read_text(encoding="utf-8")

    assert "## Critical Repo Sync Policy" in readme_text
    assert "shareable work as incomplete until it is committed and pushed through `origin`" in readme_text
    assert "If commit or push is intentionally skipped, that must be said explicitly." in readme_text
