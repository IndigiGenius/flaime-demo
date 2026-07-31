"""Shared fixtures for the flaime-demo suite."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Repo root, found by walking up to the directory holding `.git`.

    The scaffold tests read files by path instead of importing `flaime_demo`,
    so they need an anchor that resolves before `pyproject.toml` exists —
    which is exactly when `pytestconfig.rootpath` cannot be trusted.
    """
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError("repo root not found: no .git directory above tests/")
