"""Shared fixtures for the flaime-demo suite."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root(pytestconfig: pytest.Config) -> Path:
    """Repo root, from pytest's own rootdir.

    `pyproject.toml` carries `[tool.pytest.ini_options]`, so pytest resolves
    rootdir to the repo root. Deriving it from `.git` instead would raise in any
    checkout-less tree — an sdist, an exported tarball, or the suite running
    from /app inside the built container.
    """
    return Path(pytestconfig.rootpath)
