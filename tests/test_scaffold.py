"""Scaffold contract for 26Q3-REPO-08 — mirrors flaime-serving's REPO-01 gates.

These read files rather than importing `flaime_demo`, so the repo's shape is
pinned independently of whether its package happens to be installable.
"""

from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(scope="session")
def pyproject(repo_root: Path) -> dict[str, Any]:
    return tomllib.loads((repo_root / "pyproject.toml").read_text())


def test_package_is_flaime_demo(pyproject: dict[str, Any]) -> None:
    assert pyproject["project"]["name"] == "flaime-demo"
    include = pyproject["tool"]["setuptools"]["packages"]["find"]["include"]
    assert "flaime_demo*" in include


def test_declares_the_three_runtime_dependencies(pyproject: dict[str, Any]) -> None:
    deps = " ".join(pyproject["project"]["dependencies"])
    assert "flaime-serving" in deps
    assert "streamlit>=1.40.0" in deps
    assert "soundfile>=0.12.0" in deps


def test_flaime_serving_is_pinned_to_an_immutable_rev(
    pyproject: dict[str, Any],
) -> None:
    source = pyproject["tool"]["uv"]["sources"]["flaime-serving"]
    assert "flaime-serving" in source["git"]
    # A full 40-char SHA specifically: a truthy check would accept rev="main",
    # which is a moving target and defeats the point of pinning.
    assert re.fullmatch(r"[0-9a-f]{40}", source.get("rev", "")), (
        "pin a full commit SHA, not a branch or short rev — branches move under us"
    )


def test_loc_budget_is_1400(repo_root: Path) -> None:
    assert (repo_root / "LOC_BUDGET").read_text().strip() == "1400"


def test_loc_gate_passes_on_the_current_tree(repo_root: Path) -> None:
    """The gate is wired correctly and the tree is within budget.

    Pins behaviour, not just the budget number: catches a gate script that
    points at the wrong package directory and so counts nothing.
    """
    done = subprocess.run(
        ["bash", str(repo_root / "scripts" / "check_loc_budget.sh")],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    counted = int(re.search(r"OK: (\d+) /", done.stdout).group(1))  # type: ignore[union-attr]
    assert counted > 0, "gate counted 0 lines — it is looking at the wrong directory"


def test_loc_gate_script_is_executable(repo_root: Path) -> None:
    gate = repo_root / "scripts" / "check_loc_budget.sh"
    assert gate.exists(), "copy the gate script verbatim from flaime-serving"
    assert gate.stat().st_mode & 0o111, "gate script must be executable"


def test_loc_gate_runs_in_ci_and_pre_commit(repo_root: Path) -> None:
    ci = (repo_root / ".github" / "workflows" / "ci.yml").read_text()
    hooks = (repo_root / ".pre-commit-config.yaml").read_text()
    assert "check_loc_budget.sh" in ci
    assert "check_loc_budget.sh" in hooks


def test_deployment_descriptors_moved(repo_root: Path) -> None:
    for name in (".env.example", "flaime-demo.def"):
        descriptor = repo_root / name
        assert descriptor.exists(), f"{name} not moved"
        assert descriptor.stat().st_size > 0, f"{name} is empty"


def test_apptainer_files_sources_all_exist(repo_root: Path) -> None:
    """Every `%files` source resolves in this repo.

    Apptainer aborts the build on a missing source before %post ever runs, so a
    stale path here is a build-time failure, not a runtime one. This repo
    inherited a `configs` entry that only exists in FLAIME (26Q3-REPO-11 brings
    it), which is exactly the regression this guards.
    """
    inside = False
    staged: list[str] = []
    for line in (repo_root / "flaime-demo.def").read_text().splitlines():
        if line.startswith("%"):
            inside = line.strip() == "%files"
            continue
        stripped = line.strip()
        if inside and stripped and not stripped.startswith("#"):
            staged.append(stripped.split()[0])

    assert staged, "%files section is empty or was not found"
    for source in staged:
        assert (repo_root / source).exists(), (
            f"%files stages '{source}', which does not exist in this repo — "
            "apptainer build aborts before %post"
        )


def test_readme_documents_the_checkpoint_contract(repo_root: Path) -> None:
    readme = (repo_root / "README.md").read_text()
    for token in ("CHECKPOINTS_DIR", "DEMO_CHECKPOINT_FILE", "DEMO_LANGUAGES_CONFIG"):
        assert token in readme, f"README must document {token}"
    assert "offline" in readme.lower(), "README must state the offline requirement"
