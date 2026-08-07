"""Static deployment-config tests for the Apptainer build (flaime-demo.def).

Validates deployment artifacts without requiring a running container daemon or
network access.

Moved from FLAIME `tests/demo/test_deployment_config.py` (26Q3-REPO-10). Two
classes did NOT come along:

- `TestDemoSh` (19 tests) — validates `scripts/demo.sh`, which does not exist
  in this repo yet. That move is 26Q3-REPO-12; the tests travel with it.
- `TestReadme`'s apptainer/bare-metal assertions — this repo's README.md
  (written fresh in 26Q3-REPO-08) doesn't document either mode yet; that
  content arrives with the docs move in 26Q3-REPO-13.

`test_excludes_phonet_from_build` was replaced with `test_no_phonet_dependency`:
the original asserted a FLAIME-specific `uv sync --no-install-package phonet`
exclusion flag. flaime-demo's dependency graph never included PhoNet in the
first place (single graph, no extras — see flaime-demo.def's %post comment),
so there is nothing to exclude. This is a real behavior difference between the
in-repo and extracted layouts, not a path rewrite — flagged per REPO-10's
decision gate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

APPTAINER_DEF: Path
ENV_EXAMPLE: Path


@pytest.fixture(scope="module", autouse=True)
def _resolve_paths(repo_root: Path) -> None:
    """Bind the module-level path constants from the shared repo_root fixture.

    A single fixture, not per-test edits — see conftest.py's `repo_root` for
    why `pytestconfig.rootpath` and never `Path(__file__).parents[N]`.
    """
    global APPTAINER_DEF, ENV_EXAMPLE
    APPTAINER_DEF = repo_root / "flaime-demo.def"
    ENV_EXAMPLE = repo_root / ".env.example"


def _parse_def_sections(content: str) -> dict[str, str]:
    """Split an Apptainer definition file into a dict keyed by section name."""
    sections: dict[str, str] = {"header": ""}
    current = "header"
    for line in content.splitlines():
        if line.startswith("%") and not line.startswith("%%"):
            current = line.strip().lstrip("%").split()[0]
            sections.setdefault(current, "")
        else:
            sections[current] = sections.get(current, "") + line + "\n"
    return sections


# ── flaime-demo.def ───────────────────────────────────────────────────────────


class TestApptainerDef:
    def test_exists(self) -> None:
        assert APPTAINER_DEF.exists(), "flaime-demo.def not found"

    def test_bootstrap_docker(self) -> None:
        assert "Bootstrap: docker" in APPTAINER_DEF.read_text()

    def test_base_image_python312(self) -> None:
        assert "python:3.12" in APPTAINER_DEF.read_text()

    def test_non_root_user(self) -> None:
        sections = _parse_def_sections(APPTAINER_DEF.read_text())
        assert "useradd" in sections.get("post", ""), (
            "%post must create a non-root user with useradd"
        )

    def test_runscript_launches_streamlit(self) -> None:
        sections = _parse_def_sections(APPTAINER_DEF.read_text())
        runscript = sections.get("runscript", "")
        assert "streamlit" in runscript
        assert "app.py" in runscript

    def test_runscript_passes_args(self) -> None:
        """Port and address are forwarded via $@ so demo.sh controls them."""
        sections = _parse_def_sections(APPTAINER_DEF.read_text())
        assert '"$@"' in sections.get("runscript", ""), (
            "%runscript must forward $@ so demo.sh can set port/address"
        )

    def test_no_phonet_dependency(self) -> None:
        """PhoNet is training-only and never enters flaime-demo's dependency
        graph (unlike FLAIME's build, which had to explicitly exclude it)."""
        sections = _parse_def_sections(APPTAINER_DEF.read_text())
        assert "phonet" not in sections.get("post", "").lower()

    def test_no_weights_baked_in(self) -> None:
        content = APPTAINER_DEF.read_text()
        for ext in (".ckpt", ".pt", ".bin", ".safetensors"):
            assert ext not in content, f"Definition must not reference {ext} files"
        assert "checkpoints/" not in content

    def test_hf_home_set_to_tmp(self) -> None:
        sections = _parse_def_sections(APPTAINER_DEF.read_text())
        assert "HF_HOME=/tmp" in sections.get("environment", "")

    def test_telemetry_off_in_environment(self) -> None:
        sections = _parse_def_sections(APPTAINER_DEF.read_text())
        assert "FLAIME_TELEMETRY=off" in sections.get("environment", "")

    def test_phonet_pat_not_in_environment(self) -> None:
        sections = _parse_def_sections(APPTAINER_DEF.read_text())
        assert "PHONET_PAT" not in sections.get("environment", ""), (
            "PHONET_PAT must not appear in %environment — build-time only"
        )

    def test_configs_directory_baked_in(self) -> None:
        """Routing YAML configs must be copied into the image."""
        sections = _parse_def_sections(APPTAINER_DEF.read_text())
        assert "configs" in sections.get("files", ""), (
            "%files must copy configs/ so DEMO_LANGUAGES_CONFIG default works"
        )

    def test_server_headless_in_runscript(self) -> None:
        sections = _parse_def_sections(APPTAINER_DEF.read_text())
        assert "--server.headless" in sections.get("runscript", "")


# ── .env.example ─────────────────────────────────────────────────────────────


class TestEnvExample:
    def test_exists(self) -> None:
        assert ENV_EXAMPLE.exists(), ".env.example not found"

    def test_checkpoints_dir_documented(self) -> None:
        assert "CHECKPOINTS_DIR" in ENV_EXAMPLE.read_text()

    def test_demo_languages_config_documented(self) -> None:
        assert "DEMO_LANGUAGES_CONFIG" in ENV_EXAMPLE.read_text()

    def test_demo_checkpoint_file_documented(self) -> None:
        """.env.example must show the filename-relative single-model var so the
        host directory is not repeated."""
        assert "DEMO_CHECKPOINT_FILE" in ENV_EXAMPLE.read_text()

    def test_demo_model_type_documented(self) -> None:
        assert "DEMO_MODEL_TYPE" in ENV_EXAMPLE.read_text()

    def test_demo_decoder_documented(self) -> None:
        assert "DEMO_DECODER" in ENV_EXAMPLE.read_text()

    def test_no_secrets_committed(self) -> None:
        content = ENV_EXAMPLE.read_text()
        for line in content.splitlines():
            if line.startswith("CHECKPOINTS_DIR="):
                value = line.split("=", 1)[1].strip()
                if any(p in value for p in ("/path/to/your", "example", "change-me")):
                    return
                pytest.fail(
                    f"CHECKPOINTS_DIR in .env.example should be a placeholder, got: {value!r}"
                )
        pytest.fail("CHECKPOINTS_DIR not found in .env.example")
