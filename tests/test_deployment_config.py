"""Static deployment-config tests for the Apptainer build (flaime-demo.def).

Validates deployment artifacts without requiring a running container daemon or
network access.

Moved from FLAIME `tests/demo/test_deployment_config.py` (26Q3-REPO-10). Two
classes did NOT come along at that time:

- `TestDemoSh` (22 tests) — validates `scripts/demo.sh`, which did not exist
  in this repo yet. It arrives now, in 26Q3-REPO-12, alongside the script.
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

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

APPTAINER_DEF: Path
DEMO_SH: Path
ENV_EXAMPLE: Path


@pytest.fixture(scope="module", autouse=True)
def _resolve_paths(repo_root: Path) -> None:
    """Bind the module-level path constants from the shared repo_root fixture.

    A single fixture, not per-test edits — see conftest.py's `repo_root` for
    why `pytestconfig.rootpath` and never `Path(__file__).parents[N]`.
    """
    global APPTAINER_DEF, DEMO_SH, ENV_EXAMPLE
    APPTAINER_DEF = repo_root / "flaime-demo.def"
    DEMO_SH = repo_root / "scripts" / "demo.sh"
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


# ── demo.sh ───────────────────────────────────────────────────────────────────


class TestDemoSh:
    def test_exists(self) -> None:
        assert DEMO_SH.exists(), "scripts/demo.sh not found"

    def test_is_executable(self) -> None:
        assert DEMO_SH.stat().st_mode & stat.S_IXUSR

    def test_shellcheck(self) -> None:
        if shutil.which("shellcheck") is None:
            pytest.skip("shellcheck not installed")
        result = subprocess.run(
            ["shellcheck", str(DEMO_SH)], capture_output=True, text=True
        )
        assert result.returncode == 0, f"shellcheck:\n{result.stdout}"

    def test_invokes_apptainer_run(self) -> None:
        assert "apptainer run" in DEMO_SH.read_text()

    def test_builds_sif_when_absent(self) -> None:
        assert "apptainer build" in DEMO_SH.read_text()

    def test_rebuild_flag(self) -> None:
        assert "--rebuild" in DEMO_SH.read_text()

    def test_bind_checkpoints_readonly(self) -> None:
        content = DEMO_SH.read_text()
        assert "checkpoints" in content
        assert ":ro" in content

    def test_bind_configs_readonly(self) -> None:
        """configs/ must be bind-mounted so host YAML paths work in both modes."""
        content = DEMO_SH.read_text()
        assert "configs" in content and ":ro" in content

    def test_writable_tmpfs(self) -> None:
        assert "--writable-tmpfs" in DEMO_SH.read_text()

    def test_telemetry_off(self) -> None:
        assert "FLAIME_TELEMETRY=off" in DEMO_SH.read_text()

    def test_loopback_default(self) -> None:
        assert "127.0.0.1" in DEMO_SH.read_text()

    def test_port_default_7860(self) -> None:
        assert "7860" in DEMO_SH.read_text()

    def test_loads_env_file(self) -> None:
        assert ".env" in DEMO_SH.read_text()

    def test_passes_languages_config_env(self) -> None:
        assert "DEMO_LANGUAGES_CONFIG" in DEMO_SH.read_text()

    def test_languages_config_not_defaulted(self) -> None:
        """DEMO_LANGUAGES_CONFIG (Mode B) must never be defaulted.

        A baked-in fallback would silently force router mode even when the
        operator configured Mode A (DEMO_CHECKPOINT_FILE) and left Mode B
        unset — overriding their choice with no way to opt out. It must only
        be forwarded into the container when the operator actually set it.
        """
        content = DEMO_SH.read_text()
        assert "DEMO_LANGUAGES_CONFIG:-/app" not in content
        assert 'if [[ -n "${DEMO_LANGUAGES_CONFIG:-}" ]]' in content

    def test_gpu_nv_flag(self) -> None:
        content = DEMO_SH.read_text()
        assert "--nv" in content, "demo.sh must pass --nv when GPU is detected"
        assert "nvidia-smi" in content

    def test_exits_nonzero_when_checkpoints_missing(self, tmp_path: Path) -> None:
        """demo.sh must exit non-zero and print a helpful message when
        CHECKPOINTS_DIR does not exist."""
        env = dict(os.environ.items())
        env["CHECKPOINTS_DIR"] = str(tmp_path / "nonexistent")
        env["FLAIME_SIF"] = str(tmp_path / "fake.sif")
        # Create a fake SIF so the build step is skipped
        (tmp_path / "fake.sif").touch()
        result = subprocess.run(
            ["bash", str(DEMO_SH)],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode != 0
        assert "checkpoints" in result.stderr.lower() or "CHECKPOINTS" in result.stderr

    def test_sif_flag_overrides_default(self) -> None:
        """--sif flag must be accepted."""
        content = DEMO_SH.read_text()
        assert "--sif" in content

    def test_port_flag(self) -> None:
        assert "--port" in DEMO_SH.read_text()

    def test_bind_flag(self) -> None:
        assert "--bind" in DEMO_SH.read_text()

    def test_resolves_checkpoint_file_to_mount(self) -> None:
        """demo.sh must build the in-container path from a bare filename so the
        operator states CHECKPOINTS_DIR and the filename once each (no repeated
        host path)."""
        content = DEMO_SH.read_text()
        assert "DEMO_CHECKPOINT_FILE" in content
        assert "/checkpoints/${DEMO_CHECKPOINT_FILE}" in content

    def test_exits_when_checkpoint_file_missing(self, tmp_path: Path) -> None:
        """demo.sh must exit non-zero with a helpful message when
        DEMO_CHECKPOINT_FILE names a file absent from CHECKPOINTS_DIR."""
        ckpt_dir = tmp_path / "ckpts"
        ckpt_dir.mkdir()
        env = dict(os.environ.items())
        env.pop("DEMO_CHECKPOINT", None)
        env.pop("DEMO_LANGUAGES_CONFIG", None)
        env["CHECKPOINTS_DIR"] = str(ckpt_dir)
        env["DEMO_CHECKPOINT_FILE"] = "does-not-exist.pt"
        env["FLAIME_SIF"] = str(tmp_path / "fake.sif")
        (tmp_path / "fake.sif").touch()
        result = subprocess.run(
            ["bash", str(DEMO_SH)],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode != 0
        assert "checkpoint" in result.stderr.lower()


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
