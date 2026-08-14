"""Static deployment-config tests for the Apptainer build (flaime-demo.def).

Validates deployment artifacts without requiring a running container daemon or
network access.

Moved from FLAIME `tests/demo/test_deployment_config.py` (26Q3-REPO-10). Two
classes didn't come along at that time and were backfilled by later cards:

- `TestDemoSh` (22 tests) — validates `scripts/demo.sh`; landed in 26Q3-REPO-12
  alongside the script.
- `TestReadme` — landed in 26Q3-REPO-13 alongside the docs move. Its
  `test_documents_bare_metal_mode` assertion targets the real bare-metal launch
  command (`uv run python flaime_demo/app.py`), not FLAIME's old
  `uv run flaime serve ui` — a real behavior difference between the in-repo and
  extracted layouts, not a path rewrite.

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
import time
from pathlib import Path

import pytest

APPTAINER_DEF: Path
DEMO_SH: Path
ENV_EXAMPLE: Path
README: Path


@pytest.fixture(scope="module", autouse=True)
def _resolve_paths(repo_root: Path) -> None:
    """Bind the module-level path constants from the shared repo_root fixture.

    A single fixture, not per-test edits — see conftest.py's `repo_root` for
    why `pytestconfig.rootpath` and never `Path(__file__).parents[N]`.
    """
    global APPTAINER_DEF, DEMO_SH, ENV_EXAMPLE, README
    APPTAINER_DEF = repo_root / "flaime-demo.def"
    DEMO_SH = repo_root / "scripts" / "demo.sh"
    ENV_EXAMPLE = repo_root / ".env.example"
    README = repo_root / "README.md"


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


# ── demo.sh bootstrap preamble (26Q3-REPO-22) ───────────────────────────────────


class TestDemoShBootstrap:
    """Self-bootstrapping preamble: .env creation + the checkpoint-staging gate.

    Runs demo.sh from a *sandboxed* repo root (script + .env.example copied into
    tmp_path), not the real checkout — the bootstrap preamble writes `.env` next
    to demo.sh, and this suite must never touch the developer's real `.env`.
    """

    @pytest.fixture
    def sandbox(self, tmp_path: Path) -> Path:
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        sandboxed_sh = scripts_dir / "demo.sh"
        sandboxed_sh.write_bytes(DEMO_SH.read_bytes())
        sandboxed_sh.chmod(DEMO_SH.stat().st_mode)
        (tmp_path / ".env.example").write_bytes(ENV_EXAMPLE.read_bytes())
        return tmp_path

    def _run(self, sandbox: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(sandbox / "scripts" / "demo.sh")],
            capture_output=True,
            text=True,
            env=env,
        )

    def test_creates_env_from_example_when_missing(self, sandbox: Path) -> None:
        assert not (sandbox / ".env").exists()
        env = dict(os.environ.items())
        env.pop("CHECKPOINTS_DIR", None)
        env["FLAIME_SIF"] = str(sandbox / "fake.sif")
        (sandbox / "fake.sif").touch()  # skip the apptainer build step
        self._run(sandbox, env)
        assert (sandbox / ".env").exists(), "demo.sh must create .env from .env.example"

    def test_exits_with_fetch_instructions_when_checkpoints_dir_unset(
        self, sandbox: Path
    ) -> None:
        env = dict(os.environ.items())
        env.pop("CHECKPOINTS_DIR", None)
        env["FLAIME_SIF"] = str(sandbox / "fake.sif")
        (sandbox / "fake.sif").touch()  # skip the apptainer build step
        result = self._run(sandbox, env)
        assert result.returncode != 0
        assert "fetch_checkpoints.sh" in result.stderr, (
            "demo.sh must point the operator at the exact staging command, "
            f"got stderr: {result.stderr!r}"
        )

    def test_does_not_fetch_checkpoints_itself(self, sandbox: Path) -> None:
        """Staging may need credentials the launcher shouldn't assume."""
        env = dict(os.environ.items())
        env.pop("CHECKPOINTS_DIR", None)
        env["FLAIME_SIF"] = str(sandbox / "fake.sif")
        (sandbox / "fake.sif").touch()
        self._run(sandbox, env)
        assert not (sandbox / "checkpoints").exists()


class TestDemoShBareMetal:
    """Mode dispatch (bare-metal default vs. Apptainer) and the uv sync gate.

    `uv` and `apptainer` are shimmed on PATH to record their invocations
    instead of actually building/running anything, so these tests are fast
    and don't require either tool installed.
    """

    @pytest.fixture
    def sandbox(self, tmp_path: Path) -> Path:
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        sandboxed_sh = scripts_dir / "demo.sh"
        sandboxed_sh.write_bytes(DEMO_SH.read_bytes())
        sandboxed_sh.chmod(DEMO_SH.stat().st_mode)
        (tmp_path / ".env.example").write_bytes(ENV_EXAMPLE.read_bytes())
        (tmp_path / "checkpoints").mkdir()
        (tmp_path / "uv.lock").write_text("")

        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        for tool, log_var in (
            ("uv", "UV_CALL_LOG"),
            ("apptainer", "APPTAINER_CALL_LOG"),
        ):
            fake = bin_dir / tool
            fake.write_text(f'#!/usr/bin/env bash\necho "$*" >> "${{{log_var}}}"\n')
            fake.chmod(0o755)
        return tmp_path

    def _env(self, sandbox: Path) -> dict[str, str]:
        env = dict(os.environ.items())
        env["PATH"] = f"{sandbox / 'bin'}:{env['PATH']}"
        env["CHECKPOINTS_DIR"] = str(sandbox / "checkpoints")
        # Blank out (not unset) so the auto-created .env's placeholder
        # DEMO_CHECKPOINT_FILE=your-merged-checkpoint.pt — which no fixture
        # checkpoints dir satisfies — doesn't trip the unrelated
        # checkpoint-file-resolution check these tests aren't exercising.
        env["DEMO_CHECKPOINT_FILE"] = ""
        env["UV_CALL_LOG"] = str(sandbox / "uv.log")
        env["APPTAINER_CALL_LOG"] = str(sandbox / "apptainer.log")
        return env

    def _log(self, path: Path) -> str:
        return path.read_text() if path.exists() else ""

    def test_bare_metal_default_when_no_sif(self, sandbox: Path) -> None:
        env = self._env(sandbox)
        self._run(env=env, sandbox=sandbox)
        assert "run" in self._log(sandbox / "uv.log")
        assert self._log(sandbox / "apptainer.log") == ""

    def test_apptainer_mode_when_sif_present(self, sandbox: Path) -> None:
        (sandbox / "flaime-demo.sif").touch()
        env = self._env(sandbox)
        self._run(env=env, sandbox=sandbox)
        assert "run" in self._log(sandbox / "apptainer.log")
        assert self._log(sandbox / "uv.log") == ""

    def test_uv_sync_runs_when_venv_missing(self, sandbox: Path) -> None:
        env = self._env(sandbox)
        self._run(env=env, sandbox=sandbox)
        assert "sync" in self._log(sandbox / "uv.log")

    def test_uv_sync_skipped_when_venv_warm(self, sandbox: Path) -> None:
        venv = sandbox / ".venv"
        venv.mkdir()
        pyvenv_cfg = venv / "pyvenv.cfg"
        pyvenv_cfg.write_text("")
        now = time.time()
        os.utime(sandbox / "uv.lock", (now - 10, now - 10))
        os.utime(pyvenv_cfg, (now, now))
        env = self._env(sandbox)
        self._run(env=env, sandbox=sandbox)
        log = self._log(sandbox / "uv.log")
        assert "sync" not in log
        assert "run" in log

    def _run(self, env: dict[str, str], sandbox: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(sandbox / "scripts" / "demo.sh")],
            capture_output=True,
            text=True,
            env=env,
        )


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


# ── README.md ────────────────────────────────────────────────────────────────


class TestReadme:
    def test_exists(self) -> None:
        assert README.exists()

    def test_documents_apptainer_mode(self) -> None:
        assert "apptainer" in README.read_text().lower()

    def test_documents_bare_metal_mode(self) -> None:
        """Ported from FLAIME's original assertion, which targeted the old
        `uv run flaime serve ui` command. flaime-demo's app is self-bootstrapping
        (see flaime_demo/app.py's __main__ block) — there is no `flaime` CLI or
        `serve ui` subcommand here, so the real invariant is the actual launch
        command this repo's README documents."""
        assert "uv run python flaime_demo/app.py" in README.read_text()

    def test_documents_languages_config(self) -> None:
        content = README.read_text()
        assert "demo_languages" in content or "DEMO_LANGUAGES_CONFIG" in content
