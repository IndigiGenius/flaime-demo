"""Guard: no import of FLAIME survives anywhere in this repo.

flaime-demo's contract is that FLAIME is absent from the environment entirely
(26Q3-REPO epic boundary) — only `flaime_serving` is installed. An import of
`flaime.*` therefore raises ModuleNotFoundError at runtime.

Nothing else catches this. mypy runs with ignore_missing_imports, and a
function-local import inside a Streamlit callback never executes at collection
time, so ruff, mypy and pytest all pass on a file that crashes on launch. That
is not hypothetical: 26Q3-REPO-09 moved app.py carrying
`from flaime.configs.languages import LANGUAGES_32`, which made the
no-checkpoint path unreachable while every gate stayed green.

This pulls REPO-12's AC4 grep forward to the card that introduces the risk.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# `flaime.` or bare `flaime` as a module, but never `flaime_demo`/`flaime_serving`
# — the trailing (?![\w]) is what keeps the underscored package names out.
FLAIME_IMPORT = re.compile(
    r"^\s*(?:from\s+flaime(?![\w])|import\s+flaime(?![\w]))",
    re.MULTILINE,
)


def _python_files(repo_root: Path) -> list[Path]:
    return [p for p in repo_root.rglob("*.py") if ".venv" not in p.parts]


def test_no_flaime_imports_survive(pytestconfig: pytest.Config) -> None:
    """No .py file in the repo may import FLAIME itself."""
    repo_root = Path(pytestconfig.rootpath)
    offenders = [
        f"{path.relative_to(repo_root)}: {match.group(0).strip()}"
        for path in _python_files(repo_root)
        for match in FLAIME_IMPORT.finditer(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, "FLAIME is absent from the demo environment; found: " + str(
        offenders
    )


def test_guard_matches_flaime_but_not_the_demo_packages() -> None:
    """The pattern must fire on FLAIME and stay silent on the two real deps.

    Without this, a regex that never matches anything would pass the test above
    vacuously and the guard would be decorative.
    """
    assert FLAIME_IMPORT.search("from flaime.configs.languages import LANGUAGES_32")
    assert FLAIME_IMPORT.search("import flaime")
    assert FLAIME_IMPORT.search("    from flaime.serving.router import LanguageRouter")

    assert not FLAIME_IMPORT.search("from flaime_demo import errors")
    assert not FLAIME_IMPORT.search("from flaime_demo.languages import LANGUAGES_32")
    assert not FLAIME_IMPORT.search("from flaime_serving import EnginePool")
    assert not FLAIME_IMPORT.search("import flaime_serving")
