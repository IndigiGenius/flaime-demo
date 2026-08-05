"""Smoke: the app renders and reaches the no-checkpoint state (26Q3-REPO-09 AC3).

Runs the real script through Streamlit's own runtime via AppTest, so it catches
what ruff/mypy/pytest structurally cannot — notably function-local imports that
only fire once the UI actually executes. This is the check that caught app.py
reaching for `flaime.configs.languages` during the move.

Full E2E with real checkpoints is 26Q3-REPO-12's acceptance, not this.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch, pytestconfig: pytest.Config) -> AppTest:
    """Run the app with no checkpoint and no routing config configured."""
    for var in ("DEMO_CHECKPOINT", "DEMO_LANGUAGES_CONFIG", "DEMO_PUBLIC_BIND"):
        monkeypatch.delenv(var, raising=False)
    path = pytestconfig.rootpath / "flaime_demo" / "app.py"
    return AppTest.from_file(str(path), default_timeout=60).run()


def test_runs_without_uncaught_exception(app: AppTest) -> None:
    assert not app.exception, [e.value for e in app.exception]


def test_reaches_the_no_checkpoint_state(app: AppTest) -> None:
    """The no-model warning renders, and the sovereignty notice with it."""
    from flaime_demo.app import MSG_NO_MODEL, MSG_SOVEREIGNTY

    assert MSG_NO_MODEL in [w.value for w in app.warning]
    assert MSG_SOVEREIGNTY in [i.value for i in app.info]


def test_language_selector_is_populated(app: AppTest) -> None:
    """Guards the fallback path specifically — it is where LANGUAGES_32 is used."""
    assert len(app.selectbox) == 1
    # 32 languages plus the "" auto-detect entry.
    assert len(app.selectbox[0].options) == 33
