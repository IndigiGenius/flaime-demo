"""Validates the synthetic community-eval fixture (26Q3-REPO-12 Part 2 spike).

Deliberately does not import ``run_community_eval`` — that module hasn't
moved into this repo yet (it lands with Part 2, stacked on this branch).
This just proves the fixture itself (manifest + audio) is well-formed and
ready for that move to consume, via ``Sample``'s schema
(``flaime/scripts/demo/run_community_eval.py`` in FLAIME) without importing it.

See tests/fixtures/community_eval/manifest.yaml for why this fixture's audio
is synthetic (espeak-ng) and committed directly, unlike the real
community-representative set described in FLAIME's 26Q1-DEMO-05.
"""

from __future__ import annotations

from pathlib import Path

import soundfile as sf
import yaml

_REQUIRED_FIELDS = (
    "audio_path",
    "language",
    "reference_text",
    "category",
    "licensing_note",
)


def _fixture_dir(repo_root: Path) -> Path:
    return repo_root / "tests" / "fixtures" / "community_eval"


def _load_samples(repo_root: Path) -> list[dict]:
    manifest = _fixture_dir(repo_root) / "manifest.yaml"
    raw = yaml.safe_load(manifest.read_text())
    return raw["samples"]


def test_manifest_exists(repo_root: Path) -> None:
    assert (_fixture_dir(repo_root) / "manifest.yaml").exists()


def test_manifest_has_samples(repo_root: Path) -> None:
    samples = _load_samples(repo_root)
    assert len(samples) >= 5


def test_every_sample_has_required_fields(repo_root: Path) -> None:
    for s in _load_samples(repo_root):
        missing = [f for f in _REQUIRED_FIELDS if not s.get(f)]
        assert not missing, f"{s.get('audio_path', '?')} missing {missing}"


def test_every_sample_has_a_category(repo_root: Path) -> None:
    categories = {s["category"] for s in _load_samples(repo_root)}
    assert categories == {"golden", "edge"}


def test_every_audio_path_resolves(repo_root: Path) -> None:
    fixture_dir = _fixture_dir(repo_root)
    for s in _load_samples(repo_root):
        assert (fixture_dir / s["audio_path"]).exists(), s["audio_path"]


def test_every_clip_is_16khz_mono(repo_root: Path) -> None:
    fixture_dir = _fixture_dir(repo_root)
    for s in _load_samples(repo_root):
        info = sf.info(str(fixture_dir / s["audio_path"]))
        assert info.samplerate == 16_000, s["audio_path"]
        assert info.channels == 1, s["audio_path"]


def test_every_clip_is_short_and_non_silent(repo_root: Path) -> None:
    """Sanity bounds, not a WER check — this fixture doesn't run a model."""
    fixture_dir = _fixture_dir(repo_root)
    for s in _load_samples(repo_root):
        data, sr = sf.read(str(fixture_dir / s["audio_path"]))
        duration_s = len(data) / sr
        assert 0.3 < duration_s < 10.0, s["audio_path"]
        assert abs(data).max() > 0.01, f"{s['audio_path']} looks silent"
