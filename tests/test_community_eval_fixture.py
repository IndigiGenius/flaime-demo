"""Validates the community-eval fixture (26Q3-REPO-12 Part 2).

Deliberately does not import ``scripts.run_community_eval`` directly — this
proves the fixture itself (manifest + audio) is well-formed against
``Sample``'s schema independently of the runner module (see
tests/test_community_eval.py for the runner's own tests, which use a stub
transcriber instead of this fixture).

This fixture's audio is a real excerpt of LibriSpeech test-clean (CC BY 4.0;
Panayotov et al. 2015) — public, non-Indigenous, non-community speech safe to
commit directly — not the real community-representative set described in
FLAIME's 26Q1-DEMO-05 (still NOT STARTED upstream). See
tests/fixtures/community_eval/manifest.yaml for provenance and per-utterance
attribution.
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


def test_every_licensing_note_cites_librispeech(repo_root: Path) -> None:
    """Every sample must record the real license + citation, not just be non-empty.

    LibriSpeech test-clean is CC BY 4.0 (Panayotov et al. 2015) — attribution
    is a license term, not a courtesy, so this is checked per-sample rather
    than trusted to one manifest-header comment.
    """
    for s in _load_samples(repo_root):
        note = s["licensing_note"]
        assert "CC BY 4.0" in note, s["audio_path"]
        assert "Panayotov" in note, s["audio_path"]


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
