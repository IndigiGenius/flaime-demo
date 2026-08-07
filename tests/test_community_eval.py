"""Tests for ``scripts/run_community_eval.py`` (26Q1-DEMO-05).

The runner loads a sample manifest, drives the demo serving path per sample,
and reports WER + latency aggregated by category. These tests pin the manifest
schema (mandatory ``licensing_note`` per the sovereignty rule) and the
by-category aggregate schema. They use a stub — no checkpoints, no audio bytes.
"""

from __future__ import annotations

import dataclasses
import textwrap
from pathlib import Path

import pytest

from scripts.run_community_eval import (
    ManifestError,
    Sample,
    SampleResult,
    aggregate_by_category,
    evaluate_samples,
    load_manifest,
)


@dataclasses.dataclass
class _StubTranscription:
    """Structural stand-in for TranscriptionResult (no torch, no checkpoint)."""

    text: str
    latency_ms: float
    model_revision: str
    decoder: str
    confidence: float | None


def _write_manifest(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "community_samples.yaml"
    path.write_text(textwrap.dedent(body))
    return path


def test_load_manifest_parses_required_fields(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path,
        """
        samples:
          - audio_path: en/golden_001.wav
            language: en
            reference_text: hello world
            category: golden
            licensing_note: CV-CC0
        """,
    )
    samples = load_manifest(path)
    assert len(samples) == 1
    s = samples[0]
    assert s.audio_path == "en/golden_001.wav"
    assert s.language == "en"
    assert s.reference_text == "hello world"
    assert s.category == "golden"
    assert s.licensing_note == "CV-CC0"


def test_load_manifest_missing_licensing_note_raises(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path,
        """
        samples:
          - audio_path: en/golden_001.wav
            language: en
            reference_text: hello world
            category: golden
        """,
    )
    with pytest.raises(ManifestError, match="licensing_note"):
        load_manifest(path)


def test_aggregate_by_category_schema() -> None:
    results = [
        SampleResult("a", "en", "golden", "a", 0.0, 100.0, "ckpt", "ctc_greedy", False),
        SampleResult(
            "b", "en", "golden", "x", 50.0, 300.0, "ckpt", "ctc_greedy", False
        ),
        SampleResult("c", "hi", "edge", "", 100.0, 0.0, "ckpt", "ctc_greedy", True),
    ]
    agg = aggregate_by_category(results)
    assert set(agg) == {"golden", "edge"}
    golden = agg["golden"]
    assert golden["sample_count"] == 2
    assert golden["failure_count"] == 0
    assert golden["mean_wer"] == pytest.approx(25.0)
    assert golden["p50_latency_ms"] == pytest.approx(200.0)
    assert golden["p95_latency_ms"] == pytest.approx(290.0, abs=15.0)
    assert agg["edge"]["failure_count"] == 1


def test_evaluate_samples_populates_per_sample_fields(tmp_path: Path) -> None:
    def _stub_ok(audio_path: Path, language: str) -> _StubTranscription:
        return _StubTranscription(
            text="hello world",
            latency_ms=120.0,
            model_revision="ckpt-en",
            decoder="ctc_greedy",
            confidence=0.9,
        )

    samples = [Sample("en/g1.wav", "en", "hello world", "golden", "CV-CC0")]
    results = evaluate_samples(samples, _stub_ok, samples_root=tmp_path)
    assert len(results) == 1
    r = results[0]
    assert r.hypothesis == "hello world"
    assert r.wer == pytest.approx(0.0)
    assert r.cer == pytest.approx(0.0)
    assert r.latency_ms == 120.0
    assert r.checkpoint == "ckpt-en"
    assert r.decoder == "ctc_greedy"
    assert r.confidence == pytest.approx(0.9)
    assert r.failed is False


def test_evaluate_samples_records_failure(tmp_path: Path) -> None:
    def _stub_raise(audio_path: Path, language: str) -> _StubTranscription:
        raise RuntimeError("checkpoint missing")

    samples = [Sample("hi/e1.wav", "hi", "namaste", "edge", "synth")]
    results = evaluate_samples(samples, _stub_raise, samples_root=tmp_path)
    r = results[0]
    assert r.failed is True
    assert r.hypothesis == ""
    assert r.wer == pytest.approx(100.0)
