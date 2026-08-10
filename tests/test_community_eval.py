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
    build_transcriber,
    evaluate_samples,
    load_manifest,
    render_report,
    run_eval,
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


def test_load_manifest_empty_reference_text_raises(tmp_path: Path) -> None:
    # An empty reference_text would make compute_wer return inf and silently
    # poison the category mean_wer — reject it at load time, like a missing field.
    path = _write_manifest(
        tmp_path,
        """
        samples:
          - audio_path: en/golden_001.wav
            language: en
            reference_text: ""
            category: golden
            licensing_note: CV-CC0
        """,
    )
    with pytest.raises(ManifestError, match="reference_text"):
        load_manifest(path)


def test_load_manifest_blank_licensing_note_raises(tmp_path: Path) -> None:
    # Sovereignty: a whitespace-only licensing_note is as good as missing.
    path = _write_manifest(
        tmp_path,
        """
        samples:
          - audio_path: en/golden_001.wav
            language: en
            reference_text: hello world
            category: golden
            licensing_note: "   "
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
    # The cause must be captured, not swallowed — a report full of FAILs with
    # no reason forces a re-run just to see the traceback.
    assert "RuntimeError" in r.error
    assert "checkpoint missing" in r.error


def test_aggregate_includes_mean_cer() -> None:
    results = [
        SampleResult("a", "en", "golden", "abc", 0.0, 100.0, "c", "g", False, cer=10.0),
        SampleResult(
            "b", "en", "golden", "xyz", 50.0, 300.0, "c", "g", False, cer=30.0
        ),
    ]
    agg = aggregate_by_category(results)
    assert agg["golden"]["mean_cer"] == pytest.approx(20.0)


def _two_category_results() -> list[SampleResult]:
    return [
        SampleResult("en/g1.wav", "en", "golden", "hello", 0.0, 100.0, "c", "g", False),
        SampleResult(
            "hi/t1.wav",
            "hi",
            "transfer",
            "",
            100.0,
            0.0,
            "c",
            "g",
            True,
            error="RuntimeError: backend could not decode hi/t1.wav",
        ),
    ]


def test_render_report_has_required_sections() -> None:
    results = _two_category_results()
    report = render_report(results, aggregate_by_category(results))
    # Category aggregate table
    assert "Aggregate by category" in report
    assert "golden" in report and "transfer" in report
    # Per-sample table with the sample identifier
    assert "Per-sample" in report
    assert "en/g1.wav" in report
    # Worst-sample notes + a demo-readiness verdict
    assert "Worst" in report
    assert "Demo-readiness" in report


def test_render_report_flags_failed_samples() -> None:
    results = _two_category_results()
    report = render_report(results, aggregate_by_category(results))
    # The failed transfer sample must be visibly marked, not silently dropped.
    assert "FAIL" in report
    # The captured error must surface in the report so a page of FAILs is
    # diagnosable without re-running under a debugger.
    assert "## Failures" in report
    assert "backend could not decode" in report


def test_build_transcriber_routes_loads_then_transcribes() -> None:
    seen: dict[str, object] = {}

    class _FakeRoute:
        pass

    class _FakeEngine:
        def transcribe(self, audio: Path, language: str) -> _StubTranscription:
            seen["audio"] = audio
            seen["lang"] = language
            return _StubTranscription("hi", 50.0, "ckpt-x", "ctc_greedy", 0.8)

    class _FakeRouter:
        def resolve(self, language_code: str) -> _FakeRoute:
            seen["resolved"] = language_code
            return _FakeRoute()

    class _FakePool:
        def get_or_load(self, route: object) -> _FakeEngine:
            seen["route"] = route
            return _FakeEngine()

    transcribe = build_transcriber(_FakeRouter(), _FakePool())
    result = transcribe(Path("en/g1.wav"), "en")

    assert result.text == "hi"
    assert seen["resolved"] == "en"
    assert isinstance(seen["route"], _FakeRoute)
    assert seen["lang"] == "en"


def test_run_eval_end_to_end_writes_report(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path,
        """
        samples:
          - audio_path: en/g1.wav
            language: en
            reference_text: hello world
            category: golden
            licensing_note: CV-CC0
          - audio_path: hi/t1.wav
            language: hi
            reference_text: namaste
            category: transfer
            licensing_note: synthesized
        """,
    )

    def _stub(audio_path: Path, language: str) -> _StubTranscription:
        text = "hello world" if language == "en" else ""
        return _StubTranscription(text, 90.0, "ckpt", "ctc_greedy", 0.7)

    out = tmp_path / "report.md"
    results, aggregates, report = run_eval(path, tmp_path, _stub, out_path=out)

    assert len(results) == 2
    assert set(aggregates) == {"golden", "transfer"}
    assert out.exists()
    written = out.read_text()
    assert "Aggregate by category" in written
    assert written == report


def test_committed_manifest_is_valid(pytestconfig: pytest.Config) -> None:
    manifest = pytestconfig.rootpath / "tests" / "fixtures" / "community_samples.yaml"
    samples = load_manifest(manifest)
    assert len(samples) >= 10
    assert {s.category for s in samples} == {"golden", "transfer", "edge"}
    # Sovereignty: every shipped sample carries a non-empty licensing note.
    assert all(s.licensing_note for s in samples)
