"""Community-representative demo evaluation runner (26Q1-DEMO-05).

Loads a sample manifest, drives the demo serving path per sample, and reports
WER + latency aggregated by category (golden / transfer / edge). No contributor
audio bytes are ever committed — the manifest references audio in a documented
external location and carries a mandatory ``licensing_note`` per sample
(Indigenous data-sovereignty constraint).

This module deliberately keeps the report/data layer free of any model imports
so it is testable with a stub engine and no checkpoints.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

# Required keys for every manifest sample. ``licensing_note`` is mandatory: a
# missing one fails the load rather than silently shipping unlicensed audio.
_REQUIRED_FIELDS = (
    "audio_path",
    "language",
    "reference_text",
    "category",
    "licensing_note",
)


class ManifestError(ValueError):
    """Raised when the sample manifest is malformed or missing required fields."""


@dataclasses.dataclass(frozen=True)
class Sample:
    """One manifest entry to evaluate.

    Attributes:
        audio_path: Path relative to a documented samples root (never committed).
        language: BCP-47 / ISO 639-3 code (e.g. ``"en"``).
        reference_text: Ground-truth transcript.
        category: ``"golden"`` (training-set langs), ``"transfer"`` (held-out),
            or ``"edge"`` (codeswitch / noise / accent variants).
        licensing_note: Provenance/licence string. Mandatory.
    """

    audio_path: str
    language: str
    reference_text: str
    category: str
    licensing_note: str


@dataclasses.dataclass(frozen=True)
class SampleResult:
    """Per-sample evaluation outcome.

    Attributes:
        audio_path: Sample identifier (the manifest ``audio_path``).
        language: BCP-47 / ISO 639-3 code.
        category: golden / transfer / edge.
        hypothesis: Decoded transcript (empty string on failure).
        wer: Word error rate vs reference, as a percentage (0-100).
        latency_ms: Wall-clock transcription latency in milliseconds.
        checkpoint: Served checkpoint path/ID.
        decoder: Decoding strategy used (e.g. ``"ctc_greedy"``).
        failed: True when transcription raised; ``hypothesis`` is then empty and
            ``wer`` is recorded as 100.0.
    """

    audio_path: str
    language: str
    category: str
    hypothesis: str
    wer: float
    latency_ms: float
    checkpoint: str
    decoder: str
    failed: bool
    cer: float = 0.0
    confidence: float | None = None


class TranscriptionLike(Protocol):
    """Structural type for a transcription result (avoids a torch import here).

    Matched by ``flaime_serving.TranscriptionResult`` (frozen API) and by test
    stubs.
    """

    text: str
    latency_ms: float
    model_revision: str
    decoder: str
    confidence: float | None


# A transcriber maps (resolved audio path, language code) -> a transcription
# result. Production wiring (router -> engine pool -> engine) builds one of
# these; tests inject a stub so no checkpoints or audio bytes are needed.
Transcriber = Callable[[Path, str], TranscriptionLike]


def load_manifest(path: str | Path) -> list[Sample]:
    """Load and validate the community-sample manifest.

    Args:
        path: Path to the manifest YAML. Expects a top-level ``samples`` list.

    Returns:
        List of validated :class:`Sample` entries, in manifest order.

    Raises:
        ManifestError: If the file has no ``samples`` list, or any sample is
            missing a required field (including the mandatory ``licensing_note``).
    """
    with open(path, encoding="utf-8") as fh:
        raw: Any = yaml.safe_load(fh)

    if not isinstance(raw, dict) or "samples" not in raw:
        raise ManifestError(f"{path}: manifest must have a top-level 'samples' list")

    entries = raw["samples"]
    if not isinstance(entries, list) or not entries:
        raise ManifestError(f"{path}: 'samples' must be a non-empty list")

    samples: list[Sample] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ManifestError(f"{path}: sample {i} is not a mapping")
        missing = [k for k in _REQUIRED_FIELDS if k not in entry or entry[k] is None]
        if missing:
            raise ManifestError(
                f"{path}: sample {i} ({entry.get('audio_path', '?')}) "
                f"missing required field(s): {', '.join(missing)}"
            )
        samples.append(
            Sample(
                audio_path=str(entry["audio_path"]),
                language=str(entry["language"]),
                reference_text=str(entry["reference_text"]),
                category=str(entry["category"]),
                licensing_note=str(entry["licensing_note"]),
            )
        )
    return samples


def evaluate_samples(
    samples: list[Sample],
    transcribe: Transcriber,
    *,
    samples_root: Path,
) -> list[SampleResult]:
    """Drive every sample through *transcribe* and score WER/CER + latency.

    A sample whose transcription raises is flagged ``failed`` (empty hypothesis,
    WER recorded as 100.0) but does NOT abort the sweep — one bad sample must not
    cancel the demo-readiness report (DEMO-05 risk: "don't gate the demo on a
    single failure").

    Args:
        samples: Validated manifest entries.
        transcribe: Callable mapping (resolved audio path, language) to a
            transcription result. Production wires router -> engine pool here;
            tests inject a stub.
        samples_root: Directory the manifest ``audio_path`` values are relative
            to. Never read by this function directly — passed to *transcribe*.

    Returns:
        One :class:`SampleResult` per input sample, in order.
    """
    # Lazy import: keeps this module import-light, while reusing the canonical
    # WER/CER helpers rather than reimplementing jiwer wrappers.
    from scripts.metrics import compute_cer, compute_wer

    results: list[SampleResult] = []
    for s in samples:
        audio_path = samples_root / s.audio_path
        try:
            tr = transcribe(audio_path, s.language)
        except Exception:  # noqa: BLE001 - flag bad samples, never abort the sweep
            results.append(
                SampleResult(
                    audio_path=s.audio_path,
                    language=s.language,
                    category=s.category,
                    hypothesis="",
                    wer=100.0,
                    latency_ms=0.0,
                    checkpoint="",
                    decoder="",
                    failed=True,
                )
            )
            continue
        results.append(
            SampleResult(
                audio_path=s.audio_path,
                language=s.language,
                category=s.category,
                hypothesis=tr.text,
                wer=compute_wer([tr.text], [s.reference_text]),
                latency_ms=tr.latency_ms,
                checkpoint=tr.model_revision,
                decoder=tr.decoder,
                failed=False,
                cer=compute_cer([tr.text], [s.reference_text]),
                confidence=tr.confidence,
            )
        )
    return results


def aggregate_by_category(results: list[SampleResult]) -> dict[str, dict[str, float]]:
    """Aggregate per-sample results by category.

    Args:
        results: Per-sample evaluation outcomes.

    Returns:
        Mapping of category -> aggregate dict with keys ``sample_count``,
        ``failure_count``, ``mean_wer``, ``p50_latency_ms``, ``p95_latency_ms``.
        Latency percentiles use linear interpolation over all samples in the
        category (failures included — a failure still has a measured latency).
    """
    by_cat: dict[str, list[SampleResult]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)

    agg: dict[str, dict[str, float]] = {}
    for category, rows in by_cat.items():
        wers = np.array([r.wer for r in rows], dtype=float)
        latencies = np.array([r.latency_ms for r in rows], dtype=float)
        agg[category] = {
            "sample_count": float(len(rows)),
            "failure_count": float(sum(1 for r in rows if r.failed)),
            "mean_wer": float(wers.mean()),
            "p50_latency_ms": float(np.percentile(latencies, 50)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
        }
    return agg
