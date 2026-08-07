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
        error: One-line ``ExceptionType: message`` captured when ``failed`` —
            empty on success. Surfaced in the report's Failures section so a
            page of FAILs is diagnosable without a re-run.
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
    error: str = ""


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


class EngineLike(Protocol):
    """Structural type for a loaded inference engine.

    Matched by ``flaime_serving.ASRInferenceEngine`` (frozen API) and by test
    stubs.
    """

    def transcribe(self, audio: Path, language: str) -> TranscriptionLike: ...


class RouterLike(Protocol):
    """Structural type for a language router.

    Matched by ``flaime_serving.LanguageRouter`` (frozen API) and by test
    stubs.
    """

    def resolve(self, language_code: str) -> object: ...


class PoolLike(Protocol):
    """Structural type for an engine pool.

    Matched by ``flaime_serving.EnginePool`` (frozen API) and by test stubs.
    The route is opaque here: it flows from ``RouterLike.resolve`` straight
    into ``get_or_load`` without this module inspecting it.
    """

    def get_or_load(self, route: Any) -> EngineLike: ...


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
        # A blank string is treated as missing: an empty reference_text makes
        # compute_wer return inf (poisoning mean_wer), and a blank licensing_note
        # would ship unlicensed audio — both must fail the load, not slip through.
        missing = [
            k
            for k in _REQUIRED_FIELDS
            if k not in entry or entry[k] is None or not str(entry[k]).strip()
        ]
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
        except Exception as exc:  # noqa: BLE001 - flag bad samples, never abort the sweep
            one_line = " ".join(str(exc).split()) or "(no message)"
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
                    error=f"{type(exc).__name__}: {one_line[:300]}",
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


def build_transcriber(router: RouterLike, pool: PoolLike) -> Transcriber:
    """Wire router -> engine pool -> engine into a single transcriber callable.

    The returned callable resolves the language to a checkpoint route, loads (or
    reuses) the engine for that route, and transcribes — the same code path the
    Streamlit UI uses, so the eval can't diverge from production serving.

    Args:
        router: Resolves a language code to a checkpoint route
            (production: :class:`flaime_serving.LanguageRouter`).
        pool: Caches/loads an engine per resolved route
            (production: :class:`flaime_serving.EnginePool`).

    Returns:
        A :data:`Transcriber` suitable for :func:`evaluate_samples`.
    """

    def _transcribe(audio_path: Path, language: str) -> TranscriptionLike:
        route = router.resolve(language)
        engine = pool.get_or_load(route)
        return engine.transcribe(audio_path, language)

    return _transcribe


def aggregate_by_category(results: list[SampleResult]) -> dict[str, dict[str, float]]:
    """Aggregate per-sample results by category.

    Args:
        results: Per-sample evaluation outcomes.

    Returns:
        Mapping of category -> aggregate dict with keys ``sample_count``,
        ``failure_count``, ``mean_wer``, ``mean_cer``, ``p50_latency_ms``,
        ``p95_latency_ms``. Latency percentiles use linear interpolation over
        all samples in the category, including failures — which record
        ``latency_ms=0.0`` (no transcription happened), so a category with
        failures reads optimistically low on latency. Cross-check
        ``failure_count`` when reading the percentiles.
    """
    by_cat: dict[str, list[SampleResult]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)

    agg: dict[str, dict[str, float]] = {}
    for category, rows in by_cat.items():
        wers = np.array([r.wer for r in rows], dtype=float)
        cers = np.array([r.cer for r in rows], dtype=float)
        latencies = np.array([r.latency_ms for r in rows], dtype=float)
        agg[category] = {
            "sample_count": float(len(rows)),
            "failure_count": float(sum(1 for r in rows if r.failed)),
            "mean_wer": float(wers.mean()),
            "mean_cer": float(cers.mean()),
            "p50_latency_ms": float(np.percentile(latencies, 50)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
        }
    return agg


# Category display order for the report (worst-transfer story reads best last).
_CATEGORY_ORDER = ("golden", "transfer", "edge")


def _category_sort_key(category: str) -> tuple[int, str]:
    order = {c: i for i, c in enumerate(_CATEGORY_ORDER)}
    return (order.get(category, len(_CATEGORY_ORDER)), category)


def render_report(
    results: list[SampleResult],
    aggregates: dict[str, dict[str, float]],
    *,
    title: str = "DEMO-05 Community Eval",
    worst_n: int = 3,
) -> str:
    """Render a Markdown demo-readiness report from eval results.

    Args:
        results: Per-sample outcomes from :func:`evaluate_samples`.
        aggregates: Per-category aggregates from :func:`aggregate_by_category`.
        title: Report H1 title.
        worst_n: How many worst-WER samples to surface in the notes section.

    Returns:
        A Markdown document with an aggregate-by-category table, a per-sample
        table, a worst-samples notes section, and a demo-readiness verdict. The
        verdict is a scaffold for human sign-off — it summarises the numbers but
        does not auto-decide whether to demo.
    """
    lines: list[str] = [f"# {title}", ""]

    # --- Aggregate by category -------------------------------------------------
    lines += [
        "## Aggregate by category",
        "",
        "| Category | Samples | Failures | Mean WER | Mean CER | p50 latency (ms) | p95 latency (ms) |",
        "|----------|--------:|---------:|---------:|---------:|-----------------:|-----------------:|",
    ]
    for category in sorted(aggregates, key=_category_sort_key):
        a = aggregates[category]
        lines.append(
            f"| {category} | {int(a['sample_count'])} | {int(a['failure_count'])} "
            f"| {a['mean_wer']:.1f} | {a.get('mean_cer', 0.0):.1f} "
            f"| {a['p50_latency_ms']:.0f} | {a['p95_latency_ms']:.0f} |"
        )

    # --- Per-sample results ----------------------------------------------------
    lines += [
        "",
        "## Per-sample results",
        "",
        "| Sample | Lang | Category | Status | WER | CER | Latency (ms) | Conf | Hypothesis |",
        "|--------|------|----------|--------|----:|----:|-------------:|-----:|------------|",
    ]
    for r in results:
        status = "FAIL" if r.failed else "ok"
        conf = "—" if r.confidence is None else f"{r.confidence:.2f}"
        hyp = (r.hypothesis[:60] + "…") if len(r.hypothesis) > 60 else r.hypothesis
        lines.append(
            f"| {r.audio_path} | {r.language} | {r.category} | {status} "
            f"| {r.wer:.1f} | {r.cer:.1f} | {r.latency_ms:.0f} | {conf} | {hyp} |"
        )

    # --- Failures (why, not just that) -----------------------------------------
    failed_results = [r for r in results if r.failed]
    if failed_results:
        lines += ["", "## Failures", ""]
        for r in failed_results:
            lines.append(f"- `{r.audio_path}` ({r.language}): {r.error or 'unknown'}")

    # --- Worst samples ---------------------------------------------------------
    worst = sorted(results, key=lambda r: r.wer, reverse=True)[:worst_n]
    lines += ["", f"## Worst {len(worst)} samples (by WER)", ""]
    for r in worst:
        tag = " **(FAIL)**" if r.failed else ""
        lines.append(
            f"- `{r.audio_path}` ({r.language}, {r.category}){tag} — "
            f"WER {r.wer:.1f}, CER {r.cer:.1f}: "
            f"hyp `{r.hypothesis}` — _add note (compare to reference)_"
        )

    # --- Verdict scaffold ------------------------------------------------------
    total = len(results)
    failures = sum(1 for r in results if r.failed)
    lines += [
        "",
        "## Demo-readiness verdict",
        "",
        f"- Samples evaluated: **{total}** ({failures} failed)",
        "- [ ] Golden-path WER acceptable for the demo languages",
        "- [ ] No unhandled failures (stack traces) on edge samples",
        "- [ ] p95 latency within the < 5s budget (DEMO-07)",
        "- **Verdict**: _PENDING human sign-off_",
        "",
    ]
    return "\n".join(lines)


def run_eval(
    manifest_path: str | Path,
    samples_root: str | Path,
    transcribe: Transcriber,
    *,
    out_path: str | Path | None = None,
    title: str = "DEMO-05 Community Eval",
) -> tuple[list[SampleResult], dict[str, dict[str, float]], str]:
    """Load a manifest, evaluate every sample, and render the report.

    This is the testable orchestration core: it takes an injected *transcribe*
    so it runs end-to-end with a stub (no checkpoints). :func:`main` supplies the
    production transcriber.

    Args:
        manifest_path: Path to the sample manifest YAML.
        samples_root: Directory the manifest ``audio_path`` values resolve against.
        transcribe: Transcriber callable (see :func:`build_transcriber`).
        out_path: If given, the Markdown report is written here.
        title: Report title.

    Returns:
        ``(results, aggregates, report_markdown)``.
    """
    samples = load_manifest(manifest_path)
    results = evaluate_samples(samples, transcribe, samples_root=Path(samples_root))
    aggregates = aggregate_by_category(results)
    report = render_report(results, aggregates, title=title)
    if out_path is not None:
        Path(out_path).write_text(report, encoding="utf-8")
    return results, aggregates, report


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: drive the community eval through the production router.

    Args:
        argv: Argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success).
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run the DEMO-05 community eval.")
    parser.add_argument(
        "--manifest", required=True, help="Path to community_samples.yaml"
    )
    parser.add_argument(
        "--samples-root",
        required=True,
        help="Directory the manifest audio_path values resolve against",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Serving language config YAML (LanguageRouter.from_yaml)",
    )
    parser.add_argument("--out", default=None, help="Write the Markdown report here")
    parser.add_argument("--device", default=None, help="torch device (default: auto)")
    args = parser.parse_args(argv)

    # Lazy import: only the CLI path needs the torch-backed serving stack.
    from flaime_serving import EnginePool, LanguageRouter

    router = LanguageRouter.from_yaml(args.config)
    pool = EnginePool(device=args.device)
    transcribe = build_transcriber(router, pool)

    _, aggregates, report = run_eval(
        args.manifest, args.samples_root, transcribe, out_path=args.out
    )
    if args.out is None:
        print(report)
    else:
        print(f"Wrote report to {args.out} ({len(aggregates)} categories)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
