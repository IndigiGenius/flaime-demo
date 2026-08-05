"""FLAIME demo UI — audio-to-text transcription (DEMO-02).

Run locally (loopback only, default):
    uv run python flaime_demo/app.py

Run with a real checkpoint:
    uv run python flaime_demo/app.py --checkpoint /path/to/ckpt --model-type xeus

Expose on network (conference setup — shows a warning banner in the UI):
    uv run python flaime_demo/app.py --bind 0.0.0.0

Sovereignty guarantee: audio bytes are processed entirely in RAM.
Nothing is written to disk or sent off-device.
"""

from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys

import numpy as np
import soundfile as sf
from flaime_serving import (
    ASRInferenceEngine,
    EnginePool,
    LanguageNotSupportedError,
    LanguageRouter,
    RouteResult,
    TranscriptionResult,
)

from flaime_demo import errors

# ---------------------------------------------------------------------------
# Module-level constants — importable and testable without Streamlit runtime
# ---------------------------------------------------------------------------

DEFAULT_BIND: str = "127.0.0.1"
DEFAULT_PORT: int = 8501

MSG_NO_AUDIO: str = "Upload or record audio, then click Transcribe."
MSG_SOVEREIGNTY: str = (
    "All audio is processed locally on this machine — nothing leaves this device."
)
MSG_NO_MODEL: str = (
    "No checkpoint loaded. "
    "Pass --checkpoint <path> --model-type <name> to enable transcription."
)
MSG_PUBLIC_BIND: str = (
    "⚠️  App bound to a non-default interface. "
    "Audio is still processed locally, but the UI may be reachable on the network."
)

# ---------------------------------------------------------------------------
# Pure helper functions — no Streamlit dependency; fully unit-testable
# ---------------------------------------------------------------------------


def _bytes_to_audio(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode WAV/FLAC/OGG bytes to a float32 mono numpy array.

    Audio never touches disk — decoding happens entirely in RAM.

    Args:
        audio_bytes: Raw bytes from st.file_uploader or st.audio_input.

    Returns:
        Tuple of (mono float32 array, sample_rate_hz).

    Raises:
        RuntimeError: If soundfile cannot decode the format.
    """
    # TODO: handle MP3 (requires libsndfile with MP3 support; document in README)
    try:
        buf = io.BytesIO(audio_bytes)
        audio_array, sample_rate = sf.read(buf, dtype="float32")
        if audio_array.ndim > 1:
            # Collapse stereo/multi-channel to mono by averaging channels
            audio_array = audio_array.mean(axis=1)
        return audio_array, sample_rate

    except Exception as e:
        raise RuntimeError(f"Could not decode audio: {e}") from e


def _build_language_options(router: LanguageRouter) -> dict[str, str]:
    """Return ``{code: display_name}`` for all languages in the routing config.

    Used to populate the Streamlit language selector in DEMO-03.  The returned
    dict matches the YAML config order exactly (insertion-order dict, Python 3.7+).

    Args:
        router: Loaded LanguageRouter instance.

    Returns:
        Ordered ``{code: display_name}`` mapping.
    """
    return router.supported_languages()


def _format_checkpoint_badge(route: RouteResult) -> str:
    """Return a human-readable badge string for the metadata row.

    Displayed next to latency_ms so partners can see whether an expert or the
    merged checkpoint served their request.

    Args:
        route: RouteResult from LanguageRouter.resolve().

    Returns:
        Badge string, e.g. ``"Expert checkpoint"`` or ``"Merged checkpoint"``.
    """
    return "Expert" if route.checkpoint_type == "expert" else "Merged"


def _run_transcription_routed(
    audio_bytes: bytes,
    language_code: str,
    router: LanguageRouter,
    pool: EnginePool,
) -> tuple[TranscriptionResult, RouteResult]:
    """Resolve language → engine → transcription using the router and pool.

    This is the DEMO-03 replacement for ``_run_transcription()`` in router mode.
    Audio bytes are processed entirely in RAM (sovereignty constraint preserved).

    Args:
        audio_bytes: Raw audio bytes (WAV, FLAC, OGG).
        language_code: BCP-47/ISO 639-3 code selected in the UI.
        router: Loaded LanguageRouter (resolves lang → checkpoint).
        pool: EnginePool (caches engines per checkpoint path).

    Returns:
        Tuple of (TranscriptionResult, RouteResult) so the UI can display both
        the transcript and the checkpoint-type badge.

    Raises:
        LanguageNotSupportedError: If language_code is not in the routing config.
        RuntimeError: If audio decoding fails (propagated from _bytes_to_audio).
    """
    route = router.resolve(language_code)
    engine = pool.get_or_load(route)
    audio_array, sample_rate = _bytes_to_audio(audio_bytes)
    result = engine.transcribe(
        audio_array, language=language_code, sample_rate=sample_rate
    )
    return result, route


def _run_transcription(
    audio_bytes: bytes,
    language: str | None,
    engine: ASRInferenceEngine,
) -> TranscriptionResult:
    """Decode audio bytes and run inference. No Streamlit dependency.

    Separated from _run_app() so tests can call it with a stub engine
    without requiring a Streamlit runtime or real model weights.

    Args:
        audio_bytes: Raw audio bytes (WAV, FLAC, OGG).
        language: BCP-47/ISO 639-3 language code, or None for auto-detect.
        engine: Loaded ASRInferenceEngine instance.

    Returns:
        TranscriptionResult with text, latency, model info, and language.
    """
    audio_array, sample_rate = _bytes_to_audio(audio_bytes)
    return engine.transcribe(audio_array, language=language, sample_rate=sample_rate)


def _guard_audio(audio_array: np.ndarray, sample_rate: int) -> str | None:
    """Run pre-inference audio guards; return a user message or ``None``.

    Rejects silent/empty audio and over-long clips before they reach the model
    (protects latency + memory and avoids a raw ``ValueError`` from the engine).
    """
    silence_msg = errors.check_silence(audio_array)
    if silence_msg is not None:
        return silence_msg
    return errors.check_duration(len(audio_array), sample_rate)


def _safe_run_transcription(
    audio_bytes: bytes,
    language: str | None,
    engine: ASRInferenceEngine,
) -> tuple[TranscriptionResult | None, str | None]:
    """Decode → guard → transcribe, never raising to the caller.

    Returns ``(result, None)`` on success or ``(None, message)`` on any guarded
    or caught error, so the Streamlit layer just renders the message instead of
    a stack trace. Errors are logged aggregate-only (class + duration).

    Args:
        audio_bytes: Raw audio bytes (WAV, FLAC, OGG).
        language: BCP-47/ISO 639-3 code, or None for auto-detect.
        engine: Loaded ASRInferenceEngine instance.

    Returns:
        Tuple of (TranscriptionResult or None, user-facing message or None).
    """
    try:
        audio_array, sample_rate = _bytes_to_audio(audio_bytes)
    except Exception as exc:  # decode failure → friendly "couldn't read" message
        errors.log_error(exc)
        return None, errors.message_for_exception(exc)

    guard_msg = _guard_audio(audio_array, sample_rate)
    if guard_msg is not None:
        return None, guard_msg

    try:
        result = engine.transcribe(
            audio_array, language=language, sample_rate=sample_rate
        )
    except Exception as exc:  # OOM / runtime / unexpected → mapped + logged
        duration_s = len(audio_array) / sample_rate if sample_rate else None
        errors.log_error(exc, duration_s=duration_s)
        return None, errors.message_for_exception(exc)
    return result, None


def _safe_run_transcription_routed(
    audio_bytes: bytes,
    language_code: str,
    router: LanguageRouter,
    pool: EnginePool,
) -> tuple[tuple[TranscriptionResult, RouteResult] | None, str | None]:
    """Router-mode counterpart to :func:`_safe_run_transcription`.

    Resolves language → checkpoint → engine, then delegates the decode/guard/
    transcribe path. Never raises: returns ``((result, route), None)`` on
    success or ``(None, message)`` on any error.

    Args:
        audio_bytes: Raw audio bytes.
        language_code: Selected language code.
        router: Loaded LanguageRouter.
        pool: EnginePool caching engines per checkpoint.

    Returns:
        Tuple of ((TranscriptionResult, RouteResult) or None, message or None).
    """
    try:
        route = router.resolve(language_code)
        engine = pool.get_or_load(route)
    except LanguageNotSupportedError as exc:
        # Already a user-facing message from DEMO-03's router.
        return None, str(exc)
    except Exception as exc:  # missing checkpoint, load failure, etc.
        errors.log_error(exc)
        return None, errors.message_for_exception(exc)

    result, message = _safe_run_transcription(audio_bytes, language_code, engine)
    if result is None:
        return None, message
    return (result, route), None


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the launcher (python flaime_demo/app.py ...).

    These args are consumed by the __main__ launcher block and passed to
    Streamlit via environment variables. They are NOT Streamlit script args.
    """
    p = argparse.ArgumentParser(
        description="FLAIME demo UI — local audio-to-text transcription.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--bind",
        default=DEFAULT_BIND,
        metavar="ADDR",
        help=f"Address to bind (default: {DEFAULT_BIND}). "
        "Use 0.0.0.0 to expose on the network (shows a warning banner).",
    )
    p.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        metavar="PORT",
        help=f"Port to listen on (default: {DEFAULT_PORT}).",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        metavar="PATH",
        help="Path to a FLAIME checkpoint directory. "
        "If omitted, the app starts without a loaded model.",
    )
    p.add_argument(
        "--model-type",
        dest="model_type",
        default="xeus",
        metavar="TYPE",
        help="Model architecture key (e.g. xeus, whisper). Default: xeus.",
    )
    p.add_argument(
        "--device",
        default=None,
        metavar="DEVICE",
        help="Torch device string (e.g. cpu, cuda). Auto-detects if omitted.",
    )
    p.add_argument(
        "--decoder",
        default="ctc_greedy",
        metavar="DECODER",
        help="Decoding strategy: ctc_greedy or ctc_beam<N>. Default: ctc_greedy.",
    )
    # DEMO-03: language routing mode (replaces single --checkpoint in router mode)
    p.add_argument(
        "--languages-config",
        dest="languages_config",
        default=None,
        metavar="PATH",
        help=(
            "Path to the YAML language routing config "
            "(e.g. configs/serving/demo_languages.yaml). "
            "When set, --checkpoint is ignored and the router selects the checkpoint "
            "per language selection."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Engine loading — cached across Streamlit reruns via st.cache_resource
# ---------------------------------------------------------------------------


def _load_router_and_pool() -> tuple[LanguageRouter, EnginePool] | None:
    """Load LanguageRouter + EnginePool from env var set by the DEMO-03 launcher.

    Returns ``None`` when ``DEMO_LANGUAGES_CONFIG`` is not set (falls back to
    single-model DEMO-02 mode so the app is not broken by partial migration).

    This function is wrapped with @st.cache_resource in _run_app() so the
    router and pool are created once and reused across Streamlit reruns.

    Returns:
        (LanguageRouter, EnginePool) when in router mode, None otherwise.
    """
    config_path = os.getenv("DEMO_LANGUAGES_CONFIG")
    if not config_path:
        return None
    device = os.getenv("DEMO_DEVICE") or None

    try:
        router = LanguageRouter.from_yaml(config_path)
        # DEMO-07: warm engines as they load so cold-start lands at
        # language-select time (see _route_to_preload), not on first transcribe.
        pool = EnginePool(device=device, warmup=True)
        return router, pool
    except (FileNotFoundError, ValueError) as e:
        import streamlit as st

        st.error(f"Failed to load language routing config: {e}")
        return None


def _load_engine() -> ASRInferenceEngine | None:
    """Load ASRInferenceEngine from env vars set by the launcher.

    Returns None (no error) when no checkpoint was specified — the UI
    shows a warning banner in that state instead of crashing.

    This function is wrapped with @st.cache_resource in _run_app() so the
    model is loaded once and reused across Streamlit reruns.
    """
    checkpoint = os.getenv("DEMO_CHECKPOINT")
    model_type = os.getenv("DEMO_MODEL_TYPE", "xeus")
    device = os.getenv("DEMO_DEVICE") or None
    decoder = os.getenv("DEMO_DECODER", "ctc_greedy")

    if not checkpoint:
        return None

    try:
        # DEMO-07: warm only when single-engine mode is the active path. This
        # engine loads eagerly at startup under the "Loading model…" spinner, so
        # warming moves cold-start off the first transcription — but in router
        # mode (DEMO_LANGUAGES_CONFIG set) router_and_pool wins and this engine
        # is never used, so warming it would burn a throwaway forward for nothing.
        warmup = not os.getenv("DEMO_LANGUAGES_CONFIG")
        return ASRInferenceEngine.load(
            checkpoint,
            model_type=model_type,
            device=device,
            decoder=decoder,
            warmup=warmup,
        )
    except Exception as e:
        import streamlit as st

        st.error(f"Error loading model: {e}")
        return None


def _route_to_preload(
    router: LanguageRouter, pool: EnginePool, lang_code: str
) -> RouteResult | None:
    """Return the route whose engine should be warmed now, or ``None``.

    DEMO-07: the UI calls this right after the language selector so the engine
    for the chosen language can be loaded-and-warmed *before* the user clicks
    Transcribe, moving cold-start off the user-facing transcription path. Returns
    ``None`` when the language is unsupported (the real error surfaces on the
    transcribe attempt) or its engine is already cached (no work to do).

    Args:
        router: Resolves a language code to a RouteResult.
        pool: Engine cache; consulted to skip already-loaded checkpoints.
        lang_code: Selected language code.

    Returns:
        The RouteResult to preload, or ``None`` if nothing should load now.
    """
    try:
        route = router.resolve(lang_code)
    except LanguageNotSupportedError:
        return None
    if route.checkpoint_path in pool.loaded_checkpoints():
        return None
    return route


# ---------------------------------------------------------------------------
# Streamlit UI — requires Streamlit runtime; not called at import time
# ---------------------------------------------------------------------------


def _run_app() -> None:
    """Render the full Streamlit demo UI.

    Called only when Streamlit runtime is active (i.e., from within
    `streamlit run`). Never called at module import time or from tests.
    """
    import streamlit as st

    from flaime_demo.languages import LANGUAGES_32

    st.set_page_config(page_title="FLAIME Demo", page_icon="🎙️")

    # DEMO-03: try router mode first; fall back to DEMO-02 single-engine mode.
    cached_load_router = st.cache_resource(_load_router_and_pool)
    router_and_pool: tuple[LanguageRouter, EnginePool] | None = cached_load_router()

    cached_load = st.cache_resource(_load_engine)
    with st.spinner("Loading model… (first run may take a minute)"):
        engine: ASRInferenceEngine | None = cached_load()

    st.info(MSG_SOVEREIGNTY)
    if os.getenv("DEMO_PUBLIC_BIND") == "1":
        st.warning(MSG_PUBLIC_BIND)
    if router_and_pool is None and engine is None:
        st.warning(MSG_NO_MODEL)

    with st.sidebar:
        st.image(
            os.path.join(os.path.dirname(__file__), "images", "flair_logo.png"),
            use_container_width=True,
        )
        st.subheader("Model")
        if router_and_pool is not None:
            router, pool = router_and_pool
            n = len(router.supported_languages())
            st.write(f"Mode: language router ({n} languages)")
        elif engine is not None:
            st.write(f"Revision: {engine.model_revision}")
            st.write(f"Decoder: {engine.decoder}")

    upload_tab, record_tab = st.tabs(["Upload file", "Record microphone"])
    with upload_tab:
        uploaded = st.file_uploader(
            "Audio file",
            type=["wav", "flac", "ogg"],
            on_change=lambda: st.session_state.update({"_audio_source": "upload"}),
        )
    with record_tab:
        recorded = st.audio_input(
            "Record from microphone",
            on_change=lambda: st.session_state.update({"_audio_source": "record"}),
        )

    # Use whichever source was last changed; avoids an earlier upload silently
    # overriding a new recording (both widgets retain their value across reruns).
    _source = st.session_state.get("_audio_source")
    if _source == "upload":
        audio_bytes = uploaded.getvalue() if uploaded else None
    elif _source == "record":
        audio_bytes = recorded.getvalue() if recorded else None
    else:
        audio_bytes = None

    # DEMO-03: Language selector — populated from YAML config in router mode,
    # or from LANGUAGES_32 in single-engine fallback mode.
    if router_and_pool is not None:
        router, pool = router_and_pool
        lang_options = _build_language_options(router)
        lang_code: str = st.selectbox(
            "Language",
            list(lang_options.keys()),
            format_func=lambda k: lang_options.get(k, k),
        )  # type: ignore[assignment]
        # DEMO-07: warm the selected language's engine now, under an expected
        # "Loading…" spinner, so the first Transcribe doesn't pay cold-start.
        # Track attempted checkpoints in session_state so a *failed* preload is
        # tried once, not re-fired (spinner + error) on every Streamlit rerun —
        # the real error still surfaces when the user clicks Transcribe.
        preload_route = _route_to_preload(router, pool, lang_code)
        attempted_preloads: set[str] = st.session_state.setdefault(
            "_preload_attempted", set()
        )
        if (
            preload_route is not None
            and preload_route.checkpoint_path not in attempted_preloads
        ):
            attempted_preloads.add(preload_route.checkpoint_path)
            with st.spinner(f"Loading {lang_options.get(lang_code, lang_code)} model…"):
                try:
                    pool.get_or_load(preload_route)
                except Exception as e:  # surfaces again on the transcribe attempt
                    st.error(f"Error loading model: {e}")
    else:
        fallback_options = {"": "Auto-detect"} | LANGUAGES_32
        lang_code = st.selectbox(
            "Language",
            list(fallback_options.keys()),
            format_func=lambda k: fallback_options.get(k, k),
        )

    if audio_bytes is None:
        st.info(MSG_NO_AUDIO)
    else:
        transcribe_disabled = router_and_pool is None and engine is None
        if st.button("Transcribe", disabled=transcribe_disabled):
            with st.spinner("Transcribing…"):
                if router_and_pool is not None:
                    router, pool = router_and_pool
                    routed, message = _safe_run_transcription_routed(
                        audio_bytes, lang_code, router, pool
                    )
                    if routed is None:
                        st.error(message)
                    else:
                        result, route = routed
                        st.session_state["_last_model_type"] = route.model_type
                        st.text_area("Transcription", result.text, height=120)
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Latency", f"{result.latency_ms:.0f} ms")
                        col2.metric(
                            "Confidence",
                            f"{result.confidence:.2f}"
                            if result.confidence is not None
                            else "n/a",
                        )
                        col3.metric("Checkpoint", _format_checkpoint_badge(route))
                elif engine is not None:
                    result, message = _safe_run_transcription(
                        audio_bytes, lang_code or None, engine
                    )
                    if result is None:
                        st.error(message)
                    else:
                        st.text_area("Transcription", result.text, height=120)
                        col1, col2 = st.columns(2)
                        col1.metric("Latency", f"{result.latency_ms:.0f} ms")
                        col2.metric(
                            "Confidence",
                            f"{result.confidence:.2f}"
                            if result.confidence is not None
                            else "n/a",
                        )
                        st.caption(
                            f"Model: {result.model_revision} · Decoder: {result.decoder}"
                        )

    if router_and_pool is not None:
        with st.sidebar:
            last_model_type = st.session_state.get("_last_model_type", "—")
            st.write(f"Model type: {last_model_type}")

    st.caption(MSG_SOVEREIGNTY)


# ---------------------------------------------------------------------------
# Entry point — launches Streamlit when run as `python flaime_demo/app.py`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import streamlit as st

    if not st.runtime.exists():
        # Running as `python flaime_demo/app.py` — launch a Streamlit subprocess.
        # st.runtime.exists() returns False here, True inside Streamlit's runner,
        # so the subprocess does NOT re-enter this branch (no infinite loop).
        args = _parse_args()

        if args.bind != DEFAULT_BIND:
            os.environ["DEMO_PUBLIC_BIND"] = "1"
            print(  # noqa: T201
                f"WARNING: binding to {args.bind} — "
                "the demo UI will be reachable on the network.",
                file=sys.stderr,
            )

        if args.languages_config:
            os.environ["DEMO_LANGUAGES_CONFIG"] = args.languages_config
            os.environ.pop("DEMO_CHECKPOINT", None)
            os.environ.pop("DEMO_MODEL_TYPE", None)
        elif args.checkpoint:
            os.environ["DEMO_CHECKPOINT"] = args.checkpoint
            os.environ["DEMO_MODEL_TYPE"] = args.model_type
        if args.device:
            os.environ["DEMO_DEVICE"] = args.device
        os.environ["DEMO_DECODER"] = args.decoder

        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            __file__,
            "--server.address",
            args.bind,
            "--server.port",
            str(args.port),
            # Disable Streamlit's built-in telemetry (sovereignty requirement)
            "--browser.gatherUsageStats",
            "false",
        ]
        # TODO: consider adding --server.headless true for non-interactive environments
        subprocess.run(cmd)
    else:
        _run_app()
