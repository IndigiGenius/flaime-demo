"""Tests for the demo error-handling layer (flaime_demo/errors.py).

Run: uv run pytest tests/test_error_handling.py -v

The error layer is a pure, Streamlit-free mapping from exceptions and audio
guards to user-facing messages. The Streamlit UI renders whatever these pure
functions return; nothing here requires a Streamlit runtime or real weights.

Moved from FLAIME `tests/demo/test_error_handling.py` (26Q3-REPO-08). REPO-08
left `TestSafeRunTranscription` behind because it imports `apps.demo.app`,
which hadn't moved yet; those four tests are reunited here in 26Q3-REPO-10,
now that `flaime_demo.app` exists. This is also why `torch` reappears as a
test dependency (declared in pyproject.toml's dev group) — the stub models
below need it.
"""

from __future__ import annotations

import numpy as np
import torch
from flaime_serving import ASRInferenceEngine

from flaime_demo.errors import (
    DEFAULT_MAX_DURATION_S,
    MSG_BAD_FORMAT,
    MSG_EMPTY_AUDIO,
    MSG_GENERIC,
    MSG_NO_CHECKPOINT,
    MSG_TOO_LONG,
    check_duration,
    check_silence,
    message_for_exception,
)

TARGET_SR = 16_000


class TestExceptionMapping:
    """Known exception types map to calm, user-facing messages."""

    def test_decode_runtimeerror_maps_to_bad_format(self) -> None:
        exc = RuntimeError("Could not decode audio: format not recognised")
        assert message_for_exception(exc) == MSG_BAD_FORMAT

    def test_empty_waveform_valueerror_maps_to_empty_audio(self) -> None:
        exc = ValueError("Empty waveform after normalisation.")
        assert message_for_exception(exc) == MSG_EMPTY_AUDIO

    def test_missing_checkpoint_maps_to_no_checkpoint(self) -> None:
        exc = FileNotFoundError("Checkpoint not found: /ckpt/ka")
        assert message_for_exception(exc) == MSG_NO_CHECKPOINT

    def test_unknown_exception_maps_to_generic_fallback(self) -> None:
        assert message_for_exception(KeyError("surprise")) == MSG_GENERIC


class TestDurationGuard:
    """Clips longer than the cap are rejected before inference."""

    def test_long_clip_rejected(self) -> None:
        num_samples = int((DEFAULT_MAX_DURATION_S + 5) * TARGET_SR)
        assert check_duration(num_samples, TARGET_SR) == MSG_TOO_LONG

    def test_short_clip_passes(self) -> None:
        assert check_duration(TARGET_SR, TARGET_SR) is None


class TestSilenceGuard:
    """All-zero / sub-threshold audio is rejected before inference."""

    def test_silent_audio_rejected(self) -> None:
        silent = np.zeros(TARGET_SR, dtype=np.float32)
        assert check_silence(silent) == MSG_EMPTY_AUDIO

    def test_audible_audio_passes(self) -> None:
        rng = np.random.default_rng(0)
        audible = rng.standard_normal(TARGET_SR).astype(np.float32)
        assert check_silence(audible) is None


# ---------------------------------------------------------------------------
# UI wiring — flaime_demo/app.py wraps decode + guards + transcribe so no
# exception ever reaches the user as a stack trace.
# ---------------------------------------------------------------------------


def _make_audible_wav_bytes(
    seconds: float = 1.0, sample_rate: int = TARGET_SR
) -> bytes:
    """Return in-memory WAV bytes of audible noise (passes the silence guard)."""
    import io
    import wave

    rng = np.random.default_rng(0)
    n = int(seconds * sample_rate)
    samples = (rng.standard_normal(n) * 8000).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


class _StubProcessor:
    """Minimal stand-in for a model processor (mirrors test_app.py)."""

    def decode(self, token_ids: list[int], **kwargs: object) -> str:  # noqa: ARG002
        return "hello world"


class _OkModel:
    """Stub model returning flat logits — no HuggingFace or GPU needed."""

    revision = "test-stub-v0"

    def forward(
        self,
        input_features: torch.Tensor,
        **kwargs: object,  # noqa: ARG002
    ) -> dict[str, torch.Tensor]:
        t = max(1, input_features.shape[-1] // 160)
        return {"logits": torch.full((1, t, 32), fill_value=-3.0)}

    @property
    def processor(self) -> _StubProcessor:
        return _StubProcessor()

    def eval(self) -> _OkModel:
        return self

    def to(self, device: str) -> _OkModel:  # noqa: ARG002
        return self


class _RaisingModel(_OkModel):
    """Stub model whose forward() raises — simulates an OOM / runtime fault."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def forward(
        self, input_features: torch.Tensor, **kwargs: object
    ) -> dict[str, torch.Tensor]:
        raise self._exc


def _engine(model: _OkModel) -> ASRInferenceEngine:
    return ASRInferenceEngine(
        model=model,
        model_revision="test-stub-v0",
        decoder="ctc_greedy",
        device="cpu",
    )


class TestSafeRunTranscription:
    """_safe_run_transcription returns (result, None) or (None, message)."""

    def test_happy_path_returns_result(self) -> None:
        from flaime_demo.app import _safe_run_transcription

        result, message = _safe_run_transcription(
            _make_audible_wav_bytes(), "en", _engine(_OkModel())
        )
        assert message is None
        assert result is not None
        assert result.text == "hello world"

    def test_corrupt_bytes_returns_bad_format(self) -> None:
        from flaime_demo.app import _safe_run_transcription

        result, message = _safe_run_transcription(
            b"not audio", "en", _engine(_OkModel())
        )
        assert result is None
        assert message == MSG_BAD_FORMAT

    def test_silent_audio_rejected_before_inference(self) -> None:
        import io
        import wave

        from flaime_demo.app import _safe_run_transcription

        buf = io.BytesIO()
        with wave.open(buf, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(TARGET_SR)
            wf.writeframes(np.zeros(TARGET_SR, dtype=np.int16).tobytes())
        # A model that would blow up proves the guard fires *before* inference.
        engine = _engine(_RaisingModel(RuntimeError("should not run")))
        result, message = _safe_run_transcription(buf.getvalue(), "en", engine)
        assert result is None
        assert message == MSG_EMPTY_AUDIO

    def test_transcribe_error_is_caught_and_mapped(self) -> None:
        from flaime_demo.app import _safe_run_transcription

        engine = _engine(_RaisingModel(RuntimeError("CUDA out of memory")))
        result, message = _safe_run_transcription(
            _make_audible_wav_bytes(), "en", engine
        )
        assert result is None
        assert (
            message
            == "The model ran into a problem with that clip. Try a shorter clip."
        )
