"""Tests for the demo error-handling layer (flaime_demo/errors.py).

Run: uv run pytest tests/test_error_handling.py -v

The error layer is a pure, Streamlit-free mapping from exceptions and audio
guards to user-facing messages. The Streamlit UI renders whatever these pure
functions return; nothing here requires a Streamlit runtime or real weights.

Moved from FLAIME `tests/demo/test_error_handling.py` (26Q3-REPO-08). The
original file's `TestSafeRunTranscription` class is NOT here: it imports
`apps.demo.app`, which REPO-08 explicitly fences out. Those four tests travel
with `app.py` in REPO-09.
"""

from __future__ import annotations

import numpy as np

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
