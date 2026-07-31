"""Demo error-handling layer (DEMO-06) — pure, Streamlit-free.

Maps the exceptions that `ASRInferenceEngine.transcribe()` and the audio
decode path can raise into calm, human-readable messages, and provides the
pre-inference audio guards (max-duration, silence) so a live demo never
surfaces a Python stack trace or a frozen UI to a community audience.

Design: every function here is pure and importable without a Streamlit
runtime or model weights (mirrors the ``_bytes_to_audio`` helper split in
``app.py``). The Streamlit layer only renders whatever these return.

Sovereignty: logging is aggregate-only — error class + clip duration, never
audio bytes or transcripts (CLAUDE.md).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("flaime.demo")

# ---------------------------------------------------------------------------
# Guard thresholds — configurable, conservative defaults
# ---------------------------------------------------------------------------

# Clips longer than this are rejected before inference (protects latency +
# memory on the demo hardware). 30 s comfortably covers a spoken sentence.
DEFAULT_MAX_DURATION_S: float = 30.0

# RMS below this is treated as silence. Audio is float32 in roughly [-1, 1];
# 1e-4 RMS is ~ -80 dBFS, well below any real speech.
DEFAULT_RMS_THRESHOLD: float = 1e-4

# ---------------------------------------------------------------------------
# User-facing messages — add MSG_* constants, never scatter string literals
# ---------------------------------------------------------------------------

MSG_EMPTY_AUDIO: str = (
    "That clip sounds silent. Record or upload audio with speech and try again."
)
MSG_BAD_FORMAT: str = "Couldn't read that audio file. Try a WAV, FLAC, or OGG clip."
MSG_TOO_LONG: str = (
    "That clip is too long for the demo. Try a shorter one "
    f"(under {DEFAULT_MAX_DURATION_S:.0f} seconds)."
)
MSG_NO_CHECKPOINT: str = (
    "No model is loaded for the selected language yet. Pick another language."
)
MSG_MODEL_ERROR: str = (
    "The model ran into a problem with that clip. Try a shorter clip."
)
MSG_GENERIC: str = (
    "Something went wrong handling that audio. Try again with a different clip."
)


# ---------------------------------------------------------------------------
# Exception → message mapping
# ---------------------------------------------------------------------------


def message_for_exception(exc: BaseException) -> str:
    """Map a known exception to a calm, user-facing message.

    Falls back to :data:`MSG_GENERIC` for anything unrecognised so no error
    path is ever uncaught.

    Args:
        exc: The exception raised by the decode or transcribe path.

    Returns:
        A user-facing message string (never a stack trace).
    """
    name = type(exc).__name__
    text = str(exc).lower()

    # OOM can arrive as torch.cuda.OutOfMemoryError (a RuntimeError subclass)
    # or a plain RuntimeError whose message mentions memory. Match by name and
    # message so errors.py stays torch-free and fast to import.
    if name == "OutOfMemoryError" or "out of memory" in text:
        return MSG_MODEL_ERROR

    if isinstance(exc, FileNotFoundError):
        return MSG_NO_CHECKPOINT

    if isinstance(exc, RuntimeError):
        if "decode" in text:
            return MSG_BAD_FORMAT
        return MSG_MODEL_ERROR

    if isinstance(exc, ValueError):
        if "empty" in text:
            return MSG_EMPTY_AUDIO
        # e.g. malformed/unreadable audio surfaced as ValueError
        return MSG_BAD_FORMAT

    return MSG_GENERIC


# ---------------------------------------------------------------------------
# Pre-inference audio guards
# ---------------------------------------------------------------------------


def check_duration(
    num_samples: int,
    sample_rate: int,
    max_duration_s: float = DEFAULT_MAX_DURATION_S,
) -> str | None:
    """Return :data:`MSG_TOO_LONG` if the clip exceeds the cap, else ``None``.

    Args:
        num_samples: Number of audio samples in the (mono) clip.
        sample_rate: Sample rate in Hz.
        max_duration_s: Maximum allowed duration in seconds.

    Returns:
        The rejection message, or ``None`` when the clip is within the cap.
    """
    if sample_rate <= 0:
        return MSG_BAD_FORMAT
    duration_s = num_samples / sample_rate
    if duration_s > max_duration_s:
        return MSG_TOO_LONG
    return None


def check_silence(
    audio_array: np.ndarray,
    rms_threshold: float = DEFAULT_RMS_THRESHOLD,
) -> str | None:
    """Return :data:`MSG_EMPTY_AUDIO` for empty/sub-threshold audio, else ``None``.

    Args:
        audio_array: Mono float waveform.
        rms_threshold: Minimum RMS to be treated as audible.

    Returns:
        The rejection message, or ``None`` when the clip is audible.
    """
    if audio_array.size == 0:
        return MSG_EMPTY_AUDIO
    rms = float(np.sqrt(np.mean(np.square(audio_array, dtype=np.float64))))
    if rms < rms_threshold:
        return MSG_EMPTY_AUDIO
    return None


def log_error(exc: BaseException, duration_s: float | None = None) -> None:
    """Log an aggregate-only record of a demo error.

    Records the exception class and (optionally) the clip duration. Never logs
    audio bytes or transcripts (data-sovereignty constraint).

    Args:
        exc: The caught exception.
        duration_s: Clip duration in seconds, if known.
    """
    if duration_s is None:
        logger.warning("demo transcription error: %s", type(exc).__name__)
    else:
        logger.warning(
            "demo transcription error: %s (clip %.1fs)",
            type(exc).__name__,
            duration_s,
        )
