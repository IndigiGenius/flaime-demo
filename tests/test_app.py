"""Tests for the flaime-demo Streamlit app (flaime_demo/app.py).

Run: uv run pytest tests/test_app.py -v

These tests exercise the testable-without-Streamlit surface of the app:
  1. Module import does not launch a server
  2. _run_transcription() returns TranscriptionResult with the right shape
  3. DEFAULT_BIND is the loopback address
  4. Sovereignty/no-audio message constants exist and contain expected keywords

The Streamlit UI widgets (_run_app) are not tested here — use manual smoke
testing or streamlit.testing.v1.AppTest in a follow-up task.
"""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest
import torch
from flaime_serving import ASRInferenceEngine

# ---------------------------------------------------------------------------
# Stubs mirroring tests/serving/test_inference.py patterns
# ---------------------------------------------------------------------------

TARGET_SR = 16_000


def _make_wav_bytes(
    num_samples: int = TARGET_SR,
    sample_rate: int = TARGET_SR,
) -> bytes:
    """Return minimal valid WAV bytes (silence) without touching disk."""
    buf = io.BytesIO()
    samples = np.zeros(num_samples, dtype=np.int16)
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def _make_stereo_wav_bytes(
    num_samples: int = TARGET_SR,
    sample_rate: int = TARGET_SR,
) -> bytes:
    """Return minimal valid stereo WAV bytes (silence) without touching disk."""
    buf = io.BytesIO()
    # wave expects interleaved samples: L0 R0 L1 R1 …
    samples = np.zeros(num_samples * 2, dtype=np.int16)
    with wave.open(buf, "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


class _StubProcessor:
    """Minimal stand-in for a model processor."""

    def decode(self, token_ids: list[int], **kwargs: object) -> str:  # noqa: ARG002
        return "hello world"


class _StubModel:
    """Minimal stand-in for a loaded ASR model — no HuggingFace or GPU needed.

    Mirrors the interface expected by ASRInferenceEngine.transcribe():
      forward(input_tensor, wav_lengths=...) → {"logits": Tensor(1, T, vocab)}
      processor.decode(token_ids)            → str
      eval()                                 → self
      to(device)                             → self
    """

    revision = "test-stub-v0"

    def forward(
        self, input_features: torch.Tensor, **kwargs: object
    ) -> dict[str, torch.Tensor]:
        T = max(1, input_features.shape[-1] // 160)
        return {"logits": torch.full((1, T, 32), fill_value=-3.0)}

    @property
    def processor(self) -> _StubProcessor:
        return _StubProcessor()

    def eval(self) -> _StubModel:
        return self

    def to(self, device: str) -> _StubModel:  # noqa: ARG002
        return self


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def stub_engine() -> ASRInferenceEngine:
    return ASRInferenceEngine(
        model=_StubModel(),
        model_revision="test-stub-v0",
        decoder="ctc_greedy",
        device="cpu",
    )


@pytest.fixture()
def fixture_wav_bytes() -> bytes:
    """One second of silence as in-memory WAV bytes (16 kHz, mono, int16)."""
    return _make_wav_bytes()


# ---------------------------------------------------------------------------
# Test class 1 — module import
# ---------------------------------------------------------------------------


class TestDemoAppImport:
    """Module-level code must not start a server on import."""

    def test_imports_without_launching_server(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Importing flaime_demo.app must not call subprocess.run or start Streamlit."""
        import subprocess

        launched: list[object] = []
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: launched.append(a))

        import importlib

        mod = importlib.import_module("flaime_demo.app")
        assert mod is not None
        assert launched == [], "subprocess.run must not be called at import time"


# ---------------------------------------------------------------------------
# Test class 2 — _run_transcription helper
# ---------------------------------------------------------------------------


class TestTranscriptionHelper:
    """_run_transcription() produces a correctly shaped TranscriptionResult."""

    def test_returns_transcription_result_type(
        self, stub_engine: ASRInferenceEngine, fixture_wav_bytes: bytes
    ) -> None:
        from flaime_serving import TranscriptionResult

        from flaime_demo.app import _run_transcription

        result = _run_transcription(
            fixture_wav_bytes, language="en", engine=stub_engine
        )
        assert isinstance(result, TranscriptionResult)

    def test_text_field_is_str(
        self, stub_engine: ASRInferenceEngine, fixture_wav_bytes: bytes
    ) -> None:
        from flaime_demo.app import _run_transcription

        result = _run_transcription(
            fixture_wav_bytes, language="en", engine=stub_engine
        )
        assert isinstance(result.text, str)

    def test_latency_ms_is_non_negative_float(
        self, stub_engine: ASRInferenceEngine, fixture_wav_bytes: bytes
    ) -> None:
        from flaime_demo.app import _run_transcription

        result = _run_transcription(
            fixture_wav_bytes, language="en", engine=stub_engine
        )
        assert isinstance(result.latency_ms, float)
        assert result.latency_ms >= 0.0

    def test_model_revision_matches_stub(
        self, stub_engine: ASRInferenceEngine, fixture_wav_bytes: bytes
    ) -> None:
        from flaime_demo.app import _run_transcription

        result = _run_transcription(
            fixture_wav_bytes, language="en", engine=stub_engine
        )
        assert result.model_revision == "test-stub-v0"

    def test_decoder_matches_stub(
        self, stub_engine: ASRInferenceEngine, fixture_wav_bytes: bytes
    ) -> None:
        from flaime_demo.app import _run_transcription

        result = _run_transcription(
            fixture_wav_bytes, language="en", engine=stub_engine
        )
        assert result.decoder == "ctc_greedy"

    def test_language_passed_through(
        self, stub_engine: ASRInferenceEngine, fixture_wav_bytes: bytes
    ) -> None:
        from flaime_demo.app import _run_transcription

        result = _run_transcription(
            fixture_wav_bytes, language="en", engine=stub_engine
        )
        assert result.language == "en"

    def test_none_language_accepted(
        self, stub_engine: ASRInferenceEngine, fixture_wav_bytes: bytes
    ) -> None:
        from flaime_demo.app import _run_transcription

        result = _run_transcription(
            fixture_wav_bytes, language=None, engine=stub_engine
        )
        assert result.language is None

    def test_accepts_8khz_input(self, stub_engine: ASRInferenceEngine) -> None:
        from flaime_serving import TranscriptionResult

        from flaime_demo.app import _run_transcription

        wav_8k = _make_wav_bytes(sample_rate=8_000)
        result = _run_transcription(wav_8k, language=None, engine=stub_engine)
        assert isinstance(result, TranscriptionResult)

    def test_accepts_stereo_input(self, stub_engine: ASRInferenceEngine) -> None:
        from flaime_serving import TranscriptionResult

        from flaime_demo.app import _run_transcription

        result = _run_transcription(
            _make_stereo_wav_bytes(), language=None, engine=stub_engine
        )
        assert isinstance(result, TranscriptionResult)


# ---------------------------------------------------------------------------
# Test class 3 — default bind address
# ---------------------------------------------------------------------------


class TestDemoDefaults:
    """Verify loopback-only defaults to prevent accidental network exposure."""

    def test_default_bind_is_loopback(self) -> None:
        from flaime_demo.app import DEFAULT_BIND

        assert DEFAULT_BIND == "127.0.0.1"

    def test_default_port(self) -> None:
        from flaime_demo.app import DEFAULT_PORT

        assert DEFAULT_PORT == 8501

    def test_parse_args_default_bind(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        from flaime_demo.app import DEFAULT_BIND, _parse_args

        monkeypatch.setattr(sys, "argv", ["app.py"])
        args = _parse_args()
        assert args.bind == DEFAULT_BIND

    def test_parse_args_custom_bind(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        from flaime_demo.app import _parse_args

        monkeypatch.setattr(sys, "argv", ["app.py", "--bind", "0.0.0.0"])
        args = _parse_args()
        assert args.bind == "0.0.0.0"


# ---------------------------------------------------------------------------
# Test class 4 — UI message constants (sovereignty and no-audio banner)
# ---------------------------------------------------------------------------


class TestDemoBanners:
    """Verify required user-visible message strings exist with expected content."""

    def test_no_audio_message_exists(self) -> None:
        from flaime_demo.app import MSG_NO_AUDIO

        lower = MSG_NO_AUDIO.lower()
        assert "upload" in lower or "record" in lower, (
            "MSG_NO_AUDIO should reference uploading or recording audio"
        )

    def test_sovereignty_message_exists(self) -> None:
        from flaime_demo.app import MSG_SOVEREIGNTY

        lower = MSG_SOVEREIGNTY.lower()
        assert "local" in lower or "locally" in lower, (
            "MSG_SOVEREIGNTY should state that audio is processed locally"
        )

    def test_sovereignty_message_mentions_device(self) -> None:
        from flaime_demo.app import MSG_SOVEREIGNTY

        lower = MSG_SOVEREIGNTY.lower()
        assert "device" in lower or "machine" in lower or "local" in lower

    def test_public_bind_message_warns_network(self) -> None:
        from flaime_demo.app import MSG_PUBLIC_BIND

        lower = MSG_PUBLIC_BIND.lower()
        assert "network" in lower or "public" in lower or "reachable" in lower

    def test_no_model_message_references_checkpoint_flag(self) -> None:
        from flaime_demo.app import MSG_NO_MODEL

        assert "--checkpoint" in MSG_NO_MODEL


# ---------------------------------------------------------------------------
# Test class 5 — data sovereignty: no disk writes, no temp files
# ---------------------------------------------------------------------------


class TestAudioSovereignty:
    """Audio must never touch the filesystem during transcription.

    These tests catch regressions where a future change accidentally writes
    a temp file (e.g. passing a file path to soundfile or torchaudio instead
    of a BytesIO buffer).
    """

    def test_no_write_mode_open_during_transcription(
        self,
        stub_engine: ASRInferenceEngine,
        fixture_wav_bytes: bytes,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """builtins.open must not be called in any write mode during transcription."""
        import builtins

        from flaime_demo.app import _run_transcription

        write_calls: list[object] = []
        real_open = builtins.open

        def guarded_open(
            file: object, mode: str = "r", *args: object, **kwargs: object
        ) -> object:
            if any(m in mode for m in ("w", "x", "a", "+")):
                write_calls.append(file)
            return real_open(file, mode, *args, **kwargs)  # type: ignore[call-overload]

        monkeypatch.setattr(builtins, "open", guarded_open)
        _run_transcription(fixture_wav_bytes, language=None, engine=stub_engine)
        assert write_calls == [], f"open() called in write mode for: {write_calls}"

    def test_no_tempfile_during_transcription(
        self,
        stub_engine: ASRInferenceEngine,
        fixture_wav_bytes: bytes,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """tempfile must not be used during transcription."""
        import tempfile

        from flaime_demo.app import _run_transcription

        def guarded_named(*args: object, **kwargs: object) -> object:
            pytest.fail("tempfile.NamedTemporaryFile called during transcription")

        def guarded_mkstemp(*args: object, **kwargs: object) -> object:
            pytest.fail("tempfile.mkstemp called during transcription")

        monkeypatch.setattr(tempfile, "NamedTemporaryFile", guarded_named)
        monkeypatch.setattr(tempfile, "mkstemp", guarded_mkstemp)
        _run_transcription(fixture_wav_bytes, language=None, engine=stub_engine)


# ---------------------------------------------------------------------------
# Test class 6 — _bytes_to_audio() unit tests
# ---------------------------------------------------------------------------


class TestBytesToAudio:
    """Direct unit tests for _bytes_to_audio() covering dtype, shape, and error paths."""

    def test_returns_float32_array(self) -> None:
        from flaime_demo.app import _bytes_to_audio

        array, _ = _bytes_to_audio(_make_wav_bytes())
        assert array.dtype == np.float32

    def test_mono_wav_returns_1d_array(self) -> None:
        from flaime_demo.app import _bytes_to_audio

        array, _ = _bytes_to_audio(_make_wav_bytes())
        assert array.ndim == 1

    def test_stereo_wav_collapsed_to_1d(self) -> None:
        from flaime_demo.app import _bytes_to_audio

        array, _ = _bytes_to_audio(_make_stereo_wav_bytes())
        assert array.ndim == 1

    def test_preserves_sample_rate(self) -> None:
        from flaime_demo.app import _bytes_to_audio

        _, sr = _bytes_to_audio(_make_wav_bytes(sample_rate=8_000))
        assert sr == 8_000

    def test_length_matches_num_samples(self) -> None:
        from flaime_demo.app import _bytes_to_audio

        n = 4_000
        array, _ = _bytes_to_audio(
            _make_wav_bytes(num_samples=n, sample_rate=TARGET_SR)
        )
        assert len(array) == n

    def test_invalid_bytes_raises_runtime_error(self) -> None:
        from flaime_demo.app import _bytes_to_audio

        with pytest.raises(RuntimeError, match="Could not decode audio"):
            _bytes_to_audio(b"this is not audio")

    def test_empty_bytes_raises_runtime_error(self) -> None:
        from flaime_demo.app import _bytes_to_audio

        with pytest.raises(RuntimeError, match="Could not decode audio"):
            _bytes_to_audio(b"")
