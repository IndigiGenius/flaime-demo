"""WER/CER helpers, carried verbatim from FLAIME's ``flaime.utils.training_utils``
(26Q3-REPO-12).

Only ``compute_wer`` and ``compute_cer`` are ported — the rest of that module
imports torch/torchaudio for training-time batch decoding, which the demo has
no use for and does not depend on. Keeping this module jiwer-only means
``scripts/run_community_eval.py`` can import it without pulling in any heavy
deps.
"""

from __future__ import annotations

import jiwer


def compute_wer(predictions: list[str], references: list[str]) -> float:
    """Compute Word Error Rate (WER).

    Args:
        predictions: List of predicted transcriptions
        references: List of reference transcriptions

    Returns:
        WER as a percentage (0-100)
    """
    wer = jiwer.wer(references, predictions)
    return wer * 100  # Convert to percentage


def compute_cer(predictions: list[str], references: list[str]) -> float:
    """Compute Character Error Rate (CER).

    For scripts without whitespace word boundaries (Chinese, Japanese, Thai,
    Cantonese) WER goes to ~100% on a single character delta because there
    is one "word" per utterance. CER is the standard reporting metric for
    those.

    Args:
        predictions: List of predicted transcriptions
        references: List of reference transcriptions

    Returns:
        CER as a percentage (0-100). Returns 0.0 when both ref and hyp are
        entirely empty (mirrors compute_per's empty-reference contract).
    """
    if not any(r.strip() for r in references):
        return 0.0 if not any(p.strip() for p in predictions) else 100.0
    cer = jiwer.cer(references, predictions)
    return cer * 100
