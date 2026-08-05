"""Language display-name table for the demo's language selector.

Demo-owned copy of FLAIME's ``LANGUAGES_32`` (upstream:
``flaime/configs/languages.py`` at commit 0978bb5d). Transcribed verbatim,
batch comments included, so a reviewer can diff it against the original.

This is presentation config, not serving logic — the same category call
26Q3-REPO-11 makes for the routing YAMLs, which move here because they are
demo *instances* of the router's config schema. It lives in flaime-demo rather
than flaime-serving because ``flaime_serving``'s public API is frozen at eight
inference names (26Q3-REPO-04) and a display-name table is the wrong category
to reopen it for, and because this repo's contract is that FLAIME is absent
from the demo environment entirely (26Q3-REPO epic boundary).

Only ``LANGUAGES_32`` is copied — the upstream module's ``LANGUAGES_8``,
``LANGUAGES_16`` and ``LANGUAGE_STATS`` are training-side config the demo does
not use.
"""

from __future__ import annotations

# 32 languages (comprehensive multilingual)
# Covers 10 writing systems and 15+ language families
LANGUAGES_32: dict[str, str] = {
    # Original 8 (European, Latin script)
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    # Batch 2: Diverse scripts & families
    "ru": "Russian",
    "zh-CN": "Chinese",
    "ja": "Japanese",
    "ar": "Arabic",
    "tr": "Turkish",
    "sv-SE": "Swedish",
    "cs": "Czech",
    "uk": "Ukrainian",
    # Batch 3: Maximum diversity
    "ko": "Korean",
    "id": "Indonesian",
    "vi": "Vietnamese",
    "hi": "Hindi",
    "el": "Greek",
    "hu": "Hungarian",
    "ro": "Romanian",
    "fi": "Finnish",
    "da": "Danish",
    "ca": "Catalan",
    "eu": "Basque",
    "cy": "Welsh",
    "fa": "Persian",
    "th": "Thai",
    "ta": "Tamil",
    "sw": "Swahili",
}
