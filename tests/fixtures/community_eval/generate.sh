#!/usr/bin/env bash
# tests/fixtures/community_eval/generate.sh
#
# Regenerates the synthetic community-eval fixture audio (audio/*.wav) from
# the reference_text strings in manifest.yaml, via espeak-ng + ffmpeg.
#
# This is a spike fixture, not the real community-representative sample set
# (see manifest.yaml's header comment) — synthetic speech, no real speaker,
# so the output WAVs are safe to commit directly.
#
# Requires: espeak-ng, ffmpeg (both already runtime deps of the FLAIME/demo
# Apptainer images; on a bare host, `apt-get install espeak-ng ffmpeg`).
#
# Usage: bash tests/fixtures/community_eval/generate.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDIO_DIR="${SCRIPT_DIR}/audio"
mkdir -p "${AUDIO_DIR}"

gen() {
    local id="$1" text="$2"
    local raw
    raw="$(mktemp --suffix=.wav)"
    espeak-ng -v en -s 150 -w "${raw}" "${text}"
    ffmpeg -y -loglevel error -i "${raw}" -ar 16000 -ac 1 -acodec pcm_s16le \
        "${AUDIO_DIR}/${id}.wav"
    rm -f "${raw}"
    echo "==> ${AUDIO_DIR}/${id}.wav"
}

# Keep in sync with manifest.yaml's reference_text per audio_path.
gen golden_001 "hello world this is a test"
gen golden_002 "the quick brown fox jumps over the lazy dog"
gen golden_003 "please transcribe this audio clip accurately"
gen golden_004 "the weather today is sunny with a light breeze"
gen edge_001 "call me at five five five one two three four"
gen edge_002 "my name is jordan and i live in ottawa"
