#!/usr/bin/env bash
# tests/fixtures/community_eval/fetch_librispeech_samples.sh
#
# Regenerates the community-eval fixture audio (audio/*.wav) from LibriSpeech
# test-clean (CC BY 4.0; Panayotov et al. 2015 — see manifest.yaml's header).
#
# Fetches individual utterances via the Hugging Face `datasets-server` rows
# API for openslr/librispeech_asr (config "clean", split "test") rather than
# the canonical openslr.org/12 tar.gz (~346MB for the full split) — this pulls
# only the handful of files this fixture needs.
#
# Requires: curl, jq, ffmpeg.
#
# Usage: bash tests/fixtures/community_eval/fetch_librispeech_samples.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDIO_DIR="${SCRIPT_DIR}/audio"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT
mkdir -p "${AUDIO_DIR}"

# Rows 0-34 of test-clean cover both source chapters used below (6930-75918
# and 6930-76324). Re-fetched each run rather than cached, since the API
# returns short-lived signed audio URLs.
ROWS_URL="https://datasets-server.huggingface.co/rows?dataset=openslr%2Flibrispeech_asr&config=clean&split=test&offset=0&length=35"
curl -sL --max-time 30 "${ROWS_URL}" -o "${WORK_DIR}/rows.json"

# Keep in sync with manifest.yaml's "Filename -> source LibriSpeech utterance
# ID" table and its per-sample reference_text.
declare -A SAMPLES=(
  [golden_001]="6930-75918-0000"
  [golden_002]="6930-75918-0002"
  [golden_003]="6930-75918-0007"
  [golden_004]="6930-75918-0010"
  [golden_005]="6930-76324-0001"
  [edge_001]="6930-75918-0008"
  [edge_002]="6930-76324-0000"
  [edge_003]="6930-76324-0007"
)

for name in "${!SAMPLES[@]}"; do
  utt_id="${SAMPLES[${name}]}"
  url=$(jq -r --arg id "${utt_id}" \
    '.rows[] | select(.row.id == $id) | .row.audio[0].src' "${WORK_DIR}/rows.json")
  if [ -z "${url}" ] || [ "${url}" = "null" ]; then
    echo "FATAL: utterance ${utt_id} not found in the fetched rows window" >&2
    exit 1
  fi
  curl -sL --max-time 30 "${url}" -o "${WORK_DIR}/${name}.flac"
  ffmpeg -y -loglevel error -i "${WORK_DIR}/${name}.flac" -ar 16000 -ac 1 -acodec pcm_s16le \
    "${AUDIO_DIR}/${name}.wav"
  echo "==> ${AUDIO_DIR}/${name}.wav (${utt_id})"
done
