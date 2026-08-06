#!/usr/bin/env bash
# scripts/fetch_checkpoints.sh — download FLAIME demo checkpoints
#
# Supports two source types:
#   hf://org/repo   — HF Hub model repo (uses huggingface-cli download)
#   https://...     — HTTPS archive URL (curl + sha256 verification)
#
# Idempotency:
#   HTTPS: skips download when the archive exists and sha256 matches.
#          Aborts on checksum mismatch unless --force is passed.
#   HF:    skips download when a .complete sentinel exists in DEST.
#          --force removes the sentinel and re-downloads.
#
# Usage:
#   CHECKPOINT_SOURCE=hf://IndigiGenius/flaime-demo \
#     ./scripts/fetch_checkpoints.sh --dest ./checkpoints
#
#   ./scripts/fetch_checkpoints.sh \
#     --source https://example.com/flaime-demo.tar.gz \
#     --dest ./checkpoints \
#     --checksum <sha256hex>
#
# Environment variables (alternatives to flags):
#   CHECKPOINT_SOURCE   source URL (hf://... or https://...)
#   CHECKPOINT_DEST     destination directory
#   CHECKPOINT_CHECKSUM expected SHA-256 hex digest (HTTPS mode only)

set -euo pipefail

# ── defaults (overridden by flags below) ──────────────────────────────────────
SOURCE="${CHECKPOINT_SOURCE:-}"
DEST_DIR="${CHECKPOINT_DEST:-}"
EXPECTED_CHECKSUM="${CHECKPOINT_CHECKSUM:-}"
FORCE=0

# ── usage ─────────────────────────────────────────────────────────────────────
usage() {
    cat >&2 <<'USAGE'
Usage: fetch_checkpoints.sh [OPTIONS]

Download FLAIME demo checkpoints from HF Hub or an HTTPS archive.

Options:
  --source   <hf://org/repo | https://...>  Checkpoint source (or CHECKPOINT_SOURCE env)
  --dest     <dir>                          Destination directory (or CHECKPOINT_DEST env)
  --checksum <sha256hex>                    Expected SHA-256 (HTTPS mode; or CHECKPOINT_CHECKSUM env)
  --force                                   Overwrite existing downloads even if valid
  -h, --help                                Show this help

Examples:
  # HF Hub (requires HF_TOKEN for private repos)
  CHECKPOINT_SOURCE=hf://IndigiGenius/flaime-demo \
    fetch_checkpoints.sh --dest ./checkpoints

  # HTTPS with checksum verification
  fetch_checkpoints.sh \
    --source https://example.com/flaime-demo.tar.gz \
    --dest ./checkpoints \
    --checksum a3f2...

  # Overwrite a stale download
  fetch_checkpoints.sh --source hf://... --dest ./checkpoints --force

TODO: set CHECKPOINT_SOURCE to the July demo checkpoint location before running.
USAGE
}

# ── argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)   SOURCE="$2";            shift 2 ;;
        --dest)     DEST_DIR="$2";          shift 2 ;;
        --checksum) EXPECTED_CHECKSUM="$2"; shift 2 ;;
        --force)    FORCE=1;                shift   ;;
        -h|--help)  usage; exit 0          ;;
        *) echo "Error: unknown option '$1'" >&2; usage; exit 1 ;;
    esac
done

# ── validation ────────────────────────────────────────────────────────────────
if [[ -z "${SOURCE}" ]]; then
    echo "Error: checkpoint source not set." >&2
    echo "  Set CHECKPOINT_SOURCE=<hf://org/repo | https://...> or use --source." >&2
    echo "  See scripts/fetch_checkpoints.sh --help for examples." >&2
    exit 1
fi

if [[ -z "${DEST_DIR}" ]]; then
    echo "Error: --dest is required (or set CHECKPOINT_DEST)." >&2
    usage
    exit 1
fi

# ── helpers ───────────────────────────────────────────────────────────────────

# sha256 of a file, portable across Linux (sha256sum) and macOS (shasum -a 256)
file_sha256() {
    local file="$1"
    if command -v sha256sum &>/dev/null; then
        sha256sum "${file}" | cut -d' ' -f1
    elif command -v shasum &>/dev/null; then
        shasum -a 256 "${file}" | cut -d' ' -f1
    else
        echo "Error: neither sha256sum nor shasum found." >&2
        exit 1
    fi
}

# ── HF Hub mode ───────────────────────────────────────────────────────────────
fetch_hf() {
    local repo="${SOURCE#hf://}"
    local sentinel="${DEST_DIR}/.complete"

    if [[ -f "${sentinel}" ]] && [[ "${FORCE}" -eq 0 ]]; then
        echo "Checkpoints already present (${sentinel} exists) — nothing to do."
        echo "  Use --force to re-download."
        exit 0
    fi

    if [[ "${FORCE}" -eq 1 ]] && [[ -f "${sentinel}" ]]; then
        echo "Warning: --force passed; removing sentinel and re-downloading." >&2
        rm -f "${sentinel}"
    fi

    if ! command -v huggingface-cli &>/dev/null; then
        echo "Error: huggingface-cli not found." >&2
        echo "  Install with: uv pip install huggingface_hub[cli]" >&2
        exit 1
    fi

    mkdir -p "${DEST_DIR}"
    echo "Downloading from HF Hub: ${repo} → ${DEST_DIR}"
    # HF Hub verifies each file's integrity natively (etag + sha256 on the server side).
    huggingface-cli download "${repo}" --local-dir "${DEST_DIR}"
    touch "${sentinel}"
    echo "Done. Checkpoints in: ${DEST_DIR}"
}

# ── HTTPS archive mode ────────────────────────────────────────────────────────
fetch_https() {
    local archive="${DEST_DIR}/checkpoint.tar.gz"

    if [[ -z "${EXPECTED_CHECKSUM}" ]]; then
        echo "Error: --checksum is required for HTTPS sources." >&2
        usage
        exit 1
    fi

    mkdir -p "${DEST_DIR}"

    # Idempotency check: skip if the archive already exists with the right checksum.
    if [[ -f "${archive}" ]]; then
        local actual
        actual="$(file_sha256 "${archive}")"
        if [[ "${actual}" == "${EXPECTED_CHECKSUM}" ]]; then
            echo "Archive already present and checksum matches — nothing to do."
            echo "  Use --force to re-download."
            exit 0
        fi
        # Checksum mismatch on an existing file: could be corruption or a stale version.
        if [[ "${FORCE}" -eq 0 ]]; then
            echo "Error: ${archive} exists but checksum does not match." >&2
            echo "  Expected: ${EXPECTED_CHECKSUM}" >&2
            echo "  Actual:   ${actual}" >&2
            echo "  Use --force to overwrite." >&2
            exit 1
        fi
        echo "Warning: checksum mismatch on existing archive; overwriting (--force)." >&2
        rm -f "${archive}"
    fi

    if ! command -v curl &>/dev/null; then
        echo "Error: curl not found." >&2
        exit 1
    fi

    echo "Downloading: ${SOURCE} → ${archive}"
    curl -fL --progress-bar "${SOURCE}" -o "${archive}"

    # Verify checksum before extracting — abort and clean up on mismatch.
    local actual
    actual="$(file_sha256 "${archive}")"
    if [[ "${actual}" != "${EXPECTED_CHECKSUM}" ]]; then
        echo "Error: checksum mismatch after download — archive may be corrupted." >&2
        echo "  Expected: ${EXPECTED_CHECKSUM}" >&2
        echo "  Actual:   ${actual}" >&2
        rm -f "${archive}"
        exit 1
    fi

    echo "Checksum verified. Extracting…"
    tar -xzf "${archive}" -C "${DEST_DIR}" --strip-components=1
    echo "Done. Checkpoints in: ${DEST_DIR}"
}

# ── dispatch ──────────────────────────────────────────────────────────────────
if [[ "${SOURCE}" == hf://* ]]; then
    fetch_hf
else
    fetch_https
fi
