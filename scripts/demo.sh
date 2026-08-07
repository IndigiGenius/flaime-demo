#!/usr/bin/env bash
# scripts/demo.sh — one-command flaime-demo launcher
#
# Builds the Apptainer image if absent, then launches the Streamlit UI.
# The build step is skipped automatically when the .sif already exists.
#
# One-time setup:
#   cp .env.example .env   # fill in CHECKPOINTS_DIR
#   bash scripts/demo.sh
#
# Usage:
#   bash scripts/demo.sh [OPTIONS]
#
# Options:
#   --rebuild             Force rebuild of the .sif image even if it exists
#   --port <port>         Streamlit port (default: 7860)
#   --bind <addr>         Bind address (default: 127.0.0.1)
#   --sif <path>          .sif image path (overrides FLAIME_SIF)
#   --checkpoints <dir>   Host checkpoints directory (overrides CHECKPOINTS_DIR)
#   -h, --help            Show this help
#
# Required env vars (or set in .env — loaded automatically):
#   CHECKPOINTS_DIR       absolute path to local checkpoints directory
#
# Optional env vars (also loadable from .env):
#   FLAIME_SIF            path to .sif image (default: flaime-demo.sif)
#   DEMO_CHECKPOINT_FILE  checkpoint filename inside CHECKPOINTS_DIR (resolved to
#                         /checkpoints/<file> in the container)
#   DEMO_CHECKPOINT       full in-container checkpoint path (overrides _FILE)
#   DEMO_LANGUAGES_CONFIG language-routing config path inside container
#   DEMO_MODEL_TYPE       model type (default: xeus)
#   DEMO_DECODER          decoder (default: ctc_greedy)
#   DEMO_PORT             Streamlit port (default: 7860)
#   DEMO_BIND             bind address (use 0.0.0.0 for local-network exposure)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"

# Load .env if present, but only for vars not already in the environment.
# Explicit env vars always take precedence over .env defaults.
if [[ -f "${ENV_FILE}" ]]; then
    while IFS='=' read -r key value || [[ -n "$key" ]]; do
        [[ "$key" =~ ^[[:space:]]*(#|$) ]] && continue
        [[ -z "$key" ]] && continue
        # shellcheck disable=SC2163
        [[ -z "${!key+x}" ]] && export "${key}=${value}"
    done < "${ENV_FILE}"
fi

SIF="${FLAIME_SIF:-${REPO_ROOT}/flaime-demo.sif}"
CHECKPOINTS="${CHECKPOINTS_DIR:-${REPO_ROOT}/checkpoints}"
PORT="${DEMO_PORT:-7860}"
BIND_ADDR="${DEMO_BIND:-127.0.0.1}"
REBUILD=0

usage() {
    sed -n 's/^# \{0,1\}//p' "$0"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rebuild)     REBUILD=1;              shift   ;;
        --sif)         SIF="$2";               shift 2 ;;
        --checkpoints) CHECKPOINTS="$2";       shift 2 ;;
        --port)        PORT="$2";              shift 2 ;;
        --bind)        BIND_ADDR="$2";         shift 2 ;;
        -h|--help)     usage; exit 0          ;;
        *) echo "Error: unknown option '$1'" >&2; exit 1 ;;
    esac
done

# ── Step 1: build ─────────────────────────────────────────────────────────────
if [[ ! -f "${SIF}" ]] || [[ "${REBUILD}" -eq 1 ]]; then
    echo "==> Building Apptainer image (~5-10 min on first run)…"
    apptainer build "${SIF}" "${REPO_ROOT}/flaime-demo.def"
    echo "==> Image built: ${SIF}"
else
    echo "==> Image already present — skipping build  (--rebuild to force)"
fi

# ── Step 2: validate checkpoints ──────────────────────────────────────────────
if [[ ! -d "${CHECKPOINTS}" ]]; then
    echo "Error: checkpoints directory not found: ${CHECKPOINTS}" >&2
    echo "  Set CHECKPOINTS_DIR in .env or pass --checkpoints <dir>." >&2
    exit 1
fi
echo "==> Checkpoints: ${CHECKPOINTS}"

# Resolve a bare checkpoint filename to its in-container path. CHECKPOINTS_DIR is
# mounted read-only at /checkpoints, so only the filename differs between host and
# container — the operator states the directory and filename once each, never the
# full path twice. An explicit DEMO_CHECKPOINT (full container path) takes priority.
if [[ -n "${DEMO_CHECKPOINT_FILE:-}" && -z "${DEMO_CHECKPOINT:-}" ]]; then
    if [[ ! -e "${CHECKPOINTS}/${DEMO_CHECKPOINT_FILE}" ]]; then
        echo "Error: checkpoint not found: ${CHECKPOINTS}/${DEMO_CHECKPOINT_FILE}" >&2
        echo "  Set DEMO_CHECKPOINT_FILE in .env to a file inside CHECKPOINTS_DIR." >&2
        exit 1
    fi
    DEMO_CHECKPOINT="/checkpoints/${DEMO_CHECKPOINT_FILE}"
    echo "==> Checkpoint:  ${DEMO_CHECKPOINT_FILE} → ${DEMO_CHECKPOINT}"
fi

# ── Step 3: run ───────────────────────────────────────────────────────────────
apptainer_run_args=(
    --bind "${CHECKPOINTS}:/checkpoints:ro"
    --bind "${REPO_ROOT}/configs:${REPO_ROOT}/configs:ro"
    --writable-tmpfs
    --env FLAIME_TELEMETRY=off
    --env "DEMO_CHECKPOINT=${DEMO_CHECKPOINT:-}"
    --env "DEMO_MODEL_TYPE=${DEMO_MODEL_TYPE:-xeus}"
    --env "DEMO_DECODER=${DEMO_DECODER:-ctc_greedy}"
)

# DEMO_LANGUAGES_CONFIG (Mode B) overrides DEMO_CHECKPOINT (Mode A) per the .env
# contract, so it must only be forwarded when the operator actually set it —
# never defaulted. A default here would silently force router mode even when
# DEMO_CHECKPOINT_FILE/.env selected single-checkpoint mode.
if [[ -n "${DEMO_LANGUAGES_CONFIG:-}" ]]; then
    apptainer_run_args+=(--env "DEMO_LANGUAGES_CONFIG=${DEMO_LANGUAGES_CONFIG}")
fi

if command -v nvidia-smi &>/dev/null; then
    apptainer_run_args=(--nv "${apptainer_run_args[@]}")
    echo "==> GPU detected — enabling --nv"
fi

echo "==> Launching flaime-demo at http://${BIND_ADDR}:${PORT}"
echo "    SIF:         ${SIF}"
echo "    Checkpoints: ${CHECKPOINTS} → /checkpoints (read-only)"

apptainer run \
    "${apptainer_run_args[@]}" \
    "${SIF}" \
    --server.port "${PORT}" \
    --server.address "${BIND_ADDR}"
