#!/usr/bin/env bash
# scripts/demo.sh — self-bootstrapping flaime-demo launcher
#
# Creates .env from .env.example on first run, builds the Apptainer image if
# absent, then launches the Streamlit UI. The build step is skipped
# automatically when the .sif already exists.
#
# One-time setup:
#   bash scripts/demo.sh          # creates .env on first run
#   $EDITOR .env                  # set CHECKPOINTS_DIR, then re-run
#   ./scripts/fetch_checkpoints.sh --dest ./checkpoints   # if not staged yet
#
# Usage:
#   bash scripts/demo.sh [OPTIONS]
#
# Launch mode is picked automatically: Apptainer when a .sif already exists
# (the partner-laptop path — a prebuilt image, no repo checkout needed) or
# --apptainer/--rebuild is passed; bare metal (uv sync + uv run) otherwise —
# the default for a fresh clone on a dev laptop, no Apptainer install required.
#
# Options:
#   --apptainer           Force Apptainer mode even without an existing .sif
#   --rebuild             Force rebuild of the .sif image (implies --apptainer)
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
ENV_EXAMPLE_FILE="${REPO_ROOT}/.env.example"

# Bootstrap: create .env from the template on first run so a fresh clone never
# hard-fails for lack of one. Never overwrites an existing .env.
if [[ ! -f "${ENV_FILE}" ]] && [[ -f "${ENV_EXAMPLE_FILE}" ]]; then
    cp "${ENV_EXAMPLE_FILE}" "${ENV_FILE}"
    echo "==> Created .env from .env.example — set CHECKPOINTS_DIR before staging checkpoints"
fi

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
FORCE_APPTAINER=0

usage() {
    sed -n 's/^# \{0,1\}//p' "$0"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --apptainer)   FORCE_APPTAINER=1;      shift   ;;
        --rebuild)     REBUILD=1;              shift   ;;
        --sif)         : "${2:?Error: --sif requires a path}"; SIF="$2";               shift 2 ;;
        --checkpoints) : "${2:?Error: --checkpoints requires a directory}"; CHECKPOINTS="$2";       shift 2 ;;
        --port)        : "${2:?Error: --port requires a value}"; PORT="$2";              shift 2 ;;
        --bind)        : "${2:?Error: --bind requires an address}"; BIND_ADDR="$2";         shift 2 ;;
        -h|--help)     usage; exit 0          ;;
        *) echo "Error: unknown option '$1'" >&2; exit 1 ;;
    esac
done

# ── Step 1: validate checkpoints ──────────────────────────────────────────────
# Checked before the (slow) build step so a fresh clone fails fast with an
# actionable command instead of waiting 5-10 min to be told the same thing.
# demo.sh never stages checkpoints itself — fetching may need credentials
# (HF_TOKEN, etc.) the launcher shouldn't assume.
if [[ ! -d "${CHECKPOINTS}" ]]; then
    echo "Error: no checkpoints staged at ${CHECKPOINTS}" >&2
    echo "  Stage them first, then re-run:" >&2
    echo "    ./scripts/fetch_checkpoints.sh --dest ./checkpoints" >&2
    exit 1
fi
echo "==> Checkpoints: ${CHECKPOINTS}"

# ── Step 2: pick a launch mode ────────────────────────────────────────────────
# Apptainer when a .sif already exists (partner laptop: prebuilt image, no repo
# checkout needed) or the operator asked for it explicitly; bare metal
# otherwise — the fresh-clone default, no Apptainer install required.
if [[ "${FORCE_APPTAINER}" -eq 1 ]] || [[ "${REBUILD}" -eq 1 ]] || [[ -f "${SIF}" ]]; then
    MODE="apptainer"
else
    MODE="bare-metal"
fi

if [[ "${MODE}" == "apptainer" ]]; then
    # ── Apptainer: build (if absent) → run ──────────────────────────────────
    if [[ ! -f "${SIF}" ]] || [[ "${REBUILD}" -eq 1 ]]; then
        echo "==> Building Apptainer image (~5-10 min on first run)…"
        apptainer build "${SIF}" "${REPO_ROOT}/flaime-demo.def"
        echo "==> Image built: ${SIF}"
    else
        echo "==> Image already present — skipping build  (--rebuild to force)"
    fi

    # Resolve a bare checkpoint filename to its in-container path. CHECKPOINTS_DIR
    # is mounted read-only at /checkpoints, so only the filename differs between
    # host and container — the operator states the directory and filename once
    # each, never the full path twice. An explicit DEMO_CHECKPOINT (full
    # container path) takes priority.
    if [[ -z "${DEMO_LANGUAGES_CONFIG:-}" && -n "${DEMO_CHECKPOINT_FILE:-}" && -z "${DEMO_CHECKPOINT:-}" ]]; then
        if [[ ! -e "${CHECKPOINTS}/${DEMO_CHECKPOINT_FILE}" ]]; then
            echo "Error: checkpoint not found: ${CHECKPOINTS}/${DEMO_CHECKPOINT_FILE}" >&2
            echo "  Set DEMO_CHECKPOINT_FILE in .env to a file inside CHECKPOINTS_DIR." >&2
            exit 1
        fi
        DEMO_CHECKPOINT="/checkpoints/${DEMO_CHECKPOINT_FILE}"
        echo "==> Checkpoint:  ${DEMO_CHECKPOINT_FILE} → ${DEMO_CHECKPOINT}"
    fi

    apptainer_run_args=(
        --bind "${CHECKPOINTS}:/checkpoints:ro"
        --bind "${REPO_ROOT}/configs:${REPO_ROOT}/configs:ro"
        --writable-tmpfs
        --env FLAIME_TELEMETRY=off
        --env "DEMO_CHECKPOINT=${DEMO_CHECKPOINT:-}"
        --env "DEMO_MODEL_TYPE=${DEMO_MODEL_TYPE:-xeus}"
        --env "DEMO_DECODER=${DEMO_DECODER:-ctc_greedy}"
    )

    # DEMO_LANGUAGES_CONFIG (Mode B) overrides DEMO_CHECKPOINT (Mode A) per the
    # .env contract, so it must only be forwarded when the operator actually set
    # it — never defaulted. A default here would silently force router mode even
    # when DEMO_CHECKPOINT_FILE/.env selected single-checkpoint mode.
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
else
    # ── Bare metal: uv sync (if stale) → uv run ─────────────────────────────
    # Our own stamp, not anything `uv` manages: a no-op `uv sync` (nothing to
    # install) doesn't reliably bump any file inside .venv, so staleness can't
    # be inferred from .venv's own contents — only from whether *we* last
    # synced after uv.lock's current mtime.
    UV_SYNC_STAMP="${REPO_ROOT}/.venv/.uv-sync-stamp"
    if [[ ! -d "${REPO_ROOT}/.venv" ]] || [[ "${REPO_ROOT}/uv.lock" -nt "${UV_SYNC_STAMP}" ]]; then
        echo "==> Syncing dependencies (uv sync)…"
        ( cd "${REPO_ROOT}" && uv sync )
        mkdir -p "${REPO_ROOT}/.venv"
        touch "${UV_SYNC_STAMP}"
    else
        echo "==> Dependencies already synced — skipping uv sync"
    fi

    # Host-path equivalent of the Apptainer resolution above — no /checkpoints
    # container mount here, so the filename resolves directly under CHECKPOINTS.
    if [[ -z "${DEMO_LANGUAGES_CONFIG:-}" && -n "${DEMO_CHECKPOINT_FILE:-}" && -z "${DEMO_CHECKPOINT:-}" ]]; then
        if [[ ! -e "${CHECKPOINTS}/${DEMO_CHECKPOINT_FILE}" ]]; then
            echo "Error: checkpoint not found: ${CHECKPOINTS}/${DEMO_CHECKPOINT_FILE}" >&2
            echo "  Set DEMO_CHECKPOINT_FILE in .env to a file inside CHECKPOINTS_DIR." >&2
            exit 1
        fi
        DEMO_CHECKPOINT="${CHECKPOINTS}/${DEMO_CHECKPOINT_FILE}"
        echo "==> Checkpoint:  ${DEMO_CHECKPOINT_FILE} → ${DEMO_CHECKPOINT}"
    fi

    app_args=(--bind "${BIND_ADDR}" --port "${PORT}")
    if [[ -n "${DEMO_LANGUAGES_CONFIG:-}" ]]; then
        app_args+=(--languages-config "${DEMO_LANGUAGES_CONFIG}")
    elif [[ -n "${DEMO_CHECKPOINT:-}" ]]; then
        app_args+=(--checkpoint "${DEMO_CHECKPOINT}")
    fi
    app_args+=(--model-type "${DEMO_MODEL_TYPE:-xeus}" --decoder "${DEMO_DECODER:-ctc_greedy}")

    echo "==> Launching flaime-demo at http://${BIND_ADDR}:${PORT}"
    echo "    Checkpoints: ${CHECKPOINTS}"

    ( cd "${REPO_ROOT}" && exec uv run python flaime_demo/app.py "${app_args[@]}" )
fi
