# flaime-demo

Offline Streamlit demo for FLAIME ASR — runs fully local on a laptop; audio never
leaves the device (data sovereignty). Supports mic recording and file upload across
all BTM-trained languages. Depends on `flaime-serving` only; FLAIME is absent from
this environment entirely.

> **Presenting this at a gathering?** This README is the install/deploy source of
> truth. For run-of-show, talking points, FAQ, the failure playbook, and the backup
> plan, see the [Presenter Guide](docs/PRESENTER_GUIDE.md).

## Status

Scaffolded by `26Q3-REPO-08`. The UI (`app.py`) landed in `26Q3-REPO-09`; routing
configs arrived in `26Q3-REPO-11`; the offline end-to-end smoke in `26Q3-REPO-12` —
the MVP finish line. Task cards, architecture decision, and scope guardrails live in
the FLAIME repo: `docs/planning/tasks/26Q3-REPO/26Q3-REPO-00-overview.md`
(epic issue [FLAIME#608](https://github.com/IndigiGenius/FLAIME/issues/608),
cards PR [FLAIME#609](https://github.com/IndigiGenius/FLAIME/pull/609)).

---

## Quickstart (dev laptop)

```bash
git clone git@github.com:IndigiGenius/flaime-demo.git && cd flaime-demo
./scripts/fetch_checkpoints.sh --dest ./checkpoints   # first run only: stage checkpoints
./scripts/demo.sh                                     # sets up if needed, then launches
```

Two commands (three on first run, for checkpoint staging). `demo.sh` is
self-bootstrapping: it creates `.env` from `.env.example`, runs `uv sync` only when
the venv is missing or `uv.lock` is newer than it (so a warm re-run makes zero
network calls), and launches `flaime_demo/app.py` directly — no Apptainer install
required. If `CHECKPOINTS_DIR` isn't set yet, it prints the exact
`fetch_checkpoints.sh` command above instead of guessing.

This is [Mode 2](#mode-2--bare-metal-uv-run-python-flaime_demoapppy) below, driven
automatically. Partner laptops with a prebuilt `.sif` and no repo access keep using
[Mode 1](#mode-1--apptainer-linux--slurm) — `demo.sh` detects the `.sif` and switches
to the Apptainer path unchanged.

---

## Prerequisites

| Mode | Requires |
|------|----------|
| Apptainer (Linux / SLURM) | Apptainer ≥ 1.0 |
| Bare metal | `uv` ≥ 0.4, Python 3.12 |

---

## Step 0 — Fetch checkpoints (all modes)

Checkpoints are not included in the repo or image. Run this once to populate the
`checkpoints/` directory before launching.

```bash
# HF Hub (private repo — set HF_TOKEN first)
export HF_TOKEN=<your-hf-token>
export CHECKPOINT_SOURCE=hf://IndigiGenius/flaime-demo
bash scripts/fetch_checkpoints.sh --dest ./checkpoints

# HTTPS with checksum verification
bash scripts/fetch_checkpoints.sh \
    --source https://example.com/flaime-demo.tar.gz \
    --dest ./checkpoints \
    --checksum <sha256hex>
```

The script is idempotent — re-running it skips the download if the checksum matches
(HTTPS) or a completion sentinel already exists (HF Hub). Use `--force` to re-download.

---

## Mode 1 — Apptainer (Linux / SLURM)

Apptainer runs without a root daemon and integrates natively with SLURM's GPU
scheduling via `--nv`. Use this mode on HPC clusters and Linux workstations.

### One-command path

```bash
# 1. Copy the env template and fill in your values (one-time setup)
cp .env.example .env
$EDITOR .env   # set CHECKPOINTS_DIR

# 2. Build → launch (build is skipped automatically if the .sif already exists)
bash scripts/demo.sh

# Open http://127.0.0.1:7860
```

`demo.sh` loads `.env` automatically, so after the initial setup every subsequent
launch is just `bash scripts/demo.sh`.

**Flags:**

| Flag | Effect |
|------|--------|
| `--rebuild` | Force rebuild of `flaime-demo.sif` even if it exists |
| `--sif <path>` | Override the .sif image path |
| `--checkpoints <dir>` | Override host checkpoints directory |
| `--port <N>` | Override Streamlit port (default 7860) |
| `--bind <addr>` | Override bind address (default 127.0.0.1) |

### Manual steps (advanced)

<details>
<summary>Build, fetch, and run separately</summary>

**Build the image** (once, or when deps change; run from repo root):

```bash
apptainer build flaime-demo.sif flaime-demo.def
```

Build time is ~5–10 min on first run. The `.sif` can be distributed to partner
laptops or SLURM nodes without repeating the build.

**Run** (after checkpoints are fetched via Step 0):

```bash
# Single checkpoint (filename inside CHECKPOINTS_DIR, resolved by demo.sh)
DEMO_CHECKPOINT_FILE=xeus-demo.pt \
  bash scripts/demo.sh

# Full language routing
DEMO_LANGUAGES_CONFIG=/app/configs/demo_languages.yaml \
  bash scripts/demo.sh
```

GPU is auto-detected — if `nvidia-smi` is present, `--nv` is added automatically.

**SLURM sbatch snippet:**

```bash
#!/bin/bash
#SBATCH --gres=gpu:1
apptainer run --nv \
    --bind ./checkpoints:/checkpoints:ro \
    --bind ./configs:./configs:ro \
    --writable-tmpfs \
    --env FLAIME_TELEMETRY=off \
    --env DEMO_CHECKPOINT=/checkpoints/xeus-demo.pt \
    flaime-demo.sif \
    --server.port 7860 --server.address 127.0.0.1
```

</details>

The `checkpoints/` directory is bound read-only — the container cannot modify or
exfiltrate weight files. `--writable-tmpfs` provides a scratch `/tmp` for the HF
model cache without writing to the host filesystem.

---

## Mode 2 — Bare metal (`uv run python flaime_demo/app.py`)

`./scripts/demo.sh` drives this mode automatically (see Quickstart above) whenever
no `flaime-demo.sif` is present — `uv sync` (if needed) then a launch, reading the
same `.env` contract as Mode 1. The commands below are the manual, one-step-at-a-time
equivalent — useful for passing ad hoc flags or debugging a launch failure.

```bash
# Install deps (single dependency graph — no extras group here, unlike FLAIME's
# old apps/demo/ which needed --extra demo --extra asr)
uv sync

# No checkpoint (UI starts but transcription is disabled)
uv run python flaime_demo/app.py

# With a single checkpoint
uv run python flaime_demo/app.py \
    --checkpoint /path/to/checkpoint \
    --model-type xeus \
    --decoder ctc_greedy

# With full language routing
uv run python flaime_demo/app.py \
    --languages-config configs/demo_languages.yaml

# Expose on local network (conference setup — shows warning banner in UI)
uv run python flaime_demo/app.py --bind 0.0.0.0 --port 8501
```

Opens at `http://127.0.0.1:8501` by default. Self-bootstrapping: running
`python flaime_demo/app.py` directly re-launches itself under `streamlit run`
(see the `__main__` block) — there is no separate `streamlit run` step to remember.

### Bare-metal CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | *(none)* | Path to a FLAIME checkpoint directory |
| `--languages-config` | *(none)* | YAML language-routing config; overrides `--checkpoint` |
| `--model-type` | `xeus` | Architecture key (`xeus`, `whisper`, …) |
| `--device` | *(auto)* | Torch device (`cpu`, `cuda`, `cuda:0`) |
| `--decoder` | `ctc_greedy` | Decoding strategy (`ctc_greedy` or `ctc_beam<N>`) |
| `--bind` | `127.0.0.1` | Address to bind; `0.0.0.0` exposes on the network |
| `--port` | `8501` | Streamlit port |

---

## Checkpoint contract

**Model weights are never committed to this repo.** They live on the host and are
mounted read-only. Copy `.env.example` to `.env` and set:

| Variable | Meaning |
|----------|---------|
| `CHECKPOINTS_DIR` | Absolute host path holding the checkpoints. Mounted read-only at `/checkpoints` in the container. **Required.** |
| `DEMO_CHECKPOINT_FILE` | Single-model mode — the checkpoint *filename* inside `CHECKPOINTS_DIR`. |
| `DEMO_LANGUAGES_CONFIG` | Language-routing mode — path to a routing YAML. Overrides `DEMO_CHECKPOINT_FILE`. |

Set exactly one of `DEMO_CHECKPOINT_FILE` or `DEMO_LANGUAGES_CONFIG`. `.env` is
gitignored; never commit it.

---

## Offline requirement

The demo must run with no network access — that is the point of the extraction, and
`26Q3-REPO-12`'s smoke test enforces it. Practically:

- Dependencies resolve from the committed `uv.lock` (`uv sync --frozen`).
- `flaime-serving` is pinned to an immutable git rev, not a branch.
- Checkpoints are read from `CHECKPOINTS_DIR`. Nothing in the UI downloads or
  uploads a model — the operator supplies the weights.
- Streamlit telemetry is disabled in the container environment.

> ⚠️ **At the currently pinned `flaime-serving` rev, offline is a configuration
> property rather than an enforced one.** That loader treats any non-absolute
> value containing exactly one `/` as a HuggingFace Hub ID — it skips the
> local-file check and lets `from_pretrained` download at runtime. So a
> routing-YAML entry of `facebook/wav2vec2-base` would fetch over the network,
> while `./checkpoints/model.pt` would not. No UI path reaches this; only config
> does. `flaime-serving`'s `26Q3-SERVE-01` fixes it upstream — `load()` now
> refuses Hub-shaped values unless `allow_remote=True` — and this repo inherits
> the guarantee when its pin advances past that change. Until then, offline
> depends on the routing YAML being written correctly, and `26Q3-REPO-12`'s
> offline smoke should assert that no configured checkpoint is Hub-shaped.

---

## Data sovereignty and security

The demo enforces Indigenous data sovereignty **by construction**, not policy:

1. **No audio leaves the device.** All inference runs locally via `flaime-serving`'s
   `ASRInferenceEngine`. The app has no network code path for audio data.

2. **No disk writes for audio.** Uploaded and recorded bytes are decoded in RAM
   (`io.BytesIO`) and discarded after each transcription.

3. **No telemetry.** Streamlit's built-in usage stats are disabled at launch (see
   Offline requirement above). Apptainer containers add `--writable-tmpfs` for
   `/tmp` only — the host filesystem is not written to. `FLAIME_TELEMETRY=off` is
   hardcoded in both the image (`%environment`) and the launch script (`--env`).

4. **No phone-home.** The container has no outbound network calls during normal
   operation. Wi-Fi can be disabled on stage to demonstrate this.

Any community-partner audio included in a demo bundle requires **explicit written
approval** from the partner before any gathering. The default bundle ships with no
partner audio.

---

## Error handling

The demo is driven live in front of an audience, so every failure path surfaces a
calm, human-readable message instead of a Python stack trace or a frozen UI. The
mapping lives in `flaime_demo/errors.py` (pure, Streamlit-free) and the UI renders
whatever it returns via `st.error`.

| Input / failure | What the user sees |
|-----------------|--------------------|
| Silent or all-zero clip (RMS below threshold) | "That clip sounds silent. Record or upload audio with speech and try again." |
| Clip longer than the cap (default **30 s**) | "That clip is too long for the demo. Try a shorter one (under 30 seconds)." |
| Corrupt / undecodable bytes | "Couldn't read that audio file. Try a WAV, FLAC, or OGG clip." |
| No checkpoint for the selected language | "No model is loaded for the selected language yet. Pick another language." |
| Model OOM / runtime error mid-inference | "The model ran into a problem with that clip. Try a shorter clip." |
| Anything unrecognised (catch-all) | "Something went wrong handling that audio. Try again with a different clip." |
| Unsupported language code (router mode) | The router's own message, passed through unchanged. |

The duration and silence guards run **before** inference (protecting latency and
memory). The duration cap is a single constant — `DEFAULT_MAX_DURATION_S` in
`errors.py` — not hardcoded across the UI.

**Sovereignty:** errors are logged aggregate-only (exception class + clip duration).
Audio bytes and transcripts are never logged.

---

## Troubleshooting

**Port already in use (Apptainer):**
```bash
bash scripts/demo.sh --port 7861
```

**Port already in use (bare metal):**
```bash
uv run python flaime_demo/app.py --port 8502
```

**SIF image not found:**
```bash
# Build from repo root:
apptainer build flaime-demo.sif flaime-demo.def
```

**Checkpoint not found:**
Verify the path exists and contains `config.json` (produced by the FLAIME training
pipeline). In Apptainer, confirm the host `checkpoints/` directory is populated
before running `scripts/demo.sh`.

**GPU not detected in Apptainer:**
`demo.sh` auto-detects GPU via `nvidia-smi`. If detection fails, pass `--nv`
manually:
```bash
apptainer run --nv --bind ./checkpoints:/checkpoints:ro \
    --writable-tmpfs --env FLAIME_TELEMETRY=off \
    flaime-demo.sif --server.port 7860 --server.address 127.0.0.1
```

**Browser microphone not working:**
Some browsers require HTTPS for `getUserMedia`. On localhost this is usually
exempt, but on a local-network URL (`http://192.168.x.x:...`) you may need a
self-signed certificate. File upload always works over plain HTTP.

**Slow first transcription (cold start):**
Model weights are loaded on the first transcription request and cached for the
session. Cold-start time is shown in the latency metric — this is intentional.

**MP3 files not accepted:**
The app accepts WAV, FLAC, and OGG. Convert to WAV first:
```bash
ffmpeg -i recording.mp3 recording.wav
```

---

## Development

```bash
uv sync
uv run pytest
uv run ruff check flaime_demo/
uv run mypy flaime_demo/
bash scripts/check_loc_budget.sh
```

`LOC_BUDGET` caps non-test source in `flaime_demo/`. Raising it is a deliberate,
reviewed change in the same PR as the code that needs it, never a silent drift.

## Licensing

The **code** in this repository is Apache-2.0 — see `LICENSE` and `NOTICE`.

**No model weights are hosted, bundled, or distributed here.** This repository
ships software that runs a model *you* supply: you train it or obtain it
yourself, and how you use it is between you and whoever licensed it to you.
Nothing here grants or restricts rights to any checkpoint.

As a courtesy, the foundation models this code can load carry their own terms —
verified against the upstream model cards on 2026-07-31:

| Foundation model | License |
|---|---|
| `facebook/wav2vec2-base`, `-base-960h` | Apache-2.0 |
| `espnet/xeus` | CC-BY-NC-SA-4.0 |
| `facebook/mms-1b-all` | CC-BY-NC-4.0 |

Two of the three are non-commercial, and XEUS's ShareAlike term extends to
checkpoints fine-tuned from it. Apache-2.0 on this code does not alter those
terms in either direction: it grants you the software, not the weights.

The upstream model card governs; this table is a pointer, not legal advice.
