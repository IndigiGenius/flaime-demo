# flaime-demo

Offline Streamlit demo for FLAIME ASR — runs fully local on a laptop; audio never
leaves the device (data sovereignty). Depends on `flaime-serving` only; FLAIME is
absent from this environment entirely.

## Status

Scaffolded by `26Q3-REPO-08`. The UI itself (`app.py`) arrives in `26Q3-REPO-09`,
routing configs in `26Q3-REPO-11`, and the offline end-to-end smoke in
`26Q3-REPO-12` — the MVP finish line. Task cards, architecture decision, and
scope guardrails live in the FLAIME repo:
`docs/planning/tasks/26Q3-REPO/26Q3-REPO-00-overview.md`
(epic issue [FLAIME#608](https://github.com/IndigiGenius/FLAIME/issues/608),
cards PR [FLAIME#609](https://github.com/IndigiGenius/FLAIME/pull/609)).

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

## Offline requirement

The demo must run with no network access — that is the point of the extraction,
and `26Q3-REPO-12`'s smoke test enforces it. Practically:

- Dependencies resolve from the committed `uv.lock` (`uv sync --frozen`).
- `flaime-serving` is pinned to an immutable git rev, not a branch.
- No model downloads at runtime; weights come from `CHECKPOINTS_DIR`.
- Streamlit telemetry is disabled in the container environment.

## Development

```bash
uv sync
uv run pytest
bash scripts/check_loc_budget.sh
```

`LOC_BUDGET` caps non-test source in `flaime_demo/`. Raising it is a deliberate,
reviewed change in the same PR as the code that needs it, never a silent drift.

## Licensing

**Code** in this repository is Apache-2.0 (see `LICENSE` and `NOTICE`).

**Model weights are not.** No weights are committed here, but a checkpoint you
load inherits the license of the foundation model it derives from — and most of
those are non-commercial. Verified 2026-07-31:

| Foundation model | License | Commercial use | ShareAlike |
|---|---|---|---|
| `facebook/wav2vec2-base`, `-base-960h` | Apache-2.0 | ✅ permitted | no |
| `espnet/xeus` | CC-BY-NC-SA-4.0 | ❌ prohibited | **yes** |
| `facebook/mms-1b-all` | CC-BY-NC-4.0 | ❌ prohibited | no |

Two consequences worth stating plainly:

- **XEUS-derived checkpoints are CC-BY-NC-SA-4.0.** FLAIME's primary encoder is
  XEUS, so any checkpoint fine-tuned from it is a derivative work: it may not be
  used commercially, it requires attribution, and ShareAlike means that if you
  distribute it, you must distribute it under the same license.
- **Apache-2.0 on this code does not relicense those weights.** Permissive code
  plus a non-commercial checkpoint is still non-commercial in use.

If you need a commercially usable system, the wav2vec2 family is the only path
available today — but this demo is built around XEUS-BTM checkpoints, so assume
non-commercial unless you have deliberately configured otherwise.

This table is a factual record of upstream terms, not legal advice.
