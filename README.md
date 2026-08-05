# flaime-demo

Offline Streamlit demo for FLAIME ASR — runs fully local on a laptop; audio never
leaves the device (data sovereignty). Depends on `flaime-serving` only; FLAIME is
absent from this environment entirely.

## Status

Scaffolded by `26Q3-REPO-08`. The UI (`app.py`) landed in `26Q3-REPO-09`;
routing configs arrive in `26Q3-REPO-11`, and the offline end-to-end smoke in
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

## Development

```bash
uv sync
uv run pytest
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
