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
