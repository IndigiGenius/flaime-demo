# flaime-demo

Offline Streamlit demo for FLAIME ASR — runs fully local on a laptop; audio never
leaves the device (data sovereignty). Depends on `flaime-serving` only; FLAIME is
absent from this environment entirely. Model checkpoints stay **outside** the repo
(`CHECKPOINTS_DIR` env contract — weights are never committed).

**This repo is awaiting its scaffold** (task `26Q3-REPO-08`). The task cards,
architecture decision, and scope guardrails live in the FLAIME repo:
`docs/planning/tasks/26Q3-REPO/26Q3-REPO-00-overview.md`
(epic issue [FLAIME#608](https://github.com/IndigiGenius/FLAIME/issues/608),
cards PR [FLAIME#609](https://github.com/IndigiGenius/FLAIME/pull/609)).

Private under IndigiGenius, consistent with community data governance.
