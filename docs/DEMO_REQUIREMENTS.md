# FLAIME Demo Requirements

**Status**: Draft — Sprint 6 (26Q2-DEMO-PREP)
**Owner**: FLAIME team
**Target Date**: Late-June 2026 (working build), mid-July 2026 gathering (community-partner showing)
**Last Updated**: 2026-04-30

This document captures *who* the FLAIME demo is for, *what* it must do, and the architectural decisions Sprint 7 will design against. It is the input contract for Sprint 7 demo-design work, not the demo design itself.

This doc is also intended as a **reusable template** for FLAIME community-partner demos. Future partners (different language, different gathering, different model snapshot) inherit the same section structure — Audience, Goals, Platform, Languages, Inference, Data Sovereignty, Timeline, Open Questions — and re-fill the contents per engagement. The current instance is scoped to the mid-July 2026 partner; later instances should fork a copy and update the dates, model, and partner-specific decisions.

Decisions take a position by default. Items that genuinely cannot be settled until Sprint 7 are in [Open Questions](#open-questions).

---

## Audience

The demo is built for **a community partner** working with FLAIME on indigenous-language ASR — the specific partner is named in [Open Questions](#open-questions) and resolved in Sprint 7. Assumptions:

- **Primary audience**: indigenous-language community members and partner staff. Generally **non-technical**: comfortable with web/phone apps, not with command lines, Python, or model internals.
- **Secondary audience**: FLAIME researchers and the July gathering's broader attendees (mixed technical / non-technical, mixed indigenous / non-indigenous).
- **Language background**: speakers and learners of the partner's language(s); also speakers of widely-resourced languages (English, French, Spanish) that bridge to demo content.
- **Setting**: in-person at the July gathering; likely a single laptop on a table with one or two people interacting at a time. **Not a streaming demo to a remote audience.**

---

## Goals — Success Criteria

The demo succeeds if a non-technical visitor, in <5 minutes of interaction, can:

1. **Transcribe a held-out audio clip in a supported language** and see the model's output. Threshold: WER below the matching no-BTM / wav2vec2 baseline on the same clip (concrete numbers locked at Sprint 7 once XEUS-BTM-03 / ABL-01-LITE results land).
2. **See a side-by-side comparison of BTM vs. no-BTM** on the same input, illustrating the research narrative ("expert merging helps low-resource languages") without reading the paper.
3. **Run the entire transcription locally** with no network call — visibly demonstrating that audio never leaves the device, in line with [Data Sovereignty](#data-sovereignty).

Stretch (drop if budget tight at S7):

4. **Record a fresh utterance** via the laptop microphone and transcribe it, in a language present at the gathering.

Out of scope for this demo:

- Real-time streaming transcription (offline file/clip transcription only).
- Production-quality UI polish — this is a research demo.
- Multi-user or networked operation.
- Speaker diarization, punctuation restoration, translation.

---

## Platform

**Decision**: Local **web app** (Gradio or Streamlit) launched from a single command on a laptop, served on `localhost`, no cloud, no upstream API calls at demo time.

**Rationale**:
- **Web UI** beats CLI for a non-technical audience — pointing at a button and dragging a file is universal.
- **Local server** (not a hosted service) is the only platform consistent with [Data Sovereignty](#data-sovereignty): no data leaves the device by construction, and we can demonstrate that by toggling Wi-Fi off.
- **Gradio vs. Streamlit** decision deferred to Sprint 7 — both work; Gradio has lighter audio-widget overhead, Streamlit has nicer side-by-side layout. Whichever the S7 implementer prefers.
- **Python framework, not mobile/native**: we already ship a Python inference path (`flaime evaluate`, INFRA-16); native packaging is out of scope this sprint cycle.

**Constraints**:
- **Hardware**: single laptop with at least one consumer NVIDIA GPU (e.g., RTX 4070+) OR a recent Apple Silicon Mac with `mps` backend. The exact target machine is in [Open Questions](#open-questions).
- **Offline**: must boot and run with Wi-Fi disabled. All model weights, tokenizers, and audio assets bundled at install time.
- **Cold-start**: <60 s from launch to first transcription on the demo machine.
- **No login, no telemetry**, no analytics. The app does not phone home.

---

## Languages

**Decision**: Demo coverage drawn from the language set of whichever XEUS BTM checkpoint actually ships:

- **If XEUS-BTM-03 lands**: subset of the 64-language set, prioritizing (a) widely-resourced languages with strong WER for confident demos and (b) at least one typologically distant language to showcase BTM expert advantage on hard cases.
- **If we fall back to XEUS-BTM-02**: subset of the 16-language set — concretely English, French, Spanish, German, Dutch, Arabic, Georgian. These are at-bar Common Voice languages, strongest WER / clearest BTM-vs-no-BTM contrast, safe to demo without sovereignty risk (CV is open-license, consented).

Either way, we ship **one typologically distant language** (Georgian or Arabic at minimum) to make the BTM expert advantage visible on a hard case — this is the headline for the research narrative.

**Community-language coverage**: a hard gate, not a default. See [Open Questions](#open-questions) for the data + evaluation criteria that must clear before the community language ships in the demo. If the gate doesn't clear, the demo runs on widely-resourced languages and the community-language path is discussed narratively at the gathering rather than transcribed live.

**Rationale**:
- Only languages with merged-and-evaluated experts appear in the demo. We do not promise inference quality on an unevaluated language.
- The demo will **not** include any indigenous-language audio that the community partner has not explicitly cleared. The default is to show research progress *toward* indigenous-language ASR using widely-resourced languages as proxies, then discuss the path forward in person.

---

## Inference Architecture

**Model — primary**: XEUS BTM-03 merged checkpoint (64-lang full run). This is the headline model the demo is built around.

**Model — fallback**: XEUS BTM-02 merged checkpoint (16-lang, SSL+ctcfix arm — best merged WER as of Sprint 6, ~39.53%). Used if XEUS-BTM-03 has not landed in time, or if BTM-03 fails final evaluation. BTM-02 is the safety net; the demo ships with one of these two, not both.

XEUS-BTM-03 is currently ⏸️ PAUSED on PHONET-01 ([#256](https://github.com/IndigiGenius/PhoNet/issues/256)) — the same hard block as the indigenous-language data path. Slip in PHONET-01 → fall back to BTM-02. Single static checkpoint either way, loaded once at app startup.

- *Why merged, not per-language expert*: the demo's research story is "BTM merging works." A per-expert UI implies the user picks a language up-front, which buries the merging story.
- *Why XEUS, not wav2vec2*: XEUS is FLAIME's primary foundation model per the Sprint 6 strategic pivot.
- *Side-by-side baseline*: a second model — wav2vec2 baseline or BTM no-merge Phase 0 — runs in parallel for the BTM-vs-no-BTM comparison. Exact baseline locked at Sprint 7 once ABL-01-LITE results land.

**Decoder**: **Beam search**, width 5–10, via the path delivered in [DEC-01](https://github.com/IndigiGenius/FLAIME/pull/335). Greedy is the fallback if beam search introduces latency >2 s/clip on the demo machine.

**Inference path**: backend reuses the `flaime evaluate` codepath ([INFRA-16, PR #343](https://github.com/IndigiGenius/FLAIME/pull/343)) — the web app is a thin wrapper around the same inference function used in evaluation. No new training-only branches in the demo.

**Where it runs**: entirely on the demo laptop. No remote endpoint. No model download at runtime — weights ship in the bundle.

---

## Data Sovereignty

This is a **non-negotiable hard constraint** per `CLAUDE.md`. The demo enforces it by construction, not by policy:

1. **No upload of audio, ever.** All inference is local. The app has no network code path for audio. We will demonstrate this on stage by disabling Wi-Fi.
2. **No persistent storage of audio recorded at the demo.** If the stretch "record a fresh utterance" feature ships, recordings are kept in memory only — written to disk only if the user explicitly clicks "Save", and only to a path they choose.
3. **Indigenous-language audio in the bundle requires explicit community-partner approval.** Default bundle ships with Common Voice clips only (open-license, consented). Any partner-supplied indigenous audio is added under a written agreement reviewed before the July gathering.
4. **No telemetry / analytics / crash reporters.** The app is silent on the network.
5. **Visible privacy affordances in the UI**: a clear "all processing is local" line in the header; a "no data leaves this device" confirmation on the upload widget.

If any of these break during S7 implementation, that is a stop-ship bug, not a polish item.

---

## Timeline

About eleven weeks between sprint close (2026-05-01) and the mid-July gathering. Four internal checkpoints, ~2 weeks apart:

| Date (target) | Checkpoint | Deliverable |
|---|---|---|
| **2026-05-22** | **Sprint 7 design freeze** | Demo design doc: UI sketch, framework decision (Gradio vs. Streamlit), exact baseline model, exact language list, exact community partner. Closes [Open Questions](#open-questions). Feeds off the design meeting in early May. |
| **2026-06-05** | **Feature freeze + internal dry run** | App boots offline on the demo laptop, transcribes a held-out clip in each shipped language, BTM-vs-no-BTM side-by-side works. Internal team runs through the 5-minute visitor flow. |
| **2026-06-19** | **Community-partner dry run** | Remote or in-person review with the community partner. Approve language/audio bundle. Sign off on data-sovereignty affordances. Final go/no-go on shipping the community language. |
| **2026-06-29** | **Working build ready** | Bundle frozen, install instructions written, demo laptop imaged. ~2-week buffer to the gathering. |
| **2026-07-XX** | **Mid-July gathering** | In-person showing. Date confirmed in S7 once partner schedule is set. |

If any checkpoint slips by >1 week, raise immediately at standup — the gathering date is fixed.

---

## Open Questions

Decisions deferred to Sprint 7 demo-design (or earlier if the answer arrives).

1. **Community partner identity** — which partner is the demo built for? Drives language list, audio approvals, and the tone of the UI copy. Owner: PI. Needed by 2026-05-08.
2. **Community-language inclusion (hard gate)** — does the demo ship with the community partner's language, or does the gathering run on widely-resourced languages with the community-language path discussed narratively? **Both** of the following must clear, otherwise the demo runs without the community language:
   - **(a) Data**: ~10 hours of transcribed audio in the community language obtained, cleaned, and approved for use by the partner, in time to fine-tune / evaluate before the partner dry run on 2026-06-19.
   - **(b) Evaluation**: the resulting model evaluated by community partner leadership and confirmed to be *actually recognizing words in the community language* — not hallucinating, not collapsing to a related high-resource language, not producing plausible-looking gibberish. This is a qualitative judgement call by speakers of the language, not a WER number.

   Hard-blocked on PHONET-01 ([#256](https://github.com/IndigiGenius/PhoNet/issues/256)) for the data ingestion path even before the 10-hour gate is checked. Fall back to "narrative only" if either gate fails.
3. **Demo machine spec** — which physical laptop runs the demo at the gathering? GPU vs. Apple Silicon `mps` decision affects bundle build and beam-search latency budget.
4. **Framework: Gradio vs. Streamlit** — Sprint 7 implementer picks; both are viable.
5. **Side-by-side baseline** — wav2vec2 baseline, or XEUS BTM Phase 0 (no-merge)? Decided once XEUS-BTM-03 / ABL-01-LITE land.
6. **Concrete WER thresholds** — the Goals section sets a relative bar ("below the no-BTM baseline"); absolute numbers locked once XEUS-BTM-03 / ABL-01-LITE produce final per-language WERs.
7. **Live-microphone (stretch)** — ship or drop? Decision at the 2026-05-22 internal dry run based on remaining time.

---

## See Also

This doc originated in the FLAIME monorepo before the 26Q3-REPO demo extraction; the
links below point back at FLAIME's own history and planning docs rather than into
this repo, since none of that context lives here.

- [docs/planning/current-sprint.md](https://github.com/IndigiGenius/FLAIME/blob/main/docs/planning/current-sprint.md) — Sprint 6 task table, mid-June demo target line.
- [docs/planning/tasks/26Q2-06/26Q2-DEMO-PREP-demo-requirements.md](https://github.com/IndigiGenius/FLAIME/blob/main/docs/planning/tasks/26Q2-06/26Q2-DEMO-PREP-demo-requirements.md) — task definition for this doc.
- [docs/planning/26Q1-RESEARCH-PLAN.md](https://github.com/IndigiGenius/FLAIME/blob/main/docs/planning/26Q1-RESEARCH-PLAN.md) — overarching research framing.
- [`CLAUDE.md`](https://github.com/IndigiGenius/FLAIME/blob/main/CLAUDE.md) — Indigenous data sovereignty requirements (non-negotiable).
- [PR #283](https://github.com/IndigiGenius/FLAIME/pull/283) — XEUS BTM 16-lang (the merged checkpoint the demo runs).
- [PR #335](https://github.com/IndigiGenius/FLAIME/pull/335) — DEC-01 beam search.
- [PR #343](https://github.com/IndigiGenius/FLAIME/pull/343) — INFRA-16 `flaime evaluate` CLI entrypoint.
