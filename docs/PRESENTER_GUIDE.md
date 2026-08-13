# FLAIME Presenter Guide

**Audience:** community members and partners at a gathering — not ML engineers.
**Goal:** run the live transcription demo confidently, explain it honestly, and recover gracefully if something goes wrong.

This guide is the *presentation* layer. It does **not** duplicate the deploy steps —
those live in [`README.md`](../README.md) (single source of truth).
Read that first to install; come back here to present.

> **Honesty rule for everything below:** we never overclaim. The system helps
> with transcription for the languages we have trained on; it makes mistakes; and
> it does not "understand" speech. Say so plainly. Communities are trusting us
> with their languages — earned trust is the whole point of this work.

---

## (1) Pre-demo setup checklist

Do this the morning of, on the actual demo machine, in this order:

- [ ] **Hardware ready.** Demo laptop charged + charger plugged in. External mic
      tested (if using one). Projector/screen mirrored.
- [ ] **Checkpoints fetched.** Run Step 0 of [`README.md`](../README.md)
      (`scripts/fetch_checkpoints.sh`). Confirm `checkpoints/` is populated.
- [ ] **First-time Apptainer setup only.** If this machine has never run the
      demo, do the one-time env setup first: `cp .env.example .env`
      and set `CHECKPOINTS_DIR` in it. Skip this if `.env` already exists or you're
      on bare metal.
- [ ] **Launch the app** using whichever mode the demo machine uses — Apptainer
      (`bash scripts/demo.sh`) or bare metal
      (`uv run python flaime_demo/app.py --languages-config configs/demo_languages.yaml`).
      Exact commands and flags: [`README.md`](../README.md).
- [ ] **Smoke check.** Open the local URL (`http://127.0.0.1:7860` Apptainer /
      `:8501` bare metal). Upload one known-good clip and confirm a transcription
      comes back. **Do the very first transcription now** — it pays the one-time
      cold-start cost so the audience never sees it.
- [ ] **Pick your languages.** Confirm the languages you plan to demo are loaded
      and routing correctly (the language picker shows them). Have one clip ready
      per language you'll show.
- [ ] **Network off (optional but powerful).** Disable Wi-Fi. The demo runs fully
      offline — turning the network off on stage *proves* no audio leaves the room
      (see Data sovereignty in the README).
- [ ] **Backup assets open.** Pre-recorded clips / screenshots loaded in a second
      tab or folder (see section 5), in case live inference misbehaves.

If the smoke check fails, work the **Failure playbook** (section 4) *before* the
audience arrives — not live.

---

## (2) Run-of-show

A relaxed 5–10 minute flow. Plain language; pause for questions.

1. **Why we're here (~1 min).** "Most speech technology doesn't work for our
   languages, because it was never built with them. FLAIME is an effort to change
   that — to build speech tools *with* and *for* First Languages communities,
   while the community keeps control of its own recordings."

2. **What you're about to see (~30 s).** "I'll speak (or play a clip), and the
   laptop will write down what it heard — for a language the big tech systems
   don't support. Everything happens on this laptop. Nothing is uploaded."

3. **Live transcription (~3–4 min).** Record or upload a clip in one of the
   trained languages. Let the text appear. Then:
   - Show a **second language** to make the point it's multilingual.
   - Optionally show the **latency number** the UI displays — "that's how long it
     took, all on this machine."

4. **Be honest about limits (~1 min).** Pick one:
   - "It gets things wrong — here's one." (Show a mistake; don't hide it.)
   - "For languages we haven't trained on yet, it tells you that, instead of
     guessing." (Show an unsupported-language state.)

5. **Sovereignty close (~1 min).** "The recordings that train this stay with the
   community. The laptop has no internet right now — and it still works. That's by
   design, not by accident." (Reference the Data sovereignty section of the README.)

6. **Questions.** Move into the FAQ below.

---

## (3) FAQ

Honest answers to likely questions. Keep them short and plain.

**Q: Where does my voice go when I speak into it?**
Nowhere. It's processed on this laptop's memory and discarded right after. There's
no upload, no save-to-disk of your audio, and right now no internet at all. (This
is enforced in the code, not just promised — see the README's sovereignty section.)

**Q: Is this like the speech recognition on my phone?**
Similar idea, different goal. Phone systems support big commercial languages.
FLAIME is built for languages those systems ignore, and it runs locally instead of
sending your voice to a company's servers.

**Q: How accurate is it?**
It varies a lot by language and by how much training audio we had. It makes real
mistakes, especially on names, places, and unusual words. It's a helpful first
draft of a transcript, not a finished one — a human should always review it.

**Q: Which languages does it support?**
The ones loaded for this demo are shown in the language picker. We've trained on a
set of languages and are adding more. For languages we haven't trained yet, the
app says so rather than guessing.

**Q: What happens with a language you haven't trained on?**
It tells you it doesn't have a model for that language yet, instead of inventing a
wrong answer. We'd rather say "not yet" than mislead you.

**Q: Who owns the recordings used to build this?**
The community does. Audio shared by a partner is used only with explicit written
permission, and the default demo ships with no partner audio at all. Communities
decide how their language data is used.

**Q: Can it work without internet?**
Yes — and we can prove it by turning Wi-Fi off right now. All the computation
happens on this device.

**Q: How does it handle two languages mixed in one sentence (code-switching)?**
That's genuinely hard, and it's one of the cases where it's most likely to slip.
We test for it honestly rather than pretending it's solved.

**Q: Will this replace our language teachers or speakers?**
No. It's a tool to *support* documentation and learning — transcribing recordings
faster, for example. It can't teach the language or replace a fluent speaker, and
it shouldn't be treated as an authority on what's correct.

**Q: Can we get this for our community's language?**
That's exactly the conversation we want to have. It depends on having recordings to
learn from, and on the community deciding it wants to — on its terms. Let's talk.

---

## (4) Failure playbook

What to do *calmly, on stage* when something breaks. The demo is built to fail
gracefully — most failures show a friendly message, not a crash (see the
error table in [`README.md`](../README.md)).

| If this happens | The app shows / you do |
|-----------------|------------------------|
| **No / silent audio** | App says the clip sounds silent. Just re-record closer to the mic, or upload a clip. No restart needed. |
| **Obviously wrong transcription** | Lean into it: "this is where it struggles." It's a teaching moment, not a failure. Move to your next clip. |
| **Clip too long** | App says it's too long (cap ~30 s). Use a shorter clip — keep demo clips under 30 seconds anyway. |
| **Unsupported language** | App says no model for that language yet. This is *expected behavior* — frame it as honesty, then switch to a supported language. |
| **Unreadable file / wrong format** | App asks for WAV/FLAC/OGG. Use a prepared WAV clip. Convert with `ffmpeg` if needed (README troubleshooting). |
| **App crashes / freezes / blank page** | Don't debug live. Switch to the **backup plan** (section 5). Relaunch in the background if you can. |
| **Mic not working in browser** | Switch to **file upload** — it always works over plain HTTP (README troubleshooting). |
| **Very slow first response** | That's cold start — which is why you ran a transcription during setup. If it slipped, narrate it: "it's warming up." |

Golden rule: **never read a stack trace to the audience.** If you ever see one,
move to the backup plan and keep talking.

---

## (5) Backup plan

Assume the live model will misbehave at least once. Have these ready *before* you
start, so the fallback is seamless.

- **Pre-recorded clips + their expected transcripts.** 2–3 short clips per demo
  language that you've already confirmed transcribe well. Stored alongside the demo
  machine (location: _fill in on the demo machine — e.g. `~/flaime-demo-backup/`_).
- **Screenshots of a successful run** (transcription + latency visible), in case
  the app won't launch at all. Show these and narrate what *would* happen.
- **A short screen recording** of an end-to-end transcription, as the strongest
  fallback if the laptop can't run the model on the day.
  > **Produce the recording during the dry run (backlog `26Q3-DEMO-11`), not from
  > this guide.** This guide only lists the requirement and storage location;
  > recording it needs the working demo on the target hardware.
- **Sovereignty constraint on all backup assets:** use only DEMO-05
  licensed/synthesized samples. **Never** use uncleared partner audio in a clip,
  screenshot, or recording. When in doubt, leave it out.

---

## Talking-points accuracy notes (presenter, read once)

So you can answer follow-ups without overclaiming:

- **We route to a per-language expert; we don't ship a single merged model.** In
  our 64-language results, naively merging the experts into one model performed
  *worse* than the starting point, while keeping a specialist per language did
  best. If asked "is it one big model?" — no, it picks the right specialist.
- **There's early evidence it transfers to unseen languages.** In a held-out test
  (Indic languages it wasn't trained on), continued self-supervised training
  reached low error rates — promising, not proven-at-scale. Don't promise it works
  for *any* language out of the box.
- **Coverage is honest by design.** Languages we haven't trained on surface a
  "not supported yet" state rather than a confident wrong answer.
- **It's a transcription aid, not a language authority.** Always frame output as a
  draft for human review.

---

## Changelog / dry-run sign-off

> **Status 2026-07-31:** the July gathering was **cancelled**, and the dry-run tail
> (`26Q1-DEMO-09` / `26Q1-DEMO-10`) was cut from Sprint 13 unrun. Another demo is
> planned but not yet scheduled. The sign-off below is therefore **still open** —
> it belongs to backlog **`26Q3-DEMO-11`**, which also retargets the rehearsal at
> `flaime-demo` once the 26Q3-REPO extraction lands. Treat this guide as a draft
> that no one has yet run end-to-end.

- [ ] **Non-author dry run (`26Q3-DEMO-11`):** a second team member ran the demo
      *from this guide alone* and reached a working demo. Recorded here:
      _name / date / notes — fill in at the dry run._
- [ ] Backup clips recorded and stored at the documented location.
- [ ] FAQ updated with any worst-sample findings from the DEMO-05 community eval
      once that cluster run lands.

*Last updated: 2026-07-31 — dry-run references repointed to backlog `26Q3-DEMO-11`
after the July gathering was cancelled and DEMO-09/10 were cut. Original draft
2026-06-22 (DEMO-08); results-dependent FAQ items still pending DEMO-05.*
