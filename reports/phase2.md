# Phase 2 — Cross-subject word decoding on LibriBrain100: what is brain, what is transcript timing (2026-11-19)

Setup (`kaggle/xsub/xsub.py`): train on sub-0 (Sherlock2–7, 453 958 words, 16 169 types), validate on sub-1…4,
test zero-shot (no calibration) on **8 never-seen subjects** sub-5…12 listening to a different book (Sherlock1).
3.6 M-param word-window CNN (+ optional cross-word transformer), contrastive to frozen T5-large word-type embeddings.
Metric: balanced top-10 accuracy over the N most frequent training words; "unseen" = 402 word types absent from the
training text (chance 2.5 %). Mean over test subjects; range over 3 seeds where available.
Data: `results/phase2_xsub_v1/` (commit 93c23c7), `results/phase2_xsub_v2/` (commit e327d91). GPU: ~2.3 h T4 in total.

| Model input | 50 words | 250 words | 1000 words | unseen words | seeds |
|---|---|---|---|---|---|
| chance | 20.0 | 4.0 | 1.0 | 2.5 | |
| **Transcript timing only, no MEG** (gap to previous/next onset, duration, position in sentence) | **75.8–76.0** | **42.4–43.7** | 16.0–17.1 | 14.7–18.4 | 3 |
| MEG, sentence context (v1) | 60.9–62.2 | 24.6–27.3 | 7.2–7.7 | 8.5–10.7 | 3 |
| MEG, 12-word chunks, no sentence/position input (v1) | 51.5–52.4 | 18.5–19.2 | 4.2–4.5 | 6.4–7.3 | 3 |
| **MEG, one word at a time (no cross-word context)** | **26.5–27.0** | **6.0–6.6** | 1.0–1.6 | 3.4–4.0 | 3 |

Controls (50 words; chance 20):

| Model | MEG shifted 7 s (same shift for all words) | MEG at an independent random time per word | white noise |
|---|---|---|---|
| sentence context | 54.2–55.6 | 21.7 (1 seed) | 21.5–22.7 |
| 12-word chunks | 44.9–45.9 | 19.8 (1 seed) | 18.8–19.3 |
| one word at a time | 20.2–20.4 | 20.0–20.5 | 19.7–20.3 |

## What this shows
1. A model that never sees brain data, only word timing taken from the transcript alignment, reaches 76 % (50 words)
   and 43 % (250 words). That is above every MEG model here, and above the numbers published for the same metric
   on LibriBrain (MEG-XL 57.3–63.0 % / 19.8–25.4 %; different split and subject, so not a like-for-like comparison).
2. Our context models keep ~90 % of their score when the MEG is misaligned by 7 s but collapse to chance when each
   word's window is taken at an independent random time or replaced by noise. They therefore read the *relative
   timing of the word windows* (recoverable from overlapping windows), not word-locked brain responses.
3. With context removed, the brain-only zero-shot signal on unseen subjects is real but small: +6.7 points over
   chance at 50 words, +2.2 at 250, about +1 on unseen words (controls at chance). In-subject (held-out sub-0
   recordings): 38–44 % at 50 words.

## What this does NOT show (to verify before any claim)
- Not tested on MEG-XL's or d'Ascoli et al.'s own code. MEG-XL pools features inside each word window and its head
  never sees neighbouring windows, so it may be less exposed than sentence-level models. Unknown until measured.
- The timing-only model is given annotated word *duration*, which a MEG model cannot read directly; an ablation
  with onset gaps only is missing.
- One training subject, one test book shared by all test subjects (subject SD understates uncertainty), no CIs yet.
- Novelty not yet checked for this specific finding (rule 1).
