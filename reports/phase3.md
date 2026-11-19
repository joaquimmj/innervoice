# Phase 3 — Timing-free decoding on never-seen subjects: calibration, repeated observations, more training subjects (2026-11-19)

All models decode **one word window at a time** (no cross-word context, so no timing shortcut; shift / random-time /
noise controls at chance, see `phase2.md`). 3.6 M parameters, trained on sub-0 (Sherlock2–7), 3 seeds.
Test: 8 never-seen subjects (sub-5…12). Metric: balanced top-10 accuracy over the 50 / 250 most frequent words
(chance 20 % / 4 %). Unless stated, evaluation is on each subject's second recording (B), whose text is disjoint from
the calibration recording (A): 5-gram overlap 0.6 %.

## 1. Calibration with the new subject's first minutes (`results/phase3_calibration_v1`, commit 29c5c1e)
Zero-shot on recording B: 26.6 % (50 words), 6.5 % (250 words). Gains are paired per subject, averaged over seeds;
95 % CI by bootstrap over the 8 subjects.

| Method | Minutes (labelled words) | 50 words: gain [95 % CI] | subjects improved | 250 words: gain [95 % CI] | subjects improved |
|---|---|---|---|---|---|
| BatchNorm statistics only (no labels) | 10 | +0.45 [−0.06, +0.99] | 4/8 | −0.01 [−0.17, +0.15] | 3/8 |
| Input mixing matrix 306×306 | 1 (61) | −3.85 [−5.92, −1.98] | 0/8 | −1.44 [−2.11, −0.85] | 0/8 |
| Input mixing matrix | 5 (699) | +1.76 [−0.39, +3.81] | 5/8 | +0.06 [−0.59, +0.80] | 4/8 |
| Input mixing matrix | 10 (1 483) | +3.49 [+1.24, +5.77] | 6/8 | +1.19 [+0.78, +1.61] | 8/8 |
| Fine-tune all weights, 100 steps | 1 | −1.91 [−3.64, +0.06] | 2/8 | −0.83 [−1.19, −0.48] | 0/8 |
| Fine-tune all weights | 5 | +1.87 [+0.66, +3.16] | 6/8 | +0.66 [+0.37, +0.98] | 8/8 |
| **Fine-tune all weights** | **10** | **+4.42 [+2.64, +5.92] → 31.0 %** | 7/8 | **+1.47 [+1.08, +1.92] → 8.1 %** | 8/8 |

Hyper-parameters: 100 steps was chosen over 300 on validation subjects in every case. Caveats: CI over 8 subjects who
all heard the same text; the shuffled-label control for calibration is in the next run, not this one.

## 2. Repeated observations of the same word (`results/phase3_agg_v1`, commit cc6fad8) — negative
Averaging the embeddings of n occurrences, retrieval over 250 words, balanced top-1 (chance 0.4 %), zero-shot:
n=1: 0.7–0.8 %; n=10: 1.5 / 0.0 / 0.5 % by seed (noise control 0.0–0.5 %). No reliable gain on unseen subjects.
In-subject (held-out sub-0 recordings): 1.5–2.5 % → 1.2–4.7 %, unstable too.

## 3. More training subjects (`results/phase3_scaling_v1`, commit c632f9a) — no reliable gain on new text
Training = sub-0 + recording A of N other subjects (half of each batch). Test subjects never trained on.

| Extra training subjects | 50 words, recording B (text never trained on) | 250 words, recording B | 50 words, A+B (A = text heard by training subjects) |
|---|---|---|---|
| 0 | 26.4 | 6.5 | 26.3 |
| 4 | 26.2 | 6.7 | 27.2 |
| 10 | 28.3 | 7.3 | 30.6 |
| 20 | 25.8 | 6.2 | 29.3 |

The apparent improvement on A+B is stimulus sharing (same audio heard by training subjects), not transfer to new
material. On disjoint text there is no monotonic effect (+1.9 at N=10, −0.6 at N=20). Extra subjects contribute
only 6–24 min each (≈5 h in total against 45 h of sub-0).

## Reading
- Honest zero-shot decoding on a new person is weak (6.6 points above chance at 50 words).
- **Ten minutes of the person's own labelled data add 4.4 points; one minute hurts.** The gain grows with minutes
  (5 → 10), so the curve beyond 10 min is the next measurement.
- Neither repeated observations nor up to 20 extra short-recording subjects help at this signal level.
- Not a breakthrough: absolute accuracy stays far from the timing-inflated literature numbers and from in-subject
  performance (38–48 % at 50 words on held-out sub-0 recordings).
- Novelty of the calibration result: not yet checked (rule 1).
