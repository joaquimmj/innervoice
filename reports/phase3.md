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

### 1b. Calibration curve and shuffled-label control (`results/phase3_calibration_v2`, commit 82c714e)
Independent rerun (3 new trainings), fine-tuning all weights for 100 steps, zero-shot 26.4 % / 6.5 %:

| Minutes (labelled words) | 50 words: gain [95 % CI] | improved | 250 words: gain [95 % CI] | improved |
|---|---|---|---|---|
| 5 (677) | +2.04 [+0.61, +3.56] | 7/8 | +0.65 [+0.14, +1.24] | 7/8 |
| 10 (1 462) | +4.45 [+2.90, +6.00] → 30.8 % | 8/8 | +1.51 [+1.02, +1.99] | 8/8 |
| 20 (3 085) | +5.41 [+3.47, +7.26] → 31.8 % | 8/8 | +1.75 [+1.34, +2.14] | 8/8 |
| 10, **labels shuffled** (control) | −5.87 [−7.60, −4.14] → 20.5 % (chance) | 0/8 | −1.74 [−2.32, −1.09] | 1/8 |

The 10-min gain replicates (+4.42 in v1, +4.45 here). Returns diminish: +2.4 points from 5 to 10 min, +1.0 from 10
to 20. With shuffled labels the model falls to chance, so the gain needs the true brain–word pairing. 300 steps give
the same result as 100; 30 steps give less at 10–20 min.
With 10 min of calibration, averaging 10 observations of a word reaches 2.4 % balanced top-1 over 250 words
(zero-shot 0.25 %, chance 0.4 %): statistically above zero-shot (8/8 subjects) but far too low to be useful.

### 1c. Label-free functional alignment to the training subject (`results/phase3_alignment_v1`, commit 89b72fe)
Every LibriBrain100 subject heard the same book (Sherlock1) as sub-0. A 306×306 ridge map from the new subject's MEG
to sub-0's MEG on the same audio (word-locked windows paired through the shared stimulus timeline; onset intervals
match exactly, 0.0 s) is fitted on the first K minutes of recording A; the unchanged sub-0 decoder is applied to
the mapped recording B. Sub-0's Sherlock1 MEG is never used for training. Zero-shot 26.4 % / 6.5 %. 3 seeds, 8 subjects.

| Method | 5 min (50 w) | 10 min (50 w) | 20 min (50 w) | 20 min (250 w) |
|---|---|---|---|---|
| supervised fine-tune (labels) | +2.01 [+0.57, +3.52] | +4.41 [+2.86, +5.94] | +5.37 [+3.48, +7.16] | +1.71 [+1.31, +2.10] |
| **alignment, ridge 0.1 (no decoder training)** | **+3.59 [+1.67, +5.60]** | +4.32 [+2.19, +6.76] | +5.94 [+3.83, +8.13] | +1.47 [+0.80, +2.14] |
| alignment, then fine-tune | +1.69 [−0.65, +3.92] | +4.95 [+2.26, +7.63] | **+6.96 [+4.89, +8.82] → 33.3 %** | **+2.31 [+1.96, +2.66]** |
| shuffled-label fine-tune (control) | −6.71 | −5.83 | −5.14 | −1.77 |

Paired difference alignment − supervised at 50 words: +1.58 [+0.41, +2.87] at 5 min, −0.08 at 10 min, +0.58 at
20 min (n.s.). Alignment + fine-tune − supervised at 20 min: +1.60 [+0.72, +2.57] (50 w), +0.60 [+0.22, +1.06] (250 w).
**Controls (`results/phase3_alignment_v2`, commit f6d9558; same 3 decoders, same subjects):**

| 50 words, gain vs zero-shot | 5 min | 10 min | 20 min |
|---|---|---|---|
| alignment (stimulus-paired windows) | +3.59 [+1.67, +5.60] 7/8 | +4.32 [+2.19, +6.76] 7/8 | +5.94 [+3.83, +8.13] 8/8 |
| same map, **random pairing** within the shared book | −4.15 [−5.59, −2.80] 0/8 | −3.73 [−5.61, −1.97] 1/8 | −2.07 [−3.40, −0.92] 1/8 |
| **covariance matching only** (CORAL, no pairing) | −0.47 [−1.54, +0.70] 3/8 | −0.13 [−1.11, +0.79] 3/8 | +0.18 [−0.78, +1.17] 4/8 |
| alignment − random pairing (paired) | +7.74 [+6.04, +9.53] 8/8 | +8.06 [+6.01, +10.82] 8/8 | +8.02 [+6.73, +9.85] 8/8 |
| alignment − covariance only (paired) | +4.06 [+2.45, +5.90] 8/8 | +4.46 [+2.14, +7.06] 8/8 | +5.76 [+3.65, +8.21] 8/8 |

At 250 words: alignment − random pairing +1.75 / +2.36 / +2.54 (8/8 each); alignment − covariance only
+0.56 [−0.05, +1.21] / +0.94 [+0.32, +1.56] / +1.51 [+0.79, +2.22].
**The gain requires stimulus-locked correspondence between the two brains**: destroying the pairing makes the map
harmful, and matching second-order sensor statistics alone gives nothing. Alignment needs the new subject to listen
to the same audio as the training subject; it removes the need to train the decoder, not the need for a known
stimulus. Novelty: the method is published for fMRI (Ferrante et al. 2023/24) and was a public null on LibriBrain
with a weaker decoder (rohit-ravi2, P1b); a weak-decoder replication and Procrustes comparison are running.

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
