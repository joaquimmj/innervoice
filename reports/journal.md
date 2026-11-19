# Journal

## 2026-11-18
- Done: repo scaffold; secret scanner + pre-commit hook; `kaggle_run.py` + hello-GPU job (untested on Kaggle);
  novelty check (abstract level); measured MEG-XL encoder activation memory on CPU.
- Results: 80 GB = activations of 191k tokens/sample (57 GiB fp32 at 150 s); est. 7 GiB with 16-bit + checkpointing.
- Problems: no Kaggle/HF secrets; huggingface.co, kaggle.com, zenodo.org, openneuro.org blocked; GitHub repo not yet
  created/attached, so this was committed locally only.
- Kaggle GPU hours used: 0.
- Next: run hello-GPU, LibriBrain inventory, check Zenodo 23895809, then Phase 1 memory probe on T4.

## 2026-11-18 (later) — repo `joaquimmj/innervoice`, session moved to env `nervegear`
- Kaggle works (account joaquimmanjijung; 2×T4 14.6 GiB). Credentials injected by the session proxy; Kaggle Secrets
  unavailable to API-pushed kernels and not needed (all inputs public).
- LibriBrain2 inventory: 32 new subjects all on Sherlock1, 16.2 GiB h5 total; sub-13…32 partly embargoed until after the competition.
- Maryland listened/imagined MEG is public (Zenodo 23895809, CC-BY-4.0, 18 subj) → Voie B unblocked (small vocab).
- MEG-XL fine-tuning head (`CrissCrossWordEmbeddingExtractor`): 323 M params (306×512 → 2048 linear), vs 21 M encoder.
- Phase 1 memory probe launched on Kaggle T4 (`kaggle/memory_probe`).
- Phase 1 memory: 150 s full fine-tune fits on 2×T4 (8.8 + 6.0 GiB, 5.7 s/step); single T4 up to 120 s. See `reports/phase1.md`.
- First probe run failed (v2 ckpt has extra hparam `debug_nan_checks`); fixed with our own loader. Kaggle GPU used: ~0.3 h (3 short jobs).
- Next: accuracy reproduction on LibriBrain (needs a data subset: sub-0 h5 is 55 GiB, /kaggle/working has 19.5 GiB).

## 2026-11-18 (night) — pivot of the base model, data prepared
- Decision (delegated by Joaquim): do not spend >= 40 GPU-h reproducing MEG-XL's single-subject number. Evidence:
  on unseen subjects MEG-XL scores 0.43 (PNPL Broad reference) while a 5 M-param CNN + sentence transformer trained
  on sub-0 only scores 0.66 zero-shot (rohit-ravi2/libribrain-2026, MIT). Base model for Voie A = that family;
  MEG-XL stays a baseline. Phase 1 memory result stands on its own.
- Data prep on Kaggle CPU (no GPU quota): sub-0 91/91 recordings, 53.8 h, 11.05 GiB cache; sub-1…12 24/24 recordings,
  9.7 h (~48 min each), 2.0 GiB. Sub-13…32 live in `serialised_competition` (first halves): not prepared yet.
- `kaggle/xsub/xsub.py`: train sub-0 (Sherlock2–7), zero-shot test on sub-5…12 (val sub-1…4), retrieval over
  50/250/1000 words + unseen word types, permuted-label control. Launched.

## 2026-11-19 — cross-subject results and a timing confound
- Zero-shot on 8 unseen subjects: MEG sentence-context model 61 % (50 words), but the 7 s-shift control keeps 55 %.
- Timing-only model (no MEG) reaches 75.8 % / 43.6 % (50 / 250 words): transcript rhythm alone beats the MEG models.
- Honest brain-only (one word at a time): 26.8 % / 6.2 % (chance 20 / 4), controls at chance. See `reports/phase2.md`.
- Kaggle GPU used so far: ~3 h (probes 0.3, xsub runs 0.3 + 1.1 + 1.1). Container restarted 3 times; Kaggle jobs unaffected.
- Next: novelty check of the timing finding; gap-only timing ablation; test the confound on MEG-XL's own pipeline.
- Novelty check: the timing confound was published by Oxford on 2026-09-30 (arXiv 2609.40359, SimpleB2T). STOP on that
  claim (rule 1). What is left open, to our knowledge: timing-free decoding on never-seen subjects with <= 10 min of
  calibration (our single-word baseline: 26.8 % @50 zero-shot vs 38-44 % in-subject), and imagined speech (Maryland data).

## 2026-11-19 06:45 UTC — autonomous night plan (Joaquim asleep, asked to keep running)
Running on Kaggle: `innervoice-xsub` v4 (calibration study: bn / input / full at 1, 5, 10 min; + multi-subject
training; 3 seeds; ~2.5 h GPU) and `innervoice-xsub-agg` v1 (zero-shot + repeated-observation aggregation, 3 seeds; ~45 min).
Queue, in order, each step only after the previous results are read and saved under `results/`:
1. Save both results, write `reports/phase3.md` (calibration gains with SD over subjects, aggregation curves).
2. If any calibration method gives a positive gain on test subjects: rerun it with aggregation metrics.
   If none does: test output-side calibration (per-subject centring / ridge map on embeddings) from saved embeddings.
3. Prepare sub-13…32 (`serialised_competition`) with the prep job (CPU) and re-evaluate zero-shot on 28 test subjects.
4. Novelty check before any claim (include searches for critiques/controls). Never announce, only report.
Limits: <= 6 h GPU in a row without a report entry; keep 5 h weekly reserve; nothing paid; nothing sent outside.
Kaggle credentials come from the session proxy: export KAGGLE_USERNAME=joaquimmanjijung and any dummy KAGGLE_KEY.

## 2026-11-19 07:15 UTC — repeated observations on unseen subjects: negative so far
`results/phase3_agg_v1` (3 seeds, timing-free single-word model, retrieval over 250 words, balanced top-1, chance 0.4 %):
- unseen subjects, 1 observation: 0.7–0.8 % (noise control 0.3–0.5 %); 10 observations: 1.5 / 0.0 / 0.5 % by seed → no reliable gain.
- in-subject (held-out sub-0 recordings): 1.5–2.5 % with 1 observation, 1.2–4.7 % with 10 → also unstable.
- BatchNorm recalibration on 10 min: −0.06 / +0.71 / +0.69 points at 50 words (top-10). Negligible.
Reading: with one training subject and a 3.6 M model, the timing-free signal on a new subject is too weak for
aggregation to help. Next lever tested: number of training subjects (v4 `single_multi`, then a scaling run once
sub-13…32 are prepared).

## 2026-11-19 09:30 UTC — calibration works modestly, scaling and aggregation do not
- 10 min supervised calibration (fine-tune all weights, 100 steps): +4.4 points at 50 words [CI +2.6, +5.9], 7/8 subjects;
  +1.5 at 250 words, 8/8. 1 min hurts. BatchNorm-only: nothing. See `reports/phase3.md`.
- More training subjects: no reliable gain on disjoint text; the A+B improvement was stimulus sharing (caught before reporting).
- Launched v6: calibration curve 5/10/20 min, steps 30/100/300, shuffled-label control, aggregation on calibrated models.
- Kaggle GPU used so far: ~9.5 h (v4 2.5 h, scale 1.9 h, agg 0.5 h, earlier ~4.6 h).

## 2026-11-19 11:05 UTC — functional alignment launched
- Novelty check of the calibration result logged (PARTIALLY published; see novelty_log).
- v6 (calibration curve 5/10/20 min + shuffled-label control) still running on `innervoice-xsub` since 09:08.
- New idea, launched as `innervoice-xsub-align` (commit 89b72fe): label-free functional alignment. Every subject heard
  the same book as sub-0 (Sherlock1), so a 306x306 ridge map new-subject MEG → sub-0 MEG is fitted on the first K
  minutes of the shared audio and the sub-0 decoder is applied to the mapped signal. Compared in the same run with
  supervised fine-tuning, align + fine-tune, and the shuffled-label control. Sub-0's Sherlock1 MEG is never trained on.
- To do when both finish: save under results/phase3_calibration_v2 and results/phase3_alignment_v1, paired gains
  with bootstrap CIs, update phase3.md; novelty check of functional alignment for MEG word decoding before any claim.

## 2026-11-19 11:20 UTC — calibration curve
- 5 / 10 / 20 min: +2.0 / +4.45 / +5.4 points at 50 words (8/8 subjects at 10 and 20 min); diminishing returns.
- Shuffled-label control falls to chance (20.5 %): the gain needs the true pairing. v1 result replicated.
- Kaggle GPU used so far: ~11.6 h. Alignment run still in progress.

## 2026-11-19 12:40 UTC — alignment matches supervised calibration
- Label-free alignment to sub-0 on shared audio: +3.6 / +4.3 / +5.9 points (5/10/20 min); align+fine-tune 20 min +7.0 → 33.3 %.
- Launched v8 controls (`innervoice-xsub-align`, version 2): random pairing (alignshuf) and covariance-only (coral).
- Kaggle GPU used so far: ~13.3 h.
