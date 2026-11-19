# Results (master table)

| Run | Phase | What | Key number | Reproduce |
|---|---|---|---|---|
| phase0_memory_estimate | 0 | MEG-XL encoder saved activations, CPU, random weights | 57.0 GiB fp32 @150 s; 7.1 GiB 16-bit+ckpt (computed) | `python scripts/estimate_activation_memory.py <MEG-XL>` |
| phase1_memory_probe | 1 | MEG-XL full fine-tune step, 150 s context, 2×T4, fp16 + grad ckpt, head on GPU1 | 8.8 + 6.0 GiB peak, 5.7 s/step (vs README ≥80 GB) | `python scripts/kaggle_run.py kaggle/memory_probe` |
| phase2_xsub_v1/v2 | 2 | Zero-shot on 8 unseen subjects, balanced top-10 @50 / @250 words | timing-only (no MEG) 75.8 / 43.6; MEG sentence-context 61 / 26; MEG single-word 26.8 / 6.2; chance 20 / 4 | `python scripts/kaggle_run.py kaggle/xsub` |
| phase3_calibration_v1 | 3 | Timing-free model, 10 min supervised calibration on unseen subjects (rec B, 8 subjects, 3 seeds) | +4.42 pts @50 [95% CI +2.64, +5.92] → 31.0 %; +1.47 @250 [+1.08, +1.92]; 1 min: −1.9 | `python scripts/kaggle_run.py kaggle/xsub` at commit 29c5c1e |
| phase3_agg_v1 | 3 | Repeated observations, zero-shot, 250 words, balanced top-1 | n=1 0.7–0.8 %, n=10 0.0–1.5 % (chance 0.4) — negative | commit cc6fad8 |
| phase3_scaling_v1 | 3 | +0/4/10/20 extra training subjects, rec B @50 | 26.4 / 26.2 / 28.3 / 25.8 — no reliable gain | commit c632f9a |
| phase3_calibration_v2 | 3 | Calibration curve, fine-tune 100 steps, rec B, 8 unseen subjects, 3 seeds | 5 / 10 / 20 min: +2.04 / +4.45 / +5.41 pts @50; shuffled labels: −5.87 (chance) | commit 82c714e |
| phase3_alignment_v1 | 3 | Label-free ridge alignment to sub-0 on shared audio, rec B, 8 unseen subjects, 3 seeds | 5 / 10 / 20 min: +3.59 / +4.32 / +5.94 @50; align+fine-tune 20 min +6.96 (33.3 %) — controls pending | commit 89b72fe |
| phase3_alignment_v2 | 3 | Alignment controls, 8 unseen subjects, 3 seeds, @50 | align − random pairing: +7.7 / +8.1 / +8.0 (5/10/20 min, 8/8); align − covariance-only: +4.1 / +4.5 / +5.8 (8/8) | commit f6d9558 |
| phase3_weak_v1 | 3 | Weak (600-step) decoder, 10 min, @50 | ridge align +4.46 (8/8); random pairing −3.50; Procrustes +0.10; supervised +1.76 | commit d83a868 |
| phase3_identity_v1 | 3 | Identity-shrinkage alignment, 10 min, @50 | λ 0.01 / 0.1 / 1 / 10: +3.59 / +2.56 / +0.02 / −0.12 (shrink-to-zero +4.32) | commit 0266b10 |
