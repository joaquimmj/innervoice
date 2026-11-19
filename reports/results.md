# Results (master table)

| Run | Phase | What | Key number | Reproduce |
|---|---|---|---|---|
| phase0_memory_estimate | 0 | MEG-XL encoder saved activations, CPU, random weights | 57.0 GiB fp32 @150 s; 7.1 GiB 16-bit+ckpt (computed) | `python scripts/estimate_activation_memory.py <MEG-XL>` |
| phase1_memory_probe | 1 | MEG-XL full fine-tune step, 150 s context, 2×T4, fp16 + grad ckpt, head on GPU1 | 8.8 + 6.0 GiB peak, 5.7 s/step (vs README ≥80 GB) | `python scripts/kaggle_run.py kaggle/memory_probe` |
| phase2_xsub_v1/v2 | 2 | Zero-shot on 8 unseen subjects, balanced top-10 @50 / @250 words | timing-only (no MEG) 75.8 / 43.6; MEG sentence-context 61 / 26; MEG single-word 26.8 / 6.2; chance 20 / 4 | `python scripts/kaggle_run.py kaggle/xsub` |
| phase3_calibration_v1 | 3 | Timing-free model, 10 min supervised calibration on unseen subjects (rec B, 8 subjects, 3 seeds) | +4.42 pts @50 [95% CI +2.64, +5.92] → 31.0 %; +1.47 @250 [+1.08, +1.92]; 1 min: −1.9 | `python scripts/kaggle_run.py kaggle/xsub` at commit 29c5c1e |
| phase3_agg_v1 | 3 | Repeated observations, zero-shot, 250 words, balanced top-1 | n=1 0.7–0.8 %, n=10 0.0–1.5 % (chance 0.4) — negative | commit cc6fad8 |
| phase3_scaling_v1 | 3 | +0/4/10/20 extra training subjects, rec B @50 | 26.4 / 26.2 / 28.3 / 25.8 — no reliable gain | commit c632f9a |
