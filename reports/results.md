# Results (master table)

| Run | Phase | What | Key number | Reproduce |
|---|---|---|---|---|
| phase0_memory_estimate | 0 | MEG-XL encoder saved activations, CPU, random weights | 57.0 GiB fp32 @150 s; 7.1 GiB 16-bit+ckpt (computed) | `python scripts/estimate_activation_memory.py <MEG-XL>` |
| phase1_memory_probe | 1 | MEG-XL full fine-tune step, 150 s context, 2×T4, fp16 + grad ckpt, head on GPU1 | 8.8 + 6.0 GiB peak, 5.7 s/step (vs README ≥80 GB) | `python scripts/kaggle_run.py kaggle/memory_probe` |
| phase2_xsub_v1/v2 | 2 | Zero-shot on 8 unseen subjects, balanced top-10 @50 / @250 words | timing-only (no MEG) 75.8 / 43.6; MEG sentence-context 61 / 26; MEG single-word 26.8 / 6.2; chance 20 / 4 | `python scripts/kaggle_run.py kaggle/xsub` |
