# Results (master table)

| Run | Phase | What | Key number | Reproduce |
|---|---|---|---|---|
| phase0_memory_estimate | 0 | MEG-XL encoder saved activations, CPU, random weights | 57.0 GiB fp32 @150 s; 7.1 GiB 16-bit+ckpt (computed) | `python scripts/estimate_activation_memory.py <MEG-XL>` |
| phase1_memory_probe | 1 | MEG-XL full fine-tune step, 150 s context, 2×T4, fp16 + grad ckpt, head on GPU1 | 8.8 + 6.0 GiB peak, 5.7 s/step (vs README ≥80 GB) | `python scripts/kaggle_run.py kaggle/memory_probe` |
