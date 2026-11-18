# Results (master table)

| Run | Phase | What | Key number | Reproduce |
|---|---|---|---|---|
| phase0_memory_estimate | 0 | MEG-XL encoder saved activations, CPU, random weights | 57.0 GiB fp32 @150 s; 7.1 GiB 16-bit+ckpt (computed) | `python scripts/estimate_activation_memory.py <MEG-XL>` |
