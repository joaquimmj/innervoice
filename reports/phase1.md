# Phase 1 — MEG-XL fine-tuning memory on free Kaggle GPUs (2026-11-18)

One real fine-tuning step (meg-xl-med-v2.ckpt, BioCodec tokenizer, 323 M-param word MLP, AdamW, batch 1, random MEG
input), Tesla T4 14.6 GiB. Script: `kaggle/memory_probe/memory_probe.py`. Data: `results/phase1_memory_probe/`.
Memory only: **accuracy not yet reproduced**.

| Context | Precision | Grad ckpt | Head on | Peak GPU0 / GPU1 (GiB) | s/step |
|---|---|---|---|---|---|
| 15 s | fp32 | no | GPU0 | OOM | — |
| 15 s | fp16 | no | GPU0 | 7.3 | 0.47 |
| 30 s | fp16 | no | GPU0 | 13.4 | 1.07 |
| 30 s | fp16 | yes | GPU0 | 7.0 | 1.10 |
| 60 s | fp16 | yes | GPU0 | 8.5 | 2.21 |
| 90 s | fp16 | yes | GPU0 | 10.2 | 3.91 |
| 120 s | fp16 | yes | GPU0 | 11.9 | 5.28 |
| 150 s | fp16 | yes | GPU0 | OOM | — |
| 150 s | fp16 | no | GPU1 | OOM | — |
| **150 s** | **fp16** | **yes** | **GPU1** | **8.8 / 6.0** | **5.69** |
| 150 s | fp32 | yes | GPU1 | 12.6 / 7.3 | 17.5 |
| 150 s (frozen encoder, probe) | fp16 | — | GPU0 | 8.2 | 2.24 |

Findings
- Full 150 s fine-tuning runs on 2 × T4 (fp16 autocast + gradient checkpointing + word MLP on the second GPU).
  The README's 80 GB requirement is not a hard limit.
- ~5–6 GiB is the word MLP (306×512 → 2048 → 1024, 323 M params) and its AdamW state, independent of context.
- Single T4: full fine-tuning fits up to 120 s.
- Not measured: bf16 (emulated on T4), accuracy impact of fp16 vs the authors' bf16, time per epoch on real data.

Next: reproduce the paper's word-decoding accuracy with this configuration (GO/STOP criterion of Phase 1).
