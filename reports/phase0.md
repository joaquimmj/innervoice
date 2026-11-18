# Phase 0 — Live checks (2026-11-18) — PARTIAL, decision provisional

Status: done = A (novelty), C (origin of the 80 GB). Blocked = B (Kaggle), D (LibriBrain inventory): no secrets and
no network access to kaggle.com / huggingface.co in the session where this was written.

## A. Novelty (details: `novelty_log.md`)
Sources: arXiv API abstracts, GitHub READMEs, web-search snippets. No full text read; snippet-only items are marked.

| Component of Voie A | Already published? |
|---|---|
| Contrastive MEG → frozen word embeddings | **Yes.** MEG-XL itself fine-tunes with SigLIP to T5-large layer-12 embeddings (read in `configs/eval_..._libribrain.yaml`, `word_mlp.embed_dim: 1024`, `loss: SigLIP`). Also d'Ascoli et al. 2025 (snippet). |
| Retrieval at 250 words | **Yes**, MEG-XL eval already reports `retrieval_set_sizes: [50, 250]`. |
| LM prior / beam search for MEG words | **Yes**: arXiv 2609.31997 (Maryland, listened + imagined, small vocab), 2610.08229 (shallow fusion), PNPL Deep entries (+5 pts). |
| New subject with 10 min of data | **Yes, closed 50-word vocab**: PNPL 2026 Broad track. Winners not announced (due ~2026-12-01). |
| Open vocabulary (250–1000+, words unseen in training) **+ never-seen subject + ≤10 min + light adapter on MEG-XL** | **Not found.** |

Important counter-evidence (rohit-ravi2/libribrain-2026 README, read): zero-adaptation transfer from subject 0 (0.66
BalAcc@10) beat recipes that used each new subject's 10–40 min. An adapter must beat this baseline.

New since 14/11: MegNIST (arXiv 2611.21399, 2026-11-16, PNPL) — 1 subject, 12k trials of *imagined digits*, 306-ch MEG, public loader.

## B. Kaggle — NOT RUN
`scripts/kaggle_run.py` + `kaggle/hello_gpu/` ready and unit-tested; needs KAGGLE_USERNAME/KAGGLE_KEY (PARTIE 0.3/0.6).

## C. Where the 80 GB comes from (measured, CPU, random weights)
`scripts/estimate_activation_memory.py` → `results/phase0_memory_estimate/metrics.json`.
- Encoder: 21.0 M params (≈0.3 GB with Adam) → weights are not the problem.
- Input = 306 channels × 625 time tokens (150 s × 50 Hz / BioCodec stride 12) = **191k tokens** per sample, D=512, 8 layers.
- Attention already uses `F.scaled_dot_product_attention` (linear memory). Activations are ~19 D-vectors/token/layer
  (FF 4×D hidden, qkv, fp32 RMSNorm/RoPE casts).

| Context | fp32 | 16-bit autocast | 16-bit + grad checkpointing (computed) |
|---|---|---|---|
| 15 s | 5.7 GiB | 3.3 GiB | 0.7 GiB |
| 30 s | 11.4 | 6.7 | 1.4 |
| 60 s | 22.8 | 13.5 | 2.9 |
| 150 s | **57.0** | 33.6 | **7.1** |

Conclusion: the 80 GB is activations of the 191k-token sequence. The shipped fine-tune config has
`use_gradient_checkpointing: false`. With checkpointing + 16-bit, the encoder should fit in 16 GB at 150 s.
Caveats (to measure on GPU in Phase 1): BioCodec front-end, sensor embeddings and SigLIP head not counted;
README wording suggests 80 GB *with* checkpointing — if true, something outside the encoder dominates;
T4/P100 have no native bf16 → fp16 + GradScaler.

## D. LibriBrain inventory — NOT RUN (huggingface.co blocked here). From abstracts: LibriBrain ≈ 50 h of subject 0;
LibriBrain100 (arXiv 2608.25204) adds 32 subjects × ~40 min. Licence CC-BY-NC-4.0 (per 14/11 check, re-verify).

## E. Imagined-speech data
| Dataset | Status |
|---|---|
| Maryland (2605.08075) | Zenodo 23895809 "MEG recordings of listened and imagined music and speech" seen in search — licence/size **to verify**. 11 musicians, KIT. |
| bioRxiv 2025 (Csaky et al.) | access unverified; inner speech "largely unsuccessful". |
| Jhilal 2026 (2601.15909) | 21 subj, 3 vowels; no public release found. |
| Chisco | OpenNeuro ds005170, EEG, 5 subj, >900 min each; licence to verify. |
| ds003626 / ds004196 | EEG, 10 subj / 4 subj × 8 words; cross-subject imagined transfer at chance (2610.20954). |
| MegNIST (2611.21399) | MEG, 1 subject, imagined digits — usable for within-subject feasibility only. |

## Decision (provisional, pending B and D)
- **Voie A: GO, re-scoped.** The contrastive decoder alone is not novel. The claim must be: open-vocabulary
  (unseen words, 250–1000+) for a never-seen subject with ≤10 min, beating zero-shot transfer and MEG-XL fine-tuning.
  Phase 1 (16 GB) looks feasible on paper. Re-check PNPL winners on 2026-12-01 before Phase 3.
- **Voie B: HOLD.** First serious lead: Zenodo 23895809 (Maryland). Check licence/contents as soon as zenodo.org is reachable.
- GPU budget estimate: Phase 1 ≈ 8 h, Phase 2 ≈ 20 h, Phase 3 ≈ 30 h of Kaggle T4 (rough, unmeasured).
