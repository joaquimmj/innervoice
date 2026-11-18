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
