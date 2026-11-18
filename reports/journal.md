# Journal

## 2026-11-18
- Done: repo scaffold; secret scanner + pre-commit hook; `kaggle_run.py` + hello-GPU job (untested on Kaggle);
  novelty check (abstract level); measured MEG-XL encoder activation memory on CPU.
- Results: 80 GB = activations of 191k tokens/sample (57 GiB fp32 at 150 s); est. 7 GiB with 16-bit + checkpointing.
- Problems: no Kaggle/HF secrets; huggingface.co, kaggle.com, zenodo.org, openneuro.org blocked; GitHub repo not yet
  created/attached, so this was committed locally only.
- Kaggle GPU hours used: 0.
- Next: run hello-GPU, LibriBrain inventory, check Zenodo 23895809, then Phase 1 memory probe on T4.
