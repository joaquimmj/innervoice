# nervegear

Cross-subject MEG word decoding with minimal calibration.

Goal: decode words from MEG for a never-seen person, with ≤ 10 minutes of calibration and an open vocabulary,
built on the open MEG-XL foundation model. Author: Joaquim Manji-Jung, Paris. Status: Phase 0 (live checks).

- `reports/phase0.md` — novelty, data, memory analysis, GO/STOP
- `reports/results.md` — every reported number and the command that reproduces it
- `scripts/kaggle_run.py kaggle/<job>` — run a GPU job on Kaggle and fetch its outputs

Setup: `pip install -r requirements.txt && git config core.hooksPath .githooks && pytest -q tests`
