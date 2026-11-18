# Rules (condensed — full brief: PARTIE 2 of the project prompt)
1. Live novelty check (arXiv, bioRxiv, Scholar, OpenReview, GitHub, HF, PNPL page) at Phase 0, before each new method, before any result is announced. Log in `reports/novelty_log.md`. Already done → STOP + sources.
2. Never send anything external (email, post, issue/PR on third-party repos, arXiv/OpenReview/competition submission). Drafts go in `drafts/`.
3. Git: author "Joaquim Manji-Jung" <228074480+joaquimmj@users.noreply.github.com>; no Co-Authored-By, no mention of Claude/AI in commits; work on `dev`, merge to `main` per validated step; short imperative English messages.
4. Public material: affiliation "Paris" only, no university name, no personal email; honest AI-usage section in the paper.
5. Nothing paid without Joaquim's explicit OK (provider, verified hourly price, hours, total, cap).
6. Rigor: test subjects never seen; no leakage; ≥3 seeds + 95% CIs; every number reproducible from committed script+config; negative results reported as negative.
7. Secrets only in env vars / Kaggle Secrets; never printed, logged or committed; `scripts/check_secrets.sh` runs as pre-commit hook.
8. Licences noted in `reports/resources.md` (LibriBrain CC-BY-NC-4.0: non-commercial).
9. Ethics: consented public datasets only; no re-identification; discuss misuse honestly.
10. Never claim without reading: unread figures are marked "to verify".
Reports to Joaquim: French, ≤5 lines, numbers. Code/papers: English. Stop points: end of each phase, any spend, any result announcement, any external draft, making the repo public. Max 6 h GPU in a row; keep 5 h/week reserve.

# Where we are
See `reports/journal.md` (latest entry) and `reports/phase0.md`.
