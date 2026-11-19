# Novelty log

## 2026-11-18 — Phase 0
Hosts reachable: arXiv API (export.arxiv.org), GitHub, web search. Blocked: huggingface.co, kaggle.com, openneuro.org,
zenodo.org, biorxiv.org, neural-processing-lab.github.io. Abstracts only; no full text.

Web queries: "PNPL 2026 LibriBrain competition results winners"; "MEG-XL neural-processing-lab foundation model word
decoding"; "open-vocabulary MEG word decoding contrastive 2026 arXiv"; "2609.03231 PNPL ... new subject track";
"PNPL Competition 2026 Broad track leaderboard winners"; "Défossez 2023 ... successors 2025 2026 cross-subject";
"MEG word decoding language model prior sentence context beam search non-invasive 2026"; "Integrating Language Models
into Listened and Imagined Speech Decoding from MEG"; "d'Ascoli Nature Communications 2025 ..."; "few-shot subject
adaptation MEG foundation model unseen subject calibration minutes word decoding 2026"; "arXiv 2605.08075 ... Zenodo";
"bioRxiv 2025.10.13.682161"; "Jhilal 2026 imagined speech vowels"; "arXiv 2602.18253"; "OpenNeuro ds003626 / ds004196
/ ds005170"; "zenodo 23895809"; "LibriBrain100 ... MEG-XL 10 minutes"; "Csaky Towards decoding inner speech ...";
"MEG foundation model open weights 2026".
arXiv API: `abs:MEG AND abs:word AND abs:decoding`; `abs:LibriBrain`; `abs:MEG AND abs:"foundation model"`;
`abs:EEG AND abs:"open-vocabulary" AND abs:word`; `abs:"inner speech" AND (abs:MEG OR abs:EEG) AND abs:dataset`.
Repos read: MEG-XL, Chisco, BrainOmni, rohit-ravi2/libribrain-2026, kuotunyu/pnpl2026-meg-word-decoding.

Closest prior work: 2609.31997 (Maryland, contrastive + LM beam search, listened/imagined, small vocab);
d'Ascoli et al. 2025 (D-SigLIP word decoding incl. unseen words — vocab/new-subject details UNVERIFIED);
MEG-XL + PNPL Broad (new subject, 10–40 min, closed 50 words); HDND 2609.24095; 2610.08229; 2608.01481;
SA-LBC 2611.16257 (few-shot latent calibration, image retrieval).
Verdict: PARTIALLY done — see `phase0.md` §A. Re-check PNPL winner write-ups after 2026-12-01.

## 2026-11-19 — check of the "transcript timing explains contextual word decoding" finding → ALREADY PUBLISHED
Read directly: arXiv abstract + HTML full text of 2609.40359 (scan, not line by line). Via sub-agent: SimpleB2T README
and notebook, rohit-ravi2/libribrain-2026 docs, search snippets for the rest (marked).
- **Jayalath & Parker Jones, "Removing Timing Shortcuts Improves Non-Invasive Brain-to-Text", arXiv 2609.40359,
  2026-09-30** (Oxford PNPL; code github.com/neural-processing-lab/SimpleB2T). Same mechanism (overlapping
  word-locked windows reveal inter-word intervals → word duration → identity), same controls in spirit (synthetic
  signal with no brain data 22.0 % vs real MEG 22.3 % top-1 balanced @50; timing-only input 22.9 %; independent
  windows 5.8 %), reproduced on the 32 other LibriBrain100 subjects (App. A.3) and other datasets (A.4). Their fix:
  decode each window independently, then aggregate repeated observations + LLM prior (WER 36.6 % with 5 observations
  per word, subject 0). Stated future work: fewer observations; imagined speech.
- rohit-ravi2/libribrain-2026 (public draft, not a paper): no-MEG timing classifier 0.632 top-10 balanced @50.
- Related, snippets only: arXiv 2605.24524 (length shortcut in MEG-to-audio retrieval), "On Length Bias in
  EEG-to-Text Decoding" (NeurIPS 2026), Jo et al. 2405.06459.
Verdict: our Phase 2 timing result is an **independent replication**, not a new finding. Possibly not in their paper
(to verify line by line before saying so): top-10 @50/@250 figures for the no-brain model, and the constant-shift vs
independent-random-time dissociation. Neither justifies a paper on its own.
Process failure: the Phase 0 search (2026-11-18) missed 2609.40359; queries were about vocabulary size and
calibration, none about controls/shortcuts. From now on each novelty check also searches for critiques and controls.
