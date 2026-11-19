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

## 2026-11-19 — check of the timing-free cross-subject calibration result → PARTIALLY published
Sub-agent read: arXiv HTML of 2609.40359, 2609.03231, 2608.25204 (full text); SimpleB2T code; rohit-ravi2/libribrain-2026
(PREDICTIONS.md, paper_leakage_draft.md, README, RULES). Figures not read (numbers in figures UNVERIFIED).
- 2609.40359 (Oxford): cross-subject appendix (A.3) runs only the joint and synthetic decoders on other subjects; the
  timing-free decoder is never evaluated on unseen subjects (confirmed in `simpleb2t/experiments.py`). No calibration.
- 2608.25204 / 2609.03231 (LibriBrain100, PNPL competition): contextual MEG-XL, 40→20→10 min "minimal degradation";
  no timing control. Winners due 2026-12-01.
- rohit-ravi2/libribrain-2026 (public competition log, not a paper): sub-0-only single-window CNN on 4 unseen subjects
  0.211 → 0.266 with an input map; 306×306 input map on 10–30 min: 0.274 / 0.297 / 0.307 vs 0.266; BatchNorm
  recalibration on new subjects not detected; extra shallow subjects hurt; label-grouped *posterior* averaging helps
  (+0.07 with 2, +0.22 with 16 occurrences) on a contextual decoder.
Verdict: no paper reports a seeded, text-disjoint calibration curve with CIs for a strictly one-window decoder; the
qualitative pattern is already public in that repo. Our result is a quantified replication-plus-extension, not a first.
To reconcile before any write-up: (1) our embedding-averaging null vs their posterior-averaging gain; (2) report
against input-free controls (our noise control = 20 %), not only nominal chance; (3) re-check after 2026-12-01.
Not searched: bioRxiv, OpenReview, Scholar citations, Kaggle discussions.

## 2026-11-19 — check of label-free functional alignment to a reference subject → method published, positive MEG result not found
Sub-agent: arXiv abstracts (2309.00627, 2312.06467, 2607.19394, 2608.22420, 2609.30832), LibriBrain100 HTML grep,
rohit-ravi2/libribrain-2026 PREDICTIONS/REPORT/RUNS; others snippet only (marked).
- **Method is not new.** Ferrante et al., "Through their eyes" (arXiv 2309.00627, 2023; Imaging Neuroscience 2024):
  fMRI, decoder trained on one subject, others mapped onto it by ridge / hyperalignment on shared images; ridge best.
  Thual et al. (2312.06467, fMRI, FUGW). Heo et al. (2607.19394, ECoG speech, SRM shared space, decoder reused without
  retraining). MEG: Ravishankar, Toneva & Wehbe 2021 (pairwise maps on shared stimulus, denoising; snippet), Michalke &
  Rieger 2023 (MCCA, grasping; snippet), de Cheveigné MCCA (snippet). Haxby 2011, Chen 2015 SRM to cite.
- **Same recipe on LibriBrain, public null:** rohit-ravi2 P1b (2026-10-04): label-free ridge map new subject → sub-0 on
  shared Sherlock1 audio, frozen sub-0 decoder: 0.265–0.274 vs floor 0.263 → "NOT DETECTED" (their decoder: 0.498 on sub-0).
- No published positive result found for MEG word decoding with a timing-free decoder (our +3.6/+4.3/+5.9 at 5/10/20 min).
Verdict: method PARTIALLY published (fMRI/ECoG; MEG null in a repo). A positive, controlled MEG result + explanation of
the null would be a modest contribution. Required before any claim: mis-paired control (running, v8), covariance-only
control (running), Procrustes/CCA/SRM comparisons, alignment on a weaker decoder (to explain P1b), a multi-subject template.

## 2026-11-19 — check: does MEG-XL's own word decoder use the overlapping-window timing shortcut? → not tested anywhere found
Read in full (pdftotext): arXiv 2609.40359 (shortcut paper), 2609.03231 (PNPL 2026), 2608.25204 (LibriBrain100, sections
4.2–4.3, B.7). Read code: MEG-XL @97da041 (dataset grouping, training step, noise baseline). Searched: web (2 extended,
1 standard queries), rohit-ravi2/libribrain-2026 (grep shortcut/overlap/synthetic/MEG-XL).
- 2609.40359 tests the d'Ascoli decoder only (joint vs isolated, shared vs independent synthetic, timing-only), incl. 32
  LibriBrain100 subjects; MEG-XL appears only in its references.
- MEG-XL's inputs are 50 concatenated 3 s windows of consecutive words (overlapping in real time) with an encoder
  attending across all of them: the same structure. Its built-in noise baseline (commit 8a542ef) draws independent
  noise per input, i.e. the "isolated synthetic" condition, which cannot reveal the shortcut.
- Broad-track reference (2609.03231 Table 1): MEG-XL 42.96 % BAcc@10, competition vocab, new subjects fine-tuned on half of
  Sherlock1 ses-11. LibriBrain100 Fig. 3–4: MEG-XL on subjects 1–32, values in figures (not read: to verify).
- rohit-ravi2: timing twins of their own decoder, no MEG-XL shortcut test found.
Verdict: untested to our knowledge. Test: shared-synthetic vs real MEG × adjacent vs shuffled-non-overlapping grouping,
inside MEG-XL's own code (kaggle/megxl_shortcut). Not searched: bioRxiv, OpenReview, Scholar, Kaggle discussions.
