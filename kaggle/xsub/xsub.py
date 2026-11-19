"""Cross-subject open-vocabulary word decoding on LibriBrain100: train on subject 0, test on never-seen subjects.

Train text (sub-0, TRAIN_TASKS) and test text (other subjects, Sherlock1) are different books, so there is no
stimulus leakage and "unseen words" (types absent from the train text) are a true open-vocabulary test.
Model: small word-window CNN + sentence transformer (architecture after rohit-ravi2/libribrain-2026, MIT, the
best documented zero-shot cross-subject recipe of PNPL 2026), trained contrastively against frozen T5-large
word-type embeddings (targets as in MEG-XL / d'Ascoli et al.). Any vocabulary can be ranked at test time.
Inputs: outputs of kaggle/libribrain_prep (50 Hz caches + events). Output: xsub_result.json, model.pt, test embeddings.
"""
import collections, csv, glob, json, math, os, re, sys, time

import h5py, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

# (seed, segment mode). "sentence": words grouped by annotated sentence + position embedding (uses sentence
# boundaries from the transcript). "chunk": fixed runs of MAXW consecutive words, no position input: no text-side metadata.
# "single": one word per forward pass, no cross-word context (windows cannot be compared with each other).
# "timing": NO MEG at all; the model only sees annotation-derived word timing (gap to previous/next onset, duration)
#           with sentence segmentation. Measures how much of the score the transcript's rhythm explains by itself.
# v1 (3 seeds of sentence/chunk, results/phase2_xsub_v1) showed the 7 s shift control stays far above chance.
# v2 (results/phase2_xsub_v2): timing-only model beats every MEG model; context models read window timing (known
#     shortcut, arXiv 2609.40359). From here on only timing-free "single" models are used.
# "single_multi": like "single", but the first recording of each validation subject is added to the training set.
# Every "single*" run is followed by the calibration study (see calibration_study).
RUNS = [(0, "single"), (1, "single"), (2, "single"), (0, "single_multi"), (1, "single_multi"), (2, "single_multi")]
CAL_MINUTES = [1, 5, 10]
# (method, steps, lr). zero: no adaptation. bn: BatchNorm statistics re-estimated on the calibration MEG (no labels).
# input: bn + a 306x306 input mixing matrix (init identity) fitted on labelled calibration words. full: bn + all weights.
CAL_GRID = [("bn", 0, 0.0), ("input", 100, 1e-3), ("input", 300, 1e-3), ("full", 100, 1e-4), ("full", 300, 1e-4)]
TRAIN_TASKS = ["Sherlock2", "Sherlock3", "Sherlock4", "Sherlock5", "Sherlock6", "Sherlock7"]  # sub-0, != test book
N_HELDOUT_SUB0 = 2  # last sub-0 training recordings (sorted by name) kept out: in-subject sanity check
VAL_SUBJECTS = [1, 2, 3, 4]  # model selection (epoch, prior temperature); never used for the reported test numbers
STEPS, EVAL_EVERY, BATCH_SEGS, MAXW = 6000, 1000, 48, 12
LR, WD, N_NEG = 1e-3, 0.05, 4096
PRE, POST, FS = 1.0, 2.0, 50
D_CONV, D_EMB, DROPOUT = 128, 256, 0.3
RETRIEVAL_SIZES = [50, 250, 1000]
AGG_VOCAB, AGG_NS = 250, [1, 2, 5, 10]

INPUT = os.environ.get("XSUB_INPUT", "/kaggle/input")
OUT = os.environ.get("XSUB_OUT", "/kaggle/working" if os.path.isdir("/kaggle/working") else ".")
SMOKE = os.environ.get("XSUB_SMOKE") == "1"  # tiny CPU run on synthetic data (tests/test_xsub.py)
if SMOKE:
    STEPS, EVAL_EVERY, BATCH_SEGS, N_NEG, RETRIEVAL_SIZES = 30, 15, 8, 16, [5, 10]
    RUNS, CAL_MINUTES = [(0, "sentence"), (0, "single"), (0, "single_multi"), (0, "timing")], [1]
    AGG_VOCAB, AGG_NS = 10, [1, 2]
    CAL_GRID = [("bn", 0, 0.0), ("input", 3, 1e-3), ("full", 3, 1e-4)]
dev = "cuda" if torch.cuda.is_available() else "cpu"
WIN, ON = int((PRE + POST) * FS), int(PRE * FS)
t_start = time.time()


# ---------------------------------------------------------------- data
def load_recordings():
    """Every cached recording -> robustly z-scored fp16 MEG + its word table."""
    recs = []
    for cache in sorted(glob.glob(f"{INPUT}/**/cache/*.h5", recursive=True)):
        m = re.match(r"sub-(\d+)_(ses-\d+)_task-(\w+?)_(run-\d+)_preproc", os.path.basename(cache))
        sub, ses, task, run = int(m.group(1)), m.group(2), m.group(3), m.group(4)
        if sub == 0 and task not in TRAIN_TASKS:
            continue
        ev = glob.glob(f"{os.path.dirname(os.path.dirname(cache))}/*/serialized/{task}/derivatives/events/sub-{sub}_{ses}_task-{task}_{run}_events.tsv")
        if not ev:
            continue
        with h5py.File(cache, "r") as f:
            x = f["data"][:].astype(np.float32)  # (306, n) Tesla
        med = np.median(x, 1, keepdims=True)
        q75, q25 = np.percentile(x, [75, 25], axis=1, keepdims=True)
        x = np.clip((x - med) / np.maximum(q75 - q25, 1e-30), -10, 10)  # per-recording, label-free statistics
        words = []
        for r in csv.DictReader(open(ev[0]), delimiter="\t"):
            w = (r["segment"] or "").strip().lower()
            if r["kind"] != "word" or not w or w == "silence":
                continue
            s = int(round(float(r["timemeg"]) * FS)) - ON
            if s < 0 or s + WIN > x.shape[1]:
                continue
            words.append((w, s, int(float(r["sentenceidx"] or -1)), float(r["duration"] or 0)))
        if words:
            recs.append(dict(sub=sub, name=os.path.basename(cache), x=torch.from_numpy(x).half(), words=words))
    return recs


recs = load_recordings()
assert recs, f"no recordings under {INPUT}"
_held = set(sorted(r["name"] for r in recs if r["sub"] == 0)[-N_HELDOUT_SUB0:])
is_heldout = lambda r: r["name"] in _held
train_recs = [r for r in recs if r["sub"] == 0 and not is_heldout(r)]
eval_groups = {"sub0_heldout": [r for r in recs if is_heldout(r)]}
for r in recs:
    if r["sub"] != 0:
        eval_groups.setdefault(f"sub{r['sub']}", []).append(r)
eval_groups = {k: v for k, v in eval_groups.items() if v}

train_counts = collections.Counter(w for r in train_recs for w, *_ in r["words"])
vocab = [w for w, _ in train_counts.most_common()]  # train types by frequency
test_types = sorted({w for g in eval_groups.values() for r in g for w, *_ in r["words"]} - set(vocab))
all_types = vocab + test_types
tid = {w: i for i, w in enumerate(all_types)}
n_train_types = len(vocab)
log_prior = torch.log(torch.tensor([train_counts[w] for w in vocab], dtype=torch.float) / sum(train_counts.values()))
print(f"recordings train={len(train_recs)} eval_groups={ {k: len(v) for k, v in eval_groups.items()} } "
      f"train_words={sum(train_counts.values())} train_types={n_train_types} unseen_types={len(test_types)}", flush=True)


def word_type_embeddings(types):
    """Frozen T5-large encoder, layer 12, mean over the word's tokens (static word-type targets)."""
    if SMOKE:
        return F.normalize(torch.randn(len(types), 1024, generator=torch.Generator().manual_seed(1)), dim=-1)
    from transformers import T5EncoderModel, T5Tokenizer
    tok, t5 = T5Tokenizer.from_pretrained("t5-large"), T5EncoderModel.from_pretrained("t5-large").to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(types), 256):
            b = tok(types[i:i + 256], return_tensors="pt", padding=True).to(dev)
            h = t5(**b, output_hidden_states=True).hidden_states[12]
            m = b["attention_mask"][..., None].float()
            out.append(((h * m).sum(1) / m.sum(1)).float().cpu())
    del t5
    torch.cuda.empty_cache()
    return torch.cat(out)


E = word_type_embeddings(all_types)
E = F.normalize(E - E[:n_train_types].mean(0, keepdim=True), dim=-1).to(dev)  # centred with train types only


def segments(group, mode):
    """Segments of <= MAXW consecutive words (split at sentence ends in "sentence" mode): list of (rec, [word indices])."""
    segs, by_sentence, maxw = [], mode in ("sentence", "timing"), 1 if mode.startswith("single") else MAXW
    for r in group:
        cur, last = [], None
        for i, (_, _, sent, _) in enumerate(r["words"]):
            if cur and ((by_sentence and sent != last) or len(cur) == maxw):
                segs.append((r, cur)); cur = []
            cur.append(i); last = sent
        if cur:
            segs.append((r, cur))
    return segs


ctrl_rng = np.random.default_rng(123)


def batch(segs, control=None):
    """-> x (N,306,WIN) baseline-corrected, labels (N,), seg index (S,L) into N, pad (S,L), timing (N,3).
    Controls: "shift" = MEG taken 7 s later for every word (misaligned, relative timing of windows kept);
    "jitter" = each word's MEG taken at an independent random time (misaligned, relative timing destroyed);
    "noise" = MEG replaced by white noise."""
    xs, ys, tf, gi = [], [], [], torch.zeros(len(segs), MAXW, dtype=torch.long)
    pad = torch.ones(len(segs), MAXW, dtype=torch.bool)
    for s, (r, idx) in enumerate(segs):
        for j, i in enumerate(idx):
            w, st, _, dur = r["words"][i]
            gi[s, j], pad[s, j] = len(xs), False
            prev = (st - r["words"][i - 1][1]) / FS if i > 0 else 5.0
            nxt = (r["words"][i + 1][1] - st) / FS if i + 1 < len(r["words"]) else 5.0
            tf.append((min(prev, 5.0), min(nxt, 5.0), min(dur, 5.0)))
            if control == "shift":
                st = (st + 7 * FS) % (r["x"].shape[1] - WIN)
            elif control == "jitter":
                st = int(ctrl_rng.integers(0, r["x"].shape[1] - WIN))
            xs.append(r["x"][:, st:st + WIN]); ys.append(tid[w])
    x = torch.stack(xs).to(dev).float()
    if control == "noise":
        x = torch.randn_like(x)
    x = (x - x[:, :, :ON].mean(-1, keepdim=True)).clamp(-5, 5)  # baseline = pre-onset second
    return x, torch.tensor(ys, device=dev), gi.to(dev), pad.to(dev), torch.log1p(torch.tensor(tf, device=dev))


# ---------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, d, k=5, p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(d, d, k, padding=k // 2), nn.BatchNorm1d(d), nn.GELU(), nn.Dropout(p),
                                 nn.Conv1d(d, d, k, padding=k // 2), nn.BatchNorm1d(d))

    def forward(self, x):
        return F.gelu(x + self.net(x))


class Decoder(nn.Module):
    def __init__(self, use_pos, timing_only=False):
        super().__init__()
        d, D, self.use_pos, self.timing_only = D_CONV, D_EMB, use_pos, timing_only
        self.A = None  # optional per-subject input mixing matrix (calibration)
        self.tim = nn.Sequential(nn.Linear(3, D), nn.GELU(), nn.Linear(D, D), nn.GELU())
        self.cnn = nn.Sequential(nn.Conv1d(306, d, 1), nn.BatchNorm1d(d), nn.GELU(), Block(d), Block(d), Block(d),
                                 nn.Conv1d(d, d // 2, 5, 2, 2), nn.BatchNorm1d(d // 2), nn.GELU(),
                                 nn.Conv1d(d // 2, d // 2, 5, 2, 2), nn.BatchNorm1d(d // 2), nn.GELU(),
                                 nn.Flatten(), nn.Dropout(DROPOUT), nn.Linear(d // 2 * ((WIN + 3) // 4), D), nn.GELU())
        self.pos = nn.Embedding(MAXW, D)
        layer = nn.TransformerEncoderLayer(D, 4, 2 * D, dropout=0.1, batch_first=True, norm_first=True, activation="gelu")
        self.ctx, self.norm = nn.TransformerEncoder(layer, 4, enable_nested_tensor=False), nn.LayerNorm(D)
        self.proj = nn.Linear(D, 1024)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(20.0)))

    def forward(self, x, gi, pad, tf):
        if self.A is not None:
            x = torch.einsum("ij,bjt->bit", self.A, x)
        e = self.tim(tf) if self.timing_only else self.cnn(x)  # (N, D)
        h = e[gi] * (~pad)[..., None] + (self.pos.weight[None] if self.use_pos else 0)
        e = (e[gi] + self.norm(self.ctx(h, src_key_padding_mask=pad)))[~pad]  # (N, D), order of x preserved
        return F.normalize(self.proj(e).float(), dim=-1)


# ---------------------------------------------------------------- evaluation
def balanced_topk(scores, labels, k):
    """Macro average over classes of top-k accuracy. scores (N,C), labels (N,) in [0,C)."""
    hit = (scores.topk(min(k, scores.shape[1]), dim=1).indices == labels[:, None]).any(1).float()
    cls = torch.unique(labels)
    return float(torch.stack([hit[labels == c].mean() for c in cls]).mean())


@torch.no_grad()
def embed(model, group, mode, control=None, segs=None):
    model.eval()
    zs, ys = [], []
    segs = segments(group, mode) if segs is None else segs
    for i in range(0, len(segs), 128):
        x, y, gi, pad, tf = batch(segs[i:i + 128], control)
        with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            zs.append(model(x, gi, pad, tf))
        ys.append(y)
    return torch.cat(zs), torch.cat(ys)


def score(z, y, tau, gen):
    """All metrics for one subject. Retrieval set = N most frequent train types; only words in the set are scored."""
    out, scale = {"n_words": int(len(y))}, 20.0
    for n in RETRIEVAL_SIZES:
        keep = y < n
        if keep.sum() < 20:
            continue
        s = scale * z[keep] @ E[:n].T - tau * log_prior[:n].to(dev)
        yy = y[keep]
        rank = (s > s.gather(1, yy[:, None])).sum(1).float() + 1
        out[f"top{n}"] = dict(bal_top10=balanced_topk(s, yy, 10), bal_top1=balanced_topk(s, yy, 1), chance_top10=min(1.0, 10 / n),
                              median_rank=float(rank.median()), n=int(keep.sum()),
                              permuted_bal_top10=balanced_topk(s, yy[torch.randperm(len(yy), generator=gen, device=yy.device)], 10))
    # Repeated observations (after SimpleB2T, arXiv 2609.40359): average the embeddings of n distinct occurrences of
    # the same word, then retrieve among the AGG_VOCAB most frequent training words. Groups are disjoint.
    keep = y < AGG_VOCAB
    if keep.sum() >= 20:
        zz, yy = z[keep].float(), y[keep]
        perm = torch.randperm(len(yy), generator=gen, device=yy.device)
        zz, yy = zz[perm], yy[perm]
        order = torch.argsort(yy, stable=True)
        zz, yy = zz[order], yy[order]
        start = torch.searchsorted(yy, yy)  # index of the first occurrence of each word
        within = torch.arange(len(yy), device=yy.device) - start
        for n in AGG_NS:
            grp = within // n
            full = (torch.bincount(yy, minlength=AGG_VOCAB)[yy] // n) > grp  # drop incomplete groups
            if full.sum() < n * 10:
                continue
            gid = (yy * 100000 + grp)[full]
            uniq, inv = torch.unique(gid, return_inverse=True)
            zm = torch.zeros(len(uniq), zz.shape[1], device=zz.device).index_add_(0, inv, zz[full]) / n
            ym = uniq // 100000
            sc = scale * zm @ E[:AGG_VOCAB].T - tau * log_prior[:AGG_VOCAB].to(dev)
            out[f"agg{n}"] = dict(bal_top10=balanced_topk(sc, ym, 10), bal_top1=balanced_topk(sc, ym, 1),
                                  top1=float((sc.argmax(1) == ym).float().mean()), n_groups=int(len(ym)),
                                  n_word_types=int(len(torch.unique(ym))), vocab=AGG_VOCAB)
    keep = y >= n_train_types  # word types never present in the training text
    if keep.sum() >= 20:
        s, yy = z[keep] @ E[n_train_types:].T, y[keep] - n_train_types
        rank = (s > s.gather(1, yy[:, None])).sum(1).float() + 1
        out["unseen_words"] = dict(bal_top10=balanced_topk(s, yy, 10), chance_top10=min(1.0, 10 / len(test_types)), n_candidates=len(test_types),
                                   median_rank=float(rank.median()), chance_median_rank=(len(test_types) + 1) / 2, n=int(keep.sum()),
                                   top10=float((rank <= 10).float().mean()))
    return out


# ---------------------------------------------------------------- one run
neg_p = torch.tensor([train_counts[w] for w in vocab], dtype=torch.float) ** 0.5
val_keys = [f"sub{s}" for s in VAL_SUBJECTS if f"sub{s}" in eval_groups]
test_keys = [k for k in eval_groups if k != "sub0_heldout" and k not in val_keys]


# ---------------------------------------------------------------- calibration study (timing-free models only)
# Per subject: recording A (first session) provides the first K minutes as calibration; recording B (other session,
# different part of the book) is the evaluation set, identical for every K and method.
subject_recs = {k: sorted(g, key=lambda r: r["name"]) for k, g in eval_groups.items() if k != "sub0_heldout" and len(g) >= 2}
multi_recs = [subject_recs[k][0] for k in subject_recs if k in [f"sub{s}" for s in VAL_SUBJECTS]]


def ngram_overlap(a, b, n=5):
    grams = lambda r: {tuple(w for w, *_ in r["words"][i:i + n]) for i in range(len(r["words"]) - n + 1)}
    ga, gb = grams(a), grams(b)
    return len(ga & gb) / max(1, len(gb))


def adapt_and_score(state, cal, ev, method, steps, lr, tau, seed):
    model = Decoder(use_pos=False).to(dev)
    model.load_state_dict(state)
    model.eval()
    if method != "zero":
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.reset_running_stats(); m.momentum = None; m.train()  # cumulative average over the calibration set
        with torch.no_grad():
            for i in range(0, len(cal), 256):
                x, _, gi, pad, tf = batch(cal[i:i + 256])
                model(x, gi, pad, tf)
        model.eval()
    if steps:
        if method == "input":
            model.requires_grad_(False)
            model.A = nn.Parameter(torch.eye(306, device=dev))
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
        r = np.random.default_rng(seed)
        for _ in range(steps):  # eval mode: fixed BatchNorm statistics, no dropout
            x, y, gi, pad, tf = batch([cal[i] for i in r.integers(0, len(cal), min(256, len(cal)))])
            cand = torch.unique(torch.cat([y, torch.multinomial(neg_p, min(N_NEG, n_train_types)).to(dev)]))
            loss = F.cross_entropy(model.logit_scale.exp().clamp(max=100) * model(x, gi, pad, tf) @ E[cand].T,
                                   torch.searchsorted(cand, y))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    sc = score(*embed(model, None, None, segs=ev), tau, torch.Generator(device=dev).manual_seed(seed))
    out = {k: v["bal_top10"] for k, v in sc.items() if isinstance(v, dict)}
    out.update({f"{k}_top1": v["bal_top1"] for k, v in sc.items() if isinstance(v, dict) and k.startswith("agg")})
    return out


def calibration_study(state, tau, seed, grid):
    table = {}
    for k, (ra, rb) in ((k, v[:2]) for k, v in subject_recs.items()):
        ev = [(rb, [i]) for i in range(len(rb["words"]))]
        row = {"zero": adapt_and_score(state, [], ev, "zero", 0, 0.0, tau, seed), "text_overlap_5gram": ngram_overlap(ra, rb)}
        for minutes in CAL_MINUTES:
            cal = [(ra, [i]) for i, w in enumerate(ra["words"]) if w[1] + WIN <= minutes * 60 * FS]
            if len(cal) < 8:
                continue
            for method, steps, lr in grid:
                row[f"{method}_s{steps}_k{minutes}"] = dict(adapt_and_score(state, cal, ev, method, steps, lr, tau, seed), n_cal_words=len(cal))
        table[k] = row
        print("calib", seed, k, {a: (round(b.get(f"top{RETRIEVAL_SIZES[0]}", 0), 3) if isinstance(b, dict) else round(b, 3)) for a, b in row.items()}, flush=True)
    # per (method, K): hyper-parameters chosen on validation subjects, reported on test subjects
    n0, n1 = f"top{RETRIEVAL_SIZES[0]}", f"top{RETRIEVAL_SIZES[1]}"
    mean = lambda keys, name, n: float(np.mean([table[k][name][n] for k in keys if name in table.get(k, {}) and n in table[k][name]] or [np.nan]))
    cal_test = [k for k in test_keys if k in table]
    summary = {"zero": {n: dict(mean=mean(cal_test, "zero", n), sd=float(np.std([table[k]["zero"][n] for k in cal_test]))) for n in (n0, n1)}}
    for minutes in CAL_MINUTES:
        for method in sorted({g[0] for g in grid}):
            names = [f"{m}_s{st}_k{minutes}" for m, st, _ in grid if m == method]
            best_name = max(names, key=lambda nm: -1 if np.isnan(mean(val_keys, nm, n0)) else mean(val_keys, nm, n0))
            vals = {n: [table[k][best_name][n] - table[k]["zero"][n] for k in cal_test if best_name in table[k]] for n in (n0, n1)}
            summary[f"{method}_k{minutes}"] = dict(chosen=best_name, **{n: dict(mean=mean(cal_test, best_name, n),
                                                   gain_vs_zero_mean=float(np.mean(vals[n] or [np.nan])), gain_vs_zero_sd=float(np.std(vals[n] or [np.nan])),
                                                   n_subjects=len(vals[n])) for n in (n0, n1)})
    return dict(summary_test_subjects=summary, per_subject=table)


def run(seed, mode):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    t0 = time.time()
    model = Decoder(use_pos=mode in ("sentence", "timing"), timing_only=mode == "timing").to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=STEPS, pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    train_segs = segments(train_recs + (multi_recs if mode == "single_multi" else []), mode)
    history, best, n0 = [], dict(score=-1), RETRIEVAL_SIZES[0]
    for step in range(1, STEPS + 1):
        model.train()
        n_segs = BATCH_SEGS * (MAXW // 2 if mode.startswith("single") else 1)  # similar number of words per step in every mode
        x, y, gi, pad, tf = batch([train_segs[i] for i in rng.integers(0, len(train_segs), n_segs)])
        cand = torch.unique(torch.cat([y, torch.multinomial(neg_p, min(N_NEG, n_train_types)).to(dev)]))
        with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            z = model(x, gi, pad, tf)
        logits = model.logit_scale.exp().clamp(max=100) * z @ E[cand].T
        loss = F.cross_entropy(logits, torch.searchsorted(cand, y))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        if step % EVAL_EVERY == 0 or step == STEPS:
            gen = torch.Generator(device=dev).manual_seed(seed)
            vals = {k: embed(model, eval_groups[k], mode) for k in val_keys + ["sub0_heldout"] if k in eval_groups}
            row = dict(step=step, loss=float(loss), minutes=round((time.time() - t0) / 60, 1))
            for tau in (0.0, 0.5, 1.0):  # selection on validation subjects only
                sc = [score(*vals[k], tau, gen).get(f"top{n0}", {}).get("bal_top10") for k in val_keys]
                sc = [v for v in sc if v is not None]
                row[f"val_top{n0}_tau{tau}"] = float(np.mean(sc)) if sc else None
                if sc and np.mean(sc) > best["score"]:
                    best = dict(score=float(np.mean(sc)), step=step, tau=tau,
                                state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            if "sub0_heldout" in vals:
                row["sub0_heldout"] = score(*vals["sub0_heldout"], 0.0, gen).get(f"top{n0}", {}).get("bal_top10")
            history.append(row)
            print(seed, mode, row, flush=True)
    if "state" in best:
        model.load_state_dict(best["state"])
    tau, gen = best.get("tau", 0.0), torch.Generator(device=dev).manual_seed(seed)
    per_subject, emb_out, controls = {}, {}, {}
    for k, g in eval_groups.items():
        z, y = embed(model, g, mode)
        per_subject[k] = score(z, y, tau, gen)
        emb_out[k] = dict(z=z.half().cpu(), y=y.cpu())
    for control in ("shift", "jitter", "noise"):  # must fall to chance if the model reads word-locked brain activity
        controls[control] = {k: score(*embed(model, eval_groups[k], mode, control), tau, gen) for k in test_keys}

    def mean_over(table, *path):
        v = []
        for k in test_keys:
            d = table[k]
            for q in path:
                d = d.get(q) if isinstance(d, dict) else None
            if d is not None:
                v.append(d)
        return dict(mean=float(np.mean(v)), sd=float(np.std(v)), n_subjects=len(v)) if v else None

    def summarise(table):
        out = {f"top{n}": {m: mean_over(table, f"top{n}", m) for m in ("bal_top10", "bal_top1", "permuted_bal_top10")}
               for n in RETRIEVAL_SIZES}
        out["unseen_words"] = {m: mean_over(table, "unseen_words", m) for m in ("bal_top10", "top10", "median_rank")}
        for n in AGG_NS:
            out[f"agg{n}"] = {m: mean_over(table, f"agg{n}", m) for m in ("bal_top1", "bal_top10", "top1", "n_word_types")}
        return out

    tag = f"seed{seed}_{mode}"
    torch.save(dict(state=model.state_dict(), types=all_types, n_train_types=n_train_types, tau=tau, mode=mode), f"{OUT}/model_{tag}.pt")
    torch.save(emb_out, f"{OUT}/eval_embeddings_{tag}.pt")
    calibration = None
    if mode.startswith("single") and subject_recs:
        calibration = calibration_study(model.state_dict(), tau, seed, CAL_GRID if mode == "single" else CAL_GRID[:1])
    return dict(seed=seed, mode=mode, selected=dict(step=best.get("step"), tau=tau, val_score=best["score"]), calibration=calibration,
                summary_test_subjects=summarise(per_subject),
                controls_test_subjects={c: summarise(t) for c, t in controls.items()},
                per_subject=per_subject, history=history, minutes=round((time.time() - t0) / 60, 1))


result = dict(config=dict(train_tasks=TRAIN_TASKS, val_subjects=VAL_SUBJECTS, steps=STEPS, window_s=[PRE, POST],
                          params_M=round(sum(p.numel() for p in Decoder(True).parameters()) / 1e6, 2)),
              data=dict(train_words=sum(train_counts.values()), train_types=n_train_types, unseen_types=len(test_types),
                        chance_unseen_top10=min(1.0, 10 / max(1, len(test_types)))),
              test_subjects=test_keys, device=dev, runs=[])
for seed, mode in RUNS:
    result["runs"].append(run(seed, mode))
    result["minutes"] = round((time.time() - t_start) / 60, 1)
    json.dump(result, open(f"{OUT}/xsub_result.json", "w"), indent=1)
    r = result["runs"][-1]
    print(json.dumps(dict(seed=seed, mode=mode, selected=r["selected"], summary=r["summary_test_subjects"],
                          controls=r["controls_test_subjects"],
                          calibration=r["calibration"] and r["calibration"]["summary_test_subjects"]), indent=1), flush=True)
