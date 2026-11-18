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

SEED = 0
TRAIN_TASKS = ["Sherlock2", "Sherlock3", "Sherlock4", "Sherlock5", "Sherlock6", "Sherlock7"]  # sub-0, != test book
N_HELDOUT_SUB0 = 2  # last sub-0 training recordings (sorted by name) kept out: in-subject sanity check
VAL_SUBJECTS = [1, 2, 3, 4]  # model selection (epoch, prior temperature); never used for the reported test numbers
STEPS, EVAL_EVERY, BATCH_SEGS, MAXW = 6000, 1000, 48, 12
LR, WD, N_NEG = 1e-3, 0.05, 4096
PRE, POST, FS = 1.0, 2.0, 50
D_CONV, D_EMB, DROPOUT = 128, 256, 0.3
RETRIEVAL_SIZES = [50, 250, 1000]

INPUT = os.environ.get("XSUB_INPUT", "/kaggle/input")
OUT = os.environ.get("XSUB_OUT", "/kaggle/working" if os.path.isdir("/kaggle/working") else ".")
SMOKE = os.environ.get("XSUB_SMOKE") == "1"  # tiny CPU run on synthetic data (tests/test_xsub.py)
if SMOKE:
    STEPS, EVAL_EVERY, BATCH_SEGS, N_NEG, RETRIEVAL_SIZES = 30, 15, 8, 16, [5, 10]
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
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
            words.append((w, s, int(float(r["sentenceidx"] or -1))))
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

train_counts = collections.Counter(w for r in train_recs for w, _, _ in r["words"])
vocab = [w for w, _ in train_counts.most_common()]  # train types by frequency
test_types = sorted({w for g in eval_groups.values() for r in g for w, _, _ in r["words"]} - set(vocab))
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


def segments(group):
    """Sentence segments of <= MAXW consecutive words: list of (rec, [word indices])."""
    segs = []
    for r in group:
        cur, last = [], None
        for i, (_, _, sent) in enumerate(r["words"]):
            if cur and (sent != last or len(cur) == MAXW):
                segs.append((r, cur)); cur = []
            cur.append(i); last = sent
        if cur:
            segs.append((r, cur))
    return segs


def batch(segs):
    """-> x (N,306,WIN) baseline-corrected, labels (N,), seg index (S,L) into N, pad (S,L)."""
    xs, ys, gi = [], [], torch.zeros(len(segs), MAXW, dtype=torch.long)
    pad = torch.ones(len(segs), MAXW, dtype=torch.bool)
    for s, (r, idx) in enumerate(segs):
        for j, i in enumerate(idx):
            w, st, _ = r["words"][i]
            gi[s, j], pad[s, j] = len(xs), False
            xs.append(r["x"][:, st:st + WIN]); ys.append(tid[w])
    x = torch.stack(xs).to(dev).float()
    x = (x - x[:, :, :ON].mean(-1, keepdim=True)).clamp(-5, 5)  # baseline = pre-onset second
    return x, torch.tensor(ys, device=dev), gi.to(dev), pad.to(dev)


# ---------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, d, k=5, p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(d, d, k, padding=k // 2), nn.BatchNorm1d(d), nn.GELU(), nn.Dropout(p),
                                 nn.Conv1d(d, d, k, padding=k // 2), nn.BatchNorm1d(d))

    def forward(self, x):
        return F.gelu(x + self.net(x))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        d, D = D_CONV, D_EMB
        self.cnn = nn.Sequential(nn.Conv1d(306, d, 1), nn.BatchNorm1d(d), nn.GELU(), Block(d), Block(d), Block(d),
                                 nn.Conv1d(d, d // 2, 5, 2, 2), nn.BatchNorm1d(d // 2), nn.GELU(),
                                 nn.Conv1d(d // 2, d // 2, 5, 2, 2), nn.BatchNorm1d(d // 2), nn.GELU(),
                                 nn.Flatten(), nn.Dropout(DROPOUT), nn.Linear(d // 2 * ((WIN + 3) // 4), D), nn.GELU())
        self.pos = nn.Embedding(MAXW, D)
        layer = nn.TransformerEncoderLayer(D, 4, 2 * D, dropout=0.1, batch_first=True, norm_first=True, activation="gelu")
        self.ctx, self.norm = nn.TransformerEncoder(layer, 4, enable_nested_tensor=False), nn.LayerNorm(D)
        self.proj = nn.Linear(D, 1024)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(20.0)))

    def forward(self, x, gi, pad):
        e = self.cnn(x)  # (N, D)
        h = e[gi] * (~pad)[..., None] + self.pos.weight[None]
        e = (e[gi] + self.norm(self.ctx(h, src_key_padding_mask=pad)))[~pad]  # (N, D), order of x preserved
        return F.normalize(self.proj(e).float(), dim=-1)


# ---------------------------------------------------------------- evaluation
def balanced_topk(scores, labels, k):
    """Macro average over classes of top-k accuracy. scores (N,C), labels (N,) in [0,C)."""
    hit = (scores.topk(min(k, scores.shape[1]), dim=1).indices == labels[:, None]).any(1).float()
    cls = torch.unique(labels)
    return float(torch.stack([hit[labels == c].mean() for c in cls]).mean())


@torch.no_grad()
def embed(model, group):
    model.eval()
    zs, ys = [], []
    segs = segments(group)
    for i in range(0, len(segs), 128):
        x, y, gi, pad = batch(segs[i:i + 128])
        with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            zs.append(model(x, gi, pad))
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
    keep = y >= n_train_types  # word types never present in the training text
    if keep.sum() >= 20:
        s, yy = z[keep] @ E[n_train_types:].T, y[keep] - n_train_types
        rank = (s > s.gather(1, yy[:, None])).sum(1).float() + 1
        out["unseen_words"] = dict(bal_top10=balanced_topk(s, yy, 10), chance_top10=min(1.0, 10 / len(test_types)), n_candidates=len(test_types),
                                   median_rank=float(rank.median()), chance_median_rank=(len(test_types) + 1) / 2, n=int(keep.sum()),
                                   top10=float((rank <= 10).float().mean()))
    return out


# ---------------------------------------------------------------- training
model = Decoder().to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=STEPS, pct_start=0.05)
scaler = torch.amp.GradScaler(enabled=dev == "cuda")
train_segs = segments(train_recs)
neg_p = torch.tensor([train_counts[w] for w in vocab], dtype=torch.float) ** 0.5
val_keys = [f"sub{s}" for s in VAL_SUBJECTS if f"sub{s}" in eval_groups]
history, best = [], dict(score=-1)
for step in range(1, STEPS + 1):
    model.train()
    x, y, gi, pad = batch([train_segs[i] for i in rng.integers(0, len(train_segs), BATCH_SEGS)])
    cand = torch.unique(torch.cat([y, torch.multinomial(neg_p, min(N_NEG, n_train_types)).to(dev)]))
    with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
        z = model(x, gi, pad)
    logits = model.logit_scale.exp().clamp(max=100) * z @ E[cand].T
    loss = F.cross_entropy(logits, torch.searchsorted(cand, y))
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update(); sched.step()
    if step % EVAL_EVERY == 0 or step == STEPS:
        gen = torch.Generator(device=dev).manual_seed(SEED)
        vals = {k: embed(model, eval_groups[k]) for k in val_keys + ["sub0_heldout"] if k in eval_groups}
        row = dict(step=step, loss=float(loss), minutes=round((time.time() - t_start) / 60, 1))
        n0 = RETRIEVAL_SIZES[0]
        for tau in (0.0, 0.5, 1.0):
            sc = [score(*vals[k], tau, gen).get(f"top{n0}", {}).get("bal_top10") for k in val_keys]
            sc = [s for s in sc if s is not None]
            row[f"val_top{n0}_tau{tau}"] = float(np.mean(sc)) if sc else None
            if sc and np.mean(sc) > best["score"]:
                best = dict(score=float(np.mean(sc)), step=step, tau=tau,
                            state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if "sub0_heldout" in vals:
            row["sub0_heldout"] = score(*vals["sub0_heldout"], 0.0, gen).get(f"top{n0}", {}).get("bal_top10")
        history.append(row)
        print(row, flush=True)

# ---------------------------------------------------------------- final scoring at the selected checkpoint
if "state" in best:
    model.load_state_dict(best["state"])
tau = best.get("tau", 0.0)
gen = torch.Generator(device=dev).manual_seed(SEED)
per_subject, emb_out = {}, {}
for k, g in eval_groups.items():
    z, y = embed(model, g)
    per_subject[k] = score(z, y, tau, gen)
    emb_out[k] = dict(z=z.half().cpu(), y=y.cpu())
test_keys = [k for k in per_subject if k.startswith("sub") and k != "sub0_heldout" and k not in val_keys]


def mean_over(keys, *path):
    v = []
    for k in keys:
        d = per_subject[k]
        for p in path:
            d = d.get(p) if isinstance(d, dict) else None
        if d is not None:
            v.append(d)
    return dict(mean=float(np.mean(v)), sd=float(np.std(v)), n_subjects=len(v)) if v else None


summary = {f"top{n}": {m: mean_over(test_keys, f"top{n}", m) for m in ("bal_top10", "bal_top1", "permuted_bal_top10")}
           for n in RETRIEVAL_SIZES}
summary["unseen_words"] = {m: mean_over(test_keys, "unseen_words", m) for m in ("bal_top10", "top10", "median_rank")}
result = dict(config=dict(seed=SEED, train_tasks=TRAIN_TASKS, val_subjects=VAL_SUBJECTS, steps=STEPS, window_s=[PRE, POST],
                          params_M=round(sum(p.numel() for p in model.parameters()) / 1e6, 2)),
              data=dict(train_words=sum(train_counts.values()), train_types=n_train_types, unseen_types=len(test_types)),
              selected=dict(step=best.get("step"), tau=tau, val_score=best["score"]),
              test_subjects=test_keys, summary_test_subjects=summary, per_subject=per_subject, history=history,
              minutes=round((time.time() - t_start) / 60, 1), device=dev)
json.dump(result, open(f"{OUT}/xsub_result.json", "w"), indent=1)
torch.save(dict(state=model.state_dict(), types=all_types, n_train_types=n_train_types, tau=tau), f"{OUT}/model.pt")
torch.save(emb_out, f"{OUT}/eval_embeddings.pt")
print(json.dumps(dict(selected=result["selected"], summary=summary), indent=1))
