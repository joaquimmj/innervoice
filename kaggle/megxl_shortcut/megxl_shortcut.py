"""Does MEG-XL's word decoder use the overlapping-window timing shortcut?

MEG-XL builds each 150 s input by concatenating the 3 s windows (onset -0.5 s .. +2.5 s) of 50 consecutive words.
Consecutive words are ~0.3 s apart, so neighbouring windows share most of their samples, shifted by the true
inter-word interval, and the encoder attends across all windows. arXiv 2609.40359 showed that this structure alone
(a synthetic signal with no brain information) reproduces most of the gain of jointly decoding words with the
d'Ascoli et al. decoder; MEG-XL itself was not tested there, and MEG-XL's own noise baseline replaces each input with
independent noise (no shared samples), so it cannot detect this shortcut.

Two factors, everything else is MEG-XL's code at the pinned commit (same patches as kaggle/finetune):
  SIGNAL = "real"  : MEG
           "synth" : a continuous AR(1) noise stream per recording, read at the true word windows (no brain data;
                     neighbouring windows share samples exactly as MEG windows do)
  ORDER  = "adjacent" : MEG-XL's grouping (50 consecutive words)
           "shuffled" : 50 words drawn at random from the same recording, no two windows overlapping in time
Session split (train ses-1..8, val ses-9, test ses-10 of every Sherlock book): no stimulus or sample is shared
between splits, so the shuffled grouping cannot leak test samples into training.
"""
import glob, inspect, json, os, subprocess, sys, time, zlib

SIGNAL = "synth"
ORDER = "adjacent"
TRAIN_PCT = 0.05
EPOCHS = 2
EVAL_MAX = 150  # val/test segments (50 words each), fixed random subset; 0 = all
SEED = 42
MEGXL_COMMIT = "97da04171d22e8fe4822cfeb3047a57ecf86b318"
W = "/kaggle/working"
os.environ["WANDB_MODE"] = "disabled"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytorch-lightning", "einops", "hydra-core", "omegaconf",
                "wandb", "mne", "h5py", "sentencepiece"], check=True)
src = "/tmp/MEG-XL"
if not os.path.isdir(src):
    subprocess.run(["git", "clone", "-q", "https://github.com/neural-processing-lab/MEG-XL.git", src], check=True)
    subprocess.run(["git", "-C", src, "checkout", "-q", MEGXL_COMMIT], check=True)
sys.path.insert(0, src)
os.chdir(src)

import numpy as np
import torch
from scipy.signal import lfilter
from huggingface_hub import hf_hub_download
from hydra import compose, initialize_config_dir
import brainstorm.evaluate_criss_cross_word_classification as ev
from brainstorm.data.libribrain_word_aligned_dataset import LibriBrainWordAlignedDataset as DS

assert torch.cuda.device_count() >= 2, "needs 2 GPUs (T4 x2)"
roots = glob.glob("/kaggle/input/**/meg_sensors_information.json", recursive=True)
assert roots, "prep output not mounted (kernel_sources: innervoice-libribrain-prep)"
data_root = os.path.dirname(roots[0])
cache_dir = os.path.join(os.path.dirname(data_root), "cache")
ckpt = hf_hub_download("pnpl/MEG-XL", "meg-xl-med-v2.ckpt", local_dir="/tmp/ckpt")


# --- hardware patches (identical to kaggle/finetune) ---
class TwoGpuExtractor(ev.CrissCrossWordEmbeddingExtractor):
    def to(self, *a, **k):
        return super().to("cuda:1")

    def forward(self, features, subject_idx=None):
        return super().forward(features.to("cuda:1"), subject_idx).to(features.device)


ev.CrissCrossWordEmbeddingExtractor = TwoGpuExtractor


def load_criss_cross_model(path, tok, device="cuda"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    known = set(inspect.signature(ev.CrissCrossTransformerModule.__init__).parameters)
    model = ev.CrissCrossTransformerModule(tokenizer=tok, **{k: v for k, v in ck["hyper_parameters"].items() if k in known})
    sd = {k: v for k, v in ck["state_dict"].items() if "rope_embedding_layer.rotate" not in k}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected and all("rope" in k or k.startswith("tokenizer") for k in missing), (missing, unexpected)
    return model.to(device).eval()


ev.load_criss_cross_model = load_criss_cross_model
_DL = ev.DataLoader
if EVAL_MAX:
    def capped_loader(dataset, *a, shuffle=False, **k):
        if not shuffle and len(dataset) > EVAL_MAX:
            idx = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(SEED))[:EVAL_MAX]
            dataset = torch.utils.data.Subset(dataset, sorted(idx.tolist()))
        return _DL(dataset, *a, shuffle=shuffle, **k)
    ev.DataLoader = capped_loader
# fp16 instead of their bf16 autocast: bf16 attention crashes on T4 ("unspecified launch failure"), fp16 ran in
# Phase 1. Dynamic loss scaling on the gradient only (logged loss values are unchanged); overflow -> step skipped.
_autocast, _clip, scale = torch.autocast, torch.nn.utils.clip_grad_norm_, dict(s=2.0 ** 12, good=0, skipped=0)


class ScaleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * scale["s"]


def unscale_and_clip(params, max_norm, *a, **k):
    params = [p for p in params if p.grad is not None]
    for p in params:
        p.grad.div_(scale["s"])
    if not all(torch.isfinite(p.grad).all() for p in params):
        for p in params:
            p.grad = None  # AdamW skips parameters without gradient
        scale.update(s=scale["s"] / 2, good=0, skipped=scale["skipped"] + 1)
        return torch.tensor(float("inf"))
    scale["good"] += 1
    if scale["good"] % 200 == 0:
        scale["s"] = min(scale["s"] * 2, 2.0 ** 16)
    return _clip(params, max_norm, *a, **k)


torch.nn.utils.clip_grad_norm_ = unscale_and_clip
_step, step_times = ev.training_step, []
def timed_step(*a, **k):
    torch.cuda.synchronize(); t = time.time()
    torch.autocast = lambda device_type, dtype=None, **kw: _autocast(
        device_type, dtype=torch.float16 if dtype == torch.bfloat16 else dtype, **kw)
    try:
        loss, *rest = _step(*a, **k)
    finally:
        torch.autocast = _autocast
    if loss.requires_grad:
        loss = ScaleGrad.apply(loss)
        torch.cuda.synchronize(); step_times.append(time.time() - t)
    return (loss, *rest)
ev.training_step = timed_step


# --- factor 1: synthetic signal at the true word windows ---
class SynthData:
    """Deterministic AR(1) noise stream with the recording's shape; any [:, a:b] read returns the same samples."""
    B = 2500

    def __init__(self, shape, seed):
        self.shape, self.seed = shape, seed

    def _block(self, i):
        w = np.random.default_rng([self.seed, i]).standard_normal((self.shape[0], self.B + 200))
        return lfilter([1.0], [1.0, -0.9], w, axis=1)[:, 200:].astype(np.float32)  # 200-sample burn-in: stationary

    def __getitem__(self, key):
        rows, cols = key
        a, b = cols.start, cols.stop
        x = np.concatenate([self._block(i) for i in range(a // self.B, (b - 1) // self.B + 1)], axis=1)
        return x[rows, a - (a // self.B) * self.B: b - (a // self.B) * self.B]


class SynthH5:
    def __init__(self, h, seed):
        self.h, self.attrs, self.data = h, h.attrs, SynthData(h["data"].shape, seed)

    def __getitem__(self, k):
        return self.data if k == "data" else self.h[k]

    def close(self):
        self.h.close()


if SIGNAL == "synth":
    _open = DS._open_file_handles
    def synth_handles(self):
        _open(self)
        self.file_handles = [SynthH5(h, zlib.crc32(str(r.get("raw_path", i)).encode()))
                             for i, (h, r) in enumerate(zip(self.file_handles, self.recordings))]
    DS._open_file_handles = synth_handles
else:
    assert SIGNAL == "real"


# --- factor 2: shuffled, non-overlapping groups ---
def shuffled_groups(self, events_df, recording_duration):
    words = []
    for _, row in events_df.iterrows():  # same filters as MEG-XL's _build_word_groups
        value = str(row["value"]).strip().lower()
        start = row["onset"] + self.window_onset_offset
        if value == "silence" or start < 0 or start + self.subsegment_duration > recording_duration:
            continue
        words.append(dict(word=value, onset=row["onset"], window_start=start, window_end=start + self.subsegment_duration))
    groups, cur = [], []
    for i in np.random.default_rng([SEED, len(words)]).permutation(len(words)):
        w = words[i]
        # shortcut: a word whose window overlaps one already in the group is dropped (a few %), fine for a control
        if any(abs(w["onset"] - o["onset"]) < self.subsegment_duration for o in cur):
            continue
        cur.append({**w, "subsegment_idx": len(cur)})
        if len(cur) == self.words_per_segment:
            groups.append(cur); cur = []
    if self.allow_incomplete_segments and cur:
        groups.append(cur)
    return groups


if ORDER == "shuffled":
    DS._build_word_groups = shuffled_groups
else:
    assert ORDER == "adjacent"

ses = lambda r: "[" + ",".join(f"ses-{i}" for i in r) + "]"
with initialize_config_dir(config_dir=f"{src}/configs", version_base=None):
    cfg = compose("eval_criss_cross_word_classification_libribrain", overrides=[
        f"model.criss_cross_checkpoint={ckpt}", "model.use_gradient_checkpointing=true",
        f"data.root={data_root}", f"data.cache_dir={cache_dir}", "data.use_hashed_split=false",
        f"data.train_sessions={ses(range(1, 9))}", f"data.val_sessions={ses([9])}", f"data.test_sessions={ses([10])}",
        f"data.train_pct={TRAIN_PCT}", f"training.num_epochs={EPOCHS}", "training.num_workers=3",
        f"seed={SEED}", f"logging.save_dir={W}/run", "t5.cache_dir=/tmp/t5_cache"])

t0 = time.time()
metrics = ev.main.__wrapped__(cfg)
out = dict(signal=SIGNAL, order=ORDER, train_pct=TRAIN_PCT, epochs=EPOCHS, eval_max=EVAL_MAX, seed=SEED,
           megxl_commit=MEGXL_COMMIT, hours=round((time.time() - t0) / 3600, 2),
           train_step_s_median=round(sorted(step_times)[len(step_times) // 2], 2) if step_times else None,
           train_steps=len(step_times), fp16_skipped_steps=scale["skipped"], final_loss_scale=scale["s"], peak_GiB=[round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(2)],
           test_metrics_at_best_val={k: float(v) for k, v in (metrics or {}).items()})
json.dump(out, open(f"{W}/shortcut_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
for p in glob.glob(f"{W}/run/**/*", recursive=True):  # drop checkpoints (GBs), keep logs/metrics
    if os.path.isfile(p) and os.path.getsize(p) > 50 * 2**20:
        os.remove(p)
