"""Run MEG-XL's own word-decoding fine-tuning script on 2 x T4 (Kaggle), on the cache built by libribrain_prep.

We call their `main(cfg)` unchanged and patch only what the hardware forces:
  1. word MLP (323 M params) lives on cuda:1, encoder on cuda:0            (memory, see reports/phase1.md)
  2. gradient checkpointing on                                              (their config flag)
  3. checkpoint loader tolerant to the extra hparam in meg-xl-med-v2.ckpt
  4. optional cap on the number of val/test segments (EVAL_MAX)             (time; 0 = full, as in the paper)
Everything else (data split, loss, optimiser, metrics, bf16 autocast) is their code at the pinned commit.
Edit the RUN dict to change the experiment; results land in /kaggle/working/run/.
"""
import glob, inspect, json, os, subprocess, sys, time

RUN = dict(train_pct=0.03, num_epochs=2, eval_max=150, seed=42, checkpoint="meg-xl-med-v2.ckpt")
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
os.chdir(src)  # their configs use paths relative to the repo root (tokenizer checkpoint)

import torch
from huggingface_hub import hf_hub_download
from hydra import compose, initialize_config_dir
import brainstorm.evaluate_criss_cross_word_classification as ev

assert torch.cuda.device_count() >= 2, "needs 2 GPUs (T4 x2)"
roots = glob.glob("/kaggle/input/**/meg_sensors_information.json", recursive=True)
assert roots, "prep output not mounted (kernel_sources: innervoice-libribrain-prep)"
data_root = os.path.dirname(roots[0])
cache_dir = os.path.join(os.path.dirname(data_root), "cache")
sessions = sorted({os.path.basename(p).split("_")[1] for p in glob.glob(f"{data_root}/serialized/*/derivatives/serialised/*.h5")},
                  key=lambda s: int(s.split("-")[1]))
ckpt = hf_hub_download("pnpl/MEG-XL", RUN["checkpoint"], local_dir="/tmp/ckpt")


# --- patch 1: word MLP on the second GPU, transparent to their code ---
class TwoGpuExtractor(ev.CrissCrossWordEmbeddingExtractor):
    def to(self, *a, **k):
        return super().to("cuda:1")

    def forward(self, features, subject_idx=None):
        return super().forward(features.to("cuda:1"), subject_idx).to(features.device)


ev.CrissCrossWordEmbeddingExtractor = TwoGpuExtractor


# --- patch 3: same as their loader, minus unknown hparams ---
def load_criss_cross_model(path, tok, device="cuda"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    known = set(inspect.signature(ev.CrissCrossTransformerModule.__init__).parameters)
    model = ev.CrissCrossTransformerModule(tokenizer=tok, **{k: v for k, v in ck["hyper_parameters"].items() if k in known})
    sd = {k: v for k, v in ck["state_dict"].items() if "rope_embedding_layer.rotate" not in k}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected and all("rope" in k or k.startswith("tokenizer") for k in missing), (missing, unexpected)
    return model.to(device).eval()


ev.load_criss_cross_model = load_criss_cross_model

# --- patch 4: cap evaluation loaders (fixed random subset, same for every epoch) ---
_DL, timing = ev.DataLoader, {}
if RUN["eval_max"]:
    def capped_loader(dataset, *a, shuffle=False, **k):
        if not shuffle and len(dataset) > RUN["eval_max"]:
            idx = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(RUN["seed"]))[:RUN["eval_max"]]
            dataset = torch.utils.data.Subset(dataset, sorted(idx.tolist()))
        return _DL(dataset, *a, shuffle=shuffle, **k)
    ev.DataLoader = capped_loader

# --- timing of training steps (their function, wrapped) ---
_step, step_times = ev.training_step, []
def timed_step(*a, **k):
    torch.cuda.synchronize(); t = time.time()
    out = _step(*a, **k)
    torch.cuda.synchronize(); step_times.append(time.time() - t)
    return out
ev.training_step = timed_step

with initialize_config_dir(config_dir=f"{src}/configs", version_base=None):
    cfg = compose("eval_criss_cross_word_classification_libribrain", overrides=[
        f"model.criss_cross_checkpoint={ckpt}", "model.use_gradient_checkpointing=true",
        f"data.root={data_root}", f"data.cache_dir={cache_dir}", f"data.all_sessions=[{','.join(sessions)}]",
        f"data.train_pct={RUN['train_pct']}", f"training.num_epochs={RUN['num_epochs']}", "training.num_workers=3",
        f"seed={RUN['seed']}", f"logging.save_dir={W}/run", "t5.cache_dir=/tmp/t5_cache"])

t0 = time.time()
metrics = ev.main.__wrapped__(cfg)
out = dict(run=RUN, megxl_commit=MEGXL_COMMIT, sessions=sessions, hours=round((time.time() - t0) / 3600, 2),
           train_step_s_median=round(sorted(step_times)[len(step_times) // 2], 2), train_steps=len(step_times),
           peak_GiB=[round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(2)],
           test_metrics_at_best_val={k: float(v) for k, v in metrics.items()})
json.dump(out, open(f"{W}/finetune_result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
# keep outputs small: drop model checkpoints (GBs), keep logs/metrics
for p in glob.glob(f"{W}/run/**/*", recursive=True):
    if os.path.isfile(p) and os.path.getsize(p) > 50 * 2**20:
        os.remove(p)
