"""Phase 1: peak VRAM and step time of one MEG-XL fine-tuning step on a 16 GB GPU.

Mirrors MEG-XL's evaluate_criss_cross_word_classification.training_step (real checkpoint, real
BioCodec tokenizer, word MLP on C*512 pooled features, contrastive loss to 1024-d targets, AdamW),
with random MEG input: memory and speed depend on shapes, not values. Writes memory_probe.json.
"""
import json, os, subprocess, sys, time

MEGXL_COMMIT = "97da04171d22e8fe4822cfeb3047a57ecf86b318"
W = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytorch-lightning", "einops", "hydra-core", "omegaconf", "wandb", "mne", "h5py"], check=True)
src = f"{W}/MEG-XL"
if not os.path.isdir(src):
    subprocess.run(["git", "clone", "-q", "https://github.com/neural-processing-lab/MEG-XL.git", src], check=True)
    subprocess.run(["git", "-C", src, "checkout", "-q", MEGXL_COMMIT], check=True)
sys.path.insert(0, src)

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
import inspect
from brainstorm.evaluate_criss_cross_word_classification import (
    CrissCrossWordEmbeddingExtractor, load_tokenizer, map_raw_to_encoded_timesteps)
from brainstorm.models.criss_cross_transformer import CrissCrossTransformerModule


def load_criss_cross_model(path, tok, device):
    # Same as MEG-XL's loader, but drops hparams the pinned code does not know (v2 ckpt adds 'debug_nan_checks').
    ck = torch.load(path, map_location="cpu", weights_only=False)
    known = set(inspect.signature(CrissCrossTransformerModule.__init__).parameters)
    model = CrissCrossTransformerModule(tokenizer=tok, **{k: v for k, v in ck["hyper_parameters"].items() if k in known})
    sd = {k: v for k, v in ck["state_dict"].items() if "rope_embedding_layer.rotate" not in k}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected and all("rope" in k or k.startswith("tokenizer") for k in missing), (missing, unexpected)
    return model.to(device)

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
ckpt = hf_hub_download("pnpl/MEG-XL", "meg-xl-med-v2.ckpt", local_dir="/tmp/ckpt")  # outside /kaggle/working: not an output
C, SFREQ, WORD_S = 306, 50, 3.0


def probe(context_s, precision, grad_ckpt, freeze_encoder, steps=3):
    torch.cuda.empty_cache()
    tok = load_tokenizer(f"{src}/brainstorm/neuro_tokenizers/biocodec_ckpt.pt", dev)
    model = load_criss_cross_model(ckpt, tok, dev)
    model.train()
    if grad_ckpt:
        model.enable_gradient_checkpointing()
    mlp = CrissCrossWordEmbeddingExtractor(num_channels=C).to(dev).train()
    params = list(mlp.parameters()) + ([] if freeze_encoder else
                                       [p for n, p in model.named_parameters() if not n.startswith("tokenizer")])
    if freeze_encoder:
        model.requires_grad_(False)
    opt = torch.optim.AdamW(params, lr=1e-5)
    scaler = torch.amp.GradScaler(enabled=precision == "fp16")
    T = int(context_s * SFREQ)
    n_words = max(1, int(context_s / WORD_S))
    meg = torch.randn(1, C, T, device=dev) * 1e-12
    xyz, abc = torch.randn(1, C, 3, device=dev), F.normalize(torch.randn(1, C, 3, device=dev), dim=-1)
    types, mask = (torch.arange(C, device=dev) % 3 == 2).long()[None], torch.ones(1, C, device=dev, dtype=torch.bool)
    targets = torch.randn(n_words, 1024, device=dev)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(steps):
        torch.cuda.synchronize(); t0 = time.time()
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            with torch.set_grad_enabled(not freeze_encoder):
                feats = model(meg, xyz, abc, types, mask, apply_mask=False)["features"]
            embs = []
            for i in range(n_words):
                s, e = map_raw_to_encoded_timesteps(int(i * WORD_S * SFREQ), int((i + 1) * WORD_S * SFREQ))
                embs.append(mlp(feats[0, :, s:e, :]))
            z = F.normalize(torch.stack(embs).float(), dim=-1)
            loss = F.cross_entropy(z @ F.normalize(targets, dim=-1).T / 0.07, torch.arange(n_words, device=dev))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        torch.cuda.synchronize(); times.append(time.time() - t0)
    return dict(peak_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                step_s=round(sum(times[1:]) / max(1, len(times) - 1), 2), finite_loss=bool(torch.isfinite(loss)))


configs = []
for ctx in (15, 30, 60, 150):
    for prec in ("fp32", "fp16"):
        for gc in (False, True):
            configs.append(dict(context_s=ctx, precision=prec, grad_ckpt=gc, freeze_encoder=False))
configs += [dict(context_s=150, precision="bf16", grad_ckpt=True, freeze_encoder=False),
            dict(context_s=150, precision="fp16", grad_ckpt=False, freeze_encoder=True)]

gpu = torch.cuda.get_device_properties(0)
out = dict(gpu=gpu.name, vram_GiB=round(gpu.total_memory / 2**30, 1), torch=torch.__version__,
           megxl_commit=MEGXL_COMMIT, checkpoint="pnpl/MEG-XL meg-xl-med-v2.ckpt", rows=[])
for cfg in configs:
    try:
        r = probe(**cfg)
    except torch.OutOfMemoryError:
        r = dict(peak_GiB=None, oom=True)
    import gc as _gc; _gc.collect(); torch.cuda.empty_cache()
    out["rows"].append({**cfg, **r})
    print(out["rows"][-1], flush=True)
    json.dump(out, open(f"{W}/memory_probe.json", "w"), indent=1)
