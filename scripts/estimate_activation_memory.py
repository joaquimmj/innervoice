"""Count bytes of activations saved for backward by the MEG-XL criss-cross encoder.

Runs on CPU with random weights: saved-activation size depends only on shapes and dtypes,
not on weight values or device. Usage:
    python scripts/estimate_activation_memory.py /path/to/MEG-XL [--out results/x.json]
"""
import argparse, json, sys
import torch

p = argparse.ArgumentParser()
p.add_argument("megxl_root")
p.add_argument("--out")
args = p.parse_args()
sys.path.insert(0, args.megxl_root)
from brainstorm.models.attentional.attn import SpatialTemporalEncoder  # noqa: E402

C, D, DEPTH, HEADS = 306, 512, 8, 8  # train_criss_cross_multi_50hz_med.yaml, 306 Elekta channels
TOKENS_PER_S = 50 / 12  # 50 Hz input, BioCodec downsample ratio 12


def saved_bytes(enc, t_prime, dtype, ckpt):
    enc.gradient_checkpointing = ckpt
    seen, total = set(), 0

    def pack(t):
        nonlocal total
        key = (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
        if key not in seen:
            seen.add(key)
            total += key[1]
        return t

    x = torch.randn(1, C, t_prime, D, requires_grad=True)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        with torch.autocast("cpu", dtype=dtype, enabled=dtype != torch.float32):
            enc(x)
    return total


enc = SpatialTemporalEncoder(dim=D, depth=DEPTH, heads=HEADS).train()
n_params = sum(p.numel() for p in enc.parameters())
rows = []
for dtype in (torch.float32, torch.bfloat16):
    for ckpt in (False,):
        # weights are also "saved" but constant: take the slope between context sizes
        b4, b8, b12 = (saved_bytes(enc, t, dtype, ckpt) for t in (4, 8, 12))
        s1, s2 = (b8 - b4) / (C * 4), (b12 - b8) / (C * 4)
        # self-check: linear in time tokens. Caveat: on GPU, flash/mem-efficient SDPA keeps it linear too.
        assert abs(s1 - s2) / max(s1, s2) < 0.15, (s1, s2)
        bpt = s2
        for secs in (15, 30, 60, 150):
            tp = int(secs * TOKENS_PER_S)
            full = bpt * C * tp
            # grad checkpointing keeps each layer's input (C*T'*D, fp32 residual stream)
            # plus one layer's activations during recompute; computed, not hooked (shortcut:
            # non-reentrant checkpoint hides its tensors from outer hooks; measure on GPU in Phase 1)
            ckpt_b = DEPTH * C * tp * D * 4 + full / DEPTH
            rows.append(dict(dtype=str(dtype).split(".")[-1], context_s=secs, time_tokens=tp,
                             saved_GiB=round(full / 2**30, 2), saved_GiB_grad_ckpt=round(ckpt_b / 2**30, 2)))
res = dict(encoder_params=n_params, channels=C, rows=rows)
print(json.dumps(res, indent=1))
if args.out:
    json.dump(res, open(args.out, "w"), indent=1)
