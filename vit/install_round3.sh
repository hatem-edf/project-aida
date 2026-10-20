#!/bin/bash
# Installs the ViT width-pruning code (round 3) into ~/work/project-aida/vit and verifies checksums.
# Usage: bash install_round3.sh   (run from anywhere; overwrites these 13 files)
set -euo pipefail
cd /share/home/HAF9D47L/work/project-aida/vit

cat > vit_model.py <<'__END_OF_FILE__'
"""Build and load bikit-models' codebrim-classif-balanced_ViT_s8_1, and width-pruned versions of it.

Width pruning removes whole attention heads and MLP hidden neurons, so a block may have fewer than
6 heads (qkv: 3*h*64 outputs, proj: h*64 inputs) and fewer than 1536 hidden neurons. The upstream
Attention.forward assumes h*64 == 384, so this module replaces it (class-wide) with a version that
works for any head count; for the unpruned model it computes exactly the same thing. Both forwards
also apply optional gates used during sparsity training (vit_gates.py):
    attn.head_gate    (h,)  multiplies each head's output before proj
    mlp.neuron_gate   (m,)  multiplies each hidden activation before fc2

A pruned checkpoint stores cp["prune_config"] = {"heads": [12 ints], "mlp_hidden": [12 ints]};
load_vit() shrinks the layers to those widths before loading the weights.
"""
import sys
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "bikit-models" / "models"))
import vistranet_1  # noqa: E402
from vistranet_1 import VisionTransformer, LinearClassifier, Ensemble  # noqa: E402

DEFAULT_CKPT = Path.home() / ".cache/bikit-models/codebrim-classif-balanced_ViT_s8_1.pth"
NUM_HEADS = {"vit_tiny": 3, "vit_small": 6}
HEAD_DIM = 64


# ---------------------------------------------------------------- forwards for any width
def _attention_forward(self, x):
    """Same math as vistranet_1.Attention.forward, for any number of heads; optional head gates."""
    B, N, _ = x.shape
    h = self.num_heads
    d = self.qkv.out_features // (3 * h)
    qkv = self.qkv(x).reshape(B, N, 3, h, d).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = self.attn_drop(attn.softmax(dim=-1))
    out = attn @ v                                            # (B, h, N, d)
    gate = getattr(self, "head_gate", None)
    if gate is not None:
        out = out * gate.to(out.dtype)[None, :, None, None]
    x = self.proj_drop(self.proj(out.transpose(1, 2).reshape(B, N, h * d)))
    return x, attn


def _mlp_forward(self, x):
    x = self.drop(self.act(self.fc1(x)))
    gate = getattr(self, "neuron_gate", None)
    if gate is not None:
        x = x * gate.to(x.dtype)
    return self.drop(self.fc2(x))


vistranet_1.Attention.forward = _attention_forward
vistranet_1.Mlp.forward = _mlp_forward


# ---------------------------------------------------------------- pruned architectures
def shrink_encoder(encoder, cfg):
    """Replace each block's layers by narrower ones: cfg = {"heads": [...], "mlp_hidden": [...]}."""
    dim = encoder.embed_dim
    for blk, h, m in zip(encoder.blocks, cfg["heads"], cfg["mlp_hidden"]):
        a = blk.attn
        a.num_heads = h
        a.qkv = nn.Linear(dim, 3 * h * HEAD_DIM, bias=a.qkv.bias is not None)
        a.proj = nn.Linear(h * HEAD_DIM, dim)
        blk.mlp.fc1 = nn.Linear(dim, m)
        blk.mlp.fc2 = nn.Linear(m, dim)
    return encoder


def encoder_dims(encoder):
    """[(heads, mlp_hidden)] per block: what FLOP counts need."""
    return [(b.attn.num_heads, b.mlp.fc1.out_features) for b in encoder.blocks]


def build_transform(img_size=224, mode="eval"):
    """mode="eval": resize to 256x256 (squash, aspect ratio NOT kept), center-crop 224.
        Reproduces the published result: 501/632 vs the authors' 503/632 (preproc_sweep.py).
    mode="demo": short side to 256 (aspect kept), center-crop 224, as in the authors'
        run.ipynb. Only for the golden test, whose reference outputs were made this way."""
    resize = ((256, 256) if mode == "eval" else 256)
    return transforms.Compose([
        transforms.Resize(resize, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def load_vit(cp_path=DEFAULT_CKPT):
    """Returns (model, class_names_in_output_order, checkpoint_dict). Strict loading."""
    cp = torch.load(cp_path, map_location="cpu", weights_only=False)
    a = cp["args"]
    enc_sd = cp["encoder"]
    embed_dim = enc_sd["cls_token"].shape[-1]                                   # 384
    depth = len({k.split(".")[1] for k in enc_sd if k.startswith("blocks.")})   # 12

    encoder = VisionTransformer(
        img_size=[cp["img_size"]], patch_size=a.patch_size, embed_dim=embed_dim,
        depth=depth, num_heads=NUM_HEADS[a.arch_encoder], mlp_ratio=4.0,
        qkv_bias=False, num_classes=1)   # num_classes=1: unused head, kept so keys match
    if "prune_config" in cp:
        shrink_encoder(encoder, cp["prune_config"])
    encoder.load_state_dict(enc_sd, strict=True)

    n_blocks = a.n_last_blocks                                                  # 4
    decoder = LinearClassifier(embed_dim=embed_dim * (n_blocks + int(a.avgpool_patchtokens)),
                               num_cls=cp["num_class"], activation=nn.Sigmoid())
    decoder.load_state_dict(cp["decoder"], strict=True)

    # cp["relation"] is an nn.Parameter, so it is registered and follows model.to(device)
    model = Ensemble(encoder, decoder, cp["relation"], n_last_blocks=n_blocks,
                     avgpool_patchtokens=a.avgpool_patchtokens)
    model.eval()
    class_names = [cp["cat_to_name"][i] for i in range(cp["num_class"])]
    return model, class_names, cp


def reorder_index(model_names, target_names):
    """Column index list such that probs[:, idx] follows target_names' order."""
    assert sorted(model_names) == sorted(target_names), (model_names, target_names)
    return [model_names.index(n) for n in target_names]
__END_OF_FILE__

cat > vit_gates.py <<'__END_OF_FILE__'
"""Gates for structured pruning of the bikit ViT, and the surgery that removes gated-off units.

The ViT has no BatchNorm, so (unlike Network Slimming on the ResNet) there is no per-channel scale
to read importance from. We add one: a scalar gate per attention head (72) and per MLP hidden
neuron (18,432), initialised at 1 so the model is unchanged. Sparsity training pushes them toward
0 with an L1 proximal step; pruning then removes the units with the smallest gates.

Removing units is exact for the kept ones: each gate multiplies its unit's output right before a
linear layer (proj for heads, fc2 for neurons), so it is folded into that layer's input columns.

  add_gates(model)                     # gates = 1, model output unchanged
  prox_step(model, lam_head, lam_mlp)  # after each optimizer step: g <- sign(g) max(|g| - lam, 0)
  plan = select_units(model, ratio)    # which heads / neurons to keep (global ranking + floors)
  pruned, cfg = prune_model(model, plan)
"""
import copy

import torch
from torch import nn

HEAD_DIM = 64


def add_gates(model, init=None):
    """init: optional {"head": [tensors], "mlp": [tensors]} to restore saved gate values."""
    for i, blk in enumerate(model.encoder.blocks):
        h, m = blk.attn.num_heads, blk.mlp.fc1.out_features
        dev = blk.mlp.fc1.weight.device
        g_h = init["head"][i].clone() if init else torch.ones(h)
        g_m = init["mlp"][i].clone() if init else torch.ones(m)
        blk.attn.head_gate = nn.Parameter(g_h.float().to(dev))
        blk.mlp.neuron_gate = nn.Parameter(g_m.float().to(dev))
    return model


def gate_params(model):
    heads = [b.attn.head_gate for b in model.encoder.blocks]
    mlps = [b.mlp.neuron_gate for b in model.encoder.blocks]
    return heads, mlps


def gate_values(model):
    heads, mlps = gate_params(model)
    return {"head": [g.detach().cpu().clone() for g in heads],
            "mlp": [g.detach().cpu().clone() for g in mlps]}


@torch.no_grad()
def prox_step(model, lam_head, lam_mlp):
    """Proximal L1 step (soft threshold): exact zeros, unlike a plain L1 gradient."""
    heads, mlps = gate_params(model)
    for gs, lam in ((heads, lam_head), (mlps, lam_mlp)):
        for g in gs:
            g.copy_(torch.sign(g) * torch.clamp(g.abs() - lam, min=0.0))


@torch.no_grad()
def gate_stats(model, thr=0.05):
    heads, mlps = gate_params(model)
    h = torch.cat([g.detach().abs().flatten().cpu() for g in heads])
    m = torch.cat([g.detach().abs().flatten().cpu() for g in mlps])
    return {"head_lt": float((h < thr).float().mean()), "mlp_lt": float((m < thr).float().mean()),
            "head_mean": float(h.mean()), "mlp_mean": float(m.mean()),
            "head_zero": int((h == 0).sum()), "mlp_zero": int((m == 0).sum())}


def _rank_keep(gates, n_remove, floor):
    """gates: list of 1-D |g| tensors per block. Remove the n_remove smallest overall, but never
    take a block below `floor(block_size)` units. Returns kept indices per block (ascending)."""
    order = sorted(((float(v), b, j) for b, g in enumerate(gates) for j, v in enumerate(g.tolist())),
                   key=lambda t: (t[0], t[1], t[2]))
    left = [g.numel() for g in gates]
    removed = [set() for _ in gates]
    for _, b, j in order:
        if n_remove == 0:
            break
        if left[b] - 1 >= floor(gates[b].numel()):
            removed[b].add(j); left[b] -= 1; n_remove -= 1
    return [torch.tensor([j for j in range(g.numel()) if j not in removed[b]], dtype=torch.long)
            for b, g in enumerate(gates)]


def select_units(model, ratio, ratio_heads=None, min_heads=1, min_mlp_frac=0.10):
    """Global ranking by |gate|, separately for heads and neurons (a head costs ~128 neurons of
    parameters, so they are not compared directly). Same ratio for both unless ratio_heads given."""
    heads, mlps = gate_params(model)
    hg = [g.detach().abs().cpu() for g in heads]
    mg = [g.detach().abs().cpu() for g in mlps]
    rh = ratio if ratio_heads is None else ratio_heads
    keep_h = _rank_keep(hg, int(round(rh * sum(g.numel() for g in hg))), lambda n: min_heads)
    keep_m = _rank_keep(mg, int(round(ratio * sum(g.numel() for g in mg))),
                        lambda n: max(1, int(round(min_mlp_frac * n))))
    return {"heads": keep_h, "mlp": keep_m}


@torch.no_grad()
def prune_model(model, plan):
    """Fold gates into the next layer, keep only the planned units, drop the gates.
    Returns (plain pruned model, prune_config)."""
    model = copy.deepcopy(model).cpu().eval()
    for blk, kh, km in zip(model.encoder.blocks, plan["heads"], plan["mlp"]):
        a, mlp = blk.attn, blk.mlp
        h, d, dim = a.num_heads, HEAD_DIM, a.proj.out_features

        # attention: fold head gates into proj input columns, then slice qkv rows / proj columns
        g = a.head_gate.detach().cpu()
        pw = a.proj.weight.detach().cpu().clone().view(dim, h, d) * g[None, :, None]
        rows = torch.cat([t * h * d + kh[:, None] * d + torch.arange(d)[None, :] for t in range(3)]).flatten()
        qkv = nn.Linear(a.qkv.in_features, rows.numel(), bias=a.qkv.bias is not None)
        qkv.weight.copy_(a.qkv.weight.detach().cpu()[rows])
        if a.qkv.bias is not None:
            qkv.bias.copy_(a.qkv.bias.detach().cpu()[rows])
        proj = nn.Linear(kh.numel() * d, dim)
        proj.weight.copy_(pw[:, kh, :].reshape(dim, -1))
        proj.bias.copy_(a.proj.bias.detach().cpu())
        a.qkv, a.proj, a.num_heads = qkv, proj, kh.numel()
        del a.head_gate

        # MLP: fold neuron gates into fc2 input columns, slice fc1 rows / fc2 columns
        g = mlp.neuron_gate.detach().cpu()
        fc1 = nn.Linear(mlp.fc1.in_features, km.numel())
        fc1.weight.copy_(mlp.fc1.weight.detach().cpu()[km])
        fc1.bias.copy_(mlp.fc1.bias.detach().cpu()[km])
        fc2 = nn.Linear(km.numel(), mlp.fc2.out_features)
        fc2.weight.copy_((mlp.fc2.weight.detach().cpu() * g[None, :])[:, km])
        fc2.bias.copy_(mlp.fc2.bias.detach().cpu())
        mlp.fc1, mlp.fc2 = fc1, fc2
        del mlp.neuron_gate
    cfg = {"heads": [b.attn.num_heads for b in model.encoder.blocks],
           "mlp_hidden": [b.mlp.fc1.out_features for b in model.encoder.blocks]}
    return model, cfg


def plain_state(module):
    """state_dict without gate entries (for saving in the original checkpoint layout)."""
    return {k: v.detach().cpu() for k, v in module.state_dict().items()
            if not k.endswith(("head_gate", "neuron_gate"))}
__END_OF_FILE__

cat > tome_vit.py <<'__END_OF_FILE__'
"""Token Merging (ToMe; Bolya et al., ICLR 2023) for the bikit ViT. No training needed.

Inside every block, right after attention, the r most similar pairs of tokens are
merged into one (bipartite soft matching on the attention keys, averaged over heads).
A merged token remembers how many patches it stands for ("size"):
  - merging is a size-weighted average,
  - attention adds log(size) to its logits ("proportional attention"), so a token
    that stands for 5 patches gets the attention weight of 5 patches.
The CLS token is never merged and stays at position 0, because the classifier reads
the CLS token of the last 4 blocks. Tokens go 785 -> 785 - 12 r over the 12 blocks.

    from tome_vit import apply_tome
    apply_tome(model, r=16)        # model from vit_model.load_vit(); r=0 restores the exact original
    model.encoder._tome["tokens"]  # tokens entering each block during the last forward pass
"""
import math
import types

import torch


def bipartite_soft_matching(metric, r):
    """Returns merge(x, mode) that removes r tokens from (B, N, C) tensors; token 0 is protected."""
    t = metric.shape[1]
    r = min(r, (t - 1) // 2)              # at most half the tokens, CLS excluded
    if r <= 0:
        return None
    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]        # alternate tokens into sets A and B
        scores = a @ b.transpose(-1, -2)                         # cosine similarity A x B
        scores[..., 0, :] = -math.inf                            # CLS (A[0]) is never merged
        node_max, node_idx = scores.max(dim=-1)                  # best partner in B for each A token
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r:, :].sort(dim=-2)[0]           # A tokens kept, original order (CLS first)
        src_idx = edge_idx[..., :r, :]                           # A tokens merged away
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    def merge(x, mode="sum"):
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        return torch.cat([unm, dst], dim=1)
    return merge


def _attn_forward(self, x, size):
    """Attention as in vit_model (any head count, optional head gates), plus proportional attention;
    returns the keys averaged over heads, used for matching."""
    B, N, _ = x.shape
    h = self.num_heads
    d = self.qkv.out_features // (3 * h)
    qkv = self.qkv(x).reshape(B, N, 3, h, d).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    attn = (q @ k.transpose(-2, -1)) * self.scale
    if size is not None:
        attn = attn + size.log()[:, None, None, :, 0]
    out = self.attn_drop(attn.softmax(dim=-1)) @ v
    gate = getattr(self, "head_gate", None)
    if gate is not None:
        out = out * gate.to(out.dtype)[None, :, None, None]
    x = self.proj_drop(self.proj(out.transpose(1, 2).reshape(B, N, h * d)))
    return x, k.mean(dim=1)


def _block_forward(self, x, return_attention=False):
    info = self._tome
    info["tokens"].append(x.shape[1])
    y, metric = self.attn.tome_forward(self.norm1(x), info["size"])
    x = x + self.drop_path(y)
    merge = bipartite_soft_matching(metric, info["r"][self._tome_idx])
    if merge is not None:
        size = info["size"] if info["size"] is not None else torch.ones_like(x[..., :1])
        x = merge(x * size) / merge(size)          # size-weighted average
        info["size"] = merge(size)
    return x + self.drop_path(self.mlp(self.norm2(x)))


def apply_tome(model, r):
    """Patch model.encoder in place. r: int (same for every block) or a list of 12 ints."""
    enc = model.encoder
    depth = len(enc.blocks)
    r_list = list(r) if isinstance(r, (list, tuple)) else [int(r)] * depth
    assert len(r_list) == depth
    if not hasattr(enc, "_tome"):
        enc._tome = {}
        orig_prepare = enc.prepare_tokens

        def prepare_tokens(x):                    # new image batch: reset sizes and the token trace
            enc._tome["size"], enc._tome["tokens"] = None, []
            return orig_prepare(x)
        enc.prepare_tokens = prepare_tokens
        for i, blk in enumerate(enc.blocks):
            blk._tome, blk._tome_idx = enc._tome, i
            blk.attn.tome_forward = types.MethodType(_attn_forward, blk.attn)
            blk.forward = types.MethodType(_block_forward, blk)
    enc._tome.update(r=r_list, size=None, tokens=[])
    return model


def vit_flops(tokens, r_list, dims=None, dim=384, head_dim=64, patch_in=3 * 8 * 8, n_patches=784,
              head_in=1536, n_cls=6):
    """FLOPs (1 multiply-add = 2) of one image, from the tokens entering each block.
    dims: [(heads, mlp_hidden)] per block (vit_model.encoder_dims); default: the unpruned ViT-S/8."""
    dims = dims or [(6, 1536)] * len(tokens)
    macs = n_patches * patch_in * dim + head_in * n_cls          # patch embedding + classifier
    for n_in, r, (h, m) in zip(tokens, r_list, dims):
        r_eff = max(0, min(r, (n_in - 1) // 2))
        n_out = n_in - r_eff
        inner = h * head_dim
        macs += n_in * dim * 3 * inner + n_in * inner * dim      # qkv + proj
        macs += 2 * n_in * n_in * inner                          # QK^T and attn @ V
        macs += 2 * n_out * dim * m                              # MLP after merging
        if r_eff:
            macs += ((n_in + 1) // 2) * (n_in // 2) * head_dim   # matching scores
    return 2 * macs
__END_OF_FILE__

cat > tome_sweep.py <<'__END_OF_FILE__'
#!/usr/bin/env python
"""Token Merging sweep on the released ViT: accuracy vs FLOPs vs real GPU throughput.

For each r (tokens merged per block), evaluates val and test, counts FLOPs from the
actual token trace, and times batches of 64 images on the GPU (images only, no data
loading). r=0 is the unmodified model (checked against the plain model at start-up).

  sbatch tome_sweep.sh
  python tome_sweep.py --ckpt runs/vit_X/best.pth --out results/tome_X
  -> results/tome_sweep.json and results/tome_sweep_probs.npz
"""
import argparse, copy, json, os, time

import numpy as np
import torch
from torch.utils.data import DataLoader

from vit_model import DEFAULT_CKPT, build_transform, encoder_dims, load_vit, reorder_index
from eval_vit import compute_metrics
from tome_vit import apply_tome, vit_flops

SHORT = {"NoDamage": "ND", "Crack": "Cr", "Efflorescence": "Ef", "Spalling": "Sp",
         "BarsExposed": "BE", "Rust": "Ru"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(DEFAULT_CKPT), help="any checkpoint in bikit-models layout")
    p.add_argument("--r", default="0,4,8,12,16,24,32,40,48,56,64")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default="results")
    return p.parse_args()


def load_split(split, tf, a):
    from bikit.datasets import BikitDataset
    ds = BikitDataset("codebrim-classif-balanced", split=split, transform=tf)
    X, Y = [], []
    for x, y in DataLoader(ds, batch_size=a.batch_size, num_workers=a.workers):
        X.append(x); Y.append(y)
    return torch.cat(X), torch.cat(Y).numpy(), ds.class_names


@torch.inference_mode()
def predict(model, X, dev, bs):
    return np.concatenate([model(X[i:i + bs].to(dev, non_blocking=True)).float().cpu().numpy()
                           for i in range(0, len(X), bs)])


@torch.inference_mode()
def throughput(model, X, dev, bs, warmup=5, iters=20):
    x = X[:bs].to(dev)
    for _ in range(warmup):
        model(x)
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    if dev == "cuda":
        torch.cuda.synchronize()
    return iters * bs / (time.perf_counter() - t0)


def main():
    a = parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}" + (f"  gpu={torch.cuda.get_device_name(0)}" if dev == "cuda" else ""))
    model, names, cp = load_vit(a.ckpt)
    print(f"checkpoint: {a.ckpt}")
    model.to(dev).eval()
    tf = build_transform(cp["img_size"], mode="eval")
    data = {s: load_split(s, tf, a) for s in ("val", "test")}
    label_names = data["test"][2]
    perm = reorder_index(names, label_names)
    print(f"val {len(data['val'][0])}  test {len(data['test'][0])}  (images kept in memory)")

    tm = apply_tome(copy.deepcopy(model), 0)
    xs = data["test"][0][:16]
    with torch.inference_mode():
        d = (tm(xs.to(dev)) - model(xs.to(dev))).abs().max().item()
    print(f"check: patched r=0 vs plain model, max |prob diff| = {d:.2e}  (must be ~0)\n")
    assert d < 1e-4

    rows, probs_out = [], {}
    for r in [int(v) for v in a.r.split(",")]:
        apply_tome(tm, r)
        res = {"r": r}
        for split in ("val", "test"):
            X, Y, _ = data[split]
            P = predict(tm, X, dev, a.batch_size)[:, perm]
            res[split] = compute_metrics(P, Y)
            if split == "test":
                probs_out[f"r{r}"] = P
        tm(data["test"][0][:1].to(dev))                       # token trace for one image
        res["tokens"] = list(tm.encoder._tome["tokens"])
        res["gflops"] = vit_flops(res["tokens"], tm.encoder._tome["r"], encoder_dims(tm.encoder)) / 1e9
        res["img_per_s"] = throughput(tm, data["test"][0], dev, a.batch_size)
        rows.append(res)
        t = res["test"]
        print(f"r={r:2d}  tokens {res['tokens'][0]}->{res['tokens'][-1]:<4} GFLOPs {res['gflops']:5.1f}  "
              f"{res['img_per_s']:6.0f} img/s   EMR val {100 * res['val']['exact_match_ratio']:5.2f}  "
              f"test {100 * t['exact_match_ratio']:5.2f}", flush=True)

    base = rows[0]
    print(f"\n{'r':>3} {'tokens':>11} {'GFLOPs':>7} {'saved':>6} {'speedup':>8} {'val EMR':>8} "
          f"{'test EMR':>9} {'F1':>6}   recall (test): " + " ".join(f"{SHORT[n]:>5}" for n in label_names))
    for res in rows:
        t = res["test"]
        print(f"{res['r']:>3} {res['tokens'][0]:>5}->{res['tokens'][-1]:<4} {res['gflops']:7.1f} "
              f"{100 * (1 - res['gflops'] / base['gflops']):5.1f}% {res['img_per_s'] / base['img_per_s']:7.2f}x "
              f"{100 * res['val']['exact_match_ratio']:8.2f} {100 * t['exact_match_ratio']:9.2f} "
              f"{100 * t['macro']['f1']:6.2f}                 "
              + " ".join(f"{100 * v:5.1f}" for v in t["per_class"]["recall"]))

    os.makedirs(a.out, exist_ok=True)
    json.dump({"class_names": label_names, "rows": rows}, open(os.path.join(a.out, "tome_sweep.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.out, "tome_sweep_probs.npz"), targets=data["test"][1], **probs_out)
    print(f"\nsaved -> {a.out}/tome_sweep.json (+ _probs.npz)")


if __name__ == "__main__":
    main()
__END_OF_FILE__

cat > train_vit.py <<'__END_OF_FILE__'
#!/usr/bin/env python
"""Full fine-tuning of the bikit ViT-S/8 on CODEBRIM (encoder + decoder; relation matrix frozen).

The model returns PROBABILITIES (sigmoid and relation post-processing are inside it), in the
checkpoint's class order (CSV order: Spalling before Efflorescence). BikitDataset labels follow
datasets.json order. So: permute the outputs to label order, then plain BCE on probabilities,
computed in fp32 outside autocast (BCELoss refuses to run under autocast).

Checkpoints are saved in the ORIGINAL bikit-models layout, so load_vit(), eval_vit.py --ckpt
and tome_sweep.py --ckpt read them unchanged.

Fail-fast guards (non-zero exit stops a Slurm dependency chain):
  exit 2  output folder already exists (never overwrite a run)
  exit 3  G1: val EMR of the loaded model != --expect-val-emr (loading/order/preprocessing wrong)
  exit 4  G0: another of the 720 class orders fits the labels better than ours (order wrong)
  exit 5  G2: val EMR below --guard-min-emr after --guard-epoch (training destroyed the features)
  exit 6  G3: non-finite loss

  python train_vit.py --cpu-smoke --out /tmp/vit_smoke          # login node, 100 images, CPU
  python train_vit.py --epochs 100 --seed 0 --expect-val-emr 76.30 --out runs/vit_ctrl_ep100_seed0
"""
import argparse, contextlib, itertools, json, math, os, random, sys, time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from vit_model import DEFAULT_CKPT, build_transform, load_vit, reorder_index
from eval_vit import compute_metrics

DATASET = "codebrim-classif-balanced"
NORM = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(DEFAULT_CKPT), help="starting checkpoint (bikit-models layout)")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr-enc", type=float, default=1e-5)
    p.add_argument("--lr-dec", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=3, help="linear warmup epochs")
    p.add_argument("--min-lr-ratio", type=float, default=0.01, help="cosine floor, fraction of base lr")
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--expect-val-emr", type=float, default=None, help="G1: required val EMR at start (%%)")
    p.add_argument("--guard-epoch", type=int, default=3)
    p.add_argument("--guard-min-emr", type=float, default=60.0)
    p.add_argument("--cpu-smoke", action="store_true", help="100 images, batch 4, 1 epoch, CPU, no guards")
    a = p.parse_args()
    if a.cpu_smoke:
        a.epochs, a.batch_size, a.workers = 1, 4, 0
    return a


def train_transform(img_size=224):
    """Same geometry as evaluation (256x256 squash, then 224), random crop + flips."""
    return T.Compose([T.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
                      T.RandomCrop(img_size), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                      T.ToTensor(), NORM])


def param_groups(module, lr, wd):
    decay, no_decay = [], []
    for n, p in module.named_parameters():
        if p.requires_grad:
            (no_decay if p.ndim == 1 or n in ("pos_embed", "cls_token") else decay).append(p)
    return [{"params": decay, "lr": lr, "base_lr": lr, "weight_decay": wd},
            {"params": no_decay, "lr": lr, "base_lr": lr, "weight_decay": 0.0}]


def lr_factor(ep, a):
    """ep is 1-indexed. Linear warmup, then cosine down to min_lr_ratio."""
    w = min(a.warmup, a.epochs)
    if ep <= w:
        return ep / max(1, w)
    t = (ep - w) / max(1, a.epochs - w)
    return a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * t))


def bce(probs, y):
    return F.binary_cross_entropy(probs.float().clamp(1e-6, 1 - 1e-6), y)


@torch.no_grad()
def evaluate(model, loader, perm, dev):
    """fp32, no autocast: comparable to the 76.30 / 79.27 references."""
    model.eval()
    P, Y = [], []
    for x, y in loader:
        P.append(model(x.to(dev, non_blocking=True)).float()[:, perm].cpu())
        Y.append(y)
    P, Y = torch.cat(P), torch.cat(Y)
    m = compute_metrics(P.numpy(), Y.numpy())
    return m["exact_match_ratio"], bce(P, Y).item(), m


def save_ckpt(model, cp, path, **extra):
    torch.save({**{k: v for k, v in cp.items() if k not in ("encoder", "decoder", "relation", "optimizer")},
                "encoder": {k: v.cpu() for k, v in model.encoder.state_dict().items()},
                "decoder": {k: v.cpu() for k, v in model.decoder.state_dict().items()},
                "relation": torch.nn.Parameter(model.relation.detach().cpu(), requires_grad=False),
                **extra}, path)


def stop(code, msg, a=None, hist=None):
    print(f"\nSTOP (exit {code}): {msg}", flush=True)
    if a is not None and hist is not None:
        json.dump({"args": vars(a), "history": hist, "stopped": msg},
                  open(os.path.join(a.out, "history.json"), "w"), indent=2)
    sys.exit(code)


def main():
    a = parse_args()
    if os.path.exists(a.out):
        stop(2, f"{a.out} already exists; choose another TAG or move it away")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() and not a.cpu_smoke else "cpu"
    amp = dev == "cuda" and torch.cuda.is_bf16_supported()
    print(f"device={dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else "")
          + f"  bf16 autocast={amp}  seed={a.seed}")

    from bikit.datasets import BikitDataset
    model, names, cp = load_vit(a.ckpt)
    model.to(dev)
    model.relation.requires_grad_(False)
    print(f"start: {a.ckpt}")

    dtr = BikitDataset(DATASET, split="train", transform=train_transform(cp["img_size"]), devel_mode=a.cpu_smoke)
    dva = BikitDataset(DATASET, split="val", transform=build_transform(cp["img_size"], mode="eval"),
                       devel_mode=a.cpu_smoke)
    ltr = DataLoader(dtr, batch_size=a.batch_size, shuffle=True, drop_last=True,
                     num_workers=a.workers, pin_memory=(dev == "cuda"), persistent_workers=a.workers > 0)
    lva = DataLoader(dva, batch_size=64, shuffle=False, num_workers=a.workers, pin_memory=(dev == "cuda"))
    perm = reorder_index(names, dtr.class_names)
    print(f"train {len(dtr)}  val {len(dva)}  model order {names}  ->  label order {dtr.class_names}")

    # G0: among all 720 possible class orders, ours must fit the labels best (on one val batch).
    # Only for trusted starts (--expect-val-emr given): a freshly pruned model predicts too poorly.
    if not a.cpu_smoke and a.expect_val_emr is not None:
        model.eval()
        x, y = next(iter(lva))
        with torch.no_grad():
            p = model(x.to(dev)).float().cpu()
        losses = {o: bce(p[:, list(o)], y).item() for o in itertools.permutations(range(len(perm)))}
        best_o = min(losses, key=losses.get)
        print(f"G0 label order: BCE {losses[tuple(perm)]:.4f} with our order, best of {len(losses)} orders "
              f"{losses[best_o]:.4f}, no reordering {losses[tuple(range(len(perm)))]:.4f}")
        if list(best_o) != perm:
            stop(4, f"G0: order {list(best_o)} fits better than ours {perm}; label order is wrong")

    emr0, vloss0, _ = evaluate(model, lva, perm, dev)
    print(f"before fine-tuning: val EMR {100 * emr0:.2f}  val BCE {vloss0:.4f}")
    if a.expect_val_emr is not None and not a.cpu_smoke and abs(100 * emr0 - a.expect_val_emr) > 0.2:
        stop(3, f"G1: val EMR {100 * emr0:.2f} != expected {a.expect_val_emr:.2f}")

    os.makedirs(a.out)
    groups = param_groups(model.encoder, a.lr_enc, a.wd) + param_groups(model.decoder, a.lr_dec, a.wd)
    opt = torch.optim.AdamW(groups)
    trainable = [p for g in groups for p in g["params"]]
    print(f"trainable parameters {sum(p.numel() for p in trainable):,}  "
          f"(lr encoder {a.lr_enc:g}, decoder {a.lr_dec:g}, wd {a.wd:g}, warmup {a.warmup}, cosine)\n")

    hist, best = [], -1.0
    for ep in range(1, a.epochs + 1):
        f = lr_factor(ep, a)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * f
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model.train()
        tot, n, t0 = 0.0, 0, time.time()
        for x, y in ltr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with (torch.autocast("cuda", dtype=torch.bfloat16) if amp else contextlib.nullcontext()):
                p = model(x)                                   # probabilities, model order
            loss = bce(p[:, perm], y)
            if not torch.isfinite(loss):
                stop(6, f"G3: non-finite loss at epoch {ep}", a, hist)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, a.clip)
            opt.step()
            tot += loss.item() * x.size(0); n += x.size(0)
        secs = time.time() - t0
        emr, vloss, m = evaluate(model, lva, perm, dev)
        mem = torch.cuda.max_memory_allocated() / 2 ** 30 if dev == "cuda" else 0.0
        rec = {"epoch": ep, "lr_enc": a.lr_enc * f, "loss": tot / max(1, n), "val_emr": emr,
               "val_loss": vloss, "val_recall": dict(zip(dtr.class_names, m["per_class"]["recall"])),
               "secs": round(secs, 1), "peak_mem_gb": round(mem, 2)}
        hist.append(rec)
        star = ""
        if emr > best:
            best = emr
            save_ckpt(model, cp, os.path.join(a.out, "best.pth"), ft_epoch=ep, val_emr=emr,
                      source_ckpt=a.ckpt, ft_args=vars(a))
            star = "  *"
        save_ckpt(model, cp, os.path.join(a.out, "last.pth"), ft_epoch=ep, val_emr=emr,
                  source_ckpt=a.ckpt, ft_args=vars(a))
        json.dump({"args": vars(a), "before": {"val_emr": emr0, "val_loss": vloss0}, "history": hist},
                  open(os.path.join(a.out, "history.json"), "w"), indent=2)
        print(f"ep{ep:4d}  lr {a.lr_enc * f:.2e}  loss {rec['loss']:.4f}  val_loss {vloss:.4f}  "
              f"val_emr {100 * emr:.2f}  ({secs:.0f}s, {mem:.1f} GB){star}", flush=True)
        if not a.cpu_smoke and ep == a.guard_epoch and 100 * emr < a.guard_min_emr:
            stop(5, f"G2: val EMR {100 * emr:.2f} < {a.guard_min_emr} after epoch {ep}", a, hist)

    print(f"\nval EMR {100 * emr0:.2f} -> {100 * best:.2f}  (best epoch "
          f"{max(hist, key=lambda r: r['val_emr'])['epoch']})")
    print("best ->", os.path.join(a.out, "best.pth"))


if __name__ == "__main__":
    main()
__END_OF_FILE__

cat > sparsity_vit.py <<'__END_OF_FILE__'
#!/usr/bin/env python
"""Sparsity training for structured ViT pruning: train with gates on every head and MLP neuron
and an L1 proximal step that drives unimportant gates to exactly 0 (the ViT version of the
Network Slimming phase we ran on the ResNet's BatchNorm gammas).

Gate dynamics: after each optimizer step, g <- sign(g) * max(|g| - lam, 0). A gate with no useful
gradient reaches 0 after ~1/lam steps; a useful one survives because its Adam step (up to ~lr_gate
per step) outweighs lam. So lr_gate must be well above lam (default 10x).

Snapshots: sXX.pth is saved the first epoch at which XX% of the MLP-neuron gates are below --thr
(head gates are logged alongside). prune_vit.py prunes ratio 0.XX from sXX.pth. Snapshots use the
original checkpoint layout plus cp["gates"], so load_vit() reads them; vit_gates.add_gates(model,
init=cp["gates"]) restores the gates.

  python sparsity_vit.py --cpu-smoke --out /tmp/vit_sp_smoke
  python sparsity_vit.py --ckpt runs/vit_ctrl_ep100_seed0/best.pth --expect-val-emr 77.60 --out runs/vit_sparsity
"""
import argparse, contextlib, itertools, json, os, random, sys, time

import numpy as np
import torch
from torch.utils.data import DataLoader

from vit_model import build_transform, load_vit, reorder_index
from vit_gates import add_gates, gate_params, gate_stats, gate_values, plain_state, prox_step
from train_vit import DATASET, bce, evaluate, param_groups, stop, train_transform


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="starting checkpoint (the fine-tuned control)")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr-enc", type=float, default=1e-5)
    p.add_argument("--lr-dec", type=float, default=1e-4)
    p.add_argument("--lr-gate", type=float, default=2e-3)
    p.add_argument("--lam-head", type=float, default=2e-4, help="L1 shrink per step, head gates")
    p.add_argument("--lam-mlp", type=float, default=2e-4, help="L1 shrink per step, neuron gates")
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=3, help="lr and lam ramp up linearly, then constant")
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--thr", type=float, default=0.05, help="a gate below this counts as 'off'")
    p.add_argument("--milestones", default="0.15,0.3,0.5,0.65,0.8")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--expect-val-emr", type=float, default=None)
    p.add_argument("--guard-epoch", type=int, default=3)
    p.add_argument("--guard-min-emr", type=float, default=50.0)
    p.add_argument("--cpu-smoke", action="store_true")
    a = p.parse_args()
    if a.cpu_smoke:
        a.epochs, a.batch_size, a.workers, a.warmup = 2, 4, 0, 1
        a.lam_head = a.lam_mlp = 0.05            # large, so the smoke run crosses milestones
    return a


def save(model, cp, path, **extra):
    torch.save({**{k: v for k, v in cp.items() if k not in ("encoder", "decoder", "relation", "optimizer")},
                "encoder": plain_state(model.encoder), "decoder": plain_state(model.decoder),
                "relation": torch.nn.Parameter(model.relation.detach().cpu(), requires_grad=False),
                "gates": gate_values(model), **extra}, path)


def main():
    a = parse_args()
    if os.path.exists(a.out):
        stop(2, f"{a.out} already exists; choose another TAG or move it away")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() and not a.cpu_smoke else "cpu"
    amp = dev == "cuda" and torch.cuda.is_bf16_supported()
    print(f"device={dev}  bf16 autocast={amp}  seed={a.seed}")

    from bikit.datasets import BikitDataset
    model, names, cp = load_vit(a.ckpt)
    add_gates(model)
    model.to(dev)
    model.relation.requires_grad_(False)
    print(f"start: {a.ckpt}")

    dtr = BikitDataset(DATASET, split="train", transform=train_transform(cp["img_size"]), devel_mode=a.cpu_smoke)
    dva = BikitDataset(DATASET, split="val", transform=build_transform(cp["img_size"], mode="eval"),
                       devel_mode=a.cpu_smoke)
    ltr = DataLoader(dtr, batch_size=a.batch_size, shuffle=True, drop_last=True,
                     num_workers=a.workers, pin_memory=(dev == "cuda"), persistent_workers=a.workers > 0)
    lva = DataLoader(dva, batch_size=64, shuffle=False, num_workers=a.workers, pin_memory=(dev == "cuda"))
    perm = reorder_index(names, dtr.class_names)

    if not a.cpu_smoke and a.expect_val_emr is not None:      # G0, as in train_vit.py
        model.eval()
        x, y = next(iter(lva))
        with torch.no_grad():
            p = model(x.to(dev)).float().cpu()
        losses = {o: bce(p[:, list(o)], y).item() for o in itertools.permutations(range(len(perm)))}
        if list(min(losses, key=losses.get)) != perm:
            stop(4, "G0: label order is wrong")
        print(f"G0 label order OK (BCE {losses[tuple(perm)]:.4f})")

    emr0, vloss0, _ = evaluate(model, lva, perm, dev)
    print(f"start (gates = 1): val EMR {100 * emr0:.2f}")
    if a.expect_val_emr is not None and not a.cpu_smoke and abs(100 * emr0 - a.expect_val_emr) > 0.2:
        stop(3, f"G1: val EMR {100 * emr0:.2f} != expected {a.expect_val_emr:.2f}")

    os.makedirs(a.out)
    heads, mlps = gate_params(model)
    gate_set = {id(g) for g in heads + mlps}
    enc_groups = param_groups(model.encoder, a.lr_enc, a.wd)
    for g in enc_groups:                                       # gates get their own group
        g["params"] = [p for p in g["params"] if id(p) not in gate_set]
    groups = enc_groups + param_groups(model.decoder, a.lr_dec, a.wd) + \
        [{"params": heads + mlps, "lr": a.lr_gate, "base_lr": a.lr_gate, "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups)
    trainable = [p for g in groups for p in g["params"]]
    milestones = sorted(float(m) for m in a.milestones.split(","))
    steps = len(ltr)
    print(f"gates: {sum(g.numel() for g in heads)} heads, {sum(g.numel() for g in mlps)} neurons;  "
          f"lr_gate {a.lr_gate:g}, lam head {a.lam_head:g} / mlp {a.lam_mlp:g} per step "
          f"({steps} steps/epoch: a useless gate dies in ~{1 / (a.lam_mlp * steps):.0f} epochs)\n")

    hist = []
    for ep in range(1, a.epochs + 1):
        f = min(1.0, ep / max(1, a.warmup))                    # warmup, then constant
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * f
        model.train()
        tot, n, t0 = 0.0, 0, time.time()
        for x, y in ltr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with (torch.autocast("cuda", dtype=torch.bfloat16) if amp else contextlib.nullcontext()):
                p = model(x)
            loss = bce(p[:, perm], y)
            if not torch.isfinite(loss):
                stop(6, f"G3: non-finite loss at epoch {ep}", a, hist)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, a.clip)
            opt.step()
            prox_step(model, a.lam_head * f, a.lam_mlp * f)
            tot += loss.item() * x.size(0); n += x.size(0)
        secs = time.time() - t0
        emr, vloss, _ = evaluate(model, lva, perm, dev)
        gs = gate_stats(model, a.thr)
        rec = {"epoch": ep, "loss": tot / max(1, n), "val_emr": emr, "val_loss": vloss,
               "secs": round(secs, 1), **gs}
        hist.append(rec)
        saved = []
        for m in milestones:
            path = os.path.join(a.out, f"s{round(100 * m):02d}.pth")
            if gs["mlp_lt"] >= m and not os.path.exists(path):
                save(model, cp, path, sparsity_epoch=ep, val_emr=emr, gate_stats=gs, source_ckpt=a.ckpt)
                saved.append(os.path.basename(path))
        save(model, cp, os.path.join(a.out, "last.pth"), sparsity_epoch=ep, val_emr=emr, gate_stats=gs,
             source_ckpt=a.ckpt)
        json.dump({"args": vars(a), "before": {"val_emr": emr0}, "history": hist},
                  open(os.path.join(a.out, "history.json"), "w"), indent=2)
        print(f"ep{ep:4d}  loss {rec['loss']:.4f}  val_emr {100 * emr:.2f}  "
              f"heads off {100 * gs['head_lt']:5.1f}% (mean {gs['head_mean']:.3f})  "
              f"neurons off {100 * gs['mlp_lt']:5.1f}% (mean {gs['mlp_mean']:.3f})  ({secs:.0f}s)"
              + (f"  saved {' '.join(saved)}" if saved else ""), flush=True)
        if not a.cpu_smoke and ep == a.guard_epoch and 100 * emr < a.guard_min_emr:
            stop(5, f"G2: val EMR {100 * emr:.2f} < {a.guard_min_emr} after epoch {ep}", a, hist)

    missing = [f"s{round(100 * m):02d}" for m in milestones
               if not os.path.exists(os.path.join(a.out, f"s{round(100 * m):02d}.pth"))]
    print("\nsnapshots:", sorted(f for f in os.listdir(a.out) if f.endswith(".pth")),
          f"  not reached: {missing}" if missing else "")


if __name__ == "__main__":
    main()
__END_OF_FILE__

cat > prune_vit.py <<'__END_OF_FILE__'
#!/usr/bin/env python
"""Prune a sparsity-trained ViT snapshot: remove the heads and MLP neurons with the smallest gates.

Ranking is global (all 72 heads compete, all 18,432 neurons compete), with floors of 1 head and
10% of neurons per block. Gates are folded into the next layer, so the kept units compute exactly
what they did in the gated model. The result is a plain, narrower ViT saved in the original
checkpoint layout + cp["prune_config"], so load_vit / train_vit / eval_vit / tome_sweep all read it.

  python prune_vit.py --ckpt runs/vit_sparsity/s50.pth --ratio 0.5 --out pruned/vit_r05.pth --eval-val
"""
import argparse, json, os

import torch
from torch.utils.data import DataLoader

from vit_model import build_transform, encoder_dims, load_vit, reorder_index
from vit_gates import add_gates, gate_stats, plain_state, prune_model, select_units
from tome_vit import vit_flops
from train_vit import DATASET, evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="snapshot from sparsity_vit.py (has cp['gates'])")
    p.add_argument("--ratio", type=float, required=True, help="fraction of heads and of neurons to remove")
    p.add_argument("--ratio-heads", type=float, default=None, help="different fraction for heads")
    p.add_argument("--min-heads", type=int, default=1)
    p.add_argument("--min-mlp-frac", type=float, default=0.10)
    p.add_argument("--out", required=True)
    p.add_argument("--eval-val", action="store_true", help="val EMR of the gated and the pruned model")
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def main():
    a = parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"STOP: {a.out} already exists")
    model, names, cp = load_vit(a.ckpt)
    if "gates" not in cp:
        raise SystemExit(f"STOP: {a.ckpt} has no gates (not a sparsity_vit.py snapshot)")
    add_gates(model, init=cp["gates"])
    gs = gate_stats(model)
    plan = select_units(model, a.ratio, a.ratio_heads, a.min_heads, a.min_mlp_frac)
    pruned, cfg = prune_model(model, plan)

    n0 = sum(p.numel() for n, p in model.named_parameters() if not n.endswith(("head_gate", "neuron_gate")))
    n1 = sum(p.numel() for p in pruned.parameters())
    f0 = vit_flops([785] * 12, [0] * 12, [(6, 1536)] * 12)
    f1 = vit_flops([785] * 12, [0] * 12, encoder_dims(pruned.encoder))
    print(f"snapshot {a.ckpt} (sparsity epoch {cp.get('sparsity_epoch')}): "
          f"heads off {100 * gs['head_lt']:.1f}%, neurons off {100 * gs['mlp_lt']:.1f}% (|g| < 0.05)")
    print(f"ratio {a.ratio}" + (f" (heads {a.ratio_heads})" if a.ratio_heads is not None else ""))
    print(f"  heads per block   {cfg['heads']}  ({sum(cfg['heads'])}/72)")
    print(f"  neurons per block {cfg['mlp_hidden']}  ({sum(cfg['mlp_hidden'])}/18432)")
    print(f"  parameters {n0:,} -> {n1:,} ({100 * (1 - n1 / n0):.1f}% smaller)")
    print(f"  GFLOPs     {f0 / 1e9:.1f} -> {f1 / 1e9:.1f} ({100 * (1 - f1 / f0):.1f}% fewer)")

    stats = {"ratio": a.ratio, "ratio_heads": a.ratio_heads, "source": a.ckpt,
             "sparsity_epoch": cp.get("sparsity_epoch"), "gate_stats": gs, "prune_config": cfg,
             "params_before": n0, "params_after": n1, "gflops_before": f0 / 1e9, "gflops_after": f1 / 1e9}
    if a.eval_val:
        from bikit.datasets import BikitDataset
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dva = BikitDataset(DATASET, split="val", transform=build_transform(cp["img_size"], mode="eval"))
        lva = DataLoader(dva, batch_size=64, num_workers=a.workers)
        perm = reorder_index(names, dva.class_names)
        e_g = evaluate(model.to(dev), lva, perm, dev)[0]
        e_p = evaluate(pruned.to(dev), lva, perm, dev)[0]
        stats.update(val_emr_gated=e_g, val_emr_pruned=e_p)
        print(f"  val EMR: gated snapshot {100 * e_g:.2f} -> pruned, before fine-tuning {100 * e_p:.2f}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save({**{k: v for k, v in cp.items()
                   if k not in ("encoder", "decoder", "relation", "optimizer", "gates")},
                "encoder": plain_state(pruned.encoder), "decoder": plain_state(pruned.decoder),
                "relation": torch.nn.Parameter(pruned.relation.detach().cpu(), requires_grad=False),
                "prune_config": cfg, "prune_stats": stats}, a.out)
    json.dump(stats, open(a.out.replace(".pth", "_stats.json"), "w"), indent=2)
    print("saved", a.out)


if __name__ == "__main__":
    main()
__END_OF_FILE__

cat > vit_table.py <<'__END_OF_FILE__'
#!/usr/bin/env python
"""ViT compression table: accuracy vs parameters, FLOPs and MEASURED GPU speed, with the ResNet-50
as the reference to beat (success criterion of the ViT phase).

Test EMR comes from results/<name>_test.json (eval_vit.py), mean +/- std over seeds. Parameters
and FLOPs are counted on the checkpoint; throughput is timed on this GPU in fp32: images/s at
batch 64 and latency at batch 1 (same timing method as tome_sweep.py).

  python vit_table.py              # on a GPU node (vit_table.sh); --preview allows missing runs
  -> results/vit_table.json and results/vit_table.md
"""
import argparse, json, os, time

import numpy as np
import torch
from torch import nn

from vit_model import DEFAULT_CKPT, encoder_dims, load_vit
from tome_vit import vit_flops

seeds = lambda stem: [f"{stem}_seed{s}" for s in range(3)]
RATIOS = ["015", "03", "05", "065", "08"]
ROWS = ([("ViT-S/8 released", str(DEFAULT_CKPT), ["codebrim-classif-balanced_ViT_s8_1"]),
         ("ViT-S/8 fine-tuned (control)", "runs/vit_ctrl_ep100_seed0/best.pth", seeds("vit_ctrl_ep100"))]
        + [(f"ViT-S/8 {int(r) / 10 ** (len(r) - 1) * 100:.0f}% pruned", f"pruned/vit_r{r}.pth",
            seeds(f"vit_prune_r{r}_ep100")) for r in RATIOS])
RESNET = {"label": "ResNet-50 fine-tuned 160 ep (reference)", "emr": [79.43, 78.16, 80.38],
          "params": 23_771_078, "gflops": 8.175}


def resnet_like():
    """Same compute as the bikit DaclNet ResNet-50: backbone + 2048->128->6 head."""
    import torchvision
    m = torchvision.models.resnet50()
    m.fc = nn.Sequential(nn.Linear(2048, 128), nn.ELU(), nn.Linear(128, 6))
    return m


@torch.inference_mode()
def speed(model, dev, bs=64, iters=30, lat_iters=50):
    model.eval().to(dev)
    sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
    x = torch.randn(bs, 3, 224, 224, device=dev)
    for _ in range(10):
        model(x)
    sync(); t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    sync(); ips = iters * bs / (time.perf_counter() - t0)
    x1 = x[:1]
    for _ in range(10):
        model(x1)
    sync(); t0 = time.perf_counter()
    for _ in range(lat_iters):
        model(x1)
    sync(); lat = 1000 * (time.perf_counter() - t0) / lat_iters
    return ips, lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if not a.preview:                                     # check everything before timing anything
        missing = [f"{a.results}/{n}_test.json" for _, _, names in ROWS for n in names
                   if not os.path.exists(os.path.join(a.results, f"{n}_test.json"))]
        missing += [ck for _, ck, _ in ROWS if not os.path.exists(ck)]
        if missing:
            raise SystemExit(f"missing {len(missing)} files, e.g. {missing[:3]} (use --preview)")
    out = []
    for label, ckpt, names in ROWS:
        files = [os.path.join(a.results, f"{n}_test.json") for n in names]
        have = [f for f in files if os.path.exists(f)]
        if len(have) < len(files) and not a.preview:
            raise SystemExit(f"missing {[f for f in files if f not in have]} (use --preview)")
        if not have or not os.path.exists(ckpt):
            print(f"{label:<32} pending"); continue
        ms = [json.load(open(f))["metrics"] for f in have]
        emr = np.array([100 * m["exact_match_ratio"] for m in ms])
        f1 = np.array([100 * m["macro"]["f1"] for m in ms])
        model = load_vit(ckpt)[0]
        dims = encoder_dims(model.encoder)
        ips, lat = speed(model, dev)
        out.append({"label": label, "ckpt": ckpt, "runs": len(have), "emr": emr.tolist(), "f1": f1.tolist(),
                    "params": sum(p.numel() for p in model.parameters()),
                    "gflops": vit_flops([785] * 12, [0] * 12, dims) / 1e9,
                    "heads": sum(h for h, _ in dims), "neurons": sum(m for _, m in dims),
                    "img_per_s": ips, "latency_ms": lat})
        del model
    ips, lat = speed(resnet_like(), dev)
    out.append({"label": RESNET["label"], "runs": 3, "emr": RESNET["emr"], "f1": None,
                "params": RESNET["params"], "gflops": RESNET["gflops"], "heads": None, "neurons": None,
                "img_per_s": ips, "latency_ms": lat})

    base = out[0]
    hdr = (f"| {'model':<38} | test EMR | test error | params | GFLOPs | img/s (bs 64) | "
           f"latency bs 1 | speed vs released |")
    lines = [hdr, "|" + "|".join("-" * (len(c) + 0) for c in hdr.split("|")[1:-1]) + "|"]
    for r in out:
        e = np.array(r["emr"])
        emr_txt = f"{e.mean():.2f} ± {e.std(ddof=1):.2f}" if len(e) > 1 else f"{e.mean():.2f}"
        lines.append(f"| {r['label']:<38} | {emr_txt:>8} | {100 - e.mean():10.2f} | {r['params'] / 1e6:5.2f}M | "
                     f"{r['gflops']:6.2f} | {r['img_per_s']:13.0f} | {r['latency_ms']:9.2f} ms | "
                     f"{r['img_per_s'] / base['img_per_s']:16.2f}x |")
    table = "\n".join(lines)
    print(f"\ndevice: {torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'}  (fp32)\n")
    print(table)
    os.makedirs(a.results, exist_ok=True)
    json.dump(out, open(os.path.join(a.results, "vit_table.json"), "w"), indent=1)
    open(os.path.join(a.results, "vit_table.md"), "w").write(table + "\n")
    print(f"\nsaved -> {a.results}/vit_table.json, vit_table.md")


if __name__ == "__main__":
    main()
__END_OF_FILE__

cat > train_vit.sh <<'__END_OF_FILE__'
#!/bin/bash
#SBATCH -J vit-train
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
set -euo pipefail

echo "=== job $SLURM_JOB_ID on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
source /share/home/HAF9D47L/work/.venv-bikit/bin/activate
echo "python: $(which python)"
cd /share/home/HAF9D47L/work/project-aida/vit

TAG="${TAG:?set TAG}"
EPOCHS="${EPOCHS:?set EPOCHS}"
SEED="${SEED:-0}"
CKPT="${CKPT:-$HOME/.cache/bikit-models/codebrim-classif-balanced_ViT_s8_1.pth}"
EXPECT="${EXPECT:-76.30}"          # val EMR of the released model; EXPECT=none for other starts
GUARD="${GUARD:-60}"               # min val EMR after epoch 3 (lower it for freshly pruned starts)
G1=""
[ "$EXPECT" != none ] && G1="--expect-val-emr $EXPECT"

python train_vit.py --ckpt "$CKPT" --epochs "$EPOCHS" --seed "$SEED" $G1 --guard-min-emr "$GUARD" --out "runs/vit_${TAG}"
python eval_vit.py --ckpt "runs/vit_${TAG}/best.pth" --tag "vit_${TAG}" --no-golden --split test --out results

echo "=== done $(date) ==="
__END_OF_FILE__

cat > sparsity_vit.sh <<'__END_OF_FILE__'
#!/bin/bash
#SBATCH -J vit-sparse
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
set -euo pipefail
echo "=== job $SLURM_JOB_ID on $(hostname) at $(date) ==="
source /share/home/HAF9D47L/work/.venv-bikit/bin/activate
cd /share/home/HAF9D47L/work/project-aida/vit

CKPT="${CKPT:?set CKPT (the fine-tuned control best.pth)}"
EXPECT="${EXPECT:?set EXPECT (val EMR of that checkpoint, e.g. 77.60)}"
python sparsity_vit.py --ckpt "$CKPT" --expect-val-emr "$EXPECT" --epochs "${EPOCHS:-60}" \
       --lam-head "${LAM:-2e-4}" --lam-mlp "${LAM:-2e-4}" --lr-gate "${LR_GATE:-2e-3}" --out "runs/vit_${TAG:-sparsity}"
echo "=== done $(date) ==="
__END_OF_FILE__

cat > prune_all_vit.sh <<'__END_OF_FILE__'
#!/bin/bash
#SBATCH -J vit-prune
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
set -euo pipefail
echo "=== job $SLURM_JOB_ID on $(hostname) at $(date) ==="
source /share/home/HAF9D47L/work/.venv-bikit/bin/activate
cd /share/home/HAF9D47L/work/project-aida/vit

SP="runs/vit_${SP_TAG:-sparsity}"
mkdir -p pruned
for R in ${RATIOS:-0.15 0.3 0.5 0.65 0.8}; do
  N=$(python -c "print(f'{round(100 * $R):02d}')")          # 0.15 -> 15, 0.3 -> 30
  SNAP="$SP/s$N.pth"
  [ -f "$SNAP" ] || { echo "!! $SNAP not reached, pruning from last.pth"; SNAP="$SP/last.pth"; }
  echo; echo "########## ratio $R from $SNAP ##########"
  python prune_vit.py --ckpt "$SNAP" --ratio "$R" --eval-val --out "pruned/vit_r${R/./}.pth"
done
echo "=== done $(date) ==="
__END_OF_FILE__

cat > vit_table.sh <<'__END_OF_FILE__'
#!/bin/bash
#SBATCH -J vit-table
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
set -euo pipefail
source /share/home/HAF9D47L/work/.venv-bikit/bin/activate
cd /share/home/HAF9D47L/work/project-aida/vit
python vit_table.py ${PREVIEW:+--preview}
__END_OF_FILE__

cat > launch_vit_prune.sh <<'__END_OF_FILE__'
#!/bin/bash
# One-shot launch of ViT width pruning, run DETACHED (submitting frees the GPUs):
#   setsid nohup bash launch_vit_prune.sh > launch_prune.log 2>&1 < /dev/null &
# Chain: sparsity training (gates + L1) -> prune at 15/30/50/65/80% -> 100-epoch fine-tuning,
#        3 seeds each (same budget as the control) -> final table with measured throughput.
set -euo pipefail
cd /share/home/HAF9D47L/work/project-aida/vit
START=runs/vit_ctrl_ep100_seed0/best.pth          # control seed with the best val EMR
EXPECT=77.60                                      # its val EMR
RATIOS="0.15 0.3 0.5 0.65 0.8"
[ -f "$START" ] || { echo "STOP: $START missing"; exit 1; }
for d in runs/vit_sparsity pruned/vit_r015.pth runs/vit_prune_r015_ep100_seed0; do
  [ -e "$d" ] && { echo "STOP: $d already exists"; exit 1; }
done
grep -q 'GUARD=' train_vit.sh || { echo "STOP: train_vit.sh lacks the GUARD option"; exit 1; }

J1=$(sbatch --parsable --export=ALL,CKPT=$START,EXPECT=$EXPECT,TAG=sparsity sparsity_vit.sh)
J2=$(sbatch --parsable --dependency=afterok:$J1 --export=ALL,SP_TAG=sparsity prune_all_vit.sh)
FT=()
for S in 0 1 2; do
  for R in $RATIOS; do
    N=r${R/./}
    FT+=($(sbatch --parsable --dependency=afterok:$J2 \
           --export=ALL,CKPT=pruned/vit_$N.pth,EXPECT=none,GUARD=20,EPOCHS=100,SEED=$S,TAG=prune_${N}_ep100_seed$S train_vit.sh))
  done
done
DEP=$(IFS=:; echo "${FT[*]}")
J4=$(sbatch --parsable --dependency=afterok:$DEP vit_table.sh)
{ echo; echo "## Pruning launched $(date '+%F %T')"
  echo "sparsity $J1 -> prune $J2 -> fine-tune ${FT[*]} -> table $J4"; } | tee -a HANDOFF.md
__END_OF_FILE__

chmod +x *.sh
md5sum -c <<'__MD5__'
399c7e00d05d36538cc838a3ddf766d0  vit_model.py
cfad5fbb0e5aeb50ebaac6380ee00854  vit_gates.py
0f2c6389fd06e557840d7024720f1512  tome_vit.py
b7e8e444623794af7dea8a7f9a21cc75  tome_sweep.py
1a21022f040f6f878df0b77d3ff5a7df  train_vit.py
acc6d87b43e25b12d7e683ade30c012c  sparsity_vit.py
6f7499ab4615a2d3f0f70a310e037844  prune_vit.py
d1b7408be091328762f5cd457892f3ca  vit_table.py
12ebb85b130afe7e0e62ad8bb114bc2e  train_vit.sh
a9db71070faedff95fbe885ace880e66  sparsity_vit.sh
d7da0911dcc5ac2be27ff0b637221787  prune_all_vit.sh
224a5a73064b1631bb854057f95ed5a2  vit_table.sh
24e4b54e42a480bb9e34cff91e882786  launch_vit_prune.sh
__MD5__
echo "install OK: 13 files written and verified"
