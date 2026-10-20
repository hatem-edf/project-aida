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
