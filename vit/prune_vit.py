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
