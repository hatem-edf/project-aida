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
