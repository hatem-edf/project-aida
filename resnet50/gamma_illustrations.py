#!/usr/bin/env python
"""Illustrations of network slimming on our own model, in the style of
Liu et al. 2017 (blue bars, dotted grid, serif).

  10_before_after.png        |gamma| histogram, released vs sparsity-trained
  11_kept_vs_pruned.png      gammas sorted by size, split by the global threshold:
                             (a) all 7552 prunable channels, (b)(c) two single layers

Model: runs/sparsity_lam5e-5/s40.pth (epoch 30) pruned at ratio 0.5, i.e. the
"50% Pruned" row of the table. The threshold and the per-layer floor are
recomputed exactly as prune_slim.py does, so the kept counts must match the
pruning log.

    python gamma_illustrations.py
"""
import os, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BLUE, GREY, INK = "#0000EE", "#A9A9A9", "#111111"


def layer_gammas(path):
    sd = torch.load(os.path.expanduser(path), map_location="cpu", weights_only=False)["state_dict"]
    out = []
    for k, v in sd.items():
        parts = k.split(".")
        if k.startswith("base.") and k.endswith(".weight") and v.dim() == 1 and parts[-2] in ("bn1", "bn2"):
            out.append((".".join(parts[:-1]), v.abs().numpy()))
    return out


def kept_mask(w, thr, min_keep):
    keep = w > thr
    floor = max(1, int(round(min_keep * len(w))))
    if keep.sum() < floor:
        keep = np.zeros(len(w), bool)
        keep[np.argsort(-w)[:floor]] = True
    return keep


def pretty(name):
    # base.4.2.bn2 -> stage 1, block 3 · bn2
    _, st, blk, bn = name.split(".")
    return f"stage {int(st) - 3}, block {int(blk) + 1} \u00b7 {bn}"


def style():
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                         "font.size": 11, "axes.edgecolor": INK})


def grid(ax):
    ax.grid(True, ls=":", color="k", lw=0.6)
    ax.set_axisbelow(True)


def before_after(before, after, out, labels):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.9), sharex=True)
    bins = np.arange(0, 0.70 + 1e-9, 0.01)
    for ax, g, lab in zip(axes, (before, after), labels):
        ax.hist(np.clip(g, 0, 0.7), bins=bins, color=BLUE, edgecolor="none")
        grid(ax)
        ax.set_title(lab, fontsize=11)
        ax.set_xlim(-0.02, 0.7)
        ax.text(0.97, 0.93, f"{100*(g < 0.01).mean():.0f}% below 0.01",
                transform=ax.transAxes, ha="right", va="top", fontsize=9.5)
    axes[0].set_ylabel("Count")
    fig.supxlabel("|Scaling factor|", fontsize=11, y=0.02)
    fig.tight_layout()
    fig.savefig(out, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def sorted_bars(ax, w, keep, thr, title):
    order = np.argsort(-w)
    ws, ks = w[order], keep[order]
    x = np.arange(len(ws))
    stub = 0.012 * ws.max()                     # make gamma = 0 channels visible
    shown = np.where(ks, ws, np.maximum(ws, stub))
    ax.bar(x, shown, width=0.85, color=np.where(ks, BLUE, GREY), edgecolor="none")
    ax.set_ylim(0, 1.08 * ws.max())
    split = ks.sum()
    ax.axvline(split - 0.5, color=INK, ls="--", lw=1.1)
    ax.axhline(thr, color=INK, ls=":", lw=0.9)
    grid(ax)
    ax.set_xlim(-1, len(ws))
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("channels, sorted by |\u03b3|")


def kept_vs_pruned(layers, thr, min_keep, out, pick):
    g_all = np.concatenate([w for _, w in layers])
    keep_all = np.concatenate([kept_mask(w, thr, min_keep) for _, w in layers])

    fig = plt.figure(figsize=(12, 3.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.22)

    # (a) whole network as an area
    ax = fig.add_subplot(gs[0])
    order = np.argsort(-g_all)
    ws, ks = g_all[order], keep_all[order]
    x = np.arange(len(ws))
    ax.fill_between(x, ws, where=ks, color=BLUE, step="mid", lw=0)
    ax.fill_between(x, np.maximum(ws, 0.012 * ws.max()), where=~ks, color=GREY, step="mid", lw=0)
    ax.set_ylim(0, 1.08 * ws.max())
    n_keep_thr = int((g_all > thr).sum())
    ax.axvline(n_keep_thr, color=INK, ls="--", lw=1.1)
    ax.axhline(thr, color=INK, ls=":", lw=0.9)
    ax.text(n_keep_thr + 80, thr + 0.02, f"global threshold\n|\u03b3| = {thr:.3f}", fontsize=9, va="bottom")
    grid(ax)
    ax.set_xlim(0, len(ws))
    ax.set_title(f"(a) all {len(ws)} prunable channels: {int(ks.sum())} kept", fontsize=11)
    ax.set_xlabel("channels, sorted by |\u03b3|")
    ax.set_ylabel("|γ|")

    # (b)(c) two individual layers, same global threshold
    for i, name in enumerate(pick):
        w = dict(layers)[name]
        keep = kept_mask(w, thr, min_keep)
        ax = fig.add_subplot(gs[i + 1], sharey=None)
        sorted_bars(ax, w, keep, thr,
                    f"({'bc'[i]}) {pretty(name)}: {int(keep.sum())} of {len(w)} kept")
        print(f"  {name}: kept {int(keep.sum())}/{len(w)}")

    fig.legend(handles=[Patch(color=BLUE, label="kept"), Patch(color=GREY, label="pruned (many are exactly 0)"),
                        plt.Line2D([], [], color=INK, ls=":", label="global threshold")],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.17))
    fig.savefig(out, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="~/.cache/bikit-models/codebrim-classif-balanced_ResNet50_hta.pth")
    ap.add_argument("--after", default="runs/sparsity_lam5e-5/s40.pth")
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--min-keep", type=float, default=0.10)
    ap.add_argument("--layers", nargs=2, default=["base.4.2.bn2", "base.4.0.bn2"],
                    help="two layers for panels (b) and (c)")
    ap.add_argument("--out", default="figs/presentation")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    style()

    before = np.concatenate([w for _, w in layer_gammas(a.before)])
    after_layers = layer_gammas(a.after)
    after = np.concatenate([w for _, w in after_layers])
    thr = float(torch.quantile(torch.tensor(after, dtype=torch.float32), a.ratio))
    print(f"global threshold {thr:.6f} (prune log for s40 at 0.5: 0.054487)")

    before_after(before, after, os.path.join(a.out, "10_before_after.png"),
                 ["released, ordinary training ($\\lambda = 0$)",
                  "after sparsity training ($\\lambda = 5\\times10^{-5}$, epoch 30)"])
    kept_vs_pruned(after_layers, thr, a.min_keep,
                   os.path.join(a.out, "11_kept_vs_pruned.png"), a.layers)


if __name__ == "__main__":
    main()
