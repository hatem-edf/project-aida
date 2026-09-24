#!/usr/bin/env python
"""Scaling-factor histograms in the style of Figure 4 of Liu et al. 2017.

One panel per checkpoint, signed gamma of the 32 prunable BatchNorm layers
(bn1/bn2 of every bottleneck, 7552 channels). Missing checkpoints are skipped.

    python gamma_hist_paper.py
"""
import os, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANELS = [
    ("no L1  ($\\lambda = 0$)",
     "~/.cache/bikit-models/codebrim-classif-balanced_ResNet50_hta.pth"),
    ("$\\lambda = 3\\times10^{-2}$", "runs/sparsity_lam3e-2/best.pth"),
    ("$\\lambda = 5\\times10^{-5}$, epoch 30", "runs/sparsity_lam5e-5/s40.pth"),
    ("$\\lambda = 5\\times10^{-5}$, epoch 40", "runs/sparsity_lam5e-5/last.pth"),
]


def prunable_gammas(path):
    ck = torch.load(os.path.expanduser(path), map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    g = [v.flatten() for k, v in sd.items()
         if k.startswith("base.") and k.endswith(".weight") and v.dim() == 1
         and k.split(".")[-2] in ("bn1", "bn2")]
    return torch.cat(g).numpy(), ck.get("sparsity_epoch")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figs/presentation/09_scaling_factors_paper_style.png")
    a = ap.parse_args()

    found = []
    for title, path in PANELS:
        if os.path.exists(os.path.expanduser(path)):
            g, ep = prunable_gammas(path)
            found.append((title, g))
            plain = title.replace("$", "").replace("\\lambda", "lambda").replace("\\times", "x")
            print(f"{plain:<44} n={len(g)}  epoch={ep}  "
                  f"|g|<0.01: {100*(np.abs(g) < 0.01).mean():5.1f}%  mean|g| {np.abs(g).mean():.4f}")
        else:
            print(f"missing, skipped: {path}")
    if not found:
        return

    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                         "font.size": 11})
    fig, axes = plt.subplots(1, len(found), figsize=(3.6 * len(found), 2.9))
    axes = np.atleast_1d(axes)
    bins = np.arange(-0.15, 0.85 + 1e-9, 0.01)
    for i, (ax, (title, g)) in enumerate(zip(axes, found)):
        ax.hist(np.clip(g, bins[0], bins[-1]), bins=bins, color="#0000EE", edgecolor="none")
        ax.set_xlim(-0.12, 0.5)
        ax.grid(True, ls=":", color="k", lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=11)
        ax.tick_params(labelsize=9)
        ax.text(0.97, 0.93, f"{100*(np.abs(g) < 0.01).mean():.0f}% at $\\approx$0",
                transform=ax.transAxes, ha="right", va="top", fontsize=9.5)
        if i == 0:
            ax.set_ylabel("Count")
    fig.supxlabel("Scaling factor value", fontsize=11, y=0.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=220, facecolor="white", bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
