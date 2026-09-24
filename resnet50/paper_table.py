#!/usr/bin/env python
"""Reproduce the layout of Table 1 of Liu et al. 2017 (Network Slimming)
for our ResNet-50 on CODEBRIM.

Parameters and FLOPs are measured from the checkpoints; test EMR values are
the results of our runs (mean over seeds where we have 3).

    python paper_table.py          # writes figs/presentation/table_slimming.{tex,png}

FLOPs convention: 1 multiply-add = 2 FLOPs, conv + linear layers, one
224x224 image. This matches the paper, whose VGG-19 (~0.40 G multiply-adds
on 32x32) is listed as 7.97e8 FLOPs.
"""
import os, argparse
import numpy as np
import torch
from torch import nn

ROWS = [  # (label, checkpoint, test EMR per seed, bold)
    ("ResNet-50 (Baseline)",
     "~/.cache/bikit-models/codebrim-classif-balanced_ResNet50_hta.pth",
     [77.22, 74.68, 78.64]),          # unpruned, same fine-tuning, 32917-32919
    ("ResNet-50 (50% Pruned)",  "pruned/s40_r05.pth",  [64.24, 67.56, 69.30]),
    ("ResNet-50 (65% Pruned)",  "pruned/s40_r065.pth", [65.35, 67.88, 64.08]),
    ("ResNet-50 (80% Pruned)",  "pruned/s40_r08.pth",  [64.24, 64.87, 67.88]),
    ("ResNet-50 (90% Pruned)\u2020", "pruned/s40_r09.pth", [63.61]),
]
CAPTION = "(a) Test Errors on CODEBRIM"
FOOTNOTE = ("Test error = 100 \u2212 exact match ratio; mean \u00b1 std over 3 seeds, "
            "\u2020 single run. Input 224\u00d7224, so FLOPs are not comparable to CIFAR's 32\u00d732.")


def count_flops(model, res=224):
    macs = 0
    def conv(m, i, o):
        nonlocal macs
        kh, kw = m.kernel_size
        macs += (o.numel() // o.shape[0]) * (m.in_channels // m.groups) * kh * kw
    def lin(m, i, o):
        nonlocal macs
        macs += m.in_features * m.out_features
    hs = [m.register_forward_hook(conv) for m in model.modules() if isinstance(m, nn.Conv2d)]
    hs += [m.register_forward_hook(lin) for m in model.modules() if isinstance(m, nn.Linear)]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, res, res))
    for h in hs:
        h.remove()
    return 2 * macs


def fmt_params(n):
    return f"{n/1e6:.2f}M"


def sci(x):
    e = int(np.floor(np.log10(x)))
    return f"{x/10**e:.2f}", e


def build_rows():
    from prune_slim import load_pruned
    out = []
    for label, path, emr in ROWS:
        path = os.path.expanduser(path)
        model, _ = load_pruned(path)
        n = sum(p.numel() for p in model.parameters())
        f = count_flops(model)
        err = 100 - np.asarray(emr)
        out.append(dict(label=label, n=n, flops=f, err=err.mean(),
                        std=err.std(ddof=1) if len(err) > 1 else None))
        print(f"{label:<28} params {n:>12,}  FLOPs {f:.3e}  error {err.mean():.2f}")
    base = out[0]
    for r in out:
        r["p_pruned"] = None if r is base else 100 * (1 - r["n"] / base["n"])
        r["f_pruned"] = None if r is base else 100 * (1 - r["flops"] / base["flops"])
    best = min(range(len(out)), key=lambda i: out[i]["err"])
    for i, r in enumerate(out):
        r["bold"] = (i == best)
    return out


def write_tex(rows, path):
    L = [r"% requires nothing beyond standard LaTeX",
         r"\begin{table}[t]", r"\centering",
         r"{\small " + CAPTION + r"}\\[2pt]",
         r"\begin{tabular}{l|c|c|c|c|c}", r"\hline",
         r"Model & Test error (\%) & Parameters & Pruned & FLOPs & Pruned \\", r"\hline"]
    for r in rows:
        err = f"{r['err']:.2f}"
        if r["bold"]:
            err = r"\textbf{" + err + "}"
        if r["std"] is not None:
            err += r" {\scriptsize$\pm$" + f"{r['std']:.2f}" + "}"
        m, e = sci(r["flops"])
        pp = "-" if r["p_pruned"] is None else f"{r['p_pruned']:.1f}\\%"
        fp = "-" if r["f_pruned"] is None else f"{r['f_pruned']:.1f}\\%"
        label = r["label"].replace("%", r"\%").replace("\u2020", r"$^\dagger$")
        L.append(f"{label} & {err} & {fmt_params(r['n'])} & {pp} & "
                 f"${m}\\times10^{{{e}}}$ & {fp} \\\\")
    L += [r"\hline", r"\end{tabular}", r"\\[2pt]",
          r"{\scriptsize " + FOOTNOTE.replace("\u2212", "$-$").replace("\u00b1", r"$\pm$")
          .replace("\u2020", r"$^\dagger$").replace("\u00d7", r"$\times$") + "}",
          r"\end{table}"]
    open(path, "w").write("\n".join(L) + "\n")
    print("wrote", path)


def write_png(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})
    INK = "#111111"

    headers = ["Model", "Test error (%)", "Parameters", "Pruned", "FLOPs", "Pruned"]
    widths = [0.30, 0.17, 0.13, 0.10, 0.18, 0.12]
    xs = np.concatenate([[0], np.cumsum(widths)])
    cells = []
    for r in rows:
        err = f"{r['err']:.2f}"
        std = f" \u00b1{r['std']:.2f}" if r["std"] is not None else ""
        m, e = sci(r["flops"])
        cells.append([r["label"], (err, std, r["bold"]), fmt_params(r["n"]),
                      "-" if r["p_pruned"] is None else f"{r['p_pruned']:.1f}%",
                      f"${m}\\times10^{{{e}}}$",
                      "-" if r["f_pruned"] is None else f"{r['f_pruned']:.1f}%"])

    rh = 0.105
    n = len(rows) + 1
    H = rh * n
    fig = plt.figure(figsize=(9.2, 0.34 * n + 0.75))
    ax = fig.add_axes([0.02, 0.15, 0.96, 0.70])
    ax.set_xlim(0, 1); ax.set_ylim(-H, 0.02); ax.axis("off")

    def hline(y, lw=0.9):
        ax.plot([0, 1], [y, y], color=INK, lw=lw, solid_capstyle="butt")

    hline(0); hline(-rh); hline(-H)
    for x in xs[1:-1]:
        ax.plot([x, x], [0, -H], color=INK, lw=0.7)

    def put(col, row, text, bold=False, size=12.5):
        y = -(row + 0.5) * rh
        if col == 0:
            ax.text(xs[0] + 0.012, y, text, ha="left", va="center", fontsize=size,
                    color=INK, fontweight="bold" if bold else "normal")
        else:
            ax.text((xs[col] + xs[col + 1]) / 2, y, text, ha="center", va="center",
                    fontsize=size, color=INK, fontweight="bold" if bold else "normal")

    for c, h in enumerate(headers):
        put(c, 0, h)
    for ri, row in enumerate(cells, start=1):
        for c, v in enumerate(row):
            if c == 1:
                err, std, bold = v
                y = -(ri + 0.5) * rh
                cx = (xs[1] + xs[2]) / 2
                ax.text(cx, y, err, ha="right" if std else "center", va="center",
                        fontsize=12.5, color=INK, fontweight="bold" if bold else "normal")
                if std:
                    ax.text(cx, y, std, ha="left", va="center", fontsize=9.5, color="#555555")
            else:
                put(c, ri, v)

    fig.text(0.5, 0.88, CAPTION, ha="center", va="bottom", fontsize=12, color=INK)
    fig.text(0.5, 0.02, FOOTNOTE, ha="center", va="bottom", fontsize=8.5, color="#444444")
    fig.savefig(path, dpi=220, facecolor="white", bbox_inches="tight")
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figs/presentation")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = build_rows()
    write_tex(rows, os.path.join(a.out, "table_slimming.tex"))
    write_png(rows, os.path.join(a.out, "table_slimming.png"))


if __name__ == "__main__":
    main()
