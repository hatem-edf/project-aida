#!/usr/bin/env python
"""Network-Slimming results table for ResNet-50 on CODEBRIM, version 2.

Changes from paper_table.py (v1, first slide):
  - reference row for the released checkpoint, now reproduced with its original
    preprocessing (70.89 EMR vs 71.36 published, preproc_sweep.py, job 32967)
  - second pruning pipeline: sparsity training started from the fine-tuned baseline
  - test EMRs read from results/*.json instead of typed in (a missing file is an error)

    cd ~/work/project-aida/resnet50 && python paper_table_v2.py
    -> figs/presentation/table_slimming_v2.{tex,png}

Parameters and FLOPs are measured on the checkpoints (1 multiply-add = 2 FLOPs,
conv + linear, one 224x224 image), as in v1.
"""
import argparse, json, os

import numpy as np
import torch
from torch import nn

ORIG = "~/.cache/bikit-models/codebrim-classif-balanced_ResNet50_hta.pth"
RELEASED_EMR = 70.89   # short side 256 + center crop 224 (bilinear): the protocol behind the published 71.36

# ("block", title)  or  (label, checkpoint for params/FLOPs, [results name or literal EMR, ...])
TABLE = [
    ("block", "Reference"),
    ("ResNet-50 (released)\u2021", ORIG, [RELEASED_EMR]),
    ("ResNet-50 (Baseline)", ORIG, ["ft_codebrim-classif-balanced_ResNet50_hta_seed0",
                                    "ft_codebrim-classif-balanced_ResNet50_hta_seed1",
                                    "ft_codebrim-classif-balanced_ResNet50_hta_seed2"]),
    ("block", "Sparsity from the released model"),
    ("ResNet-50 (50% Pruned)", "pruned/s40_r05.pth", ["ft_s40_r05_seed0", "ft_s40_r05_seed1", "ft_s40_r05_seed2"]),
    ("ResNet-50 (65% Pruned)", "pruned/s40_r065.pth", ["ft_s40_r065_seed0", "ft_s40_r065_seed1", "ft_s40_r065_seed2"]),
    ("ResNet-50 (80% Pruned)", "pruned/s40_r08.pth", ["ft_s40_r08_seed0", "ft_s40_r08_seed1", "ft_s40_r08_seed2"]),
    ("ResNet-50 (90% Pruned)\u2020", "pruned/s40_r09.pth", ["ft_s40_r09_seed0"]),
    ("block", "Sparsity from the fine-tuned baseline"),
    # seed 1's results file was overwritten by the 160-epoch run (same TAG); 70.25 is from its log, job 32936
    ("ResNet-50 (50% Pruned)", "pruned/strong_r05.pth", ["ft_strong_r05_seed0", 70.25, "ft_strong_r05_seed2"]),
    ("ResNet-50 (65% Pruned)", "pruned/strong_r065.pth", ["ft_strong_r065_seed0", "ft_strong_r065_seed1", "ft_strong_r065_seed2"]),
    ("ResNet-50 (80% Pruned)", "pruned/strong_r08.pth", ["ft_strong_r08_seed0", "ft_strong_r08_seed1", "ft_strong_r08_seed2"]),
    ("ResNet-50 (50% Pruned, 160 ep)\u2020", "pruned/strong_r05.pth", ["ft_strong_r05_ep160_seed1"]),
]
CAPTION = "Test Errors on CODEBRIM"
FOOTNOTE = ("Test error = 100 \u2212 exact match ratio; mean \u00b1 std over 3 seeds, \u2020 single run.\n"
            "\u2021 Released checkpoint, evaluated with its original preprocessing (short side 256, center crop 224);\n"
            "all other rows are trained and evaluated with a 224\u00d7224 resize.  FLOPs for one 224\u00d7224 image.")


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


def sci(x):
    e = int(np.floor(np.log10(x)))
    return f"{x / 10 ** e:.2f}", e


def read_emr(entry, results_dir):
    if isinstance(entry, (int, float)):
        return float(entry)
    path = os.path.join(results_dir, f"{entry}_test.json")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}  (check the name with: ls {results_dir} | grep test.json)")
    return 100 * json.load(open(path))["metrics"]["exact_match_ratio"]


def build_rows(results_dir):
    from prune_slim import load_pruned
    rows, cache, base = [], {}, None
    for item in TABLE:
        if item[0] == "block":
            rows.append({"block": item[1]})
            continue
        label, ckpt, entries = item
        ckpt = os.path.expanduser(ckpt)
        if ckpt not in cache:
            model, _ = load_pruned(ckpt)
            cache[ckpt] = (sum(p.numel() for p in model.parameters()), count_flops(model))
        n, f = cache[ckpt]
        base = base or (n, f)          # the first model row (released) is the size reference
        emr = [read_emr(e, results_dir) for e in entries]
        err = 100 - np.asarray(emr)
        rows.append(dict(label=label, n=n, flops=f, err=err.mean(),
                         std=err.std(ddof=1) if len(err) > 1 else None,
                         p_pruned=None if n == base[0] else 100 * (1 - n / base[0]),
                         f_pruned=None if f == base[1] else 100 * (1 - f / base[1])))
        print(f"{label:<34} EMR per run {', '.join(f'{v:.2f}' for v in emr):<22} "
              f"error {err.mean():5.2f}  params {n:>11,}  FLOPs {f:.3e}")
    # bold = lowest error inside each block, like the paper does per architecture
    block = []
    for r in rows + [{"block": None}]:
        if "block" in r:
            if block:
                min(block, key=lambda x: x["err"])["bold"] = True
            block = []
        else:
            block.append(r)
    return rows


def cells(r):
    m, e = sci(r["flops"])
    return (r["label"], f"{r['n'] / 1e6:.2f}M",
            "-" if r["p_pruned"] is None else f"{r['p_pruned']:.1f}%", (m, e),
            "-" if r["f_pruned"] is None else f"{r['f_pruned']:.1f}%")


def write_tex(rows, path):
    L = [r"\begin{table}[t]", r"\centering", r"{\small " + CAPTION + r"}\\[2pt]",
         r"\begin{tabular}{l|c|c|c|c|c}", r"\hline",
         r"Model & Test error (\%) & Parameters & Pruned & FLOPs & Pruned \\"]
    for r in rows:
        if "block" in r:
            L += [r"\hline", r"\multicolumn{6}{l}{\textit{" + r["block"] + r"}} \\"]
            continue
        label, n, pp, (m, e), fp = cells(r)
        err = f"{r['err']:.2f}"
        if r.get("bold"):
            err = r"\textbf{" + err + "}"
        if r["std"] is not None:
            err += r" {\scriptsize$\pm$" + f"{r['std']:.2f}" + "}"
        label = label.replace("%", r"\%").replace("\u2020", r"$^\dagger$").replace("\u2021", r"$^\ddagger$")
        L.append(f"{label} & {err} & {n} & {pp.replace('%', chr(92) + '%')} & "
                 f"${m}\\times10^{{{e}}}$ & {fp.replace('%', chr(92) + '%')} \\\\")
    foot = (FOOTNOTE.replace("\u2212", "$-$").replace("\u00b1", r"$\pm$").replace("\u2020", r"$^\dagger$")
            .replace("\u2021", r"$^\ddagger$").replace("\u00d7", r"$\times$").replace("\n", " "))
    L += [r"\hline", r"\end{tabular}", r"\\[2pt]", r"{\scriptsize " + foot + "}", r"\end{table}"]
    open(path, "w").write("\n".join(L) + "\n")
    print("wrote", path)


def write_png(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})
    INK, GREY = "#111111", "#555555"
    headers = ["Model", "Test error (%)", "Parameters", "Pruned", "FLOPs", "Pruned"]
    widths = [0.37, 0.17, 0.12, 0.09, 0.14, 0.11]
    xs = np.concatenate([[0], np.cumsum(widths)])
    rh, bh = 0.10, 0.072                       # data row / block-title row heights
    ys, y = [], -rh
    for r in rows:
        h = bh if "block" in r else rh
        ys.append((y, h)); y -= h
    H = -y
    fig, ax = plt.subplots(figsize=(9.6, 4.2 * H))
    ax.set_xlim(0, 1); ax.set_ylim(-H, 0); ax.axis("off")
    hl = lambda yy, lw=0.9: ax.plot([0, 1], [yy, yy], color=INK, lw=lw, solid_capstyle="butt")
    hl(0); hl(-rh); hl(-H)
    for x in xs[1:-1]:                        # continuous vertical rules
        ax.plot([x, x], [0, -H], color=INK, lw=0.7)

    def put(col, yc, text, bold=False):
        x = xs[0] + 0.012 if col == 0 else (xs[col] + xs[col + 1]) / 2
        ax.text(x, yc, text, ha="left" if col == 0 else "center", va="center",
                fontsize=12.5, color=INK, fontweight="bold" if bold else "normal")

    for c, h in enumerate(headers):
        put(c, -rh / 2, h)
    for r, (y0, h) in zip(rows, ys):
        yc = y0 - h / 2
        if "block" in r:
            if y0 != -rh:
                hl(y0, 0.6)
            ax.text(xs[0] + 0.012, yc, r["block"], ha="left", va="center", fontsize=10.5,
                    color=GREY, style="italic")
            continue
        label, n, pp, (m, e), fp = cells(r)
        put(0, yc, label)
        cx = (xs[1] + xs[2]) / 2
        std = f" \u00b1{r['std']:.2f}" if r["std"] is not None else ""
        ax.text(cx, yc, f"{r['err']:.2f}", ha="right" if std else "center", va="center",
                fontsize=12.5, color=INK, fontweight="bold" if r.get("bold") else "normal")
        if std:
            ax.text(cx, yc, std, ha="left", va="center", fontsize=9.5, color=GREY)
        put(2, yc, n); put(3, yc, pp); put(4, yc, f"${m}\\times10^{{{e}}}$"); put(5, yc, fp)
    ax.text(0.5, 0.03, CAPTION, ha="center", va="bottom", fontsize=12, color=INK)
    ax.text(0.5, -H - 0.035, FOOTNOTE, ha="center", va="top", fontsize=8.3, color="#444444",
            linespacing=1.4)
    fig.savefig(path, dpi=220, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figs/presentation")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = build_rows(a.results)
    write_tex(rows, os.path.join(a.out, "table_slimming_v2.tex"))
    write_png(rows, os.path.join(a.out, "table_slimming_v2.png"))


if __name__ == "__main__":
    main()
