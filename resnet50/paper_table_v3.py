#!/usr/bin/env python
"""Network-Slimming results table for ResNet-50 on CODEBRIM, version 3.

Same method, two fine-tuning budgets side by side (160 vs 40 epochs), all pruned
models coming from the fine-tuned baseline. The first experiment (sparsity from the
released model, 40 epochs) stays in paper_table_v2.py for a backup slide.

Every results file is checked against its run folder before use: it must come from
the row's starting checkpoint, have trained the row's number of epochs to the end,
and not have been overwritten since. Any mismatch stops the script.

    cd ~/work/project-aida/resnet50 && python paper_table_v3.py            # final table
    python paper_table_v3.py --preview      # while jobs still run: uses what exists

    -> figs/presentation/table_slimming_v3.{tex,png}   (_PREVIEW when --preview)
"""
import argparse, json, os

import numpy as np
import torch
from torch import nn

ORIG = "~/.cache/bikit-models/codebrim-classif-balanced_ResNet50_hta.pth"
RELEASED_EMR = 70.89   # short side 256 + center crop 224 (bilinear): the protocol behind the published 71.36
runs = lambda stem: [f"ft_{stem}_seed{s}" for s in range(3)]
of632 = lambda *k: [100 * n / 632 for n in k]   # correct test images out of 632 -> EMR in %

# ("block", title)  or  (label, starting checkpoint, fine-tuning epochs, [results name or literal EMR, ...])
TABLE = [
    ("block", "Reference"),
    ("ResNet-50 (released)\u2021", ORIG, None, [RELEASED_EMR]),
    ("block", "Fine-tuning: 160 epochs"),
    ("ResNet-50 (Baseline)", ORIG, 160, runs("control_ep160")),
    ("ResNet-50 (50% Pruned)", "pruned/strong_r05.pth", 160, runs("strong_r05_ep160")),
    ("ResNet-50 (65% Pruned)", "pruned/strong_r065.pth", 160, runs("strong_r065_ep160")),
    ("ResNet-50 (80% Pruned)", "pruned/strong_r08.pth", 160, runs("strong_r08_ep160")),
    ("block", "Fine-tuning: 40 epochs"),
    # These 40-epoch run folders were overwritten on Sep 28 by 160-epoch runs whose TAG was ignored.
    # Their results survive in the original logs and in paper_table_v2's output from that morning.
    ("ResNet-50 (Baseline)", ORIG, 40, of632(488, 472, 497)),
    ("ResNet-50 (50% Pruned)", "pruned/strong_r05.pth", 40, of632(429, 444, 422)),   # logs 32935-32937
    ("ResNet-50 (65% Pruned)", "pruned/strong_r065.pth", 40, of632(402, 416, 431)),
    ("ResNet-50 (80% Pruned)", "pruned/strong_r08.pth", 40, of632(408, 397, 397)),
]
CAPTION = "Test Errors on CODEBRIM"
FOOTNOTE = ("Test error = 100 \u2212 exact match ratio; mean \u00b1 std over 3 seeds.  "
            "Pruned models: sparsity training ($\\lambda = 5\\times10^{-5}$, 40 epochs) from the\n"
            "40-epoch baseline, channel pruning, then fine-tuning.  \u2021 Released checkpoint, evaluated with its "
            "original\npreprocessing (short side 256, center crop 224); all other rows use a 224\u00d7224 resize.  "
            "FLOPs for one 224\u00d7224 image.")


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


def read_emr(entry, ckpt, epochs, a):
    """A literal number, or a results file whose run is verified to match the row."""
    if isinstance(entry, (int, float)):
        return float(entry)
    res = os.path.join(a.results, f"{entry}_test.json")
    hist = os.path.join("runs", entry, "history.json")
    if not os.path.exists(res):
        if a.preview:
            return None
        raise SystemExit(f"missing {res}  (still running? --preview builds a partial table)")
    if not os.path.exists(hist):
        raise SystemExit(f"{entry}: results exist but {hist} doesn't, so its origin can't be checked")
    H = json.load(open(hist))
    args, n_logged = H["args"], len(H["history"])
    problems = []
    if int(args["epochs"]) != epochs:
        problems.append(f"trained with {args['epochs']} epochs, this row is {epochs}")
    if n_logged != int(args["epochs"]):
        problems.append(f"only {n_logged}/{args['epochs']} epochs completed")
    if os.path.basename(args["ckpt"]) != os.path.basename(ckpt):
        problems.append(f"started from {args['ckpt']}, this row starts from {os.path.basename(ckpt)}")
    if os.path.getmtime(hist) > os.path.getmtime(res) + 60:
        problems.append("its run folder changed after the result was written (overwritten?)")
    if problems:
        raise SystemExit(f"REFUSED {entry}: " + "; ".join(problems))
    return 100 * json.load(open(res))["metrics"]["exact_match_ratio"]


def build_rows(a):
    from prune_slim import load_pruned
    rows, cache, base = [], {}, None
    for item in TABLE:
        if item[0] == "block":
            rows.append({"block": item[1]})
            continue
        label, ckpt, epochs, entries = item
        ckpt = os.path.expanduser(ckpt)
        if ckpt not in cache:
            model, _ = load_pruned(ckpt)
            cache[ckpt] = (sum(p.numel() for p in model.parameters()), count_flops(model))
        n, f = cache[ckpt]
        base = base or (n, f)          # the first model row (released) is the size reference
        emr = [v for v in (read_emr(e, ckpt, epochs, a) for e in entries) if v is not None]
        tag = f"{label}{'' if epochs is None else f' [{epochs} ep]'}"
        if not emr:
            print(f"{tag:<40} pending (no finished run yet)")
            continue
        err = 100 - np.asarray(emr)
        rows.append(dict(label=label, n=n, flops=f, err=err.mean(),
                         std=err.std(ddof=1) if len(err) > 1 else None,
                         p_pruned=None if n == base[0] else 100 * (1 - n / base[0]),
                         f_pruned=None if f == base[1] else 100 * (1 - f / base[1])))
        runs_txt = ", ".join(f"{v:.2f}" for v in emr) + (f"  ({len(emr)}/{len(entries)})" if len(emr) < len(entries) else "")
        print(f"{tag:<40} EMR {runs_txt:<30} error {err.mean():5.2f}  params {n:>11,}  FLOPs {f:.3e}")
    # bold = lowest error inside each block, like the paper does per architecture
    block = []
    for r in rows + [{"block": None}]:
        if "block" in r:
            if len(block) > 1:          # a lone reference row isn't a "best"
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
    global CAPTION
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figs/presentation")
    ap.add_argument("--preview", action="store_true", help="allow unfinished runs (partial table)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = build_rows(a)
    name = "table_slimming_v3" + ("_PREVIEW" if a.preview else "")
    if a.preview:
        CAPTION += "  (PREVIEW: some runs unfinished)"
    write_tex(rows, os.path.join(a.out, name + ".tex"))
    write_png(rows, os.path.join(a.out, name + ".png"))


if __name__ == "__main__":
    main()
