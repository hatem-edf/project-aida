#!/usr/bin/env python
"""All presentation plots for the CODEBRIM ResNet50 slimming project.

Every number below is transcribed from the job logs (job ids in comments).
Plots 01-07 need nothing but this file. Plot 08 reads the sparsity
checkpoints, so run it on the cluster:

    python make_plots.py            # 01-07
    python make_plots.py --gammas   # also 08 (needs runs/ and ~/.cache)
"""
import os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ------------------------------------------------------------------ style
INK, MUTED, GRID = "#1B2530", "#5A6470", "#D8D3CA"
TEAL, RUST, AMBER, GREY = "#2A6F74", "#B5451B", "#D9962B", "#B9B3A8"
SLATE, SAND = "#4F6D8A", "#E3DFD8"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#9A958C",
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
    "savefig.transparent": True, "savefig.bbox": "tight", "savefig.dpi": 200,
})
STEPS_PER_EPOCH = 93   # 6013 train images, batch 64, drop_last

# ------------------------------------------------------------------ data
CLASSES = ["background", "crack", "spallation", "efflorescence", "exposed bars", "corrosion stain"]
REPORTED = dict(EMR=71.36, Accuracy=92.59, Precision=85.33, Recall=83.20, F1=84.13, AUROC=96.99)
MEASURED = dict(EMR=64.24, Accuracy=89.90, Precision=78.78, Recall=80.09, F1=78.92, AUROC=95.01)  # 32862
REC_REPORTED = [93.33, 85.33, 84.67, 77.18, 85.33, 73.33]
REC_MEASURED = [95.33, 68.00, 86.00, 78.52, 78.67, 74.00]   # 32862, released weights
REC_PRUNED   = [93.33, 86.00, 85.33, 73.83, 88.00, 80.00]   # 32879, s40 -> 50% -> fine-tune

# sparsity training, share of prunable gammas below 0.05 (%) and val EMR (%)
ADAM = {  # L1 added to the gradient, optimizer = Adam
    "1e-4": dict(val=[64.77,66.07,67.21,68.18,68.34,67.86,69.32,69.64,68.83,70.45,69.48,70.45,
                      68.34,68.18,69.16,68.67,70.29,70.13], start=1),                    # 32864
    "1e-2": dict(val=[68.83,69.16,69.97,69.64,71.43,70.62,69.81,70.62], start=18),        # 32865
    "3e-2": dict(val=[65.10,66.07,67.21,69.32,68.99,68.34,69.32,70.13,69.81,68.34,68.83,69.48,
                      68.67,69.32,68.67,69.16,68.99,69.16,68.83,69.32,69.81,69.48,71.43,71.27,
                      70.94], start=1),                                                  # 32866
}
PROX = {  # proximal soft-threshold after the optimizer step
    1e-5: dict(job=32869, lt05=[0.2]*25,
               val=[65.10,66.56,67.21,69.48,68.67,68.67,69.64,70.29,69.81,68.83,68.67,69.32,68.99,
                    68.99,68.99,69.32,69.48,69.97,69.32,69.48,69.97,71.10,71.10,71.27,71.59]),
    5e-5: dict(job=32877,
               lt05=[0.2,0.2,0.2,0.2,0.2,0.3,0.3,0.4,0.6,0.7,1.0,1.4,1.9,2.4,3.2,4.1,5.3,6.8,8.5,
                     10.3,12.2,14.5,16.9,19.7,23.2,26.7,30.9,35.1,39.6,44.8,50.1,55.2,61.5,66.8,
                     71.6,76.4,80.7,84.8,87.9,90.7],
               val=[64.29,66.07,68.02,69.64,70.29,69.97,69.97,69.48,69.64,68.67,69.16,68.02,69.16,
                    68.83,68.51,68.51,67.05,68.02,69.48,68.83,68.67,67.86,68.51,67.21,66.40,64.45,
                    65.10,63.47,62.66,60.23,61.36,60.23,56.82,55.52,51.62,50.49,46.75,46.43,43.99,
                    40.75]),
    8e-5: dict(job=32875,
               lt05=[0.2,0.2,0.2,0.3,0.4,0.6,1.1,1.8,2.6,4.1,6.1,8.9,11.8,15.6,19.9,25.3,31.9,38.8,
                     47.1,55.5,64.9,72.7,79.9,86.2,90.7,93.7,95.9,97.3,98.2,98.6],
               val=[64.45,65.26,67.86,69.32,69.64,68.83,67.86,68.18,67.86,66.88,68.51,67.86,68.51,
                    67.05,66.07,62.66,62.34,61.04,57.63,56.49,51.30,48.70,42.53,38.80,35.55,34.42,
                    31.98,31.98,31.98,31.98]),
    1.2e-4: dict(job=32873,
                 lt05=[0.2,0.2,0.3,0.7,1.4,2.6,5.1,8.8,13.6,19.9,28.7,38.9,51.3,65.2,76.6,86.3,92.4,
                       95.9,97.8,98.6,99.0,99.4,99.6,99.8,99.8,99.9,99.9,99.9,99.9,99.9],
                 val=[64.12,66.23,67.21,68.34,68.34,67.21,67.53,65.58,64.12,61.04,58.44,55.03,50.32,
                      43.99,36.04,32.31,29.55,30.84,31.82,31.66,33.12,33.60,34.74,35.71,37.18,36.36,
                      38.31,38.15,40.10,39.29]),
    2e-4: dict(job=32871,
               lt05=[0.2,0.4,1.4,4.1,10.3,20.0,35.4,55.8,76.7,90.7,96.8,98.6,99.3,99.7,99.8,99.9,
                     99.9,99.9,99.9,100.0,100.0,100.0,100.0,100.0,100.0],
               val=[64.12,66.23,66.56,64.12,61.69,55.36,44.16,34.58,25.16,24.19,28.08,31.33,31.49,
                    33.93,33.93,34.74,35.55,35.06,34.90,35.71,36.69,37.66,38.15,39.77,40.42]),
}
MEAN_GAMMA0 = 0.1912   # mean |gamma| of the 32 prunable BNs in the released checkpoint

# pruned + fine-tuned, test split. seeds from 32903-32911, singles from 32912-32916
PARAMS_FULL = 23_771_078
PRUNED = [  # (label, params, source, [EMR per seed], [F1], [AUROC])
    ("50%", 12_592_289, "s40", [64.24, 67.56, 69.30], [79.47, 81.22, 82.36], [95.11, 95.40, 95.53]),
    ("65%",  8_917_792, "s40", [65.35, 67.88, 64.08], [80.28, 81.96, 80.14], [94.77, 95.44, 94.51]),
    ("80%",  5_957_070, "s40", [64.24, 64.87, 67.88], [80.41, 81.76, 82.29], [94.85, 95.02, 95.32]),
    ("85%",  5_208_971, "s40", [64.72], [80.76], [94.82]),
    ("90%",  4_635_089, "s40", [63.61], [79.59], [94.68]),
    ("90%*", 4_146_188, "last", [62.66], [78.38], [93.89]),
    ("95%*", 3_566_984, "last", [61.08], [77.12], [93.03]),
    ("98%*", 3_330_622, "last", [59.34], [77.03], [93.10]),
]
# unpruned model fine-tuned with the same recipe (3 seeds). Fill in from
# `python summarize.py`, e.g. CONTROL = dict(EMR=[..], F1=[..], AUROC=[..])
CONTROL = dict(EMR=[77.22, 74.68, 78.64], F1=[88.58, 86.34, 89.19], AUROC=[98.15, 97.80, 98.22])  # 32917-32919
UNPRUNABLE = 3.0e6     # downsample convs + stem + head + fixed conv sides

# fine-tuning recovery: val EMR before -> best val EMR after
RECOVERY = [  # (label, before, [after per seed])
    ("cut 50%  \u00b7 from 23%-sparse",  23.38, [67.21]),
    ("cut 50%  \u00b7 from 45%-sparse",   5.03, [66.40, 66.23, 66.40]),
    ("cut 65%  \u00b7 from 45%-sparse",   8.93, [63.15, 61.69, 62.18]),
    ("cut 80%  \u00b7 from 45%-sparse",   3.41, [61.20, 64.12, 61.69]),
    ("cut 85%  \u00b7 from 45%-sparse",   3.57, [62.82]),
    ("cut 90%  \u00b7 from 45%-sparse",   2.92, [63.31]),
    ("cut 90%  \u00b7 from 91%-sparse",  23.05, [59.25]),
    ("cut 95%  \u00b7 from 91%-sparse",  25.00, [58.28]),
    ("cut 98%  \u00b7 from 91%-sparse",  24.68, [56.82]),
]

# channels kept per prunable BN, s40 pruned at ratio 0.5 (prune_slim.py output)
LAYERS = [  # (stage, block, bn1 kept, bn2 kept, width)
    (1, 0, 25, 13, 64), (1, 1, 21, 23, 64), (1, 2, 24, 57, 64),
    (2, 0, 80, 73, 128), (2, 1, 13, 23, 128), (2, 2, 35, 55, 128), (2, 3, 21, 83, 128),
    (3, 0, 229, 76, 256), (3, 1, 26, 103, 256), (3, 2, 26, 113, 256), (3, 3, 33, 51, 256),
    (3, 4, 45, 72, 256), (3, 5, 89, 112, 256),
    (4, 0, 459, 264, 512), (4, 1, 282, 429, 512), (4, 2, 437, 402, 512),
]


def lam(x):
    m, e = f"{x:.1e}".split("e")
    m = m.rstrip("0").rstrip(".")
    return f"{m}e{int(e)}"


def mean_std(v):
    v = np.asarray(v, float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def binom_se(p_pct, n=632):
    p = p_pct / 100
    return 100 * np.sqrt(p * (1 - p) / n)


def save(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# ------------------------------------------------------------------ 01
def plot_reproduction(out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6),
                                 gridspec_kw={"width_ratios": [1, 1.05], "wspace": 0.55})
    for ax, labels, rep, got, title in [
        (a1, list(REPORTED), list(REPORTED.values()), list(MEASURED.values()), "aggregate metrics"),
        (a2, CLASSES, REC_REPORTED, REC_MEASURED, "recall per class"),
    ]:
        y = np.arange(len(labels))[::-1]
        for yi, r, g in zip(y, rep, got):
            gap = g - r
            col = RUST if gap < -5 else (AMBER if gap < -1.5 else TEAL)
            ax.plot([r, g], [yi, yi], color=col, lw=3, alpha=0.55, zorder=1,
                    solid_capstyle="round")
            ax.scatter(r, yi, s=70, color="white", edgecolor=INK, lw=1.5, zorder=3)
            ax.scatter(g, yi, s=70, color=col, zorder=3)
            ax.text(max(r, g) + 1.2, yi, f"{gap:+.1f}", va="center", fontsize=10,
                    color=col, fontweight="bold")
        ax.set_yticks(y); ax.set_yticklabels(labels)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("%")
        ax.set_title(title, loc="left", fontsize=11, color=MUTED)
        ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
    a1.set_xlim(60, 102); a2.set_xlim(60, 102)
    handles = [Line2D([], [], marker="o", ls="", mfc="white", mec=INK, ms=8, label="reported (bikit-models)"),
               Line2D([], [], marker="o", ls="", color=INK, ms=8, label="measured, released weights")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    save(fig, out, "01_reproduction.png")


# ------------------------------------------------------------------ 02
def plot_adam_vs_prox(out):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    cols = {5e-5: TEAL, 8e-5: SLATE, 1.2e-4: AMBER, 2e-4: RUST}
    for l, c in cols.items():
        d = PROX[l]
        ep = np.arange(1, len(d["lt05"]) + 1)
        ax.plot(ep, d["lt05"], color=c, lw=2.4, label=f"proximal step, \u03bb = {lam(l)}")
    for name, d in ADAM.items():
        ep = np.arange(d["start"], d["start"] + len(d["val"]))
        ax.plot(ep, [0.2] * len(ep), color=INK, lw=3, ls=(0, (4, 2)), zorder=5,
                label="L1 in the gradient + Adam" if name == "1e-4" else None)
    ax.annotate("L1 in the gradient + Adam\n\u03bb = 1e-4, 1e-2, 3e-2 (300\u00d7)\nall stuck at 0.2%",
                xy=(25, 0.6), xytext=(30.8, 9), fontsize=9.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.set_xlabel("epoch of sparsity training")
    ax.set_ylabel("prunable channels with |\u03b3| < 0.05  (%)")
    ax.set_ylim(-3, 103); ax.set_xlim(0, 41)
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9.5)
    save(fig, out, "02_adam_vs_proximal.png")


# ------------------------------------------------------------------ 03
def plot_shrinkage(out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"wspace": 0.28})
    cols = {1e-5: GREY, 5e-5: TEAL, 8e-5: SLATE, 1.2e-4: AMBER, 2e-4: RUST}
    for l, c in cols.items():
        d = PROX[l]
        ep = np.arange(1, len(d["lt05"]) + 1)
        budget = l * STEPS_PER_EPOCH * ep
        a1.plot(budget, d["lt05"], color=c, lw=2.4, marker="o", ms=3, label=f"\u03bb = {lam(l)}")
        if l != 1e-5:
            sp, va = np.array(d["lt05"]), np.array(d["val"])
            stop = np.argmax(sp >= 95) + 1 if (sp >= 95).any() else len(sp)
            a2.plot(sp[:stop], va[:stop], color=c, lw=2.2, marker="o", ms=3)
    a1.axvline(MEAN_GAMMA0, color=INK, ls=":", lw=1.3)
    a1.text(MEAN_GAMMA0 + 0.006, 8, "mean |\u03b3| at start\n= 0.19", fontsize=9, color=INK)
    a1.set_xlim(0, 0.36)
    a1.set_xlabel("cumulative shrinkage per \u03b3   (\u03bb \u00d7 steps)")
    a1.set_ylabel("channels with |\u03b3| < 0.05  (%)")
    a1.set_title("one curve: only \u03bb \u00d7 steps matters", loc="left", fontsize=11, color=MUTED)
    a1.legend(loc="lower right", fontsize=9)
    a2.set_xlabel("channels with |\u03b3| < 0.05  (%)")
    a2.set_ylabel("val exact match ratio (%)")
    a2.set_title("same sparsity, higher accuracy when slower", loc="left", fontsize=11, color=MUTED)
    a2.set_xlim(-2, 97)
    for ax in (a1, a2):
        ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    save(fig, out, "03_shrinkage_budget.png")


# ------------------------------------------------------------------ 04
def plot_layers(out):
    stage_col = {1: SLATE, 2: TEAL, 3: AMBER, 4: RUST}
    fig, ax = plt.subplots(figsize=(12, 4.4))
    x, labels, xt = 0, [], []
    for st, blk, k1, k2, w in LAYERS:
        for j, k in enumerate((k1, k2)):
            ax.bar(x, 100 * k / w, color=stage_col[st], width=0.8,
                   alpha=0.95 if j == 0 else 0.6)
            x += 1
        xt.append(x - 1.5); labels.append(f"{st}.{blk}")
        x += 0.6
    ax.axhline(50, color=INK, ls="--", lw=1)
    ax.axhline(10, color=MUTED, ls=":", lw=1.2)
    ax.set_xticks(xt); ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("bottleneck block (stage.block)    darker bar = bn1, lighter = bn2")
    ax.set_ylabel("channels kept (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=stage_col[s]) for s in stage_col]
    labels = [f"stage {s} ({w} wide)" for s, w in zip(stage_col, (64, 128, 256, 512))]
    handles += [Line2D([], [], color=INK, ls="--"), Line2D([], [], color=MUTED, ls=":", lw=1.2)]
    labels += ["50% kept overall", "10% floor per layer"]
    ax.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=9)
    ax.set_ylim(0, 100)
    save(fig, out, "04_channels_kept_per_layer.png")


# ------------------------------------------------------------------ 05
def plot_recovery(out):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    y = np.arange(len(RECOVERY))[::-1]
    for yi, (lab, before, after) in zip(y, RECOVERY):
        m, s = mean_std(after)
        c = RUST if "91%" in lab else TEAL
        ax.plot([before, m], [yi, yi], color=c, lw=2.5, alpha=0.45, zorder=1)
        ax.scatter(before, yi, s=55, color="white", edgecolor=c, lw=1.6, zorder=3)
        ax.errorbar(m, yi, xerr=s if s else None, fmt="o", color=c, ms=7, capsize=3, zorder=3)
        ax.text(m + 1.5 + s, yi, f"{m:.1f}", va="center", fontsize=9.5, color=c)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in RECOVERY])
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 78)
    ax.set_xlabel("val exact match ratio (%)")
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    handles = [Line2D([], [], marker="o", ls="", mfc="white", mec=INK, ms=8, label="right after pruning"),
               Line2D([], [], marker="o", ls="", color=INK, ms=8, label="after 40 epochs fine-tuning")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=2, fontsize=9.5)
    save(fig, out, "05_finetune_recovery.png")


# ------------------------------------------------------------------ 06
def _curve(ax, idx, ref_meas, ref_rep, ylabel, show_ceiling=True):
    for lab, n, src, *vals in PRUNED:
        v = vals[idx]
        m, s = mean_std(v)
        x = 100 * (1 - n / PARAMS_FULL)
        if len(v) > 1:
            ax.errorbar(x, m, yerr=s, fmt="o", color=TEAL, ms=8, capsize=4, lw=1.6, zorder=3)
        else:
            c = TEAL if src == "s40" else RUST
            if idx == 0:
                ax.errorbar(x, m, yerr=binom_se(m), fmt="none", ecolor="#CFC8BC", lw=5, zorder=1)
            ax.plot(x, m, "o" if src == "s40" else "D", mfc="white", mec=c, mew=1.8, ms=8, zorder=3)
    ax.axhline(ref_rep, color=RUST, ls="--", lw=1.3)
    ax.axhline(ref_meas, color=INK, ls=":", lw=1.5)
    if CONTROL is not None:
        key = ["EMR", "F1", "AUROC"][idx]
        m, s = mean_std(CONTROL[key])
        ax.axhspan(m - s, m + s, color=TEAL, alpha=0.12, lw=0)
        ax.axhline(m, color=TEAL, lw=1.3)
    if show_ceiling:
        lim = 100 * (1 - UNPRUNABLE / PARAMS_FULL)
        ax.axvspan(lim, 100, color=SAND, lw=0, zorder=0)
    ax.set_xlim(40, 92)
    ax.set_ylabel(ylabel)
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)


def plot_curve(out):
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    _curve(ax, 0, MEASURED["EMR"], REPORTED["EMR"], "test exact match ratio (%)")
    ax.set_ylim(55, 81)
    ax.set_xlabel("parameters removed (%)")
    ax.text(41, REPORTED["EMR"] + 0.35, "reported for the full model (71.4)", fontsize=9, color=RUST)
    ax.text(41, MEASURED["EMR"] - 0.75, "released checkpoint as measured (64.2)", fontsize=9, color=INK)
    if CONTROL is not None:
        m, _ = mean_std(CONTROL["EMR"])
        ax.text(41, m + 2.3, f"unpruned, same fine-tuning ({m:.1f} \u00b1 {mean_std(CONTROL['EMR'])[1]:.1f})",
                fontsize=9, color=TEAL)
    lim = 100 * (1 - UNPRUNABLE / PARAMS_FULL)
    ax.text(lim + 0.4, 80.4, "method limit\n(~3M unprunable)", fontsize=8.5, color=MUTED, va="top")
    xs = [100 * (1 - p[1] / PARAMS_FULL) for p in PRUNED]
    ms = [mean_std(p[3])[0] for p in PRUNED]
    ax.plot(xs, ms, color=MUTED, lw=1, alpha=0.5, zorder=0)
    for x, m, p in zip(xs, ms, PRUNED):
        ax.annotate(f"{p[1]/1e6:.1f}M", (x, m), textcoords="offset points",
                    xytext=(9, 4), ha="left", fontsize=8, color=MUTED)
    handles = [
        Line2D([], [], marker="o", ls="-", color=TEAL, ms=8, label="3 seeds, mean \u00b1 std"),
        Line2D([], [], marker="o", ls="", mfc="white", mec=TEAL, mew=1.8, ms=8, label="1 seed, from 45%-sparse checkpoint"),
        Line2D([], [], marker="D", ls="", mfc="white", mec=RUST, mew=1.8, ms=8, label="1 seed, from 91%-sparse checkpoint"),
        Line2D([], [], color="#CFC8BC", lw=5, label="\u00b11 SE of the 632-image test set"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=9)
    save(fig, out, "06_accuracy_vs_size.png")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"wspace": 0.3})
    for i, (ax, key, yl) in enumerate(zip(axes, ["EMR", "F1", "AUROC"],
                                          ["exact match ratio (%)", "F1 macro (%)", "AUROC macro (%)"])):
        _curve(ax, i, MEASURED[key], REPORTED[key], yl)
        ax.set_xlabel("parameters removed (%)")
        ax.set_title(key, loc="left", fontsize=11, color=MUTED)
    handles = [Line2D([], [], color=TEAL, lw=1.3, label="unpruned, same fine-tuning (\u00b1 std band)"),
               Line2D([], [], color=RUST, ls="--", label="reported, full model"),
               Line2D([], [], color=INK, ls=":", label="released checkpoint, measured")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    save(fig, out, "06b_metrics_vs_size.png")


# ------------------------------------------------------------------ 07
def plot_per_class(out):
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    y = np.arange(len(CLASSES))[::-1]
    for yi, r, g, p in zip(y, REC_REPORTED, REC_MEASURED, REC_PRUNED):
        ax.annotate("", xy=(p, yi), xytext=(g, yi),
                    arrowprops=dict(arrowstyle="-|>", color=TEAL if p >= g else RUST,
                                    lw=1.8, alpha=0.6, shrinkA=6, shrinkB=6))
        ax.scatter(r, yi, s=150, facecolor="none", edgecolor=MUTED, lw=1.6, zorder=3)
        ax.scatter(g, yi, s=65, color=SLATE, zorder=4)
        ax.scatter(p, yi, s=65, color=TEAL, zorder=4)
    ax.set_yticks(y); ax.set_yticklabels(CLASSES)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlim(64, 100); ax.set_xlabel("test recall (%)")
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    iy = CLASSES.index("crack"); ie = CLASSES.index("efflorescence")
    ax.text(REC_PRUNED[iy] + 1.8, y[iy], "+18 pts after fine-tuning", va="center",
            fontsize=9.5, color=TEAL)
    ax.text(REC_MEASURED[ie] + 1.8, y[ie], "lowest class after fine-tuning", va="center",
            fontsize=9.5, color=RUST)
    handles = [Line2D([], [], marker="o", ls="", mfc="none", mec=MUTED, mew=1.6, ms=11, label="reported, full model"),
               Line2D([], [], marker="o", ls="", color=SLATE, ms=8, label="released weights, measured"),
               Line2D([], [], marker="o", ls="", color=TEAL, ms=8, label="47% smaller, fine-tuned (one run)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=9.5)
    save(fig, out, "07_per_class_recall.png")


# ------------------------------------------------------------------ 08 (cluster)
def plot_gammas(out):
    import torch
    ckpts = [("released", "~/.cache/bikit-models/codebrim-classif-balanced_ResNet50_hta.pth"),
             ("epoch 25 (s20)", "runs/sparsity_lam5e-5/s20.pth"),
             ("epoch 30 (s40)", "runs/sparsity_lam5e-5/s40.pth"),
             ("epoch 33 (s60)", "runs/sparsity_lam5e-5/s60.pth"),
             ("epoch 40 (last)", "runs/sparsity_lam5e-5/last.pth")]
    ckpts = [(l, os.path.expanduser(p)) for l, p in ckpts if os.path.exists(os.path.expanduser(p))]
    if not ckpts:
        print("08: no checkpoints found, skipped"); return
    cols = [SLATE, TEAL, AMBER, RUST, INK]
    fig, axes = plt.subplots(len(ckpts), 1, figsize=(9, 1.25 * len(ckpts) + 0.8), sharex=True)
    axes = np.atleast_1d(axes)
    bins = np.linspace(0, 0.6, 61)
    for ax, (lab, p), c in zip(axes, ckpts, cols):
        sd = torch.load(p, map_location="cpu", weights_only=False)["state_dict"]
        g = torch.cat([v.abs().flatten() for k, v in sd.items()
                       if k.startswith("base.") and k.endswith(".weight") and v.dim() == 1
                       and k.split(".")[-2] in ("bn1", "bn2")]).numpy()
        ax.hist(np.clip(g, 0, 0.6), bins=bins, color=c, alpha=0.85)
        ax.axvline(0.05, color=INK, ls=":", lw=1)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.text(0.99, 0.75, f"{lab}\n{100*(g < 0.05).mean():.0f}% below 0.05",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color=c)
    axes[-1].set_xlabel("|\u03b3| of prunable BatchNorm channels   (\u03bb = 5e-5)")
    save(fig, out, "08_gamma_histograms.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figs/presentation")
    ap.add_argument("--gammas", action="store_true", help="also plot 08 from checkpoints")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    plot_reproduction(a.out)
    plot_adam_vs_prox(a.out)
    plot_shrinkage(a.out)
    plot_layers(a.out)
    plot_recovery(a.out)
    plot_curve(a.out)
    plot_per_class(a.out)
    if a.gammas:
        plot_gammas(a.out)


if __name__ == "__main__":
    main()
