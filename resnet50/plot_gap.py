#!/usr/bin/env python
"""Reported vs reproduced: EMR headline + per-class recall."""
import json, os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["NoDamage", "Crack", "Efflorescence", "Spalling", "BarsExposed", "Rust"]
REPORTED_RECALL = [93.33, 85.33, 77.18, 84.67, 85.33, 73.33]
REPORTED_EMR = 71.36

ap = argparse.ArgumentParser()
ap.add_argument("--json", default="results/codebrim-classif-balanced_ResNet50_hta_test.json")
ap.add_argument("--out", default="figs/gap.png")
a = ap.parse_args()

m = json.load(open(a.json))["metrics"]
ours_emr = 100 * m["exact_match_ratio"]
ours_recall = [100 * r for r in m["per_class"]["recall"]]

BLUE, ORANGE = "#4C72B0", "#DD8452"
fig = plt.figure(figsize=(12, 5.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 2.1, 0.02], wspace=0.28)

# --- EMR headline ---
ax0 = fig.add_subplot(gs[0])
bars = ax0.bar(["Reported", "Ours"], [REPORTED_EMR, ours_emr],
               color=[BLUE, ORANGE], width=0.62)
for b, v in zip(bars, [REPORTED_EMR, ours_emr]):
    ax0.text(b.get_x() + b.get_width()/2, v + 1.2, f"{v:.2f}",
             ha="center", fontsize=13, fontweight="bold")
d = ours_emr - REPORTED_EMR
ax0.set_title(f"Exact Match Ratio\n({d:+.2f} pts)", fontsize=13, fontweight="bold")
ax0.set_ylim(0, max(REPORTED_EMR, ours_emr) * 1.25)
ax0.set_ylabel("%", fontsize=11)
ax0.grid(axis="y", alpha=0.25)
ax0.set_axisbelow(True)
for s in ("top", "right"):
    ax0.spines[s].set_visible(False)

# --- per-class recall ---
ax1 = fig.add_subplot(gs[1])
x = np.arange(len(CLASSES)); w = 0.38
ax1.bar(x - w/2, REPORTED_RECALL, w, label="Reported", color=BLUE)
ax1.bar(x + w/2, ours_recall, w, label="Ours", color=ORANGE)
for i, (r, o) in enumerate(zip(REPORTED_RECALL, ours_recall)):
    ax1.text(i - w/2, r + 1.0, f"{r:.0f}", ha="center", fontsize=8.5, color=BLUE)
    ax1.text(i + w/2, o + 1.0, f"{o:.0f}", ha="center", fontsize=8.5, color=ORANGE)
    if abs(o - r) >= 5:
        ax1.text(i, max(r, o) + 6.5, f"{o-r:+.0f}", ha="center",
                 fontsize=9, fontweight="bold", color="#B23A48")
ax1.set_xticks(x)
ax1.set_xticklabels(CLASSES, fontsize=10)
ax1.set_ylim(0, 108)
ax1.set_ylabel("Recall (%)", fontsize=11)
ax1.set_title("Per-class recall", fontsize=13, fontweight="bold")
ax1.legend(frameon=False, fontsize=10, loc="lower right")
ax1.grid(axis="y", alpha=0.25)
ax1.set_axisbelow(True)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)

fig.suptitle("CODEBRIM · ResNet50 (bikit) — reported vs reproduced, test split",
             fontsize=14, fontweight="bold", y=0.99)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
plt.savefig(a.out, dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", a.out, f"| EMR {ours_emr:.2f} vs {REPORTED_EMR}")
