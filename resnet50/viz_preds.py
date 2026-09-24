#!/usr/bin/env python
"""Contact sheet of test images with true labels vs model predictions."""
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

CLASSES = ["NoDamage", "Crack", "Spalling", "Efflorescence", "BarsExposed", "Rust"]
SHORT   = ["ND", "Cr", "Sp", "Ef", "BE", "Ru"]

p = argparse.ArgumentParser()
p.add_argument("--npz", default="results/codebrim-classif-balanced_ResNet50_hta_test_raw.npz")
p.add_argument("--mode", default="wrong", choices=["wrong", "right", "all"],
               help="which samples to show")
p.add_argument("--n", type=int, default=24)
p.add_argument("--cols", type=int, default=6)
p.add_argument("--thr", type=float, default=0.5)
p.add_argument("--out", default="figs/preds.png")
a = p.parse_args()

d = np.load(a.npz)
probs, tgts = d["probs"], d["targets"]
preds = (probs >= a.thr).astype(int)

csv = os.path.expanduser("~/work/project-aida/building-inspection-toolkit/bikit/data/codebrim-classif-balanced.csv")
root = os.path.expanduser("~/.cache/bikit/")
df = pd.read_csv(csv)
test = df[df.split_type == "test"].reset_index(drop=True)
assert len(test) == len(probs), f"{len(test)} rows vs {len(probs)} preds"

exact = (preds == tgts).all(axis=1)
if a.mode == "wrong":
    idx = np.where(~exact)[0]
elif a.mode == "right":
    idx = np.where(exact)[0]
else:
    idx = np.arange(len(probs))
print(f"{a.mode}: {len(idx)} samples (showing up to {a.n})")

rng = np.random.default_rng(0)
if len(idx) > a.n:
    idx = rng.choice(idx, a.n, replace=False)
idx = np.sort(idx)

cols = a.cols
rows = int(np.ceil(len(idx) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.6))
axes = np.atleast_1d(axes).ravel()

def fmt(vec, pr=None):
    parts = []
    for i, v in enumerate(vec):
        if v:
            parts.append(SHORT[i] if pr is None else f"{SHORT[i]}{pr[i]:.2f}"[:7])
    return ",".join(parts) if parts else "-"

for ax, i in zip(axes, idx):
    img = Image.open(os.path.join(root, test.img_path[i])).convert("RGB")
    ax.imshow(img)
    ok = exact[i]
    ax.set_title(f"T: {fmt(tgts[i])}\nP: {fmt(preds[i], probs[i])}",
                 fontsize=7, color=("green" if ok else "red"), loc="left")
    ax.axis("off")
for ax in axes[len(idx):]:
    ax.axis("off")

os.makedirs(os.path.dirname(a.out), exist_ok=True)
plt.tight_layout()
plt.savefig(a.out, dpi=110, bbox_inches="tight")
print("wrote", a.out)
