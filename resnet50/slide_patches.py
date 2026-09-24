#!/usr/bin/env python
"""Example test patches for the slide: all right / one defect missed / wrong.

For every chosen patch it writes
  <cat>_<k>_img.png    the patch with "Target: ..." / "Pred: ..." written on it
  <cat>_<k>_crop.png   same, center-cropped to the slide box aspect
  <cat>_<k>_probs.png  the six sigmoid probabilities with the 0.5 threshold
plus contact_sheet.png to choose from.

  python slide_patches.py
"""
import os, argparse
import numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch

# datasets.json order, slide wording
NAMES = ["background", "crack", "efflorescence", "spallation", "exposed bars", "corrosion stain"]

INK = "#1B2530"          # slide title colour
COL = {"found": "#2A6F74",       # teal   (ALL SIX RIGHT)
       "missed": "#D9962B",      # amber
       "false alarm": "#B5451B", # rust   (WRONG)
       "absent": "#B9B3A8"}      # neutral grey


# ---------------------------------------------------------------- selection
def pick(P, T, n_keep, sizes, min_aspect):
    pred = (P >= 0.5).astype(int)
    fn = (T == 1) & (pred == 0)
    fp = (T == 0) & (pred == 1)
    exact = (pred == T).all(1)
    n_def = T[:, 1:].sum(1)

    margin = np.abs(P - 0.5).min(1)
    right = np.where(exact & (n_def >= 2))[0]
    if len(right) < n_keep:
        right = np.where(exact & (n_def >= 1))[0]
    right = right[np.argsort(-margin[right])]

    miss = np.where((fn.sum(1) == 1) & (fp.sum(1) == 0) & (n_def >= 2))[0]
    if len(miss):
        miss = miss[np.argsort([P[i][fn[i]][0] for i in miss])]

    wrong = np.where(((pred & T).sum(1) == 0) & (pred.sum(1) >= 1))[0]
    if len(wrong):
        wrong = wrong[np.argsort([-P[i][pred[i] == 1].max() for i in wrong])]

    def prefer_wide(idxs):
        good = [i for i in idxs if sizes(i)[0] / sizes(i)[1] >= min_aspect]
        return np.array(good + [i for i in idxs if i not in set(good)], dtype=int)

    return pred, {"all_right": prefer_wide(right),
                  "one_missed": prefer_wide(miss),
                  "wrong": prefer_wide(wrong)}


# ---------------------------------------------------------------- drawing
def names(vec):
    idx = np.where(vec)[0]
    return ", ".join(NAMES[i] for i in idx) if len(idx) else "none"


def crop_to_aspect(img, r):
    w, h = img.size
    if w / h > r:
        nw = int(h * r); x = (w - nw) // 2
        return img.crop((x, 0, x + nw, h))
    nh = int(w / r); y = (h - nh) // 2
    return img.crop((0, y, w, y + nh))


def label_image(img, truth, pred, width=1200, font_frac=0.024):
    img = img.convert("RGB")
    img = img.resize((width, int(img.height * width / img.width)), Image.BICUBIC)
    size = max(12, int(font_frac * width))
    font = ImageFont.truetype(fm.findfont(fm.FontProperties(family="DejaVu Sans")), size)
    lines = [f"Target: {names(truth)}", f"Pred:   {names(pred)}"]

    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    pad = int(0.6 * size)
    lh = d.textbbox((0, 0), "Ag", font=font)[3] + int(0.25 * size)
    tw = max(d.textbbox((0, 0), l, font=font)[2] for l in lines)
    x0, y1 = pad, img.height - pad
    y0 = y1 - len(lines) * lh - pad
    d.rectangle([x0 - pad // 2, y0 - pad // 2, x0 + tw + pad, y1],
                fill=(15, 20, 26, 170))
    for k, l in enumerate(lines):
        d.text((x0 + pad // 2, y0 + k * lh), l, font=font, fill=(245, 245, 245, 255))
    return Image.alpha_composite(img.convert("RGBA"), over).convert("RGB")


def prob_chart(p, truth, pred, path, title=None):
    status = []
    for i in range(len(NAMES)):
        if truth[i] and pred[i]:   status.append("found")
        elif truth[i]:             status.append("missed")
        elif pred[i]:              status.append("false alarm")
        else:                      status.append("absent")

    fig, ax = plt.subplots(figsize=(6.9, 3.0), dpi=200)
    y = np.arange(len(NAMES))[::-1]
    ax.barh(y, [1] * len(NAMES), color="#E3DFD8", height=0.62, zorder=1)
    ax.barh(y, p, color=[COL[s] for s in status], height=0.62, zorder=2)
    for yi, pi in zip(y, p):
        if pi > 0.88:   # label inside the bar, in white
            ax.text(pi - 0.015, yi, f"{pi:.2f}", va="center", ha="right",
                    fontsize=8.5, color="white", zorder=3)
        else:
            ax.text(pi + 0.015, yi, f"{pi:.2f}", va="center",
                    fontsize=8.5, color=INK, zorder=5,
                    bbox=dict(facecolor="#E3DFD8", edgecolor="none", pad=0.6))
    ax.axvline(0.5, color=INK, ls="--", lw=1.2, zorder=4)
    ax.text(0.5, len(NAMES) - 0.2, "threshold 0.5", ha="center", va="bottom",
            fontsize=8, color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels(NAMES, fontsize=9.5, color=INK)
    for lbl, t in zip(ax.get_yticklabels(), truth):
        if t:
            lbl.set_fontweight("bold")          # bold = actually present
    ax.set_xlim(0, 1); ax.set_ylim(-0.6, len(NAMES) - 0.1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.tick_params(axis="x", labelsize=8, colors=INK)
    ax.tick_params(axis="y", length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#9A958C")
    ax.set_xlabel("predicted probability", fontsize=8.5, color=INK)
    if title:
        ax.set_title(title, fontsize=9, color=INK, loc="left")

    used = [s for s in ("found", "missed", "false alarm", "absent") if s in status]
    ax.legend(handles=[Patch(color=COL[s], label=s) for s in used],
              loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=len(used),
              frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, transparent=True, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="results/codebrim-classif-balanced_ResNet50_hta_test_raw.npz")
    ap.add_argument("--csv", default=os.path.expanduser(
        "~/work/project-aida/building-inspection-toolkit/bikit/data/codebrim-classif-balanced.csv"))
    ap.add_argument("--root", default=os.path.expanduser("~/.cache/bikit/"))
    ap.add_argument("--n", type=int, default=4, help="candidates per category")
    ap.add_argument("--aspect", type=float, default=2.3, help="slide box width/height")
    ap.add_argument("--min-aspect", type=float, default=0.8)
    ap.add_argument("--font-frac", type=float, default=0.024,
                    help="text height as a fraction of image width")
    ap.add_argument("--out", default="figs/slide_patches")
    a = ap.parse_args()

    d = np.load(a.npz)
    P, T = d["probs"], d["targets"].astype(int)
    print(f"sanity: EMR from this file = {100*((P >= 0.5) == T).all(1).mean():.2f} (expect 64.24)\n")

    test = pd.read_csv(a.csv).query("split_type == 'test'").reset_index(drop=True)
    assert len(test) == len(P), f"{len(test)} rows vs {len(P)} predictions"
    path = lambda i: os.path.join(a.root, test.img_path[i])
    cache = {}
    sizes = lambda i: cache.setdefault(i, Image.open(path(i)).size)

    pred, cats = pick(P, T, a.n, sizes, a.min_aspect)
    os.makedirs(a.out, exist_ok=True)
    sheet = []
    for cat, idxs in cats.items():
        print(f"=== {cat}: {len(idxs)} candidates, saving {min(a.n, len(idxs))}")
        for k, i in enumerate(idxs[:a.n]):
            base = os.path.join(a.out, f"{cat}_{k}")
            img = Image.open(path(i)).convert("RGB")
            full = label_image(img, T[i], pred[i], font_frac=a.font_frac)
            full.save(base + "_img.png")
            label_image(crop_to_aspect(img, a.aspect), T[i], pred[i],
                        font_frac=a.font_frac).save(base + "_crop.png")
            prob_chart(P[i], T[i], pred[i], base + "_probs.png")
            print(f"  [{cat}_{k}] {test.img_name[i]}  {img.size[0]}x{img.size[1]}")
            print(f"      Target: {names(T[i])}")
            print(f"      Pred:   {names(pred[i])}")
            sheet.append((f"{cat}_{k}", full, base + "_probs.png"))

    if sheet:
        fig, axes = plt.subplots(len(sheet), 2, figsize=(9, 2.6 * len(sheet)),
                                 gridspec_kw={"width_ratios": [1, 1.3]})
        axes = np.atleast_2d(axes)
        for (name, im, pp), (a1, a2) in zip(sheet, axes):
            a1.imshow(im); a1.set_title(name, fontsize=9, loc="left"); a1.axis("off")
            a2.imshow(Image.open(pp)); a2.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(a.out, "contact_sheet.png"), dpi=110)
    print(f"\nwrote {a.out}/  (open contact_sheet.png first)")


if __name__ == "__main__":
    main()
