#!/usr/bin/env python
"""Final benchmark: every ViT and ResNet-50 variant timed on the same GPU, the same way.

Accuracy is test EMR, mean +/- std over seeds:
  ViT     from the ToMe sweeps (results/tome_*/tome_sweep.json): one json per run, one row per r
  ResNet  from ../resnet50/results/ft_*_test.json
Speed is measured here for each architecture (and each ToMe r): fp32, images/s at batch 64 and
latency at batch 1 (vit_table.speed). A point is on the Pareto frontier if no other point is both
faster and more accurate.

  python bench_all.py            # GPU node (bench_all.sh); --preview skips configs not finished yet
  -> results/bench_all.json, results/bench_all.md, results/bench_all.png
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch

from vit_model import DEFAULT_CKPT, encoder_dims, load_vit
from tome_vit import apply_tome, vit_flops
from vit_table import speed

HERE = Path(__file__).resolve().parent
RESNET_DIR = HERE.parent / "resnet50"
R_LIST = [0, 16, 32, 48, 64]
seeds = lambda f: [f.format(s=s) for s in range(3)]
pct = lambda r: f"{int(r) / 10 ** (len(r) - 1) * 100:.0f}%"


def tome_tokens(r, n=785, depth=12):
    """Tokens entering each block with r merges per block (as in tome_vit, CLS protected)."""
    out = []
    for _ in range(depth):
        out.append(n)
        n -= max(0, min(r, (n - 1) // 2))
    return out

# (family, label, architecture checkpoint, ToMe result jsons, ToMe r values)
VIT = ([("vit_released", "ViT-S/8 released", str(DEFAULT_CKPT), ["results/tome_sweep.json"], R_LIST),
        ("vit_tome", "ViT-S/8 fine-tuned", "runs/vit_ctrl_ep100_seed0/best.pth",
         seeds("results/tome_ctrl_ep100_seed{s}/tome_sweep.json"), R_LIST)]
       + [("vit_prune", f"ViT-S/8 {pct(r)} pruned", f"runs/vit_prune_r{r}_ep100_seed0/best.pth",
           seeds(f"results/tome_vit_prune_r{r}_ep100_seed{{s}}/tome_sweep.json"), R_LIST)
          for r in ["015", "03", "05", "065", "08"]]
       + [("vit_prune_mlp", f"ViT-S/8 {pct(r)} MLP-only", f"runs/vit_prune_mlp_r{r}_ep100_seed0/best.pth",
           seeds(f"results/tome_vit_prune_mlp_r{r}_ep100_seed{{s}}/tome_sweep.json"), R_LIST)
          for r in ["05", "065", "08"]])
RESNET = ([("resnet_released", "ResNet-50 released", str(Path.home() / ".cache/bikit-models/codebrim-classif-balanced_ResNet50_hta.pth"), [70.89]),
           ("resnet", "ResNet-50 fine-tuned", "runs/ft_control_ep160_seed0/best.pth",
            seeds("results/ft_control_ep160_seed{s}_test.json"))]
          + [("resnet", f"ResNet-50 {pct(r)} pruned", f"pruned/b160_r{r}.pth",
              seeds(f"results/ft_b160_r{r}_ep160_seed{{s}}_test.json"))
             for r in ["015", "03", "05", "065", "08"]])


def vit_points(dev, preview):
    pts = []
    for fam, label, ckpt, files, rs in VIT:
        have = [f for f in files if os.path.exists(f)]
        if len(have) < len(files) or not os.path.exists(ckpt):
            if not preview:
                raise SystemExit(f"{label}: missing {[f for f in files if f not in have] or ckpt} (use --preview)")
            if not have or not os.path.exists(ckpt):
                print(f"  {label:<26} pending"); continue
        rows = [{r["r"]: r for r in json.load(open(f))["rows"]} for f in have]
        model = load_vit(ckpt)[0]
        dims = encoder_dims(model.encoder)
        n_params = sum(p.numel() for p in model.parameters())
        for r in rs:
            if not all(r in d for d in rows):
                continue
            emr = [100 * d[r]["test"]["exact_match_ratio"] for d in rows]
            val = [100 * d[r]["val"]["exact_match_ratio"] for d in rows]
            apply_tome(model, r)
            ips, lat = speed(model, dev)
            pts.append({"family": fam, "label": label + (f" + ToMe r={r}" if r else ""), "tome_r": r,
                        "params": n_params, "gflops": vit_flops(tome_tokens(r), [r] * 12, dims) / 1e9,
                        "emr": emr, "val_emr": val, "img_per_s": ips, "latency_ms": lat})
            print(f"  {pts[-1]['label']:<40} {np.mean(emr):6.2f}  {ips:6.0f} img/s", flush=True)
        del model
    return pts


def resnet_points(dev, preview):
    sys.path.insert(0, str(RESNET_DIR))
    from prune_slim import load_pruned
    from paper_table import count_flops
    pts, cwd = [], os.getcwd()
    os.chdir(RESNET_DIR)
    try:
        for fam, label, ckpt, files in RESNET:
            if isinstance(files[0], (int, float)):
                emr = list(files)
            else:
                have = [f for f in files if os.path.exists(f)]
                if len(have) < len(files) and not preview:
                    raise SystemExit(f"{label}: missing {[f for f in files if f not in have]} (use --preview)")
                if not have:
                    print(f"  {label:<26} pending"); continue
                emr = [100 * json.load(open(f))["metrics"]["exact_match_ratio"] for f in have]
            model = load_pruned(os.path.expanduser(ckpt))[0].eval()
            gfl = count_flops(model) / 1e9
            ips, lat = speed(model, dev)
            pts.append({"family": fam, "label": label, "tome_r": None,
                        "params": sum(p.numel() for p in model.parameters()), "gflops": gfl,
                        "emr": emr, "val_emr": None, "img_per_s": ips, "latency_ms": lat})
            print(f"  {label:<40} {np.mean(emr):6.2f}  {ips:6.0f} img/s", flush=True)
            del model
    finally:
        os.chdir(cwd)
    return pts


def mark_frontier(pts):
    for p in pts:
        m, s = np.mean(p["emr"]), p["img_per_s"]
        p["frontier"] = not any((np.mean(q["emr"]) >= m and q["img_per_s"] >= s) and
                                (np.mean(q["emr"]) > m or q["img_per_s"] > s) for q in pts)


def write_md(pts, path):
    lines = ["| model | test EMR | params | GFLOPs | img/s (bs 64) | latency bs 1 | Pareto |",
             "|---|---|---|---|---|---|---|"]
    for p in sorted(pts, key=lambda p: -p["img_per_s"]):
        e = np.array(p["emr"])
        et = f"{e.mean():.2f} ± {e.std(ddof=1):.2f}" if len(e) > 1 else f"{e.mean():.2f}"
        lines.append(f"| {p['label']} | {et} | {p['params'] / 1e6:.2f}M | {p['gflops']:.2f} | "
                     f"{p['img_per_s']:.0f} | {p['latency_ms']:.2f} ms | {'yes' if p['frontier'] else ''} |")
    open(path, "w").write("\n".join(lines) + "\n")
    return "\n".join(lines)


def place_labels(fig, ax, labels, points, color):
    """Put each label at the first candidate offset that overlaps no placed label and no data point."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pix = [ax.transData.transform(pt) for pt in points]
    box = ax.get_window_extent(r)
    obstacles = [(px - 7, py - 7, px + 7, py + 7) for px, py in pix]          # markers (display px)
    hit = lambda b, o: not (b[2] < o[0] or b[0] > o[2] or b[3] < o[1] or b[1] > o[3])
    for text, x, y, pref, ha in labels:
        cands = [(pref, ha), ((7, 6), "left"), ((7, -14), "left"), ((-8, 6), "right"), ((-8, -14), "right"),
                 ((0, 11), "center"), ((0, -19), "center"), ((12, 16), "left"), ((-12, -24), "right")]
        for k, (off, h) in enumerate(cands):
            a = ax.annotate(text, (x, y), textcoords="offset points", xytext=off, ha=h, fontsize=9, color=color)
            bb = a.get_window_extent(r).expanded(1.05, 1.15)
            b = (bb.x0, bb.y0, bb.x1, bb.y1)
            inside = b[0] >= box.x0 + 2 and b[2] <= box.x1 - 2 and b[1] >= box.y0 + 2 and b[3] <= box.y1 - 2
            if (inside and not any(hit(b, o) for o in obstacles)) or k == len(cands) - 1:
                obstacles.append(b)
                break
            a.remove()


def plot(pts, path, gpu):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
    BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"      # categorical slots 1-3 (all-pairs safe)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(11, 6.6), facecolor=SURF)
    ax.set_facecolor(SURF)
    labels = []                                    # (text, x, y, preferred offset, ha), placed at the end

    def series(fam_filter, color, name, ls="-", hollow=False, label_fn=None, order=None,
               offset=(7, 6), ha="left"):
        s = [p for p in pts if fam_filter(p)]
        if not s:
            return
        s.sort(key=order or (lambda p: p["img_per_s"]))
        x = [p["img_per_s"] for p in s]
        y = [np.mean(p["emr"]) for p in s]
        e = [np.std(p["emr"], ddof=1) if len(p["emr"]) > 1 else 0 for p in s]
        ax.plot(x, y, ls=ls, color=color, lw=2, zorder=2, label=name)
        ax.errorbar(x, y, yerr=e, fmt="none", ecolor=color, elinewidth=1.2, capsize=0, alpha=0.6, zorder=2)
        ax.scatter(x, y, s=64, zorder=3, facecolor=SURF if hollow else color, edgecolor=color if hollow else SURF,
                   linewidths=2)
        if label_fn:
            for p, xi, yi in zip(s, x, y):
                t = label_fn(p)
                if t:
                    labels.append((t, xi, yi, offset, ha))

    short = lambda p: p["label"].split()[-2] if "pruned" in p["label"] or "MLP-only" in p["label"] else None
    series(lambda p: p["family"] == "resnet", BLUE, "ResNet-50: fine-tuned, then pruned (Network Slimming)",
           label_fn=lambda p: short(p) or "baseline")
    series(lambda p: p["family"] == "vit_prune" and p["tome_r"] == 0, ORANGE,
           "ViT-S/8: width-pruned (heads + MLP)", label_fn=short)
    series(lambda p: p["family"] == "vit_prune_mlp" and p["tome_r"] == 0, ORANGE,
           "ViT-S/8: width-pruned, MLP only (all heads kept)", ls="--", hollow=True, label_fn=short,
           offset=(-8, 7), ha="right")
    series(lambda p: p["family"] == "vit_tome", AQUA, "ViT-S/8: fine-tuned + Token Merging",
           label_fn=lambda p: {0: "fine-tuned", 16: None}.get(p["tome_r"], f"r={p['tome_r']}"))
    for fam, txt in (("vit_released", "ViT released"), ("resnet_released", "ResNet released")):
        for p in pts:
            if p["family"] == fam and not p["tome_r"]:
                ax.scatter([p["img_per_s"]], [np.mean(p["emr"])], s=64, marker="D", facecolor=MUTED,
                           edgecolor=SURF, linewidths=2, zorder=3)
                labels.append((txt, p["img_per_s"], np.mean(p["emr"]), (7, -13), "left"))

    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
    lo, hi = min(p["img_per_s"] for p in pts) / 1.25, max(p["img_per_s"] for p in pts) * 1.15
    ticks = [t for t in (100, 200, 300, 500, 1000, 2000, 3000, 5000, 10000) if lo <= t <= hi]
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{t:,}" for t in ticks]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(f"Measured throughput, images/s ({gpu}, fp32, batch 64, log scale)", color=INK2)
    ax.set_ylabel("Test exact match ratio (%), mean ± std over 3 seeds", color=INK2)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.grid(True, which="minor", axis="x", color=GRID, lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=INK2)
    ax.set_title("Accuracy vs real speed on CODEBRIM: higher and further right is better",
                 loc="left", color=INK, fontsize=13, pad=12)
    ax.scatter([], [], s=64, marker="D", facecolor=MUTED, edgecolor=SURF, label="Released checkpoints")
    place_labels(fig, ax, labels, [(p["img_per_s"], np.mean(p["emr"])) for p in pts], INK2)
    leg = ax.legend(loc="lower left", frameon=False, fontsize=10)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURF)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else "CPU"
    if not a.preview:                                       # check everything before timing anything
        need = [f for _, _, ck, fs, _ in VIT for f in fs + [ck]]
        need += [str(RESNET_DIR / f) for _, _, ck, fs in RESNET for f in [ck] + [x for x in fs if isinstance(x, str)]
                 if not f.startswith("/")] + [ck for _, _, ck, _ in RESNET if ck.startswith("/")]
        missing = [f for f in need if not os.path.exists(f)]
        if missing:
            raise SystemExit(f"missing {len(missing)} files, e.g. {missing[:3]} (use --preview)")
    print(f"device: {gpu}\nViT:")
    pts = vit_points(dev, a.preview)
    print("ResNet-50:")
    pts += resnet_points(dev, a.preview)
    mark_frontier(pts)
    os.makedirs("results", exist_ok=True)
    json.dump(pts, open("results/bench_all.json", "w"), indent=1)
    print("\n" + write_md(pts, "results/bench_all.md"))
    plot(pts, "results/bench_all.png", gpu)
    print("\nsaved -> results/bench_all.json, bench_all.md, bench_all.png")


if __name__ == "__main__":
    main()
