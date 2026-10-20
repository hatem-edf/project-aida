#!/usr/bin/env python
"""Reproduce the bikit-models README row for codebrim-classif-balanced_ViT_s8_1.

README row : EMR 79.59 | F1 89.82 | P 91.30 | R 88.54 | Acc 86.65 | AUROC -
Authors    : 503/632 test images fully correct = 79.59 % (codebrim_challenge README)

Two steps:
  1. Golden test: 5 test images whose per-class probabilities the model's authors
     published (run.ipynb in github.com/OALcementys/codebrim_challenge). A match
     means weights, preprocessing, class order AND image files equal theirs.
  2. Full evaluation of --split with the same metric definitions as the ResNet
     eval (../resnet50/eval_codebrim.py), so the two models are directly comparable.

  python eval_vit.py --golden-only     # login node, CPU, ~10 s
  sbatch eval_vit.sh                   # full test set on a GPU
"""
import argparse, json, os, time

import numpy as np
import torch
from torch.utils.data import DataLoader

from vit_model import DEFAULT_CKPT, build_transform, load_vit, reorder_index

DATASET = "codebrim-classif-balanced"

# Published outputs, in % and in the model's output order
# (NoDamage, Crack, Spalling, Efflorescence, BarsExposed, Rust).
GOLDEN_ORDER = ["NoDamage", "Crack", "Spalling", "Efflorescence", "BarsExposed", "Rust"]
GOLDEN = {
    "image_0000468_crop_0000001.png": [0.01, 0.01, 99.66, 0.00, 99.46, 92.00],
    "image_0000761_crop_0000006.png": [0.00, 6.06, 99.66, 0.01, 100.00, 96.13],
    "image_0001039_crop_0000004.png": [75.26, 0.00, 0.10, 0.03, 0.00, 0.00],
    "image_0001251_crop_0000002.png": [0.00, 17.40, 51.19, 0.01, 46.29, 83.54],
    "image_0001550_crop_0000005.png": [0.00, 54.15, 94.19, 0.69, 99.63, 63.89],
}
GOLDEN_TOL = 0.5   # percentage points; the reference is rounded to 0.01

# README targets. Its "Accuracy" (86.65) is below its EMR, impossible for the
# element-wise accuracy computed here, so it is a different metric: not compared.
TARGETS = {"exact_match_ratio": 79.59, "f1": 89.82, "precision": 91.30, "recall": 88.54}
TARGET_RECALL = {"NoDamage": 94.00, "Crack": 89.33, "Efflorescence": 85.91,
                 "Spalling": 84.00, "BarsExposed": 90.00, "Rust": 88.00}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out", default="results")
    p.add_argument("--tag", default=None, help="name for output files (default: checkpoint name)")
    p.add_argument("--golden-only", action="store_true")
    p.add_argument("--no-golden", action="store_true", help="skip the golden test (not released weights)")
    return p.parse_args()


def compute_metrics(probs, targets, threshold=0.5):
    """Identical definitions to the ResNet eval. probs, targets: (N, C) arrays."""
    preds = (probs >= threshold).astype(np.float32)
    tgt = targets.astype(np.float32)
    emr = float((preds == tgt).all(axis=1).mean())
    acc = float((preds == tgt).mean())
    tp = (preds * tgt).sum(axis=0)
    fp = (preds * (1 - tgt)).sum(axis=0)
    fn = ((1 - preds) * tgt).sum(axis=0)
    eps = 1e-12
    prec_c = tp / (tp + fp + eps)
    rec_c = tp / (tp + fn + eps)
    f1_c = 2 * prec_c * rec_c / (prec_c + rec_c + eps)
    macro = {"precision": float(prec_c.mean()), "recall": float(rec_c.mean()),
             "f1": float(f1_c.mean())}
    mp = tp.sum() / (tp.sum() + fp.sum() + eps)
    mr = tp.sum() / (tp.sum() + fn.sum() + eps)
    micro = {"precision": float(mp), "recall": float(mr),
             "f1": float(2 * mp * mr / (mp + mr + eps))}
    try:
        from sklearn.metrics import roc_auc_score
        aucs = [roc_auc_score(tgt[:, c], probs[:, c])
                for c in range(tgt.shape[1]) if len(np.unique(tgt[:, c])) > 1]
        auroc = float(np.mean(aucs)) if aucs else None
    except ImportError:
        auroc = None
    return {"exact_match_ratio": emr, "accuracy": acc, "macro": macro,
            "micro": micro, "auroc_macro": auroc,
            "per_class": {"precision": prec_c.tolist(), "recall": rec_c.tolist(),
                          "f1": f1_c.tolist(), "support": tgt.sum(axis=0).tolist()}}


@torch.no_grad()
def golden_test(model, model_names, tf, device):
    from bikit.datasets import BikitDataset
    assert model_names == GOLDEN_ORDER, model_names
    ds = BikitDataset(DATASET, split="test", transform=tf)
    names = ds.df["img_name"].tolist()
    print("golden test (ours vs published, %, model order: ND Cr Sp Ef BE Ru)")
    worst = 0.0
    for img_name, ref in GOLDEN.items():
        x, _ = ds[names.index(img_name)]
        ours = model(x.unsqueeze(0).to(device))[0].cpu().numpy() * 100
        diff = float(np.abs(ours - np.array(ref)).max())
        worst = max(worst, diff)
        print(f"  {img_name}  " + " ".join(f"{v:6.2f}" for v in ours) + f"   max|diff| {diff:.3f}")
    ok = worst <= GOLDEN_TOL
    print(f"golden test: {'PASS' if ok else 'FAIL  <-- do not trust the numbers below'}"
          f"  (worst {worst:.3f} pts, tolerance {GOLDEN_TOL})\n")
    return ok


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}" + (f"  gpu={torch.cuda.get_device_name(0)}" if device == "cuda" else ""))

    model, model_names, cp = load_vit(args.ckpt)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"checkpoint: {args.ckpt}")
    print(f"parameters: {n_params:,}   output order: {model_names}")

    tf = build_transform(cp["img_size"], mode="eval")
    golden_ok = None
    if not args.no_golden:
        golden_ok = golden_test(model, model_names, build_transform(cp["img_size"], mode="demo"), device)
    if args.golden_only:
        return

    from bikit.datasets import BikitDataset
    ds = BikitDataset(DATASET, split=args.split, transform=tf, return_type="pt")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=(device == "cuda"))
    perm = reorder_index(model_names, ds.class_names)   # model order -> label order
    print(f"{args.split} samples: {len(ds)}   label order: {ds.class_names}")

    P, P_raw, T = [], [], []
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for x, y in loader:
            raw = model(x.to(device, non_blocking=True), postprocess=False)  # sigmoid probs
            P.append(model.postprocess_pred(raw.clone()).cpu().numpy())       # = model(x)
            P_raw.append(raw.cpu().numpy())
            T.append(y.numpy())
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    probs = np.concatenate(P)[:, perm]
    probs_raw = np.concatenate(P_raw)[:, perm]
    tgts = np.concatenate(T)
    print(f"inference: {dt:.1f}s for {len(probs)} images ({1000 * dt / len(probs):.2f} ms/img, "
          f"incl. data loading)")

    m = compute_metrics(probs, tgts, args.threshold)
    m_raw = compute_metrics(probs_raw, tgts, args.threshold)
    show = args.split == "test"

    def line(label, val, key=None):
        tgt = f"   (target {TARGETS[key]:.2f})" if show and key else ""
        return f"{label:<16} {val}{tgt}"

    print("\n=== results (%) ===")
    print(line("ExactMatchRatio", f"{100 * m['exact_match_ratio']:.2f}", "exact_match_ratio"))
    print(line("Accuracy", f"{100 * m['accuracy']:.2f}   (element-wise, README's differs)"))
    print(line("F1 macro", f"{100 * m['macro']['f1']:.2f}   micro {100 * m['micro']['f1']:.2f}", "f1"))
    print(line("Precision macro", f"{100 * m['macro']['precision']:.2f}   micro {100 * m['micro']['precision']:.2f}", "precision"))
    print(line("Recall macro", f"{100 * m['macro']['recall']:.2f}   micro {100 * m['micro']['recall']:.2f}", "recall"))
    print(line("AUROC macro", f"{100 * m['auroc_macro']:.2f}   (not reported)"))
    print(f"correct samples  {int(round(m['exact_match_ratio'] * len(probs)))}/{len(probs)}"
          + ("   (authors: 503/632)" if show else ""))

    print("\nper-class recall (label order):")
    for i, name in enumerate(ds.class_names):
        r = 100 * m["per_class"]["recall"][i]
        tgt = f"(target {TARGET_RECALL[name]:5.2f}, " if show else "("
        print(f"  {name:<14} {r:6.2f}   {tgt}support {int(m['per_class']['support'][i])})")

    print(f"\nwithout relation post-processing: EMR {100 * m_raw['exact_match_ratio']:.2f}"
          f"   F1 {100 * m_raw['macro']['f1']:.2f}   P {100 * m_raw['macro']['precision']:.2f}"
          f"   R {100 * m_raw['macro']['recall']:.2f}")

    tag = args.tag or os.path.splitext(os.path.basename(args.ckpt))[0]
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{tag}_{args.split}.json"), "w") as f:
        json.dump({"model": tag, "split": args.split, "n_params": n_params,
                   "n_samples": int(len(probs)), "class_names": ds.class_names,
                   "golden_test_passed": golden_ok, "metrics": m,
                   "metrics_without_relation": m_raw}, f, indent=2)
    # same keys as the ResNet npz (probs, targets) so viz_preds.py etc. work unchanged
    np.savez_compressed(os.path.join(args.out, f"{tag}_{args.split}_raw.npz"),
                        probs=probs, probs_no_relation=probs_raw, targets=tgts)
    print(f"\nsaved -> {args.out}/{tag}_{args.split}.json (+ _raw.npz)")


if __name__ == "__main__":
    main()
