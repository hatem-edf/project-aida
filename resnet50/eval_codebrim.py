#!/usr/bin/env python
"""Reproduce bikit's reported test metrics for a CODEBRIM checkpoint.

Target (codebrim-classif-balanced_ResNet50_hta):
  EMR 71.36 | F1 84.13 | P 85.33 | R 83.20 | Acc 92.59 | AUROC 96.99
"""
import argparse, json, os, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# CSV column order (what BikitDataset returns).
CLASSES = ["NoDamage", "Crack", "Efflorescence", "Spalling", "BarsExposed", "Rust"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="codebrim-classif-balanced_ResNet50_hta")
    p.add_argument("--dataset", default="codebrim-classif-balanced")
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out", default="results")
    p.add_argument("--ckpt", default=None,
                   help="path to a checkpoint (pruned or not); overrides --model")
    p.add_argument("--tag", default=None, help="name for output files")
    return p.parse_args()


def compute_metrics(probs, targets, threshold=0.5):
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


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    if device == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    from bikit.datasets import BikitDataset
    from prune_slim import load_pruned

    cp_path = args.ckpt or os.path.expanduser(
        f"~/.cache/bikit-models/{args.model}.pth")
    model, cp = load_pruned(cp_path)
    model.to(device).eval()
    tag = args.tag or os.path.splitext(os.path.basename(cp_path))[0]
    print(f"checkpoint: {cp_path}")
    if "prune_stats" in cp:
        ps = cp["prune_stats"]
        print(f"pruned: {ps['channels_before']} -> {ps['channels_after']} channels, "
              f"{ps['params_before']:,} -> {ps['params_after']:,} params")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}")

    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = BikitDataset(args.dataset, split=args.split, transform=tf, return_type="pt")
    print(f"{args.split} samples: {len(ds)}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=(device == "cuda"))

    all_probs, all_tgts = [], []
    t0 = time.time()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            all_probs.append(torch.sigmoid(model(imgs)).cpu().numpy())
            all_tgts.append(labels.numpy())
    dt = time.time() - t0
    probs, tgts = np.concatenate(all_probs), np.concatenate(all_tgts)
    print(f"inference: {dt:.1f}s for {len(probs)} images "
          f"({1000*dt/len(probs):.2f} ms/img)")

    m = compute_metrics(probs, tgts, args.threshold)
    show_targets = args.ckpt is None
    print("\n=== results (%) ===")
    print(f"ExactMatchRatio  {100*m['exact_match_ratio']:.2f}" +
          ("   (target 71.36)" if show_targets else ""))
    print(f"Accuracy         {100*m['accuracy']:.2f}   (target 92.59)")
    print(f"F1 macro         {100*m['macro']['f1']:.2f}   micro {100*m['micro']['f1']:.2f}   (target 84.13)")
    print(f"Precision macro  {100*m['macro']['precision']:.2f}   micro {100*m['micro']['precision']:.2f}   (target 85.33)")
    print(f"Recall macro     {100*m['macro']['recall']:.2f}   micro {100*m['micro']['recall']:.2f}   (target 83.20)")
    if m["auroc_macro"] is not None:
        print(f"AUROC macro      {100*m['auroc_macro']:.2f}   (target 96.99)")

    targets_tbl = {"NoDamage": 93.33, "Crack": 85.33, "Spalling": 84.67,
                   "Efflorescence": 77.18, "BarsExposed": 85.33, "Rust": 73.33}
    print("\nper-class recall (json order):")
    for i, r in enumerate(m["per_class"]["recall"]):
        name = CLASSES[i]
        extra = f"(target {targets_tbl[name]:5.2f}, " if show_targets else "("
        print(f"  {name:<14} {100*r:6.2f}   {extra}"
              f"support {int(m['per_class']['support'][i])})")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{tag}_{args.split}.json"), "w") as f:
        json.dump({"model": tag, "split": args.split, "n_params": n_params,
                   "n_samples": int(len(probs)), "metrics": m}, f, indent=2)
    np.savez_compressed(os.path.join(args.out, f"{tag}_{args.split}_raw.npz"),
                        probs=probs, targets=tgts)
    print(f"\nsaved -> {args.out}/")


if __name__ == "__main__":
    main()
