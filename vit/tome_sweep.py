#!/usr/bin/env python
"""Token Merging sweep on the released ViT: accuracy vs FLOPs vs real GPU throughput.

For each r (tokens merged per block), evaluates val and test, counts FLOPs from the
actual token trace, and times batches of 64 images on the GPU (images only, no data
loading). r=0 is the unmodified model (checked against the plain model at start-up).

  sbatch tome_sweep.sh
  -> results/tome_sweep.json and results/tome_sweep_probs.npz
"""
import argparse, copy, json, os, time

import numpy as np
import torch
from torch.utils.data import DataLoader

from vit_model import DEFAULT_CKPT, build_transform, load_vit, reorder_index
from eval_vit import compute_metrics
from tome_vit import apply_tome, vit_flops

SHORT = {"NoDamage": "ND", "Crack": "Cr", "Efflorescence": "Ef", "Spalling": "Sp",
         "BarsExposed": "BE", "Rust": "Ru"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    p.add_argument("--r", default="0,4,8,12,16,24,32,40,48,56,64")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default="results")
    return p.parse_args()


def load_split(split, tf, a):
    from bikit.datasets import BikitDataset
    ds = BikitDataset("codebrim-classif-balanced", split=split, transform=tf)
    X, Y = [], []
    for x, y in DataLoader(ds, batch_size=a.batch_size, num_workers=a.workers):
        X.append(x); Y.append(y)
    return torch.cat(X), torch.cat(Y).numpy(), ds.class_names


@torch.inference_mode()
def predict(model, X, dev, bs):
    return np.concatenate([model(X[i:i + bs].to(dev, non_blocking=True)).float().cpu().numpy()
                           for i in range(0, len(X), bs)])


@torch.inference_mode()
def throughput(model, X, dev, bs, warmup=5, iters=20):
    x = X[:bs].to(dev)
    for _ in range(warmup):
        model(x)
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    if dev == "cuda":
        torch.cuda.synchronize()
    return iters * bs / (time.perf_counter() - t0)


def main():
    a = parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}" + (f"  gpu={torch.cuda.get_device_name(0)}" if dev == "cuda" else ""))
    model, names, cp = load_vit(a.ckpt)
    model.to(dev).eval()
    tf = build_transform(cp["img_size"], mode="eval")
    data = {s: load_split(s, tf, a) for s in ("val", "test")}
    label_names = data["test"][2]
    perm = reorder_index(names, label_names)
    print(f"val {len(data['val'][0])}  test {len(data['test'][0])}  (images kept in memory)")

    tm = apply_tome(copy.deepcopy(model), 0)
    xs = data["test"][0][:16]
    with torch.inference_mode():
        d = (tm(xs.to(dev)) - model(xs.to(dev))).abs().max().item()
    print(f"check: patched r=0 vs plain model, max |prob diff| = {d:.2e}  (must be ~0)\n")
    assert d < 1e-4

    rows, probs_out = [], {}
    for r in [int(v) for v in a.r.split(",")]:
        apply_tome(tm, r)
        res = {"r": r}
        for split in ("val", "test"):
            X, Y, _ = data[split]
            P = predict(tm, X, dev, a.batch_size)[:, perm]
            res[split] = compute_metrics(P, Y)
            if split == "test":
                probs_out[f"r{r}"] = P
        tm(data["test"][0][:1].to(dev))                       # token trace for one image
        res["tokens"] = list(tm.encoder._tome["tokens"])
        res["gflops"] = vit_flops(res["tokens"], tm.encoder._tome["r"]) / 1e9
        res["img_per_s"] = throughput(tm, data["test"][0], dev, a.batch_size)
        rows.append(res)
        t = res["test"]
        print(f"r={r:2d}  tokens {res['tokens'][0]}->{res['tokens'][-1]:<4} GFLOPs {res['gflops']:5.1f}  "
              f"{res['img_per_s']:6.0f} img/s   EMR val {100 * res['val']['exact_match_ratio']:5.2f}  "
              f"test {100 * t['exact_match_ratio']:5.2f}", flush=True)

    base = rows[0]
    print(f"\n{'r':>3} {'tokens':>11} {'GFLOPs':>7} {'saved':>6} {'speedup':>8} {'val EMR':>8} "
          f"{'test EMR':>9} {'F1':>6}   recall (test): " + " ".join(f"{SHORT[n]:>5}" for n in label_names))
    for res in rows:
        t = res["test"]
        print(f"{res['r']:>3} {res['tokens'][0]:>5}->{res['tokens'][-1]:<4} {res['gflops']:7.1f} "
              f"{100 * (1 - res['gflops'] / base['gflops']):5.1f}% {res['img_per_s'] / base['img_per_s']:7.2f}x "
              f"{100 * res['val']['exact_match_ratio']:8.2f} {100 * t['exact_match_ratio']:9.2f} "
              f"{100 * t['macro']['f1']:6.2f}                 "
              + " ".join(f"{100 * v:5.1f}" for v in t["per_class"]["recall"]))

    os.makedirs(a.out, exist_ok=True)
    json.dump({"class_names": label_names, "rows": rows}, open(os.path.join(a.out, "tome_sweep.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.out, "tome_sweep_probs.npz"), targets=data["test"][1], **probs_out)
    print(f"\nsaved -> {a.out}/tome_sweep.json (+ _probs.npz)")


if __name__ == "__main__":
    main()
