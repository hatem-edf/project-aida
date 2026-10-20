#!/usr/bin/env python
"""ViT compression table: accuracy vs parameters, FLOPs and MEASURED GPU speed, with the ResNet-50
as the reference to beat (success criterion of the ViT phase).

Test EMR comes from results/<name>_test.json (eval_vit.py), mean +/- std over seeds. Parameters
and FLOPs are counted on the checkpoint; throughput is timed on this GPU in fp32: images/s at
batch 64 and latency at batch 1 (same timing method as tome_sweep.py).

  python vit_table.py              # on a GPU node (vit_table.sh); --preview allows missing runs
  -> results/vit_table.json and results/vit_table.md
"""
import argparse, json, os, time

import numpy as np
import torch
from torch import nn

from vit_model import DEFAULT_CKPT, encoder_dims, load_vit
from tome_vit import vit_flops

seeds = lambda stem: [f"{stem}_seed{s}" for s in range(3)]
RATIOS = ["015", "03", "05", "065", "08"]
ROWS = ([("ViT-S/8 released", str(DEFAULT_CKPT), ["codebrim-classif-balanced_ViT_s8_1"]),
         ("ViT-S/8 fine-tuned (control)", "runs/vit_ctrl_ep100_seed0/best.pth", seeds("vit_ctrl_ep100"))]
        + [(f"ViT-S/8 {int(r) / 10 ** (len(r) - 1) * 100:.0f}% pruned", f"pruned/vit_r{r}.pth",
            seeds(f"vit_prune_r{r}_ep100")) for r in RATIOS])
RESNET = {"label": "ResNet-50 fine-tuned 160 ep (reference)", "emr": [79.43, 78.16, 80.38],
          "params": 23_771_078, "gflops": 8.175}


def resnet_like():
    """Same compute as the bikit DaclNet ResNet-50: backbone + 2048->128->6 head."""
    import torchvision
    m = torchvision.models.resnet50()
    m.fc = nn.Sequential(nn.Linear(2048, 128), nn.ELU(), nn.Linear(128, 6))
    return m


@torch.inference_mode()
def speed(model, dev, bs=64, iters=30, lat_iters=50):
    model.eval().to(dev)
    sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
    x = torch.randn(bs, 3, 224, 224, device=dev)
    for _ in range(10):
        model(x)
    sync(); t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    sync(); ips = iters * bs / (time.perf_counter() - t0)
    x1 = x[:1]
    for _ in range(10):
        model(x1)
    sync(); t0 = time.perf_counter()
    for _ in range(lat_iters):
        model(x1)
    sync(); lat = 1000 * (time.perf_counter() - t0) / lat_iters
    return ips, lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if not a.preview:                                     # check everything before timing anything
        missing = [f"{a.results}/{n}_test.json" for _, _, names in ROWS for n in names
                   if not os.path.exists(os.path.join(a.results, f"{n}_test.json"))]
        missing += [ck for _, ck, _ in ROWS if not os.path.exists(ck)]
        if missing:
            raise SystemExit(f"missing {len(missing)} files, e.g. {missing[:3]} (use --preview)")
    out = []
    for label, ckpt, names in ROWS:
        files = [os.path.join(a.results, f"{n}_test.json") for n in names]
        have = [f for f in files if os.path.exists(f)]
        if len(have) < len(files) and not a.preview:
            raise SystemExit(f"missing {[f for f in files if f not in have]} (use --preview)")
        if not have or not os.path.exists(ckpt):
            print(f"{label:<32} pending"); continue
        ms = [json.load(open(f))["metrics"] for f in have]
        emr = np.array([100 * m["exact_match_ratio"] for m in ms])
        f1 = np.array([100 * m["macro"]["f1"] for m in ms])
        model = load_vit(ckpt)[0]
        dims = encoder_dims(model.encoder)
        ips, lat = speed(model, dev)
        out.append({"label": label, "ckpt": ckpt, "runs": len(have), "emr": emr.tolist(), "f1": f1.tolist(),
                    "params": sum(p.numel() for p in model.parameters()),
                    "gflops": vit_flops([785] * 12, [0] * 12, dims) / 1e9,
                    "heads": sum(h for h, _ in dims), "neurons": sum(m for _, m in dims),
                    "img_per_s": ips, "latency_ms": lat})
        del model
    ips, lat = speed(resnet_like(), dev)
    out.append({"label": RESNET["label"], "runs": 3, "emr": RESNET["emr"], "f1": None,
                "params": RESNET["params"], "gflops": RESNET["gflops"], "heads": None, "neurons": None,
                "img_per_s": ips, "latency_ms": lat})

    base = out[0]
    hdr = (f"| {'model':<38} | test EMR | test error | params | GFLOPs | img/s (bs 64) | "
           f"latency bs 1 | speed vs released |")
    lines = [hdr, "|" + "|".join("-" * (len(c) + 0) for c in hdr.split("|")[1:-1]) + "|"]
    for r in out:
        e = np.array(r["emr"])
        emr_txt = f"{e.mean():.2f} ± {e.std(ddof=1):.2f}" if len(e) > 1 else f"{e.mean():.2f}"
        lines.append(f"| {r['label']:<38} | {emr_txt:>8} | {100 - e.mean():10.2f} | {r['params'] / 1e6:5.2f}M | "
                     f"{r['gflops']:6.2f} | {r['img_per_s']:13.0f} | {r['latency_ms']:9.2f} ms | "
                     f"{r['img_per_s'] / base['img_per_s']:16.2f}x |")
    table = "\n".join(lines)
    print(f"\ndevice: {torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'}  (fp32)\n")
    print(table)
    os.makedirs(a.results, exist_ok=True)
    json.dump(out, open(os.path.join(a.results, "vit_table.json"), "w"), indent=1)
    open(os.path.join(a.results, "vit_table.md"), "w").write(table + "\n")
    print(f"\nsaved -> {a.results}/vit_table.json, vit_table.md")


if __name__ == "__main__":
    main()
