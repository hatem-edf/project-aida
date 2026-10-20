#!/usr/bin/env python
"""Full fine-tuning of the bikit ViT-S/8 on CODEBRIM (encoder + decoder; relation matrix frozen).

The model returns PROBABILITIES (sigmoid and relation post-processing are inside it), in the
checkpoint's class order (CSV order: Spalling before Efflorescence). BikitDataset labels follow
datasets.json order. So: permute the outputs to label order, then plain BCE on probabilities,
computed in fp32 outside autocast (BCELoss refuses to run under autocast).

Checkpoints are saved in the ORIGINAL bikit-models layout, so load_vit(), eval_vit.py --ckpt
and tome_sweep.py --ckpt read them unchanged.

Fail-fast guards (non-zero exit stops a Slurm dependency chain):
  exit 2  output folder already exists (never overwrite a run)
  exit 3  G1: val EMR of the loaded model != --expect-val-emr (loading/order/preprocessing wrong)
  exit 4  G0: another of the 720 class orders fits the labels better than ours (order wrong)
  exit 5  G2: val EMR below --guard-min-emr after --guard-epoch (training destroyed the features)
  exit 6  G3: non-finite loss

  python train_vit.py --cpu-smoke --out /tmp/vit_smoke          # login node, 100 images, CPU
  python train_vit.py --epochs 100 --seed 0 --expect-val-emr 76.30 --out runs/vit_ctrl_ep100_seed0
"""
import argparse, contextlib, itertools, json, math, os, random, sys, time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from vit_model import DEFAULT_CKPT, build_transform, load_vit, reorder_index
from eval_vit import compute_metrics

DATASET = "codebrim-classif-balanced"
NORM = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(DEFAULT_CKPT), help="starting checkpoint (bikit-models layout)")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr-enc", type=float, default=1e-5)
    p.add_argument("--lr-dec", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=3, help="linear warmup epochs")
    p.add_argument("--min-lr-ratio", type=float, default=0.01, help="cosine floor, fraction of base lr")
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--expect-val-emr", type=float, default=None, help="G1: required val EMR at start (%%)")
    p.add_argument("--guard-epoch", type=int, default=3)
    p.add_argument("--guard-min-emr", type=float, default=60.0)
    p.add_argument("--cpu-smoke", action="store_true", help="100 images, batch 4, 1 epoch, CPU, no guards")
    a = p.parse_args()
    if a.cpu_smoke:
        a.epochs, a.batch_size, a.workers = 1, 4, 0
    return a


def train_transform(img_size=224):
    """Same geometry as evaluation (256x256 squash, then 224), random crop + flips."""
    return T.Compose([T.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
                      T.RandomCrop(img_size), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                      T.ToTensor(), NORM])


def param_groups(module, lr, wd):
    decay, no_decay = [], []
    for n, p in module.named_parameters():
        if p.requires_grad:
            (no_decay if p.ndim == 1 or n in ("pos_embed", "cls_token") else decay).append(p)
    return [{"params": decay, "lr": lr, "base_lr": lr, "weight_decay": wd},
            {"params": no_decay, "lr": lr, "base_lr": lr, "weight_decay": 0.0}]


def lr_factor(ep, a):
    """ep is 1-indexed. Linear warmup, then cosine down to min_lr_ratio."""
    w = min(a.warmup, a.epochs)
    if ep <= w:
        return ep / max(1, w)
    t = (ep - w) / max(1, a.epochs - w)
    return a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * t))


def bce(probs, y):
    return F.binary_cross_entropy(probs.float().clamp(1e-6, 1 - 1e-6), y)


@torch.no_grad()
def evaluate(model, loader, perm, dev):
    """fp32, no autocast: comparable to the 76.30 / 79.27 references."""
    model.eval()
    P, Y = [], []
    for x, y in loader:
        P.append(model(x.to(dev, non_blocking=True)).float()[:, perm].cpu())
        Y.append(y)
    P, Y = torch.cat(P), torch.cat(Y)
    m = compute_metrics(P.numpy(), Y.numpy())
    return m["exact_match_ratio"], bce(P, Y).item(), m


def save_ckpt(model, cp, path, **extra):
    torch.save({**{k: v for k, v in cp.items() if k not in ("encoder", "decoder", "relation", "optimizer")},
                "encoder": {k: v.cpu() for k, v in model.encoder.state_dict().items()},
                "decoder": {k: v.cpu() for k, v in model.decoder.state_dict().items()},
                "relation": torch.nn.Parameter(model.relation.detach().cpu(), requires_grad=False),
                **extra}, path)


def stop(code, msg, a=None, hist=None):
    print(f"\nSTOP (exit {code}): {msg}", flush=True)
    if a is not None and hist is not None:
        json.dump({"args": vars(a), "history": hist, "stopped": msg},
                  open(os.path.join(a.out, "history.json"), "w"), indent=2)
    sys.exit(code)


def main():
    a = parse_args()
    if os.path.exists(a.out):
        stop(2, f"{a.out} already exists; choose another TAG or move it away")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() and not a.cpu_smoke else "cpu"
    amp = dev == "cuda" and torch.cuda.is_bf16_supported()
    print(f"device={dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else "")
          + f"  bf16 autocast={amp}  seed={a.seed}")

    from bikit.datasets import BikitDataset
    model, names, cp = load_vit(a.ckpt)
    model.to(dev)
    model.relation.requires_grad_(False)
    print(f"start: {a.ckpt}")

    dtr = BikitDataset(DATASET, split="train", transform=train_transform(cp["img_size"]), devel_mode=a.cpu_smoke)
    dva = BikitDataset(DATASET, split="val", transform=build_transform(cp["img_size"], mode="eval"),
                       devel_mode=a.cpu_smoke)
    ltr = DataLoader(dtr, batch_size=a.batch_size, shuffle=True, drop_last=True,
                     num_workers=a.workers, pin_memory=(dev == "cuda"), persistent_workers=a.workers > 0)
    lva = DataLoader(dva, batch_size=64, shuffle=False, num_workers=a.workers, pin_memory=(dev == "cuda"))
    perm = reorder_index(names, dtr.class_names)
    print(f"train {len(dtr)}  val {len(dva)}  model order {names}  ->  label order {dtr.class_names}")

    # G0: among all 720 possible class orders, ours must fit the labels best (on one val batch)
    if not a.cpu_smoke:
        model.eval()
        x, y = next(iter(lva))
        with torch.no_grad():
            p = model(x.to(dev)).float().cpu()
        losses = {o: bce(p[:, list(o)], y).item() for o in itertools.permutations(range(len(perm)))}
        best_o = min(losses, key=losses.get)
        print(f"G0 label order: BCE {losses[tuple(perm)]:.4f} with our order, best of {len(losses)} orders "
              f"{losses[best_o]:.4f}, no reordering {losses[tuple(range(len(perm)))]:.4f}")
        if list(best_o) != perm:
            stop(4, f"G0: order {list(best_o)} fits better than ours {perm}; label order is wrong")

    emr0, vloss0, _ = evaluate(model, lva, perm, dev)
    print(f"before fine-tuning: val EMR {100 * emr0:.2f}  val BCE {vloss0:.4f}")
    if a.expect_val_emr is not None and not a.cpu_smoke and abs(100 * emr0 - a.expect_val_emr) > 0.2:
        stop(3, f"G1: val EMR {100 * emr0:.2f} != expected {a.expect_val_emr:.2f}")

    os.makedirs(a.out)
    groups = param_groups(model.encoder, a.lr_enc, a.wd) + param_groups(model.decoder, a.lr_dec, a.wd)
    opt = torch.optim.AdamW(groups)
    trainable = [p for g in groups for p in g["params"]]
    print(f"trainable parameters {sum(p.numel() for p in trainable):,}  "
          f"(lr encoder {a.lr_enc:g}, decoder {a.lr_dec:g}, wd {a.wd:g}, warmup {a.warmup}, cosine)\n")

    hist, best = [], -1.0
    for ep in range(1, a.epochs + 1):
        f = lr_factor(ep, a)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * f
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model.train()
        tot, n, t0 = 0.0, 0, time.time()
        for x, y in ltr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with (torch.autocast("cuda", dtype=torch.bfloat16) if amp else contextlib.nullcontext()):
                p = model(x)                                   # probabilities, model order
            loss = bce(p[:, perm], y)
            if not torch.isfinite(loss):
                stop(6, f"G3: non-finite loss at epoch {ep}", a, hist)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, a.clip)
            opt.step()
            tot += loss.item() * x.size(0); n += x.size(0)
        secs = time.time() - t0
        emr, vloss, m = evaluate(model, lva, perm, dev)
        mem = torch.cuda.max_memory_allocated() / 2 ** 30 if dev == "cuda" else 0.0
        rec = {"epoch": ep, "lr_enc": a.lr_enc * f, "loss": tot / max(1, n), "val_emr": emr,
               "val_loss": vloss, "val_recall": dict(zip(dtr.class_names, m["per_class"]["recall"])),
               "secs": round(secs, 1), "peak_mem_gb": round(mem, 2)}
        hist.append(rec)
        star = ""
        if emr > best:
            best = emr
            save_ckpt(model, cp, os.path.join(a.out, "best.pth"), ft_epoch=ep, val_emr=emr,
                      source_ckpt=a.ckpt, ft_args=vars(a))
            star = "  *"
        save_ckpt(model, cp, os.path.join(a.out, "last.pth"), ft_epoch=ep, val_emr=emr,
                  source_ckpt=a.ckpt, ft_args=vars(a))
        json.dump({"args": vars(a), "before": {"val_emr": emr0, "val_loss": vloss0}, "history": hist},
                  open(os.path.join(a.out, "history.json"), "w"), indent=2)
        print(f"ep{ep:4d}  lr {a.lr_enc * f:.2e}  loss {rec['loss']:.4f}  val_loss {vloss:.4f}  "
              f"val_emr {100 * emr:.2f}  ({secs:.0f}s, {mem:.1f} GB){star}", flush=True)
        if not a.cpu_smoke and ep == a.guard_epoch and 100 * emr < a.guard_min_emr:
            stop(5, f"G2: val EMR {100 * emr:.2f} < {a.guard_min_emr} after epoch {ep}", a, hist)

    print(f"\nval EMR {100 * emr0:.2f} -> {100 * best:.2f}  (best epoch "
          f"{max(hist, key=lambda r: r['val_emr'])['epoch']})")
    print("best ->", os.path.join(a.out, "best.pth"))


if __name__ == "__main__":
    main()
