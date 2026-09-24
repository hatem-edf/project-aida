#!/usr/bin/env python
"""Stage 3 of network slimming: fine-tune a pruned model to recover accuracy.

Loads a checkpoint written by prune_slim.py (narrowed architecture), trains with
no sparsity penalty and a cosine-decayed LR, and saves a checkpoint that keeps
channel_config/absorb_bias so load_pruned() can read it back.

  python finetune_pruned.py --ckpt pruned/s40_r50.pth --epochs 40 --out runs/ft_s40_r50
"""
import argparse, json, os, time
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode


CLASSES = ["NoDamage", "Crack", "Efflorescence", "Spalling", "BarsExposed", "Rust"]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="pruned checkpoint from prune_slim.py")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--warmup", type=int, default=2, help="linear warmup epochs")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    a = get_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")

    from bikit.datasets import BikitDataset
    from prune_slim import load_pruned

    model, cp = load_pruned(a.ckpt)
    model.to(dev)
    for p in model.parameters():
        p.requires_grad_(True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded {a.ckpt}")
    if "prune_stats" in cp:
        ps = cp["prune_stats"]
        print(f"  pruned {ps['channels_before']} -> {ps['channels_after']} channels, "
              f"{ps['params_before']:,} -> {ps['params_after']:,} params "
              f"({100*(1-ps['params_after']/ps['params_before']):.1f}% smaller)")
    print(f"  trainable parameters: {n_params:,}")

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    rs = transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC)
    tf_train = transforms.Compose([rs, transforms.RandomHorizontalFlip(),
                                   transforms.RandomVerticalFlip(),
                                   transforms.ToTensor(), norm])
    tf_eval = transforms.Compose([rs, transforms.ToTensor(), norm])

    dtr = BikitDataset("codebrim-classif-balanced", split="train", transform=tf_train)
    dva = BikitDataset("codebrim-classif-balanced", split="val", transform=tf_eval)
    ltr = DataLoader(dtr, batch_size=a.batch_size, shuffle=True,
                     num_workers=a.workers, pin_memory=(dev == "cuda"), drop_last=True)
    lva = DataLoader(dva, batch_size=a.batch_size, shuffle=False,
                     num_workers=a.workers, pin_memory=(dev == "cuda"))
    print(f"train {len(dtr)}  val {len(dva)}")

    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)

    def lr_at(ep):
        """Linear warmup then cosine decay to min_lr. ep is 1-indexed."""
        if ep <= a.warmup:
            return a.lr * ep / max(1, a.warmup)
        t = (ep - a.warmup) / max(1, a.epochs - a.warmup)
        return a.min_lr + 0.5 * (a.lr - a.min_lr) * (1 + np.cos(np.pi * t))

    @torch.no_grad()
    def evaluate():
        model.eval()
        P, T = [], []
        for x, y in lva:
            P.append(torch.sigmoid(model(x.to(dev))).cpu().numpy())
            T.append(y.numpy())
        P, T = np.concatenate(P), np.concatenate(T)
        pred = (P >= 0.5)
        emr = float((pred == T).all(1).mean())
        acc = float((pred == T).mean())
        rec = [float(((pred[:, c] == 1) & (T[:, c] == 1)).sum() / max(1, (T[:, c] == 1).sum()))
               for c in range(T.shape[1])]
        return emr, acc, rec

    def save(path, ep, emr, acc):
        """Carry channel_config/absorb_bias forward or load_pruned cannot rebuild."""
        torch.save({**{k: v for k, v in cp.items() if k != "state_dict"},
                    "state_dict": model.state_dict(),
                    "finetune_epoch": ep, "val_emr": emr, "val_acc": acc,
                    "source_ckpt": a.ckpt}, path)

    emr0, acc0, _ = evaluate()
    print(f"before fine-tuning: val_emr {100*emr0:.2f}  val_acc {100*acc0:.2f}\n")

    hist, best = [], -1.0
    for ep in range(1, a.epochs + 1):
        lr = lr_at(ep)
        for gp in opt.param_groups:
            gp["lr"] = lr
        model.train(); tot = 0.0; t0 = time.time()
        for x, y in ltr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
        emr, acc, rec = evaluate()
        rec_ent = {"epoch": ep, "lr": lr, "loss": tot / len(dtr),
                   "val_emr": emr, "val_acc": acc,
                   "val_recall": {CLASSES[i]: rec[i] for i in range(len(CLASSES))},
                   "secs": round(time.time() - t0, 1)}
        hist.append(rec_ent)
        star = ""
        if emr > best:
            best = emr
            save(os.path.join(a.out, "best.pth"), ep, emr, acc)
            star = "  *"
        save(os.path.join(a.out, "last.pth"), ep, emr, acc)
        json.dump({"args": vars(a), "before": {"val_emr": emr0, "val_acc": acc0},
                   "history": hist}, open(os.path.join(a.out, "history.json"), "w"), indent=2)
        print(f"ep{ep:3d} lr {lr:.2e}  loss {rec_ent['loss']:.4f}  "
              f"val_emr {100*emr:.2f}  val_acc {100*acc:.2f}  "
              f"({rec_ent['secs']}s){star}", flush=True)

    print(f"\nval EMR {100*emr0:.2f} -> {100*best:.2f}  (+{100*(best-emr0):.2f} pts)")
    print("best ->", os.path.join(a.out, "best.pth"))


if __name__ == "__main__":
    main()
