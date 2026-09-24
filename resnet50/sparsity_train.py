#!/usr/bin/env python
"""Stage 1 of network slimming: fine-tune with L1 on BN gammas."""
import argparse, json, os, time
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

torch.manual_seed(0); np.random.seed(0)

CLASSES = ["NoDamage", "Crack", "Efflorescence", "Spalling", "BarsExposed", "Rust"]

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.expanduser(
        "~/.cache/bikit-models/codebrim-classif-balanced_ResNet50_hta.pth"))
    p.add_argument("--lam", type=float, default=1e-4, help="L1 strength on BN gammas")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--all-bn", action="store_true",
                   help="L1 on every BN (default: only prunable bn1/bn2)")
    p.add_argument("--out", default="runs/sparsity")
    return p.parse_args()

def prunable_bns(model):
    """bn1/bn2 inside bottlenecks only: bn3 and downsample feed the residual add."""
    out = []
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) and name.startswith("base."):
            leaf = name.split(".")[-1]
            if leaf in ("bn1", "bn2") and len(name.split(".")) > 3:
                out.append((name, m))
    return out

def main():
    a = get_args()
    os.makedirs(a.out, exist_ok=True)
    existing = [f for f in ("best.pth", "last.pth", "s20.pth", "s40.pth", "s60.pth")
                if os.path.exists(os.path.join(a.out, f))]
    if existing:
        print(f"WARNING: {a.out} already contains {existing} -- milestones will be skipped")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")

    from bikit.models import DaclNet
    from bikit.datasets import BikitDataset

    cp = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = DaclNet(base_name=cp["base"], resolution=cp["resolution"],
                    hidden_layers=cp["hidden_layers"], drop_prob=cp["drop_prob"],
                    num_class=cp["num_class"], freeze_base=False)
    model.load_state_dict(cp["state_dict"])
    for p in model.parameters():
        p.requires_grad_(True)
    model.to(dev)

    bns = prunable_bns(model) if not a.all_bn else \
          [(n, m) for n, m in model.named_modules()
           if isinstance(m, nn.BatchNorm2d) and n.startswith("base.")]
    n_ch = sum(m.weight.numel() for _, m in bns)
    print(f"L1 on {len(bns)} BN layers, {n_ch} channels, lambda={a.lam}")

    norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    rs = transforms.Resize((224,224), interpolation=InterpolationMode.BICUBIC)
    tf_train = transforms.Compose([rs, transforms.RandomHorizontalFlip(),
                                   transforms.RandomVerticalFlip(),
                                   transforms.ToTensor(), norm])
    tf_eval = transforms.Compose([rs, transforms.ToTensor(), norm])

    dtr = BikitDataset("codebrim-classif-balanced", split="train", transform=tf_train)
    dva = BikitDataset("codebrim-classif-balanced", split="val", transform=tf_eval)
    ltr = DataLoader(dtr, batch_size=a.batch_size, shuffle=True,
                     num_workers=a.workers, pin_memory=True, drop_last=True)
    lva = DataLoader(dva, batch_size=a.batch_size, shuffle=False,
                     num_workers=a.workers, pin_memory=True)
    print(f"train {len(dtr)}  val {len(dva)}")

    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    def gamma_stats():
        g = torch.cat([m.weight.detach().abs().flatten() for _, m in bns])
        return {"mean": g.mean().item(),
                "lt_0.01": (g < 0.01).float().mean().item(),
                "lt_0.05": (g < 0.05).float().mean().item(),
                "p50": g.median().item()}

    @torch.no_grad()
    def evaluate():
        model.eval()
        P, T = [], []
        for x, y in lva:
            P.append(torch.sigmoid(model(x.to(dev))).cpu().numpy()); T.append(y.numpy())
        P, T = np.concatenate(P), np.concatenate(T)
        pred = (P >= 0.5)
        return float((pred == T).all(1).mean()), float((pred == T).mean())

    hist, best = [], -1
    for ep in range(1, a.epochs + 1):
        model.train(); tot = 0.0; t0 = time.time()
        for x, y in ltr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            # proximal L1: soft-threshold gammas toward zero
            if a.lam > 0:
                with torch.no_grad():
                    for _, m in bns:
                        w = m.weight.data
                        w.copy_(torch.sign(w) * torch.clamp(w.abs() - a.lam, min=0.0))
            tot += loss.item() * x.size(0)
        emr, acc = evaluate()
        gs = gamma_stats()
        rec = {"epoch": ep, "loss": tot/len(dtr), "val_emr": emr, "val_acc": acc,
               "secs": round(time.time()-t0, 1), **gs}
        hist.append(rec)
        print(f"ep{ep:3d} loss {rec['loss']:.4f}  val_emr {100*emr:.2f}  "
              f"gamma<0.01 {100*gs['lt_0.01']:.1f}%  <0.05 {100*gs['lt_0.05']:.1f}%  "
              f"({rec['secs']}s)", flush=True)
        if emr > best:
            best = emr
            torch.save({**{k: v for k, v in cp.items() if k != "state_dict"},
                        "state_dict": model.state_dict(), "lam": a.lam,
                        "sparsity_epoch": ep, "val_emr": emr},
                       os.path.join(a.out, "best.pth"))
        json.dump(hist, open(os.path.join(a.out, "history.json"), "w"), indent=2)
        torch.save({**{k: v for k, v in cp.items() if k != "state_dict"},
                    "state_dict": model.state_dict(), "lam": a.lam,
                    "sparsity_epoch": ep, "val_emr": emr,
                    "frac_lt_005": gs["lt_0.05"]},
                   os.path.join(a.out, "last.pth"))
        for tag, thr in [("s20", 0.20), ("s40", 0.40), ("s60", 0.60)]:
            pth = os.path.join(a.out, f"{tag}.pth")
            if gs["lt_0.05"] >= thr and not os.path.exists(pth):
                torch.save({**{k: v for k, v in cp.items() if k != "state_dict"},
                            "state_dict": model.state_dict(), "lam": a.lam,
                            "sparsity_epoch": ep, "val_emr": emr,
                            "frac_lt_005": gs["lt_0.05"]}, pth)
                print(f"  saved {tag} at ep{ep} ({100*gs['lt_0.05']:.1f}%, EMR {100*emr:.2f})")
    print("best val EMR", best)
    print("\ncheckpoints written:")
    for f in sorted(os.listdir(a.out)):
        if f.endswith(".pth"):
            c = torch.load(os.path.join(a.out, f), map_location="cpu", weights_only=False)
            print(f"  {f:<10} ep{c.get('sparsity_epoch','?'):<3} "
                  f"val_emr {100*c.get('val_emr',0):.2f}  "
                  f"sparsity {100*c.get('frac_lt_005',0):.1f}%")

if __name__ == "__main__":
    main()
