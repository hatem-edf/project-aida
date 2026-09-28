#!/usr/bin/env python
"""Can the evaluation preprocessing explain the reproduction gaps?

Data (md5, CRC, labels) and weights are verified identical, so for a fixed
checkpoint the only freedom left in an evaluation is how images become 224x224
tensors. This evaluates both published checkpoints under the standard options.

  published : ViT test 79.59 | ResNet test 71.36 | ResNet val 79.06 (stored in its checkpoint)
  ours      : ViT 72.31 with short256_crop224 | ResNet 64.24 / 61.36 with squash224
              -> those two cells must come out identical (built-in consistency check)

  sbatch sweep.sh
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "vit"))
from vit_model import load_vit, reorder_index   # noqa: E402
from eval_vit import compute_metrics            # noqa: E402

BIC = InterpolationMode.BICUBIC
VARIANTS = {   # everything after this: ToTensor + ImageNet Normalize
    "squash224":         [T.Resize((224, 224), interpolation=BIC)],              # bikit preprocess_img (our ResNet eval)
    "short256_crop224":  [T.Resize(256, interpolation=BIC), T.CenterCrop(224)],  # ViT demo + bikit-models process_img_daclnet
    "short224_crop224":  [T.Resize(224, interpolation=BIC), T.CenterCrop(224)],
    "squash256_crop224": [T.Resize((256, 256), interpolation=BIC), T.CenterCrop(224)], "squash224_bl":         [T.Resize((224, 224))],              # torchvision default interpolation = bilinear
    "short256_crop224_bl":  [T.Resize(256), T.CenterCrop(224)],  # the classic torchvision eval transform
    "short224_crop224_bl":  [T.Resize(224), T.CenterCrop(224)],
    "squash256_crop224_bl": [T.Resize((256, 256)), T.CenterCrop(224)],
}
RUNS = [("ViT_s8_1", "test", 79.59), ("ResNet50", "test", 71.36), ("ResNet50", "val", 79.06)]


def load_resnet():
    from bikit.models import DaclNet
    cp = torch.load(Path.home() / ".cache/bikit-models/codebrim-classif-balanced_ResNet50_hta.pth",
                    map_location="cpu", weights_only=False)
    m = DaclNet(base_name=cp["base"], resolution=cp["resolution"], hidden_layers=cp["hidden_layers"],
                drop_prob=cp["drop_prob"], num_class=cp["num_class"])
    m.load_state_dict(cp["state_dict"])
    names = [cp["cat_to_name"][i] for i in range(cp["num_class"])]
    return m.eval(), names, torch.sigmoid          # DaclNet returns logits


def load_vit_probs():
    m, names, _ = load_vit()
    return m.eval(), names, lambda p: p            # the ViT already returns probabilities


@torch.no_grad()
def emr(model, names, act, steps, split, device):
    from bikit.datasets import BikitDataset
    tf = T.Compose(steps + [T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    ds = BikitDataset("codebrim-classif-balanced", split=split, transform=tf)
    dl = DataLoader(ds, batch_size=64, num_workers=8, pin_memory=(device == "cuda"))
    P, Y = [], []
    for x, y in dl:
        P.append(act(model(x.to(device, non_blocking=True))).cpu().numpy())
        Y.append(y.numpy())
    probs = np.concatenate(P)[:, reorder_index(names, ds.class_names)]
    m = compute_metrics(probs, np.concatenate(Y))
    return 100 * m["exact_match_ratio"], int(round(m["exact_match_ratio"] * len(probs))), len(probs)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}\n")
    models = {"ViT_s8_1": load_vit_probs, "ResNet50": load_resnet}
    table = {}
    for name in models:
        model, names, act = models[name]()
        model.to(device)
        for _, split, _ in [r for r in RUNS if r[0] == name]:
            for v, steps in VARIANTS.items():
                table[(name, split, v)] = emr(model, names, act, steps, split, device)
                e, k, n = table[(name, split, v)]
                print(f"  {name:<9} {split:<5} {v:<18} EMR {e:6.2f}  ({k}/{n})", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    cols = [f"{n} {s}" for n, s, _ in RUNS]
    print("\nEMR (%)             " + "".join(f"{c:>14}" for c in cols))
    print("published           " + "".join(f"{p:>14.2f}" for _, _, p in RUNS))
    for v in VARIANTS:
        print(f"{v:<20}" + "".join(f"{table[(n, s, v)][0]:>14.2f}" for n, s, _ in RUNS))


if __name__ == "__main__":
    main()
