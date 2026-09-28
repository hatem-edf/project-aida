#!/usr/bin/env python
"""Is our CODEBRIM copy the data the model authors used?  CPU only, about a minute.

1. Image bytes : CRC32 of every extracted val/test image vs the CRC32 stored in the
                 zip's own index (computed when the zip was made, so it is the reference).
2. Labels      : bikit's CSV vs the dataset's official annotations
                 (classification_dataset_balanced/metadata/{background,defects}.xml),
                 matched by folder + crop name like the dataset authors' loader,
                 and read by tag name (their own code disagrees on the positions).
3. Re-scoring  : the saved predictions (*_raw.npz) of both models, scored against
                 bikit's labels and against the official ones. No GPU needed.

  cd ~/work/project-aida && python check_data.py
"""
import argparse, sys, zipfile, zlib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "vit"))
from eval_vit import compute_metrics  # same metric definitions as both evals  # noqa: E402

LABEL_ORDER = ["NoDamage", "Crack", "Efflorescence", "Spalling", "BarsExposed", "Rust"]  # npz order
SHORT = dict(zip(LABEL_ORDER, ["ND", "Cr", "Ef", "Sp", "BE", "Ru"]))
XML_TAGS = {"background": "NoDamage", "crack": "Crack", "spallation": "Spalling",
            "efflorescence": "Efflorescence", "exposedbars": "BarsExposed",
            "corrosionstain": "Rust"}
# (model, saved predictions, split, published EMR for that split)
PREDICTIONS = [
    ("ViT_s8_1", "vit/results/codebrim-classif-balanced_ViT_s8_1_test_raw.npz", "test", 79.59),
    ("ResNet50", "resnet50/results/codebrim-classif-balanced_ResNet50_hta_test_raw.npz", "test", 71.36),
    ("ResNet50", "resnet50/results/codebrim-classif-balanced_ResNet50_hta_val_raw.npz", "val", 79.06),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(PROJECT / "building-inspection-toolkit/bikit/data/codebrim-classif-balanced.csv"))
    p.add_argument("--root", default=str(Path.home() / ".cache/bikit"))
    p.add_argument("--zip", default=str(PROJECT / "data/CODEBRIM_classification_balanced_dataset.zip"))
    return p.parse_args()


def check_bytes(df, root, zip_path):
    print(f"[1] image bytes vs the CRC32 recorded in {zip_path.name}")
    if not zip_path.exists():
        print("    zip not found, skipped\n")
        return
    with zipfile.ZipFile(zip_path) as z:          # reading the index works; only data reads were broken
        crc = {i.filename: i.CRC for i in z.infolist()}
    bad, missing, not_in_zip = [], [], []
    for p in df.img_path:
        arc = p.split("/", 1)[1]                  # drop the leading "codebrim-classif-balanced/"
        f = root / p
        if arc not in crc:
            not_in_zip.append(p)
        elif not f.exists():
            missing.append(p)
        elif zlib.crc32(f.read_bytes()) != crc[arc]:
            bad.append(p)
    ok = not (bad or missing or not_in_zip)
    print(f"    {len(df)} val/test images: {len(bad)} differ, {len(missing)} missing on disk, "
          f"{len(not_in_zip)} not in zip" + ("  -> all byte-identical to the zip" if ok else ""))
    for p in (bad + missing + not_in_zip)[:5]:
        print("     ", p)
    print()


def load_official(meta_dir):
    labels, tag_orders, seen = {}, set(), {}
    for folder in ("background", "defects"):
        for elem in ET.parse(meta_dir / f"{folder}.xml").getroot():
            name = list(elem.attrib.values())[0]
            tags = [c.tag for c in elem]
            tag_orders.add(tuple(tags))
            vec = {}
            for c in elem:
                key = c.tag.lower().replace("_", "")
                if key not in XML_TAGS:
                    sys.exit(f"unknown tag {c.tag!r} in {folder}.xml, element tags: {tags}")
                vec[XML_TAGS[key]] = int(c.text.strip() == "1")
            labels[(folder, name)] = [vec[c] for c in LABEL_ORDER]
            seen.setdefault(name, set()).add(folder)
    both = sum(len(v) > 1 for v in seen.values())
    return labels, tag_orders, both


def check_labels(df, meta_dir):
    print(f"[2] bikit CSV labels vs official annotations in {meta_dir}")
    labels, tag_orders, both = load_official(meta_dir)
    print(f"    {len(labels)} annotated crops; XML tag order(s): {sorted(tag_orders)}")
    print(f"    crop names present in BOTH background.xml and defects.xml: {both}")
    official = {}
    for split, sub in df.groupby("split_type", sort=False):
        keys = list(zip(sub["type"], sub["img_name"]))
        absent = [k for k in keys if k not in labels]
        off = np.array([labels.get(k, [-1] * 6) for k in keys])
        ours = sub[LABEL_ORDER].to_numpy()
        diff = (off != ours).any(axis=1)
        official[split] = off
        print(f"    {split:<5} {len(sub)} rows: {int(diff.sum())} with different labels, "
              f"{len(absent)} without an official entry")
        print("          positives  bikit:    " + "  ".join(f"{SHORT[c]} {int(ours[:, i].sum()):3d}" for i, c in enumerate(LABEL_ORDER)))
        print("          positives  official: " + "  ".join(f"{SHORT[c]} {int((off[:, i] == 1).sum()):3d}" for i, c in enumerate(LABEL_ORDER)))
        fmt = lambda v: ",".join(SHORT[c] for c, x in zip(LABEL_ORDER, v) if x == 1) or "-"
        for (t, n), o, b in [x for x in zip(keys, off, ours) if (x[1] != x[2]).any()][:5]:
            print(f"          e.g. {t}/{n}: bikit {fmt(b)}  official {fmt(o)}")
    print()
    return official


def rescore(df, official):
    print("[3] saved predictions scored against both label sets (EMR / F1 macro, %)")
    for model, rel, split, published in PREDICTIONS:
        f = PROJECT / rel
        if not f.exists():
            print(f"    {rel}: not found, skipped")
            continue
        d = np.load(f)
        bikit = df[df.split_type == split][LABEL_ORDER].to_numpy()
        if not np.array_equal(d["targets"], bikit):
            print(f"    {rel}: rows don't line up with the CSV, skipped")
            continue
        variants = [("", d["probs"])]
        if "probs_no_relation" in d.files:
            variants.append((" no relation", d["probs_no_relation"]))
        for tag, probs in variants:
            a = compute_metrics(probs, bikit)
            b = compute_metrics(probs, official[split])
            print(f"    {model + tag:<21} {split:<4}  bikit labels {100 * a['exact_match_ratio']:6.2f} / {100 * a['macro']['f1']:5.2f}"
                  f"   official {100 * b['exact_match_ratio']:6.2f} / {100 * b['macro']['f1']:5.2f}"
                  f"   published EMR {published}")


def main():
    a = parse_args()
    root = Path(a.root)
    df = pd.read_csv(a.csv)
    df = df[df.split_type.isin(["val", "test"])].reset_index(drop=True)
    check_bytes(df, root, Path(a.zip))
    meta = root / "codebrim-classif-balanced/classification_dataset_balanced/metadata"
    official = check_labels(df, meta)
    rescore(df, official)


if __name__ == "__main__":
    main()
