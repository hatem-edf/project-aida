"""Build and load bikit-models' codebrim-classif-balanced_ViT_s8_1.

Where the model comes from
    Trained by the SOFIA/Cerema team for the dacl.ai CODEBRIM challenge
    (github.com/OALcementys/codebrim_challenge). bikit-models redistributes
    their weights unchanged and only adds metadata (dataset, cat_to_name...).

Architecture (upstream Ensemble in bikit-models/models/vistranet_1.py)
    image 224x224 -> 28x28 = 784 patches of 8x8 pixels + 1 CLS token
    -> ViT-Small encoder: 12 blocks, 6 heads, width 384, no qkv bias
    -> CLS token of each of the LAST 4 blocks, LayerNorm'd, concatenated (4*384 = 1536)
    -> Linear(1536, 6) -> Sigmoid
    -> "relation" post-processing with a learned 6x6 matrix R (values clamped to [0.1, 1]):
       pick the anchor class i of the most confident pair (max_j p_i * p_j * R[i, j]),
       then multiply all probabilities by R[i, :]. It can only lower probabilities,
       e.g. damp "Crack" when "NoDamage" is the anchor.

Two traps, both handled here or in eval_vit.py
    1. The model returns PROBABILITIES (sigmoid is inside). Never apply sigmoid again.
    2. Its output order is the checkpoint's cat_to_name = CSV header order
       (NoDamage, Crack, Spalling, Efflorescence, ...), whereas BikitDataset returns
       labels in datasets.json order (NoDamage, Crack, Efflorescence, Spalling, ...).
       Use reorder_index() before comparing with labels. (The ResNet had it the other way round.)
"""
import sys
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "bikit-models" / "models"))
from vistranet_1 import VisionTransformer, LinearClassifier, Ensemble  # noqa: E402

DEFAULT_CKPT = Path.home() / ".cache/bikit-models/codebrim-classif-balanced_ViT_s8_1.pth"
NUM_HEADS = {"vit_tiny": 3, "vit_small": 6}  # not recoverable from the weights themselves


def build_transform(img_size=224, mode="eval"):
    """mode="eval": resize to 256x256 (squash, aspect ratio NOT kept), center-crop 224.
        Reproduces the published result: 501/632 vs the authors' 503/632 (preproc_sweep.py).
    mode="demo": short side to 256 (aspect kept), center-crop 224, as in the authors'
        run.ipynb. Only for the golden test, whose reference outputs were made this way.
        It keeps just the middle half of wide crops and scores 457/632."""
    resize = ((256, 256) if mode == "eval" else 256)
    return transforms.Compose([
        transforms.Resize(resize, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

def load_vit(cp_path=DEFAULT_CKPT):
    """Returns (model, class_names_in_output_order, checkpoint_dict).

    Same model as upstream build_vistra_1(), but loaded strictly (a missing or
    unexpected key raises instead of being silently ignored) and with
    weights_only=False, which torch >= 2.6 needs because the checkpoint pickles
    an argparse.Namespace (upstream's loader crashes on it)."""
    cp = torch.load(cp_path, map_location="cpu", weights_only=False)
    a = cp["args"]
    enc_sd = cp["encoder"]
    embed_dim = enc_sd["cls_token"].shape[-1]                                   # 384
    depth = len({k.split(".")[1] for k in enc_sd if k.startswith("blocks.")})   # 12

    encoder = VisionTransformer(
        img_size=[cp["img_size"]], patch_size=a.patch_size, embed_dim=embed_dim,
        depth=depth, num_heads=NUM_HEADS[a.arch_encoder], mlp_ratio=4.0,
        qkv_bias=False, num_classes=1)   # num_classes=1: unused head, kept so keys match
    encoder.load_state_dict(enc_sd, strict=True)

    n_blocks = a.n_last_blocks                                                  # 4
    decoder = LinearClassifier(embed_dim=embed_dim * (n_blocks + int(a.avgpool_patchtokens)),
                               num_cls=cp["num_class"], activation=nn.Sigmoid())
    decoder.load_state_dict(cp["decoder"], strict=True)

    # cp["relation"] is an nn.Parameter, so it is registered and follows model.to(device)
    model = Ensemble(encoder, decoder, cp["relation"], n_last_blocks=n_blocks,
                     avgpool_patchtokens=a.avgpool_patchtokens)
    model.eval()
    class_names = [cp["cat_to_name"][i] for i in range(cp["num_class"])]
    return model, class_names, cp


def reorder_index(model_names, target_names):
    """Column index list such that probs[:, idx] follows target_names' order."""
    assert sorted(model_names) == sorted(target_names), (model_names, target_names)
    return [model_names.index(n) for n in target_names]
