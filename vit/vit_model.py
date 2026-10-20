"""Build and load bikit-models' codebrim-classif-balanced_ViT_s8_1, and width-pruned versions of it.

Width pruning removes whole attention heads and MLP hidden neurons, so a block may have fewer than
6 heads (qkv: 3*h*64 outputs, proj: h*64 inputs) and fewer than 1536 hidden neurons. The upstream
Attention.forward assumes h*64 == 384, so this module replaces it (class-wide) with a version that
works for any head count; for the unpruned model it computes exactly the same thing. Both forwards
also apply optional gates used during sparsity training (vit_gates.py):
    attn.head_gate    (h,)  multiplies each head's output before proj
    mlp.neuron_gate   (m,)  multiplies each hidden activation before fc2

A pruned checkpoint stores cp["prune_config"] = {"heads": [12 ints], "mlp_hidden": [12 ints]};
load_vit() shrinks the layers to those widths before loading the weights.
"""
import sys
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "bikit-models" / "models"))
import vistranet_1  # noqa: E402
from vistranet_1 import VisionTransformer, LinearClassifier, Ensemble  # noqa: E402

DEFAULT_CKPT = Path.home() / ".cache/bikit-models/codebrim-classif-balanced_ViT_s8_1.pth"
NUM_HEADS = {"vit_tiny": 3, "vit_small": 6}
HEAD_DIM = 64


# ---------------------------------------------------------------- forwards for any width
def _attention_forward(self, x):
    """Same math as vistranet_1.Attention.forward, for any number of heads; optional head gates."""
    B, N, _ = x.shape
    h = self.num_heads
    d = self.qkv.out_features // (3 * h)
    qkv = self.qkv(x).reshape(B, N, 3, h, d).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = self.attn_drop(attn.softmax(dim=-1))
    out = attn @ v                                            # (B, h, N, d)
    gate = getattr(self, "head_gate", None)
    if gate is not None:
        out = out * gate.to(out.dtype)[None, :, None, None]
    x = self.proj_drop(self.proj(out.transpose(1, 2).reshape(B, N, h * d)))
    return x, attn


def _mlp_forward(self, x):
    x = self.drop(self.act(self.fc1(x)))
    gate = getattr(self, "neuron_gate", None)
    if gate is not None:
        x = x * gate.to(x.dtype)
    return self.drop(self.fc2(x))


vistranet_1.Attention.forward = _attention_forward
vistranet_1.Mlp.forward = _mlp_forward


# ---------------------------------------------------------------- pruned architectures
def shrink_encoder(encoder, cfg):
    """Replace each block's layers by narrower ones: cfg = {"heads": [...], "mlp_hidden": [...]}."""
    dim = encoder.embed_dim
    for blk, h, m in zip(encoder.blocks, cfg["heads"], cfg["mlp_hidden"]):
        a = blk.attn
        a.num_heads = h
        a.qkv = nn.Linear(dim, 3 * h * HEAD_DIM, bias=a.qkv.bias is not None)
        a.proj = nn.Linear(h * HEAD_DIM, dim)
        blk.mlp.fc1 = nn.Linear(dim, m)
        blk.mlp.fc2 = nn.Linear(m, dim)
    return encoder


def encoder_dims(encoder):
    """[(heads, mlp_hidden)] per block: what FLOP counts need."""
    return [(b.attn.num_heads, b.mlp.fc1.out_features) for b in encoder.blocks]


def build_transform(img_size=224, mode="eval"):
    """mode="eval": resize to 256x256 (squash, aspect ratio NOT kept), center-crop 224.
        Reproduces the published result: 501/632 vs the authors' 503/632 (preproc_sweep.py).
    mode="demo": short side to 256 (aspect kept), center-crop 224, as in the authors'
        run.ipynb. Only for the golden test, whose reference outputs were made this way."""
    resize = ((256, 256) if mode == "eval" else 256)
    return transforms.Compose([
        transforms.Resize(resize, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def load_vit(cp_path=DEFAULT_CKPT):
    """Returns (model, class_names_in_output_order, checkpoint_dict). Strict loading."""
    cp = torch.load(cp_path, map_location="cpu", weights_only=False)
    a = cp["args"]
    enc_sd = cp["encoder"]
    embed_dim = enc_sd["cls_token"].shape[-1]                                   # 384
    depth = len({k.split(".")[1] for k in enc_sd if k.startswith("blocks.")})   # 12

    encoder = VisionTransformer(
        img_size=[cp["img_size"]], patch_size=a.patch_size, embed_dim=embed_dim,
        depth=depth, num_heads=NUM_HEADS[a.arch_encoder], mlp_ratio=4.0,
        qkv_bias=False, num_classes=1)   # num_classes=1: unused head, kept so keys match
    if "prune_config" in cp:
        shrink_encoder(encoder, cp["prune_config"])
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
