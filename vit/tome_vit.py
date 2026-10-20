"""Token Merging (ToMe; Bolya et al., ICLR 2023) for the bikit ViT. No training needed.

Inside every block, right after attention, the r most similar pairs of tokens are
merged into one (bipartite soft matching on the attention keys, averaged over heads).
A merged token remembers how many patches it stands for ("size"):
  - merging is a size-weighted average,
  - attention adds log(size) to its logits ("proportional attention"), so a token
    that stands for 5 patches gets the attention weight of 5 patches.
The CLS token is never merged and stays at position 0, because the classifier reads
the CLS token of the last 4 blocks. Tokens go 785 -> 785 - 12 r over the 12 blocks.

    from tome_vit import apply_tome
    apply_tome(model, r=16)        # model from vit_model.load_vit(); r=0 restores the exact original
    model.encoder._tome["tokens"]  # tokens entering each block during the last forward pass
"""
import math
import types

import torch


def bipartite_soft_matching(metric, r):
    """Returns merge(x, mode) that removes r tokens from (B, N, C) tensors; token 0 is protected."""
    t = metric.shape[1]
    r = min(r, (t - 1) // 2)              # at most half the tokens, CLS excluded
    if r <= 0:
        return None
    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]        # alternate tokens into sets A and B
        scores = a @ b.transpose(-1, -2)                         # cosine similarity A x B
        scores[..., 0, :] = -math.inf                            # CLS (A[0]) is never merged
        node_max, node_idx = scores.max(dim=-1)                  # best partner in B for each A token
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r:, :].sort(dim=-2)[0]           # A tokens kept, original order (CLS first)
        src_idx = edge_idx[..., :r, :]                           # A tokens merged away
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    def merge(x, mode="sum"):
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        return torch.cat([unm, dst], dim=1)
    return merge


def _attn_forward(self, x, size):
    """Attention as in vit_model (any head count, optional head gates), plus proportional attention;
    returns the keys averaged over heads, used for matching."""
    B, N, _ = x.shape
    h = self.num_heads
    d = self.qkv.out_features // (3 * h)
    qkv = self.qkv(x).reshape(B, N, 3, h, d).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    attn = (q @ k.transpose(-2, -1)) * self.scale
    if size is not None:
        attn = attn + size.log()[:, None, None, :, 0]
    out = self.attn_drop(attn.softmax(dim=-1)) @ v
    gate = getattr(self, "head_gate", None)
    if gate is not None:
        out = out * gate.to(out.dtype)[None, :, None, None]
    x = self.proj_drop(self.proj(out.transpose(1, 2).reshape(B, N, h * d)))
    return x, k.mean(dim=1)


def _block_forward(self, x, return_attention=False):
    info = self._tome
    info["tokens"].append(x.shape[1])
    y, metric = self.attn.tome_forward(self.norm1(x), info["size"])
    x = x + self.drop_path(y)
    merge = bipartite_soft_matching(metric, info["r"][self._tome_idx])
    if merge is not None:
        size = info["size"] if info["size"] is not None else torch.ones_like(x[..., :1])
        x = merge(x * size) / merge(size)          # size-weighted average
        info["size"] = merge(size)
    return x + self.drop_path(self.mlp(self.norm2(x)))


def apply_tome(model, r):
    """Patch model.encoder in place. r: int (same for every block) or a list of 12 ints."""
    enc = model.encoder
    depth = len(enc.blocks)
    r_list = list(r) if isinstance(r, (list, tuple)) else [int(r)] * depth
    assert len(r_list) == depth
    if not hasattr(enc, "_tome"):
        enc._tome = {}
        orig_prepare = enc.prepare_tokens

        def prepare_tokens(x):                    # new image batch: reset sizes and the token trace
            enc._tome["size"], enc._tome["tokens"] = None, []
            return orig_prepare(x)
        enc.prepare_tokens = prepare_tokens
        for i, blk in enumerate(enc.blocks):
            blk._tome, blk._tome_idx = enc._tome, i
            blk.attn.tome_forward = types.MethodType(_attn_forward, blk.attn)
            blk.forward = types.MethodType(_block_forward, blk)
    enc._tome.update(r=r_list, size=None, tokens=[])
    return model


def vit_flops(tokens, r_list, dims=None, dim=384, head_dim=64, patch_in=3 * 8 * 8, n_patches=784,
              head_in=1536, n_cls=6):
    """FLOPs (1 multiply-add = 2) of one image, from the tokens entering each block.
    dims: [(heads, mlp_hidden)] per block (vit_model.encoder_dims); default: the unpruned ViT-S/8."""
    dims = dims or [(6, 1536)] * len(tokens)
    macs = n_patches * patch_in * dim + head_in * n_cls          # patch embedding + classifier
    for n_in, r, (h, m) in zip(tokens, r_list, dims):
        r_eff = max(0, min(r, (n_in - 1) // 2))
        n_out = n_in - r_eff
        inner = h * head_dim
        macs += n_in * dim * 3 * inner + n_in * inner * dim      # qkv + proj
        macs += 2 * n_in * n_in * inner                          # QK^T and attn @ V
        macs += 2 * n_out * dim * m                              # MLP after merging
        if r_eff:
            macs += ((n_in + 1) // 2) * (n_in // 2) * head_dim   # matching scores
    return 2 * macs
