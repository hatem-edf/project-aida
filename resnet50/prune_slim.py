#!/usr/bin/env python
"""Stage 2 of network slimming: physically remove low-gamma channels.

Only bn1/bn2 inside each Bottleneck are prunable. bn3, conv3's outputs and the
downsample path all feed the residual addition, so their widths are fixed by
the block's input and must stay untouched.

Channel selection is a single GLOBAL threshold over every prunable gamma in the
network (the paper's method: layers compete for budget), with a per-layer floor
so no layer can be wiped out entirely.
"""
import argparse, json, os, copy
import torch
from torch import nn


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------
def bottlenecks(model):
    """[(name, block)] for every Bottleneck in the backbone, in forward order."""
    return [(n, m) for n, m in model.named_modules()
            if type(m).__name__ == "Bottleneck"]


def all_prunable_gammas(model):
    g = []
    for _, blk in bottlenecks(model):
        g.append(blk.bn1.weight.detach().abs().flatten())
        g.append(blk.bn2.weight.detach().abs().flatten())
    return torch.cat(g)


# --------------------------------------------------------------------------
# mask computation
# --------------------------------------------------------------------------
def compute_masks(model, ratio, min_keep=0.10, verbose=True):
    """Global threshold at `ratio` quantile; keep >= min_keep fraction per layer.

    Returns {block_name: {"bn1": LongTensor(keep_idx), "bn2": ...}} plus stats.
    """
    g = all_prunable_gammas(model)
    thr = torch.quantile(g.float(), ratio).item()

    masks, stats = {}, []
    for name, blk in bottlenecks(model):
        entry = {}
        for tag in ("bn1", "bn2"):
            bn = getattr(blk, tag)
            w = bn.weight.detach().abs()
            keep = (w > thr).nonzero(as_tuple=True)[0]
            floor = max(1, int(round(min_keep * w.numel())))
            if keep.numel() < floor:
                # keep the `floor` largest gammas instead
                keep = torch.argsort(w, descending=True)[:floor].sort().values
            entry[tag] = keep
            stats.append((f"{name}.{tag}", w.numel(), keep.numel()))
        masks[name] = entry

    tot_before = sum(s[1] for s in stats)
    tot_after = sum(s[2] for s in stats)
    if verbose:
        print(f"global threshold {thr:.6f} (ratio {ratio})")
        print(f"prunable channels {tot_before} -> {tot_after} "
              f"({100*(1-tot_after/tot_before):.1f}% removed)")
        for n, b, a in stats:
            bar = "#" * int(30 * a / b)
            print(f"  {n:<24} {b:4d} -> {a:4d}  {bar}")
    return masks, {"threshold": thr, "ratio": ratio,
                   "channels_before": tot_before, "channels_after": tot_after,
                   "per_layer": [{"layer": n, "before": b, "after": a}
                                 for n, b, a in stats]}


# --------------------------------------------------------------------------
# surgery helpers
# --------------------------------------------------------------------------
def _slice_bn(bn, keep):
    new = nn.BatchNorm2d(keep.numel(), eps=bn.eps, momentum=bn.momentum,
                         affine=bn.affine, track_running_stats=bn.track_running_stats)
    new.weight.data.copy_(bn.weight.data[keep])
    new.bias.data.copy_(bn.bias.data[keep])
    if bn.track_running_stats:
        new.running_mean.copy_(bn.running_mean[keep])
        new.running_var.copy_(bn.running_var[keep])
        new.num_batches_tracked.copy_(bn.num_batches_tracked)
    return new


def _slice_conv(conv, keep_out=None, keep_in=None, extra_bias=None):
    w = conv.weight.data
    if keep_out is not None:
        w = w[keep_out]
    if keep_in is not None:
        w = w[:, keep_in]
    need_bias = conv.bias is not None or extra_bias is not None
    new = nn.Conv2d(w.shape[1], w.shape[0], kernel_size=conv.kernel_size,
                    stride=conv.stride, padding=conv.padding,
                    dilation=conv.dilation, groups=conv.groups, bias=need_bias)
    new.weight.data.copy_(w)
    if need_bias:
        b = torch.zeros(w.shape[0])
        if conv.bias is not None:
            b += conv.bias.data[keep_out] if keep_out is not None else conv.bias.data
        if extra_bias is not None:
            b += extra_bias
        new.bias.data.copy_(b)
    return new


def _dropped_contribution(bn, conv_next, dropped, keep_out):
    """Constant a removed channel still injects into the next conv.

    A channel with gamma~0 outputs the constant beta, so after ReLU it is
    relu(beta). Folding that into the next conv's bias keeps the pruned model
    numerically close to the original before any fine-tuning. Approximate at
    spatial borders because of padding, exact in the interior.
    """
    if dropped.numel() == 0:
        return None
    c = torch.relu(bn.bias.data[dropped])                 # (D,)
    w = conv_next.weight.data[:, dropped]                 # (O, D, kh, kw)
    if keep_out is not None:
        w = conv_next.weight.data[keep_out][:, dropped]
    return (w.sum(dim=(2, 3)) * c.unsqueeze(0)).sum(dim=1)


# --------------------------------------------------------------------------
# main surgery
# --------------------------------------------------------------------------
def prune_model(model, masks, absorb_bias=True):
    model = copy.deepcopy(model).cpu().eval()
    for name, blk in bottlenecks(model):
        k1, k2 = masks[name]["bn1"], masks[name]["bn2"]
        n1, n2 = blk.bn1.weight.numel(), blk.bn2.weight.numel()
        d1 = torch.tensor([i for i in range(n1) if i not in set(k1.tolist())], dtype=torch.long)
        d2 = torch.tensor([i for i in range(n2) if i not in set(k2.tolist())], dtype=torch.long)

        eb2 = _dropped_contribution(blk.bn1, blk.conv2, d1, k2) if absorb_bias else None
        eb3 = _dropped_contribution(blk.bn2, blk.conv3, d2, None) if absorb_bias else None

        new_conv1 = _slice_conv(blk.conv1, keep_out=k1)
        new_bn1 = _slice_bn(blk.bn1, k1)
        new_conv2 = _slice_conv(blk.conv2, keep_out=k2, keep_in=k1, extra_bias=eb2)
        new_bn2 = _slice_bn(blk.bn2, k2)
        new_conv3 = _slice_conv(blk.conv3, keep_out=None, keep_in=k2, extra_bias=eb3)

        blk.conv1, blk.bn1 = new_conv1, new_bn1
        blk.conv2, blk.bn2 = new_conv2, new_bn2
        blk.conv3 = new_conv3
    return model


def channel_config(model):
    """Per-block widths, needed to rebuild this architecture at load time."""
    return {n: [b.conv1.out_channels, b.conv2.out_channels] for n, b in bottlenecks(model)}


def rebuild_from_config(ref_model, config, absorb_bias=True):
    """Shrink a fresh DaclNet to `config` widths so a pruned state_dict loads."""
    model = copy.deepcopy(ref_model).cpu()
    for name, blk in bottlenecks(model):
        w1, w2 = config[name]
        blk.conv1 = nn.Conv2d(blk.conv1.in_channels, w1, 1, bias=False)
        blk.bn1 = nn.BatchNorm2d(w1)
        blk.conv2 = nn.Conv2d(w1, w2, 3, stride=blk.conv2.stride,
                              padding=blk.conv2.padding, bias=absorb_bias)
        blk.bn2 = nn.BatchNorm2d(w2)
        blk.conv3 = nn.Conv2d(w2, blk.conv3.out_channels, 1, bias=absorb_bias)
    return model


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def load_pruned(path, map_location="cpu"):
    """Load a checkpoint written by this script (or an unpruned bikit one)."""
    from bikit.models import DaclNet
    cp = torch.load(path, map_location=map_location, weights_only=False)
    model = DaclNet(base_name=cp["base"], resolution=cp["resolution"],
                    hidden_layers=cp["hidden_layers"], drop_prob=cp["drop_prob"],
                    num_class=cp["num_class"], freeze_base=False)
    if "channel_config" in cp:
        model = rebuild_from_config(model, cp["channel_config"],
                                    absorb_bias=cp.get("absorb_bias", True))
    model.load_state_dict(cp["state_dict"])
    model.eval()
    return model, cp


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="sparsity-trained checkpoint")
    ap.add_argument("--ratio", type=float, default=0.4, help="fraction of channels to remove")
    ap.add_argument("--min-keep", type=float, default=0.10)
    ap.add_argument("--no-absorb", action="store_true",
                    help="do not fold removed channels' constant into next conv bias")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from bikit.models import DaclNet

    cp = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = DaclNet(base_name=cp["base"], resolution=cp["resolution"],
                    hidden_layers=cp["hidden_layers"], drop_prob=cp["drop_prob"],
                    num_class=cp["num_class"], freeze_base=False)
    model.load_state_dict(cp["state_dict"])
    model.eval()

    masks, stats = compute_masks(model, a.ratio, a.min_keep)
    pruned = prune_model(model, masks, absorb_bias=not a.no_absorb)

    p0, p1 = count_params(model), count_params(pruned)
    print(f"\nparameters {p0:,} -> {p1:,}  ({100*(1-p1/p0):.1f}% smaller)")

    # agreement check on random input -- only meaningful pre-fine-tuning
    torch.manual_seed(0)
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        a0 = torch.sigmoid(model(x))
        a1 = torch.sigmoid(pruned(x))
    print(f"max |prob diff| vs unpruned: {(a0-a1).abs().max().item():.4f}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save({**{k: v for k, v in cp.items() if k != "state_dict"},
                "state_dict": pruned.state_dict(),
                "channel_config": channel_config(pruned),
                "absorb_bias": not a.no_absorb,
                "prune_stats": {**stats, "params_before": p0, "params_after": p1}},
               a.out)
    print("saved", a.out)
    json.dump(stats, open(a.out.replace(".pth", "_stats.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
