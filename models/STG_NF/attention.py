"""Attention mechanisms for ST-GCN blocks.

Provides learnable ``nn.Module`` wrappers for Dual-Attention (DAM),
Triplet-Attention, Skeleton-Only, and Frame-Only attention that operate
on ``(B, C, T, V)`` pose tensors.

All modules accept explicit constructor parameters instead of relying on
global CLI args, so the module works cleanly when imported from a notebook
or any context that does not call ``argparse`` at import time.
"""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """Single multi-head attention layer.

    Args:
        in_channels: Channels fed to the per-head Conv2d (pooling dimension).
        out_channels: Output channels per head.
        num_heads: Number of parallel attention heads.
        ctv_size: Tuple ``(C, T, V)`` used to size the projection layer.
        attention_type: One of ``'skeleton'``, ``'frame'``, ``'channel'``.
        opt: Pooling strategy — ``'maxpool'`` or ``'zpool'``.
        n_mecatt_inside: Number of inner iterations per forward call.
        device: Device string (e.g. ``'cuda:0'``).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        ctv_size,
        num_heads=1,
        attention_type="skeleton",
        opt="maxpool",
        n_mecatt_inside=1,
        device="cuda:0",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.attention_type = attention_type
        self.opt = opt
        self.n_mecatt_inside = n_mecatt_inside
        C, T, V = ctv_size

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 7), padding="same", bias=True)
            for _ in range(num_heads)
        ])
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm2d(out_channels) for _ in range(num_heads)
        ])
        self.sigmoid = nn.Sigmoid()
        self.projection = nn.Linear(num_heads * C * T * V, C * T * V)

        self.to(device)

    def forward(self, x):
        for _ in range(self.n_mecatt_inside):
            head_outputs = []
            for conv, batch_norm in zip(self.convs, self.batch_norms):
                if self.attention_type == "skeleton":
                    x_permuted = x.permute(0, 2, 1, 3)
                elif self.attention_type == "frame":
                    x_permuted = x.permute(0, 3, 2, 1)
                elif self.attention_type == "channel":
                    x_permuted = x
                else:
                    raise ValueError(f"Unknown attention_type: {self.attention_type!r}")

                if self.opt == "maxpool":
                    x_max, _ = x_permuted.max(dim=1, keepdim=True)
                    x_conv = conv(x_max)
                elif self.opt == "zpool":
                    # Pool the *permuted* tensor so the conv output shape is
                    # broadcast-compatible with x_permuted in the multiply below.
                    x_zpool = torch.cat(
                        (torch.max(x_permuted, 1)[0].unsqueeze(1),
                         torch.mean(x_permuted, 1).unsqueeze(1)),
                        dim=1,
                    )
                    x_conv = conv(x_zpool)
                else:
                    raise ValueError(f"Unknown opt: {self.opt!r}")

                x_final = self.sigmoid(batch_norm(x_conv))
                head_outputs.append(torch.mul(x_final, x_permuted))

            concatenated = torch.cat(head_outputs, dim=1)
            projected = self.projection(concatenated.reshape(concatenated.size(0), -1))
            x = projected.view_as(x)
        return x


class DualAttention(nn.Module):
    """Dual-Attention (DAM): average of skeleton + frame attention + residual."""

    def __init__(self, ctv_size, n_heads=1, n_mecatt_inside=1, device="cuda:0"):
        super().__init__()
        self.att_skl = MultiHeadAttention(
            1, 1, ctv_size,
            num_heads=n_heads, attention_type="skeleton",
            opt="maxpool", n_mecatt_inside=n_mecatt_inside, device=device,
        )
        self.att_frame = MultiHeadAttention(
            1, 1, ctv_size,
            num_heads=n_heads, attention_type="frame",
            opt="maxpool", n_mecatt_inside=n_mecatt_inside, device=device,
        )

    def forward(self, x):
        return 0.5 * (self.att_skl(x) + self.att_frame(x)) + x


class TripletAttention(nn.Module):
    """Triplet-Attention: average of skeleton + frame + channel attention + residual."""

    def __init__(self, ctv_size, n_heads=1, n_mecatt_inside=1, device="cuda:0"):
        super().__init__()
        self.att_skl = MultiHeadAttention(
            2, 1, ctv_size,
            num_heads=n_heads, attention_type="skeleton",
            opt="zpool", n_mecatt_inside=n_mecatt_inside, device=device,
        )
        self.att_frame = MultiHeadAttention(
            2, 1, ctv_size,
            num_heads=n_heads, attention_type="frame",
            opt="zpool", n_mecatt_inside=n_mecatt_inside, device=device,
        )
        self.att_identity = MultiHeadAttention(
            2, 1, ctv_size,
            num_heads=n_heads, attention_type="channel",
            opt="zpool", n_mecatt_inside=n_mecatt_inside, device=device,
        )

    def forward(self, x):
        return (1.0 / 3.0) * (self.att_skl(x) + self.att_frame(x) + self.att_identity(x)) + x


class SkeletonAttention(nn.Module):
    """Skeleton-only attention + residual."""

    def __init__(self, ctv_size, n_heads=1, n_mecatt_inside=1, device="cuda:0"):
        super().__init__()
        self.att = MultiHeadAttention(
            1, 1, ctv_size,
            num_heads=n_heads, attention_type="skeleton",
            opt="maxpool", n_mecatt_inside=n_mecatt_inside, device=device,
        )

    def forward(self, x):
        return self.att(x)


class FrameAttention(nn.Module):
    """Frame-only attention + residual."""

    def __init__(self, ctv_size, n_heads=1, n_mecatt_inside=1, device="cuda:0"):
        super().__init__()
        self.att = MultiHeadAttention(
            1, 1, ctv_size,
            num_heads=n_heads, attention_type="frame",
            opt="maxpool", n_mecatt_inside=n_mecatt_inside, device=device,
        )

    def forward(self, x):
        return self.att(x)


def build_attention_module(attention_type, ctv_size, n_heads, n_mecatt_inside, device):
    """Factory: return an ``nn.Module`` for the requested attention variant.

    Args:
        attention_type: ``'dual'``, ``'triplet'``, ``'skeleton'``, ``'frame'``.
        ctv_size: ``(C, T, V)`` tuple from the pose shape.
        n_heads: Number of attention heads.
        n_mecatt_inside: Inner iterations per forward call.
        device: Target device string.

    Returns:
        An ``nn.Module`` whose ``forward(x)`` applies the attention, or
        ``None`` if *attention_type* is ``'none'``.
    """
    if attention_type == "none":
        return None
    if attention_type == "dual":
        return DualAttention(ctv_size, n_heads, n_mecatt_inside, device)
    if attention_type == "triplet":
        return TripletAttention(ctv_size, n_heads, n_mecatt_inside, device)
    if attention_type == "skeleton":
        return SkeletonAttention(ctv_size, n_heads, n_mecatt_inside, device)
    if attention_type == "frame":
        return FrameAttention(ctv_size, n_heads, n_mecatt_inside, device)
    raise ValueError(f"Unknown attention_type: {attention_type!r}")
