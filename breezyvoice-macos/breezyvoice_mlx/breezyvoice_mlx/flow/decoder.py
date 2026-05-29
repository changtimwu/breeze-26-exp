"""UNet1D flow-matching decoder (the CFM estimator).  [PORT STATUS: IMPLEMENTED — parity-tested]

MLX port of BreezyVoice's Matcha-TTS non-causal ConditionalDecoder
(../BreezyVoice/cosyvoice/flow/decoder.py): SinusoidalPosEmb + TimestepEmbedding,
down/mid/up blocks of [ResnetBlock1D + N x BasicTransformerBlock] with
Downsample1D / ConvTranspose Upsample1D, then Block1D + 1x1 final_proj.

The conv/norm blocks are reimplemented here because mlx-audio-plus's non-causal
Block1D applies GroupNorm in (B,C,T) layout (MLX needs channels-last) — its
shipped path is causal, so the non-causal branch is unexercised/broken. The
verified BasicTransformerBlock (diffusers attention + GELU FFN, operates in
(B,T,C)) is reused from mlx-audio-plus. Attribute names match mlx-audio's
"sanitized" scheme so `remap_decoder_weights` maps BreezyVoice's torch checkpoint
keys onto this module. Parity vs the torch source: tests/test_decoder_parity.py.
"""

from __future__ import annotations

import math
import re

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.codec.models.s3gen.matcha.transformer import BasicTransformerBlock


# --- building blocks (channels-correct) -------------------------------------

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def __call__(self, x: mx.array, scale: float = 1000) -> mx.array:
        if x.ndim < 1:
            x = mx.expand_dims(x, 0)
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = mx.exp(mx.arange(half, dtype=mx.float32) * -emb)
        emb = scale * mx.expand_dims(x, 1) * mx.expand_dims(emb, 0)
        return mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int, act_fn: str = "silu"):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_2(nn.silu(self.linear_1(x)))


class Block1D(nn.Module):
    def __init__(self, dim: int, dim_out: int, groups: int = 8):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out, pytorch_compatible=True)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        # x: (B, C, T), mask: (B, 1, T). Work in (B, T, C) for conv + groupnorm.
        h = mx.swapaxes(x * mask, 1, 2)        # (B, T, C)
        h = self.conv(h)
        h = self.norm(h)                        # GroupNorm channels-last
        h = nn.mish(h)
        h = mx.swapaxes(h, 1, 2)               # (B, C', T)
        return h * mask


class ResnetBlock1D(nn.Module):
    def __init__(self, dim: int, dim_out: int, time_emb_dim: int, groups: int = 8):
        super().__init__()
        self.mlp_linear = nn.Linear(time_emb_dim, dim_out)  # torch mlp.1 (after Mish)
        self.block1 = Block1D(dim, dim_out, groups)
        self.block2 = Block1D(dim_out, dim_out, groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1)

    def __call__(self, x: mx.array, mask: mx.array, time_emb: mx.array) -> mx.array:
        h = self.block1(x, mask)
        h = h + mx.expand_dims(self.mlp_linear(nn.mish(time_emb)), -1)
        h = self.block2(h, mask)
        r = mx.swapaxes(self.res_conv(mx.swapaxes(x * mask, 1, 2)), 1, 2)
        return h + r


class Downsample1D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, stride=2, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return mx.swapaxes(self.conv(mx.swapaxes(x, 1, 2)), 1, 2)


class Upsample1D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, stride=2, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return mx.swapaxes(self.conv(mx.swapaxes(x, 1, 2)), 1, 2)


class _PlainConv1d(nn.Conv1d):
    """A bare Conv1d operating on (B, C, T) — the is_last down/up sampler.
    Subclasses nn.Conv1d so its params are .weight/.bias directly (torch's
    is_last sampler is a plain Conv1d, key '...downsample.weight')."""
    def __init__(self, dim: int):
        super().__init__(dim, dim, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return mx.swapaxes(super().__call__(mx.swapaxes(x, 1, 2)), 1, 2)


# --- block containers (indexed attrs -> sanitized keys) ---------------------

class _DownBlock(nn.Module):
    def __init__(self, resnet, transformers, downsample):
        super().__init__()
        self.resnet = resnet
        for i, b in enumerate(transformers):
            setattr(self, f"transformer_{i}", b)
        self.n = len(transformers)
        self.downsample = downsample

    @property
    def transformers(self):
        return [getattr(self, f"transformer_{i}") for i in range(self.n)]


class _MidBlock(nn.Module):
    def __init__(self, resnet, transformers):
        super().__init__()
        self.resnet = resnet
        for i, b in enumerate(transformers):
            setattr(self, f"transformer_{i}", b)
        self.n = len(transformers)

    @property
    def transformers(self):
        return [getattr(self, f"transformer_{i}") for i in range(self.n)]


class _UpBlock(nn.Module):
    def __init__(self, resnet, transformers, upsample):
        super().__init__()
        self.resnet = resnet
        for i, b in enumerate(transformers):
            setattr(self, f"transformer_{i}", b)
        self.n = len(transformers)
        self.upsample = upsample

    @property
    def transformers(self):
        return [getattr(self, f"transformer_{i}") for i in range(self.n)]


def _attn_bias(mask: mx.array) -> mx.array:
    """mask (B,1,T) -> additive attention bias (B,T,T): matmul(mask^T, mask)."""
    m = mx.swapaxes(mask, 1, 2) @ mask          # (B, T, T) outer product (0/1)
    return (1.0 - m) * -1.0e10


# --- the decoder ------------------------------------------------------------

class ConditionalDecoder(nn.Module):
    def __init__(self, in_channels: int = 320, out_channels: int = 80,
                 channels=(256, 256), dropout: float = 0.0,
                 attention_head_dim: int = 64, n_blocks: int = 4,
                 num_mid_blocks: int = 12, num_heads: int = 8, act_fn: str = "gelu"):
        super().__init__()
        channels = tuple(channels)
        self.out_channels = out_channels
        self.time_embeddings = SinusoidalPosEmb(in_channels)
        time_embed_dim = channels[0] * 4
        self.time_mlp = TimestepEmbedding(in_channels, time_embed_dim, "silu")

        def transformers(ch):
            return [BasicTransformerBlock(ch, num_heads, attention_head_dim, dropout, act_fn)
                    for _ in range(n_blocks)]

        out_ch = in_channels
        for i, ch in enumerate(channels):
            in_ch, out_ch = out_ch, ch
            is_last = i == len(channels) - 1
            ds = Downsample1D(out_ch) if not is_last else _PlainConv1d(out_ch)
            setattr(self, f"down_blocks_{i}",
                    _DownBlock(ResnetBlock1D(in_ch, out_ch, time_embed_dim),
                               transformers(out_ch), ds))
        self.n_down = len(channels)

        for i in range(num_mid_blocks):
            setattr(self, f"mid_blocks_{i}",
                    _MidBlock(ResnetBlock1D(channels[-1], channels[-1], time_embed_dim),
                              transformers(channels[-1])))
        self.n_mid = num_mid_blocks

        rev = list(reversed(channels)) + [channels[0]]
        for i in range(len(rev) - 1):
            in_ch, out_ch = rev[i] * 2, rev[i + 1]
            is_last = i == len(rev) - 2
            us = Upsample1D(out_ch) if not is_last else _PlainConv1d(out_ch)
            setattr(self, f"up_blocks_{i}",
                    _UpBlock(ResnetBlock1D(in_ch, out_ch, time_embed_dim),
                             transformers(out_ch), us))
        self.n_up = len(rev) - 1

        self.final_block = Block1D(rev[-1], rev[-1])
        self.final_proj = nn.Conv1d(rev[-1], out_channels, 1)

    @property
    def down_blocks(self):
        return [getattr(self, f"down_blocks_{i}") for i in range(self.n_down)]

    @property
    def mid_blocks(self):
        return [getattr(self, f"mid_blocks_{i}") for i in range(self.n_mid)]

    @property
    def up_blocks(self):
        return [getattr(self, f"up_blocks_{i}") for i in range(self.n_up)]

    def __call__(self, x, mask, mu, t, spks=None, cond=None):
        t = self.time_mlp(self.time_embeddings(t))
        x = mx.concatenate([x, mu], axis=1)
        if spks is not None:
            spks_e = mx.broadcast_to(mx.expand_dims(spks, -1),
                                     (spks.shape[0], spks.shape[1], x.shape[-1]))
            x = mx.concatenate([x, spks_e], axis=1)
        if cond is not None:
            x = mx.concatenate([x, cond], axis=1)

        hiddens, masks = [], [mask]
        for blk in self.down_blocks:
            md = masks[-1]
            x = blk.resnet(x, md, t)
            xt = mx.swapaxes(x, 1, 2)
            bias = _attn_bias(md)
            for tb in blk.transformers:
                xt = tb(xt, attention_mask=bias, timestep=t)
            x = mx.swapaxes(xt, 1, 2)
            hiddens.append(x)
            x = blk.downsample(x * md)
            masks.append(md[:, :, ::2])
        masks = masks[:-1]
        mm = masks[-1]

        for blk in self.mid_blocks:
            x = blk.resnet(x, mm, t)
            xt = mx.swapaxes(x, 1, 2)
            bias = _attn_bias(mm)
            for tb in blk.transformers:
                xt = tb(xt, attention_mask=bias, timestep=t)
            x = mx.swapaxes(xt, 1, 2)

        for blk in self.up_blocks:
            mu_ = masks.pop()
            skip = hiddens.pop()
            x = mx.concatenate([x[:, :, : skip.shape[-1]], skip], axis=1)
            x = blk.resnet(x, mu_, t)
            xt = mx.swapaxes(x, 1, 2)
            bias = _attn_bias(mu_)
            for tb in blk.transformers:
                xt = tb(xt, attention_mask=bias, timestep=t)
            x = mx.swapaxes(xt, 1, 2)
            x = blk.upsample(x * mu_)

        x = self.final_block(x, mu_)
        out = mx.swapaxes(self.final_proj(mx.swapaxes(x * mu_, 1, 2)), 1, 2)
        return out * mask


# --- checkpoint key remap ---------------------------------------------------

def _remap_key(k: str) -> str:
    def block(m):
        grp, i, sub = m.group(1), m.group(2), m.group(3)
        tail = {"down": "downsample", "up": "upsample"}[grp]
        if sub == "0":
            return f"{grp}_blocks_{i}.resnet."
        if sub.startswith("1."):
            return f"{grp}_blocks_{i}.transformer_{sub.split('.')[1]}."
        if sub == "2":
            return f"{grp}_blocks_{i}.{tail}."
        return m.group(0)
    k = re.sub(r"(down|up)_blocks\.(\d+)\.(0|1\.\d+|2)\.", block, k)
    k = re.sub(r"mid_blocks\.(\d+)\.0\.", r"mid_blocks_\1.resnet.", k)
    k = re.sub(r"mid_blocks\.(\d+)\.1\.(\d+)\.", r"mid_blocks_\1.transformer_\2.", k)
    k = k.replace("mlp.1.", "mlp_linear.")
    k = k.replace("block1.block.0.", "block1.conv.").replace("block1.block.1.", "block1.norm.")
    k = k.replace("block2.block.0.", "block2.conv.").replace("block2.block.1.", "block2.norm.")
    k = k.replace("final_block.block.0.", "final_block.conv.").replace("final_block.block.1.", "final_block.norm.")
    k = k.replace("attn1.to_q.", "attn.query_proj.").replace("attn1.to_k.", "attn.key_proj.")
    k = k.replace("attn1.to_v.", "attn.value_proj.").replace("attn1.to_out.0.", "attn.out_proj.")
    k = k.replace("ff.net.0.proj.", "ff.layers.0.").replace("ff.net.2.", "ff.layers.1.")
    return k


def remap_decoder_weights(torch_state: dict) -> list:
    """BreezyVoice torch decoder state_dict -> MLX (key rename + conv layout)."""
    import numpy as np
    out = []
    for k, v in torch_state.items():
        a = np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v, dtype=np.float32)
        nk = _remap_key(k)
        if k.endswith(".weight") and a.ndim == 3:
            if ".upsample.conv." in nk:    # ConvTranspose1d (in,out,k) -> (out,k,in)
                a = np.transpose(a, (1, 2, 0))
            else:                     # Conv1d (out,in,k) -> (out,k,in)
                a = np.transpose(a, (0, 2, 1))
        out.append((nk, mx.array(a)))
    return out
