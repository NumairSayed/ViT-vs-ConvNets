"""
vit.py
------
PyTorch Vision Transformer for DLCV 2026 Assignment 2.
Ported from the official Google JAX implementation.

Uses F.scaled_dot_product_attention (PyTorch 2.0+) which automatically
dispatches to Flash Attention / memory-efficient attention on supported hardware.

Supports all 7 experiments:
    Exp 2: patch_size
    Exp 3: pooling         ('cls' | 'mean')
    Exp 4: pos_encoding    ('learnable' | 'sinusoidal' | 'none')
    Exp 5: return_attention=True  →  returns attention maps (disables Flash Attn)
    Exp 6: patch_stride    (set < patch_size for overlapping patches)
    Exp 7: return_all_layers=True  →  returns per-layer CLS/pooled features
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional Encodings  (Experiment 4)
# ---------------------------------------------------------------------------

class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, num_tokens: int, embed_dim: int):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.normal_(self.pos_embedding, std=0.02)

    def forward(self, x):
        return x + self.pos_embedding


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, num_tokens: int, embed_dim: int):
        super().__init__()
        pe  = torch.zeros(num_tokens, embed_dim)
        pos = torch.arange(num_tokens).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pos_embedding', pe.unsqueeze(0))  # (1, N, D)

    def forward(self, x):
        return x + self.pos_embedding


class NoPositionalEmbedding(nn.Module):
    def forward(self, x):
        return x


# ---------------------------------------------------------------------------
# Self-Attention using F.scaled_dot_product_attention
#
# Why not nn.MultiheadAttention?
#   nn.MultiheadAttention calls F.scaled_dot_product_attention internally BUT
#   only when need_weights=False. The moment you ask for attention weights
#   (Experiment 5), it falls back to the slow math path.
#
#   Writing our own thin wrapper gives us:
#     - Flash Attention / memory-efficient attention when return_attention=False
#     - Explicit fallback to math SDPA when return_attention=True (needed for maps)
#   Same level of correctness, much more control.
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    """
    Multi-head self-attention using F.scaled_dot_product_attention.

    Flash Attention is active during normal training (return_attention=False).
    When return_attention=True (Experiment 5), we disable it so we can
    extract the attention weights — Flash Attention fuses the kernel and
    doesn't materialise the attention matrix.
    """

    def __init__(self, embed_dim: int, num_heads: int, attn_dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.attn_drop  = attn_dropout

        # Single fused projection for Q, K, V  (faster than three separate linears)
        self.qkv  = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, return_attention: bool = False):
        B, N, D = x.shape

        # Project and split into Q, K, V
        qkv = self.qkv(x)                          # (B, N, 3D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)           # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)                    # each: (B, heads, N, head_dim)

        if return_attention:
            # --- Explicit softmax path so we can read attention weights ---
            # (needed for Experiment 5 — Flash Attn doesn't expose the matrix)
            scale   = self.head_dim ** -0.5
            attn_w  = (q @ k.transpose(-2, -1)) * scale   # (B, heads, N, N)
            attn_w  = attn_w.softmax(dim=-1)
            out     = attn_w @ v                           # (B, heads, N, head_dim)
        else:
            # --- Fast path: Flash Attention / mem-efficient / math (auto-selected) ---
            dropout = self.attn_drop if self.training else 0.0
            out     = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout)
            attn_w  = None

        # Merge heads and project
        out = out.transpose(1, 2).reshape(B, N, D)   # (B, N, D)
        out = self.proj(out)

        return (out, attn_w) if return_attention else out


# ---------------------------------------------------------------------------
# MLP Block
# ---------------------------------------------------------------------------

class MlpBlock(nn.Module):
    def __init__(self, embed_dim: int, mlp_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Encoder Block
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    """Pre-LN transformer block: LN → Attn → residual → LN → MLP → residual"""

    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int,
                 dropout: float = 0.1, attn_dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = SelfAttention(embed_dim, num_heads, attn_dropout)
        self.drop  = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = MlpBlock(embed_dim, mlp_dim, dropout)

    def forward(self, x, return_attention: bool = False):
        normed = self.norm1(x)

        if return_attention:
            attn_out, attn_w = self.attn(normed, return_attention=True)
            x = x + self.drop(attn_out)
            x = x + self.mlp(self.norm2(x))
            return x, attn_w                      # attn_w: (B, heads, N, N)
        else:
            x = x + self.drop(self.attn(normed))
            x = x + self.mlp(self.norm2(x))
            return x


# ---------------------------------------------------------------------------
# Encoder (full stack)
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, num_layers: int, embed_dim: int, num_heads: int,
                 mlp_dim: int, num_tokens: int, pos_encoding: str = 'learnable',
                 dropout: float = 0.1, attn_dropout: float = 0.0):
        super().__init__()

        if pos_encoding == 'learnable':
            self.pos_embed = LearnablePositionalEmbedding(num_tokens, embed_dim)
        elif pos_encoding == 'sinusoidal':
            self.pos_embed = SinusoidalPositionalEmbedding(num_tokens, embed_dim)
        elif pos_encoding == 'none':
            self.pos_embed = NoPositionalEmbedding()
        else:
            raise ValueError(f"Unknown pos_encoding: '{pos_encoding}'")

        self.dropout = nn.Dropout(dropout)
        self.blocks  = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, mlp_dim, dropout, attn_dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, return_attention: bool = False, return_all_layers: bool = False):
        x = self.pos_embed(x)
        x = self.dropout(x)

        attn_maps     = []
        layer_outputs = []

        for block in self.blocks:
            if return_attention:
                x, attn = block(x, return_attention=True)
                attn_maps.append(attn)
            else:
                x = block(x)

            if return_all_layers:
                layer_outputs.append(x)    # before final norm, good for linear probing

        x = self.norm(x)
        return x, attn_maps, layer_outputs


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """
    Conv2d-based patch tokeniser.
    stride=patch_size  →  non-overlapping  (default, Experiments 1-5, 7)
    stride<patch_size  →  overlapping      (Experiment 6)
    """

    def __init__(self, img_size: int = 32, patch_size: int = 4,
                 stride: int = None, in_channels: int = 3, embed_dim: int = 192):
        super().__init__()
        stride = stride if stride is not None else patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=stride, padding=0)
        h_out = (img_size - patch_size) // stride + 1
        self.num_patches = h_out * h_out

    def forward(self, x):
        x = self.proj(x)           # (B, D, h, w)
        x = x.flatten(2)           # (B, D, N)
        return x.transpose(1, 2)   # (B, N, D)


# ---------------------------------------------------------------------------
# Vision Transformer
# ---------------------------------------------------------------------------

class VisionTransformer(nn.Module):
    """
    Args:
        img_size          : Input image side length.
        patch_size        : Patch side length.                     [Exp 2]
        patch_stride      : Patch stride (default = patch_size).   [Exp 6]
        num_classes       : Output classes.
        embed_dim         : Token embedding dimension.
        depth             : Number of transformer layers.
        num_heads         : Attention heads.
        mlp_ratio         : MLP hidden = embed_dim * mlp_ratio.
        dropout           : Dropout rate.
        attn_dropout      : Attention dropout rate.
        pooling           : 'cls' | 'mean'                         [Exp 3]
        pos_encoding      : 'learnable' | 'sinusoidal' | 'none'   [Exp 4]
        return_attention  : Also return attention maps.            [Exp 5]
        return_all_layers : Also return per-layer features.        [Exp 7]
    """

    def __init__(
        self,
        img_size:          int   = 32,
        patch_size:        int   = 4,
        patch_stride:      int   = None,
        num_classes:       int   = 10,
        embed_dim:         int   = 192,
        depth:             int   = 9,
        num_heads:         int   = 12,
        mlp_ratio:         float = 2.0,
        dropout:           float = 0.1,
        attn_dropout:      float = 0.0,
        pooling:           str   = 'cls',
        pos_encoding:      str   = 'learnable',
        return_attention:  bool  = False,
        return_all_layers: bool  = False,
    ):
        super().__init__()
        self.pooling           = pooling
        self.return_attention  = return_attention
        self.return_all_layers = return_all_layers

        self.patch_embed = PatchEmbedding(img_size, patch_size, patch_stride, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        if pooling == 'cls':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            num_tokens = num_patches + 1
        else:
            self.cls_token = None
            num_tokens = num_patches

        self.encoder = Encoder(
            num_layers   = depth,
            embed_dim    = embed_dim,
            num_heads    = num_heads,
            mlp_dim      = int(embed_dim * mlp_ratio),
            num_tokens   = num_tokens,
            pos_encoding = pos_encoding,
            dropout      = dropout,
            attn_dropout = attn_dropout,
        )

        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.head.weight)
        if self.cls_token is not None:
            nn.init.zeros_(self.cls_token)

    def _pool(self, x):
        """Extract a single (B, D) vector from the sequence."""
        if self.pooling == 'cls':
            return x[:, 0]
        # mean over patch tokens only (skip CLS slot if it was prepended)
        start = 1 if self.cls_token is not None else 0
        return x[:, start:].mean(dim=1)

    def forward(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)

        if self.cls_token is not None:
            cls = self.cls_token.expand(B, -1, -1)
            x   = torch.cat([cls, x], dim=1)

        x, attn_maps, layer_outputs = self.encoder(
            x,
            return_attention  = self.return_attention,
            return_all_layers = self.return_all_layers,
        )

        logits = self.head(self._pool(x))

        extras = []
        if self.return_attention:
            extras.append(attn_maps)           # list[ (B, heads, N, N) ] × depth

        if self.return_all_layers:
            layer_feats = [self._pool(lx) for lx in layer_outputs]
            extras.append(layer_feats)         # list[ (B, D) ] × depth

        return (logits, *extras) if extras else logits


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def vit_tiny(num_classes=10, **kwargs):
    """ViT-Tiny — fast, good for CIFAR-10 experimentation."""
    return VisionTransformer(embed_dim=192, depth=9, num_heads=12,
                             mlp_ratio=2.0, num_classes=num_classes, **kwargs)

def vit_small(num_classes=10, **kwargs):
    """ViT-Small — more capacity, slower."""
    return VisionTransformer(embed_dim=384, depth=12, num_heads=12,
                             mlp_ratio=4.0, num_classes=num_classes, **kwargs)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    x = torch.randn(4, 3, 32, 32)

    tests = [
        ("default",          vit_small()),
        ("patch_size=8",     vit_small(patch_size=8)),
        ("pooling=mean",     vit_small(pooling='mean')),
        ("pos=sinusoidal",   vit_small(pos_encoding='sinusoidal')),
        ("pos=none",         vit_small(pos_encoding='none')),
        ("overlap stride=2", VisionTransformer(patch_size=4, patch_stride=2,
                                               embed_dim=192, depth=9, num_heads=12)),
        ("return_attention", vit_small(return_attention=True)),
        ("return_all_layers",vit_small(return_all_layers=True)),
    ]

    print("=" * 55)
    for name, model in tests:
        model.eval()
        with torch.no_grad():
            out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        extra  = f", extras={[type(e).__name__ for e in out[1:]]}" if isinstance(out, tuple) else ""
        print(f"  [{name:22s}]  logits={tuple(logits.shape)}{extra}")

    # Confirm attention map shapes
    model = vit_small(return_attention=True)
    
    num_trainable_params = sum([p.numel() for p in model.parameters()])
    print('\n' + 'num_trainable_params = ' + str(num_trainable_params) + '\n')
    
    model.eval()
    with torch.no_grad():
        logits, attn_maps = model(x)
    print(f"\n  Attention maps: {len(attn_maps)} layers, each {tuple(attn_maps[0].shape)}")
    print(f"  (B=4, heads=12, N+1={attn_maps[0].shape[-1]} tokens)")

    # Confirm layer feature shapes
    model = vit_small(return_all_layers=True)
    model.eval()
    with torch.no_grad():
        logits, layer_feats = model(x)
    print(f"\n  Layer features: {len(layer_feats)} layers, each {tuple(layer_feats[0].shape)}")
    print("=" * 55)
    print("All checks passed.")