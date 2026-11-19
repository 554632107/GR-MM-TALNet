# -*- coding: utf-8 -*-
"""
FT-Transformer for the tabular baseline.
Each feature is represented as a token. A [CLS] token is added, and all tokens pass through one attention layer.
The number of input features is configurable; this study uses 116.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureTransformer(nn.Module):
    def __init__(self, n_features: int = 32, d_model: int = 32, n_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.n_features = n_features
        self.w = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))
        self.col_emb = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 1))

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        tok = x.unsqueeze(-1) * self.w + self.bias + self.col_emb
        cls = self.cls.expand(x.size(0), -1, -1)
        return torch.cat([cls, tok], dim=1)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        h = self.tokenize(x)
        attn_out, w = self.attn(h, h, h, need_weights=True, average_attn_weights=True)
        h = self.norm1(h + attn_out)
        h = self.norm2(h + self.ff(h))
        logit = self.head(h[:, 0]).squeeze(-1)
        if return_attn:
            return logit, w[:, 0, 1:]
        return logit
