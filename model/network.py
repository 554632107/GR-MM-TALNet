# -*- coding: utf-8 -*-
"""
Define multimodal networks and modality encoders.
MM_TALNet uses one shared residual; MM_TALNetGateSmall implements GR-MM-TALNet with separate gated text and ECG residuals.
Mask missing auxiliary modalities and use paired laboratory measurements when available.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalPairEncoder(nn.Module):
    """Shared temporal encoder mapping two measurements of each laboratory item to one token."""

    def __init__(self, d_model: int, dropout: float = 0.25):
        super().__init__()
        self.val = nn.Linear(2, d_model)
        self.time_emb = nn.Embedding(2, d_model)
        self.delta = nn.Linear(1, d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=2, dropout=dropout, batch_first=True)
        self.n1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.n2 = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model * 2, d_model)

    def forward(self, x_first, x_second, m_first, m_second, delta):
        b, n = x_second.shape
        t0 = self.val(torch.stack([x_first, m_first], dim=-1)) + self.time_emb.weight[0]
        t1 = self.val(torch.stack([x_second, m_second], dim=-1)) + self.time_emb.weight[1]
        pair = torch.stack([t0, t1], dim=2).reshape(b * n, 2, -1)
        attn_out, _ = self.attn(pair, pair, pair)
        h = self.n1(pair + attn_out)
        h = self.n2(h + self.ff(h))
        pooled = torch.cat([h[:, 1], self.delta(delta.reshape(b * n, 1))], dim=-1)
        return self.out(pooled).reshape(b, n, -1)


class TableSwitch(nn.Module):
    """Tabular branch. A switch of 0 uses the first value only. A switch of 1 uses both time points."""

    def __init__(self, n_labs: int, n_ctx: int, d_model: int = 48, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.25):
        super().__init__()
        self.pair = TemporalPairEncoder(d_model, dropout=dropout)
        self.lab_type = nn.Embedding(n_labs, d_model)
        self.ctx = nn.Sequential(
            nn.Linear(n_ctx, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 1),
        )
        # Unused by the forward pass and the loss. Kept so the parameter count matches the paper.
        self.t1_head = nn.Linear(d_model, 1)

    def forward(self, x_first, x_second, m_first, m_second, delta, ctx, switched):
        alone = self.pair.val(torch.stack([x_second, m_second], dim=-1)) + self.lab_type.weight.unsqueeze(0)
        paired = self.pair(x_first, x_second, m_first, m_second, delta) + self.lab_type.weight.unsqueeze(0)
        # Each item chooses pairing or the first measurement alone.
        gate = switched.view(switched.size(0), x_second.size(1), 1)
        lab = gate * paired + (1.0 - gate) * alone
        ctx_tok = self.ctx(ctx).unsqueeze(1)
        cls = self.cls.expand(x_second.size(0), -1, -1)
        h = self.enc(torch.cat([cls, ctx_tok, lab], dim=1))
        return self.head(h[:, 0]).squeeze(-1), h[:, 0]


class ECGEncoder(nn.Module):
    """Shallow 1-D convolution. Fusion uses the 64-dimensional features. The classification head is not used in fusion."""

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(12, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, out_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(out_dim, 16), nn.ReLU(), nn.Dropout(0.4), nn.Linear(16, 1),
        )

    def forward(self, x):
        feat = self.net(x)
        return self.head(feat).squeeze(-1), feat


class MM_TALNet(nn.Module):
    def __init__(self, n_labs: int, n_ctx: int, d_model: int = 48, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.25):
        super().__init__()
        self.tab = TableSwitch(n_labs, n_ctx, d_model, n_heads, n_layers, dropout)
        self.ecg = ECGEncoder(out_dim=64)
        self.txt_proj = nn.Sequential(
            nn.Linear(768, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.ecg_proj = nn.Linear(64, d_model)
        self.mod_emb = nn.Embedding(3, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead=4, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.fuse = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x_first, x_second, m_first, m_second, delta, ctx, switched, ecg, has_ecg, txt, has_txt):
        tab_logit, tab_h = self.tab(x_first, x_second, m_first, m_second, delta, ctx, switched)
        b = tab_h.size(0)
        _, feat = self.ecg(ecg)
        feat = feat * has_ecg.view(b, 1).float()
        tok = torch.stack([
            tab_h + self.mod_emb.weight[0],
            self.txt_proj(txt) + self.mod_emb.weight[1],
            self.ecg_proj(feat) + self.mod_emb.weight[2],
        ], dim=1)
        pad = torch.stack([
            torch.zeros(b, dtype=torch.bool, device=tab_h.device),
            ~has_txt.view(b).bool(),
            ~has_ecg.view(b).bool(),
        ], dim=1)
        fused = self.fuse(tok, src_key_padding_mask=pad)
        # If both auxiliary modalities are absent, fusion still runs and the output is not the tabular logit. The residual starts at zero.
        return tab_logit + self.head(fused[:, 0]).squeeze(-1), tab_logit


def _zero_residual_head(d_model: int, dropout: float) -> nn.Sequential:
    """Residual head. Its last layer starts at zero, so it does not change the tabular logit at initialization."""
    head = nn.Sequential(
        nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 1),
    )
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


class MM_TALNetGateSmall(nn.Module):
    """
    GR-MM-TALNet. Encoders and three-token fusion match MM_TALNet. The shared residual is split into two terms.
    Final logit = tabular logit + has_text × text gate × text residual + has_ecg × ECG gate × ECG residual.
    Gates are not applied to tokens before fusion. A missing modality contributes zero.
    When both are missing, the output equals the tabular logit.
    """

    def __init__(self, n_labs: int, n_ctx: int, d_model: int = 48, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.25):
        super().__init__()
        self.tab = TableSwitch(n_labs, n_ctx, d_model, n_heads, n_layers, dropout)
        self.ecg = ECGEncoder(out_dim=64)
        self.txt_proj = nn.Sequential(
            nn.Linear(768, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.ecg_proj = nn.Linear(64, d_model)
        self.mod_emb = nn.Embedding(3, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead=4, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.fuse = nn.TransformerEncoder(layer, num_layers=1)
        self.head_txt = _zero_residual_head(d_model, dropout)
        self.head_ecg = _zero_residual_head(d_model, dropout)
        self.gate_txt = nn.Linear(d_model * 2, 1)
        self.gate_ecg = nn.Linear(d_model * 2, 1)
        nn.init.zeros_(self.gate_txt.weight)
        nn.init.zeros_(self.gate_txt.bias)
        nn.init.zeros_(self.gate_ecg.weight)
        nn.init.zeros_(self.gate_ecg.bias)

    def forward(self, x_first, x_second, m_first, m_second, delta, ctx, switched, ecg, has_ecg, txt, has_txt):
        tab_logit, tab_h = self.tab(x_first, x_second, m_first, m_second, delta, ctx, switched)
        b = tab_h.size(0)
        _, feat = self.ecg(ecg)
        feat = feat * has_ecg.view(b, 1).float()
        txt_h = self.txt_proj(txt)
        ecg_h = self.ecg_proj(feat)
        tok = torch.stack([
            tab_h + self.mod_emb.weight[0],
            txt_h + self.mod_emb.weight[1],
            ecg_h + self.mod_emb.weight[2],
        ], dim=1)
        pad = torch.stack([
            torch.zeros(b, dtype=torch.bool, device=tab_h.device),
            ~has_txt.view(b).bool(),
            ~has_ecg.view(b).bool(),
        ], dim=1)
        fused = self.fuse(tok, src_key_padding_mask=pad)
        g_txt = torch.sigmoid(self.gate_txt(torch.cat([tab_h, txt_h], dim=-1)).squeeze(-1))
        g_ecg = torch.sigmoid(self.gate_ecg(torch.cat([tab_h, ecg_h], dim=-1)).squeeze(-1))
        g_txt = g_txt * has_txt.view(b).float()
        g_ecg = g_ecg * has_ecg.view(b).float()
        r_txt = self.head_txt(fused[:, 1]).squeeze(-1)
        r_ecg = self.head_ecg(fused[:, 2]).squeeze(-1)
        fuse = tab_logit + g_txt * r_txt + g_ecg * r_ecg
        return fuse, tab_logit, g_txt, g_ecg


class MM_TALNetGate(nn.Module):
    """
    Same as MM_TALNetGateSmall, except the gates also multiply the text and ECG tokens before fusion.
    A missing modality has gate 0. When both are missing, the output equals the tabular logit.
    A negative text_gate_bias starts the text gate closer to closed.
    """

    def __init__(self, n_labs: int, n_ctx: int, d_model: int = 48, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.25, text_gate_bias: float = 0.0):
        super().__init__()
        self.tab = TableSwitch(n_labs, n_ctx, d_model, n_heads, n_layers, dropout)
        self.ecg = ECGEncoder(out_dim=64)
        self.txt_proj = nn.Sequential(
            nn.Linear(768, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.ecg_proj = nn.Linear(64, d_model)
        self.mod_emb = nn.Embedding(3, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead=4, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.fuse = nn.TransformerEncoder(layer, num_layers=1)
        self.head_txt = _zero_residual_head(d_model, dropout)
        self.head_ecg = _zero_residual_head(d_model, dropout)
        self.gate_txt = nn.Linear(d_model * 2, 1)
        self.gate_ecg = nn.Linear(d_model * 2, 1)
        nn.init.zeros_(self.gate_txt.weight)
        nn.init.constant_(self.gate_txt.bias, text_gate_bias)
        nn.init.zeros_(self.gate_ecg.weight)
        nn.init.zeros_(self.gate_ecg.bias)

    def forward(self, x_first, x_second, m_first, m_second, delta, ctx, switched, ecg, has_ecg, txt, has_txt):
        tab_logit, tab_h = self.tab(x_first, x_second, m_first, m_second, delta, ctx, switched)
        b = tab_h.size(0)
        _, feat = self.ecg(ecg)
        feat = feat * has_ecg.view(b, 1).float()
        txt_h = self.txt_proj(txt)
        ecg_h = self.ecg_proj(feat)
        # A missing modality has gate 0, so its residual is not added.
        g_txt = torch.sigmoid(self.gate_txt(torch.cat([tab_h, txt_h], dim=-1)).squeeze(-1))
        g_ecg = torch.sigmoid(self.gate_ecg(torch.cat([tab_h, ecg_h], dim=-1)).squeeze(-1))
        g_txt = g_txt * has_txt.view(b).float()
        g_ecg = g_ecg * has_ecg.view(b).float()
        tok = torch.stack([
            tab_h + self.mod_emb.weight[0],
            g_txt.unsqueeze(1) * (txt_h + self.mod_emb.weight[1]),
            g_ecg.unsqueeze(1) * (ecg_h + self.mod_emb.weight[2]),
        ], dim=1)
        pad = torch.stack([
            torch.zeros(b, dtype=torch.bool, device=tab_h.device),
            ~has_txt.view(b).bool(),
            ~has_ecg.view(b).bool(),
        ], dim=1)
        fused = self.fuse(tok, src_key_padding_mask=pad)
        r_txt = self.head_txt(fused[:, 1]).squeeze(-1)
        r_ecg = self.head_ecg(fused[:, 2]).squeeze(-1)
        fuse = tab_logit + g_txt * r_txt + g_ecg * r_ecg
        return fuse, tab_logit, g_txt, g_ecg
