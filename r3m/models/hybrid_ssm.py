"""R3M Hybrid SSM Base LM (Nemotron-H style, tiny)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


@dataclass
class R3MHybridSSMConfig:
    vocab_size: int = 50257
    max_seq_len: int = 1024
    d_model: int = 512
    n_layers: int = 10
    attention_layers: Tuple[int, ...] = (2, 7)
    n_heads: int = 8
    ssm_expand: int = 2
    ssm_conv_kernel: int = 3
    ffn_mult: int = 2
    dropout: float = 0.0
    layernorm_eps: float = 1e-5


class SSMCore(nn.Module):
    def __init__(self, d_model: int, *, expand: int, conv_kernel: int):
        super().__init__()
        self.d_model = int(d_model)
        self.expand = int(expand)
        self.d_inner = int(self.expand * self.d_model)
        k = int(conv_kernel)
        if k < 1:
            raise ValueError("conv_kernel must be >= 1")
        self.conv_kernel = int(k)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=k,
            padding=0,  # IMPORTANT: we do explicit left-padding to keep this conv strictly causal
            groups=self.d_inner,
            bias=True,
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_and_gate = self.in_proj(x)
        x_proj, gate = x_and_gate.chunk(2, dim=-1)
        x_proj = self.act(x_proj)
        # Causal depthwise conv: token t must not see tokens > t.
        # Input is [B,T,C] -> conv wants [B,C,T]. We left-pad time by (k-1) and do padding=0 in Conv1d.
        xt = x_proj.transpose(1, 2)
        if self.conv_kernel > 1:
            xt = F.pad(xt, (self.conv_kernel - 1, 0))
        xc = self.conv1d(xt).transpose(1, 2)
        xc = xc * self.act(gate)
        return self.out_proj(xc)


class AttentionCore(nn.Module):
    def __init__(self, d_model: int, *, n_heads: int, dropout: float):
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.head_dim = self.d_model // self.n_heads
        self.dropout = float(dropout)
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.out = nn.Linear(self.d_model, self.d_model, bias=False)

    def forward(self, x: torch.Tensor, *, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_mask = None
        if attention_mask is not None:
            keep = attention_mask.to(dtype=torch.bool)
            attn_mask = keep[:, None, None, :]

        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=(self.dropout if self.training and self.dropout > 0 else 0.0),
                is_causal=True,
            )
        else:
            scale = self.head_dim**-0.5
            scores = (q * scale) @ k.transpose(-2, -1)
            causal = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal, float("-inf"))
            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            if self.training and self.dropout > 0:
                probs = F.dropout(probs, p=self.dropout)
            y = probs @ v

        y = y.transpose(1, 2).contiguous().view(b, t, d)
        return self.out(y)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, *, mult: int, dropout: float):
        super().__init__()
        d_ff = int(mult * d_model)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.w1(x))
        x = self.dropout(x)
        return self.w2(x)


class HybridBlock(nn.Module):
    def __init__(self, cfg: R3MHybridSSMConfig, *, use_attention: bool):
        super().__init__()
        self.use_attention = bool(use_attention)
        self.norm1 = RMSNorm(cfg.d_model, eps=cfg.layernorm_eps)
        self.core = AttentionCore(cfg.d_model, n_heads=cfg.n_heads, dropout=cfg.dropout) if self.use_attention else SSMCore(cfg.d_model, expand=cfg.ssm_expand, conv_kernel=cfg.ssm_conv_kernel)
        self.drop1 = nn.Dropout(float(cfg.dropout))
        self.norm2 = RMSNorm(cfg.d_model, eps=cfg.layernorm_eps)
        self.ffn = FeedForward(cfg.d_model, mult=cfg.ffn_mult, dropout=cfg.dropout)
        self.drop2 = nn.Dropout(float(cfg.dropout))

    def forward(self, x: torch.Tensor, *, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.norm1(x)
        h = self.core(h, attention_mask=attention_mask) if self.use_attention else self.core(h)
        x = x + self.drop1(h)
        h2 = self.ffn(self.norm2(x))
        x = x + self.drop2(h2)
        return x


class R3MHybridSSMLM(nn.Module):
    def __init__(self, cfg: R3MHybridSSMConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(int(cfg.vocab_size), int(cfg.d_model))
        self.wpe = nn.Embedding(int(cfg.max_seq_len), int(cfg.d_model))
        self.drop = nn.Dropout(float(cfg.dropout))
        attn_set = set(int(i) for i in cfg.attention_layers)
        self.layers = nn.ModuleList([HybridBlock(cfg, use_attention=(i in attn_set)) for i in range(int(cfg.n_layers))])
        self.norm_f = RMSNorm(cfg.d_model, eps=cfg.layernorm_eps)
        self.lm_head = nn.Linear(int(cfg.d_model), int(cfg.vocab_size), bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        b, t = input_ids.shape
        if t > int(self.cfg.max_seq_len):
            raise ValueError(f"sequence length {t} exceeds max_seq_len {self.cfg.max_seq_len}")
        device = input_ids.device
        pos = torch.arange(t, device=device).unsqueeze(0).expand(b, -1)
        x = self.wte(input_ids) + self.wpe(pos)
        x = self.drop(x)
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return {"logits": logits, "loss": loss}

    def state_dict_with_config(self) -> Dict[str, Any]:
        return {"config": asdict(self.cfg), "model_state_dict": self.state_dict()}
