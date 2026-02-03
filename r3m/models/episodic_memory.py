from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class EpisodicMemory(nn.Module):
    """Differentiable soft read/write memory."""

    def __init__(self, d_model: int, slots: int):
        super().__init__()
        self.d_model = int(d_model)
        self.slots = int(slots)
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.write_logits = nn.Linear(self.d_model, self.slots, bias=True)
        self.write_k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.write_v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        nn.init.eye_(self.write_k_proj.weight)
        nn.init.eye_(self.write_v_proj.weight)

    def init_memory(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        dt = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        k = torch.zeros(batch_size, self.slots, self.d_model, device=device, dtype=dt)
        v = torch.zeros(batch_size, self.slots, self.d_model, device=device, dtype=dt)
        return k, v

    def read(self, s: torch.Tensor, mem_k: torch.Tensor, mem_v: torch.Tensor) -> torch.Tensor:
        work_dtype = (
            torch.float32
            if (s.dtype in (torch.float16, torch.bfloat16) or mem_k.dtype in (torch.float16, torch.bfloat16))
            else s.dtype
        )
        q = self.q_proj(s.to(dtype=work_dtype)).unsqueeze(1)  # [B,1,D]
        mk = mem_k.to(dtype=work_dtype)
        mv = mem_v.to(dtype=work_dtype)
        att = (q * mk).sum(dim=-1) / (self.d_model**0.5)  # [B,M]
        w = F.softmax(att, dim=-1).unsqueeze(-1)  # [B,M,1]
        out = (w * mv).sum(dim=1)  # [B,D]
        return out

    def write(
        self,
        s: torch.Tensor,
        write_vec: torch.Tensor,
        mem_k: torch.Tensor,
        mem_v: torch.Tensor,
        gate: torch.Tensor,
        *,
        topk: int = 0,
        straight_through: bool = True,
        return_info: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:
        w_logits = self.write_logits(s)  # [B,M]
        info: Dict[str, torch.Tensor] = {}

        if topk is not None and int(topk) > 0:
            k = int(topk)
            if k == 1:
                idx = torch.argmax(w_logits, dim=-1)  # [B]
                w_hard = F.one_hot(idx, num_classes=self.slots).to(dtype=w_logits.dtype)
                w_soft = F.softmax(w_logits, dim=-1)
                w = (w_hard - w_soft.detach() + w_soft) if straight_through else w_hard
                w = w.unsqueeze(-1)
                info["write_idx"] = idx.unsqueeze(-1)
                info["write_w"] = torch.ones_like(idx, dtype=w_logits.dtype).unsqueeze(-1)
            else:
                v, idx = torch.topk(w_logits, k=min(k, w_logits.size(-1)), dim=-1)
                w_sparse = F.softmax(v, dim=-1)
                w = torch.zeros_like(w_logits)
                w.scatter_(dim=-1, index=idx, src=w_sparse)
                w = w.unsqueeze(-1)
                info["write_idx"] = idx
                info["write_w"] = w_sparse
        else:
            w = F.softmax(w_logits, dim=-1).unsqueeze(-1)
            v, idx = torch.topk(w_logits, k=min(3, w_logits.size(-1)), dim=-1)
            info["write_idx"] = idx
            info["write_w"] = F.softmax(v, dim=-1)

        work_dtype = (
            torch.float32
            if (mem_k.dtype in (torch.float16, torch.bfloat16) or write_vec.dtype in (torch.float16, torch.bfloat16))
            else mem_k.dtype
        )
        mk = mem_k.to(dtype=work_dtype)
        mv = mem_v.to(dtype=work_dtype)
        g = gate.to(dtype=work_dtype)
        ww = w.to(dtype=work_dtype)

        wv = write_vec.to(dtype=work_dtype)
        write_k = self.write_k_proj(wv)
        write_v = self.write_v_proj(wv)

        mk = mk + g.unsqueeze(1) * (ww * (write_k.unsqueeze(1) - mk))
        mv = mv + g.unsqueeze(1) * (ww * (write_v.unsqueeze(1) - mv))

        mem_k2 = mk.to(dtype=mem_k.dtype)
        mem_v2 = mv.to(dtype=mem_v.dtype)
        write_strength = gate.squeeze(-1)

        if return_info:
            info["write_gate"] = gate.squeeze(-1)
            return mem_k2, mem_v2, write_strength, info
        return mem_k2, mem_v2, write_strength
