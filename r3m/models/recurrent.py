"""
R³M Fix-B: Recurrent / Causal-by-construction base model.

This model avoids the "future leakage" bug by updating an internal state sequentially:
  s_{t+1} = f(s_t, x_t, memory)
and predicting token_{t+1} only from s_{t+1}.

It implements:
- token-by-token recurrent state update (GRU-style)
- per-token "thinking loop" (K steps) that refines s_t with episodic memory reads/writes
- stable convex mixing between update streams (base vs memory)

MVP constraints:
- episodic memory is per sequence / forward pass (not persisted across batches)
- tools and consolidation are not included yet
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class R3MRecurrentConfig:
    vocab_size: int = 50257
    d_model: int = 512
    max_seq_len: int = 512
    dropout: float = 0.1

    # Recurrent core
    state_init: str = "zeros"  # zeros | learned

    # Thinking loop
    k_train: int = 4
    k_max: int = 32
    step_size_eta: float = 0.1

    # Adaptive halting (ACT-style)
    adaptive_halting: bool = False
    halting_epsilon: float = 1e-2  # stop when remaining mass < eps
    ponder_lambda: float = 0.0  # add ponder cost to loss in training script if desired

    # Episodic memory
    episodic_enabled: bool = True
    episodic_slots: int = 128

    # Episodic write content
    # Old behavior wrote the current state directly into memory. This often collapses into
    # "store noise" or "ignore memory". Learned writes let the model choose what to store.
    episodic_learned_write: bool = True
    # Episodic stability knobs (important for AMP/bf16/fp16 training)
    episodic_fp32: bool = True  # do memory read/write math in fp32
    episodic_write_gate_max: float = 0.2  # cap per-step write strength
    episodic_write_vec_rmsnorm: bool = True  # normalize write vector before storing
    episodic_slot_rms_clip: float = 10.0  # clip per-slot RMS to avoid exploding keys/values
    episodic_detach_after_token: bool = True  # stop gradients through memory across token steps (stability)

    # Episodic write routing (avoid soft smear)
    write_topk: int = 0  # 0=soft write to all slots; 1=top-1; k>1=top-k
    write_straight_through: bool = True

    # Optional persistent episodic memory across generate() calls (session memory)
    persistent_gen_memory: bool = False
    persistent_decay: float = 0.0  # 0=overwrite with latest, 0.99=slow update

    # MoE (optional) + mHC-like Sinkhorn routing option
    moe_enabled: bool = False
    moe_num_experts: int = 4
    moe_router: str = "topk"  # topk | sinkhorn
    moe_top_k: int = 1
    moe_sinkhorn_iters: int = 8
    moe_temperature: float = 1.0

    # mHC-on-R3M (minimal): widen ONLY the thinking-loop residual stream into n substreams
    # and apply manifold-constrained (doubly-stochastic) stream mixing there.
    mhc_enabled: bool = False
    mhc_n: int = 4
    mhc_alpha_init: float = 0.01  # mix strength toward projected DS matrix
    mhc_sinkhorn_iters: int = 12
    mhc_temperature: float = 1.0

    # Stable mixing
    mix_temperature: float = 1.0
    # Optional regularization knob (applied in training scripts): encourage non-degenerate mixing
    mix_entropy_lambda: float = 0.0

    # Neuromodulator (gates writes)
    neuromodulator_enabled: bool = True

    # Debug: expose memory write decisions (slot indices/weights) for logging/inspection.
    debug_memory_write_trace: bool = False


class EpisodicMemory(nn.Module):
    """
    Per-forward episodic memory (differentiable soft write/read).
    """

    def __init__(self, d_model: int, slots: int):
        super().__init__()
        self.d_model = d_model
        self.slots = slots
        # Memory stores keys/values directly in model space [D].
        # We only project the query; keys/values are stored as-is to keep behavior simple/stable.
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.write_logits = nn.Linear(d_model, slots, bias=True)

        # Write projections (separate key/value to decouple retrieval vs content).
        self.write_k_proj = nn.Linear(d_model, d_model, bias=False)
        self.write_v_proj = nn.Linear(d_model, d_model, bias=False)
        if d_model == d_model:
            # start near identity so "learned write" can behave like old "write state" early on
            nn.init.eye_(self.write_k_proj.weight)
            nn.init.eye_(self.write_v_proj.weight)

    def init_memory(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        # Memory can be numerically sensitive; default to fp32 buffers even under AMP.
        dt = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        k = torch.zeros(batch_size, self.slots, self.d_model, device=device, dtype=dt)
        v = torch.zeros(batch_size, self.slots, self.d_model, device=device, dtype=dt)
        return k, v

    def read(self, s: torch.Tensor, mem_k: torch.Tensor, mem_v: torch.Tensor) -> torch.Tensor:
        # s: [B,D], mem_*: [B,M,D]
        # Do attention math in fp32 for stability
        work_dtype = torch.float32 if (s.dtype in (torch.float16, torch.bfloat16) or mem_k.dtype in (torch.float16, torch.bfloat16)) else s.dtype
        q = self.q_proj(s.to(dtype=work_dtype)).unsqueeze(1)  # [B,1,D]
        # keys/values are stored directly
        mk = mem_k.to(dtype=work_dtype)
        mv = mem_v.to(dtype=work_dtype)
        att = (q * mk).sum(dim=-1) / (self.d_model**0.5)  # [B,M]
        w = F.softmax(att, dim=-1).unsqueeze(-1)  # [B,M,1]
        out = (w * mv).sum(dim=1)  # [B,D]
        # Keep fp32 output for stability; caller can cast if desired.
        return out

    def write(
        self,
        s: torch.Tensor,
        write_vec: torch.Tensor,
        mem_k: torch.Tensor,
        mem_v: torch.Tensor,
        gate: torch.Tensor,
        topk: int = 0,
        straight_through: bool = True,
        return_info: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:
        # gate: [B,1]
        w_logits = self.write_logits(s)  # [B,M]
        info: Dict[str, torch.Tensor] = {}
        if topk is not None and int(topk) > 0:
            k = int(topk)
            if k == 1:
                # Straight-through hard one-hot routing to a single slot
                idx = torch.argmax(w_logits, dim=-1)  # [B]
                w_hard = F.one_hot(idx, num_classes=self.slots).to(dtype=w_logits.dtype)  # [B,M]
                w_soft = F.softmax(w_logits, dim=-1)  # [B,M]
                w = (w_hard - w_soft.detach() + w_soft) if straight_through else w_hard
                w = w.unsqueeze(-1)  # [B,M,1]
                info["write_idx"] = idx.unsqueeze(-1)  # [B,1]
                info["write_w"] = torch.ones_like(idx, dtype=w_logits.dtype).unsqueeze(-1)  # [B,1]
            else:
                # Top-k sparse softmax over selected slots
                v, idx = torch.topk(w_logits, k=min(k, w_logits.size(-1)), dim=-1)
                w_sparse = F.softmax(v, dim=-1)  # [B,k]
                w = torch.zeros_like(w_logits)
                w.scatter_(dim=-1, index=idx, src=w_sparse)
                w = w.unsqueeze(-1)  # [B,M,1]
                info["write_idx"] = idx  # [B,k]
                info["write_w"] = w_sparse  # [B,k]
        else:
            w = F.softmax(w_logits, dim=-1).unsqueeze(-1)  # [B,M,1]
            # For debug/logging: report top-3 indices/weights even for soft write
            v, idx = torch.topk(w_logits, k=min(3, w_logits.size(-1)), dim=-1)  # [B,k]
            info["write_idx"] = idx  # [B,k]
            info["write_w"] = F.softmax(v, dim=-1)  # [B,k] (approx, over top-k)
        # Do write math in fp32 for stability
        work_dtype = torch.float32 if (mem_k.dtype in (torch.float16, torch.bfloat16) or write_vec.dtype in (torch.float16, torch.bfloat16)) else mem_k.dtype
        mk = mem_k.to(dtype=work_dtype)
        mv = mem_v.to(dtype=work_dtype)
        g = gate.to(dtype=work_dtype)
        ww = w.to(dtype=work_dtype)

        # Project write content into key/value spaces.
        wv = write_vec.to(dtype=work_dtype)
        write_k = self.write_k_proj(wv)  # [B,D]
        write_v = self.write_v_proj(wv)  # [B,D]

        # Update selected slots toward write content (EMA-style).
        mk = mk + g.unsqueeze(1) * (ww * (write_k.unsqueeze(1) - mk))
        mv = mv + g.unsqueeze(1) * (ww * (write_v.unsqueeze(1) - mv))

        mem_k = mk.to(dtype=mem_k.dtype)
        mem_v = mv.to(dtype=mem_v.dtype)
        write_strength = gate.squeeze(-1)  # [B]
        if return_info:
            info["write_gate"] = gate.squeeze(-1)  # [B]
            return mem_k, mem_v, write_strength, info
        return mem_k, mem_v, write_strength


def sinkhorn_knopp(
    logits: torch.Tensor,
    n_iters: int = 8,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Sinkhorn-Knopp normalization for balanced routing.

    Given logits [B,E], returns P [B,E] with rows ~1 and columns ~B/E.
    This is "mHC-like" in the sense that we constrain routing onto a (scaled)
    doubly-stochastic manifold using Sinkhorn iterations.
    """
    if logits.dim() != 2:
        raise ValueError("sinkhorn_knopp expects logits [B,E]")
    # Important for AMP stability: do Sinkhorn normalization in fp32 (exp / normalization can overflow in fp16).
    b, e = logits.shape
    work_dtype = torch.float32 if logits.dtype in (torch.float16, torch.bfloat16) else logits.dtype
    z = logits.to(dtype=work_dtype)
    x = torch.exp(z - z.max(dim=-1, keepdim=True).values) + float(eps)
    col_target = torch.full((e,), float(b) / float(e), device=logits.device, dtype=work_dtype)

    for _ in range(max(1, int(n_iters))):
        x = x / (x.sum(dim=-1, keepdim=True) + eps)
        col = x.sum(dim=0) + eps
        x = x * (col_target / col).unsqueeze(0)
    x = x / (x.sum(dim=-1, keepdim=True) + eps)
    return x.to(dtype=logits.dtype)


def sinkhorn_birkhoff(
    logits: torch.Tensor,
    n_iters: int = 12,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Sinkhorn-Knopp for square matrices -> approximate doubly-stochastic matrix.

    logits: [n,n] -> returns P: [n,n] with rows/cols ~1.
    """
    if logits.dim() != 2 or logits.size(0) != logits.size(1):
        raise ValueError("sinkhorn_birkhoff expects square logits [n,n]")
    # Important for AMP stability: do Sinkhorn normalization in fp32 (exp / normalization can overflow in fp16).
    work_dtype = torch.float32 if logits.dtype in (torch.float16, torch.bfloat16) else logits.dtype
    z = logits.to(dtype=work_dtype)
    x = torch.exp(z - z.max()) + float(eps)
    for _ in range(max(1, int(n_iters))):
        x = x / (x.sum(dim=-1, keepdim=True) + eps)  # rows
        x = x / (x.sum(dim=0, keepdim=True) + eps)  # cols
    x = x / (x.sum(dim=-1, keepdim=True) + eps)
    return x.to(dtype=logits.dtype)


class MoEFeedForward(nn.Module):
    """
    Small MoE FFN for state vectors.

    We keep this intentionally simple: small E (4-16), compute experts densely,
    and use either top-k or Sinkhorn-balanced routing.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        router: str = "topk",
        top_k: int = 1,
        sinkhorn_iters: int = 8,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.num_experts = int(num_experts)
        self.router = str(router)
        self.top_k = int(top_k)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.temperature = float(temperature)

        self.router_head = nn.Linear(self.d_model, self.num_experts, bias=True)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.d_model, 4 * self.d_model, bias=False),
                    nn.GELU(),
                    nn.Linear(4 * self.d_model, self.d_model, bias=False),
                )
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # s: [B,D]
        logits = self.router_head(s) / max(self.temperature, 1e-6)  # [B,E]
        if self.router == "sinkhorn":
            p = sinkhorn_knopp(logits, n_iters=self.sinkhorn_iters)  # [B,E]
        else:
            p = F.softmax(logits, dim=-1)

        # sparsify to top_k for efficiency / stability
        if 0 < self.top_k < self.num_experts:
            v, idx = torch.topk(p, k=self.top_k, dim=-1)
            p_sparse = torch.zeros_like(p)
            p_sparse.scatter_(dim=-1, index=idx, src=v)
            p = p_sparse / (p_sparse.sum(dim=-1, keepdim=True) + 1e-8)

        # dense expert compute (fine for small E)
        expert_outs = torch.stack([ex(s) for ex in self.experts], dim=1)  # [B,E,D]
        y = torch.sum(p.unsqueeze(-1) * expert_outs, dim=1)  # [B,D]

        usage = p.mean(dim=0)  # [E]
        moe_load_balance = self.num_experts * torch.sum(usage * usage)  # min at uniform
        moe_entropy = (-torch.sum(p * torch.log(p + 1e-8), dim=-1)).mean()
        return y, {"moe_load_balance": moe_load_balance, "moe_entropy": moe_entropy, "moe_usage": usage}


class R3MRecurrentCore(nn.Module):
    """
    Processes one token step AFTER token assimilation:
      - thinking loop: K steps refine s with memory read/write and convex mixing
    """

    def __init__(self, d_model: int, episodic_slots: int, mix_temperature: float):
        super().__init__()
        self.d_model = d_model
        self.mix_temperature = mix_temperature

        self.mem = EpisodicMemory(d_model=d_model, slots=episodic_slots)

        # stream updates (operate on state, not sequence)
        self.base_mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
        )
        self.moe_ff: Optional[MoEFeedForward] = None
        self.mem_mlp = nn.Sequential(
            nn.Linear(2 * d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
        )

        # Learned episodic write content (what to store), separate from state update.
        self.write_content = nn.Sequential(
            nn.Linear(2 * d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
        )

        # convex mix weights for [base, mem]
        self.mix_head = nn.Linear(2 * d_model, 2, bias=True)

        # neuromodulator and write gate
        self.neuromod = nn.Linear(2 * d_model, 1, bias=True)
        self.write_gate = nn.Linear(d_model + 1, 1, bias=True)
        self.halt_head = nn.Linear(2 * d_model, 1, bias=True)

        self.post_ln = nn.LayerNorm(d_model)

        # mHC-on-R3M (minimal): parameters for widened thinking-loop stream mixing
        self.mhc_res_logits: Optional[nn.Parameter] = None
        self.mhc_pre_logits: Optional[nn.Parameter] = None
        self.mhc_post_logits: Optional[nn.Parameter] = None
        self.mhc_alpha_logit: Optional[nn.Parameter] = None

    def _mhc_init_if_needed(self, n: int, alpha_init: float) -> None:
        if self.mhc_res_logits is not None:
            return
        n = int(n)
        if n < 2:
            return
        # Ensure params are created on the same device/dtype as the module (important if created lazily during forward).
        ref = self.post_ln.weight
        dev = ref.device
        dt = ref.dtype
        # Initialize near-identity: H_res ≈ I by setting alpha small.
        alpha_init = float(alpha_init)
        alpha_init = max(1e-6, min(alpha_init, 1.0 - 1e-6))
        a = torch.tensor(alpha_init, device=dev, dtype=dt)
        self.mhc_alpha_logit = nn.Parameter(torch.log(a / (1.0 - a)))

        # logits for DS projection; start near zeros so Sinkhorn gives ~uniform
        self.mhc_res_logits = nn.Parameter(torch.zeros(n, n, device=dev, dtype=dt))
        # pre/post as simplex vectors over streams
        self.mhc_pre_logits = nn.Parameter(torch.zeros(n, device=dev, dtype=dt))
        self.mhc_post_logits = nn.Parameter(torch.zeros(n, device=dev, dtype=dt))

    def _mhc_mats(self, n: int, iters: int, temperature: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (H_res, h_pre, h_post, alpha)
        - H_res: [n,n] near-identity DS mix: (1-alpha)I + alpha*P
        - h_pre: [n] simplex (compress streams -> state)
        - h_post: [n] simplex (expand delta -> streams)
        """
        assert self.mhc_res_logits is not None
        assert self.mhc_pre_logits is not None
        assert self.mhc_post_logits is not None
        assert self.mhc_alpha_logit is not None

        n = int(n)
        alpha = torch.sigmoid(self.mhc_alpha_logit)  # scalar
        P = sinkhorn_birkhoff(self.mhc_res_logits / max(float(temperature), 1e-6), n_iters=int(iters))
        I = torch.eye(n, device=P.device, dtype=P.dtype)
        H_res = (1.0 - alpha) * I + alpha * P
        h_pre = F.softmax(self.mhc_pre_logits, dim=-1)  # [n]
        h_post = F.softmax(self.mhc_post_logits, dim=-1)  # [n]
        return H_res, h_pre, h_post, alpha

    def init_memory(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.mem.init_memory(batch_size, device=device, dtype=dtype)

    def step(
        self,
        s: torch.Tensor,  # [B,D] (already assimilated for this token)
        mem_k: torch.Tensor,
        mem_v: torch.Tensor,
        k_steps: int,
        eta: float,
        episodic_enabled: bool,
        neuromod_enabled: bool,
        adaptive_halting: bool,
        halting_epsilon: float,
        write_topk: int,
        write_straight_through: bool,
        moe_enabled: bool,
        moe_num_experts: int,
        moe_router: str,
        moe_top_k: int,
        moe_sinkhorn_iters: int,
        moe_temperature: float,
        mhc_enabled: bool,
        mhc_n: int,
        mhc_alpha_init: float,
        mhc_sinkhorn_iters: int,
        mhc_temperature: float,
        keep_mask: Optional[torch.Tensor] = None,  # [B] bool; if provided, do not update state/memory when False
        episodic_fp32: bool = True,
        episodic_write_gate_max: float = 0.2,
        episodic_write_vec_rmsnorm: bool = True,
        episodic_slot_rms_clip: float = 10.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        s_in = s
        # If requested, force fp32 math for the entire thinking/memory update.
        # This avoids bf16/fp16 overflow in gates / memory interactions during AMP training.
        use_fp32 = bool(episodic_fp32) and s.is_cuda
        if use_fp32:
            s = s.to(torch.float32)
            mem_k = mem_k.to(torch.float32)
            mem_v = mem_v.to(torch.float32)
        if keep_mask is not None:
            active = keep_mask.to(dtype=s.dtype).unsqueeze(-1)  # [B,1]
        else:
            active = None

        write_strengths = []
        g_stats = []
        ponder_costs = []
        moe_lb_stats = []
        moe_ent_stats = []
        mix_ent_stats = []
        alpha_mhc: Optional[torch.Tensor] = None
        last_write_info: Optional[Dict[str, torch.Tensor]] = None

        # mHC stream setup (minimal): initialize stream ONCE per token-step and keep it through micro-steps
        use_mhc = bool(mhc_enabled) and int(mhc_n) >= 2
        if use_mhc:
            self._mhc_init_if_needed(int(mhc_n), float(mhc_alpha_init))
            H_res, h_pre, h_post, alpha_mhc = self._mhc_mats(
                int(mhc_n), int(mhc_sinkhorn_iters), float(mhc_temperature)
            )
            # Start from the assimilated state and widen into n parallel substreams.
            S = s.unsqueeze(1).expand(-1, int(mhc_n), -1).contiguous()  # [B,n,D]
        else:
            H_res = None
            h_pre = None
            h_post = None
            S = None

        # 2) thinking loop (optionally adaptive halting)
        if adaptive_halting:
            # ACT-style halting: build a convex combination of intermediate states.
            bsz = s.size(0)
            remainder = torch.ones(bsz, 1, device=s.device, dtype=s.dtype)  # [B,1]
            s_acc = torch.zeros_like(s)  # [B,D]
            steps_used = torch.zeros(bsz, device=s.device, dtype=s.dtype)  # [B]

            max_steps = max(1, int(k_steps))
            eps = max(float(halting_epsilon), 1e-6)

            for _ in range(max_steps):
                # Effective state for memory read
                if S is not None:
                    assert h_pre is not None
                    s_eff = torch.einsum("n,bnd->bd", h_pre, S)  # [B,D]
                    m_e = self.mem.read(s_eff, mem_k, mem_v) if episodic_enabled else torch.zeros_like(s_eff)
                else:
                    s_eff = s
                    m_e = self.mem.read(s, mem_k, mem_v) if episodic_enabled else torch.zeros_like(s)

                if neuromod_enabled:
                    g = torch.sigmoid(self.neuromod(torch.cat([s_eff, m_e], dim=-1)))  # [B,1]
                else:
                    g = torch.ones(s.size(0), 1, device=s.device, dtype=s.dtype)
                if active is not None:
                    g = g * active

                # update candidates
                base_in = s_eff
                if moe_enabled:
                    if (
                        self.moe_ff is None
                        or self.moe_ff.num_experts != int(moe_num_experts)
                        or self.moe_ff.router != str(moe_router)
                        or self.moe_ff.top_k != int(moe_top_k)
                    ):
                        self.moe_ff = MoEFeedForward(
                            d_model=self.d_model,
                            num_experts=int(moe_num_experts),
                            router=str(moe_router),
                            top_k=int(moe_top_k),
                            sinkhorn_iters=int(moe_sinkhorn_iters),
                            temperature=float(moe_temperature),
                        ).to(s.device)
                    d_base, moe_aux = self.moe_ff(base_in)
                    moe_lb_stats.append(moe_aux["moe_load_balance"])
                    moe_ent_stats.append(moe_aux["moe_entropy"])
                else:
                    d_base = self.base_mlp(base_in)  # [B,D]
                d_mem = self.mem_mlp(torch.cat([base_in, m_e], dim=-1))  # [B,D]

                # convex mixing
                mix_logits = self.mix_head(torch.cat([base_in, m_e], dim=-1)) / max(self.mix_temperature, 1e-6)
                alpha = F.softmax(mix_logits, dim=-1)  # [B,2]
                mix_ent_stats.append((-torch.sum(alpha * torch.log(alpha + 1e-8), dim=-1)).mean())
                delta = alpha[:, 0:1] * d_base + alpha[:, 1:2] * d_mem

                # apply update
                if S is not None:
                    assert H_res is not None and h_post is not None and h_pre is not None
                    S = torch.einsum("ij,bjd->bid", H_res, S)
                    S = S + (eta * delta).unsqueeze(1) * h_post.view(1, -1, 1)
                    s = self.post_ln(torch.einsum("n,bnd->bd", h_pre, S))
                else:
                    s = self.post_ln(s + eta * delta)
                if active is not None:
                    s = torch.where(keep_mask.unsqueeze(-1), s, s_in)

                # write episodic memory
                if episodic_enabled:
                    w_gate = torch.sigmoid(self.write_gate(torch.cat([s, g], dim=-1)))  # [B,1]
                    if active is not None:
                        w_gate = w_gate * active
                    w_gate = torch.clamp(w_gate, 0.0, float(episodic_write_gate_max))
                    write_vec = self.write_content(torch.cat([s_eff, m_e], dim=-1)) if bool(getattr(self, "episodic_learned_write", False)) else s
                    if episodic_write_vec_rmsnorm:
                        rms = write_vec.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6)
                        write_vec = write_vec / rms
                    if return_debug := bool(getattr(self, "debug_memory_write_trace", False)):
                        mem_k, mem_v, w_strength, last_write_info = self.mem.write(
                            s,
                            write_vec,
                            mem_k,
                            mem_v,
                            gate=w_gate,
                            topk=write_topk,
                            straight_through=write_straight_through,
                            return_info=True,
                        )
                    else:
                        mem_k, mem_v, w_strength = self.mem.write(
                            s,
                            write_vec,
                            mem_k,
                            mem_v,
                            gate=w_gate,
                            topk=write_topk,
                            straight_through=write_straight_through,
                        )
                    if episodic_slot_rms_clip and float(episodic_slot_rms_clip) > 0.0:
                        clip = float(episodic_slot_rms_clip)
                        mk_rms = mem_k.to(torch.float32).pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6)
                        mv_rms = mem_v.to(torch.float32).pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6)
                        mem_k = (mem_k.to(torch.float32) * torch.clamp(clip / mk_rms, max=1.0)).to(mem_k.dtype)
                        mem_v = (mem_v.to(torch.float32) * torch.clamp(clip / mv_rms, max=1.0)).to(mem_v.dtype)
                    write_strengths.append(w_strength)
                else:
                    write_strengths.append(torch.zeros(s.size(0), device=s.device, dtype=s.dtype))

                g_stats.append(g.squeeze(-1))

                # halting probability from current state + memory read
                p_halt = torch.sigmoid(self.halt_head(torch.cat([s, m_e], dim=-1)))  # [B,1]
                p_eff = torch.minimum(p_halt, remainder)  # [B,1]
                s_acc = s_acc + p_eff * s
                remainder = remainder - p_eff
                steps_used = steps_used + (remainder.squeeze(-1) > eps).to(dtype=s.dtype)
                if bool((remainder < eps).all().item()):
                    break

            # whatever remains gets last state
            s = s_acc + remainder * s
            ponder_costs.append(steps_used)
        else:
            for _ in range(k_steps):
                if S is not None:
                    assert h_pre is not None
                    s_eff = torch.einsum("n,bnd->bd", h_pre, S)  # [B,D]
                    m_e = self.mem.read(s_eff, mem_k, mem_v) if episodic_enabled else torch.zeros_like(s_eff)
                else:
                    s_eff = s
                    m_e = self.mem.read(s, mem_k, mem_v) if episodic_enabled else torch.zeros_like(s)

                if neuromod_enabled:
                    g = torch.sigmoid(self.neuromod(torch.cat([s_eff, m_e], dim=-1)))  # [B,1]
                else:
                    g = torch.ones(s.size(0), 1, device=s.device, dtype=s.dtype)
                if active is not None:
                    g = g * active

                base_in = s_eff
                if moe_enabled:
                    if (
                        self.moe_ff is None
                        or self.moe_ff.num_experts != int(moe_num_experts)
                        or self.moe_ff.router != str(moe_router)
                        or self.moe_ff.top_k != int(moe_top_k)
                    ):
                        self.moe_ff = MoEFeedForward(
                            d_model=self.d_model,
                            num_experts=int(moe_num_experts),
                            router=str(moe_router),
                            top_k=int(moe_top_k),
                            sinkhorn_iters=int(moe_sinkhorn_iters),
                            temperature=float(moe_temperature),
                        ).to(s.device)
                    d_base, moe_aux = self.moe_ff(base_in)
                    moe_lb_stats.append(moe_aux["moe_load_balance"])
                    moe_ent_stats.append(moe_aux["moe_entropy"])
                else:
                    d_base = self.base_mlp(base_in)
                d_mem = self.mem_mlp(torch.cat([base_in, m_e], dim=-1))

                mix_logits = self.mix_head(torch.cat([base_in, m_e], dim=-1)) / max(self.mix_temperature, 1e-6)
                alpha = F.softmax(mix_logits, dim=-1)
                mix_ent_stats.append((-torch.sum(alpha * torch.log(alpha + 1e-8), dim=-1)).mean())
                delta = alpha[:, 0:1] * d_base + alpha[:, 1:2] * d_mem

                if S is not None:
                    assert H_res is not None and h_post is not None and h_pre is not None
                    S = torch.einsum("ij,bjd->bid", H_res, S)
                    S = S + (eta * delta).unsqueeze(1) * h_post.view(1, -1, 1)
                    s = self.post_ln(torch.einsum("n,bnd->bd", h_pre, S))
                else:
                    s = self.post_ln(s + eta * delta)
                if active is not None:
                    s = torch.where(keep_mask.unsqueeze(-1), s, s_in)

                if episodic_enabled:
                    w_gate = torch.sigmoid(self.write_gate(torch.cat([s, g], dim=-1)))
                    if active is not None:
                        w_gate = w_gate * active
                    w_gate = torch.clamp(w_gate, 0.0, float(episodic_write_gate_max))
                    write_vec = self.write_content(torch.cat([s_eff, m_e], dim=-1)) if bool(getattr(self, "episodic_learned_write", False)) else s
                    if episodic_write_vec_rmsnorm:
                        rms = write_vec.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6)
                        write_vec = write_vec / rms
                    if return_debug := bool(getattr(self, "debug_memory_write_trace", False)):
                        mem_k, mem_v, w_strength, last_write_info = self.mem.write(
                            s, write_vec, mem_k, mem_v, gate=w_gate, topk=write_topk, straight_through=write_straight_through, return_info=True
                        )
                    else:
                        mem_k, mem_v, w_strength = self.mem.write(
                            s, write_vec, mem_k, mem_v, gate=w_gate, topk=write_topk, straight_through=write_straight_through
                        )
                    if episodic_slot_rms_clip and float(episodic_slot_rms_clip) > 0.0:
                        clip = float(episodic_slot_rms_clip)
                        mk_rms = mem_k.to(torch.float32).pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6)
                        mv_rms = mem_v.to(torch.float32).pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6)
                        mem_k = (mem_k.to(torch.float32) * torch.clamp(clip / mk_rms, max=1.0)).to(mem_k.dtype)
                        mem_v = (mem_v.to(torch.float32) * torch.clamp(clip / mv_rms, max=1.0)).to(mem_v.dtype)
                    write_strengths.append(w_strength)
                else:
                    write_strengths.append(torch.zeros(s.size(0), device=s.device, dtype=s.dtype))

                g_stats.append(g.squeeze(-1))

        aux = {
            "avg_write_strength": torch.stack(write_strengths, dim=0).mean(dim=0) if write_strengths else torch.zeros(s.size(0), device=s.device, dtype=s.dtype),
            "avg_neuromod": torch.stack(g_stats, dim=0).mean(dim=0) if g_stats else torch.zeros(s.size(0), device=s.device, dtype=s.dtype),
            "ponder_cost": torch.stack(ponder_costs, dim=0).mean(dim=0) if ponder_costs else torch.zeros(s.size(0), device=s.device, dtype=s.dtype),
            "moe_load_balance": torch.stack(moe_lb_stats, dim=0).mean() if moe_lb_stats else torch.tensor(0.0, device=s.device, dtype=s.dtype),
            "moe_entropy": torch.stack(moe_ent_stats, dim=0).mean() if moe_ent_stats else torch.tensor(0.0, device=s.device, dtype=s.dtype),
            "mix_entropy": torch.stack(mix_ent_stats, dim=0).mean() if mix_ent_stats else torch.tensor(0.0, device=s.device, dtype=s.dtype),
            "mhc_alpha": (alpha_mhc.detach() if alpha_mhc is not None else torch.tensor(0.0, device=s.device, dtype=s.dtype)),
        }
        # Debug: last write decision (slot indices/weights + gate). Returned for logging/inspection.
        if last_write_info is not None:
            aux["write_gate"] = last_write_info.get("write_gate", torch.zeros(s.size(0), device=s.device, dtype=s.dtype))
            aux["write_idx"] = last_write_info.get("write_idx", torch.zeros(s.size(0), 1, device=s.device, dtype=torch.long))
            aux["write_w"] = last_write_info.get("write_w", torch.zeros(s.size(0), 1, device=s.device, dtype=s.dtype))
        # Ensure inactive positions do not update state
        if keep_mask is not None:
            s = torch.where(keep_mask.unsqueeze(-1), s, s_in)
        # If we forced fp32 internally, keep state/memory in fp32 (caller can cast if desired).
        return s, mem_k, mem_v, aux


class R3MRecurrentLM(nn.Module):
    """
    Causal LM via recurrent state.

At token position t:
  - consume token x_t
  - update state s_{t+1}
  - produce logits for token_{t+1} from s_{t+1}
    """

    def __init__(self, cfg: R3MRecurrentConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        # Token assimilation is done with a fused GRU over the whole sequence to avoid
        # a Python per-token GRUCell loop. Thinking/memory is still sequential.
        self.in_ln = nn.LayerNorm(cfg.d_model)
        self.gru = nn.GRU(input_size=cfg.d_model, hidden_size=cfg.d_model, batch_first=True)

        if cfg.state_init == "learned":
            self.s0 = nn.Parameter(torch.zeros(cfg.d_model))
        else:
            self.s0 = None

        self.core = R3MRecurrentCore(
            d_model=cfg.d_model,
            episodic_slots=cfg.episodic_slots,
            mix_temperature=cfg.mix_temperature,
        )
        # allow core to pick learned vs state writes
        self.core.episodic_learned_write = bool(getattr(cfg, "episodic_learned_write", True))
        # Debug: let core know if we want to expose memory write traces.
        self.core.debug_memory_write_trace = bool(getattr(cfg, "debug_memory_write_trace", False))
        # Important: mHC params are created lazily inside the core. If we save a checkpoint after
        # training with mHC enabled, the state_dict will contain these params. To ensure checkpoints
        # load cleanly (strict=True) we must register them up-front when mHC is enabled.
        if bool(cfg.mhc_enabled) and int(cfg.mhc_n) >= 2:
            self.core._mhc_init_if_needed(int(cfg.mhc_n), float(cfg.mhc_alpha_init))

        self.out_ln = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # weight tying
        self.lm_head.weight = self.tok_emb.weight

        # Persistent generation memory/state (optional, session-like)
        self._gen_state: Optional[torch.Tensor] = None
        self._gen_mem_k: Optional[torch.Tensor] = None
        self._gen_mem_v: Optional[torch.Tensor] = None

        # Optional: compile a *single-token* step (core thinking+memory) and call it inside the token loop.
        # Compiling the whole sequence forward can be very slow (big unrolled graphs). This keeps compile graphs small.
        self._compiled_token_step: Optional[Callable[..., Tuple[torch.Tensor, ...]]] = None

    def reset_generation_memory(self) -> None:
        """Reset cached state/memory used when cfg.persistent_gen_memory=True."""
        self._gen_state = None
        self._gen_mem_k = None
        self._gen_mem_v = None

    def compile_token_step(self, *, backend: str = "inductor", mode: str = "reduce-overhead") -> None:
        """
        Compile a single-token step for faster training/inference loops.

        This avoids compiling the full sequence loop (which can explode compile time due to unrolling).
        The fast path is used only when `episodic_enabled_override` is not set and debug traces are off.
        """
        if not hasattr(torch, "compile"):
            raise RuntimeError("compile_token_step requires PyTorch 2.x (torch.compile)")

        cfg = self.cfg

        def _token_step(
            s_assim: torch.Tensor,
            mem_k: torch.Tensor,
            mem_v: torch.Tensor,
            k_eff: int,
        ) -> Tuple[torch.Tensor, ...]:
            s_new, mem_k2, mem_v2, aux = self.core.step(
                s=s_assim,
                mem_k=mem_k,
                mem_v=mem_v,
                k_steps=k_eff,
                eta=cfg.step_size_eta,
                episodic_enabled=bool(cfg.episodic_enabled),
                neuromod_enabled=bool(cfg.neuromodulator_enabled),
                adaptive_halting=bool(cfg.adaptive_halting),
                halting_epsilon=float(cfg.halting_epsilon),
                write_topk=int(cfg.write_topk),
                write_straight_through=bool(cfg.write_straight_through),
                moe_enabled=bool(cfg.moe_enabled),
                moe_num_experts=int(cfg.moe_num_experts),
                moe_router=str(cfg.moe_router),
                moe_top_k=int(cfg.moe_top_k),
                moe_sinkhorn_iters=int(cfg.moe_sinkhorn_iters),
                moe_temperature=float(cfg.moe_temperature),
                mhc_enabled=bool(cfg.mhc_enabled),
                mhc_n=int(cfg.mhc_n),
                mhc_alpha_init=float(cfg.mhc_alpha_init),
                mhc_sinkhorn_iters=int(cfg.mhc_sinkhorn_iters),
                mhc_temperature=float(cfg.mhc_temperature),
                keep_mask=None,
                episodic_fp32=bool(cfg.episodic_fp32),
                episodic_write_gate_max=float(cfg.episodic_write_gate_max),
                episodic_write_vec_rmsnorm=bool(cfg.episodic_write_vec_rmsnorm),
                episodic_slot_rms_clip=float(cfg.episodic_slot_rms_clip),
            )

            # Return tensors only (avoid dict payload in compiled graph)
            avg_write = aux["avg_write_strength"]
            avg_neuromod = aux["avg_neuromod"]
            ponder = aux.get("ponder_cost", torch.zeros(s_new.size(0), device=s_new.device, dtype=s_new.dtype))
            moe_lb = aux.get("moe_load_balance", torch.tensor(0.0, device=s_new.device, dtype=s_new.dtype))
            moe_ent = aux.get("moe_entropy", torch.tensor(0.0, device=s_new.device, dtype=s_new.dtype))
            mix_ent = aux.get("mix_entropy", torch.tensor(0.0, device=s_new.device, dtype=s_new.dtype))
            mhc_alpha = aux.get("mhc_alpha", torch.tensor(0.0, device=s_new.device, dtype=s_new.dtype))
            return s_new, mem_k2, mem_v2, avg_write, avg_neuromod, ponder, moe_lb, moe_ent, mix_ent, mhc_alpha

        self._compiled_token_step = torch.compile(_token_step, backend=str(backend), mode=str(mode))  # type: ignore[attr-defined]

    def _forward_stateful(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        k_steps: Optional[int],
        s: torch.Tensor,
        mem_k: torch.Tensor,
        mem_v: torch.Tensor,
        episodic_enabled_override: Optional[bool] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Run a forward pass starting from provided state + memory.
        Returns:
          (logits, s_final, mem_k_final, mem_v_final, avg_write, avg_neuromod, ponder_cost, moe_load_balance, moe_entropy)
        """
        b, t = input_ids.shape
        device = input_ids.device

        pos = torch.arange(t, device=device).unsqueeze(0).expand(b, -1)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)
        x = self.in_ln(x)

        k_eff = int(k_steps) if k_steps is not None else int(self.cfg.k_train)
        k_eff = max(1, min(k_eff, self.cfg.k_max))

        # Assimilate tokens with fused GRU (optionally packed to skip padding-at-end).
        h0 = s.unsqueeze(0)  # [1,B,D]
        if attention_mask is not None:
            # Assume padding is at the end (standard tokenizer padding).
            lengths = attention_mask.to(dtype=torch.long).sum(dim=1).clamp(min=0, max=t).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths=lengths, batch_first=True, enforce_sorted=False)
            packed_out, h_last = self.gru(packed, h0)
            s_seq, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=t)  # [B,T,D]
        else:
            s_seq, h_last = self.gru(x, h0)  # [B,T,D]

        logits_seq = []
        write_stats = []
        g_stats = []
        ponder_stats = []
        moe_lb_stats: List[torch.Tensor] = []
        moe_ent_stats: List[torch.Tensor] = []
        mhc_alpha_stats: List[torch.Tensor] = []
        mix_ent_stats: List[torch.Tensor] = []
        write_gate_stats: List[torch.Tensor] = []
        write_idx_stats: List[torch.Tensor] = []
        write_w_stats: List[torch.Tensor] = []

        for i in range(t):
            if attention_mask is not None:
                keep = attention_mask[:, i].to(dtype=torch.bool)  # [B]
            else:
                keep = None

            episodic_enabled_eff = self.cfg.episodic_enabled if episodic_enabled_override is None else bool(episodic_enabled_override)
            # Start from assimilated state for this token; keep inactive positions unchanged.
            s_assim = s_seq[:, i, :]  # [B,D]
            if keep is not None:
                s_assim = torch.where(keep.unsqueeze(-1), s_assim, s)

            # Fast path: compiled single-token step (only when episodic override is not used and debug traces are off).
            if (
                self._compiled_token_step is not None
                and episodic_enabled_override is None
                and (not bool(getattr(self.cfg, "debug_memory_write_trace", False)))
                and bool(episodic_enabled_eff) == bool(self.cfg.episodic_enabled)
            ):
                (
                    s_new,
                    mem_k,
                    mem_v,
                    avg_write_t,
                    avg_neuromod_t,
                    ponder_t,
                    moe_lb_t,
                    moe_ent_t,
                    mix_ent_t,
                    mhc_alpha_t,
                ) = self._compiled_token_step(s_assim, mem_k, mem_v, k_eff)
                aux = {
                    "avg_write_strength": avg_write_t,
                    "avg_neuromod": avg_neuromod_t,
                    "ponder_cost": ponder_t,
                    "moe_load_balance": moe_lb_t,
                    "moe_entropy": moe_ent_t,
                    "mix_entropy": mix_ent_t,
                    "mhc_alpha": mhc_alpha_t,
                }
            else:
                s_new, mem_k, mem_v, aux = self.core.step(
                    s=s_assim,
                    mem_k=mem_k,
                    mem_v=mem_v,
                    k_steps=k_eff,
                    eta=self.cfg.step_size_eta,
                    episodic_enabled=episodic_enabled_eff,
                    neuromod_enabled=self.cfg.neuromodulator_enabled,
                    adaptive_halting=self.cfg.adaptive_halting,
                    halting_epsilon=self.cfg.halting_epsilon,
                    write_topk=self.cfg.write_topk,
                    write_straight_through=self.cfg.write_straight_through,
                    moe_enabled=self.cfg.moe_enabled,
                    moe_num_experts=self.cfg.moe_num_experts,
                    moe_router=self.cfg.moe_router,
                    moe_top_k=self.cfg.moe_top_k,
                    moe_sinkhorn_iters=self.cfg.moe_sinkhorn_iters,
                    moe_temperature=self.cfg.moe_temperature,
                    mhc_enabled=self.cfg.mhc_enabled,
                    mhc_n=self.cfg.mhc_n,
                    mhc_alpha_init=self.cfg.mhc_alpha_init,
                    mhc_sinkhorn_iters=self.cfg.mhc_sinkhorn_iters,
                    mhc_temperature=self.cfg.mhc_temperature,
                    keep_mask=keep,
                    episodic_fp32=bool(self.cfg.episodic_fp32),
                    episodic_write_gate_max=float(self.cfg.episodic_write_gate_max),
                    episodic_write_vec_rmsnorm=bool(self.cfg.episodic_write_vec_rmsnorm),
                    episodic_slot_rms_clip=float(self.cfg.episodic_slot_rms_clip),
                )

            if keep is not None:
                s = torch.where(keep.unsqueeze(-1), s_new, s)
            else:
                s = s_new

            # Stability: do not backprop through the memory state across token positions.
            if self.training and episodic_enabled_eff and bool(getattr(self.cfg, "episodic_detach_after_token", True)):
                mem_k = mem_k.detach()
                mem_v = mem_v.detach()

            logits_t = self.lm_head(self.out_ln(s))
            logits_seq.append(logits_t.unsqueeze(1))
            write_stats.append(aux["avg_write_strength"].unsqueeze(1))
            g_stats.append(aux["avg_neuromod"].unsqueeze(1))
            ponder_stats.append(aux.get("ponder_cost", torch.zeros(b, device=device, dtype=s.dtype)).unsqueeze(1))
            moe_lb_stats.append(aux.get("moe_load_balance", torch.tensor(0.0, device=device, dtype=s.dtype)).to(device=device, dtype=s.dtype))
            moe_ent_stats.append(aux.get("moe_entropy", torch.tensor(0.0, device=device, dtype=s.dtype)).to(device=device, dtype=s.dtype))
            mhc_alpha_stats.append(aux.get("mhc_alpha", torch.tensor(0.0, device=device, dtype=s.dtype)).to(device=device, dtype=s.dtype))
            mix_ent_stats.append(aux.get("mix_entropy", torch.tensor(0.0, device=device, dtype=s.dtype)).to(device=device, dtype=s.dtype))
            if bool(getattr(self.cfg, "debug_memory_write_trace", False)):
                if "write_gate" in aux and "write_idx" in aux and "write_w" in aux:
                    write_gate_stats.append(aux["write_gate"].to(device=device, dtype=s.dtype))  # [B]
                    write_idx_stats.append(aux["write_idx"].to(device=device))  # [B,k]
                    write_w_stats.append(aux["write_w"].to(device=device, dtype=s.dtype))  # [B,k]

        logits = torch.cat(logits_seq, dim=1)  # [B,T,V]
        avg_write = torch.cat(write_stats, dim=1).mean(dim=1)
        avg_neuromod = torch.cat(g_stats, dim=1).mean(dim=1)
        ponder_cost = torch.cat(ponder_stats, dim=1).mean(dim=1)
        moe_load_balance = torch.stack(moe_lb_stats, dim=0).mean() if moe_lb_stats else torch.tensor(0.0, device=device, dtype=s.dtype)
        moe_entropy = torch.stack(moe_ent_stats, dim=0).mean() if moe_ent_stats else torch.tensor(0.0, device=device, dtype=s.dtype)
        mhc_alpha = torch.stack(mhc_alpha_stats, dim=0).mean() if mhc_alpha_stats else torch.tensor(0.0, device=device, dtype=s.dtype)
        mix_entropy = torch.stack(mix_ent_stats, dim=0).mean() if mix_ent_stats else torch.tensor(0.0, device=device, dtype=s.dtype)
        # For debug: take the last token's write info (if present)
        if bool(getattr(self.cfg, "debug_memory_write_trace", False)) and write_gate_stats:
            last_write_gate = write_gate_stats[-1]  # [B]
            last_write_idx = write_idx_stats[-1]  # [B,k]
            last_write_w = write_w_stats[-1]  # [B,k]
        else:
            last_write_gate = torch.zeros(b, device=device, dtype=s.dtype)
            last_write_idx = torch.zeros(b, 1, device=device, dtype=torch.long)
            last_write_w = torch.zeros(b, 1, device=device, dtype=s.dtype)

        return (
            logits,
            s,
            mem_k,
            mem_v,
            avg_write,
            avg_neuromod,
            ponder_cost,
            moe_load_balance,
            moe_entropy,
            mix_entropy,
            mhc_alpha,
            last_write_gate,
            last_write_idx,
            last_write_w,
        )

    def _init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.s0 is None:
            return torch.zeros(batch_size, self.cfg.d_model, device=device, dtype=dtype)
        return self.s0.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1).contiguous()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        k_steps: Optional[int] = None,
        episodic_enabled_override: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        b, t = input_ids.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length {t} exceeds max_seq_len {self.cfg.max_seq_len}")

        device = input_ids.device
        dtype = torch.float32  # keep state stable in fp32; embeddings can be fp16/bf16 in training if desired

        # init state + memory
        s = self._init_state(b, device=device, dtype=torch.float32)
        mem_k, mem_v = self.core.init_memory(b, device=device, dtype=torch.float32)

        (
            logits,
            s_final,
            mem_k_final,
            mem_v_final,
            avg_write,
            avg_neuromod,
            ponder_cost,
            moe_load_balance,
            moe_entropy,
            mix_entropy,
            mhc_alpha,
            last_write_gate,
            last_write_idx,
            last_write_w,
        ) = self._forward_stateful(
            input_ids=input_ids,
            attention_mask=attention_mask,
            k_steps=k_steps,
            s=s,
            mem_k=mem_k,
            mem_v=mem_v,
            episodic_enabled_override=episodic_enabled_override,
        )

        out: Dict[str, torch.Tensor] = {
            "logits": logits,
            "avg_write_strength": avg_write,
            "avg_neuromod": avg_neuromod,
            "ponder_cost": ponder_cost,
            "moe_load_balance": moe_load_balance,
            "moe_entropy": moe_entropy,
            "mix_entropy": mix_entropy,
            "mhc_alpha": mhc_alpha,
        }
        # Optional debug traces (slot indices/weights + gate for the *last* token's last write)
        if bool(getattr(self.cfg, "debug_memory_write_trace", False)):
            out["last_write_gate"] = last_write_gate
            out["last_write_idx"] = last_write_idx
            out["last_write_w"] = last_write_w

        if labels is not None:
            # shift: position i predicts labels at i+1
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            out["loss"] = loss

        return out

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: Optional[float] = None,
        k_steps: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
    ) -> torch.Tensor:
        self.eval()
        device = input_ids.device

        out = input_ids

        for _ in range(max_new_tokens):
            # Truncate context if needed. If we truncate, cached memory is invalid; reset.
            if out.size(1) > self.cfg.max_seq_len:
                out = out[:, -self.cfg.max_seq_len :]
                if self.cfg.persistent_gen_memory:
                    self.reset_generation_memory()

            if self.cfg.persistent_gen_memory:
                b = out.size(0)
                # Initialize cache by processing the full prompt once
                if self._gen_state is None or self._gen_mem_k is None or self._gen_mem_v is None:
                    s0 = self._init_state(b, device=out.device, dtype=torch.float32)
                    mem_k, mem_v = self.core.init_memory(b, device=out.device, dtype=torch.float32)
                    logits, s_f, mem_k_f, mem_v_f, *_ = self._forward_stateful(
                        input_ids=out,
                        attention_mask=None,
                        k_steps=k_steps,
                        s=s0,
                        mem_k=mem_k,
                        mem_v=mem_v,
                    )
                    self._gen_state, self._gen_mem_k, self._gen_mem_v = s_f.detach(), mem_k_f.detach(), mem_v_f.detach()
                    next_logits = logits[:, -1, :] / max(temperature, 1e-6)
                else:
                    # Incremental update using only the last token embedding
                    last_id = out[:, -1:]
                    # run one token step with cached state+mem
                    logits, s_f, mem_k_f, mem_v_f, *_ = self._forward_stateful(
                        input_ids=last_id,
                        attention_mask=None,
                        k_steps=k_steps,
                        s=self._gen_state,
                        mem_k=self._gen_mem_k,
                        mem_v=self._gen_mem_v,
                    )
                    # optional EMA update of persistent memory
                    decay = float(self.cfg.persistent_decay)
                    if decay > 0.0:
                        mem_k_f = decay * self._gen_mem_k + (1.0 - decay) * mem_k_f
                        mem_v_f = decay * self._gen_mem_v + (1.0 - decay) * mem_v_f
                    self._gen_state, self._gen_mem_k, self._gen_mem_v = s_f.detach(), mem_k_f.detach(), mem_v_f.detach()
                    next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            else:
                logits = self.forward(out, attention_mask=None, labels=None, k_steps=k_steps)["logits"]
                next_logits = logits[:, -1, :] / max(temperature, 1e-6)

            if repetition_penalty is not None and repetition_penalty > 1.0:
                for b in range(out.size(0)):
                    seen = out[b].unique()
                    next_logits[b, seen] = next_logits[b, seen] / repetition_penalty

            # No-repeat ngram (simple per-batch implementation)
            if no_repeat_ngram_size is not None and int(no_repeat_ngram_size) > 0:
                n = int(no_repeat_ngram_size)
                for b in range(out.size(0)):
                    seq = out[b].tolist()
                    if len(seq) < n:
                        continue
                    prefix = tuple(seq[-(n - 1) :]) if n > 1 else tuple()
                    banned = set()
                    # Collect all tokens that previously followed this prefix
                    for i in range(len(seq) - n + 1):
                        if tuple(seq[i : i + n - 1]) == prefix:
                            banned.add(seq[i + n - 1])
                    if banned:
                        idx = torch.tensor(list(banned), device=next_logits.device, dtype=torch.long)
                        next_logits[b, idx] = -float("inf")

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(next_logits, k=min(top_k, next_logits.size(-1)))
                cutoff = v[:, -1].unsqueeze(-1)
                next_logits = torch.where(next_logits < cutoff, torch.full_like(next_logits, -float("inf")), next_logits)

            # Nucleus sampling
            if top_p is not None and 0.0 < float(top_p) < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cum = torch.cumsum(sorted_probs, dim=-1)
                # Mask tokens with cumulative prob above top_p (keep at least 1 token)
                mask = cum > float(top_p)
                mask[:, 0] = False
                sorted_logits = sorted_logits.masked_fill(mask, -float("inf"))
                # Unsort back to original vocab order
                next_logits = torch.full_like(next_logits, -float("inf"))
                next_logits.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return out


