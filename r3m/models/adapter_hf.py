"""
R3M-on-Transformer Adapter
-------------------------
Wrap a pretrained Hugging Face causal LM (dense Transformer) and add a lightweight
R3M-style episodic memory module on top of the final hidden states.

Key idea for single-GPU feasibility:
- Keep the pretrained base model (language prior) intact.
- Train only the adapter/memory modules (fast -> hours/days).

This is NOT a full replacement for R3MRecurrentLM; it's an augmentation layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from r3m.models.recurrent import EpisodicMemory


@dataclass
class R3MHFAdapterConfig:
    base_model_name: str = "Qwen/Qwen2.5-0.5B"
    # Training / numerical
    dropout: float = 0.0
    # Episodic memory
    episodic_enabled: bool = True
    episodic_slots: int = 64
    write_topk: int = 1  # 0=soft, 1=top-1, k=top-k
    write_straight_through: bool = True
    write_gate_max: float = 0.2
    detach_memory_across_tokens: bool = True
    # How strongly memory read influences logits
    mem_scale: float = 0.5
    # Clamp memory residual to avoid adapter overpowering the base model (prevents repetition loops)
    mem_delta_clip: float = 2.0
    # Freeze base model weights (recommended)
    freeze_base: bool = True
    # If detach_memory_across_tokens is False, optionally detach every N tokens (truncated BPTT for memory)
    detach_every_tokens: int = 0


class R3MHFAdapterLM(nn.Module):
    """
    Pretrained HF CausalLM + episodic memory adapter.

    We compute base hidden states causally, then run a causal memory read/write pass
    over the sequence and add a learned memory residual before the base LM head.
    """

    def __init__(self, cfg: R3MHFAdapterConfig, *, torch_dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.cfg = cfg

        self.base = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map=None,
        )
        # Infer model dim from embeddings.
        emb = self.base.get_input_embeddings()
        if emb is None:
            raise RuntimeError("Base model has no input embeddings; cannot infer hidden size.")
        d_model = int(emb.weight.shape[1])

        self.d_model = d_model
        self.mem = EpisodicMemory(d_model=d_model, slots=int(cfg.episodic_slots))
        self.mem_ln = nn.LayerNorm(d_model)
        self.mem_out = nn.Linear(d_model, d_model, bias=False)

        # neuromod + write gate (simple, stable)
        self.neuromod = nn.Linear(2 * d_model, 1, bias=True)
        self.write_gate = nn.Linear(d_model + 1, 1, bias=True)

        self.dropout = nn.Dropout(float(cfg.dropout))

        if bool(cfg.freeze_base):
            for p in self.base.parameters():
                p.requires_grad = False

    def adapter_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Return ONLY adapter weights (exclude base model weights).
        Use this for lightweight checkpoints.
        """
        sd = self.state_dict()
        # Drop base.* keys
        return {k: v for k, v in sd.items() if not k.startswith("base.")}

    def load_adapter_state_dict(self, sd: Dict[str, torch.Tensor], strict: bool = True) -> None:
        cur = self.state_dict()
        filtered = {k: v for k, v in sd.items() if k in cur and not k.startswith("base.")}
        missing, unexpected = self.load_state_dict({**{k: cur[k] for k in cur if k.startswith("base.")}, **filtered}, strict=False)
        if strict and missing:
            # It's okay to miss base keys; but adapter keys should match.
            missing_adapter = [k for k in missing if not k.startswith("base.")]
            if missing_adapter:
                raise RuntimeError(f"Missing adapter keys: {missing_adapter[:20]}")
        if strict and unexpected:
            unexpected_adapter = [k for k in unexpected if not k.startswith("base.")]
            if unexpected_adapter:
                raise RuntimeError(f"Unexpected adapter keys: {unexpected_adapter[:20]}")

    def _shift_labels(self, logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # logits: [B,T,V], labels: [B,T]
        # Predict token t+1 at position t (standard causal LM).
        return logits[:, :-1, :].contiguous(), labels[:, 1:].contiguous()

    def init_memory_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create a persistent episodic memory state (mem_k, mem_v) in fp32.
        This can be carried across multiple turns for multi-turn evaluation.
        """
        return self.mem.init_memory(batch_size=batch_size, device=device, dtype=torch.float32)

    def _mem_step(
        self,
        h_t: torch.Tensor,  # [B,D] base dtype
        mem_k: torch.Tensor,
        mem_v: torch.Tensor,
        keep: Optional[torch.Tensor] = None,  # [B] bool
        update: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One causal memory step for a single token hidden state.
        Returns: (delta_residual [B,D] in base dtype, mem_k, mem_v)
        """
        h_f = h_t.to(torch.float32)
        m_f = self.mem.read(h_f, mem_k, mem_v).to(torch.float32)

        g = torch.sigmoid(self.neuromod(torch.cat([h_f, m_f], dim=-1)))  # [B,1]
        w_gate = torch.sigmoid(self.write_gate(torch.cat([h_f, g], dim=-1)))  # [B,1]
        w_gate = torch.clamp(w_gate, 0.0, float(self.cfg.write_gate_max))
        if keep is not None:
            w_gate = w_gate * keep.to(dtype=torch.float32).unsqueeze(-1)

        if update:
            mem_k, mem_v, _ = self.mem.write(
                s=h_f,
                write_vec=h_f,
                mem_k=mem_k,
                mem_v=mem_v,
                gate=w_gate,
                topk=int(self.cfg.write_topk),
                straight_through=bool(self.cfg.write_straight_through),
            )

        delta_f = self.mem_out(self.mem_ln(m_f)) * float(self.cfg.mem_scale)  # fp32
        if float(getattr(self.cfg, "mem_delta_clip", 0.0)) > 0.0:
            c = float(self.cfg.mem_delta_clip)
            delta_f = torch.clamp(delta_f, -c, c)
        delta = delta_f.to(dtype=h_t.dtype)
        return delta, mem_k, mem_v

    @torch.no_grad()
    def encode_with_memory_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        mem_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
        *,
        update_memory: bool = True,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass that (optionally) updates an external memory state.
        Useful for multi-turn conversations where we truncate visible context but keep memory.
        Returns (logits, (mem_k, mem_v)).
        """
        device = input_ids.device
        bsz = int(input_ids.size(0))
        if mem_state is None:
            mem_k, mem_v = self.init_memory_state(batch_size=bsz, device=device)
        else:
            mem_k, mem_v = mem_state

        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        h = out.hidden_states[-1]  # [B,T,D]
        b, t, _ = h.shape
        h_mod: List[torch.Tensor] = []
        for i in range(t):
            hi = h[:, i, :]
            keep_i = attention_mask[:, i].to(torch.bool) if attention_mask is not None else None
            delta, mem_k, mem_v = self._mem_step(hi, mem_k, mem_v, keep=keep_i, update=update_memory)
            h_mod.append((hi + self.dropout(delta)).unsqueeze(1))
            if update_memory and self.training:
                if bool(self.cfg.detach_memory_across_tokens):
                    mem_k = mem_k.detach()
                    mem_v = mem_v.detach()
                else:
                    n = int(getattr(self.cfg, "detach_every_tokens", 0))
                    if n > 0 and ((i + 1) % n == 0):
                        mem_k = mem_k.detach()
                        mem_v = mem_v.detach()
        h2 = torch.cat(h_mod, dim=1)
        head = self.base.get_output_embeddings()
        if head is None:
            raise RuntimeError("Base model has no output embeddings (lm_head).")
        logits = head(h2)
        return logits, (mem_k, mem_v)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Any = None,
    ) -> Dict[str, Any]:
        # Ask base for hidden states so we can apply adapter before lm_head.
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_hidden_states=True,
            return_dict=True,
        )

        h = out.hidden_states[-1]  # [B,T,D]
        b, t, d = h.shape

        if not bool(self.cfg.episodic_enabled):
            logits = out.logits  # base logits
        else:
            # Initialize per-forward memory.
            mem_k, mem_v = self.mem.init_memory(batch_size=b, device=h.device, dtype=h.dtype)

            h_mod = []
            for i in range(t):
                hi = h[:, i, :]  # [B,D]
                # respect padding
                if attention_mask is not None:
                    keep = attention_mask[:, i].to(dtype=torch.bool)  # [B]
                else:
                    keep = None

                # Adapter math in fp32 (base may be bf16/fp16 on GPU).
                hi_f = hi.to(torch.float32)
                m_f = self.mem.read(hi_f, mem_k, mem_v).to(torch.float32)  # [B,D] fp32

                g = torch.sigmoid(self.neuromod(torch.cat([hi_f, m_f], dim=-1)))  # [B,1] fp32
                w_gate = torch.sigmoid(self.write_gate(torch.cat([hi_f, g], dim=-1)))  # [B,1] fp32
                w_gate = torch.clamp(w_gate, 0.0, float(self.cfg.write_gate_max))
                if keep is not None:
                    w_gate = w_gate * keep.to(dtype=torch.float32).unsqueeze(-1)

                mem_k, mem_v, _ = self.mem.write(
                    s=hi_f,
                    write_vec=hi_f,
                    mem_k=mem_k,
                    mem_v=mem_v,
                    gate=w_gate,
                    topk=int(self.cfg.write_topk),
                    straight_through=bool(self.cfg.write_straight_through),
                )
                if self.training and bool(self.cfg.detach_memory_across_tokens):
                    mem_k = mem_k.detach()
                    mem_v = mem_v.detach()

                # memory residual into hidden state
                delta_f = self.mem_out(self.mem_ln(m_f)) * float(self.cfg.mem_scale)  # fp32
                delta = delta_f.to(dtype=hi.dtype)
                h_mod.append((hi + self.dropout(delta)).unsqueeze(1))

            h2 = torch.cat(h_mod, dim=1)  # [B,T,D]
            head = self.base.get_output_embeddings()
            if head is None:
                raise RuntimeError("Base model has no output embeddings (lm_head).")
            logits = head(h2)  # [B,T,V]

        loss = None
        if labels is not None:
            l2, y2 = self._shift_labels(logits, labels)
            loss = F.cross_entropy(l2.view(-1, l2.size(-1)), y2.view(-1), ignore_index=-100)

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": out.past_key_values,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: Optional[float] = 0.9,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Simple incremental generation using base KV-cache + adapter on last hidden state.
        """
        self.eval()
        device = input_ids.device

        bsz = int(input_ids.size(0))
        # Initialize memory fresh for this generation call.
        # Keep episodic buffers in fp32 for stability.
        mem_k, mem_v = self.mem.init_memory(batch_size=bsz, device=device, dtype=torch.float32)

        # Prime with full prompt (no cache) then continue token-by-token with past_key_values.
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = out.past_key_values
        h_last = out.hidden_states[-1][:, -1, :]  # [B,D] (base dtype)
        # Update memory through the prompt tokens (cheap approximation: only last token).
        if bool(self.cfg.episodic_enabled):
            h_f = h_last.to(torch.float32)
            m_f = self.mem.read(h_f, mem_k, mem_v).to(torch.float32)
            g = torch.sigmoid(self.neuromod(torch.cat([h_f, m_f], dim=-1)))
            w_gate = torch.clamp(
                torch.sigmoid(self.write_gate(torch.cat([h_f, g], dim=-1))), 0.0, float(self.cfg.write_gate_max)
            )
            mem_k, mem_v, _ = self.mem.write(
                s=h_f,
                write_vec=h_f,
                mem_k=mem_k,
                mem_v=mem_v,
                gate=w_gate,
                topk=int(self.cfg.write_topk),
                straight_through=bool(self.cfg.write_straight_through),
            )

        head = self.base.get_output_embeddings()
        if head is None:
            raise RuntimeError("Base model has no output embeddings (lm_head).")

        seq = input_ids
        for _ in range(int(max_new_tokens)):
            if bool(self.cfg.episodic_enabled):
                h_f = h_last.to(torch.float32)
                m_f = self.mem.read(h_f, mem_k, mem_v).to(torch.float32)
                delta_f = self.mem_out(self.mem_ln(m_f)) * float(self.cfg.mem_scale)
                h_use = h_last + delta_f.to(dtype=h_last.dtype)
                logits = head(h_use).unsqueeze(1)  # [B,1,V]
            else:
                logits = out.logits[:, -1:, :]

            next_logits = logits[:, -1, :] / max(float(temperature), 1e-6)
            if top_k is not None and int(top_k) > 0:
                k = min(int(top_k), next_logits.size(-1))
                vals, idx = torch.topk(next_logits, k)
                mask = torch.full_like(next_logits, float("-inf"))
                mask.scatter_(1, idx, vals)
                next_logits = mask
            if top_p is not None and 0.0 < float(top_p) < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cum = torch.cumsum(probs, dim=-1)
                cutoff = cum > float(top_p)
                cutoff[..., 0] = False
                sorted_logits[cutoff] = float("-inf")
                next_logits = torch.full_like(next_logits, float("-inf"))
                next_logits.scatter_(1, sorted_idx, sorted_logits)

            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # [B,1]
            seq = torch.cat([seq, next_id], dim=1)
            if eos_token_id is not None:
                if bool((next_id.squeeze(-1) == int(eos_token_id)).all().item()):
                    break

            # Step base model with cache
            out = self.base(
                input_ids=next_id,
                attention_mask=None,
                use_cache=True,
                past_key_values=past,
                output_hidden_states=True,
                return_dict=True,
            )
            past = out.past_key_values
            h_last = out.hidden_states[-1][:, -1, :]
            if bool(self.cfg.episodic_enabled):
                h_f = h_last.to(torch.float32)
                m_f = self.mem.read(h_f, mem_k, mem_v).to(torch.float32)
                g = torch.sigmoid(self.neuromod(torch.cat([h_f, m_f], dim=-1)))
                w_gate = torch.clamp(
                    torch.sigmoid(self.write_gate(torch.cat([h_f, g], dim=-1))), 0.0, float(self.cfg.write_gate_max)
                )
                mem_k, mem_v, _ = self.mem.write(
                    s=h_f,
                    write_vec=h_f,
                    mem_k=mem_k,
                    mem_v=mem_v,
                    gate=w_gate,
                    topk=int(self.cfg.write_topk),
                    straight_through=bool(self.cfg.write_straight_through),
                )

        return seq

    @torch.no_grad()
    def generate_with_memory_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        mem_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
        *,
        max_new_tokens: int = 64,
        do_sample: bool = True,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: Optional[float] = 0.9,
        eos_token_id: Optional[int] = None,
        update_memory: bool = True,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generation that carries an external episodic memory state across calls.
        This enables meaningful multi-turn evaluation with truncated visible context.
        """
        self.eval()
        device = input_ids.device
        bsz = int(input_ids.size(0))
        if mem_state is None:
            mem_k, mem_v = self.init_memory_state(batch_size=bsz, device=device)
        else:
            mem_k, mem_v = mem_state

        head = self.base.get_output_embeddings()
        if head is None:
            raise RuntimeError("Base model has no output embeddings (lm_head).")

        # 1) Prime on the prompt (no cache), then update memory across prompt tokens.
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = out.past_key_values
        h_seq = out.hidden_states[-1]  # [B,T,D]
        if bool(self.cfg.episodic_enabled):
            t = int(h_seq.size(1))
            for i in range(t):
                keep_i = attention_mask[:, i].to(torch.bool) if attention_mask is not None else None
                _, mem_k, mem_v = self._mem_step(h_seq[:, i, :], mem_k, mem_v, keep=keep_i, update=update_memory)

        h_last = h_seq[:, -1, :]
        seq = input_ids

        # 2) Autoregressive loop
        for _ in range(int(max_new_tokens)):
            if bool(self.cfg.episodic_enabled):
                delta, mem_k, mem_v = self._mem_step(h_last, mem_k, mem_v, keep=None, update=update_memory)
                logits = head(h_last + delta).unsqueeze(1)  # [B,1,V]
            else:
                logits = out.logits[:, -1:, :]

            if not bool(do_sample):
                # Greedy decode (deterministic)
                next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # [B,1]
                seq = torch.cat([seq, next_id], dim=1)
                if eos_token_id is not None and bool((next_id.squeeze(-1) == int(eos_token_id)).all().item()):
                    break
                out = self.base(
                    input_ids=next_id,
                    attention_mask=None,
                    use_cache=True,
                    past_key_values=past,
                    output_hidden_states=True,
                    return_dict=True,
                )
                past = out.past_key_values
                h_last = out.hidden_states[-1][:, -1, :]
                continue

            next_logits = logits[:, -1, :] / max(float(temperature), 1e-6)
            if top_k is not None and int(top_k) > 0:
                k = min(int(top_k), next_logits.size(-1))
                vals, idx = torch.topk(next_logits, k)
                mask = torch.full_like(next_logits, float("-inf"))
                mask.scatter_(1, idx, vals)
                next_logits = mask
            if top_p is not None and 0.0 < float(top_p) < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cum = torch.cumsum(probs, dim=-1)
                cutoff = cum > float(top_p)
                cutoff[..., 0] = False
                sorted_logits[cutoff] = float("-inf")
                next_logits = torch.full_like(next_logits, float("-inf"))
                next_logits.scatter_(1, sorted_idx, sorted_logits)

            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # [B,1]
            seq = torch.cat([seq, next_id], dim=1)
            if eos_token_id is not None and bool((next_id.squeeze(-1) == int(eos_token_id)).all().item()):
                break

            out = self.base(
                input_ids=next_id,
                attention_mask=None,
                use_cache=True,
                past_key_values=past,
                output_hidden_states=True,
                return_dict=True,
            )
            past = out.past_key_values
            h_last = out.hidden_states[-1][:, -1, :]

        return seq, (mem_k, mem_v)


def save_adapter_checkpoint(path: str, cfg: R3MHFAdapterConfig, adapter_state: Dict[str, torch.Tensor]) -> None:
    torch.save({"config": asdict(cfg), "adapter_state_dict": adapter_state}, path)



