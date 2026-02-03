## R3M (Recurrent + Reasoning + Memory) — Architecture Overview (with diagrams)

This document explains the **implemented** R3M recurrent language model architecture in this repo and includes the diagrams we discussed.

### Where the implementation lives

- **Model**: `src/models/r3m_recurrent.py`
  - `R3MRecurrentConfig`
  - `R3MRecurrentLM`
  - `R3MRecurrentCore`
  - `EpisodicMemory` (KV memory + routed writes)
  - **mHC (minimal)** stream mixing utilities (e.g. `sinkhorn_birkhoff`, core mHC params)
  - **MoE (optional)** feedforward (`MoEFeedForward`)
- **HF streaming pretrain**: `scripts/pretrain_r3m_recurrent_hf.py`
- **Sampling from checkpoints**: `scripts/generate_r3m_recurrent_samples.py`

---

## Diagram 1 — High-level model flow (per token)

```mermaid
flowchart TD
  subgraph Input["Input"]
    T["token_ids[t]"]
    E["Token+Pos Embedding"]
    T --> E
  end

  subgraph Core["Recurrent Core (per token t)"]
    S0["state s_t"]
    M0["episodic memory (K,V)"]
    E --> GRU["GRU / token assimilation"]
    S0 --> GRU
    GRU --> S1["state s' (post-assimilation)"]

    subgraph Think["Thinking Loop (k steps / optional ACT halting)"]
      direction TB

      %% mHC minimal stream (optional)
      S1 --> MHC["mHC stream widen+mix (optional)\nS ∈ R^{n×D}, H_res (Sinkhorn DS), h_pre/h_post"]
      MHC --> SEFF["effective state s_eff"]

      %% if mHC off, s_eff = S1
      S1 -. "if mHC disabled" .-> SEFF

      SEFF --> READ["Episodic Read\nm_e = Attn(s_eff, K,V)"]
      READ --> MIX["Convex Mixture\nΔ = α·MLP(s_eff) + (1-α)·MLP([s_eff,m_e])"]
      MIX --> UPDATE["State Update\ns ← LN(s + η·Δ)\n(or stream update if mHC)"]

      UPDATE --> WRITE["Routed Episodic Write (top‑k)\nK,V ← write(s)"]
      WRITE --> HALT["(optional) ACT Halting\nponder cost"]
      HALT -->|next inner step| SEFF
    end

    Think --> S2["final state s_{t+1}"]
    WRITE --> M1["updated memory (K,V)"]
  end

  subgraph Output["Output"]
    S2 --> LN["Output Norm"]
    LN --> HEAD["LM Head (tied to embeddings)"]
    HEAD --> LOGITS["logits for token_{t+1}"]
  end

  M0 --> READ
  M1 --> M0
```

### What this means

- The model is **causal by construction**: it processes tokens **one-by-one** and produces logits for the next token from the updated state.
- Each token step has:
  - **Assimilation** (GRU)
  - **Thinking** (k micro-steps, optionally adaptive halting)
  - **Memory read/write** (episodic KV slots)

---

## Diagram 2 — Generation dataflow (persistent memory on/off)

```mermaid
flowchart TD
  subgraph Prompt["Given prompt tokens"]
    P["input_ids = prompt"]
  end

  subgraph A["Mode A: persistent_gen_memory = false (recompute every step)"]
    P --> A1["Forward over full sequence\n(for each token: GRU + think-loop + mem)"]
    A1 --> A2["Take last logits"]
    A2 --> A3["Sample next token"]
    A3 --> A4["Append token to prompt"]
    A4 --> A1
  end

  subgraph B["Mode B: persistent_gen_memory = true (cached state+memory)"]
    P --> B0["If cache empty:\nrun full prompt once"]
    B0 --> C1["cache: state s, memory (K,V)"]
    C1 --> B1["For each new token:\nrun ONE token-step only\n(last token)"]
    B1 --> B2["Update cache (s, K,V)\n(optional EMA decay)"]
    B2 --> B3["Take logits -> sample next token"]
    B3 --> B4["Append token"]
    B4 --> B1
  end

  subgraph Notes["Notes"]
    N1["If context is truncated to max_seq_len:\ncache must be reset (history changed)"]
    N2["k_steps during generation is usually small (1–2)\nfor speed; training can use larger k_train"]
  end
```

### What this means

- **Without** persistent generation memory: generation cost grows with prompt length (recompute).
- **With** persistent generation memory: after the first prompt pass, each new token is ~constant work (one token-step).

---

## Diagram 3 — mHC + MoE placement (where Sinkhorn is used)

```mermaid
flowchart TD
  S["state (or s_eff if mHC)"] --> CORE["RecurrentCore.step()"]

  subgraph MHC["mHC (minimal, inside thinking loop)"]
    S --> W["Widen into n streams: S_stream ∈ R^{n×D}"]
    Hlog["H_res logits (n×n)"] --> SK1["Sinkhorn-Knopp\n(project to doubly-stochastic P)"]
    SK1 --> Hres["H_res = (1-α)I + αP"]
    Hres --> MIXS["Mix streams: S_stream ← H_res · S_stream"]
    MIXS --> PRE["Compress: s_eff = h_preᵀ · S_stream"]
    PRE --> S
    POST["Expand delta: add h_post ⊗ Δ to streams"] --> MIXS
  end

  subgraph Think["Thinking micro-step (repeated k times / optional ACT)"]
    S --> READ["Episodic read m_e"]
    S --> FFN["Base update branch"]
    READ --> MEMFFN["Memory update branch"]
    FFN --> MIX["ConvexMixture (2-way) -> Δ"]
    MEMFFN --> MIX
    MIX --> UPD["Update state (or streams if mHC)"]
    UPD --> WRITE["Top‑k routed episodic write"]
  end

  subgraph MoE["MoE (optional replacement for base FFN)"]
    S --> ROUTE["Router logits over experts"]
    ROUTE --> MOESEL["Top‑k routing\n(or Sinkhorn-balanced routing)"]
    MOESEL --> EXP["Apply selected expert MLPs"]
    EXP --> FFN
    subgraph MoE_Sinkhorn["If moe_router = sinkhorn"]
      ROUTE --> SK2["Sinkhorn-Knopp\n(balance token↔expert assignment)"]
      SK2 --> MOESEL
    end
  end
```

### What this means

- **mHC (minimal)**: uses Sinkhorn to constrain the **stream mixing matrix** inside the thinking loop.
- **MoE (optional)**: uses either top‑k routing or Sinkhorn-balanced routing for **expert selection**.

---

## Key properties (in plain language)

- **Causal + streaming-friendly**: token-by-token state update enables incremental generation and avoids future leakage.
- **Variable compute per token**: `k_train` / optional adaptive halting means “think more on harder tokens”.
- **Episodic memory**: external KV slots for retrieval and selectively routed writes (top‑k).
- **Stability knobs**:
  - write-rate regularization prevents “always write” collapse
  - mHC-style constrained mixing preserves identity-like behavior in deeper inner loops

---

## Notes for training/evaluation

- When evaluating early pretraining, use prompts that match the data distribution (e.g. book-style prompts for Gutenberg).
- Instruction-style behavior is best introduced via a small instruction stage or distillation.



