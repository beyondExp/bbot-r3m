### R³M Algorithms (Pseudocode)

This file gives concrete pseudocode for:

- the forward “cognitive cycle”
- episodic memory read/write
- adaptive halting
- consolidation via replay

Notation: tensors are batch-first.

---

### 1) Forward pass (text-first MVP, no tools)

```
Inputs:
  x_ids: [B, T] token ids
  K_max: maximum reasoning steps

Parameters:
  Embed, Encoder, Core, Router (optional), Heads
  EpisodicMemory (keys/values + metadata)

Outputs:
  logits: [B, T, V] or [B, V] depending on training format
  aux: halting stats, step counts, memory usage

Algorithm:
  x = Embed(x_ids)                     # [B, T, D]
  p = EncoderPool(x)                   # [B, D]  (e.g., last token or mean pool)

  s = InitState(p)                     # [B, D]  (often a linear proj of p)
  y = InitCandidate(s)                 # [B, D]  (optional)

  halted = [False]*B
  steps  = [0]*B

  for k in 0..K_max-1:
      # 1) Read episodic + semantic memory
      m_e = EpisodicRead(s)            # [B, D]
      m_s = SemanticRead(s)            # [B, D]  (can be 0-vector in MVP)

      # 2) Compute neuromodulator signal (surprise/utility proxy)
      g = Neuromodulator(s, p, m_e)    # [B, 1]

      # 3) Compute update candidates (streams)
      Δ_base = CoreBase(s, p)          # [B, D]
      Δ_mem  = CoreMem(s, m_e, m_s)    # [B, D]

      # 4) Stable mixing (convex combination)
      α = softmax(MixHead(s, p, m_e))  # [B, 2]  (nonneg, sums to 1)
      Δ = α[:,0]*Δ_base + α[:,1]*Δ_mem # [B, D]

      # 5) Update state (bounded step size)
      s = LayerNorm(s + eta * Δ)       # [B, D]

      # 6) Optional: update candidate
      y = CandidateHead(s, y)          # [B, D]

      # 7) Decide whether to write episodic memory
      p_write = WriteHead(s, g)        # [B, 1]
      EpisodicWrite(s, y, p, gate=p_write)

      # 8) Adaptive halting
      p_halt = HaltHead(s, g)          # [B, 1]
      for b in 0..B-1:
          if not halted[b] and p_halt[b] > tau_halt:
              halted[b] = True
              steps[b] = k+1

      if all halted:
          break

  # Final state
  s_final = s
  logits  = OutputHead(s_final, x)     # choose an output format
  return logits, {steps, halted, memory_stats, g_stats}
```

---

### 2) Episodic memory read/write (simple)

#### 2.1 Read (attention over slots)

```
E_k: [B, M_e, D]
E_v: [B, M_e, D]
q  = W_q s                      # [B, D]
att = softmax((E_k · q) / sqrt(D))    # [B, M_e]
m_e = Σ_i att_i * E_v[i]        # [B, D]
```

#### 2.2 Write (gated FIFO)

```
gate in [0,1], optionally thresholded
write_vec = W_wv concat(s, y, p)       # [B, D]
if gate > tau_write:
   write into next slot (FIFO pointer)
```

---

### 3) Halting training options

R³M can be trained with either:

- **Fixed K** during training, halting only used at inference (simplest)
- **Learned halting** with a compute penalty:
  - `L = L_task + λ_steps * E[steps]`

For initial stability, do:

- train with fixed `K_train` (e.g., 8)
- enable halting at eval time first
- then train halting head later

---

### 4) Consolidation via replay (periodic phase)

Consolidation is a separate training phase that runs every N updates (or every epoch):

```
Inputs:
  ReplayBuffer of episodic traces:
    (x_ids, intermediate states s_k, candidate y_k, tool results if any, final correct solution)

Goal:
  Improve semantic memory / slow adapter so future tasks need fewer steps and fewer writes.

Algorithm (sketch):
  sample batch of traces
  run model with episodic memory disabled
  train to match:
     - final answer
     - optionally intermediate state targets (distillation)
  update only semantic components (slow adapter or semantic bank)
```

This is the “sleep-like” stage: stabilize what matters, reduce reliance on huge episodic buffers.



