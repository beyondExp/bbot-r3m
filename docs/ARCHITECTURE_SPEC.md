### R³M Architecture Spec (Recursive, Routed, Replay‑Consolidating Memory)

This is a **base model architecture** proposal designed to be:

- **Trainable on limited compute** (single 24GB GPU for fast iteration at small scale)
- **Brain-principled** (multi-timescale memory, sparse activation, adaptive compute, consolidation)
- **Potentially novel** through the *combination* of: iterative refinement + stable mixing + dual-memory + neuromodulated gating

This spec is intentionally concrete: it defines modules, tensor shapes, interfaces, and constraints.

---

### 1) High-level overview

R³M replaces the “single forward pass” LM with a **cognitive cycle**:

- **Perception**: encode input into a compact latent state
- **Recursive thinking loop** (repeat up to \(K_{max}\) steps):
  - read episodic memory
  - (optionally) call a tool/solver
  - update internal state
  - update a candidate output plan/answer
  - decide whether to halt
- **Action**: emit output tokens (or a structured action)
- **Consolidation** (periodic): replay episodic traces and distill into semantic memory

---

### 2) Data types and notation

- **Batch** \(B\)
- **Sequence length** \(T\)
- **Model width** \(D\)
- **Reasoning steps** \(K\) (adaptive, per sample)
- **Episodic memory slots** \(M_e\) with key/value sizes \(D_k, D_v\) (often \(D_k=D_v=D\))
- **Semantic memory size** \(M_s\) (can be a smaller bank, or a learned “slow weights” adapter)

We use the following tensors:

- **Input tokens**: `x_ids` \(\in \mathbb{N}^{B\times T}\)
- **Token embeddings**: `x` \(\in \mathbb{R}^{B\times T\times D}\)
- **Latent state**: `s_k` \(\in \mathbb{R}^{B\times D}\) (one per sample per thinking step)
- **Candidate output state** (optional): `y_k` \(\in \mathbb{R}^{B\times D}\)
- **Episodic memory**:
  - keys: `E_k` \(\in \mathbb{R}^{B\times M_e\times D}\)
  - values: `E_v` \(\in \mathbb{R}^{B\times M_e\times D}\)
  - ages / priorities: `E_meta` \(\in \mathbb{R}^{B\times M_e\times r}\) (small metadata)
- **Semantic memory** (one option):
  - bank: `S` \(\in \mathbb{R}^{M_s\times D}\) (global learned parameters)
  - optional low-rank adapter weights updated only during consolidation

---

### 3) Modules (minimal set)

#### 3.1 Perception encoder (text-first MVP)

**Goal**: compress the token sequence into an initial latent state.

- **Encoder**: small Transformer/SSM stack (shared with core) or a simple pooling head.
- **Output**: `s_0 = Pool(x)` where `Pool` could be:
  - last token embedding
  - mean pooling
  - attention pooling

**MVP choice**: mean or last-token pooling to minimize confounders.

#### 3.2 Recurrent brain-core \(f_\theta\)

**Goal**: update state using recurrence, not full-sequence attention each step.

Inputs per step:
- current state `s_k` \(\in \mathbb{R}^{B\times D}\)
- perceptual summary `p` \(\in \mathbb{R}^{B\times D}\) (from encoder)
- memory reads `m_e`, `m_s` \(\in \mathbb{R}^{B\times D}\)
- optional tool result embedding `t_k` \(\in \mathbb{R}^{B\times D}\)

Outputs per step:
- updated state `s_{k+1}`
- updated candidate `y_{k+1}` (optional)

Recommended structure:
- gated update (GRU-like) or SSM-style update:
  - `u = concat(s_k, p, m_e, m_s, t_k)`
  - `Δ = MLP(u)` then `s_{k+1} = Norm(s_k + g ⊙ Δ)`

**Key requirement**: keep updates stable across many steps.

#### 3.3 Router (sparse expert selection)

**Goal**: activate specialized compute without paying full cost.

- Experts: `{language, planning, memory_ops, tool_ops, puzzle_ops}` (start with 2–3)
- Router logits: `r_k = W_r s_k` \(\in \mathbb{R}^{B\times E}\)
- Select top-`k_exp` experts per step, with weights.

**MVP**: start with `E=2` (base + memory_ops) and `k_exp=1` to keep training stable.

#### 3.4 Episodic memory (fast writes)

**Goal**: store intermediate reasoning traces and tool results.

Memory read:
- query `q = W_q s_k`
- attention over keys `E_k` to produce `m_e`

Memory write (sparse, gated):
- write gate `w_k = sigmoid(W_w s_k)` \(\in \mathbb{R}^{B\times 1}\)
- candidate write vector `v_write = W_v s_k`
- write only when `w_k > τ_write` OR via straight-through estimator.

Eviction:
- simple: FIFO or lowest priority
- better: priority based on surprise/utility signal (see neuromodulator below)

#### 3.5 Semantic memory (slow)

**Goal**: stable knowledge store updated during consolidation, not every step.

Two viable implementations:

- **Bank**: `S` is learned and read like attention memory each step; updated only via optimizer (standard training), consolidation changes *what* is replayed rather than direct writes.
- **Slow adapter**: a small low-rank adapter whose parameters are only updated in consolidation phases (a closer “two-speed” system).

**MVP**: bank-only (simpler), add slow adapter later.

#### 3.6 Neuromodulator \(g_k\) (surprise/utility)

**Goal**: decide when to spend compute and when to write memory.

Compute a scalar per sample per step:
- `g_k = sigmoid(w_g · features_k)` where `features_k` can include:
  - prediction error proxy (negative logprob of candidate)
  - uncertainty (entropy)
  - novelty w.r.t. semantic memory (low similarity)
  - verifier feedback (when available)

Use `g_k` to gate:
- episodic write probability
- tool call probability
- halting probability (inverse)

#### 3.7 Halting head (adaptive compute)

**Goal**: stop thinking when confident.

- halt logit `h_k = W_h s_k`
- halt probability `p_halt = sigmoid(h_k)`

Training:
- supervised: encourage halting near the first correct solution step
- regularization: penalize long thinking (compute budget), e.g. `λ * expected_steps`

---

### 4) Stability constraints (mHC-inspired “don’t explode” rules)

Recursive loops are fragile. We enforce stability through **constrained mixing** when combining signals:

When mixing multiple streams into an update:
- base update candidates: `Δ_base`, `Δ_mem`, `Δ_tool`, …
- mixture weights `α` must be **non-negative** and sum to 1 (convex combination)

Practical parameterization:
- `α = softmax(logits)` (guarantees simplex)
- update `Δ = Σ_i α_i * Δ_i`
- then `s_{k+1} = s_k + η * Δ` with small `η` (learned or fixed)

This is the “lightweight cousin” of the doubly-stochastic constraint in mHC; it gives you:
- bounded updates
- reduced risk of exploding internal dynamics

---

### 5) Output heads (text-first)

Two-layer approach:

- **State-to-token head**: `logits = LMHead(s_final)` where `s_final` is the halted state.
- Optionally, for richer generation: condition a small decoder on `s_final` and generate tokens autoregressively.

**MVP**: produce text with a standard decoder, but allow `s_final` to influence the logits via a bias/adapter.

---

### 6) Interfaces for tools (neurosymbolic loop)

Tool calling is a structured action:

- tool choice `tool_id` \(\in \{1..N\}\)
- tool args `tool_args` (JSON-like; in model: tokenized or latent structured vector)
- tool result `tool_res` embedded back into `t_k`

Tools are optional in early phases; keep the interface stable from day 1 even if tools are disabled.

---

### 7) MVP scope (to avoid chaos)

Start with:

- text-only input
- recurrent core with K-step refinement
- halting head
- episodic memory read/write (simple FIFO)
- convex mixing stability
- no tools yet

Then add:
- routing experts
- tool interface + verifiers
- consolidation (replay) + semantic memory adapter



