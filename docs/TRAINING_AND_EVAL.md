### R³M Training & Evaluation Plan (Single GPU Friendly)

This plan is staged to minimize confounders and get early signal.

---

### 1) First results: what “counts” as progress

You should see at least one of:

- **Language sanity**: coherent sentences after short training (pipeline correctness)
- **Compute helps**: accuracy improves as `K` increases (iterative refinement works)
- **Memory helps**: episodic memory improves tasks requiring intermediate state retention

If none of these show up, fix training correctness before adding complexity.

---

### 2) Stage 0 — correctness baseline (1–3 days)

**Goal**: prove the training loop is correct and produces coherent text.

- Train a small baseline (dense Transformer or SSM) with correct causal masking and pad masking.
- Evaluate:
  - next-token loss curve
  - short prompt completions (manual spot check)

**Exit criteria**
- non-degenerate generation (no instant repetition/EOS spam)

---

### 3) Stage 1 — R³M without tools (1–2 weeks)

**Goal**: demonstrate “thinking steps” produce measurable gains.

Start with:
- fixed `K_train` (e.g. 8)
- no learned halting yet
- episodic memory on (simple FIFO write)
- semantic memory read can be disabled initially

Datasets (text-first, verifiable):
- arithmetic
- list/sequence transforms
- tiny logic tasks with a verifier
- “instruction following” mini set for conversational behavior

Metrics:
- accuracy vs K at eval: `K ∈ {1,2,4,8,16}`
- average number of writes per sample
- stability: gradient norms, NaNs, collapse rate

**Exit criteria**
- accuracy improves with larger K on at least one task family

---

### 4) Stage 2 — adaptive halting (1–2 weeks)

**Goal**: save compute on easy tasks, spend compute on hard tasks.

Add:
- halting head
- compute penalty term

Metrics:
- steps distribution by difficulty
- same or better accuracy at lower average steps

**Exit criteria**
- comparable accuracy with fewer average steps vs fixed-K

---

### 5) Stage 3 — neurosymbolic tools (2–4 weeks)

**Goal**: small model + tools beats larger dense baseline on structured puzzles.

Add:
- tool call head
- 1–2 verifiable tools:
  - solver/verifier for a puzzle domain
  - search in a small DSL space

Training:
- imitation: teach correct tool call format + when to call
- verifier feedback: reward correctness
- distill tool results back into the model

Metrics:
- success rate with and without tools
- tool call precision/recall
- correctness of final outputs under verification

---

### 6) Stage 4 — consolidation (replay) (4–8+ weeks)

**Goal**: reduce reliance on episodic memory and improve generalization.

Add:
- periodic replay phase
- update only semantic components (slow adapter)

Metrics:
- same accuracy with fewer episodic writes
- less catastrophic forgetting across task mixes

---

### 7) Pitfalls to watch (early warning signals)

- **No “K helps” curve**: indicates the loop isn’t learning iterative refinement (or training bug).
- **Always halts immediately**: halting head collapse; increase penalty shaping or supervise halting.
- **Memory write spam**: write gate saturates; add write budget penalty.
- **Instability over steps**: exploding norms; reduce step size `eta`, enforce convex mixing, tighten norms.



