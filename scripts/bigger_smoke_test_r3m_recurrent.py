# Ensure R3M/ is on sys.path\nimport os\nimport sys\nR3M_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))\nif R3M_ROOT not in sys.path:\n    sys.path.insert(0, R3M_ROOT)\n\nimport \nimport sys
import \nimport sys
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


def build_tiny_corpus() -> List[str]:
    # A small mixed corpus: chat-like + plain text. Kept tiny on purpose for fast iteration.
    return [
        "Human: Hello!\nAssistant: Hi. How can I help you today?\n",
        "Human: What is 2+2?\nAssistant: 2+2 is 4.\n",
        "Human: Explain what a transformer model is.\nAssistant: A transformer is a neural network that uses attention to process sequences.\n",
        "Human: Write a short haiku about winter.\nAssistant: Cold wind through bare trees\nSnow hushes the restless earth\nNight holds a clear moon\n",
        "Human: Summarize: Cats are small domesticated carnivores.\nAssistant: Cats are small domesticated carnivores.\n",
        "In the beginning, there was only a quiet page and the promise of words.\n",
        "The quick brown fox jumps over the lazy dog.\n",
        "To be, or not to be, that is the question.\n",
        "A curious mind asks; a patient mind answers.\n",
        "Numbers can be added, multiplied, and compared.\n",
    ]


def sample_batch_from_buffer(
    token_buffer: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    token_buffer: [N] token ids
    returns input_ids: [B, T]
    """
    n = int(token_buffer.numel())
    if n <= seq_len + 2:
        raise ValueError("Token buffer too small for requested seq_len; add more corpus text.")
    starts = torch.randint(0, n - seq_len - 1, (batch_size,), device=device)
    chunks = [token_buffer[s : s + seq_len].unsqueeze(0) for s in starts]
    return torch.cat(chunks, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="outputs/r3m_rec_big_smoke")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--sample-every", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--tokenizer", type=str, default="gpt2")
    ap.add_argument("--model-dim", type=int, default=192)
    ap.add_argument("--k-train", type=int, default=2)
    ap.add_argument("--compare-mhc", action="store_true", help="Run baseline vs mHC back-to-back with same seed/data into subfolders")
    ap.add_argument("--corpus-repeat", type=int, default=1, help="Repeat the built-in tiny corpus N times to enlarge the token buffer")

    # Architecture toggles
    ap.add_argument("--mhc-enabled", action="store_true")
    ap.add_argument("--mhc-n", type=int, default=4)
    ap.add_argument("--mhc-alpha-init", type=float, default=0.01)
    ap.add_argument("--mhc-sinkhorn-iters", type=int, default=12)
    ap.add_argument("--mhc-temperature", type=float, default=1.0)

    ap.add_argument("--moe-enabled", action="store_true")
    ap.add_argument("--moe-num-experts", type=int, default=4)
    ap.add_argument("--moe-router", type=str, default="topk", choices=["topk", "sinkhorn"])
    ap.add_argument("--moe-top-k", type=int, default=1)
    ap.add_argument("--moe-sinkhorn-iters", type=int, default=8)
    ap.add_argument("--moe-temperature", type=float, default=1.0)
    args = ap.parse_args()

    # Local import to keep smoke test self-contained
    import sys

    sys.path.insert(0, os.path.abspath("."))
    from r3m.models.recurrent import R3MRecurrentConfig, R3MRecurrentLM

    root_out_dir = Path(args.out)
    root_out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Build a single long token buffer for fast random span sampling
    corpus = build_tiny_corpus()
    rep = max(1, int(args.corpus_repeat))
    if rep > 1:
        corpus = corpus * rep
    ids: List[int] = []
    for s in corpus:
        ids.extend(tok.encode(s))
        ids.append(tok.eos_token_id)
    token_buffer = torch.tensor(ids, dtype=torch.long, device=device)

    prompt = "Human: Hello!\nAssistant:"

    def run_one(run_name: str, mhc_enabled: bool) -> float:
        run_dir = root_out_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "train.log"

        # reset RNG so both runs see identical batches/sampling
        torch.manual_seed(int(args.seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(args.seed))

        cfg = R3MRecurrentConfig(
            vocab_size=len(tok),
            d_model=int(args.model_dim),
            max_seq_len=int(args.max_len),
            k_train=int(args.k_train),
            k_max=max(32, int(args.k_train)),
            episodic_enabled=True,
            episodic_slots=64,
            neuromodulator_enabled=True,
            write_topk=1,
            write_straight_through=True,
            adaptive_halting=False,
            moe_enabled=bool(args.moe_enabled),
            moe_num_experts=int(args.moe_num_experts),
            moe_router=str(args.moe_router),
            moe_top_k=int(args.moe_top_k),
            moe_sinkhorn_iters=int(args.moe_sinkhorn_iters),
            moe_temperature=float(args.moe_temperature),
            mhc_enabled=bool(mhc_enabled),
            mhc_n=int(args.mhc_n),
            mhc_alpha_init=float(args.mhc_alpha_init),
            mhc_sinkhorn_iters=int(args.mhc_sinkhorn_iters),
            mhc_temperature=float(args.mhc_temperature),
        )

        model = R3MRecurrentLM(cfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=0.1)

        last_loss = float("nan")
        with log_path.open("w", encoding="utf-8") as f:
            def log(msg: str) -> None:
                print(msg, flush=True)
                f.write(msg + "\n")
                f.flush()

            log(f"run={run_name} | device={device} | mhc_enabled={mhc_enabled} | moe_enabled={bool(args.moe_enabled)}")

            model.train()
            for step in range(int(args.steps)):
                input_ids = sample_batch_from_buffer(
                    token_buffer=token_buffer,
                    batch_size=int(args.batch_size),
                    seq_len=int(args.max_len),
                    device=device,
                )
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100

                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, k_steps=cfg.k_train)
                loss = out["loss"]

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                last_loss = float(loss.item())

                if (step + 1) % max(1, int(args.log_every)) == 0:
                    avg_write = float(out.get("avg_write_strength", torch.tensor(0.0)).mean().item())
                    ponder = float(out.get("ponder_cost", torch.tensor(0.0)).mean().item())
                    moe_lb = float(out.get("moe_load_balance", torch.tensor(0.0)).item())
                    moe_ent = float(out.get("moe_entropy", torch.tensor(0.0)).item())
                    mix_ent = float(out.get("mix_entropy", torch.tensor(0.0)).item())
                    mhc_alpha = float(out.get("mhc_alpha", torch.tensor(0.0)).item())
                    log(
                        f"step {step+1}/{args.steps} | loss {loss.item():.4f} "
                        f"| avg_write {avg_write:.4f} | ponder {ponder:.4f} "
                        f"| moe_lb {moe_lb:.4f} | moe_ent {moe_ent:.4f} | mix_ent {mix_ent:.4f} | mhc_alpha {mhc_alpha:.4f}"
                    )

                if (step + 1) % max(1, int(args.sample_every)) == 0 or (step + 1) == int(args.steps):
                    model.eval()
                    with torch.no_grad():
                        p_ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
                        gen = model.generate(
                            p_ids,
                            max_new_tokens=120,
                            temperature=0.9,
                            top_k=50,
                            top_p=0.95,
                            repetition_penalty=1.05,
                            no_repeat_ngram_size=3,
                            eos_token_id=tok.eos_token_id,
                            k_steps=1,
                        )
                        text = tok.decode(gen[0].tolist())
                    (run_dir / f"sample_step_{step+1}.txt").write_text(text, encoding="utf-8")
                    model.train()

            torch.save(
                {"step": int(args.steps), "config": cfg.__dict__, "model_state_dict": model.state_dict()},
                run_dir / "checkpoint_final.pt",
            )
            log(f"done | final_loss {last_loss:.6f}")

        return last_loss

    if bool(args.compare_mhc):
        loss_base = run_one("baseline", mhc_enabled=False)
        loss_mhc = run_one("mhc", mhc_enabled=True)
        (root_out_dir / "compare_summary.txt").write_text(
            f"baseline_final_loss: {loss_base:.6f}\n"
            f"mhc_final_loss: {loss_mhc:.6f}\n",
            encoding="utf-8",
        )
    else:
        # Single run, controlled by --mhc-enabled
        run_name = "mhc" if bool(args.mhc_enabled) else "baseline"
        run_one(run_name, mhc_enabled=bool(args.mhc_enabled))


if __name__ == "__main__":
    main()



