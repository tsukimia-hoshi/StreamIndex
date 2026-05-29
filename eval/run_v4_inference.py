"""V4-Flash inference harness for addition 1.

Loads V4-Flash, optionally monkey-patches its CSA attention forward to use
`flash_csa_forward`, then runs a JSONL of prompts and records:
- prediction (model's output text)
- TTFT (time-to-first-token, ms)
- decode tokens/sec
- peak HBM (GB) during prefill+decode

The integration patch on V4-Flash modelling code is the day-1 surgery and
is explicitly marked as TODO below. The rest of the harness (loading,
looping, metrics) is fully implemented.

Usage:
python eval/run_v4_inference.py \\
--model deepseek-ai/DeepSeek-V4-Flash \\
--indexer chunked \\
--samples eval/data/needle_64K.jsonl \\
--out eval/runs/needle_64K_chunked.jsonl \\
--max-new-tokens 64 \\
--tp-size 2

`--indexer` is one of {materialize, chunked}. The chunked path requires
the V4 modelling code to be patched — see `_apply_chunked_indexer_patch`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


# --------------------------------------------------------------------------
# THE DAY-1 SURGERY POINT
# --------------------------------------------------------------------------
def _apply_chunked_indexer_patch(model) -> int:
    """Replace V4-Flash's CSA attention with `flash_csa_forward`.

    NOT YET IMPLEMENTED — depends on the exact V4-Flash modelling code,
    which we will inspect on day 1 of the integration.

    Expected surgery (based on DeepSeek-V3.2-exp lineage):

    1. Locate the attention module per layer:
    for layer in model.model.layers: attn = layer.self_attn
    Class name is something like `DeepseekV3SparseAttention` or
    `DeepseekV4Attention` depending on the V4-Flash release.

    2. Wrap its forward to detect prefill mode (start_pos == 0) and
    optionally route through our chunked indexer + the existing
    attention kernel:

    original_forward = attn.forward
    def patched_forward(self, hidden_states, position_ids=None, ...):
    if not self._use_chunked or hidden_states.shape[1] < 4096:
    return original_forward(hidden_states, position_ids=position_ids, ...)
    # Project q, kv, q_idx, k_idx_compressed exactly as the
    # original forward does, then call flash_csa_forward.
    ...
    return o, ...
    attn.forward = types.MethodType(patched_forward, attn)

    3. Verify by running a small parity check: feed a 2K-token prompt
    through both paths and assert max |Δlogit| < 1e-3 in BF16.

    Returns: number of layers patched. Asserts failure if 0.
    """
    raise NotImplementedError(
        "V4-Flash chunked-indexer patch — see this function's docstring. "
        "Day-1 task: inspect modelling_deepseek_v*.py, implement the patch, "
        "run a parity check at S=2048."
    )


# --------------------------------------------------------------------------
# Inference loop
# --------------------------------------------------------------------------
def _run_one(model, tokenizer, prompt: str, max_new_tokens: int, device: str) -> dict:
    """Run a single prompt; return prediction + timing metrics."""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    n_input_tokens = inputs["input_ids"].shape[1]

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    t0 = time.perf_counter
    # First-token: prefill only.
    with torch.no_grad:
        _ = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        torch.cuda.synchronize()
        t_first = time.perf_counter

        # Continue decode for the rest.
        with torch.no_grad:
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            torch.cuda.synchronize()
            t_done = time.perf_counter

            peak_gb = torch.cuda.max_memory_allocated() / 1024**3
            n_new = out.shape[1] - n_input_tokens
            decode_time = max(t_done - t_first, 1e-9)
            decode_tps = (n_new - 1) / decode_time if n_new > 1 else 0.0
            ttft_ms = (t_first - t0) * 1000

            pred_ids = out[0, n_input_tokens:]
            prediction = tokenizer.decode(pred_ids, skip_special_tokens=True)

            return {
                "prediction": prediction,
                "n_input_tokens": n_input_tokens,
                "n_new_tokens": int(n_new),
                "ttft_ms": ttft_ms,
                "decode_tokens_per_sec": decode_tps,
                "peak_hbm_gb": peak_gb,
            }


def main():
    p = argparse.ArgumentParser
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--indexer", choices=["materialize", "chunked"], required=True)
    p.add_argument("--samples", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--tp-size", type=int, default=2)
    p.add_argument("--max-samples", type=int, default=10**9)
    p.add_argument("--device", default="cuda")
    args = p.parse_args

    # Lazy imports.
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"required libs missing: {e}; pip install torch transformers", file=sys.stderr)
        sys.exit(1)

        print(f"Loading {args.model} (TP={args.tp_size}) ...", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        # Loading strategy depends on whether the V4-Flash release ships
        # native HF integration. If yes, AutoModelForCausalLM works. If no,
        # we'll need DeepSeek's inference repo (`references/DeepSeek-V4-Pro/inference/`).
        # Day-1 spike determines which path.
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval

        if args.indexer == "chunked":
            n_patched = _apply_chunked_indexer_patch(model)
            print(f"Patched {n_patched} attention layers to use chunked indexer", file=sys.stderr)
        else:
            print("Using stock (materialize) indexer", file=sys.stderr)

            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            n_done = 0
            with open(args.samples) as f_in, open(args.out, "w") as f_out:
                for line in f_in:
                    if n_done >= args.max_samples:
                        break
                    rec = json.loads(line)
                    try:
                        metrics = _run_one(
                            model,
                            tokenizer,
                            rec["prompt"],
                            args.max_new_tokens,
                            args.device,
                        )
                        metrics["id"] = rec["id"]
                        metrics["status"] = "ok"
                    except torch.cuda.OutOfMemoryError as e:
                        metrics = {"id": rec["id"], "status": "OOM", "msg": str(e)[:200]}
                        torch.cuda.empty_cache()
                    except Exception as e:
                        metrics = {
                            "id": rec["id"],
                            "status": "ERROR",
                            "msg": f"{type(e).__name__}: {str(e)[:200]}",
                        }
                        f_out.write(json.dumps(metrics, ensure_ascii=False) + "\n")
                        f_out.flush
                        n_done += 1
                        if metrics.get("status") == "ok":
                            print(
                                f"[{n_done}] {rec['id']} S={metrics['n_input_tokens']} "
                                f"TTFT={metrics['ttft_ms']:.0f}ms "
                                f"decode={metrics['decode_tokens_per_sec']:.1f}tok/s "
                                f"HBM={metrics['peak_hbm_gb']:.1f}GB",
                                file=sys.stderr,
                            )
                        else:
                            print(
                                f"[{n_done}] {rec['id']} {metrics.get('status')}: "
                                f"{metrics.get('msg', '')[:80]}",
                                file=sys.stderr,
                            )

                            print(f"Wrote {n_done} predictions to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main
