"""Long-context benchmark — the regime where the score matrix dominates.

Compares two indexer + top-k paths:

Path A: ``indexer_score`` (Triton, materializes full [B, S, T] FP32 score
matrix to HBM) followed by ``torch.topk``. The current production
path; baseline for comparison.

Path B: ``chunked_indexer_topk`` — processes (chunk_S × chunk_T) tiles,
merges per-chunk top-k incrementally. Peak HBM is ~chunk_T/T of
path A. The streaming-top-k theorem (``docs/streaming_topk.md``)
guarantees set-equivalence.

The story we expect:
- At small S (≤ 4K): paths are similar; chunked has overhead from many small
kernel launches, possibly slower.
- At medium S (8K-32K): paths similar wall-clock, but path A's score matrix
is starting to consume tens of MB of HBM.
- At large S (≥ 64K): path A's score matrix is hundreds of MB to GBs;
path B continues with bounded peak HBM.
- At very large S (≥ 256K): path A OOMs, path B doesn't. *That's the
scaling crossover the paper claims.*

Run as:
python benchmarks/bench_long_context.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch

from flash_sparse.triton.chunked_indexer import chunked_indexer_topk
from flash_sparse.triton.indexer_score import indexer_score


@dataclass
class LongContextConfig:
    S: int
    T: int
    H_I: int = 16
    D_I: int = 64
    K: int = 32
    chunk_S: int = 1024
    chunk_T: int = 4096


def _time_us(fn, n_iter=10, n_warmup=3):
    for _ in range(n_warmup):
        fn
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iter):
            fn
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) * 1000.0 / n_iter


def bench_path_a(cfg: LongContextConfig) -> dict:
    """Path A: full score matrix + torch.topk. May OOM at large S."""
    torch.manual_seed(42)
    B = 1

    try:
        q = torch.randn(B, cfg.S, cfg.H_I, cfg.D_I, dtype=torch.bfloat16, device="cuda") * 0.5
        k_idx = torch.randn(B, cfg.T, cfg.D_I, dtype=torch.bfloat16, device="cuda") * 0.5
        weights = torch.randn(B, cfg.S, cfg.H_I, dtype=torch.float32, device="cuda")
        torch.cuda.synchronize()
        # Reset peak so the measurement reflects only what fn allocates ON TOP of inputs.
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
    except torch.OutOfMemoryError:
        return {"status": "OOM_at_inputs", "us": float("inf"), "peak_gb": float("nan")}

    def fn():
        scores = indexer_score(q, k_idx, weights)
        _, _ = scores.topk(cfg.K, dim=-1)

    try:
        t = _time_us(fn)
        peak_total = torch.cuda.max_memory_allocated()
        peak_extra = max(0, peak_total - baseline)
        del q, k_idx, weights
        return {"status": "ok", "us": t, "peak_gb": peak_extra / 1024**3}
    except torch.OutOfMemoryError:
        return {"status": "OOM", "us": float("inf"), "peak_gb": float("nan")}
    except RuntimeError as e:
        if "invalid argument" in str(e).lower() or "65535" in str(e):
            return {"status": "GRID_LIMIT", "us": float("inf"), "peak_gb": float("nan")}
        raise


def bench_path_b(cfg: LongContextConfig) -> dict:
    """Path B: chunked indexer. Peak HBM bounded by chunk size."""
    torch.manual_seed(42)
    B = 1

    try:
        q = torch.randn(B, cfg.S, cfg.H_I, cfg.D_I, dtype=torch.bfloat16, device="cuda") * 0.5
        k_idx = torch.randn(B, cfg.T, cfg.D_I, dtype=torch.bfloat16, device="cuda") * 0.5
        weights = torch.randn(B, cfg.S, cfg.H_I, dtype=torch.float32, device="cuda")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
    except torch.OutOfMemoryError:
        return {"status": "OOM_at_inputs", "us": float("inf"), "peak_gb": float("nan")}

    def fn():
        chunked_indexer_topk(
            q,
            k_idx,
            weights,
            top_k=cfg.K,
            chunk_s=cfg.chunk_S,
            chunk_t=cfg.chunk_T,
        )

    try:
        t = _time_us(fn)
        peak_total = torch.cuda.max_memory_allocated()
        peak_extra = max(0, peak_total - baseline)
        del q, k_idx, weights
        return {"status": "ok", "us": t, "peak_gb": peak_extra / 1024**3}
    except torch.OutOfMemoryError:
        return {"status": "OOM", "us": float("inf"), "peak_gb": float("nan")}
    except RuntimeError as e:
        if "invalid argument" in str(e).lower() or "65535" in str(e):
            return {"status": "GRID_LIMIT", "us": float("inf"), "peak_gb": float("nan")}
        raise


def main():
    if not torch.cuda.is_available():
        return
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")

    # Configs spanning short → very long context.
    # T = S for "every position has a compressed counterpart" (m=1) — pessimistic
    # since real m=4 means T = S/4. Stresses HBM more.
    configs = [
        LongContextConfig(S=1024, T=1024, chunk_S=512, chunk_T=1024),
        LongContextConfig(S=4096, T=4096, chunk_S=1024, chunk_T=2048),
        LongContextConfig(S=16384, T=16384, chunk_S=1024, chunk_T=4096),
        LongContextConfig(S=32768, T=32768, chunk_S=1024, chunk_T=4096),
        LongContextConfig(S=65536, T=65536, chunk_S=1024, chunk_T=4096),
        LongContextConfig(S=131072, T=131072, chunk_S=1024, chunk_T=4096),
        LongContextConfig(S=262144, T=262144, chunk_S=1024, chunk_T=4096),
    ]
    # At S = T = 131072: materialized score matrix is 64 GB FP32 (feasible on
    # H200 140 GB but autotuner overlap sweeps push past); the chunked path
    # works fine. At S=T=262144 the score matrix is 256 GB — impossible on a
    # single H200 — but the chunked path still runs.

    print(
        f"{'S':>8} {'T':>8} {'K':>6}"
        f" {'A: time (ms)':>14} {'A: peak HBM (GB)':>18}"
        f" {'B: time (ms)':>14} {'B: peak HBM (GB)':>18}"
        f" {'speedup':>9}"
    )
    print("-" * 110)

    for cfg in configs:
        r_a = bench_path_a(cfg)
        torch.cuda.empty_cache()
        r_b = bench_path_b(cfg)
        torch.cuda.empty_cache()

        def _fail_label(status):
            if status == "OOM" or status == "OOM_at_inputs":
                return "OOM"
            if status == "GRID_LIMIT":
                return "GRID>65k"
            return status

        a_str = f"{r_a['us'] / 1000:>14.2f}" if r_a["status"] == "ok" else f"{_fail_label(r_a['status']):>14}"
        a_hbm = f"{r_a['peak_gb']:>18.2f}" if r_a["status"] == "ok" else f"{'-':>18}"
        b_str = f"{r_b['us'] / 1000:>14.2f}" if r_b["status"] == "ok" else f"{_fail_label(r_b['status']):>14}"
        b_hbm = f"{r_b['peak_gb']:>18.2f}" if r_b["status"] == "ok" else f"{'-':>18}"
        if r_a["status"] == "ok" and r_b["status"] == "ok":
            speedup = f"{r_a['us'] / r_b['us']:>9.2f}x"
        elif r_a["status"] != "ok" and r_b["status"] == "ok":
            speedup = f"{'∞':>9}"
        else:
            speedup = f"{'-':>9}"

            print(f"{cfg.S:>8} {cfg.T:>8} {cfg.K:>6} {a_str} {a_hbm} {b_str} {b_hbm} {speedup}")


if __name__ == "__main__":
    main
