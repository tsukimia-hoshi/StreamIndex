"""Pipeline-composition bench — does the chunked indexer extend the regime
when paired with TileLang's attention kernel (the production-quality path)?

This is the REFRAMED gating experiment after `bench_vs_tilelang_pipelined.py`
showed that TileLang's `sparse_mla_fwd_pipelined` does NOT OOM at S=256K and
runs ~6× faster than our Triton attention. The conclusion from that bench:
the contribution of FlashSparse is on the *indexer* side, not the attention
side.

This bench tests the full V4-Flash pipeline at long context with TileLang's
attention kernel as the common backend, comparing two indexer paths:
(A) materialize-then-topk: full [B, S, T] FP32 score matrix in HBM.
(B) chunked partition-merge top-k.

Both paths feed the same TileLang attention kernel. The headline question:
does (A) OOM where (B) survives, with the production-quality attention
kernel in the loop?

Run as: python benchmarks/bench_pipeline_composition.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import torch

_TILELANG_EX = os.path.expanduser("~/flash-sparse/references/tilelang/examples/deepseek_v32")
if _TILELANG_EX not in sys.path:
    sys.path.insert(0, _TILELANG_EX)

from flash_sparse.triton.indexer_score import indexer_score
from flash_sparse.triton.chunked_indexer import chunked_indexer_topk


@dataclass
class PipelineCfg:
    B: int = 1
    S: int = 4096
    H_q: int = 64  # V4-Flash query heads
    H_kv: int = 1  # MQA
    dim: int = 512
    tail_dim: int = 64
    H_I: int = 64  # V4-Flash indexer heads
    D_I: int = 128  # indexer head dim
    m: int = 4  # compressor stride: T = S / m
    top_k: int = 512
    kv_stride: int = 1


def _build_inputs(cfg: PipelineCfg):
    """Build all pipeline-stage inputs at V4-Flash dims."""
    DQK = cfg.dim + cfg.tail_dim
    T = cfg.S // cfg.m

    # Attention-side inputs (TileLang's split-MLA layout).
    q_attn = torch.randn(cfg.B, cfg.S, cfg.H_q, DQK, dtype=torch.bfloat16, device="cuda") * 0.1
    kv_attn = torch.randn(cfg.B, cfg.S, cfg.H_kv, DQK, dtype=torch.bfloat16, device="cuda") * 0.1
    q_attn.clamp_(-10, 10)
    kv_attn.clamp_(-10, 10)
    q_start = torch.tensor([0], dtype=torch.int32, device="cuda")

    # Indexer-side inputs (our single-D layout).
    sigma = cfg.D_I**-0.5
    q_idx = torch.randn(cfg.B, cfg.S, cfg.H_I, cfg.D_I, dtype=torch.bfloat16, device="cuda") * sigma
    k_idx = torch.randn(cfg.B, T, cfg.D_I, dtype=torch.bfloat16, device="cuda") * sigma
    weights = torch.randn(cfg.B, cfg.S, cfg.H_I, dtype=torch.float32, device="cuda")

    return {
        "q_attn": q_attn,
        "kv_attn": kv_attn,
        "q_start": q_start,
        "q_idx": q_idx,
        "k_idx": k_idx,
        "weights": weights,
        "T": T,
    }


def _time_us(fn, n_iter=3, n_warmup=1):
    for _ in range(n_warmup):
        fn
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(n_iter):
            fn
            e1.record()
            torch.cuda.synchronize()
            return e0.elapsed_time(e1) * 1000.0 / n_iter  # us


def _materialize_indexer_topk(q_idx, k_idx, weights, top_k):
    """Reference path: full [B, S, T] FP32 score matrix → torch.topk."""
    scores = indexer_score(q_idx, k_idx, weights)
    _, top_idx = scores.topk(top_k, dim=-1)
    return top_idx.to(torch.int32)


def _chunked_indexer_topk(q_idx, k_idx, weights, top_k, *, chunk_s=4096, chunk_t=4096):
    top_idx, _ = chunked_indexer_topk(
        q_idx,
        k_idx,
        weights,
        top_k,
        chunk_s=chunk_s,
        chunk_t=chunk_t,
    )
    return top_idx.to(torch.int32)


def _compressed_to_decompressed_indices(top_idx_compressed: torch.Tensor, m: int, S_kv: int):
    """Map compressed-key top-k indices [B, S, K] (range [0, T)) to
    decompressed-key indices [B, S, K*m] (range [0, S_kv)) for TileLang's
    attention kernel.

    Each compressed key represents `m` decompressed keys; expand each entry.
    Final shape [B, S, K*m, 1] for kv_group=1, kept at [B, S, K*m] for now.
    """
    B, S, K = top_idx_compressed.shape
    expanded = top_idx_compressed.unsqueeze(-1) * m + torch.arange(
        m, device=top_idx_compressed.device, dtype=torch.int32
    )
    expanded = expanded.reshape(B, S, K * m).clamp(0, S_kv - 1)
    return expanded


def bench_one(cfg: PipelineCfg, *, indexer: str) -> dict:
    """One full-pipeline run with `indexer` ∈ {'materialize', 'chunked'}."""
    try:
        from sparse_mla_fwd_pipelined import sparse_mla_fwd as tl_sparse_mla_fwd
    except Exception as e:
        return {"status": "import_fail", "msg": str(e)[:120]}

    try:
        inputs = _build_inputs(cfg)
        S_kv = cfg.S

        # TileLang attention's K is in compressed-key space — we use cfg.top_k
        # as the *compressed* top-k count, and TileLang sees that many
        # decompressed keys (top_k * m = 2048 for V4-Flash). For an apples-
        # to-apples comparison with the materialize path, the indexer
        # selects top_k compressed keys, and the attention kernel is
        # compiled with topk=top_k * m.
        attn_topk = cfg.top_k * cfg.m

        # Materialize and chunked produce identical [B, S, top_k] indices in
        # compressed-key space; only the HBM peak path differs.
        def run_indexer():
            if indexer == "materialize":
                return _materialize_indexer_topk(
                    inputs["q_idx"],
                    inputs["k_idx"],
                    inputs["weights"],
                    cfg.top_k,
                )
            return _chunked_indexer_topk(
                inputs["q_idx"],
                inputs["k_idx"],
                inputs["weights"],
                cfg.top_k,
                chunk_s=2048,
                chunk_t=8192,
            )

        # Build attention kernel once (JIT-compiled per shape).
        kernel = tl_sparse_mla_fwd(
            cfg.B,
            cfg.S,
            S_kv,
            cfg.H_q,
            cfg.dim,
            cfg.tail_dim,
            attn_topk,
            cfg.kv_stride,
            kv_group=cfg.H_kv,
            sm_scale=None,
            is_causal=True,
            CP0=True,
        )

        def fn():
            top_idx = run_indexer()
            attn_idx = _compressed_to_decompressed_indices(top_idx, cfg.m, S_kv)
            attn_idx = attn_idx.unsqueeze(2)  # [B, S, 1, attn_topk]
            return kernel(
                inputs["q_attn"],
                inputs["kv_attn"],
                attn_idx,
                inputs["q_start"],
            )

        # Warmup (JIT compile + autotune).
        out = fn()
        torch.cuda.synchronize()

        # Peak measurement.
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        fn()
        torch.cuda.synchronize()
        peak_gb = max(0, torch.cuda.max_memory_allocated() - baseline) / 1024**3

        us = _time_us(fn)

        for v in inputs.values():
            del v
        del kernel, out
        torch.cuda.empty_cache()
        return {"status": "ok", "us": us, "peak_gb": peak_gb}
    except torch.OutOfMemoryError:
        return {"status": "OOM"}
    except RuntimeError as e:
        msg = str(e).lower()
        if "invalid argument" in msg or "65535" in msg:
            return {"status": "GRID_LIMIT"}
        return {"status": "ERROR", "msg": str(e)[:120]}


def main():
    if not torch.cuda.is_available():
        return
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")
    print("Pipeline-composition bench: TileLang attention + {materialize, chunked} indexer")
    print("V4-Flash dims: H_q=64, dim=512, tail=64, H_I=64, D_I=128, m=4, top_k=512")
    print("=" * 92)
    print(
        f"{'S':>8} {'mat (ms)':>10} {'mat HBM':>10} {'chunk (ms)':>11} {'chunk HBM':>11} {'mat status':>14}"
    )
    print("-" * 92)

    configs = [
        PipelineCfg(S=8192),
        PipelineCfg(S=32768),
        PipelineCfg(S=65536),
        PipelineCfg(S=131072),
        PipelineCfg(S=262144),
    ]

    for cfg in configs:
        r_mat = bench_one(cfg, indexer="materialize")
        torch.cuda.empty_cache()
        r_chk = bench_one(cfg, indexer="chunked")
        torch.cuda.empty_cache()

        def fmt_t(r):
            return f"{r['us'] / 1000:>10.1f}" if r["status"] == "ok" else f"{r['status']:>10}"

        def fmt_h(r):
            return f"{r['peak_gb']:>8.2f} GB" if r["status"] == "ok" else f"{'-':>11}"

        print(
            f"{cfg.S:>8}"
            f" {fmt_t(r_mat)} {fmt_h(r_mat):>10}"
            f" {fmt_t(r_chk):>11} {fmt_h(r_chk):>11}"
            f" {r_mat.get('status', 'ERR'):>14}"
        )
        if r_mat.get("status") == "ERROR":
            print(f" (mat error: {r_mat.get('msg', '?')})")
            if r_chk.get("status") == "ERROR":
                print(f" (chunk error: {r_chk.get('msg', '?')})")


if __name__ == "__main__":
    main
