"""Benchmark: Triton sparse_attn forward + backward vs the DeepSeek TileLang
reference (`kernel.py:sparse_attn`).

this script's output is committed alongside
the kernel. Re-run with `python benchmarks/bench_sparse_attn.py` and commit the
updated table when kernel changes are merged.

Reports per-config latency (ms) and effective TFLOPS, where for our purposes
TFLOPS counts `2 · n_heads · K · D` FMAs per query token (one Q·K^T plus one
P·V matmul, both `[H × K × D]`).
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import torch

from flash_sparse.triton import sparse_attn_fwd
from flash_sparse.triton.sparse_attn_bwd import sparse_attn_bwd
from flash_sparse.triton.sparse_attn_bwd_v2 import sparse_attn_bwd_v2

# Optional: TileLang baseline.
_INFERENCE_DIRS = [
 os.path.expanduser("~/flash-sparse/references/DeepSeek-V4-Pro/inference"),
]
_DS_SPARSE_ATTN = None
for d in _INFERENCE_DIRS:
 if os.path.isdir(d):
 if d not in sys.path:
 sys.path.insert(0, d)
 try:
 import importlib

 ds = importlib.import_module("kernel")
 _DS_SPARSE_ATTN = getattr(ds, "sparse_attn", None)
 break
 except Exception:
 pass


@dataclass
class BenchConfig:
 B: int
 S: int
 H: int
 D: int
 N_kv: int
 K: int


def _flops_per_query(H: int, K: int, D: int) -> int:
 """Per-query FMAs: Q·K^T (H · K · D) + softmax (negligible) + P·V (H · K · D).
 Each FMA = 2 FLOPs."""
 return 2 * 2 * H * K * D


def _time_us(fn, n_iter: int = 50, n_warmup: int = 10) -> float:
 for _ in range(n_warmup):
 fn
 torch.cuda.synchronize
 start = torch.cuda.Event(enable_timing=True)
 end = torch.cuda.Event(enable_timing=True)
 start.record
 for _ in range(n_iter):
 fn
 end.record
 torch.cuda.synchronize
 return start.elapsed_time(end) * 1000.0 / n_iter # ms → µs


def bench_fwd(cfg: BenchConfig) -> dict:
 torch.manual_seed(0)
 q = torch.randn(cfg.B, cfg.S, cfg.H, cfg.D, dtype=torch.bfloat16, device="cuda")
 kv = torch.randn(cfg.B, cfg.N_kv, cfg.D, dtype=torch.bfloat16, device="cuda")
 attn_sink = torch.randn(cfg.H, dtype=torch.float32, device="cuda")
 topk_idxs = torch.randint(0, cfg.N_kv, (cfg.B, cfg.S, cfg.K), dtype=torch.int32, device="cuda")
 sm = cfg.D ** -0.5

 triton_fn = lambda: sparse_attn_fwd(q, kv, attn_sink, topk_idxs, sm)
 tilelang_fn = (lambda: _DS_SPARSE_ATTN(q, kv, attn_sink, topk_idxs, sm)) if _DS_SPARSE_ATTN else None

 triton_us = _time_us(triton_fn)
 tilelang_us = _time_us(tilelang_fn) if tilelang_fn else float("nan")

 n_queries = cfg.B * cfg.S
 flops = _flops_per_query(cfg.H, cfg.K, cfg.D) * n_queries
 triton_tflops = flops / (triton_us * 1e-6) / 1e12 if triton_us > 0 else 0
 tilelang_tflops = flops / (tilelang_us * 1e-6) / 1e12 if tilelang_us > 0 else float("nan")

 return {
 "config": cfg,
 "triton_us": triton_us,
 "tilelang_us": tilelang_us,
 "triton_tflops": triton_tflops,
 "tilelang_tflops": tilelang_tflops,
 "speedup": (tilelang_us / triton_us) if (triton_us > 0 and tilelang_us == tilelang_us) else float("nan"),
 }


def bench_bwd(cfg: BenchConfig) -> dict:
 """Backward bench: v1 (atomic FP32 scatter) vs v2 (inverted-topk reduction).
 We don't have a TileLang sparse_mla_bwd plumbed in for cross-comparison;
 full perf comparison vs FlashMLA is the the kernel-level work task."""
 torch.manual_seed(0)
 q = torch.randn(cfg.B, cfg.S, cfg.H, cfg.D, dtype=torch.bfloat16, device="cuda")
 kv = torch.randn(cfg.B, cfg.N_kv, cfg.D, dtype=torch.bfloat16, device="cuda")
 attn_sink = torch.randn(cfg.H, dtype=torch.float32, device="cuda")
 # Realistic top-k: unique-per-query (matches torch.topk).
 rows = []
 for _ in range(cfg.B * cfg.S):
 rows.append(torch.randperm(cfg.N_kv)[: cfg.K])
 topk_idxs = torch.stack(rows).reshape(cfg.B, cfg.S, cfg.K).to(torch.int32).cuda
 do = torch.randn(cfg.B, cfg.S, cfg.H, cfg.D, dtype=torch.bfloat16, device="cuda")
 sm = cfg.D ** -0.5

 o, lse = sparse_attn_fwd(q, kv, attn_sink, topk_idxs, sm)

 bwd_v1_fn = lambda: sparse_attn_bwd(q, kv, attn_sink, topk_idxs, o, lse, do, sm)
 bwd_v2_fn = lambda: sparse_attn_bwd_v2(q, kv, attn_sink, topk_idxs, o, lse, do, sm)

 v1_us = _time_us(bwd_v1_fn)
 v2_us = _time_us(bwd_v2_fn)

 n_queries = cfg.B * cfg.S
 # Backward has ~5x the FMAs of forward (recompute P, plus dP·V, dQ, dK, dV).
 flops = 5 * _flops_per_query(cfg.H, cfg.K, cfg.D) * n_queries
 v1_tflops = flops / (v1_us * 1e-6) / 1e12 if v1_us > 0 else 0
 v2_tflops = flops / (v2_us * 1e-6) / 1e12 if v2_us > 0 else 0
 return {
 "config": cfg,
 "v1_us": v1_us,
 "v2_us": v2_us,
 "v1_tflops": v1_tflops,
 "v2_tflops": v2_tflops,
 "speedup": (v1_us / v2_us) if v2_us > 0 else float("nan"),
 }


def main:
 if not torch.cuda.is_available:
 print("No CUDA — bench skipped.")
 return

 print(f"Device: {torch.cuda.get_device_name}")
 print(f"PyTorch: {torch.__version__}")

 configs = [
 # (B, S, H, D, N_kv, K)
 BenchConfig(B=1, S=128, H=64, D=64, N_kv=512, K=128),
 BenchConfig(B=1, S=512, H=64, D=64, N_kv=2048, K=256),
 BenchConfig(B=1, S=1024, H=64, D=64, N_kv=4096, K=512),
 BenchConfig(B=1, S=2048, H=64, D=64, N_kv=8192, K=1024),
 # H = 128 (V4-Pro)
 BenchConfig(B=1, S=1024, H=128, D=64, N_kv=4096, K=512),
 ]

 print("\n=== FORWARD ===")
 print(
 f"{'config':<48} {'triton (us)':>12} {'tilelang (us)':>14}"
 f" {'triton (TFLOPS)':>17} {'tilelang (TFLOPS)':>19} {'speedup':>9}"
 )
 print("-" * 130)
 for cfg in configs:
 r = bench_fwd(cfg)
 cfg_s = f"B={cfg.B} S={cfg.S} H={cfg.H} D={cfg.D} N_kv={cfg.N_kv} K={cfg.K}"
 print(
 f"{cfg_s:<48} {r['triton_us']:>12.1f} {r['tilelang_us']:>14.1f}"
 f" {r['triton_tflops']:>17.2f} {r['tilelang_tflops']:>19.2f} {r['speedup']:>9.2f}"
 )

 print("\n=== BACKWARD (v1 atomic vs v2 inverted-topk reduction) ===")
 print(
 f"{'config':<48} {'v1 (us)':>10} {'v2 (us)':>10}"
 f" {'v1 (TFLOPS)':>13} {'v2 (TFLOPS)':>13} {'speedup':>9}"
 )
 print("-" * 110)
 for cfg in configs:
 r = bench_bwd(cfg)
 cfg_s = f"B={cfg.B} S={cfg.S} H={cfg.H} D={cfg.D} N_kv={cfg.N_kv} K={cfg.K}"
 print(
 f"{cfg_s:<48} {r['v1_us']:>10.1f} {r['v2_us']:>10.1f}"
 f" {r['v1_tflops']:>13.2f} {r['v2_tflops']:>13.2f} {r['speedup']:>9.2f}"
 )


if __name__ == "__main__":
 main
