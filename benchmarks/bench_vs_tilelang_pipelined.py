"""THE verification experiment — does TileLang's pipelined kernel survive
long context at V4-realistic dims, or does it OOM like the materialize path?

whether the FlashSparse contribution is "we enable a regime they can't"
(paper) or "we're memory-efficient but slower" (blog post).

Three possible outcomes:
 (a) TileLang OOMs at S=256K, V4-Flash-realistic → our story holds.
 (b) TileLang runs but slower than ours → memory-efficiency story holds.
 (c) TileLang runs and is faster → no paper, write a CUDA kernel or stop.

Run as: python benchmarks/bench_vs_tilelang_pipelined.py
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import torch

# Make TileLang's deepseek_v32 examples importable.
_TILELANG_EX = os.path.expanduser(
 "~/flash-sparse/references/tilelang/examples/deepseek_v32"
)
if _TILELANG_EX not in sys.path:
 sys.path.insert(0, _TILELANG_EX)

# Will import lazily inside main to handle missing tilelang gracefully.


@dataclass
class TileLangCfg:
 B: int = 1
 S: int = 4096
 S_kv: int = 4096
 H: int = 64 # V4-Flash
 H_kv: int = 1 # MQA
 dim: int = 512 # V4 head_dim
 tail_dim: int = 64 # V4 RoPE dim (separate in TileLang's layout)
 topk: int = 512
 kv_stride: int = 1


def _build_inputs(cfg: TileLangCfg):
 """Build q, kv, indices, q_start in TileLang's layout (vectorized)."""
 DQK = cfg.dim + cfg.tail_dim
 q = torch.randn(cfg.B, cfg.S, cfg.H, DQK, dtype=torch.bfloat16, device="cuda") * 0.1
 kv = torch.randn(cfg.B, cfg.S_kv, cfg.H_kv, DQK, dtype=torch.bfloat16, device="cuda") * 0.1
 q.clamp_(-10, 10)
 kv.clamp_(-10, 10)

 # Random top-k indices in [0, S_kv); causal clamp per query position.
 # TileLang accepts any indices in [0, S_kv); the exact values don't affect
 # benchmark workload — only the lookup pattern does.
 indices = torch.randint(
 0, cfg.S_kv,
 (cfg.B, cfg.S, cfg.H_kv, cfg.topk),
 dtype=torch.int32, device="cuda",
 )
 t_idx = torch.arange(cfg.S, dtype=torch.int32, device="cuda").view(1, cfg.S, 1, 1)
 indices = torch.minimum(indices, t_idx).contiguous

 # 1-D shape [1] — TileLang's compiled kernel signature requires ndim=1
 # (verified empirically: passing 0-D scalar raises
 # "kernel main input q_start_index_s ndim expected 1, but got 0").
 q_start = torch.tensor([0], dtype=torch.int32, device="cuda")
 return q, kv, indices, q_start


def _time_us(fn, n_iter=5, n_warmup=2):
 for _ in range(n_warmup):
 fn
 torch.cuda.synchronize
 e0 = torch.cuda.Event(enable_timing=True)
 e1 = torch.cuda.Event(enable_timing=True)
 e0.record
 for _ in range(n_iter):
 fn
 e1.record
 torch.cuda.synchronize
 return e0.elapsed_time(e1) * 1000.0 / n_iter # us


def _measure_peak(fn, baseline_after_warmup: bool = True) -> float:
 """Measure peak kernel-only working set (excludes inputs, JIT compile).

 1. Run fn once (compile/JIT warmup, fills caches).
 2. Sync, then capture baseline (inputs + persistent workspace) + reset peak.
 3. Run fn again — delta is per-call kernel peak.
 Returns peak working set in GB.
 """
 if baseline_after_warmup:
 fn
 torch.cuda.synchronize
 baseline = torch.cuda.memory_allocated
 torch.cuda.reset_peak_memory_stats
 fn
 torch.cuda.synchronize
 return max(0, torch.cuda.max_memory_allocated - baseline) / 1024**3


def _validate_output(out, expected_bsh, name: str):
 """Sanity check kernel output: leading (B, S, H) match and finiteness.

 The (B, S, H) prefix is what we care about for silent-truncation detection
 — the trailing D-dim may differ between layouts. Returns None on success,
 error string on failure.
 """
 if out is None:
 return f"{name} returned None"
 o = out[0] if isinstance(out, (tuple, list)) else out
 if not hasattr(o, "shape"):
 return f"{name} returned non-tensor of type {type(o).__name__}"
 if tuple(o.shape[:3]) != tuple(expected_bsh):
 return f"{name} leading shape {tuple(o.shape[:3])} != expected {tuple(expected_bsh)}"
 # Sample-check finiteness: full isfinite on multi-GB tensors is expensive.
 sample = o.flatten[: min(o.numel, 1 << 20)]
 if not torch.isfinite(sample).all:
 return f"{name} produced non-finite values"
 return None


def bench_tilelang(cfg: TileLangCfg) -> dict:
 """Run TileLang pipelined sparse_mla_fwd at the given config."""
 try:
 from sparse_mla_fwd_pipelined import sparse_mla_fwd as tl_sparse_mla_fwd
 except Exception as e:
 return {"status": "import_fail", "msg": str(e)[:120]}

 try:
 q, kv, indices, q_start = _build_inputs(cfg)
 DQK = cfg.dim + cfg.tail_dim

 kernel = tl_sparse_mla_fwd(
 cfg.B, cfg.S, cfg.S_kv, cfg.H, cfg.dim, cfg.tail_dim,
 cfg.topk, cfg.kv_stride, kv_group=cfg.H_kv,
 sm_scale=None, is_causal=True, CP0=True,
 )

 def fn:
 return kernel(q, kv, indices, q_start)

 # Sanity check the warmup output (silent-truncation detector).
 out = fn
 torch.cuda.synchronize
 err = _validate_output(out, (cfg.B, cfg.S, cfg.H), "TileLang")
 if err:
 del q, kv, indices, q_start
 torch.cuda.empty_cache
 return {"status": "INVALID_OUT", "msg": err[:120]}

 peak = _measure_peak(fn, baseline_after_warmup=True)
 us = _time_us(fn, n_warmup=1) # warmup already done above; one more for caches
 del q, kv, indices, q_start, out
 torch.cuda.empty_cache
 # FLOPs: 2x (Q·K^T) + 2x (P·V) ≈ 2 · 2 · S · H · topk · (dim + tail_dim) per batch
 flops = 2 * 2 * cfg.B * cfg.S * cfg.H * cfg.topk * DQK
 tflops = flops / (us * 1e-6) / 1e12
 return {"status": "ok", "us": us, "peak_gb": peak, "tflops": tflops}
 except torch.OutOfMemoryError:
 return {"status": "OOM", "us": float("inf"), "peak_gb": float("nan"), "tflops": 0}
 except RuntimeError as e:
 msg = str(e).lower
 if "invalid argument" in msg or "65535" in msg:
 return {"status": "GRID_LIMIT", "us": float("inf"), "peak_gb": float("nan"), "tflops": 0}
 return {"status": "ERROR", "msg": str(e)[:120]}


def bench_ours(cfg: TileLangCfg) -> dict:
 """Run our Triton sparse_attn at matched dims.

 Layout difference: TileLang uses split (dim=512 + tail_dim=64) MLA layout
 matching V4 weights; our kernel uses single-D layout and requires D to be a
 power of 2. We use D=512 (the closest pow2 ≤ 576) for our side. The FLOP
 count differs by ~11% (D=512 vs 576) — we report TFLOPS using each side's
 actual D so peak utilization is comparable. The headline question is the
 scaling regime (HBM, time vs S), not exact FLOP equality.
 """
 from flash_sparse.triton.sparse_attn_fwd import sparse_attn_fwd

 try:
 D = cfg.dim # 512, pow2-friendly for our kernel
 q = torch.randn(cfg.B, cfg.S, cfg.H, D, dtype=torch.bfloat16, device="cuda") * 0.1
 kv = torch.randn(cfg.B, cfg.S_kv, D, dtype=torch.bfloat16, device="cuda") * 0.1
 attn_sink = torch.zeros(cfg.H, dtype=torch.float32, device="cuda")
 # Vectorized random causal indices.
 indices = torch.randint(
 0, cfg.S_kv,
 (cfg.B, cfg.S, cfg.topk),
 dtype=torch.int32, device="cuda",
 )
 t_idx = torch.arange(cfg.S, dtype=torch.int32, device="cuda").view(1, cfg.S, 1)
 indices = torch.minimum(indices, t_idx).contiguous

 sm = D ** -0.5

 def fn:
 return sparse_attn_fwd(q, kv, attn_sink, indices, sm)

 out = fn
 torch.cuda.synchronize
 err = _validate_output(out, (cfg.B, cfg.S, cfg.H), "ours")
 if err:
 del q, kv, attn_sink, indices
 torch.cuda.empty_cache
 return {"status": "INVALID_OUT", "msg": err[:120]}

 peak = _measure_peak(fn, baseline_after_warmup=True)
 us = _time_us(fn, n_warmup=1)
 del q, kv, attn_sink, indices, out
 torch.cuda.empty_cache
 flops = 2 * 2 * cfg.B * cfg.S * cfg.H * cfg.topk * D
 tflops = flops / (us * 1e-6) / 1e12
 return {"status": "ok", "us": us, "peak_gb": peak, "tflops": tflops, "D": D}
 except torch.OutOfMemoryError:
 return {"status": "OOM", "us": float("inf"), "peak_gb": float("nan"), "tflops": 0}
 except RuntimeError as e:
 return {"status": "ERROR", "msg": str(e)[:120]}


def main:
 if not torch.cuda.is_available:
 return
 print(f"Device: {torch.cuda.get_device_name}")
 print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
 print
 print("V4-Flash-realistic dims: H=64, dim=512, tail_dim=64, topk=512")
 print(" TileLang: D_qk = dim + tail_dim = 576 (split MLA layout)")
 print(" Ours: D = 512 (single-D, pow2-required by Triton kernel)")
 print(" TFLOPS reported per-side using each side's actual D (~11% FLOP-count delta)")
 print("=" * 92)
 print(
 f"{'S':>8}"
 f" {'TileLang time':>14} {'TileLang HBM':>13} {'TileLang TFLOPS':>17}"
 f" | {'ours time':>10} {'ours HBM':>10} {'ours TFLOPS':>13}"
 )
 print("-" * 92)

 configs = [
 TileLangCfg(S=4096, S_kv=4096),
 TileLangCfg(S=16384, S_kv=16384),
 TileLangCfg(S=65536, S_kv=65536),
 TileLangCfg(S=131072, S_kv=131072),
 TileLangCfg(S=262144, S_kv=262144),
 ]

 for cfg in configs:
 r_tl = bench_tilelang(cfg)
 torch.cuda.empty_cache
 r_us = bench_ours(cfg)
 torch.cuda.empty_cache

 def fmt_time(r):
 if r.get("status") == "ok":
 return f"{r['us']/1000:>14.1f}"
 return f"{r.get('status', 'ERR'):>14}"

 def fmt_hbm(r):
 if r.get("status") == "ok":
 return f"{r['peak_gb']:>10.2f} GB"
 return f"{'-':>13}"

 def fmt_tflops(r):
 if r.get("status") == "ok":
 return f"{r['tflops']:>13.1f}"
 return f"{'-':>13}"

 print(
 f"{cfg.S:>8}"
 f" {fmt_time(r_tl)} {fmt_hbm(r_tl):>13} {fmt_tflops(r_tl):>17}"
 f" | {fmt_time(r_us):>10} {fmt_hbm(r_us)} {fmt_tflops(r_us):>13}"
 )
 for label, r in (("TileLang", r_tl), ("ours", r_us)):
 st = r.get("status")
 if st in ("ERROR", "INVALID_OUT", "import_fail"):
 print(f" ({label} {st}: {r.get('msg', '?')})")


if __name__ == "__main__":
 main
