"""Three ablations of the chunked indexer for paper §6.4.

A1 — "no per-chunk merge". After computing per-chunk top-k, do NOT re-merge
 across chunks. Instead, keep early chunks' entries and truncate at top_k
 at the end. Should crater recall: queries whose true top entries come
 from later chunks lose those entries.

A2 — "skip saturated chunks". When ``chunk_T < top_k`` (the saturated branch
 in `chunked_indexer.py:108-110` where each chunk's topk(min(top_k, chunk_T))
 returns ALL chunk entries), skip the chunk entirely instead. Tests
 whether the saturated branch is doing real work.

A3 — "FP16 score accumulation". Cast per-chunk scores to FP16 before topk,
 and keep the running buffer in FP16. The production path uses FP32.
 Tests the precision floor.

For each, we run at small_s=16384 (T=4096) where materialize is the bit-
exact ground truth, with V4-Flash dims (n_heads=64, head_dim=128, top_k=512,
ratio=4). Compute mean/min set-overlap recall vs materialize. Production
chunked path's recall (1.0/0.998) is the reference upper bound.

We also report wall-clock at big_s=262144 for A3 (FP16) to see if the
precision drop buys speed. A1/A2 wall-clock isn't interesting (they're
broken-by-design variants, not production candidates).
"""
from __future__ import annotations

import argparse
import sys
from typing import Tuple

import torch
import torch.nn as nn

from flash_sparse.triton.indexer_score import indexer_score
from flash_sparse.triton.chunked_indexer import chunked_indexer_topk


# --------------------------------------------------------------------------
# Synthesis (same as test_v4_indexer_parity.py)
# --------------------------------------------------------------------------
def _synthesize_inputs(*, B=1, S, n_heads=64, head_dim=128, dim=4096,
 ratio=4, seed=2026, device="cuda", dtype=torch.bfloat16):
 g = torch.Generator(device=device).manual_seed(seed)
 T = S // ratio
 softmax_scale = head_dim ** -0.5
 q = torch.randn(B, S, n_heads, head_dim,
 generator=g, device=device, dtype=dtype) * (head_dim ** -0.5)
 kv_cache = torch.randn(B, T, head_dim,
 generator=g, device=device, dtype=dtype) * (head_dim ** -0.5)
 torch.manual_seed(seed)
 x = torch.randn(B, S, dim, generator=g, device=device, dtype=dtype)
 weights_proj = nn.Linear(dim, n_heads, bias=False, dtype=dtype, device=device)
 with torch.no_grad:
 weights = weights_proj(x).float * (softmax_scale * (n_heads ** -0.5))
 return q, kv_cache, weights, T


def _materialize_topk(q, kv_cache, weights, top_k, ratio):
 B, S, _, _ = q.shape
 T = kv_cache.shape[1]
 score = torch.einsum("bshd,btd->bsht", q.float, kv_cache.float)
 score = (score.relu_ * weights.unsqueeze(-1)).sum(dim=2)
 illegal = (
 torch.arange(T, device=q.device).repeat(S, 1)
 >= torch.arange(1, S + 1, device=q.device).unsqueeze(1) // ratio
 )
 score = score.masked_fill(illegal.unsqueeze(0), float("-inf"))
 k_eff = min(top_k, T)
 topk = score.topk(k_eff, dim=-1)[1]
 boundary = torch.arange(1, S + 1, device=q.device).unsqueeze(1) // ratio
 invalid = topk >= boundary.unsqueeze(0)
 return torch.where(invalid, torch.full_like(topk, -1), topk)


# --------------------------------------------------------------------------
# Production reference (uses our shipped path)
# --------------------------------------------------------------------------
def _production(q, kv_cache, weights, top_k, ratio, chunk_s, chunk_t):
 out, _ = chunked_indexer_topk(
 q, kv_cache, weights, top_k=top_k,
 causal_ratio=ratio, chunk_s=chunk_s, chunk_t=chunk_t,
 )
 return out


# --------------------------------------------------------------------------
# A1 — no per-chunk merge
# --------------------------------------------------------------------------
def _ablation_no_merge(q, kv_cache, weights, top_k, ratio, chunk_s, chunk_t):
 """Keep only the FIRST chunk's contribution per query (no merge).

 Concretely: compute per-chunk top-k as the production path would, but
 do not merge across chunks — simply keep whatever entries the FIRST
 T-chunk produces for each S-chunk. Later T-chunks are ignored.
 """
 B, S, H_I, D_I = q.shape
 T = kv_cache.shape[1]
 out_idx = torch.full((B, S, top_k), -1, dtype=torch.int64, device=q.device)

 for s_start in range(0, S, chunk_s):
 s_end = min(s_start + chunk_s, S)
 q_s = q[:, s_start:s_end].contiguous
 w_s = weights[:, s_start:s_end].contiguous
 chunk_S = s_end - s_start
 # ONLY first T-chunk
 t_start = 0
 t_end = min(chunk_t, T)
 k_t = kv_cache[:, t_start:t_end].contiguous
 chunk_T = t_end - t_start
 scores = indexer_score(q_s, k_t, w_s)
 # Apply causal mask per chunk.
 s_idx = torch.arange(s_start, s_end, device=q.device).unsqueeze(1)
 t_idx = torch.arange(t_start, t_end, device=q.device).unsqueeze(0)
 legal = t_idx < (s_idx + 1) // ratio
 scores = scores.masked_fill(~legal.unsqueeze(0), float("-inf"))
 k_chunk = min(top_k, chunk_T)
 chunk_v, chunk_i = scores.topk(k_chunk, dim=-1)
 chunk_i = chunk_i + t_start
 # Mask out -inf entries.
 chunk_i = torch.where(torch.isinf(chunk_v),
 torch.full_like(chunk_i, -1), chunk_i)
 out_idx[:, s_start:s_end, :k_chunk] = chunk_i.to(torch.int64)
 return out_idx


# --------------------------------------------------------------------------
# A2 — skip saturated chunks
# --------------------------------------------------------------------------
def _ablation_skip_saturated(q, kv_cache, weights, top_k, ratio,
 chunk_s, chunk_t):
 """When chunk_T (the actual T-block size) < top_k, skip the chunk entirely
 instead of including all its entries.

 Production behavior in `chunked_indexer.py` is to take
 ``k_chunk = min(top_k, chunk_T)`` — so a chunk smaller than top_k
 contributes ALL its entries. The ablation skips them.
 """
 B, S, H_I, D_I = q.shape
 T = kv_cache.shape[1]
 out_idx = torch.full((B, S, top_k), -1, dtype=torch.int64, device=q.device)

 for s_start in range(0, S, chunk_s):
 s_end = min(s_start + chunk_s, S)
 q_s = q[:, s_start:s_end].contiguous
 w_s = weights[:, s_start:s_end].contiguous
 chunk_S = s_end - s_start

 run_v = torch.full((B, chunk_S, top_k), float("-inf"),
 dtype=torch.float32, device=q.device)
 run_i = torch.full((B, chunk_S, top_k), -1,
 dtype=torch.int64, device=q.device)

 for t_start in range(0, T, chunk_t):
 t_end = min(t_start + chunk_t, T)
 chunk_T = t_end - t_start
 # Skip chunks smaller than top_k (the saturated branch).
 if chunk_T < top_k:
 continue
 k_t = kv_cache[:, t_start:t_end].contiguous
 scores = indexer_score(q_s, k_t, w_s)
 s_idx = torch.arange(s_start, s_end, device=q.device).unsqueeze(1)
 t_idx = torch.arange(t_start, t_end, device=q.device).unsqueeze(0)
 legal = t_idx < (s_idx + 1) // ratio
 scores = scores.masked_fill(~legal.unsqueeze(0), float("-inf"))
 chunk_v, chunk_i = scores.topk(top_k, dim=-1)
 chunk_i = chunk_i + t_start
 combined_v = torch.cat([run_v, chunk_v], dim=-1)
 combined_i = torch.cat([run_i, chunk_i.to(torch.int64)], dim=-1)
 run_v, perm = combined_v.topk(top_k, dim=-1)
 run_i = combined_i.gather(-1, perm)
 run_i = torch.where(torch.isinf(run_v),
 torch.full_like(run_i, -1), run_i)
 out_idx[:, s_start:s_end] = run_i
 return out_idx


# --------------------------------------------------------------------------
# A3 — FP16 score accumulation
# --------------------------------------------------------------------------
def _ablation_fp16(q, kv_cache, weights, top_k, ratio, chunk_s, chunk_t):
 """Cast per-chunk scores and the running buffer to FP16 before topk.
 Production path uses FP32. Tests precision floor.
 """
 B, S, H_I, D_I = q.shape
 T = kv_cache.shape[1]
 out_idx = torch.full((B, S, top_k), -1, dtype=torch.int64, device=q.device)

 for s_start in range(0, S, chunk_s):
 s_end = min(s_start + chunk_s, S)
 q_s = q[:, s_start:s_end].contiguous
 w_s = weights[:, s_start:s_end].contiguous
 chunk_S = s_end - s_start

 run_v = torch.full((B, chunk_S, top_k), float("-inf"),
 dtype=torch.float16, device=q.device)
 run_i = torch.full((B, chunk_S, top_k), -1,
 dtype=torch.int64, device=q.device)

 for t_start in range(0, T, chunk_t):
 t_end = min(t_start + chunk_t, T)
 chunk_T = t_end - t_start
 k_t = kv_cache[:, t_start:t_end].contiguous
 scores = indexer_score(q_s, k_t, w_s).to(torch.float16)
 s_idx = torch.arange(s_start, s_end, device=q.device).unsqueeze(1)
 t_idx = torch.arange(t_start, t_end, device=q.device).unsqueeze(0)
 legal = t_idx < (s_idx + 1) // ratio
 scores = scores.masked_fill(~legal.unsqueeze(0), float("-inf"))
 k_chunk = min(top_k, chunk_T)
 chunk_v, chunk_i = scores.topk(k_chunk, dim=-1)
 chunk_i = chunk_i + t_start
 combined_v = torch.cat([run_v, chunk_v], dim=-1)
 combined_i = torch.cat([run_i, chunk_i.to(torch.int64)], dim=-1)
 run_v, perm = combined_v.topk(top_k, dim=-1)
 run_i = combined_i.gather(-1, perm)
 run_i = torch.where(torch.isinf(run_v.float),
 torch.full_like(run_i, -1), run_i)
 out_idx[:, s_start:s_end] = run_i
 return out_idx


# --------------------------------------------------------------------------
# Recall + timing
# --------------------------------------------------------------------------
def _set_overlap_recall(idx_a, idx_b):
 B, S, _ = idx_a.shape
 sa = idx_a.clone.to(torch.int64)
 sb = idx_b.clone.to(torch.int64)
 valid_a = (sa >= 0)
 valid_b = (sb >= 0)
 n_valid_a = valid_a.sum(dim=-1)
 n_valid_b = valid_b.sum(dim=-1)
 sa = torch.where(valid_a, sa, torch.full_like(sa, -2))
 sb = torch.where(valid_b, sb, torch.full_like(sb, -3))
 overlap = torch.zeros((B, S), dtype=torch.float32, device=idx_a.device)
 chunk = 512
 for s0 in range(0, S, chunk):
 s1 = min(s0 + chunk, S)
 a = sa[:, s0:s1]
 b = sb[:, s0:s1]
 match = (a.unsqueeze(-1) == b.unsqueeze(-2)) & valid_a[:, s0:s1].unsqueeze(-1)
 overlap[:, s0:s1] = match.any(dim=-1).sum(dim=-1).float
 both_empty = (n_valid_a == 0) & (n_valid_b == 0)
 a_empty_b_nonempty = (n_valid_a == 0) & (n_valid_b > 0)
 denom = n_valid_a.clamp(min=1).float
 rec = overlap / denom
 rec = torch.where(both_empty, torch.ones_like(rec), rec)
 rec = torch.where(a_empty_b_nonempty, torch.zeros_like(rec), rec)
 return rec


def _time_us(fn, n_iter=3, n_warmup=2):
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
 return e0.elapsed_time(e1) * 1000.0 / n_iter


def _measure_peak(fn):
 fn
 torch.cuda.synchronize
 torch.cuda.empty_cache
 base = torch.cuda.memory_allocated
 torch.cuda.reset_peak_memory_stats
 fn
 torch.cuda.synchronize
 return max(0, torch.cuda.max_memory_allocated - base) / 1024**3


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main:
 ap = argparse.ArgumentParser
 ap.add_argument("--small-s", type=int, default=16384)
 ap.add_argument("--big-s", type=int, default=262144)
 ap.add_argument("--n-heads", type=int, default=64)
 ap.add_argument("--head-dim", type=int, default=128)
 ap.add_argument("--top-k", type=int, default=512)
 ap.add_argument("--dim", type=int, default=4096)
 ap.add_argument("--ratio", type=int, default=4)
 ap.add_argument("--chunk-s", type=int, default=2048)
 ap.add_argument("--chunk-t", type=int, default=8192)
 ap.add_argument("--saturated-chunk-t", type=int, default=256,
 help="Small chunk_T < top_k for A2 (saturated branch test).")
 args = ap.parse_args

 if not torch.cuda.is_available:
 print("CUDA required.", file=sys.stderr); sys.exit(1)

 print(f"Device: {torch.cuda.get_device_name}")
 print(f"V4-Flash dims: n_heads={args.n_heads} head_dim={args.head_dim} "
 f"top_k={args.top_k} ratio={args.ratio}")
 print(f"Recall test at S={args.small_s} (T={args.small_s // args.ratio}); "
 f"timing test at S={args.big_s}.")
 print

 # Build small-S inputs once for all recall tests.
 q_s, kv_s, w_s, T_s = _synthesize_inputs(
 S=args.small_s, n_heads=args.n_heads, head_dim=args.head_dim,
 dim=args.dim, ratio=args.ratio,
 )
 idx_mat = _materialize_topk(q_s, kv_s, w_s, args.top_k, args.ratio)

 # ----------------------------------------------------------------------
 # Recall comparisons at small_s
 # ----------------------------------------------------------------------
 print("=" * 80)
 print("RECALL ABLATIONS (small_s, vs materialize ground truth)")
 print("=" * 80)
 print(f"{'variant':<32} {'mean rec':>9} {'min rec':>9} {'%=1':>7}")
 print("-" * 80)

 # Production reference
 idx_prod = _production(q_s, kv_s, w_s, args.top_k, args.ratio,
 args.chunk_s, args.chunk_t)
 rec = _set_overlap_recall(idx_mat, idx_prod)
 print(f"{'production (FP32, full merge)':<32} "
 f"{float(rec.mean):>9.4f} {float(rec.min):>9.4f} "
 f"{float((rec == 1.0).float.mean) * 100:>6.2f}%")

 # A1 — no merge (first-chunk only)
 idx_a1 = _ablation_no_merge(q_s, kv_s, w_s, args.top_k, args.ratio,
 args.chunk_s, args.chunk_t)
 rec = _set_overlap_recall(idx_mat, idx_a1)
 print(f"{'A1: no per-chunk merge':<32} "
 f"{float(rec.mean):>9.4f} {float(rec.min):>9.4f} "
 f"{float((rec == 1.0).float.mean) * 100:>6.2f}%")

 # A2 — skip saturated chunks. Use chunk_t < top_k to trigger saturation.
 idx_a2 = _ablation_skip_saturated(q_s, kv_s, w_s, args.top_k, args.ratio,
 args.chunk_s, args.saturated_chunk_t)
 rec = _set_overlap_recall(idx_mat, idx_a2)
 print(f"{f'A2: skip saturated (ct={args.saturated_chunk_t})':<32} "
 f"{float(rec.mean):>9.4f} {float(rec.min):>9.4f} "
 f"{float((rec == 1.0).float.mean) * 100:>6.2f}%")

 # A2-control — production at the same small chunk_t (should give 100%)
 idx_a2_ctrl = _production(q_s, kv_s, w_s, args.top_k, args.ratio,
 args.chunk_s, args.saturated_chunk_t)
 rec = _set_overlap_recall(idx_mat, idx_a2_ctrl)
 print(f"{f'A2-ctrl: production (ct={args.saturated_chunk_t})':<32} "
 f"{float(rec.mean):>9.4f} {float(rec.min):>9.4f} "
 f"{float((rec == 1.0).float.mean) * 100:>6.2f}%")

 # A3 — FP16 accumulation
 idx_a3 = _ablation_fp16(q_s, kv_s, w_s, args.top_k, args.ratio,
 args.chunk_s, args.chunk_t)
 rec = _set_overlap_recall(idx_mat, idx_a3)
 print(f"{'A3: FP16 score accumulation':<32} "
 f"{float(rec.mean):>9.4f} {float(rec.min):>9.4f} "
 f"{float((rec == 1.0).float.mean) * 100:>6.2f}%")

 del q_s, kv_s, w_s, idx_mat, idx_prod, idx_a1, idx_a2, idx_a2_ctrl, idx_a3
 torch.cuda.empty_cache

 # ----------------------------------------------------------------------
 # Speed comparison: A3 (FP16) vs production at big_s
 # ----------------------------------------------------------------------
 print
 print("=" * 80)
 print(f"WALL-CLOCK ABLATIONS at S={args.big_s}")
 print("=" * 80)
 print(f"{'variant':<32} {'time (ms)':>10} {'peak HBM':>11}")
 print("-" * 80)

 q_b, kv_b, w_b, T_b = _synthesize_inputs(
 S=args.big_s, n_heads=args.n_heads, head_dim=args.head_dim,
 dim=args.dim, ratio=args.ratio,
 )

 fn_prod = lambda: _production(q_b, kv_b, w_b, args.top_k, args.ratio,
 args.chunk_s, args.chunk_t)
 peak_prod = _measure_peak(fn_prod)
 us_prod = _time_us(fn_prod)
 print(f"{'production (FP32)':<32} {us_prod/1000:>10.1f} {peak_prod:>9.2f} GB")

 fn_a3 = lambda: _ablation_fp16(q_b, kv_b, w_b, args.top_k, args.ratio,
 args.chunk_s, args.chunk_t)
 peak_a3 = _measure_peak(fn_a3)
 us_a3 = _time_us(fn_a3)
 print(f"{'A3: FP16':<32} {us_a3/1000:>10.1f} {peak_a3:>9.2f} GB")
 print(f"{' (speedup vs prod)':<32} {us_prod/us_a3:>10.2f}x {(peak_prod/peak_a3 if peak_a3 else 0):>7.2f}x lower peak")


if __name__ == "__main__":
 main
