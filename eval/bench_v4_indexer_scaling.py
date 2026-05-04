"""V4-Flash Indexer S-scaling — materialize OOM threshold vs chunked.

Run the indexer-step at V4-Flash dims (n_heads=64, head_dim=128, top_k=512,
ratio=4) on a single H200 over a sequence-length sweep, comparing the
materialize+topk reference path against the chunked partition-merge path.

Pre-tested for parity by `eval/test_v4_indexer_parity.py` (100% bit-exact
match at S∈{2K, 4K, 8K} on V4-Flash-realistic distributions).

The result this bench produces is the OOM threshold for the materialize path
and the wall-clock + peak HBM that chunked sustains in the regime materialize
cannot. This is the single-GPU layer-level addition to the kernel-level
results in `benchmarks/results/2026-04-25/RESULTS.md`.

Run on H200:
 python eval/bench_v4_indexer_scaling.py
"""
from __future__ import annotations

import argparse
import sys

import torch
import torch.nn as nn


def _synthesize_inputs(
 *,
 B: int = 1,
 S: int,
 n_heads: int = 64,
 head_dim: int = 128,
 dim: int = 4096,
 ratio: int = 4,
 seed: int = 2026,
 device: str = "cuda",
 dtype=torch.bfloat16,
):
 """Same synthesis pipeline as test_v4_indexer_parity.py — synthetic but
 distribution-matched to V4-Flash post-projection variance."""
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


def _materialize_topk(q, kv_cache, weights, top_k: int, ratio: int):
 """Reference materialize+topk+postmask code path from V4-Flash model.py."""
 B, S, _, _ = q.shape
 T = kv_cache.shape[1]
 index_score = torch.einsum("bshd,btd->bsht", q.float, kv_cache.float)
 index_score = (index_score.relu_ * weights.unsqueeze(-1)).sum(dim=2)
 illegal = (
 torch.arange(T, device=q.device).repeat(S, 1)
 >= torch.arange(1, S + 1, device=q.device).unsqueeze(1) // ratio
 )
 index_score = index_score.masked_fill(illegal.unsqueeze(0), float("-inf"))
 k_eff = min(top_k, T)
 topk_idxs = index_score.topk(k_eff, dim=-1)[1]
 boundary = torch.arange(1, S + 1, device=q.device).unsqueeze(1) // ratio
 invalid_pick = topk_idxs >= boundary.unsqueeze(0)
 topk_idxs = torch.where(invalid_pick, torch.full_like(topk_idxs, -1), topk_idxs)
 return topk_idxs


def _chunked_topk(q, kv_cache, weights, top_k: int, ratio: int,
 chunk_s: int, chunk_t: int):
 """Chunked partition-merge replacement.

 Uses ``causal_ratio`` (per-chunk mask) instead of ``causal_mask``
 (global [B, S, T] bool) — O(chunk_s · chunk_t) mask memory vs O(S · T).
 Critical at long context: at S=1M, T=256K the global mask is 256 GB.
 """
 from flash_sparse.triton.chunked_indexer import chunked_indexer_topk
 top_idx, _ = chunked_indexer_topk(
 q, kv_cache, weights, top_k=top_k,
 causal_ratio=ratio,
 chunk_s=chunk_s, chunk_t=chunk_t,
 )
 return top_idx


def _time_us(fn, n_iter=3, n_warmup=1):
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


def bench_one(S: int, indexer: str, *, n_heads, head_dim, dim, top_k, ratio,
 chunk_s, chunk_t):
 """One S × {materialize, chunked} run. Returns dict with time + peak HBM."""
 try:
 q, kv_cache, weights, T = _synthesize_inputs(
 S=S, n_heads=n_heads, head_dim=head_dim, dim=dim, ratio=ratio,
 )
 if indexer == "materialize":
 fn = lambda: _materialize_topk(q, kv_cache, weights, top_k, ratio)
 elif indexer == "chunked":
 fn = lambda: _chunked_topk(q, kv_cache, weights, top_k, ratio,
 chunk_s=chunk_s, chunk_t=chunk_t)
 else:
 raise ValueError(indexer)

 # Warmup + peak measurement.
 out = fn
 torch.cuda.synchronize
 del out
 torch.cuda.empty_cache
 baseline = torch.cuda.memory_allocated
 torch.cuda.reset_peak_memory_stats
 out = fn
 torch.cuda.synchronize
 peak_gb = max(0, torch.cuda.max_memory_allocated - baseline) / 1024**3
 del out
 torch.cuda.empty_cache

 us = _time_us(fn, n_iter=3, n_warmup=0)
 del q, kv_cache, weights
 torch.cuda.empty_cache
 return {"status": "ok", "us": us, "peak_gb": peak_gb, "T": T}
 except torch.cuda.OutOfMemoryError:
 torch.cuda.empty_cache
 return {"status": "OOM"}
 except RuntimeError as e:
 torch.cuda.empty_cache
 return {"status": "ERROR", "msg": str(e)[:120]}


def main:
 ap = argparse.ArgumentParser
 ap.add_argument("--seq-lens", default="32768,65536,131072,262144,524288,1048576",
 help="Comma-separated S values.")
 ap.add_argument("--n-heads", type=int, default=64)
 ap.add_argument("--head-dim", type=int, default=128)
 ap.add_argument("--top-k", type=int, default=512)
 ap.add_argument("--dim", type=int, default=4096)
 ap.add_argument("--ratio", type=int, default=4)
 ap.add_argument("--chunk-s", type=int, default=2048)
 ap.add_argument("--chunk-t", type=int, default=8192)
 args = ap.parse_args

 if not torch.cuda.is_available:
 print("CUDA required.", file=sys.stderr)
 sys.exit(1)

 print(f"Device: {torch.cuda.get_device_name}")
 print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
 print
 print(f"V4-Flash dims: n_heads={args.n_heads} head_dim={args.head_dim} "
 f"top_k={args.top_k} ratio={args.ratio} dim={args.dim}")
 print(f"Chunked path: chunk_s={args.chunk_s} chunk_t={args.chunk_t}")
 print("=" * 90)
 print(
 f"{'S':>10} {'T':>10}"
 f" {'mat ms':>10} {'mat HBM':>11}"
 f" {'chunk ms':>10} {'chunk HBM':>11}"
 f" {'speedup':>8}"
 )
 print("-" * 90)

 seq_lens = [int(s) for s in args.seq_lens.split(",")]
 for S in seq_lens:
 T = S // args.ratio
 r_mat = bench_one(
 S, "materialize",
 n_heads=args.n_heads, head_dim=args.head_dim, dim=args.dim,
 top_k=args.top_k, ratio=args.ratio,
 chunk_s=args.chunk_s, chunk_t=args.chunk_t,
 )
 torch.cuda.empty_cache
 r_chk = bench_one(
 S, "chunked",
 n_heads=args.n_heads, head_dim=args.head_dim, dim=args.dim,
 top_k=args.top_k, ratio=args.ratio,
 chunk_s=args.chunk_s, chunk_t=args.chunk_t,
 )
 torch.cuda.empty_cache

 def fmt_t(r):
 return f"{r['us']/1000:>10.1f}" if r["status"] == "ok" else f"{r['status']:>10}"

 def fmt_h(r):
 return f"{r['peak_gb']:>8.2f} GB" if r["status"] == "ok" else f"{'-':>11}"

 if r_mat["status"] == "ok" and r_chk["status"] == "ok":
 speedup = r_mat["us"] / r_chk["us"]
 speedup_s = f"{speedup:>7.2f}x"
 elif r_mat["status"] != "ok" and r_chk["status"] == "ok":
 speedup_s = f"{'∞':>8}"
 else:
 speedup_s = f"{'-':>8}"

 print(
 f"{S:>10} {T:>10}"
 f" {fmt_t(r_mat)} {fmt_h(r_mat):>11}"
 f" {fmt_t(r_chk)} {fmt_h(r_chk):>11}"
 f" {speedup_s}"
 )
 if r_mat.get("status") == "ERROR":
 print(f" (mat error: {r_mat.get('msg', '?')})")
 if r_chk.get("status") == "ERROR":
 print(f" (chunk error: {r_chk.get('msg', '?')})")


if __name__ == "__main__":
 main
