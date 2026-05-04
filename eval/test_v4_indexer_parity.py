"""V4-Flash Indexer parity test — chunked path vs reference materialize path.

**Scope (honest)**: this is a *synthetic-but-realistic algorithmic parity*
test at V4-Flash dims (H=64, head_dim=128, top_k=512, ratio=4). It does NOT
load real V4-Flash checkpoint weights or run the real Compressor pipeline;
it draws q, kv_cache, and weights from distributions matching the shapes
and per-element variance the real pipeline produces.

Reproduces the EXACT score-then-topk-then-postmask code path of
`Indexer.forward` in `references/DeepSeek-V4-Flash/inference/model.py`,
lines ~415-431:

 index_score = torch.einsum("bshd,btd->bsht", q, kv_cache_slice)
 index_score = (index_score.relu_ * weights.unsqueeze(-1)).sum(dim=2)
 if start_pos == 0:
 mask = (torch.arange(T).repeat(S, 1)
 >= torch.arange(1, S + 1).unsqueeze(1) // ratio)
 index_score += torch.where(mask, float("-inf"), 0)
 topk_idxs = index_score.topk(min(top_k, T), dim=-1)[1]
 if start_pos == 0:
 invalid = topk_idxs >= torch.arange(1, S + 1).unsqueeze(1) // ratio
 topk_idxs = torch.where(invalid, -1, topk_idxs) # post-mask is critical

The chunked path uses our `flash_sparse.triton.chunked_indexer.chunked_indexer_topk`
with the same causal mask. Pass criterion is bit-exact set match on every
valid (non-pad) row at V4-Flash dims — earlier kernel-level recall study
showed 100% recall, so any drop here is a real regression.

Run on H200:
 python eval/test_v4_indexer_parity.py [--seq-lens 2048,4096,8192]
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn

# --------------------------------------------------------------------------
# Realistic input synthesis matching V4-Flash's Indexer.forward
# --------------------------------------------------------------------------
# We replicate just the projections/transforms that produce (q, kv_cache,
# weights) from a hidden state x. This is the upstream of model.py:415-423.
# We do NOT instantiate the full Indexer because (a) it depends on
# fast_hadamard_transform + fp4_act_quant + the Compressor which has many
# more deps; (b) for the indexer-step parity test, what matters are the
# *distributions* of (q, kv_cache, weights), not the exact rotation. The
# claim under test is "materialize → topk and chunked → topk produce the
# same top-k SET on V4-Flash-realistic distributions."
#
# We approximate the upstream by:
# - q from nn.Linear(q_lora_rank, n_heads*head_dim) of a random qr.
# V4 also applies RoPE on the last 64 dims and a Hadamard rotation;
# for the indexer-step we test the score *after* those transforms,
# so we approximate by drawing q ~ N(0, 1/head_dim**0.5).
# - kv_cache: produced by the Compressor; its output is roughly
# N(0, 1) with a 1/sqrt(d) normalization. Approximate by N(0, 1/head_dim**0.5).
# - weights: produced by nn.Linear(dim, n_heads), then scaled by
# softmax_scale * n_heads**-0.5. We use the actual nn.Linear with
# bias=False (matching V4) to get the realistic distribution.
# --------------------------------------------------------------------------


def _synthesize_indexer_inputs(
 *,
 B: int = 1,
 S: int = 2048,
 n_heads: int = 64,
 head_dim: int = 128,
 dim: int = 4096,
 ratio: int = 4,
 seed: int = 2026,
 device: str = "cuda",
 dtype=torch.bfloat16,
):
 """Return (q, kv_cache, weights, T) with synthetic distributions
 approximating V4-Flash post-projection variance.

 Caveat (per code review): this is *synthetic*. q skips the real
 wq_b → RoPE → Hadamard → FP4-quant pipeline; kv_cache skips the real
 Compressor (conv + softmax pool + RMSNorm + RoPE); weights uses a
 fresh-init Linear instead of trained checkpoint weights. The inputs
 match per-element variance scaling (dot products land in [-O(1), O(1)])
 but not the structured correlations a real model produces.
 """
 g = torch.Generator(device=device).manual_seed(seed)
 T = S // ratio
 softmax_scale = head_dim ** -0.5

 q = (
 torch.randn(B, S, n_heads, head_dim, generator=g, device=device, dtype=dtype)
 * (head_dim ** -0.5)
 )

 kv_cache = (
 torch.randn(B, T, head_dim, generator=g, device=device, dtype=dtype)
 * (head_dim ** -0.5)
 )

 # Deterministically initialize weights_proj. PyTorch's default Linear init
 # uses generator=None so we set the global seed in addition to passing g.
 torch.manual_seed(seed)
 x = torch.randn(B, S, dim, generator=g, device=device, dtype=dtype)
 weights_proj = nn.Linear(dim, n_heads, bias=False, dtype=dtype, device=device)
 with torch.no_grad:
 weights = weights_proj(x).float * (softmax_scale * (n_heads ** -0.5))

 return q, kv_cache, weights, T


# --------------------------------------------------------------------------
# Reference and chunked paths — identical to model.py:415-423
# --------------------------------------------------------------------------
def _materialize_topk(q, kv_cache, weights, top_k: int, ratio: int):
 """Replicates Indexer.forward materialize+topk+postmask for start_pos==0.

 Critical: the reference (V4-Flash model.py ~lines 415-431) post-masks
 invalid topk indices to -1. Without that step, top-k slots that came
 from -inf-scored positions return arbitrary indices and would falsely
 deflate set-overlap recall vs the chunked path.
 """
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

 # Post-mask: indices selected from -inf positions are invalid.
 # Reference: topk_idxs >= floor((s+1)/ratio) → set to -1.
 boundary = torch.arange(1, S + 1, device=q.device).unsqueeze(1) // ratio
 invalid_pick = topk_idxs >= boundary.unsqueeze(0) # [B, S, k_eff]
 topk_idxs = torch.where(invalid_pick, torch.full_like(topk_idxs, -1), topk_idxs)
 return topk_idxs


def _chunked_topk(q, kv_cache, weights, top_k: int, ratio: int,
 chunk_s: int = 2048, chunk_t: int = 8192):
 """Drop-in replacement using flash_sparse.triton.chunked_indexer.

 Uses ``causal_ratio`` for memory-efficient per-chunk masking. The parity
 test exercises both small S (where global causal_mask would also fit)
 and the long-context mode the bench will exercise.
 """
 from flash_sparse.triton.chunked_indexer import chunked_indexer_topk

 top_idx, _ = chunked_indexer_topk(
 q, kv_cache, weights, top_k=top_k,
 causal_ratio=ratio,
 chunk_s=chunk_s, chunk_t=chunk_t,
 )
 return top_idx


# --------------------------------------------------------------------------
# Set-overlap recall
# --------------------------------------------------------------------------
def _set_overlap_recall(idx_a: torch.Tensor, idx_b: torch.Tensor) -> torch.Tensor:
 """Per-query |valid_a ∩ valid_b| / |valid_a|. Negative entries are
 padding sentinels excluded from both sets.

 Returns shape [B, S]. Rows where |valid_a| == 0 AND |valid_b| == 0 — both
 paths agree there are no legal entries to choose, vacuously match — get
 recall = 1.0. Rows where |valid_a| == 0 but |valid_b| > 0 (chunked
 produces phantom valid entries materialize doesn't) get recall = 0.
 """
 B, S, K = idx_a.shape
 sa = idx_a.clone.to(torch.int64)
 sb = idx_b.clone.to(torch.int64)
 valid_a = (sa >= 0)
 valid_b = (sb >= 0)
 n_valid_a = valid_a.sum(dim=-1) # [B, S]
 n_valid_b = valid_b.sum(dim=-1) # [B, S]

 sa = torch.where(valid_a, sa, torch.full_like(sa, -2))
 sb = torch.where(valid_b, sb, torch.full_like(sb, -3))

 overlap_count = torch.zeros((B, S), dtype=torch.float32, device=idx_a.device)
 chunk = 512
 for s0 in range(0, S, chunk):
 s1 = min(s0 + chunk, S)
 a = sa[:, s0:s1]
 b = sb[:, s0:s1]
 match = (a.unsqueeze(-1) == b.unsqueeze(-2)) & valid_a[:, s0:s1].unsqueeze(-1)
 overlap_count[:, s0:s1] = match.any(dim=-1).sum(dim=-1).float

 both_empty = (n_valid_a == 0) & (n_valid_b == 0)
 a_empty_b_nonempty = (n_valid_a == 0) & (n_valid_b > 0)
 denom = n_valid_a.clamp(min=1).float
 recall = overlap_count / denom
 recall = torch.where(both_empty, torch.ones_like(recall), recall)
 recall = torch.where(a_empty_b_nonempty, torch.zeros_like(recall), recall)
 return recall


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main:
 ap = argparse.ArgumentParser
 ap.add_argument("--seq-lens", default="2048,4096,8192",
 help="Comma-separated S values to test.")
 ap.add_argument("--ratio", type=int, default=4)
 ap.add_argument("--n-heads", type=int, default=64)
 ap.add_argument("--head-dim", type=int, default=128)
 ap.add_argument("--top-k", type=int, default=512)
 ap.add_argument("--dim", type=int, default=4096)
 ap.add_argument("--chunk-s", type=int, default=2048)
 ap.add_argument("--chunk-t", type=int, default=8192)
 ap.add_argument("--threshold", type=float, default=1.0,
 help="Pass criterion: minimum mean recall (default: bit-exact, 1.0).")
 ap.add_argument("--require-perfect-min", action="store_true", default=True,
 help="Also require min recall == 1.0 (kernel-level study showed 100%).")
 args = ap.parse_args

 if not torch.cuda.is_available:
 print("CUDA required.", file=sys.stderr)
 sys.exit(1)

 print(f"Device: {torch.cuda.get_device_name}")
 print(f"V4-Flash dims: n_heads={args.n_heads} head_dim={args.head_dim} "
 f"top_k={args.top_k} ratio={args.ratio} dim={args.dim}")
 print(f"Pass threshold: mean recall ≥ {args.threshold}")
 print(f"{'S':>8} {'T':>8} {'mean':>8} {'min':>8} {'%=1':>7} {'%<.99':>7} {'verdict':>9}")
 print("-" * 60)

 seq_lens = [int(s) for s in args.seq_lens.split(",")]
 overall_pass = True
 for S in seq_lens:
 q, kv_cache, weights, T = _synthesize_indexer_inputs(
 S=S, n_heads=args.n_heads, head_dim=args.head_dim,
 dim=args.dim, ratio=args.ratio,
 )
 idx_mat = _materialize_topk(q, kv_cache, weights, args.top_k, args.ratio)
 idx_chk = _chunked_topk(
 q, kv_cache, weights, args.top_k, args.ratio,
 chunk_s=args.chunk_s, chunk_t=args.chunk_t,
 )
 # Both should produce [B, S, k_eff] where k_eff = min(top_k, T).
 # If they differ in shape (e.g. chunked padded vs materialize trimmed),
 # truncate to the common min for the overlap calc.
 K = min(idx_mat.shape[-1], idx_chk.shape[-1])
 rec = _set_overlap_recall(idx_mat[..., :K], idx_chk[..., :K])
 mean_r = float(rec.mean)
 min_r = float(rec.min)
 pct_perfect = float((rec == 1.0).float.mean) * 100
 pct_bad = float((rec < 0.99).float.mean) * 100
 # Strict gate: bit-exact match required (mean and min == 1.0).
 passed = (mean_r >= args.threshold) and (
 (not args.require_perfect_min) or (min_r >= 1.0 - 1e-6)
 )
 verdict = "PASS" if passed else "FAIL"
 if not passed:
 overall_pass = False
 print(f"{S:>8} {T:>8} {mean_r:>8.4f} {min_r:>8.4f} "
 f"{pct_perfect:>6.2f}% {pct_bad:>6.3f}% {verdict:>9}")
 del q, kv_cache, weights, idx_mat, idx_chk, rec
 torch.cuda.empty_cache

 print
 if overall_pass:
 print("OVERALL: PASS — chunked indexer is a drop-in replacement on V4-Flash dims.")
 sys.exit(0)
 else:
 print("OVERALL: FAIL — see per-row verdicts.")
 sys.exit(2)


if __name__ == "__main__":
 main
