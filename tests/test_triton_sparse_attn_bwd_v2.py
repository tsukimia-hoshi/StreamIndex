"""the kernel-level work.b correctness gate for the inverted-topk backward kernel.

Tests that v2 (atomic-free dKV via inverted-topk reduction) produces gradients
equivalent to v1 (atomic FP32 scatter) and to autograd-through-reference.
"""
from __future__ import annotations

import pytest
import torch

from flash_sparse.reference import reference_sparse_attn
from flash_sparse.triton.sparse_attn_fwd import sparse_attn_fwd
from flash_sparse.triton.sparse_attn_bwd import sparse_attn_bwd as bwd_v1
from flash_sparse.triton.sparse_attn_bwd_v2 import sparse_attn_bwd_v2


def _norm_corr(a: torch.Tensor, b: torch.Tensor) -> float:
 a32 = a.detach.float.reshape(-1)
 b32 = b.detach.float.reshape(-1)
 denom = (a32 * a32 + b32 * b32).sum
 if denom.item == 0.0:
 return 0.0
 return float((1.0 - 2.0 * (a32 * b32).sum / denom).item)


def _make_inputs(B, S, H, D, N_kv, K, seed=0, mask_every=4):
 torch.manual_seed(seed)
 q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda") * 0.5
 kv = torch.randn(B, N_kv, D, dtype=torch.bfloat16, device="cuda") * 0.5
 attn_sink = torch.randn(H, dtype=torch.float32, device="cuda")

 # Use realistic top-k (unique-per-query, like torch.topk would produce).
 rows = []
 for _ in range(B * S):
 rows.append(torch.randperm(N_kv)[:K])
 topk_idxs = torch.stack(rows).reshape(B, S, K).to(torch.int32).cuda
 if mask_every > 0:
 topk_idxs[:, :, ::mask_every] = -1

 do = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda") * 0.5
 return q, kv, attn_sink, topk_idxs, do


@pytest.mark.skipif(not torch.cuda.is_available, reason="needs CUDA")
def test_v2_matches_v1:
 """v2 (atomic-free) must produce the same dq/dkv/dsink as v1 (atomic)."""
 B, S, H, D, N_kv, K = 1, 32, 16, 64, 64, 16
 q, kv, attn_sink, topk_idxs, do = _make_inputs(B, S, H, D, N_kv, K, seed=700)
 sm = D ** -0.5

 o, lse = sparse_attn_fwd(q, kv, attn_sink, topk_idxs, sm)

 dq_v1, dkv_v1, ds_v1 = bwd_v1(q, kv, attn_sink, topk_idxs, o, lse, do, sm)
 dq_v2, dkv_v2, ds_v2 = sparse_attn_bwd_v2(q, kv, attn_sink, topk_idxs, o, lse, do, sm)

 diff_dq = _norm_corr(dq_v1, dq_v2)
 diff_dkv = _norm_corr(dkv_v1, dkv_v2)
 diff_ds = _norm_corr(ds_v1, ds_v2)
 print(f"\n[v1 vs v2] dq={diff_dq:.2e}, dkv={diff_dkv:.2e}, dsink={diff_ds:.2e}")

 assert diff_dq < 1e-2, f"dq mismatch: {diff_dq:.2e}"
 assert diff_dkv < 1e-2, f"dkv mismatch: {diff_dkv:.2e}"
 assert diff_ds < 5e-2, f"dsink mismatch: {diff_ds:.2e}"


@pytest.mark.skipif(not torch.cuda.is_available, reason="needs CUDA")
def test_v2_matches_reference_autograd:
 """v2 must match autograd-through-reference within BF16 noise."""
 B, S, H, D, N_kv, K = 1, 32, 16, 64, 64, 16
 q, kv, attn_sink, topk_idxs, do = _make_inputs(B, S, H, D, N_kv, K, seed=701)
 sm = D ** -0.5

 o, lse = sparse_attn_fwd(q, kv, attn_sink, topk_idxs, sm)
 dq_v2, dkv_v2, ds_v2 = sparse_attn_bwd_v2(q, kv, attn_sink, topk_idxs, o, lse, do, sm)

 # Reference path
 q_r = q.detach.clone.requires_grad_(True)
 kv_r = kv.detach.clone.requires_grad_(True)
 s_r = attn_sink.detach.clone.requires_grad_(True)
 o_r = reference_sparse_attn(q_r, kv_r, s_r, topk_idxs, sm)
 o_r.backward(do)

 diff_dq = _norm_corr(dq_v2, q_r.grad)
 diff_dkv = _norm_corr(dkv_v2, kv_r.grad)
 diff_ds = _norm_corr(ds_v2, s_r.grad)
 print(f"\n[v2 vs autograd] dq={diff_dq:.2e}, dkv={diff_dkv:.2e}, dsink={diff_ds:.2e}")

 assert diff_dq < 1e-2
 assert diff_dkv < 1e-2
 assert diff_ds < 5e-2


@pytest.mark.skipif(not torch.cuda.is_available, reason="needs CUDA")
def test_v2_falls_back_for_h_gt_64:
 """The current v2 dKV kernel handles all heads in one CTA; for H > 64 we
 transparently fall back to v1. Verify this works."""
 B, S, H, D, N_kv, K = 1, 16, 128, 64, 64, 16
 q, kv, attn_sink, topk_idxs, do = _make_inputs(B, S, H, D, N_kv, K, seed=702)
 sm = D ** -0.5

 o, lse = sparse_attn_fwd(q, kv, attn_sink, topk_idxs, sm)
 # Should fall back to v1 silently.
 dq_v2, dkv_v2, ds_v2 = sparse_attn_bwd_v2(q, kv, attn_sink, topk_idxs, o, lse, do, sm)
 assert torch.isfinite(dq_v2).all and torch.isfinite(dkv_v2).all
