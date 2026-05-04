"""the kernel-level work.a correctness gate for the Triton sparse_attn backward kernel.

Tests:
 test_bwd_matches_reference_autograd
 End-to-end fwd+bwd: gradients from Triton match autograd-through-reference
 within BF16 noise tolerance.
 test_bwd_dq_only
 Isolated dQ check — when only Q has requires_grad.
 test_bwd_dkv_only
 Isolated dKV check.
 test_bwd_dsink_only
 Isolated dsink check (the trickiest — sink gradient is a global per-head
 reduction).
 test_bwd_handles_partial_masking
 Bwd with -1 masked entries scattered through topk_idxs.
"""
from __future__ import annotations

import pytest
import torch

from flash_sparse.reference import reference_sparse_attn
from flash_sparse.triton import sparse_attn


def _normalized_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
 a32 = a.detach.float.reshape(-1)
 b32 = b.detach.float.reshape(-1)
 denom = (a32 * a32 + b32 * b32).sum
 if denom.item == 0.0:
 return 0.0
 sim = 2.0 * (a32 * b32).sum / denom
 return float((1.0 - sim).item)


def _make_inputs(B, S, H, D, N_kv, K, seed=0, mask_every=4, requires_grad=True):
 torch.manual_seed(seed)
 q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda") * 0.5
 kv = torch.randn(B, N_kv, D, dtype=torch.bfloat16, device="cuda") * 0.5
 attn_sink = torch.randn(H, dtype=torch.float32, device="cuda")
 topk_idxs = torch.randint(0, N_kv, (B, S, K), dtype=torch.int32, device="cuda")
 if mask_every > 0:
 topk_idxs[:, :, ::mask_every] = -1
 do = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda") * 0.5
 if requires_grad:
 q.requires_grad_(True)
 kv.requires_grad_(True)
 attn_sink.requires_grad_(True)
 return q, kv, attn_sink, topk_idxs, do


def _bwd_through_reference(q, kv, attn_sink, topk_idxs, do, scale):
 """Run reference forward, then autograd backward. Returns (dq, dkv, dsink)."""
 q_r = q.detach.clone.requires_grad_(True)
 kv_r = kv.detach.clone.requires_grad_(True)
 s_r = attn_sink.detach.clone.requires_grad_(True)
 o_r = reference_sparse_attn(q_r, kv_r, s_r, topk_idxs, scale)
 o_r.backward(do)
 return q_r.grad.detach, kv_r.grad.detach, s_r.grad.detach


def _bwd_through_triton(q, kv, attn_sink, topk_idxs, do, scale):
 """Run Triton forward (autograd-tracked), then backward. Returns (dq, dkv, dsink)."""
 q_t = q.detach.clone.requires_grad_(True)
 kv_t = kv.detach.clone.requires_grad_(True)
 s_t = attn_sink.detach.clone.requires_grad_(True)
 o_t = sparse_attn(q_t, kv_t, s_t, topk_idxs, scale)
 o_t.backward(do)
 return q_t.grad.detach, kv_t.grad.detach, s_t.grad.detach


@pytest.mark.skipif(not torch.cuda.is_available, reason="needs CUDA")
def test_bwd_matches_reference_autograd:
 """End-to-end: dq, dkv, dattn_sink all match reference autograd within BF16."""
 B, S, H, D, N_kv, K = 1, 32, 16, 64, 128, 64
 q, kv, attn_sink, topk_idxs, do = _make_inputs(B, S, H, D, N_kv, K, seed=200)
 sm = D ** -0.5

 dq_t, dkv_t, ds_t = _bwd_through_triton(q, kv, attn_sink, topk_idxs, do, sm)
 dq_r, dkv_r, ds_r = _bwd_through_reference(q, kv, attn_sink, topk_idxs, do, sm)

 diff_dq = _normalized_correlation(dq_t, dq_r)
 diff_dkv = _normalized_correlation(dkv_t, dkv_r)
 diff_ds = _normalized_correlation(ds_t, ds_r)
 print(f"\n[bwd parity] dq={diff_dq:.2e}, dkv={diff_dkv:.2e}, dsink={diff_ds:.2e}")

 assert diff_dq < 1e-2, f"dq mismatch: corr_diff = {diff_dq:.2e}"
 assert diff_dkv < 1e-2, f"dkv mismatch: corr_diff = {diff_dkv:.2e}"
 assert diff_ds < 5e-2, f"dsink mismatch: corr_diff = {diff_ds:.2e}"


@pytest.mark.skipif(not torch.cuda.is_available, reason="needs CUDA")
def test_bwd_dq_only:
 """Only Q requires grad — dQ alone should match."""
 B, S, H, D, N_kv, K = 1, 16, 16, 64, 64, 32
 torch.manual_seed(201)
 q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda") * 0.5
 kv = torch.randn(B, N_kv, D, dtype=torch.bfloat16, device="cuda") * 0.5
 attn_sink = torch.randn(H, dtype=torch.float32, device="cuda")
 topk_idxs = torch.randint(0, N_kv, (B, S, K), dtype=torch.int32, device="cuda")
 do = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
 sm = D ** -0.5

 q_t = q.clone.requires_grad_(True)
 o_t = sparse_attn(q_t, kv, attn_sink, topk_idxs, sm)
 o_t.backward(do)

 q_r = q.clone.requires_grad_(True)
 o_r = reference_sparse_attn(q_r, kv, attn_sink, topk_idxs, sm)
 o_r.backward(do)

 diff = _normalized_correlation(q_t.grad, q_r.grad)
 assert diff < 1e-2, f"dQ mismatch: corr_diff = {diff:.2e}"


@pytest.mark.skipif(not torch.cuda.is_available, reason="needs CUDA")
def test_bwd_dkv_only:
 B, S, H, D, N_kv, K = 1, 16, 16, 64, 64, 32
 torch.manual_seed(202)
 q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda") * 0.5
 kv = torch.randn(B, N_kv, D, dtype=torch.bfloat16, device="cuda") * 0.5
 attn_sink = torch.randn(H, dtype=torch.float32, device="cuda")
 topk_idxs = torch.randint(0, N_kv, (B, S, K), dtype=torch.int32, device="cuda")
 do = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
 sm = D ** -0.5

 kv_t = kv.clone.requires_grad_(True)
 o_t = sparse_attn(q, kv_t, attn_sink, topk_idxs, sm)
 o_t.backward(do)

 kv_r = kv.clone.requires_grad_(True)
 o_r = reference_sparse_attn(q, kv_r, attn_sink, topk_idxs, sm)
 o_r.backward(do)

 diff = _normalized_correlation(kv_t.grad, kv_r.grad)
 assert diff < 1e-2, f"dKV mismatch: corr_diff = {diff:.2e}"


@pytest.mark.skipif(not torch.cuda.is_available, reason="needs CUDA")
def test_bwd_dsink_only:
 B, S, H, D, N_kv, K = 1, 32, 16, 64, 128, 32
 torch.manual_seed(203)
 q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda") * 0.5
 kv = torch.randn(B, N_kv, D, dtype=torch.bfloat16, device="cuda") * 0.5
 attn_sink = torch.randn(H, dtype=torch.float32, device="cuda")
 topk_idxs = torch.randint(0, N_kv, (B, S, K), dtype=torch.int32, device="cuda")
 do = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
 sm = D ** -0.5

 s_t = attn_sink.clone.requires_grad_(True)
 o_t = sparse_attn(q, kv, s_t, topk_idxs, sm)
 o_t.backward(do)

 s_r = attn_sink.clone.requires_grad_(True)
 o_r = reference_sparse_attn(q, kv, s_r, topk_idxs, sm)
 o_r.backward(do)

 diff = _normalized_correlation(s_t.grad, s_r.grad)
 assert diff < 5e-2, f"dsink mismatch: corr_diff = {diff:.2e}"


@pytest.mark.skipif(not torch.cuda.is_available, reason="needs CUDA")
def test_bwd_handles_partial_masking:
 """Backward works with random -1 entries throughout topk_idxs."""
 B, S, H, D, N_kv, K = 2, 64, 16, 64, 256, 64
 q, kv, attn_sink, topk_idxs, do = _make_inputs(B, S, H, D, N_kv, K, seed=204, mask_every=3)
 sm = D ** -0.5

 dq_t, dkv_t, ds_t = _bwd_through_triton(q, kv, attn_sink, topk_idxs, do, sm)
 dq_r, dkv_r, ds_r = _bwd_through_reference(q, kv, attn_sink, topk_idxs, do, sm)

 diff_dq = _normalized_correlation(dq_t, dq_r)
 diff_dkv = _normalized_correlation(dkv_t, dkv_r)
 diff_ds = _normalized_correlation(ds_t, ds_r)
 print(f"\n[bwd partial-mask] dq={diff_dq:.2e}, dkv={diff_dkv:.2e}, dsink={diff_ds:.2e}")

 assert diff_dq < 1e-2
 assert diff_dkv < 1e-2
 assert diff_ds < 5e-2
