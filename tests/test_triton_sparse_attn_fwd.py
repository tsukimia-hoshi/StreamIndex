"""the kernel-level work.a correctness gate for the Triton sparse_attn forward kernel.

Three tests:
 test_triton_dense_fallthrough — when topk_idxs covers all of kv (no -1), Triton
 matches a manual dense-attention computation.
 test_triton_matches_reference — Triton matches reference_sparse_attn (BF16 noise).
 test_triton_matches_tilelang — Triton matches DeepSeek's TileLang sparse_attn
 (the absolute ground truth).
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

from flash_sparse.reference import reference_sparse_attn
from flash_sparse.triton.sparse_attn_fwd import sparse_attn_fwd as triton_sparse_attn_fwd


# Try to import the DeepSeek TileLang reference.
_INFERENCE_DIRS = [
 os.path.expanduser("~/flash-sparse/references/DeepSeek-V4-Pro/inference"),
 os.path.normpath(
 os.path.join(
 os.path.dirname(__file__),
 "..",
 "..",
 "references",
 "DeepSeek-V4-Pro",
 "inference",
 )
 ),
]


def _import_deepseek_kernel:
 for d in _INFERENCE_DIRS:
 if not os.path.isdir(d):
 continue
 if d not in sys.path:
 sys.path.insert(0, d)
 try:
 import importlib

 ds_kernel = importlib.import_module("kernel")
 return getattr(ds_kernel, "sparse_attn", None)
 except Exception:
 continue
 return None


_DS_SPARSE_ATTN = _import_deepseek_kernel
HAVE_DEEPSEEK = _DS_SPARSE_ATTN is not None


def _normalized_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
 a32 = a.detach.float.reshape(-1)
 b32 = b.detach.float.reshape(-1)
 denom = (a32 * a32 + b32 * b32).sum
 if denom.item == 0.0:
 return 0.0
 sim = 2.0 * (a32 * b32).sum / denom
 return float((1.0 - sim).item)


@pytest.mark.skipif(not torch.cuda.is_available, reason="needs CUDA")
def test_triton_dense_fallthrough:
 """When topk_idxs covers all of kv with no masking, Triton must match a manual
 sink-augmented dense softmax.
 """
 torch.manual_seed(100)
 B, S, H, D, N_kv = 1, 16, 16, 64, 64
 q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda") * 0.5
 kv = torch.randn(B, N_kv, D, dtype=torch.bfloat16, device="cuda") * 0.5
 attn_sink = torch.randn(H, dtype=torch.float32, device="cuda")
 topk_idxs = (
 torch.arange(N_kv, device="cuda")
 .view(1, 1, N_kv)
 .expand(B, S, N_kv)
 .contiguous
 .int
 )
 sm = D ** -0.5

 o_triton, lse_triton = triton_sparse_attn_fwd(q, kv, attn_sink, topk_idxs, sm)
 o_ref = reference_sparse_attn(q, kv, attn_sink, topk_idxs, sm)

 diff = _normalized_correlation(o_triton.float, o_ref.float)
 assert diff < 5e-3, f"Triton fwd dense fallthrough diff = {diff:.2e}"


@pytest.mark.skipif(not torch.cuda.is_available, reason="needs CUDA")
def test_triton_matches_reference_with_masking:
 """Triton vs reference_sparse_attn with -1 masked entries and random top-k."""
 torch.manual_seed(101)
 B, S, H, D, N_kv, K = 2, 32, 16, 64, 256, 64
 q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda") * 0.5
 kv = torch.randn(B, N_kv, D, dtype=torch.bfloat16, device="cuda") * 0.5
 attn_sink = torch.randn(H, dtype=torch.float32, device="cuda")
 topk_idxs = torch.randint(0, N_kv, (B, S, K), dtype=torch.int32, device="cuda")
 # Sprinkle some -1 (masked) entries.
 topk_idxs[:, :, ::4] = -1
 sm = D ** -0.5

 o_triton, _ = triton_sparse_attn_fwd(q, kv, attn_sink, topk_idxs, sm)
 o_ref = reference_sparse_attn(q, kv, attn_sink, topk_idxs, sm)

 diff = _normalized_correlation(o_triton.float, o_ref.float)
 assert diff < 5e-3, f"Triton vs reference diff = {diff:.2e}"


@pytest.mark.skipif(not torch.cuda.is_available, reason="needs CUDA")
def test_triton_handles_all_masked_query:
 """Some queries have all -1 indices; output should be 0 (no real V), no NaN."""
 torch.manual_seed(102)
 B, S, H, D, N_kv, K = 1, 8, 16, 32, 64, 16
 q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
 kv = torch.randn(B, N_kv, D, dtype=torch.bfloat16, device="cuda")
 attn_sink = torch.zeros(H, dtype=torch.float32, device="cuda")
 topk_idxs = torch.randint(0, N_kv, (B, S, K), dtype=torch.int32, device="cuda")
 # Mask the entire first query.
 topk_idxs[:, 0, :] = -1
 sm = D ** -0.5

 o_triton, _ = triton_sparse_attn_fwd(q, kv, attn_sink, topk_idxs, sm)
 o_ref = reference_sparse_attn(q, kv, attn_sink, topk_idxs, sm)

 assert torch.isfinite(o_triton).all, "Triton output must be finite"
 # Both should give 0 for the all-masked first query (sink absorbs everything).
 assert o_triton[:, 0, :, :].abs.max.item < 1e-3, "all-masked output should be ~0"
 diff = _normalized_correlation(o_triton.float, o_ref.float)
 assert diff < 5e-3, f"Triton vs reference (with all-masked query) diff = {diff:.2e}"


@pytest.mark.skipif(not torch.cuda.is_available, reason="needs CUDA")
@pytest.mark.skipif(not HAVE_DEEPSEEK, reason="DeepSeek kernel.py not importable")
def test_triton_matches_tilelang:
 """Triton matches the DeepSeek TileLang sparse_attn within BF16 noise."""
 torch.manual_seed(103)
 B, S, H, D, N_kv, K = 2, 64, 16, 64, 256, 64
 q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda") * 0.5
 kv = torch.randn(B, N_kv, D, dtype=torch.bfloat16, device="cuda") * 0.5
 attn_sink = torch.randn(H, dtype=torch.float32, device="cuda")
 topk_idxs = torch.randint(0, N_kv, (B, S, K), dtype=torch.int32, device="cuda")
 topk_idxs[:, :, ::8] = -1
 sm = D ** -0.5

 o_triton, _ = triton_sparse_attn_fwd(q, kv, attn_sink, topk_idxs, sm)
 o_tilelang = _DS_SPARSE_ATTN(q, kv, attn_sink, topk_idxs, sm)

 diff = _normalized_correlation(o_triton.float, o_tilelang.float)
 print(f"\n[triton vs tilelang] correlation_diff={diff:.2e}")
 assert diff < 5e-3, f"Triton vs TileLang diff = {diff:.2e}"
