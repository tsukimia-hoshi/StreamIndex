"""the kernel-level work: HCA forward integration test.

HCA reuses the same `sparse_attn` kernel as CSA. The only difference is
``topk_idxs`` — HCA includes ALL causally-legal compressed positions
(no top-k selection; compress_ratio = m_prime = 128).
"""
from __future__ import annotations

import pytest
import torch

from flash_sparse.hca import flash_hca_forward


def _norm_corr(a: torch.Tensor, b: torch.Tensor) -> float:
 a32 = a.detach.float.reshape(-1)
 b32 = b.detach.float.reshape(-1)
 denom = (a32 * a32 + b32 * b32).sum
 if denom.item == 0.0:
 return 0.0
 return float((1.0 - 2.0 * (a32 * b32).sum / denom).item)


@pytest.mark.skipif(not torch.cuda.is_available, reason="needs CUDA")
def test_hca_forward_triton_matches_reference:
 torch.manual_seed(500)
 # Smaller m_prime in test so n_compressed isn't trivially small.
 B, S, n_h, d = 1, 64, 16, 64
 m_prime, n_win = 8, 16
 n_compressed = S // m_prime # 8

 q = torch.randn(B, S, n_h, d, dtype=torch.bfloat16, device="cuda") * 0.5
 kv = torch.randn(B, S, d, dtype=torch.bfloat16, device="cuda") * 0.5
 kv_compressed = torch.randn(B, n_compressed, d, dtype=torch.bfloat16, device="cuda") * 0.5
 attn_sink = torch.randn(n_h, dtype=torch.float32, device="cuda")

 o_triton = flash_hca_forward(
 q, kv, kv_compressed, attn_sink,
 n_win=n_win, m_prime=m_prime, use_triton=True,
 )
 o_ref = flash_hca_forward(
 q, kv, kv_compressed, attn_sink,
 n_win=n_win, m_prime=m_prime, use_triton=False,
 )

 assert o_triton.shape == (B, S, n_h, d)
 diff = _norm_corr(o_triton, o_ref)
 print(f"\n[hca_forward parity] correlation_diff = {diff:.2e}")
 assert diff < 5e-3


@pytest.mark.skipif(not torch.cuda.is_available, reason="needs CUDA")
def test_hca_grad_flows:
 torch.manual_seed(501)
 B, S, n_h, d = 1, 32, 16, 64
 m_prime, n_win = 4, 8
 n_compressed = S // m_prime

 q = torch.randn(B, S, n_h, d, dtype=torch.bfloat16, device="cuda", requires_grad=True)
 kv = torch.randn(B, S, d, dtype=torch.bfloat16, device="cuda", requires_grad=True)
 kv_compressed = torch.randn(B, n_compressed, d, dtype=torch.bfloat16, device="cuda", requires_grad=True)
 attn_sink = torch.randn(n_h, dtype=torch.float32, device="cuda", requires_grad=True)

 o = flash_hca_forward(
 q, kv, kv_compressed, attn_sink,
 n_win=n_win, m_prime=m_prime, use_triton=True,
 )
 loss = o.float.sum
 loss.backward

 assert q.grad is not None and torch.isfinite(q.grad).all
 assert kv.grad is not None and torch.isfinite(kv.grad).all
 assert kv_compressed.grad is not None and torch.isfinite(kv_compressed.grad).all
 assert attn_sink.grad is not None and torch.isfinite(attn_sink.grad).all
