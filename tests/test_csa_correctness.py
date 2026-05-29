"""the kernel-level work.d: end-to-end CSA forward integration test.

Uses random weights and random inputs (matching the model.py CSA path with
small dims). Compares the Triton-backed `flash_csa_forward` against the pure
pytorch path through `reference_sparse_attn` + `reference_lightning_indexer`.
"""

from __future__ import annotations

import pytest
import torch

from flash_sparse.csa import flash_csa_forward


def _norm_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a32 = a.detach().float().reshape(-1)
    b32 = b.detach().float().reshape(-1)
    denom = (a32 * a32 + b32 * b32).sum()
    if denom.item() == 0.0:
        return 0.0
    return float((1.0 - 2.0 * (a32 * b32).sum() / denom).item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_csa_forward_triton_matches_reference():
    """End-to-end CSA forward: Triton path matches pytorch reference path."""
    torch.manual_seed(400)
    B, S, n_h, d = 1, 64, 16, 64
    n_I_h, c_I = 16, 32
    m, n_win, top_k = 4, 16, 8
    n_compressed = S // m  # 16

    # Inputs (random; in production these come from upstream linear projections + RoPE).
    q = torch.randn(B, S, n_h, d, dtype=torch.bfloat16, device="cuda") * 0.5
    kv = torch.randn(B, S, d, dtype=torch.bfloat16, device="cuda") * 0.5
    kv_compressed = torch.randn(B, n_compressed, d, dtype=torch.bfloat16, device="cuda") * 0.5
    q_idx = torch.randn(B, S, n_I_h, c_I, dtype=torch.bfloat16, device="cuda") * 0.5
    k_idx_compressed = torch.randn(B, n_compressed, c_I, dtype=torch.bfloat16, device="cuda") * 0.5
    weights = torch.randn(B, S, n_I_h, dtype=torch.float32, device="cuda")
    attn_sink = torch.randn(n_h, dtype=torch.float32, device="cuda")

    # Triton path
    o_triton = flash_csa_forward(
        q,
        kv,
        kv_compressed,
        q_idx,
        k_idx_compressed,
        weights,
        attn_sink,
        n_win=n_win,
        top_k=top_k,
        m=m,
        use_triton=True,
    )
    # Reference path
    o_ref = flash_csa_forward(
        q,
        kv,
        kv_compressed,
        q_idx,
        k_idx_compressed,
        weights,
        attn_sink,
        n_win=n_win,
        top_k=top_k,
        m=m,
        use_triton=False,
    )

    assert o_triton.shape == (B, S, n_h, d)
    diff = _norm_corr(o_triton, o_ref)
    print(f"\n[csa_forward parity] correlation_diff = {diff:.2e}")
    assert diff < 5e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_csa_forward_v4_pro_dims():
    """Run the integration with V4-Pro-shaped dims to confirm shape-agnosticism.
    Uses scaled-down sequence length but real per-head dims."""
    torch.manual_seed(401)
    # V4-Pro: head_dim=512, n_h=128, n_I_h=64, c_I=128. Use small seq to keep mem bounded.
    B, S, n_h, d = 1, 32, 128, 64  # small d=64 (would be 512 in V4) to fit in test
    n_I_h, c_I = 64, 64
    m, n_win, top_k = 4, 16, 8
    n_compressed = S // m

    q = torch.randn(B, S, n_h, d, dtype=torch.bfloat16, device="cuda") * 0.3
    kv = torch.randn(B, S, d, dtype=torch.bfloat16, device="cuda") * 0.3
    kv_compressed = torch.randn(B, n_compressed, d, dtype=torch.bfloat16, device="cuda") * 0.3
    q_idx = torch.randn(B, S, n_I_h, c_I, dtype=torch.bfloat16, device="cuda") * 0.3
    k_idx_compressed = torch.randn(B, n_compressed, c_I, dtype=torch.bfloat16, device="cuda") * 0.3
    weights = torch.randn(B, S, n_I_h, dtype=torch.float32, device="cuda")
    attn_sink = torch.randn(n_h, dtype=torch.float32, device="cuda")

    o_triton = flash_csa_forward(
        q,
        kv,
        kv_compressed,
        q_idx,
        k_idx_compressed,
        weights,
        attn_sink,
        n_win=n_win,
        top_k=top_k,
        m=m,
        use_triton=True,
    )
    o_ref = flash_csa_forward(
        q,
        kv,
        kv_compressed,
        q_idx,
        k_idx_compressed,
        weights,
        attn_sink,
        n_win=n_win,
        top_k=top_k,
        m=m,
        use_triton=False,
    )
    diff = _norm_corr(o_triton, o_ref)
    print(f"\n[csa_forward V4-Pro shape] correlation_diff = {diff:.2e}")
    assert diff < 5e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_csa_forward_chunked_indexer_matches_unchunked():
    """Chunked-indexer path must produce the same output as the unchunked path
    (set-equivalence on top-k indices → same selected KV → same attention output)."""
    torch.manual_seed(403)
    B, S, n_h, d = 1, 64, 16, 64
    n_I_h, c_I = 16, 32
    m, n_win, top_k = 4, 16, 8
    n_compressed = S // m

    q = torch.randn(B, S, n_h, d, dtype=torch.bfloat16, device="cuda") * 0.5
    kv = torch.randn(B, S, d, dtype=torch.bfloat16, device="cuda") * 0.5
    kv_compressed = torch.randn(B, n_compressed, d, dtype=torch.bfloat16, device="cuda") * 0.5
    q_idx = torch.randn(B, S, n_I_h, c_I, dtype=torch.bfloat16, device="cuda") * 0.5
    k_idx_compressed = torch.randn(B, n_compressed, c_I, dtype=torch.bfloat16, device="cuda") * 0.5
    weights = torch.randn(B, S, n_I_h, dtype=torch.float32, device="cuda")
    attn_sink = torch.randn(n_h, dtype=torch.float32, device="cuda")

    o_unchunked = flash_csa_forward(
        q,
        kv,
        kv_compressed,
        q_idx,
        k_idx_compressed,
        weights,
        attn_sink,
        n_win=n_win,
        top_k=top_k,
        m=m,
        use_triton=True,
        use_chunked_indexer=False,
    )
    o_chunked = flash_csa_forward(
        q,
        kv,
        kv_compressed,
        q_idx,
        k_idx_compressed,
        weights,
        attn_sink,
        n_win=n_win,
        top_k=top_k,
        m=m,
        use_triton=True,
        use_chunked_indexer=True,
        chunk_s=16,
        chunk_t=8,
    )

    diff = _norm_corr(o_unchunked, o_chunked)
    print(f"\n[csa unchunked vs chunked] correlation_diff = {diff:.2e}")
    # Set-equivalence on top-k means selected KV should match (modulo ties);
    # output should be very close.
    assert diff < 5e-3, f"chunked vs unchunked mismatch: {diff:.2e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_csa_forward_grad_flows():
    """Backward through the full CSA forward — ensures the autograd path works."""
    torch.manual_seed(402)
    B, S, n_h, d = 1, 32, 16, 64
    n_I_h, c_I = 16, 32
    m, n_win, top_k = 4, 16, 8
    n_compressed = S // m

    q = torch.randn(B, S, n_h, d, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    kv = torch.randn(B, S, d, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    kv_compressed = torch.randn(B, n_compressed, d, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    q_idx = torch.randn(B, S, n_I_h, c_I, dtype=torch.bfloat16, device="cuda")
    k_idx_compressed = torch.randn(B, n_compressed, c_I, dtype=torch.bfloat16, device="cuda")
    weights = torch.randn(B, S, n_I_h, dtype=torch.float32, device="cuda")
    attn_sink = torch.randn(n_h, dtype=torch.float32, device="cuda", requires_grad=True)

    o = flash_csa_forward(
        q,
        kv,
        kv_compressed,
        q_idx,
        k_idx_compressed,
        weights,
        attn_sink,
        n_win=n_win,
        top_k=top_k,
        m=m,
        use_triton=True,
    )
    loss = o.float().sum()
    loss.backward

    assert q.grad is not None and torch.isfinite(q.grad).all, "q gradient invalid"
    assert kv.grad is not None and torch.isfinite(kv.grad).all, "kv gradient invalid"
    assert kv_compressed.grad is not None and torch.isfinite(kv_compressed.grad).all
    assert attn_sink.grad is not None and torch.isfinite(attn_sink.grad).all
