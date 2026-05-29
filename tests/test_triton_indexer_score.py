"""the kernel-level work.b correctness gate for the Triton indexer-score kernel.

Tests parity vs `reference_lightning_indexer` (the pure-pytorch reference of
eq. 16 of the V4 paper).
"""

from __future__ import annotations

import pytest
import torch

from flash_sparse.reference import reference_lightning_indexer
from flash_sparse.triton.indexer_score import indexer_score, indexer_score_topk


def _norm_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a32 = a.detach().float().reshape(-1)
    b32 = b.detach().float().reshape(-1)
    denom = (a32 * a32 + b32 * b32).sum()
    if denom.item() == 0.0:
        return 0.0
    return float((1.0 - 2.0 * (a32 * b32).sum() / denom).item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_indexer_score_matches_reference():
    torch.manual_seed(300)
    B, S, H_I, D_I, T = 1, 64, 16, 64, 128
    q = torch.randn(B, S, H_I, D_I, dtype=torch.bfloat16, device="cuda") * 0.5
    k_idx = torch.randn(B, T, D_I, dtype=torch.bfloat16, device="cuda") * 0.5
    weights = torch.randn(B, S, H_I, dtype=torch.float32, device="cuda")

    # Triton
    scores_triton = indexer_score(q, k_idx, weights)
    # Reference
    scores_ref = reference_lightning_indexer(q, k_idx, weights)

    assert scores_triton.shape == (B, S, T)
    assert scores_ref.shape == (B, S, T)

    diff = _norm_corr(scores_triton, scores_ref)
    print(f"\n[indexer_score] correlation_diff = {diff:.2e}")
    assert diff < 5e-3, f"indexer_score mismatch: corr_diff = {diff:.2e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_indexer_score_with_h_64():
    """V4-Pro indexer has n_I_h=64 — exercise that exact head count."""
    torch.manual_seed(301)
    B, S, H_I, D_I, T = 1, 32, 64, 128, 64
    q = torch.randn(B, S, H_I, D_I, dtype=torch.bfloat16, device="cuda") * 0.3
    k_idx = torch.randn(B, T, D_I, dtype=torch.bfloat16, device="cuda") * 0.3
    weights = torch.randn(B, S, H_I, dtype=torch.float32, device="cuda")

    scores_triton = indexer_score(q, k_idx, weights)
    scores_ref = reference_lightning_indexer(q, k_idx, weights)

    diff = _norm_corr(scores_triton, scores_ref)
    print(f"\n[indexer_score H_I=64] correlation_diff = {diff:.2e}")
    assert diff < 5e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_indexer_score_topk():
    """Top-k indices from Triton-computed scores must match reference top-k."""
    torch.manual_seed(302)
    B, S, H_I, D_I, T, K = 1, 32, 16, 64, 64, 16
    q = torch.randn(B, S, H_I, D_I, dtype=torch.bfloat16, device="cuda")
    k_idx = torch.randn(B, T, D_I, dtype=torch.bfloat16, device="cuda")
    weights = torch.randn(B, S, H_I, dtype=torch.float32, device="cuda")

    top_idx_t, top_scores_t = indexer_score_topk(q, k_idx, weights, K)

    # Reference path: same scores, then torch.topk
    scores_ref = reference_lightning_indexer(q, k_idx, weights)
    top_scores_r, top_idx_r = scores_ref.topk(K, dim=-1)

    # Compare scores (will match in BF16 noise) and index sets.
    diff_scores = _norm_corr(top_scores_t, top_scores_r)
    assert diff_scores < 5e-3, f"top-k scores mismatch: {diff_scores:.2e}"

    # Index-set overlap (should be 100% when scores have no FP-equality ties).
    overlap = 0
    total = B * S * K
    for b in range(B):
        for s in range(S):
            set_t = set(top_idx_t[b, s].cpu.numpy.tolist)
            set_r = set(top_idx_r[b, s].cpu.numpy.tolist)
            overlap += len(set_t & set_r)
            overlap_pct = overlap / total
            print(f"\n[indexer_score_topk] index-set overlap = {overlap_pct:.4f}")
            assert overlap_pct >= 0.999, f"index-set overlap too low: {overlap_pct:.4f}"
