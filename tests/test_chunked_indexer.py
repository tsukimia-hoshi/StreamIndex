"""Correctness gate for the chunked indexer.

The chunked path must produce the same top-k SET as the unchunked
materialize-then-torch.topk path within FP32 noise (we tolerate index
reordering on ties).
"""

from __future__ import annotations

import pytest
import torch

from flash_sparse.reference import reference_lightning_indexer
from flash_sparse.triton.chunked_indexer import chunked_indexer_topk


def _random_indexer_inputs(B: int, S: int, H_I: int, D_I: int, T: int):
    q = torch.randn(B, S, H_I, D_I, dtype=torch.bfloat16, device="cuda") * 0.5
    k_idx = torch.randn(B, T, D_I, dtype=torch.bfloat16, device="cuda") * 0.5
    weights = torch.randn(B, S, H_I, dtype=torch.float32, device="cuda")
    return q, k_idx, weights


def _assert_topk_sets_match(actual: torch.Tensor, expected: torch.Tensor, *, min_overlap: float = 0.99):
    overlap = 0
    total = actual.numel()
    B, S, _ = actual.shape
    for b in range(B):
        for s in range(S):
            set_actual = set(int(x) for x in actual[b, s].cpu().tolist())
            set_expected = set(int(x) for x in expected[b, s].cpu().tolist())
            overlap += len(set_actual & set_expected)
    overlap_pct = overlap / total
    print(f"\n[chunked indexer] index-set overlap = {overlap_pct:.4f}")
    assert overlap_pct >= min_overlap, f"overlap too low: {overlap_pct:.4f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_chunked_matches_torch_topk_set_without_mask():
    """Chunked top-k indices match `torch.topk(reference)` when no mask is provided."""
    torch.manual_seed(900)
    B, S, H_I, D_I, T, K = 1, 64, 16, 32, 256, 32
    q, k_idx, weights = _random_indexer_inputs(B, S, H_I, D_I, T)

    top_idx_chunked, _ = chunked_indexer_topk(
        q,
        k_idx,
        weights,
        top_k=K,
        chunk_s=16,
        chunk_t=64,
    )

    scores_full = reference_lightning_indexer(q, k_idx, weights)
    _, top_idx_full = scores_full.topk(K, dim=-1)
    _assert_topk_sets_match(top_idx_chunked, top_idx_full)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_chunked_matches_torch_topk_set_with_materialized_causal_mask():
    """Materialized causal masks should be applied per chunk before the merge."""
    torch.manual_seed(901)
    B, S, H_I, D_I, T, K, ratio = 1, 64, 16, 32, 32, 8, 4
    q, k_idx, weights = _random_indexer_inputs(B, S, H_I, D_I, T)
    q_pos = torch.arange(S, device="cuda").unsqueeze(1)
    t_pos = torch.arange(T, device="cuda").unsqueeze(0)
    causal_mask = (t_pos < (q_pos + 1) // ratio).unsqueeze(0).expand(B, -1, -1)

    top_idx_chunked, _ = chunked_indexer_topk(
        q,
        k_idx,
        weights,
        top_k=K,
        causal_mask=causal_mask,
        chunk_s=13,
        chunk_t=7,
    )

    scores_full = reference_lightning_indexer(q, k_idx, weights).masked_fill(~causal_mask, float("-inf"))
    _, top_idx_full = scores_full.topk(K, dim=-1)
    top_idx_full = torch.where(torch.isinf(scores_full.gather(-1, top_idx_full)), -1, top_idx_full)
    _assert_topk_sets_match(top_idx_chunked, top_idx_full)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_chunked_matches_torch_topk_set_with_causal_ratio():
    """On-the-fly causal-ratio masking should match an equivalent materialized mask."""
    torch.manual_seed(902)
    B, S, H_I, D_I, T, K, ratio = 1, 64, 16, 32, 32, 8, 4
    q, k_idx, weights = _random_indexer_inputs(B, S, H_I, D_I, T)
    q_pos = torch.arange(S, device="cuda").unsqueeze(1)
    t_pos = torch.arange(T, device="cuda").unsqueeze(0)
    causal_mask = (t_pos < (q_pos + 1) // ratio).unsqueeze(0).expand(B, -1, -1)

    top_idx_ratio, _ = chunked_indexer_topk(
        q,
        k_idx,
        weights,
        top_k=K,
        causal_ratio=ratio,
        chunk_s=13,
        chunk_t=7,
    )

    scores_full = reference_lightning_indexer(q, k_idx, weights).masked_fill(~causal_mask, float("-inf"))
    _, top_idx_full = scores_full.topk(K, dim=-1)
    top_idx_full = torch.where(torch.isinf(scores_full.gather(-1, top_idx_full)), -1, top_idx_full)
    _assert_topk_sets_match(top_idx_ratio, top_idx_full)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_chunked_handles_unaligned_chunks():
    """Chunk sizes that don't evenly divide S or T must still work."""
    torch.manual_seed(903)
    B, S, H_I, D_I, T, K = 1, 47, 16, 32, 73, 16
    q, k_idx, weights = _random_indexer_inputs(B, S, H_I, D_I, T)

    top_idx, top_scores = chunked_indexer_topk(
        q,
        k_idx,
        weights,
        top_k=K,
        chunk_s=10,
        chunk_t=20,
    )

    assert top_idx.shape == (B, S, K)
    assert top_scores.shape == (B, S, K)
    # Indices must be in [0, T) or -1.
    valid = top_idx >= 0
    assert (top_idx[valid] < T).all().item(), "valid index out of range"
