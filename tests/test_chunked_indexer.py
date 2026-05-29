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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_chunked_matches_torch_topk_set():
    """Chunked top-k indices match `torch.topk(reference)` on the set level."""
    torch.manual_seed(900)
    B, S, H_I, D_I, T, K = 1, 64, 16, 32, 256, 32
    q = torch.randn(B, S, H_I, D_I, dtype=torch.bfloat16, device="cuda") * 0.5
    k_idx = torch.randn(B, T, D_I, dtype=torch.bfloat16, device="cuda") * 0.5
    weights = torch.randn(B, S, H_I, dtype=torch.float32, device="cuda")

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

    overlap = 0
    total = B * S * K
    for b in range(B):
        for s in range(S):
            set_chunked = set(int(x) for x in top_idx_chunked[b, s].cpu.tolist)
            set_full = set(int(x) for x in top_idx_full[b, s].cpu.tolist)
            overlap += len(set_chunked & set_full)
            overlap_pct = overlap / total
            print(f"\n[chunked indexer] index-set overlap = {overlap_pct:.4f}")
            assert overlap_pct >= 0.99, f"overlap too low: {overlap_pct:.4f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_chunked_handles_unaligned_chunks():
    """Chunk sizes that don't evenly divide S or T must still work."""
    torch.manual_seed(901)
    B, S, H_I, D_I, T, K = 1, 47, 16, 32, 73, 16
    q = torch.randn(B, S, H_I, D_I, dtype=torch.bfloat16, device="cuda") * 0.5
    k_idx = torch.randn(B, T, D_I, dtype=torch.bfloat16, device="cuda") * 0.5
    weights = torch.randn(B, S, H_I, dtype=torch.float32, device="cuda")

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
    assert (top_idx[valid] < T).all.item(), "valid index out of range"
