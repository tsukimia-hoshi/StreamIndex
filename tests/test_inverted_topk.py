"""the kernel-level work.b correctness gate for the inverted-topk index builder.

Tests that the inverted index correctly inverts the top-k mapping: for every
``(b, s, k)`` with ``topk_idxs[b, s, k] = s_global``, the entry ``b, s_global``
of the inverted index contains ``s`` somewhere in its first ``inv_count``
positions.
"""
from __future__ import annotations

import pytest
import torch

from flash_sparse.triton.inverted_topk import build_inverted_topk


@pytest.mark.skipif(not torch.cuda.is_available, reason="needs CUDA")
def test_inverted_topk_correctness:
 """For every valid forward entry, the inverted index contains it.

 Realistic top-k: each query's top-k indices are unique (matches torch.topk).
 """
 torch.manual_seed(600)
 B, S, K, N_kv = 1, 32, 8, 64
 K_MAX = 32 # generous

 # Sample without replacement to mimic torch.topk output (unique per query).
 topk_idxs_rows = []
 for _ in range(B * S):
 topk_idxs_rows.append(torch.randperm(N_kv)[:K])
 topk_idxs = torch.stack(topk_idxs_rows).reshape(B, S, K).to(torch.int32).cuda
 # Mask some entries
 topk_idxs[:, :, ::4] = -1

 inv_topk, inv_count = build_inverted_topk(topk_idxs, n_kv=N_kv, k_max=K_MAX)

 # Build the expected reverse mapping in pytorch.
 expected: dict[tuple[int, int], set[int]] = {}
 for b in range(B):
 for s in range(S):
 for k in range(K):
 s_global = int(topk_idxs[b, s, k].item)
 if s_global >= 0:
 expected.setdefault((b, s_global), set).add(s)

 # Verify: for each (b, s_global), the inverted index covers all queries that selected it.
 for (b, s_global), expected_queries in expected.items:
 cnt = int(inv_count[b, s_global].item)
 assert cnt == len(expected_queries), (
 f"count mismatch at ({b}, {s_global}): expected {len(expected_queries)}, got {cnt}"
 )
 cnt_capped = min(cnt, K_MAX)
 actual_queries = set(int(x) for x in inv_topk[b, s_global, :cnt_capped].cpu.tolist)
 assert actual_queries == expected_queries, (
 f"set mismatch at ({b}, {s_global}): expected {expected_queries}, got {actual_queries}"
 )

 # Verify: kv rows that no one selected have inv_count = 0
 selected_set = set(s_global for (_, s_global) in expected.keys)
 for n in range(N_kv):
 if (0, n) not in expected:
 assert inv_count[0, n].item == 0, f"non-selected n={n} should have count 0"


@pytest.mark.skipif(not torch.cuda.is_available, reason="needs CUDA")
def test_inverted_topk_counts_overflow:
 """When more than K_MAX queries select the same s, count should still
 reflect the true number (caller can detect overflow)."""
 torch.manual_seed(601)
 B, S, K, N_kv = 1, 64, 4, 16 # 64 queries × 4 picks each = 256 picks ÷ 16 keys = avg 16 per key
 K_MAX = 8

 # Force many queries to all select s_global=0
 topk_idxs = torch.zeros((B, S, K), dtype=torch.int32, device="cuda") # all queries select s=0 in slot 0
 topk_idxs[:, :, 1:] = torch.randint(1, N_kv, (B, S, K - 1), dtype=torch.int32, device="cuda")

 inv_topk, inv_count = build_inverted_topk(topk_idxs, n_kv=N_kv, k_max=K_MAX)

 # All 64 queries selected s=0, but K_MAX=8.
 assert inv_count[0, 0].item == 64, f"count for s=0 should be 64, got {inv_count[0, 0].item}"
 # First 8 entries are valid (some 8 of the 64 queries — race-order-dependent).
 valid_entries = set(int(x) for x in inv_topk[0, 0, :K_MAX].cpu.tolist)
 assert valid_entries.issubset(set(range(S))), "all entries should be valid query indices"
 assert len(valid_entries) == K_MAX, "first K_MAX slots should be filled"
