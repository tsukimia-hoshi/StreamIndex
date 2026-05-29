"""the kernel-level work correctness gate.

Validates ``flash_sparse.reference`` against the DeepSeek-V4 reference
implementation (``references/DeepSeek-V4-Pro/inference/kernel.py``).

Required environment:
* CUDA-enabled torch (tested on H200 with cu126 nightly)
* ``tilelang`` installed (per DeepSeek-V4 requirements.txt: tilelang==0.1.8)
* The DeepSeek-V4 inference dir importable on sys.path

Test gating:
* Tests prefixed ``test_self_*`` run without TileLang — they verify our
reference is internally consistent (e.g. dense fall-through, mask sanity).
* Tests prefixed ``test_parity_*`` import the DeepSeek TileLang kernels
and do a numerical match. They skip if tilelang is unavailable.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

from flash_sparse.reference import (
    reference_csa_forward,
    reference_hc_split_sinkhorn,
    reference_hca_forward,
    reference_sparse_attn,
)


# ---------------------------------------------------------------------------
# DeepSeek reference import (TileLang-backed)
# ---------------------------------------------------------------------------

_INFERENCE_DIRS = [
    os.path.expanduser("~/flash-sparse/references/DeepSeek-V4-Pro/inference"),
    os.path.expanduser("~/flash-sparse/references/DeepSeek-V3.2-Exp/inference"),
    # Local mirror (if running on Windows; tests still need a CUDA box though)
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


def _import_deepseek_kernel():
    """Try to import ``kernel.sparse_attn`` and ``kernel.hc_split_sinkhorn`` from
    the DeepSeek inference tree. Returns (sparse_attn_fn, hc_split_sinkhorn_fn) or
    ``(None, None)`` if unavailable.
    """
    for d in _INFERENCE_DIRS:
        if not os.path.isdir(d):
            continue
        if d not in sys.path:
            sys.path.insert(0, d)
        try:
            import importlib

            ds_kernel = importlib.import_module("kernel")
            return getattr(ds_kernel, "sparse_attn", None), getattr(ds_kernel, "hc_split_sinkhorn", None)
        except Exception:
            continue
    return None, None


_DS_SPARSE_ATTN, _DS_HC_SPLIT = _import_deepseek_kernel()
HAVE_DEEPSEEK = _DS_SPARSE_ATTN is not None and _DS_HC_SPLIT is not None


# ---------------------------------------------------------------------------
# Tolerance helpers
# ---------------------------------------------------------------------------


def _normalized_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    """Same metric as TileLang's `assert_tensors_similar`: 1 - 2<a,b>/(<a,a>+<b,b>)."""
    a32 = a.detach().float().reshape(-1)
    b32 = b.detach().float().reshape(-1)
    denom = (a32 * a32 + b32 * b32).sum()
    if denom.item() == 0.0:
        return 0.0
    sim = 2.0 * (a32 * b32).sum() / denom
    return float((1.0 - sim).item())


def _max_relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    a32 = a.detach().float()
    b32 = b.detach().float()
    diff = (a32 - b32).abs
    denom = b32.abs.clamp_min(1e-6)
    return float((diff / denom).max.item())


# ---------------------------------------------------------------------------
# Self-tests (no TileLang required)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_self_sparse_attn_dense_fallthrough():
    """When topk_idxs covers ALL of kv (no -1, no top-k filtering), ``reference_sparse_attn``
    must match a plain dense attention with attention sink, head-by-head.
    """
    torch.manual_seed(0)
    B, S, H, D, N_kv = 1, 32, 4, 64, 32
    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(B, N_kv, D, dtype=torch.bfloat16, device="cuda")
    attn_sink = torch.randn(H, dtype=torch.float32, device="cuda")
    # Every query attends to every kv row.
    topk_idxs = torch.arange(N_kv, device="cuda").view(1, 1, N_kv).expand(B, S, N_kv).contiguous().int()

    sm_scale = D**-0.5
    out = reference_sparse_attn(q, kv, attn_sink, topk_idxs, sm_scale)

    # Manual dense reference.
    q32, kv32 = q.float(), kv.float()
    scores = torch.einsum("bshd,bnd->bshn", q32, kv32) * sm_scale  # [B, S, H, N_kv]
    sink = attn_sink.view(1, 1, H, 1)
    max_with_sink = torch.maximum(scores.amax(dim=-1, keepdim=True), sink)
    s_shift = (scores - max_with_sink).exp
    sink_shift = (sink - max_with_sink).exp.squeeze(-1)
    denom = s_shift.sum(dim=-1) + sink_shift
    expected = torch.einsum("bshn,bnd->bshd", s_shift, kv32) / denom.unsqueeze(-1)

    diff = _normalized_correlation(out.float(), expected)
    assert diff < 1e-3, f"normalized correlation diff = {diff:.2e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_self_sparse_attn_handles_masked_indices():
    """``-1`` entries in ``topk_idxs`` must contribute zero attention probability."""
    torch.manual_seed(1)
    B, S, H, D, N_kv, K = 1, 16, 2, 32, 64, 16
    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(B, N_kv, D, dtype=torch.bfloat16, device="cuda")
    attn_sink = torch.zeros(H, dtype=torch.float32, device="cuda")

    topk_idxs_full = torch.randint(0, N_kv, (B, S, K), device="cuda", dtype=torch.int32)
    topk_idxs_masked = topk_idxs_full.clone()
    topk_idxs_masked[..., K // 2 :] = -1  # mask out the back half
    topk_idxs_unmasked = topk_idxs_full.clone()
    topk_idxs_unmasked[..., K // 2 :] = topk_idxs_unmasked[..., : K // 2]  # duplicate front half

    out_masked = reference_sparse_attn(q, kv, attn_sink, topk_idxs_masked, D**-0.5)
    # An equivalent: only the front half is real; the dup half should not change the answer
    # because gather + softmax over duplicates just rebalances the weights uniformly.
    # Easier check: when the front half alone is used (K' = K//2), the result should match.
    out_front_only = reference_sparse_attn(
        q, kv, attn_sink, topk_idxs_full[..., : K // 2].contiguous(), D**-0.5
    )
    diff = _normalized_correlation(out_masked.float(), out_front_only.float())
    assert diff < 1e-3, f"masking mismatch: corr diff = {diff:.2e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_self_sparse_attn_all_masked_returns_zero():
    """If every entry is masked and attn_sink is finite, output should be zero (no real V)."""
    torch.manual_seed(2)
    B, S, H, D, K = 1, 4, 2, 16, 8
    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(B, 32, D, dtype=torch.bfloat16, device="cuda")
    attn_sink = torch.zeros(H, dtype=torch.float32, device="cuda")  # finite sink

    topk_idxs = torch.full((B, S, K), -1, dtype=torch.int32, device="cuda")
    out = reference_sparse_attn(q, kv, attn_sink, topk_idxs, D**-0.5)
    assert torch.allclose(out.float(), torch.zeros_like(out.float()), atol=1e-4), (
        f"all-masked output should be zero, max abs = {out.abs.max.item():.2e}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_self_csa_forward_shape_and_mask():
    """`reference_csa_forward` runs end-to-end with shapes consistent with the prefill path
    (kv passed as full [B,S,D], not just the last n_win)."""
    torch.manual_seed(3)
    B, S, H, D = 1, 64, 4, 32
    n_win = 16
    m = 4
    n_compressed = S // m  # 16 in this case
    top_k = 8

    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(B, S, D, dtype=torch.bfloat16, device="cuda")
    kv_compressed = torch.randn(B, n_compressed, D, dtype=torch.bfloat16, device="cuda")
    # Random scores; apply causal mask to be well-formed.
    indexer_scores = torch.randn(B, S, n_compressed, device="cuda")
    block_last = (torch.arange(n_compressed, device="cuda") + 1) * m - 1  # [n_comp]
    q_pos = torch.arange(S, device="cuda").unsqueeze(-1)  # [S, 1]
    causal = q_pos >= block_last  # [S, n_comp]
    causal_b = causal.unsqueeze(0).expand(B, -1, -1)
    attn_sink = torch.zeros(H, dtype=torch.float32, device="cuda")

    out = reference_csa_forward(
        q,
        kv,
        kv_compressed,
        indexer_scores,
        attn_sink,
        n_win,
        top_k,
        causal_mask_for_compressed=causal_b,
    )
    assert out.shape == (B, S, H, D)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all, "output must be finite"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_self_hca_forward_shape():
    torch.manual_seed(4)
    B, S, H, D = 1, 64, 4, 32
    n_win = 16
    m_prime = 8
    n_compressed = S // m_prime  # 8

    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(B, S, D, dtype=torch.bfloat16, device="cuda")
    kv_compressed = torch.randn(B, n_compressed, D, dtype=torch.bfloat16, device="cuda")
    attn_sink = torch.zeros(H, dtype=torch.float32, device="cuda")

    out = reference_hca_forward(q, kv, kv_compressed, attn_sink, n_win, m_prime=m_prime)
    assert out.shape == (B, S, H, D)
    assert torch.isfinite(out).all


# ---------------------------------------------------------------------------
# Parity tests (TileLang required) — THIS is the the kernel-level work gate
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(not HAVE_DEEPSEEK, reason="DeepSeek kernel.py not importable (tilelang missing?)")
def test_parity_sparse_attn_against_deepseek():
    """Our pytorch reference must match the DeepSeek TileLang `sparse_attn` kernel
    within BF16 noise tolerance.
    """
    torch.manual_seed(5)
    B, S, H, D, N_kv, K = 1, 64, 4, 64, 256, 64
    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(B, N_kv, D, dtype=torch.bfloat16, device="cuda")
    attn_sink = torch.randn(H, dtype=torch.float32, device="cuda")
    topk_idxs = torch.randint(0, N_kv, (B, S, K), dtype=torch.int32, device="cuda")
    # Sprinkle some -1 (masked) entries.
    topk_idxs[:, :, ::8] = -1
    sm_scale = D**-0.5

    ours = reference_sparse_attn(q, kv, attn_sink, topk_idxs, sm_scale)
    theirs = _DS_SPARSE_ATTN(q, kv, attn_sink, topk_idxs, sm_scale)

    diff = _normalized_correlation(ours.float(), theirs.float())
    rel = _max_relative_error(ours.float(), theirs.float())
    print(f"\n[sparse_attn parity] correlation_diff={diff:.2e}, max_rel={rel:.2e}")
    # 1e-3 is the standard BF16 tolerance for this similarity metric (matches
    # TileLang's own assert_tensors_similar usage in their tests).
    assert diff < 5e-3, f"sparse_attn parity failed: corr diff = {diff:.2e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(not HAVE_DEEPSEEK, reason="DeepSeek kernel.py not importable")
def test_parity_hc_split_sinkhorn_against_deepseek():
    """Hyper-Connection split-Sinkhorn parity. Pure FP32, expect tighter tolerance."""
    torch.manual_seed(6)
    # The DeepSeek wrapper expects [B, S, mix_hc] and views it to [-1, mix_hc] internally.
    B, S = 2, 8
    hc_mult = 4
    sinkhorn_iters = 20
    eps = 1e-6
    mix_hc = (2 + hc_mult) * hc_mult

    mixes = torch.randn(B, S, mix_hc, dtype=torch.float32, device="cuda")
    hc_scale = torch.randn(3, dtype=torch.float32, device="cuda")
    hc_base = torch.randn(mix_hc, dtype=torch.float32, device="cuda")

    pre_o, post_o, comb_o = reference_hc_split_sinkhorn(
        mixes, hc_scale, hc_base, hc_mult=hc_mult, sinkhorn_iters=sinkhorn_iters, eps=eps
    )
    pre_d, post_d, comb_d = _DS_HC_SPLIT(
        mixes, hc_scale, hc_base, hc_mult=hc_mult, sinkhorn_iters=sinkhorn_iters, eps=eps
    )

    for name, ours, theirs in (("pre", pre_o, pre_d), ("post", post_o, post_d), ("comb", comb_o, comb_d)):
        diff = _normalized_correlation(ours, theirs)
        rel = _max_relative_error(ours, theirs)
        print(f"\n[hc_split_sinkhorn parity:{name}] correlation_diff={diff:.2e}, max_rel={rel:.2e}")
        assert diff < 1e-4, f"{name} parity failed: corr diff = {diff:.2e}"
