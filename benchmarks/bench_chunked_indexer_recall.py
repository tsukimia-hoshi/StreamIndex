"""Chunked indexer recall study — does the partition-merge top-k actually
match the materialize-then-topk ground truth?

The streaming-top-k theorem (docs/streaming_topk.md) proves the chunked path
is mathematically equivalent to materialize. But FP32 reduction order across
chunks can break exact ties in the score distribution, and ties between the
k-th and (k+1)-th score will swap entries between the two paths. This study
measures the empirical set-overlap recall across chunk sizes at V4-Flash-
realistic shapes.

What we report:
- mean / min set-overlap recall over (B, S) queries
- rate of queries where recall < 1.0 (any non-tie disagreement)
- score-distribution diagnostic (median tied-score gap)

Run as: python benchmarks/bench_chunked_indexer_recall.py
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from flash_sparse.triton.indexer_score import indexer_score
from flash_sparse.triton.chunked_indexer import chunked_indexer_topk


@dataclass
class RecallCfg:
    B: int = 1
    S: int = 8192
    T: int = 8192  # T = S/m; for m=4, S=32K → T=8K
    H_I: int = 64  # V4-Flash indexer-head count
    D_I: int = 128  # V4-Flash indexer-head dim
    top_k: int = 512


def _build_inputs(cfg: RecallCfg, seed: int = 2026):
    """Synthetic indexer inputs.

    Q, K_idx ~ N(0, 1/D_I), so q·k has variance 1/D_I under this iid model.
    This is a synthetic stress distribution, not a claim that it matches real
    V4 attention traces. Weights ~ N(0, 1) for unconstrained sign — the
    indexer's `weights_proj(x)` output is unconstrained per V4 spec, which
    means scores I(t,s) = Σ_h w_h ReLU(q·k) can be arbitrarily signed.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    sigma = cfg.D_I**-0.5
    q = torch.randn(cfg.B, cfg.S, cfg.H_I, cfg.D_I, generator=g, device="cuda", dtype=torch.bfloat16) * sigma
    k_idx = torch.randn(cfg.B, cfg.T, cfg.D_I, generator=g, device="cuda", dtype=torch.bfloat16) * sigma
    weights = torch.randn(cfg.B, cfg.S, cfg.H_I, generator=g, device="cuda", dtype=torch.float32)
    return q, k_idx, weights


def _materialize_topk(q, k_idx, w, top_k: int):
    """Ground truth: full [B, S, T] score matrix, then torch.topk."""
    scores = indexer_score(q, k_idx, w)  # [B, S, T] FP32
    vals, idx = scores.topk(top_k, dim=-1)
    return idx.to(torch.int64), vals, scores


def _set_overlap_recall(idx_a: torch.Tensor, idx_b: torch.Tensor, top_k: int) -> torch.Tensor:
    """Per-query set overlap: |a ∩ b| / k. Shape: [B, S]."""
    B, S, K = idx_a.shape
    # Ignore -1 padding entries (chunked path uses -1 for unfilled slots).
    sa = idx_a.clone()
    sb = idx_b.clone()
    sa[sa < 0] = -2  # different sentinels so they don't intersect
    sb[sb < 0] = -3
    sa, _ = sa.sort(dim=-1)
    sb, _ = sb.sort(dim=-1)
    # Intersection via two-pointer / set ops in tensor form: count common values.
    # Simple approach: torch.isin is O(K*K) per query but K=512 so 256K ops/query,
    # times B*S → manageable for B=1, S=8K. For larger S we'd need a hash approach.
    overlap = torch.zeros((B, S), dtype=torch.float32, device=idx_a.device)
    # Process in chunks to bound memory.
    chunk = 512
    for s0 in range(0, S, chunk):
        s1 = min(s0 + chunk, S)
        a = sa[:, s0:s1]  # [B, c, K]
        b = sb[:, s0:s1]  # [B, c, K]
        # broadcast equality: [B, c, K, 1] vs [B, c, 1, K]
        match = a.unsqueeze(-1) == b.unsqueeze(-2)  # [B, c, K, K]
        overlap[:, s0:s1] = match.any(dim=-1).sum(dim=-1).float() / top_k
        return overlap


def _tied_gap(scores: torch.Tensor, top_k: int) -> dict:
    """How close is the k-th score to the (k+1)-th? Tight gaps mean FP noise
    can swap them between materialize and chunked paths.

    Uses topk(k+1) to avoid full-sort's hidden 8 GB int64-index allocation
    (code review): at S=T=32K, scores.sort allocates 4 GB FP32 values
    + 8 GB int64 indices on top of the 4 GB score matrix.
    """
    if scores.shape[-1] <= top_k:
        return {"median_gap": float("nan"), "p1_gap": float("nan"), "min_gap": float("nan")}
    boundary, _ = scores.topk(top_k + 1, dim=-1)
    s_k = boundary[..., top_k - 1]
    s_kp1 = boundary[..., top_k]
    gap = (s_k - s_kp1).flatten
    return {
        "median_gap": float(gap.median),
        "p1_gap": float(gap.kthvalue(max(1, gap.numel // 100)).values),
        "min_gap": float(gap.min),
    }


def measure(cfg: RecallCfg, chunk_s: int, chunk_t: int, *, ground_truth=None):
    """Compute recall of chunked vs materialize at one (chunk_s, chunk_t)."""
    if ground_truth is None:
        q, k_idx, w = _build_inputs(cfg)
        idx_mat, val_mat, scores = _materialize_topk(q, k_idx, w, cfg.top_k)
        gap = _tied_gap(scores, cfg.top_k)
    else:
        q, k_idx, w, idx_mat, gap = ground_truth

        idx_chk, _ = chunked_indexer_topk(
            q,
            k_idx,
            w,
            cfg.top_k,
            chunk_s=chunk_s,
            chunk_t=chunk_t,
        )

        rec = _set_overlap_recall(idx_mat, idx_chk, cfg.top_k)
        return {
            "mean_recall": float(rec.mean()),
            "min_recall": float(rec.min),
            "frac_perfect": float((rec == 1.0).float().mean()),
            "frac_below_99": float((rec < 0.99).float().mean()),
            "tied_gap": gap,
        }


def main():
    if not torch.cuda.is_available():
        return
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")

    # V4-Flash-realistic indexer config (n_h=64, c_I=128).
    # Try two scales: small (S=8K) for fine-grained sweep, larger (S=32K) for sanity.
    flights = [
        ("V4-Flash, S=8K", RecallCfg(S=8192, T=8192, top_k=512)),
        ("V4-Flash, S=32K", RecallCfg(S=32768, T=32768, top_k=512)),
    ]
    chunk_pairs = [
        # (chunk_s, chunk_t) — small to large
        (512, 2048),
        (1024, 4096),
        (2048, 4096),
        (2048, 8192),
        (4096, 4096),
        (4096, 8192),
        (8192, 16384),
    ]
    # review coverage: small chunk_t < top_k exercises the saturated-chunk branch
    # in chunked_indexer where each chunk's topk(min(k, |chunk|)) returns ALL
    # entries rather than a true top-k. Only run on the small flight to keep
    # total bench time bounded.
    sub_topk_pair = (512, 256)

    for name, cfg in flights:
        print(f"=== {name} (S={cfg.S}, T={cfg.T}, H_I={cfg.H_I}, D_I={cfg.D_I}, k={cfg.top_k}) ===")
        # Compute ground truth once per cfg.
        q, k_idx, w = _build_inputs(cfg)
        try:
            idx_mat, _, scores = _materialize_topk(q, k_idx, w, cfg.top_k)
            gap = _tied_gap(scores, cfg.top_k)
            print(
                f" ground-truth tie gap: median={gap['median_gap']:.3e},"
                f" p1={gap['p1_gap']:.3e}, min={gap['min_gap']:.3e}"
            )
        except torch.OutOfMemoryError:
            print(f" ground-truth materialize OOM at this S — skipping flight")
            continue
        del scores
        torch.cuda.empty_cache()

        gt = (q, k_idx, w, idx_mat, gap)
        print(f" {'(chunk_s, chunk_t)':>20} {'mean':>8} {'min':>8} {'%=1':>7} {'%<.99':>7}")
        pairs = ([sub_topk_pair] if cfg.S <= 8192 else []) + chunk_pairs
        for cs, ct in pairs:
            try:
                r = measure(cfg, cs, ct, ground_truth=gt)
                print(
                    f" {f'({cs}, {ct})':>20} "
                    f"{r['mean_recall']:>8.4f} {r['min_recall']:>8.4f} "
                    f"{100 * r['frac_perfect']:>6.2f}% "
                    f"{100 * r['frac_below_99']:>6.3f}%"
                )
            except Exception as e:
                print(f" {f'({cs}, {ct})':>20} FAIL: {type(e).__name__}: {str(e)[:60]}")
                torch.cuda.empty_cache()
                del q, k_idx, w, idx_mat
                torch.cuda.empty_cache()
                print


if __name__ == "__main__":
    main
