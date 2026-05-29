"""Design-space sweep for the chunked indexer at V4-Flash dims.

Three sweeps for paper §6.4:
1. chunk_S sweep (chunk_T fixed at 8192, top_k=512): find query-chunk-size knee.
2. chunk_T sweep (chunk_S fixed at 2048, top_k=512): find key-chunk-size knee.
3. top_k sweep (chunk_S=2048, chunk_T=8192): how does cost scale with sparsity.

For (1) and (2), recall is computed at S=8192 (where materialize fits) against
the bit-exact V4-Flash reference materialize+postmask path. Time and peak HBM
at S=8192 are reported as reference; the headline cost numbers come from
S=262144 (where only chunked runs).

For (3), recall + time/HBM at S=8192 only — at S=262144 the wall-clock dominates
and we want the recall measurement before that.

Run on H200:
python eval/sweep_design_space.py
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# Shared synthesis (same as test_v4_indexer_parity.py / bench_v4_indexer_scaling.py)
# --------------------------------------------------------------------------
def _synthesize_inputs(
    *,
    B=1,
    S,
    n_heads=64,
    head_dim=128,
    dim=4096,
    ratio=4,
    seed=2026,
    device="cuda",
    dtype=torch.bfloat16,
):
    g = torch.Generator(device=device).manual_seed(seed)
    T = S // ratio
    softmax_scale = head_dim**-0.5
    q = torch.randn(B, S, n_heads, head_dim, generator=g, device=device, dtype=dtype) * (head_dim**-0.5)
    kv_cache = torch.randn(B, T, head_dim, generator=g, device=device, dtype=dtype) * (head_dim**-0.5)
    torch.manual_seed(seed)
    x = torch.randn(B, S, dim, generator=g, device=device, dtype=dtype)
    weights_proj = nn.Linear(dim, n_heads, bias=False, dtype=dtype, device=device)
    with torch.no_grad:
        weights = weights_proj(x).float() * (softmax_scale * (n_heads**-0.5))
        return q, kv_cache, weights, T


def _materialize_topk(q, kv_cache, weights, top_k, ratio):
    B, S, _, _ = q.shape
    T = kv_cache.shape[1]
    score = torch.einsum("bshd,btd->bsht", q.float(), kv_cache.float())
    score = (score.relu_ * weights.unsqueeze(-1)).sum(dim=2)
    illegal = (
        torch.arange(T, device=q.device).repeat(S, 1)
        >= torch.arange(1, S + 1, device=q.device).unsqueeze(1) // ratio
    )
    score = score.masked_fill(illegal.unsqueeze(0), float("-inf"))
    k_eff = min(top_k, T)
    topk = score.topk(k_eff, dim=-1)[1]
    boundary = torch.arange(1, S + 1, device=q.device).unsqueeze(1) // ratio
    invalid = topk >= boundary.unsqueeze(0)
    return torch.where(invalid, torch.full_like(topk, -1), topk)


def _chunked_topk(q, kv_cache, weights, top_k, ratio, chunk_s, chunk_t):
    from flash_sparse.triton.chunked_indexer import chunked_indexer_topk

    top_idx, _ = chunked_indexer_topk(
        q,
        kv_cache,
        weights,
        top_k=top_k,
        causal_ratio=ratio,
        chunk_s=chunk_s,
        chunk_t=chunk_t,
    )
    return top_idx


# --------------------------------------------------------------------------
# Set-overlap recall (excludes -1 padding sentinels)
# --------------------------------------------------------------------------
def _set_overlap_recall(idx_a, idx_b):
    B, S, K = idx_a.shape
    sa = idx_a.clone().to(torch.int64)
    sb = idx_b.clone().to(torch.int64)
    valid_a = sa >= 0
    valid_b = sb >= 0
    n_valid_a = valid_a.sum(dim=-1)
    n_valid_b = valid_b.sum(dim=-1)
    sa = torch.where(valid_a, sa, torch.full_like(sa, -2))
    sb = torch.where(valid_b, sb, torch.full_like(sb, -3))
    overlap_count = torch.zeros((B, S), dtype=torch.float32, device=idx_a.device)
    chunk = 512
    for s0 in range(0, S, chunk):
        s1 = min(s0 + chunk, S)
        a = sa[:, s0:s1]
        b = sb[:, s0:s1]
        match = (a.unsqueeze(-1) == b.unsqueeze(-2)) & valid_a[:, s0:s1].unsqueeze(-1)
        overlap_count[:, s0:s1] = match.any(dim=-1).sum(dim=-1).float()
        both_empty = (n_valid_a == 0) & (n_valid_b == 0)
        a_empty_b_nonempty = (n_valid_a == 0) & (n_valid_b > 0)
        denom = n_valid_a.clamp(min=1).float()
        rec = overlap_count / denom
        rec = torch.where(both_empty, torch.ones_like(rec), rec)
        rec = torch.where(a_empty_b_nonempty, torch.zeros_like(rec), rec)
        return rec


# --------------------------------------------------------------------------
# Time + peak measurement
# --------------------------------------------------------------------------
def _time_us(fn, n_iter=3, n_warmup=2):
    """Self-contained warmup so timing is independent of any prior peak-
    measurement call sequence (code review §5)."""
    for _ in range(n_warmup):
        fn
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(n_iter):
            fn
            e1.record()
            torch.cuda.synchronize()
            return e0.elapsed_time(e1) * 1000.0 / n_iter


def _measure_peak_post_warmup(fn):
    fn
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn
    torch.cuda.synchronize()
    return max(0, torch.cuda.max_memory_allocated() - baseline) / 1024**3


# --------------------------------------------------------------------------
# A run = (S, chunk_s, chunk_t, top_k) → (recall_or_None, us, peak_gb, status)
# --------------------------------------------------------------------------
def _run_one(S, *, chunk_s, chunk_t, top_k, n_heads, head_dim, dim, ratio, measure_recall: bool):
    try:
        q, kv_cache, weights, T = _synthesize_inputs(
            S=S,
            n_heads=n_heads,
            head_dim=head_dim,
            dim=dim,
            ratio=ratio,
        )
        # Cap chunks to actual sizes (no point in chunk_s > S).
        cs = min(chunk_s, S)
        ct = min(chunk_t, T)
        fn = lambda: _chunked_topk(q, kv_cache, weights, top_k, ratio, cs, ct)

        recall = None
        if measure_recall:
            idx_chk = fn()
            torch.cuda.synchronize()
            idx_mat = _materialize_topk(q, kv_cache, weights, top_k, ratio)
            K = min(idx_mat.shape[-1], idx_chk.shape[-1])
            r = _set_overlap_recall(idx_mat[..., :K], idx_chk[..., :K])
            recall = (float(r.mean()), float(r.min))
            del idx_chk, idx_mat, r
            torch.cuda.empty_cache()

            peak_gb = _measure_peak_post_warmup(fn)
            us = _time_us(fn, n_iter=3, n_warmup=2)
            del q, kv_cache, weights
            torch.cuda.empty_cache()
            return {"status": "ok", "us": us, "peak_gb": peak_gb, "recall": recall}
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"status": "OOM"}
    except RuntimeError as e:
        torch.cuda.empty_cache()
        return {"status": "ERROR", "msg": str(e)[:140]}


# --------------------------------------------------------------------------
# Three sweeps
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser
    ap.add_argument(
        "--small-s",
        type=int,
        default=16384,
        help="S where materialize fits (recall measurement). Set "
        "to 16384 (T=4096) to avoid top_k=2048 saturation per "
        "code review §4.",
    )
    ap.add_argument("--big-s", type=int, default=262144, help="S where only chunked runs (cost measurement).")
    ap.add_argument("--n-heads", type=int, default=64)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--dim", type=int, default=4096)
    ap.add_argument("--ratio", type=int, default=4)
    args = ap.parse_args

    if not torch.cuda.is_available():
        print("CUDA required.", file=sys.stderr)
        sys.exit(1)

        print(f"Device: {torch.cuda.get_device_name()}")
        print(
            f"V4-Flash dims: n_heads={args.n_heads} head_dim={args.head_dim} "
            f"dim={args.dim} ratio={args.ratio}"
        )
        print(f"Recall measured at S={args.small_s}; cost measured at S={args.big_s}.")
        print

        # ----------------------------------------------------------------------
        # Sweep 1: chunk_S
        # ----------------------------------------------------------------------
        print("=" * 90)
        print("SWEEP 1 — chunk_S (chunk_T=8192, top_k=512)")
        print("=" * 90)
        print(
            f"{'chunk_S':>10}"
            f" {'mean rec':>9} {'min rec':>9}"
            f" {'small ms':>9} {'small HBM':>10}"
            f" {'big ms':>9} {'big HBM':>10}"
        )
        print("-" * 90)
        for chunk_s in (1024, 4096, 16384, 65536, 262144):
            r_small = _run_one(
                args.small_s,
                chunk_s=chunk_s,
                chunk_t=8192,
                top_k=512,
                n_heads=args.n_heads,
                head_dim=args.head_dim,
                dim=args.dim,
                ratio=args.ratio,
                measure_recall=True,
            )
            torch.cuda.empty_cache()
            r_big = _run_one(
                args.big_s,
                chunk_s=chunk_s,
                chunk_t=8192,
                top_k=512,
                n_heads=args.n_heads,
                head_dim=args.head_dim,
                dim=args.dim,
                ratio=args.ratio,
                measure_recall=False,
            )
            torch.cuda.empty_cache()
            rec = (
                (f"{r_small['recall'][0]:.4f}", f"{r_small['recall'][1]:.4f}")
                if r_small["status"] == "ok"
                else ("-", "-")
            )
            sm_t = (
                (f"{r_small['us'] / 1000:.1f}", f"{r_small['peak_gb']:.2f}GB")
                if r_small["status"] == "ok"
                else (r_small["status"], "-")
            )
            bg_t = (
                (f"{r_big['us'] / 1000:.1f}", f"{r_big['peak_gb']:.2f}GB")
                if r_big["status"] == "ok"
                else (r_big["status"], "-")
            )
            print(
                f"{chunk_s:>10} {rec[0]:>9} {rec[1]:>9} {sm_t[0]:>9} {sm_t[1]:>10} {bg_t[0]:>9} {bg_t[1]:>10}"
            )

            # ----------------------------------------------------------------------
            # Sweep 2: chunk_T
            # ----------------------------------------------------------------------
            print
            print("=" * 90)
            print("SWEEP 2 — chunk_T (chunk_S=2048, top_k=512)")
            print("=" * 90)
            print(
                f"{'chunk_T':>10}"
                f" {'mean rec':>9} {'min rec':>9}"
                f" {'small ms':>9} {'small HBM':>10}"
                f" {'big ms':>9} {'big HBM':>10}"
            )
            print("-" * 90)
            for chunk_t in (1024, 4096, 16384, 65536, 262144):
                r_small = _run_one(
                    args.small_s,
                    chunk_s=2048,
                    chunk_t=chunk_t,
                    top_k=512,
                    n_heads=args.n_heads,
                    head_dim=args.head_dim,
                    dim=args.dim,
                    ratio=args.ratio,
                    measure_recall=True,
                )
                torch.cuda.empty_cache()
                r_big = _run_one(
                    args.big_s,
                    chunk_s=2048,
                    chunk_t=chunk_t,
                    top_k=512,
                    n_heads=args.n_heads,
                    head_dim=args.head_dim,
                    dim=args.dim,
                    ratio=args.ratio,
                    measure_recall=False,
                )
                torch.cuda.empty_cache()
                rec = (
                    (f"{r_small['recall'][0]:.4f}", f"{r_small['recall'][1]:.4f}")
                    if r_small["status"] == "ok"
                    else ("-", "-")
                )
                sm_t = (
                    (f"{r_small['us'] / 1000:.1f}", f"{r_small['peak_gb']:.2f}GB")
                    if r_small["status"] == "ok"
                    else (r_small["status"], "-")
                )
                bg_t = (
                    (f"{r_big['us'] / 1000:.1f}", f"{r_big['peak_gb']:.2f}GB")
                    if r_big["status"] == "ok"
                    else (r_big["status"], "-")
                )
                print(
                    f"{chunk_t:>10} {rec[0]:>9} {rec[1]:>9}"
                    f" {sm_t[0]:>9} {sm_t[1]:>10}"
                    f" {bg_t[0]:>9} {bg_t[1]:>10}"
                )

                # ----------------------------------------------------------------------
                # Sweep 3: top_k
                # ----------------------------------------------------------------------
                print
                print("=" * 90)
                print("SWEEP 3 — top_k (chunk_S=2048, chunk_T=8192)")
                print("=" * 90)
                print(
                    f"{'top_k':>10}"
                    f" {'mean rec':>9} {'min rec':>9}"
                    f" {'small ms':>9} {'small HBM':>10}"
                    f" {'big ms':>9} {'big HBM':>10}"
                )
                print("-" * 90)
                for top_k in (64, 256, 512, 1024, 2048):
                    r_small = _run_one(
                        args.small_s,
                        chunk_s=2048,
                        chunk_t=8192,
                        top_k=top_k,
                        n_heads=args.n_heads,
                        head_dim=args.head_dim,
                        dim=args.dim,
                        ratio=args.ratio,
                        measure_recall=True,
                    )
                    torch.cuda.empty_cache()
                    r_big = _run_one(
                        args.big_s,
                        chunk_s=2048,
                        chunk_t=8192,
                        top_k=top_k,
                        n_heads=args.n_heads,
                        head_dim=args.head_dim,
                        dim=args.dim,
                        ratio=args.ratio,
                        measure_recall=False,
                    )
                    torch.cuda.empty_cache()
                    rec = (
                        (f"{r_small['recall'][0]:.4f}", f"{r_small['recall'][1]:.4f}")
                        if r_small["status"] == "ok"
                        else ("-", "-")
                    )
                    sm_t = (
                        (f"{r_small['us'] / 1000:.1f}", f"{r_small['peak_gb']:.2f}GB")
                        if r_small["status"] == "ok"
                        else (r_small["status"], "-")
                    )
                    bg_t = (
                        (f"{r_big['us'] / 1000:.1f}", f"{r_big['peak_gb']:.2f}GB")
                        if r_big["status"] == "ok"
                        else (r_big["status"], "-")
                    )
                    print(
                        f"{top_k:>10} {rec[0]:>9} {rec[1]:>9}"
                        f" {sm_t[0]:>9} {sm_t[1]:>10}"
                        f" {bg_t[0]:>9} {bg_t[1]:>10}"
                    )


if __name__ == "__main__":
    main
