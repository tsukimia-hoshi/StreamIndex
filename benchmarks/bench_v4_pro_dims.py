"""V4-Pro production-dim benchmark — own the Triton-vs-CUDA gap honestly.

leadership over a tuned MLA kernel at production dims. This bench measures
exactly *what* peak utilization our Triton sparse_attn_fwd hits at the V4-
Pro shape (H=128, D=512, K=1024) so we can report it as a limitation rather
than gloss over it.

Three configs:
- V4-Flash (H=64, D=128, K=512)
- V4-Flash-D (H=64, D=512, K=512) — V4-Flash with V4-Pro head-dim
- V4-Pro (H=128, D=512, K=1024)

For each, we measure forward time, peak HBM, achieved TFLOPS, and the H200
peak utilization (vs. 989 TFLOPS BF16 dense peak).

Run as: python benchmarks/bench_v4_pro_dims.py
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from flash_sparse.triton.sparse_attn_fwd import sparse_attn_fwd
from flash_sparse.triton.sparse_attn_bwd import sparse_attn_bwd


# H200 BF16 dense peak (per nvidia.com/h200): ~989 TFLOPS.
H200_BF16_TFLOPS = 989.0


@dataclass
class V4Cfg:
    name: str
    H: int
    D: int
    K: int
    S: int = 4096
    N_kv: int = 4096
    B: int = 1


def _time_us(fn, n_iter=20, n_warmup=5):
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
            return e0.elapsed_time(e1) * 1000.0 / n_iter  # us


def _measure_peak(fn) -> float:
    """Kernel-only peak working set (excludes inputs)."""
    fn  # JIT warmup
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn
    torch.cuda.synchronize()
    return max(0, torch.cuda.max_memory_allocated() - baseline) / 1024**3


def bench_fwd(cfg: V4Cfg) -> dict:
    try:
        q = torch.randn(cfg.B, cfg.S, cfg.H, cfg.D, dtype=torch.bfloat16, device="cuda") * 0.1
        kv = torch.randn(cfg.B, cfg.N_kv, cfg.D, dtype=torch.bfloat16, device="cuda") * 0.1
        attn_sink = torch.randn(cfg.H, dtype=torch.float32, device="cuda")
        # Vectorized random causal indices.
        idx = torch.randint(0, cfg.N_kv, (cfg.B, cfg.S, cfg.K), dtype=torch.int32, device="cuda")
        t = torch.arange(cfg.S, dtype=torch.int32, device="cuda").view(1, cfg.S, 1)
        idx = torch.minimum(idx, t).contiguous()
        sm = cfg.D**-0.5

        def fn():
            return sparse_attn_fwd(q, kv, attn_sink, idx, sm)

        peak = _measure_peak(fn)
        us = _time_us(fn)
        flops = 2 * 2 * cfg.B * cfg.S * cfg.H * cfg.K * cfg.D
        tflops = flops / (us * 1e-6) / 1e12
        del q, kv, attn_sink, idx
        torch.cuda.empty_cache()
        return {
            "status": "ok",
            "us": us,
            "peak_gb": peak,
            "tflops": tflops,
            "peak_pct": 100 * tflops / H200_BF16_TFLOPS,
        }
    except torch.OutOfMemoryError:
        return {"status": "OOM"}
    except RuntimeError as e:
        return {"status": "ERROR", "msg": str(e)[:120]}


def bench_bwd(cfg: V4Cfg) -> dict:
    try:
        q = torch.randn(cfg.B, cfg.S, cfg.H, cfg.D, dtype=torch.bfloat16, device="cuda") * 0.1
        kv = torch.randn(cfg.B, cfg.N_kv, cfg.D, dtype=torch.bfloat16, device="cuda") * 0.1
        attn_sink = torch.randn(cfg.H, dtype=torch.float32, device="cuda")
        idx = torch.randint(0, cfg.N_kv, (cfg.B, cfg.S, cfg.K), dtype=torch.int32, device="cuda")
        t = torch.arange(cfg.S, dtype=torch.int32, device="cuda").view(1, cfg.S, 1)
        idx = torch.minimum(idx, t).contiguous()
        do = torch.randn(cfg.B, cfg.S, cfg.H, cfg.D, dtype=torch.bfloat16, device="cuda") * 0.1
        sm = cfg.D**-0.5

        o, lse = sparse_attn_fwd(q, kv, attn_sink, idx, sm)

        def fn():
            return sparse_attn_bwd(q, kv, attn_sink, idx, o, lse, do, sm)

        peak = _measure_peak(fn)
        us = _time_us(fn)
        # Backward: ~5x forward FMAs (recompute P, dP·V, dQ, dK, dV).
        flops = 5 * 2 * 2 * cfg.B * cfg.S * cfg.H * cfg.K * cfg.D
        tflops = flops / (us * 1e-6) / 1e12
        del q, kv, attn_sink, idx, do, o, lse
        torch.cuda.empty_cache()
        return {
            "status": "ok",
            "us": us,
            "peak_gb": peak,
            "tflops": tflops,
            "peak_pct": 100 * tflops / H200_BF16_TFLOPS,
        }
    except torch.OutOfMemoryError:
        return {"status": "OOM"}
    except RuntimeError as e:
        return {"status": "ERROR", "msg": str(e)[:120]}


def main():
    if not torch.cuda.is_available():
        return
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"H200 BF16 dense peak (assumed): {H200_BF16_TFLOPS:.0f} TFLOPS\n")

    configs = [
        V4Cfg(name="V4-Flash", H=64, D=128, K=512, S=4096, N_kv=4096),
        V4Cfg(name="V4-Flash-D", H=64, D=512, K=512, S=4096, N_kv=4096),
        V4Cfg(name="V4-Pro K=1024", H=128, D=512, K=1024, S=4096, N_kv=4096),
        # K=2048 widest V4-Pro operating point — Triton bwd autotune sweep
        # at this shape was costing >20 min on our setup. Run that row
        # separately via PYTHONPATH=... K_ONLY=2048 if needed for §6.1.
    ]

    print("=== FORWARD (sparse_attn_fwd) ===")
    print(f"{'config':<32} {'time (ms)':>10} {'peak (GB)':>10} {'TFLOPS':>9} {'% peak':>8}")
    print("-" * 75)
    for cfg in configs:
        r = bench_fwd(cfg)
        head = f"{cfg.name} (H={cfg.H} D={cfg.D} K={cfg.K})"
        if r["status"] == "ok":
            print(
                f"{head:<32} {r['us'] / 1000:>10.2f} {r['peak_gb']:>10.3f}"
                f" {r['tflops']:>9.1f} {r['peak_pct']:>7.1f}%"
            )
        else:
            print(f"{head:<32} {r['status']:>10}" + (f" {r.get('msg', '')}" if r.get("msg") else ""))

            print("\n=== BACKWARD (sparse_attn_bwd v1, atomic-scatter) ===")
            print(f"{'config':<32} {'time (ms)':>10} {'peak (GB)':>10} {'TFLOPS':>9} {'% peak':>8}")
            print("-" * 75)
            for cfg in configs:
                r = bench_bwd(cfg)
                head = f"{cfg.name} (H={cfg.H} D={cfg.D} K={cfg.K})"
                if r["status"] == "ok":
                    print(
                        f"{head:<32} {r['us'] / 1000:>10.2f} {r['peak_gb']:>10.3f}"
                        f" {r['tflops']:>9.1f} {r['peak_pct']:>7.1f}%"
                    )
                else:
                    print(f"{head:<32} {r['status']:>10}" + (f" {r.get('msg', '')}" if r.get("msg") else ""))

                    print("\nNotes:")
                    print("- % peak = TFLOPS / H200 BF16 dense peak. Lower = larger Triton-vs-CUDA gap.")
                    print("- Backward FLOP count = 5x forward (standard FA convention).")
                    print("- Per paper outline §6.1: this gap is the limitation we own; closing it")
                    print(" is the path for the CUDA forward (FA3 patterns).")


if __name__ == "__main__":
    main
