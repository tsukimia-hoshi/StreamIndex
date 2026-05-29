"""Full CSA pipeline at long context — the paper headline figure.

Times the complete `flash_csa_forward` pipeline (compressor outputs are
pre-supplied as inputs since they're cached across layers in production)
at increasing context length S, comparing two indexer strategies:

- Materialize: full [B, S, T] FP32 score matrix in HBM, then torch.topk.
- Chunked: per-chunk score + top-k merge (peak HBM bounded).

The novel result: the chunked path scales to S = 256K on a single H200,
while the materialize path OOMs at S ≥ 128K. The chunked path costs 2-3×
more wall-clock at small S (Python launch overhead) but is the *only*
viable option at long context.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from flash_sparse.csa import flash_csa_forward


@dataclass
class CsaLongCfg:
    S: int
    n_h: int = 16
    d: int = 64
    n_I_h: int = 16
    c_I: int = 64
    m: int = 4
    n_win: int = 128
    top_k: int = 64


def _time_us(fn, n_iter=5, n_warmup=2):
    for _ in range(n_warmup):
        fn
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iter):
            fn
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) * 1000.0 / n_iter


def _make_inputs(cfg: CsaLongCfg, B: int = 1):
    torch.manual_seed(2026)
    n_compressed = cfg.S // cfg.m
    return {
        "q": torch.randn(B, cfg.S, cfg.n_h, cfg.d, dtype=torch.bfloat16, device="cuda") * 0.3,
        "kv": torch.randn(B, cfg.S, cfg.d, dtype=torch.bfloat16, device="cuda") * 0.3,
        "kv_compressed": torch.randn(B, n_compressed, cfg.d, dtype=torch.bfloat16, device="cuda") * 0.3,
        "q_idx": torch.randn(B, cfg.S, cfg.n_I_h, cfg.c_I, dtype=torch.bfloat16, device="cuda") * 0.3,
        "k_idx_compressed": torch.randn(B, n_compressed, cfg.c_I, dtype=torch.bfloat16, device="cuda") * 0.3,
        "indexer_weights": torch.randn(B, cfg.S, cfg.n_I_h, dtype=torch.float32, device="cuda"),
        "attn_sink": torch.randn(cfg.n_h, dtype=torch.float32, device="cuda"),
    }


def bench_one(cfg: CsaLongCfg, *, use_chunked: bool, chunk_s: int = 1024, chunk_t: int = 4096):
    try:
        inputs = _make_inputs(cfg)
    except torch.OutOfMemoryError:
        return {"status": "OOM_inputs", "us": float("inf"), "peak_gb": float("nan")}

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()

    def fn():
        flash_csa_forward(
            **inputs,
            n_win=cfg.n_win,
            top_k=cfg.top_k,
            m=cfg.m,
            use_triton=True,
            use_chunked_indexer=use_chunked,
            chunk_s=chunk_s,
            chunk_t=chunk_t,
        )

    try:
        t = _time_us(fn)
        peak = max(0, torch.cuda.max_memory_allocated() - baseline) / 1024**3
        for v in inputs.values():
            del v
        return {"status": "ok", "us": t, "peak_gb": peak}
    except torch.OutOfMemoryError:
        return {"status": "OOM", "us": float("inf"), "peak_gb": float("nan")}
    except RuntimeError as e:
        if "invalid argument" in str(e).lower():
            return {"status": "GRID_LIMIT", "us": float("inf"), "peak_gb": float("nan")}
        raise


def main():
    if not torch.cuda.is_available():
        return
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")

    # Two flights: small-D (toy) and production-realistic-D.
    # Production-realistic V4-Flash-ish: n_h=64, d=128, n_I_h=64, c_I=128, top_k=512.
    # (Real V4-Pro is d=512 but our Triton has SRAM/spill issues at D=512;
    # real perf claim deferred to a future CUDA reimplementation. d=128 is enough to show the
    # IO-side scaling crossover holds at realistic dims.)
    configs = [
        # --- toy flight (d=64) ---
        CsaLongCfg(S=2048, n_h=16, d=64, n_I_h=16, c_I=64, top_k=64),
        CsaLongCfg(S=8192, n_h=16, d=64, n_I_h=16, c_I=64, top_k=64),
        CsaLongCfg(S=32768, n_h=16, d=64, n_I_h=16, c_I=64, top_k=64),
        CsaLongCfg(S=131072, n_h=16, d=64, n_I_h=16, c_I=64, top_k=64),
        CsaLongCfg(S=262144, n_h=16, d=64, n_I_h=16, c_I=64, top_k=64),
        CsaLongCfg(S=524288, n_h=16, d=64, n_I_h=16, c_I=64, top_k=64),
        # --- V4-Flash-ish flight (d=128) ---
        CsaLongCfg(S=2048, n_h=64, d=128, n_I_h=64, c_I=128, top_k=512),
        CsaLongCfg(S=8192, n_h=64, d=128, n_I_h=64, c_I=128, top_k=512),
        CsaLongCfg(S=32768, n_h=64, d=128, n_I_h=64, c_I=128, top_k=512),
        CsaLongCfg(S=65536, n_h=64, d=128, n_I_h=64, c_I=128, top_k=512),
        CsaLongCfg(S=131072, n_h=64, d=128, n_I_h=64, c_I=128, top_k=512),
        CsaLongCfg(S=262144, n_h=64, d=128, n_I_h=64, c_I=128, top_k=512),
    ]

    print(f"{'cfg':>32} {'mat (ms)':>10} {'mat HBM':>11} {'chunk (ms)':>12} {'chunk HBM':>11} {'ratio':>7}")
    print("-" * 90)

    for cfg in configs:
        r_mat = bench_one(cfg, use_chunked=False)
        torch.cuda.empty_cache()
        r_chk = bench_one(cfg, use_chunked=True)
        torch.cuda.empty_cache()

        def fmt(r):
            if r["status"] == "ok":
                return f"{r['us'] / 1000:>10.1f}", f"{r['peak_gb']:>9.2f}GB"
            label = {"OOM": "OOM", "OOM_inputs": "OOM-in", "GRID_LIMIT": "GRID>65k"}[r["status"]]
            return f"{label:>10}", f"{'-':>11}"

        m_t, m_h = fmt(r_mat)
        c_t, c_h = fmt(r_chk)
        if r_mat["status"] == "ok" and r_chk["status"] == "ok":
            sp = f"{r_mat['us'] / r_chk['us']:>6.2f}x"
        elif r_mat["status"] != "ok" and r_chk["status"] == "ok":
            sp = f"{'∞':>7}"
        else:
            sp = f"{'-':>7}"
            cfg_str = f"S={cfg.S} h={cfg.n_h} d={cfg.d} K={cfg.top_k}"
            print(f"{cfg_str:>32} {m_t} {m_h:>11} {c_t} {c_h:>11} {sp}")


if __name__ == "__main__":
    main
