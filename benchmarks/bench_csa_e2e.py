"""End-to-end CSA forward benchmark.

Times the full `flash_csa_forward` pipeline (indexer score → top-k → sparse_attn)
vs the pure-pytorch reference path. This is the closest thing we have to a
real-workload measurement until future work (FlashMLA comparison) lands.

Reports per-config latency (ms) and effective TFLOPS for both paths.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from flash_sparse.csa import flash_csa_forward


@dataclass
class CsaConfig:
    B: int
    S: int
    n_h: int
    d: int
    n_I_h: int
    c_I: int
    m: int
    n_win: int
    top_k: int


def _time_us(fn, n_iter=20, n_warmup=5):
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
            return start.elapsed_time(end) * 1000.0 / n_iter  # us


def bench_csa_e2e(cfg: CsaConfig) -> dict:
    torch.manual_seed(0)
    n_compressed = cfg.S // cfg.m

    q = torch.randn(cfg.B, cfg.S, cfg.n_h, cfg.d, dtype=torch.bfloat16, device="cuda") * 0.5
    kv = torch.randn(cfg.B, cfg.S, cfg.d, dtype=torch.bfloat16, device="cuda") * 0.5
    kv_compressed = torch.randn(cfg.B, n_compressed, cfg.d, dtype=torch.bfloat16, device="cuda") * 0.5
    q_idx = torch.randn(cfg.B, cfg.S, cfg.n_I_h, cfg.c_I, dtype=torch.bfloat16, device="cuda") * 0.5
    k_idx_compressed = torch.randn(cfg.B, n_compressed, cfg.c_I, dtype=torch.bfloat16, device="cuda") * 0.5
    weights = torch.randn(cfg.B, cfg.S, cfg.n_I_h, dtype=torch.float32, device="cuda")
    attn_sink = torch.randn(cfg.n_h, dtype=torch.float32, device="cuda")

    triton_fn = lambda: flash_csa_forward(
        q,
        kv,
        kv_compressed,
        q_idx,
        k_idx_compressed,
        weights,
        attn_sink,
        n_win=cfg.n_win,
        top_k=cfg.top_k,
        m=cfg.m,
        use_triton=True,
    )
    ref_fn = lambda: flash_csa_forward(
        q,
        kv,
        kv_compressed,
        q_idx,
        k_idx_compressed,
        weights,
        attn_sink,
        n_win=cfg.n_win,
        top_k=cfg.top_k,
        m=cfg.m,
        use_triton=False,
    )

    triton_us = _time_us(triton_fn)
    ref_us = _time_us(ref_fn)

    # Approximate FLOPs: indexer (n_I_h · c_I · n_compressed · S) + sparse_attn (~ 2 · n_h · top_k · d · S)
    indexer_flops = cfg.B * cfg.S * cfg.n_I_h * cfg.c_I * n_compressed * 2
    attn_flops = 2 * 2 * cfg.B * cfg.S * cfg.n_h * (cfg.top_k + cfg.n_win) * cfg.d
    total_flops = indexer_flops + attn_flops

    return {
        "config": cfg,
        "triton_us": triton_us,
        "ref_us": ref_us,
        "triton_tflops": total_flops / (triton_us * 1e-6) / 1e12,
        "ref_tflops": total_flops / (ref_us * 1e-6) / 1e12,
        "speedup": ref_us / triton_us if triton_us > 0 else 0,
    }


def main():
    if not torch.cuda.is_available():
        return
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}\n")

    configs = [
        # B, S, n_h, d, n_I_h, c_I, m, n_win, top_k
        CsaConfig(B=1, S=256, n_h=16, d=64, n_I_h=16, c_I=64, m=4, n_win=16, top_k=32),
        CsaConfig(B=1, S=512, n_h=32, d=64, n_I_h=32, c_I=64, m=4, n_win=32, top_k=64),
        CsaConfig(B=1, S=1024, n_h=64, d=64, n_I_h=32, c_I=64, m=4, n_win=64, top_k=128),
        CsaConfig(B=1, S=2048, n_h=64, d=64, n_I_h=64, c_I=64, m=4, n_win=64, top_k=256),
        # V4-Pro-shaped (with reduced d to fit memory in test):
        CsaConfig(B=1, S=1024, n_h=128, d=64, n_I_h=64, c_I=64, m=4, n_win=128, top_k=128),
    ]

    print(
        f"{'config':<60} {'triton (us)':>13} {'ref (us)':>11}"
        f" {'triton TFLOPS':>15} {'ref TFLOPS':>13} {'speedup':>9}"
    )
    print("-" * 130)
    for cfg in configs:
        r = bench_csa_e2e(cfg)
        cfg_s = (
            f"S={cfg.S} n_h={cfg.n_h} d={cfg.d} n_I_h={cfg.n_I_h} c_I={cfg.c_I} m={cfg.m} top_k={cfg.top_k}"
        )
        print(
            f"{cfg_s:<60} {r['triton_us']:>13.1f} {r['ref_us']:>11.1f}"
            f" {r['triton_tflops']:>15.2f} {r['ref_tflops']:>13.2f} {r['speedup']:>9.2f}"
        )


if __name__ == "__main__":
    main
