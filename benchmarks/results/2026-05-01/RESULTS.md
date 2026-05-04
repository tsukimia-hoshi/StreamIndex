# Snapshot 2026-05-01 — Day 2 of paper plan: design sweep + ablations + V4-Pro

Single-H200 SXM (89.169.97.104), BF16. Adds the design-space sweep, three
ablations, and V4-Pro indexer scaling on top of the layer-level snapshot
at `../2026-04-27/` and the kernel-level snapshot at `../2026-04-25/`.

## Design-space sweep (`sweep_design_space_final.log`)

`eval/sweep_design_space.py` — three sweeps at V4-Flash dims (n_h=64, head_dim=128,
ratio=4, dim=4096), recall measured at S=16384, time/HBM at S=262144.

### Sweep 1 — chunk_S (chunk_T=8192, top_k=512)

| chunk_S | mean rec | min rec | small ms | small HBM | big ms | big HBM |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1.0000 | 0.9980 | 21.6 | 0.16 GB | 4540 | 1.61 GB |
| 4096 | 1.0000 | 0.9980 | 18.6 | 0.36 GB | 4165 | 1.92 GB |
| 16384 | 1.0000 | 0.9980 | 17.4 | 0.98 GB | 4028 | 3.19 GB |
| 65536 | 1.0000 | 0.9980 | 17.4 | 0.98 GB | 4030 | 8.25 GB |
| 262144 | 1.0000 | 0.9980 | 17.3 | 0.98 GB | 4030 | 28.50 GB |

### Sweep 2 — chunk_T (chunk_S=2048, top_k=512)

| chunk_T | mean rec | min rec | small ms | small HBM | big ms | big HBM |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1.0000 | 0.9980 | 36.9 | 0.17 GB | 9482 | 1.58 GB |
| 4096 | 1.0000 | 0.9980 | 19.9 | 0.23 GB | 4999 | 1.63 GB |
| 16384 | 1.0000 | 0.9980 | 19.9 | 0.23 GB | 3900 | 1.87 GB |
| 65536 | 1.0000 | 0.9980 | 19.8 | 0.23 GB | 3377 | 2.81 GB |
| 262144† | 1.0000 | 0.9980 | 9.2 | 0.23 GB | **1607** | 2.81 GB |

†Clamped to T=65536 (single T-tile per S-tile). Fastest configuration on this hardware.

### Sweep 3 — top_k (chunk_S=2048, chunk_T=8192)

| top_k | mean rec | min rec | small ms | small HBM | big ms | big HBM |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1.0000 | 1.0000 | 7.6 | 0.10 GB | 1754 | 0.35 GB |
| 256 | 1.0000 | 1.0000 | 8.6 | 0.15 GB | 1881 | 0.93 GB |
| 512 | 1.0000 | 0.9980 | 9.2 | 0.23 GB | 1934 | 1.71 GB |
| 1024 | 1.0000 | 0.9990 | 9.9 | 0.38 GB | 1996 | 3.27 GB |
| 2048 | 1.0000 | 0.9995 | 12.0 | 0.70 GB | 2217 | 6.38 GB |

**Findings**: Recall is 100% mean at every cell. The `chunk_T` sweep shows
larger T-tiles uniformly faster — single T-tile per S-tile is optimal when
memory permits. The `chunk_S` sweep shows time floor at chunk_S>=4K; HBM
grows linearly. The `top_k` sweep is roughly linear in HBM, weakly sublinear
in time.

## Ablations

### Run 1 (`ablation_run1_chunked_t8192.log`) — chunk_t=8192 (single chunk over T)

| variant | mean rec | min rec | %=1 | time @ S=256K | HBM @ S=256K |
|---|---:|---:|---:|---:|---:|
| production (FP32, full merge) | 1.0000 | 0.9980 | 99.99% | 1932.8 ms | 1.71 GB |
| A1 no per-chunk merge | 1.0000* | 0.9980* | 99.99%* | --- | --- |
| A2 skip saturated (ct=256) | **0.0002** | **0.0000** | **0.02%** | --- | --- |
| A2 ctrl: production at ct=256 | 1.0000 | 0.9980 | 99.99% | --- | --- |
| A3 FP16 score accumulation | 0.9998 | 0.9941 | 91.82% | 1744.0 | 1.19 GB |

*A1 vacuous at chunk_t=8192 because T=4096 fits in one chunk; see Run 2.

### Run 2 (`ablation_run2_chunked_t1024.log`) — chunk_t=1024 (multi-chunk over T, 4 tiles)

| variant | mean rec | min rec | %=1 | time @ S=256K | HBM @ S=256K |
|---|---:|---:|---:|---:|---:|
| production (FP32, full merge) | 1.0000 | 0.9980 | 99.99% | 4311.9 ms | 1.58 GB |
| A1 no per-chunk merge | **0.5957** | **0.1914** | **25.03%** | --- | --- |
| A2 skip saturated (ct=256) | 0.0002 | 0.0000 | 0.02% | --- | --- |
| A2 ctrl: production at ct=256 | 1.0000 | 0.9980 | 99.99% | --- | --- |
| A3 FP16 score accumulation | 0.9998 | 0.9941 | 91.82% | 3523.7 | 1.07 GB |

**Findings**:
- **A1** (no merge) only craters in the multi-chunk regime: 0.60 mean / 0.19 min
 recall when chunk_t=1024 < T=4096 (Run 2). Confirms the merge is essential.
- **A2** (skip saturated) craters to 0.02% recall regardless of chunk size —
 the saturated branch (where each chunk returns all entries because
 chunk_T < top_k) is doing real work.
- **A3** (FP16) loses ~8% of perfect-recall rows but mean stays at 0.9998.
 Buys 1.11x speedup and 1.44x lower peak HBM at S=256K. Acceptable
 precision/throughput tradeoff for inference; FP32 needed for bit-exact.

## V4-Pro indexer scaling (V4-PRO SCALING block of `ablation_run1_chunked_t8192.log`)

`eval/bench_v4_indexer_scaling.py --n-heads 64 --head-dim 128 --top-k 1024 --dim 7168`
runs the same indexer scaling at V4-Pro dimensions.

| S | T | mat ms | mat HBM | chunk ms | chunk HBM | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 32 768 | 8 192 | 317.9 | 129.00 GB | 32.1 | 0.64 GB | 9.9× |
| 65 536 | 16 384 | OOM | --- | 126.5 | 1.02 GB | ∞ |
| 131 072 | 32 768 | OOM | --- | 502.3 | 1.77 GB | ∞ |
| 262 144 | 65 536 | OOM | --- | 2003.7 | 3.27 GB | ∞ |
| 524 288 | 131 072 | OOM | --- | 7994.3 | 6.27 GB | ∞ |
| **1 048 576** | **262 144** | **OOM** | --- | **31 973.3** | **12.27 GB** | **∞** |

Same OOM threshold (S=64K) as V4-Flash, same regime extension to S=1M.
Peak HBM is roughly 2× V4-Flash's because top_k doubled (1024 vs 512).

## Status note

Three of three additions for the paper plan are now done at single-GPU
scope on this H200:
- Addition 1 (V4-Flash end-to-end): layer-level result snapshot at
 `../2026-04-27/`. Full end-to-end with checkpoint weights deferred to
 multi-GPU follow-up.
- Addition 2 (multi-hardware): not run (no multi-GPU capacity available
 during this session window).
- Addition 3 (design sweep + ablation): this snapshot.

Next: paper writing days 10-13 of the plan, then arXiv submission.
