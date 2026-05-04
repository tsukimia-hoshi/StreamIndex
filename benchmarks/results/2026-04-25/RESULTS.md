# Snapshot: 2026-04-25 paper-experiment run

H200 SXM (140 GB), BF16, single GPU. All raw logs in this directory.

## 1. TileLang verification (`tilelang_verification_20260425_215807.log`)

`bench_vs_tilelang_pipelined.py` — TileLang's `sparse_mla_fwd_pipelined`
attention kernel vs our Triton `sparse_attn_fwd`, V4-Flash dims (H=64,
dim=512, tail=64, K=512). Indices pre-computed; this is the *attention*
kernel comparison.

| S | TileLang ms | TileLang HBM | TileLang TFLOPS | Ours ms | Ours HBM | Ours TFLOPS |
|---:|---:|---:|---:|---:|---:|---:|
| 4 096 | 0.8 | 0.25 GB | 397.6 | 5.0 | 0.25 GB | 55.3 |
| 16 384 | 3.0 | 1.00 GB | 410.4 | 20.8 | 1.00 GB | 52.9 |
| 65 536 | 12.1 | 4.02 GB | 409.2 | 72.2 | 4.02 GB | 60.9 |
| 131 072 | 24.7 | 8.03 GB | 401.2 | 147.5 | 8.03 GB | 59.6 |
| 262 144 | 50.0 | 16.06 GB | 395.4 | 298.9 | 16.06 GB | 58.9 |

**Finding**: TileLang attention does NOT OOM at S=256K and is ~6× faster
than ours at every S (~40% H200 peak vs ~6%). The attention kernel is
not the bottleneck the paper can claim — that's the indexer.

## 2. Pipeline composition (HEADLINE) (`pipeline_20260425_224416.log`)

`bench_pipeline_composition.py` — full V4-Flash CSA pipeline (compressor +
indexer + TileLang attention) at V4-Flash dims, comparing materialize vs
chunked indexer with TileLang attention as the common backend.

| S | mat ms | mat HBM | chunk ms | chunk HBM | mat status |
|---:|---:|---:|---:|---:|:---|
| 8 192 | 6.5 | 0.58 GB | 7.5 | 0.58 GB | ok |
| 32 768 | 42.6 | 2.32 GB | 47.2 | 2.32 GB | ok |
| 65 536 | 132.9 | 4.64 GB | 152.3 | 4.64 GB | ok |
| 131 072 | 460.4 | **17.00 GB** | 531.3 | **9.28 GB** | ok |
| **262 144** | **OOM** | — | **1968.8** | **18.56 GB** | **OOM** |

**Finding**: at S=256K, the materialize indexer OOMs even when paired with
TileLang's production attention kernel. The chunked indexer + TileLang
attention is the only configuration that runs the full pipeline at S=256K
on a single H200. Wall-clock overhead: ~10–15% at sub-OOM S. Peak HBM at
S=128K: 1.83× lower with chunked.

## 3. Chunked-indexer recall (`recall_20260425_220154.log`)

`bench_chunked_indexer_recall.py` — set-overlap recall of chunked vs
materialize-then-`torch.topk`, V4-Flash indexer dims (H_I=64, D_I=128,
top_k=512). Tested two flights at S∈{8K, 32K} across (chunk_S, chunk_T)
∈ {(512,256), (512,2048), (1024,4096), (2048,4096), (2048,8192),
(4096,4096), (4096,8192), (8192,16384)}.

**Result**: 100% recall at every config. Mean = min = 1.0000;
%≤0.99 = 0.000%; ground-truth tie gap median = 2.3–2.8e-4 (ties exist
but FP non-associativity does not flip top-k entries).

## 4. V4-Pro dim limitations (`v4_pro_dims_20260425_222401.log`)

`bench_v4_pro_dims.py` — Triton sparse_attn_fwd + sparse_attn_bwd at
production-shape dims, S=4096, N_kv=4096.

| Config | fwd ms | fwd TFLOPS | fwd % peak | bwd ms | bwd TFLOPS | bwd % peak |
|---|---:|---:|---:|---:|---:|---:|
| V4-Flash (H=64, D=128, K=512) | 0.24 | 291.9 | **29.5%** | 1.87 | 183.3 | **18.5%** |
| V4-Flash-D (H=64, D=512, K=512) | 4.53 | 60.7 | **6.1%** | 335.4 | 4.1 | **0.4%** |
| V4-Pro K=1024 (H=128, D=512, K=1024) | 13.96 | 78.7 | **8.0%** | autotune-timeout | — | — |

K=2048 row dropped from this run (autotune sweep exceeded session budget).
H200 BF16 dense peak assumed = 989 TFLOPS.

**Finding**: Triton-vs-CUDA gap at production D=512 is real and large
(~6% peak fwd, <1% bwd). The chunked indexer composes with TileLang's
CUDA attention, removing this gap from the production path.

## 5. Auxiliary toy-dim scaling (`long_context_fwdbwd_20260425_224918.log`)

`bench_long_context_fwd_bwd.py` — Triton sparse_attn fwd+bwd at toy dims
(H=16, D=64, K=64), demonstrating scaling.

| S | fwd ms | fwd TFLOPS | bwd ms | bwd TFLOPS | peak HBM |
|---:|---:|---:|---:|---:|---:|
| 4 096 | 0.05 | 23.3 | 0.14 | 39.7 | 0.27 GB |
| 16 384 | 0.07 | 61.3 | 0.26 | 81.2 | 0.07 GB |
| 65 536 | 0.25 | 68.9 | 1.02 | 83.9 | 0.28 GB |
| 131 072 | 0.49 | 70.5 | 2.28 | 75.5 | 0.56 GB |
| 262 144 | 0.97 | 70.9 | 4.86 | 70.7 | 1.13 GB |
| **524 288** | **1.94** | **70.7** | **10.14** | **67.8** | **2.25 GB** |

**Finding**: linear scaling, sustained ~70 TFLOPS, bounded HBM up to 524K.
Auxiliary kernel; not the headline.

## Status note (honest)

These are kernel-level results. They are necessary but not sufficient for
a top-tier paper — the technique (partition-merge top-k) is folklore from
streaming algorithms; the contribution is the engineering application to
V4 CSA and the open-source impl. Three additions are needed to upgrade
this from "kernel benchmark" to "drop-in optimization for a frontier
model" — see the project roadmap.
