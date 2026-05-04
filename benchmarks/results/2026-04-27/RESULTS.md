# Snapshot 2026-04-27 — V4-Flash layer-level result on 1× H200

Single-H200 SXM (89.169.97.104), BF16. Adds the *layer-level* result on top
of the kernel-level snapshot at `../2026-04-25/`. Raw logs in this directory.

## V4-Flash Indexer parity test (`parity_v4_flash_indexer_final.log`)

`eval/test_v4_indexer_parity.py` — chunked indexer (`chunked_indexer_topk`
with the new `causal_ratio` per-chunk-mask path) vs the V4-Flash reference
materialize+topk+postmask code path from `references/DeepSeek-V4-Flash/
inference/model.py:415-431`. Inputs are synthetic-but-realistic at V4-Flash
dims (n_h=64, head_dim=128, top_k=512, ratio=4, dim=4096; q ∼ N(0, 1/√d),
kv_cache ∼ N(0, 1/√d), weights from a fresh `nn.Linear(4096, 64)` scaled
by `1/√(d · n_h)`).

Pass criterion: bit-exact set match per row (mean = min = 1.0).

| S | T | mean recall | min recall | %=1 | %<.99 | verdict |
|---:|---:|---:|---:|---:|---:|---|
| 2048 | 512 | 1.0000 | 1.0000 | 100.00% | 0.000% | **PASS** |
| 4096 | 1024 | 1.0000 | 1.0000 | 100.00% | 0.000% | **PASS** |
| 8192 | 2048 | 1.0000 | 1.0000 | 100.00% | 0.000% | **PASS** |

The chunked indexer is bit-exact identical to the materialize+postmask
reference at V4-Flash dims with realistic per-element variance.

## V4-Flash Indexer S-scaling (`scaling_v4_flash_indexer_final.log`)

`eval/bench_v4_indexer_scaling.py` — same code paths, sweeps S ∈ {32K, 64K,
128K, 256K, 512K, 1M}. Materialize uses the reference's `[B, S, H_I, T]`
FP32 intermediate (the einsum output, before the head-sum); chunked uses
the per-chunk path with `causal_ratio=4`.

| S | T | mat ms | mat HBM | chunk ms | chunk HBM | speedup |
|---:|---:|---:|---:|---:|---:|---|
| 32 768 | 8 192 | 317.0 | **129.00 GB** | 30.8 | 0.40 GB | 10.3× |
| 65 536 | 16 384 | **OOM** | — | 122.2 | 0.59 GB | ∞ |
| 131 072 | 32 768 | OOM | — | 484.7 | 0.96 GB | ∞ |
| 262 144 | 65 536 | OOM | — | 1935.3 | 1.71 GB | ∞ |
| 524 288 | 131 072 | OOM | — | 7730.1 | 3.21 GB | ∞ |
| **1 048 576** | **262 144** | **OOM** | — | **30 899.6** | **6.21 GB** | **∞** |

Materialize OOMs at S=64K (the `[B, S, H_I, T]` intermediate is 256 GB
FP32 at S=64K with V4-Flash's H_I=64 indexer heads — 64× the size of
the kernel-level `[B, S, T]` score matrix). Chunked runs to S=1M with
6.21 GB peak.

**Regime extension on a single H200: 32×** (S=32K → S=1M).

## Engineering changes shipped this snapshot

`flash_sparse/triton/chunked_indexer.py`: added `causal_ratio: Optional[int]`
parameter that computes the V4 causal mask `t < (s+1)//ratio` per `[chunk_S,
chunk_T]` block instead of materializing the global `[B, S, T]` bool tensor.
Required to scale chunked path past S=128K — the global mask alone is
4 GB at S=128K, 256 GB at S=1M.

## What this is and isn't

**Is**: layer-level algorithmic parity at V4-Flash dims, validated on
synthetic distributions matching post-projection variance. The materialize
OOM threshold and chunked extension are real and reproduce on H200.

**Isn't**: end-to-end inference with checkpoint-trained V4-Flash weights.
That requires 2-8× H200/H100 with weight offloading and is the natural
next addition once multi-GPU capacity is available.
