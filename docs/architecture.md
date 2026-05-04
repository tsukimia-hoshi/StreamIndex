# Architecture

## Project layout

```
StreamIndex/
├── flash_sparse/
│ ├── __init__.py public API surface
│ ├── csa.py flash_csa_forward (CSA wrapper)
│ ├── hca.py flash_hca_forward (HCA wrapper)
│ ├── reference.py pure-pytorch ground truth (slow, correct)
│ ├── cuda/ CUDA kernel sources (future work+)
│ ├── triton/ Triton kernel prototypes (future work+)
│ └── utils/
│ ├── compression.py softmax-gated KV pooling
│ └── quantization.py FP8 / FP4 simulation helpers
├── tests/ correctness gates
├── benchmarks/ perf measurement harnesses
├── docs/ math, IO complexity, this file
└── paper/ arXiv submission (future work)
```

## Reference identity

The single optimized kernel that powers both CSA and HCA is **`sparse_attn`**:

```
o = sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)
```

with shapes `q[b, s, h, d_qk]`, `kv[b, n_kv, d_qk]` (MQA), `attn_sink[h]`, `topk_idxs[b, s, K]` (int32, ‑1 = masked). The two attention modes differ only in what's stuffed into `topk_idxs`:

| Mode | `topk_idxs` payload |
|------|-------------------------------------------------------------------------------|
| CSA | `cat([sliding_window(n_win), indexer_topk(top_k of compressed seq, m=4)])` |
| HCA | `cat([sliding_window(n_win), all_compressed_positions(m_prime=128)])` |

This means:
- The **lightning indexer** (FP4 q × FP8 k_compressed → ReLU-summed score → top-k) is required only for CSA layers.
- The **token compressor** (softmax-gated weighted pooling) is required for both.
- The **sparse attention** kernel is identical for both — same code, different inputs.

## What FlashSparse adds

The DeepSeek production stack (FlashMLA + DeepGEMM + TileLang `deepseek_v32`) already implements these on H800. FlashSparse contributes:

1. **Streaming top-k inside the kernel.** The reference materializes a full `[s, n_compressed]` indexer score matrix in HBM. We maintain a per-query top-k heap incrementally as we stream the compressed K, eliminating the score-matrix HBM round-trip.

2. **Indexer + top-k + sparse attention fused into one persistent kernel.** The reference issues 3 separate kernels (indexer, top-k selector, sparse MLA). We fuse them with producer-consumer warp specialization on Hopper.

3. **Backward without dKV atomics.** TileLang's reference uses atomic_addx4 to scatter dKV gradients back to selected indices. We use a 2-pass design (one CTA per K block, sum-reduce dQ from saved P) inspired by FA3 / FA4.

4. **Tuned for H200's 132 SMs.** DeepSeek's published numbers are H800/B200; FlashMLA + DeepGEMM cite no H200 throughput. We tile and persist for the H200 SM count specifically.

## Distribution channels (future work)

- pip-installable: `pip install flash-sparse`.
- Python entry points: `flash_sparse.flash_csa_forward`, `flash_sparse.flash_hca_forward`.
- Drop-in replacement behind SGLang's `--nsa-prefill-backend` / `--nsa-decode-backend` hooks.
- arXiv paper documenting the streaming top-k correctness proof + IO complexity analysis.
