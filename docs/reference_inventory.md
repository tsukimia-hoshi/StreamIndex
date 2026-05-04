# Reference inventory

Every kernel and helper we will replicate or beat, with file paths and language. All paths are relative to
`references/` in the repo (cloned 2026-04-25, no LFS weights). DeepSeek-V4-Pro inference code = correctness reference. TileLang DSA + FlashMLA + DeepGEMM = performance baselines.

## Correctness reference: DeepSeek-V4 inference

`DeepSeek-V4-Pro/` (and `DeepSeek-V4-Flash/`, same layout):

| File | Lang | What it has |
|---------------------------------|--------|-------------|
| `inference/model.py` | python | `Compressor`, `Indexer`, `Attention`, `Block`, `MTPBlock`, `Transformer`, `Gate`, `MoE`, `Expert`. Calls TileLang kernels via `kernel.py`. The single source of truth for the math. |
| `inference/kernel.py` | python (tilelang) | `act_quant`, `fp4_act_quant`, `fp8_gemm`, `fp4_gemm`, `sparse_attn`, `hc_split_sinkhorn`. Each is a `@tilelang.jit` `T.prim_func`. Disables warp_specialized + tma lowering for portability. |
| `inference/generate.py` | python | torchrun entry point. Distributed inference loop. |
| `inference/convert.py` | python | HF safetensors → DeepSeek-internal weight layout converter. |
| `inference/config.json` | json | Mirror of HF config.json with renamed keys (matches `ModelArgs` dataclass in model.py). |
| `inference/requirements.txt` | txt | `torch>=2.10.0, transformers>=5.0.0, safetensors>=0.7.0, fast_hadamard_transform, tilelang==0.1.8`. |
| `encoding/encoding_dsv4.py` | python | OpenAI-format-message → token-string encoder/parser (chat template). Out of scope for kernels. |
| `config.json` | json | HF transformers config. Authoritative for hyperparameters (see `docs/hyperparameters.yaml`). |
| `DeepSeek_V4.pdf` | pdf | Tech report. Section 2.3 covers CSA/HCA architecture (extracted to `DeepSeek_V4.txt`). |

## Performance baselines: TileLang `deepseek_v32` examples

`tilelang/examples/deepseek_v32/`:

| File | Lang | What it does | Maps to |
|-----------------------------------|----------|--------------|---------|
| `fp8_lighting_indexer.py` | tilelang | `mqa_attn_return_logits` — FP8 indexer kernel: q×K + ReLU + weighted sum over heads. Variable-len sequences (`cu_seqlen_ks/ke`). Has pure-pytorch `ref_fp8_mqa_logits` for testing. | Step 1 of CSA: lightning indexer |
| `topk_selector.py` | tilelang | `tl_topk` — radix-sort-based exact top-k. 8-bit histogram + 4-round refine. NOT streaming. | Step 2 of CSA: top-k selection |
| `sparse_mla_fwd.py` | tilelang | `sparse_mla_fwd` — sparse-MLA forward over selected indices. Online softmax + LSE return. Has pure-pytorch `ref_sparse_mla_fwd_interface` for testing. | Step 3 of CSA & full HCA: sparse attention |
| `sparse_mla_fwd_pipelined.py` | tilelang | Pipelined producer-consumer version of `sparse_mla_fwd`. Manually splits warp groups. ~600 TFlops on H800 SXM. | Phase-2 inspiration for our fused kernel |
| `sparse_mla_fwd_seesaw.py` | tilelang | Variant for testing alternative scheduling. | Reference for tile scheduling experiments |
| `sparse_mla_bwd.py` | tilelang | Sparse-MLA backward: preprocess (Δ), main (dQ + dKV w/ atomic_addx4), postprocess. ~115 TFlops on H200 SXM (per tilelang README). | Phase-3 reference |
| `topk_selector.py` | tilelang | Exact top-k via histogram. | Phase-2 streaming-top-k baseline |
| `regression_tilelang_example_deepseek_v32.py` | python | Driver that runs the perf regression on all kernels. | Use for our perf comparisons |
| `test_tilelang_example_deepseek_v32.py` | python | Correctness test driver. | Use for our correctness comparisons |
| `utils.py` | python | `assert_tensors_similar`, `generate_random_cu_seqlens`, `per_custom_dims_cast_to_fp8`. | Bench / test helpers |

## Performance baselines: FlashMLA (production CUDA)

`FlashMLA/`:

| File | Lang | What it does |
|--------------------------------------------|-------|--------------|
| `csrc/params.h` | C++ | Param structs: `SparseAttnFwdParams`, `SparseAttnDecodeParams`, `DenseAttnDecodeParams`, `CombineParams`, `GetDecodeSchedMetaParams`, `DecodingSchedMeta`. Defines `ModelType {V32, MODEL1}` (likely `MODEL1` = V4). |
| `csrc/defines.h`, `csrc/utils.h` | C++ | shared CUDA/cutlass helpers |
| `flash_mla/flash_mla_interface.py` | python | Public python API: `flash_mla_with_kvcache` (decode), `flash_mla_sparse_fwd` (prefill), `flash_attn_varlen_func` (dense). Returns `(out, lse)`. Supports `attn_sink`, `extra_k_cache`/`extra_indices_in_kvcache` for hybrid (CSA + HCA?). |
| `flash_mla/__init__.py` | python | exports |
| `tests/ref.py` | python | Pytorch reference for sparse fwd / decode. Almost identical to what we need. |
| `tests/test_flash_mla_sparse_prefill.py` | python | Sparse prefill correctness test (CUDA vs ref). |
| `tests/test_flash_mla_sparse_decoding.py` | python | Sparse decode correctness + throughput. FP8 KV cache. |
| `tests/test_flash_mla_dense_decoding.py` | python | Dense decode (no top-k). |
| `tests/test_fmha_sm100.py` | python | Dense MHA prefill on SM100 (B200). NVIDIA contributed. |
| `tests/quant.py`, `tests/lib.py` | python | Quantization helpers + test harness types. |
| `benchmark/bench_flash_mla.py` | python | Reproduces published TFLOPS numbers. |
| `setup.py` | python | CUDA extension build. Targets SM90 + SM100. CUDA 12.8+. |
| `README.md` | md | Performance: 660 TFLOPS dense decode, 410 TFLOPS sparse decode, 640 TFLOPS sparse prefill on H800 SXM5. 1450 TFLOPS sparse prefill on B200. **No H200 numbers published.** |

## Performance baselines: DeepGEMM (production CUDA, FP8/FP4 GEMM + MQA indexer scoring)

`DeepGEMM/`:

| File | Lang | What it does |
|---------------------------------------|--------|--------------|
| `deep_gemm/__init__.py` | python | Exports JIT-compiled GEMM kernels. |
| `tests/test_attention.py` | python | Tests `fp8_mqa_logits` (non-paged) + `fp8_paged_mqa_logits` (paged) — these are the production lightning-indexer-equivalent kernels. |
| `tests/test_hyperconnection.py` | python | Tests Hyper-Connection mixing (mHC). |
| `tests/test_fp8_fp4.py` | python | FP8 + FP4 GEMM correctness. |
| `tests/test_mega_moe.py` | python | Fused dispatch + GEMM + SwiGLU + GEMM + combine kernel. |
| `tests/test_bf16.py`, `test_einsum.py`, `test_lazy_init.py`, `test_layout.py`, `test_legacy.py`, `test_sanitizer.py`, `generators.py` | python | misc unit tests + test data generators |
| `setup.py` | python | JIT-only install (no CUDA build at install). |
| `README.md` | md | Performance: 1550 TFLOPS GEMM on H800. PR #200 added MQA indexer scoring (Sep 2025). PR #304 added FP4 indexer + Mega MoE (Apr 2026). |

## Other reference clones

| Repo | Path under `references/` | Use |
|-------------------------------------|---------------------------------|-----|
| FlashAttention | `flash-attention/` | FA2/FA3 reference for tiling, online softmax, varlen, paged attention patterns. |
| Triton | `triton/` | Triton DSL source — for understanding what TritonIR primitives are available. |
| TileLang | `tilelang/` | DSA/CSA reference kernels (above) plus the broader TileLang library. |
| DeepSeek-V3.2-Exp | `DeepSeek-V3.2-Exp/` | Predecessor model with DSA only (no CSA/HCA). Useful for comparing the V3.2→V4 architectural diff. |
| DeepSeek-V4-Flash | `DeepSeek-V4-Flash/` | Smaller V4 variant. Same architecture, smaller dims. Good for fast tests. |

## Checklist for future work hyperparameters and reference paths

- [x] CSA compression ratio `m` = 4 — `compress_ratios[2..60:2]` in V4-Pro config.
- [x] HCA compression ratio `m_prime` = 128 — `compress_ratios[0,1,3..59:2]` in V4-Pro config.
- [x] hyper-connection width `n_hc` = 4 — `hc_mult`.
- [x] CSA top-k `top_k` = 1024 — `index_topk`.
- [x] sliding window `n_win` = 128 — `sliding_window`.
- [x] head_dim `c` = 512 — `head_dim`.
- [x] num query heads `n_h` = 128 — `num_attention_heads`.
- [x] indexer head_dim `c_I` = 128 — `index_head_dim`.
- [x] indexer head count `n_I_h` = 64 — `index_n_heads`.
- [x] output groups `g` = 16 — `o_groups`.
- [x] per-group intermediate dim `d_g` = 1024 — `o_lora_rank`.
- [x] query compression dim `d_c` = 1536 — `q_lora_rank`.
- [x] num layers = 61 — `num_hidden_layers`.
- [x] CSA layer indices = `{2,4,...,60}` — derived from `compress_ratios`.
- [x] HCA layer indices = `{0,1,3,5,...,59}` — derived from `compress_ratios`.
- [x] precision policy: FP8 e4m3 weights (block 128×128, ue8m0 scale), FP4 e2m1 MoE experts (block 32 along K), FP4 indexer queries (Hadamard rotated), FP8 indexer keys, BF16 RoPE dims.
- [x] Reference paths recorded above.
