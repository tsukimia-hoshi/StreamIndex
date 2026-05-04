# V4-Flash End-to-End Integration — Plan

> **Status**: 2026-04-25 plan only. Execution is **blocked on V4-Flash
> checkpoint access**. This doc captures the integration path so the moment
> a checkpoint is on the VM, the work is unambiguous.

## What we are claiming and why this matters

The paper currently has kernel-level evidence: chunked CSA forward at
S=256K with V4-Flash-realistic dims (H=64, D=128, n_I_h=64, c_I=128, top_k=512)
runs in 2.5 s with 24.5 GB peak HBM, where the materialize path OOMs.
What we **do not have** is end-to-end evidence — we never load real V4 weights
and check that swapping the kernel preserves model behaviour and produces
correct tokens at long context.

A reviewer (and the user) will rightly demand:
1. *Perplexity drift*: with our kernel substituted for the reference, does
 perplexity on a held-out long-context corpus stay within ε of reference?
2. *Tokens/sec*: at S=256K prefill, what is the wall-clock end-to-end
 throughput vs the materialize-OOM-path baseline (which doesn't exist —
 so vs an alternative implementation, e.g. attention-streamed-block).
3. *Quality at the chunk boundary*: does the chunked top-k select KV pairs
 that, when fed through the actual sparse-attn kernel, produce outputs
 indistinguishable (within FP32 noise) from the reference path?

This plan describes how to answer all three.

## Required artifacts (block-list)

- [ ] V4-Flash checkpoint (HF format or DeepSeek's released format).
 Likely candidates as of 2026-04-25:
 - `deepseek-ai/DeepSeek-V4-Flash-Base` (HF)
 - `deepseek-ai/DeepSeek-V3.2-exp` (HF, released 2025; CSA precursor)
 - The user's own internal V4-Flash trained variant if available.
- [ ] HuggingFace Transformers ≥ 4.50 (or the DeepSeek-released loader).
- [ ] A long-context evaluation corpus, e.g.
 - `RULER-256K` (or a subset thereof)
 - PG19 long-form fiction (stitch documents to S=256K)
 - The user's domain-specific eval set if one exists
- [ ] H200 access with the package installed
 (`pip install -e .` from the repo root).

## Integration steps (ordered)

### Step 1: locate the CSA forward call in the V4-Flash modelling code

The DeepSeek-V3.2-exp HF release exposes
`modeling_deepseek_v32.DeepseekV32SparseAttention.forward` (or similar). The
V4-Flash variant follows the same structure with different head counts.
Trace which call sites materialize the indexer score matrix and the sparse-
attention kernel. Likely two paths:
- `prefill` (long-context): called once per layer per request.
- `decode` (streaming): one query at a time. **Out of scope** for this
 initial integration — our chunked path optimizes prefill.

### Step 2: replace the indexer + sparse-attn calls with `flash_csa_forward`

The reference path computes:
```python
# scores: [B, S, T] FP32 — the OOM-prone matrix
scores = lightning_indexer(q_idx, k_idx_compressed, weights)
top_idx = scores.topk(top_k, dim=-1).indices
o = sparse_attn(q, kv, attn_sink, top_idx, sm_scale)
```

Our drop-in replacement:
```python
from flash_sparse.csa import flash_csa_forward

o = flash_csa_forward(
 q=q, kv=kv, kv_compressed=kv_compressed,
 q_idx=q_idx, k_idx_compressed=k_idx_compressed,
 indexer_weights=weights, attn_sink=attn_sink,
 n_win=n_win, top_k=top_k, m=m,
 use_triton=True, use_chunked_indexer=None, # auto-detect
)
```

`use_chunked_indexer=None` triggers the auto-threshold check: if the
materialize matrix would exceed `auto_chunk_threshold_bytes` (default 1 GB),
the chunked path is used. At S=256K, T=64K (m=4), the matrix is 64 GB → auto-
chunked. At S=4K, T=1K, it's 16 MB → materialize.

### Step 3: golden-output parity at small S

Before any benchmark runs, verify perfect output equivalence at small S.
Pick a single prompt at S ∈ {2K, 8K, 32K} and run two paths:
1. Reference V4-Flash modelling code with stock indexer + sparse-attn.
2. Same code with `flash_csa_forward` substituted.

Compare logits at each layer's output. Pass criterion: `max |Δlogits| < 1e-3`
in BF16 (this is the standard threshold for kernel substitution). If the
delta exceeds this, do not proceed — debug the integration.

This step does NOT need long context; it tests only the kernel-substitution
correctness at the regime where both paths fit.

### Step 4: long-context perplexity drift

Pick the eval corpus. Run perplexity at S ∈ {32K, 128K, 256K}:
- 32K: both paths fit; perplexity must be within 0.01 (relative).
- 128K: materialize path uses 16 GB matrix; ours uses chunked. Compare
 reference-stock vs ours.
- 256K: materialize OOMs (the regime we enable). No reference baseline at
 this S exists in the V4-Flash modelling code without our patch — so we
 compare ours-at-256K to ours-at-128K (intra-method scaling sanity).
 This tests: does perplexity stay flat or improve as context extends?
 (For coherent long-context content, perplexity should *decrease*.)

Pass criterion at 32K: `|Δppl| < 0.01` (essentially zero).
Pass criterion at 128K: `|Δppl| < 0.05`.
Pass criterion at 256K: perplexity is finite and monotonically decreasing
(or stable) vs 128K.

### Step 5: tokens/sec end-to-end

Same configs. Wall-clock time per prefill step. Report:
- Time per token at the prefill stage.
- Comparison vs materialize path where it fits.
- Comparison vs an alternative streaming-attention strategy at S=256K
 (this is what the paper actually claims throughput-wise: "we're the only
 open-source CSA implementation that runs at this S; here is the wall-clock
 cost"). No claim of "X× faster than Y".

### Step 6: aggregate into Fig. 4 of the paper

A single bar chart: x-axis S ∈ {32K, 128K, 256K}, y-axis tokens/sec,
two bars per S (reference + ours). Reference column at 256K is OOM-marked.
This is the headline figure of the experiments section that goes alongside
Fig. 1 (kernel-level scaling).

## What we do not need (clarity for the user)

- We do **not** need to retrain anything.
- We do **not** need a CUDA forward to do this integration; the Triton fwd
 is sufficient (just slower at production D — see paper §6.1).
- We do **not** need the backward path for inference benchmarks. (Backward
 integration is a separate task for fine-tuning workflows.)

## Estimated effort

If checkpoint is in hand: 2–3 days end-to-end.
- Day 1: Step 1–3 (parity at small S).
- Day 2: Step 4 (perplexity at long S; ~12-20 hr compute on H200).
- Day 3: Step 5–6 (tokens/sec, figure prep).

## What to do today (without checkpoint)

The bench scripts already simulate the kernel-level workload at V4-Flash
dims with synthetic inputs. They produce all the kernel-level numbers we
need for paper Fig. 1, 2, 3. The end-to-end integration is the *evidence*
that the kernel-level claim translates to model behaviour — necessary for
the final paper, not for the gating "do we have a paper" decision.

If the TileLang verification (`bench_vs_tilelang_pipelined.py`) returns
outcome (a) — TileLang OOMs at S=256K — the paper direction is gated.
End-to-end integration becomes the next priority.

If outcome (b) or (c), the paper direction reframes (memory-efficiency
narrative or no-paper-without-CUDA-kernel) and end-to-end integration may
not be needed for the workshop submission.

## Open questions for the user

1. Which checkpoint and corpus? (Likely DeepSeek-V3.2-exp + RULER, but the
 user may have internal preferences.)
2. Acceptable perplexity drift threshold at 32K and 128K?
3. Is decode-path integration in scope, or prefill-only?
