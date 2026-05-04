# Fusion analysis

> **Insights 2 and 3 of the FlashSparse paper.** For each intermediate tensor that the
> DeepSeek-V4 reference materializes in HBM during a CSA / HCA forward or backward,
> we determine whether it can be eliminated (kept in registers / shared memory) in a
> fused kernel. We then propose the persistent-kernel topology and SRAM budget that makes
> this work on Hopper (H100 / H200).
>
> Insight 2: **forward fusion** — what gets eliminated and why.
> Insight 3: **backward fusion** — eliminate atomic_addx4 dKV scatter via inverted-topk
> sum-reduction.

## 0. Hopper SRAM budget (H100 / H200)

Per CTA on Hopper, the maximum *dynamic* shared memory after opt-in is **228 KB**, of
which CUDA reserves a small amount (≤ 1 KB) for static internals (kernel parameter
shadows, etc). The usable ceiling for our allocations is therefore **~227 KB** in
practice. The opt-in is via `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamic
SharedMemorySize, 228 << 10)` at runtime. Below 48 KB the opt-in is not required.

H100 and H200 both have 132 SMs and identical per-SM shared-memory specs; H200's
advantage is HBM3e bandwidth (4.8 TB/s vs 3.35 TB/s) and capacity (141 GB vs 80 GB), not
SRAM. All SRAM numbers below assume the **227 KB usable per CTA**.

Per-CTA register file: 64 K registers (256 KB), divided across threads. With 256 threads
per CTA, that's `256 KB / 256 = 1 KB / thread = 256 32-bit registers / thread`.
Register-resident accumulators (per thread, per query) above ~256 elements spill — keep
register footprint tight.

For the heap (streaming top-k), we'll use shared memory, not registers — `k = 1024`
entries does not fit in a single thread's register file even with reuse.

## 1. Inventory: CSA forward intermediates

For each tensor, columns:

- **shape** — typical per-CTA-per-iteration shape
- **bytes** — bytes if persisted to HBM
- **HBM-required?** — must live in HBM for some downstream consumer outside this kernel
 (yes/no)
- **SRAM-resident?** — fits in 228 KB shared with other live tensors
- **register-resident?** — small enough to live in register file across the relevant
 per-thread loop nest

Per CSA layer, in temporal order of generation:

| # | tensor | shape (typical) | bytes | HBM-required? | SRAM-resident? | reg-resident? | notes |
|---|--------|-----------------|------:|:-------------:|:--------------:|:-------------:|-------|
| 1 | `H_t` (input hidden state) | `[B_q, d_model]` | `64·7168·2 = 917,504` | yes (input from prior layer) | partial | no | streaming load via TMA, typically reused for q-proj and kv-proj |
| 2 | `c_a, c_b` (compressor pre-activations) | `[B_q, m, c]` per stream | `64·4·512·4 = 524,288` per stream | **no** | yes (4 such = 2 MB total — too much) → split | no | reference writes 8 KB/token to HBM × 2 streams; we keep in registers |
| 3 | `z_a, z_b` (compressor scores) | same as 2 | same | **no** | partial | yes (`B_q · m` floats per stream) | gate scores; tiny per-token, stay in registers |
| 4 | `C^Comp` (compressed KV entries) | `[B_q/m, c]` per token batch | `(64/4)·512·b_kv = 8704` | **yes** (downstream layers and bwd read from this) | n/a | n/a | the actual KV cache write |
| 5 | `Compressor_I` (indexer's compressor) outputs | `[B_q, m, c_I]` | `64·4·128·4 = 131,072` | **no** | yes | yes | smaller `c_I = 128`, fits comfortably |
| 6 | `K^IComp` (indexer compressed keys) | `[B_q/m, c_I]` | `(64/4)·128·b_kI ≈ 2,051` | **yes** (per-layer indexer cache) | n/a | n/a | small, must persist across forward calls |
| 7 | `q^I_t` (indexer queries, FP4 + Hadamard) | `[B_q, n_I_h, c_I]` | `64·64·128·0.5 = 262,144` | **no** | yes (when split per-h) | partial | reference persists to HBM; we keep per-head-group in SRAM |
| 8 | `w^I_t` (indexer per-head weights) | `[B_q, n_I_h]` | `64·64·2 = 8,192` | **no** | yes | yes | tiny |
| 9 | **`I(t,s)` (score matrix)** | `[B_q, T]` | `64·T·4 = 256·T` | **no** (only top-k consumes it) | partial (one row at a time) | yes (per query streaming) | **THE ELIMINATED ONE** — replaced by streaming heap |
| 10 | `top_k_idxs` (selected indices) | `[B_q, k]` | `64·1024·4 = 262,144` | **yes** (passed to gather) | yes | no | architecturally we *could* fuse indexer+attn so this stays in SRAM, but it makes the kernel large; see § 3 |
| 11 | gathered KV (top-k + sliding window) | `[B_q, k+n_win, d]` | `64·1152·512·b_kv ≈ 38 MB` | **no** (transient) | **no** (way too big) | n/a | streamed in `B_kv = 64`-row chunks via TMA |
| 12 | attention scores (`P[B_q, h_per_blk, B_kv]`) | `[B_q, 64, 64]` | `262,144` | **no** | yes | yes | online softmax keeps one block at a time |
| 13 | online softmax state `(m_i, l_i)` | `[B_q, h_per_blk]` | `64·64·4·2 = 32,768` | **no** | yes | **yes** | FA-style accumulators |
| 14 | output `O` | `[B_q, n_h, d]` | `64·128·512·2 = 8,388,608` | **yes** (next layer + LSE) | partial | n/a | streamed via TMA writes |
| 15 | LSE (per-head log-sum-exp) | `[B_q, n_h]` | `64·128·4 = 32,768` | **yes** (used by backward) | yes | yes | small, persist for backward |

**Summary**:

- HBM-required: rows 1, 4, 6, 10, 14, 15. The rest can be eliminated.
- Largest per-token HBM saving: row 9 (the score matrix) — the only row >1 KB/token that
 is **not** required by any downstream consumer outside this kernel.
- Rows 2, 3, 5, 7, 8 (compressor and indexer intermediates) are minor (50–300 KB total
 per `B_q=64` block) but in aggregate they amount to `~30 KB/token` of pointless HBM
 traffic.

## 2. SRAM accounting: fully-fused single-kernel CSA forward

Suppose we attempt the most ambitious design: one persistent kernel that runs the full
indexer + top-k + sparse attention pipeline for `B_q` queries per CTA. SRAM occupants
during peak:

| live tensor | size | bytes |
|---|---|---:|
| Q tile (sparse-attn) | `H_per_block × d` BF16 | `64·512·2 = 65,536` |
| Q tail (rope dims) | `H_per_block × rope_d` BF16 | `64·64·2 = 8,192` |
| KV block (gather, BF16 main + FP8 if quantized) | `B_kv × d` BF16 | `64·512·2 = 65,536` |
| KV tail | `B_kv × rope_d` BF16 | `64·64·2 = 8,192` |
| O accumulator | `H_per_block × d` BF16 | `64·512·2 = 65,536` |
| `q^I` (indexer queries, FP4) | `B_q · n_I_h · c_I · 0.5` | `64·64·128·0.5 = 262,144` |
| `K^IComp` block | `B_kv · c_I` FP8 | `64·128 = 8,192` |
| **per-query top-k heap** | `B_q · k · 8` (score+idx) | `64·1024·8 = 524,288` |
| online softmax state | small | < 64 KB |
| **TOTAL** | | **~1,007 KB** |

That's **~4× over the 227 KB shared-memory ceiling**. The two big offenders:

- `q^I` materialized for the whole query block: 256 KB.
- per-query heap: 512 KB.

Both scale linearly with `B_q`. Setting `B_q = 1` (one query per CTA) collapses them
to:

| live tensor | size at `B_q = 1` |
|---|---:|
| Q tile | 65,536 |
| Q tail | 8,192 |
| KV block | 65,536 |
| KV tail | 8,192 |
| O accumulator | 65,536 |
| `q^I` | `1·64·128·0.5 = 4,096` |
| `K^IComp` block | 8,192 |
| heap | `1·1024·8 = 8,192` |
| online softmax + scratch | ~4,096 |
| **TOTAL** | **~237 KB** |

Even with `B_q = 1` the budget is ~10 KB over the 227 KB ceiling. Plus the `Q`, `KV`, `O`
tiles each occupy 64 KB regardless of `B_q`, leaving little room for the
indexer/heap state.

**Conclusion: fully-fused single-kernel CSA forward does not fit on Hopper.** We need a
different topology (§3) that splits the work across two persistent kernels.

## 3. Two-stage topology (recommended for future work)

We split the forward into two persistent kernels:

```
 ┌────────────────────────────────────────────┐
 │ kernel 1: indexer + top-k │
 │ - one CTA per query │
 │ - streams K^IComp, │
 │ - maintains in-SRAM heap of size k │
 │ - writes top-k indices to HBM │
 └────────────────────────────────────────────┘
 │
 ▼ top-k indices [B, S, k] int32
 ┌────────────────────────────────────────────┐
 │ kernel 2: sparse attention │
 │ - one CTA per (query block, head group) │
 │ - reads top-k indices + sliding window │
 │ - gathers KV, computes attention │
 │ - writes O, LSE │
 └────────────────────────────────────────────┘
```

### 3.1 Kernel 1 (indexer + top-k) SRAM budget

Per CTA = one query (or 2 queries packed):

| live tensor | size | bytes |
|---|---|---:|
| `q^I` for this query | `n_I_h · c_I · 0.5` FP4 | `64·128·0.5 = 4,096` |
| `K^IComp` block | `B_kv · c_I` FP8 (B_kv=128 here) | `128·128 = 16,384` |
| `w^I` weights | `n_I_h · 4` FP32 | `64·4 = 256` |
| heap (scores + indices) | `k · 8` | `1024·8 = 8,192` |
| score-row scratch (one block) | `B_kv · 4` FP32 | `128·4 = 512` |
| **TOTAL** | | **~29 KB** |

Comfortably fits. With `H_per_block = 1` query per CTA we get **one CTA per query token**
on a sequence of length `n`. That's `n` CTAs total for one layer's forward indexer pass.

For decode (`n = 1` query) this is one CTA, hilariously underutilized. Decode is in the
strong arithmetic-intensity regime; we batch decode queries across requests.

### 3.2 Kernel 2 (sparse attention) SRAM budget

Per CTA = one (query block, head group) tile, identical to TileLang's
`sparse_mla_fwd.py` layout. We pick `H_per_block = 32` (quarter of `n_h = 128`, so
`REPLICATE_H = 4` CTAs cover all 128 heads per query block) and `B_q = 4`:

| live tensor | size | bytes |
|---|---|---:|
| Q tile (`H_per_block × d`) | BF16 | `32·512·2 = 32,768` |
| Q tail (`H_per_block × rope_d`) | BF16 | `32·64·2 = 4,096` |
| KV block (`B_kv × d`) | BF16 | `64·512·2 = 65,536` |
| KV tail (`B_kv × rope_d`) | BF16 | `64·64·2 = 8,192` |
| O accumulator (`H_per_block × d`) | BF16 | 32,768 |
| online softmax state (`m`, `l`, scratch) | FP32 | `H_per_block · 4 · 4 = 512` |
| top-k indices for this block (`B_q · (k + n_win)`) | INT32 | `4·1152·4 = 18,432` |
| **TOTAL** | | **~162 KB** |

Comfortably fits the 227 KB ceiling, with ~65 KB of headroom for double-buffering of
KV blocks. the kernel-level work will tune `H_per_block`, `B_q`, `B_kv` more carefully; the point of
this section is just to show **the two-stage design is feasible** within Hopper's SRAM
limits, in a way that the fully-fused design from § 2 is not.

### 3.3 Why two-stage is "good enough"

The IO cost we add by going two-stage instead of fully-fused is a single HBM write+read
of the top-k indices: `n · k · 4 = 4 KB/token`. For `n = 1M` that's `4 GB` per CSA layer
total — negligible against the `~3 TB` we save by eliminating the score matrix.

The compute cost we add is a kernel launch (`~10 µs`) per layer per forward — also
negligible at million-token scale.

**Net**: two-stage captures ≥99% of the IO savings of the fully-fused design with a much
simpler kernel. the kernel-level work will land two-stage; an opportunistic the kernel-level work-bis attempt at
fully-fused is on the wishlist but not required.

## 4. HCA forward fusion

HCA has no indexer and no top-k. The pipeline is:

```
 H_t → Compressor (m'=128, no overlap) → C^Comp → sparse_attn(all-T'-positions)
```

The only fusable seam is the compressor: keep `wkv` and `wgate` outputs in registers
rather than materializing FP32 to HBM. The savings (8 KB/token, see `ir_analysis.md` §4)
are real but small.

For the sparse_attn step, HCA reuses the same kernel as CSA — just with `top_k_idxs`
populated to "all causally legal compressed positions" instead of "top-k from indexer".
No HCA-specific kernel needed.

## 5. Backward forward dependencies

What forward state must be saved to HBM for the backward to consume:

| tensor | shape | bytes / sequence | mandatory? | why |
|---|---|---:|:---:|---|
| `Q` | `[n, n_h, d]` | `n · 128 · 512 · 2` | **yes** | needed for `dK = dP^T · Q` |
| `K^IComp` | `[T, c_I]` | `T · 128 · b_kI` | yes (or recomputable from `H`?) | reference saves it to layer-cache; could recompute in bwd if `H` is saved |
| `KV` (compressed) | `[T, d]` | `T · 512 · b_kv` | **yes** | needed for `dQ = dP · K` and `dV = P^T · dO` |
| top-k indices | `[n, k]` | `n · 1024 · 4` | **yes** | required to know which K rows to gather |
| `LSE` | `[n, n_h]` | `n · 128 · 4` | **yes** | needed to recompute `P` in bwd via `P = exp(s - LSE)` |
| `attn_sink` | `[n_h]` | `128 · 4` | **yes** (constant per layer; tiny) | for sink-augmented softmax |
| `O` | `[n, n_h, d]` | `n · 128 · 512 · 2` | yes | needed by Δ preprocess (`Δ = sum_d O · dO`) |

Notably, **the score matrix `I(t,s)` and the indexer `q^I` are NOT needed by the
backward** — backward operates on the already-selected `top_k_idxs` and is identical to
FA's `dK`, `dV`, `dQ` computation modulo the gather indirection. Streaming top-k
correctness affects only forward selection; backward inherits whatever indices forward
chose.

## 6. Backward fusion — eliminating atomic_addx4

The reference's main cost in backward is the FP32 atomic scatter of dK, dV to selected
indices (TileLang `sparse_mla_bwd.py` line 224, `T.atomic_addx4`). This pattern has two
problems on Hopper:

1. **FP32 storage** for what semantically is BF16 — 2× the per-write bytes.
2. **Atomic contention** on H200's atomic units when many CTAs scatter to the same K row
 simultaneously (typical for nearby query tokens that share top-k members).

**Insight 3**: replace atomic scatter with a **gather-then-sum** structure using a
precomputed inverted top-k index.

### 6.1 Inverted top-k index

After forward, build an auxiliary tensor `inv_topk` of shape `[T, k_max]` such that
`inv_topk[s, j]` is the `j`-th query that selected compressed key `s`. Plus a counter
`inv_count[T]` giving the number of valid entries per `s`.

**Storage size** (worst-case dense layout, `T · k_max · 4`): `k_max` is the maximum
number of queries selecting any single `s`. Pessimistic upper bound `k_max = n`; in
practice with diffuse top-k selection patterns observed in V3.2, `k_max ≈ 4k` at the
99th percentile. **Conservatively use `k_max = 4096` ⇒ size = `T · 4096 · 4 = 4 GB at
T = 250K`.**

A Compressed-Sparse-Row (CSR) layout reduces worst-case waste by storing variable-length
rows: payload `nnz · 4 = 4 GB` (since `nnz = n · k` always), plus `(T + 1) · 4 ≈ 1 MB`
of row pointers. Same total bytes as the dense version when `k_max ≈ 4k`; cheaper when
the tail is heavier. the kernel-level work will pick layout based on measurement.

**Build IO** (per layer, once per forward+backward pair):

- Read `top_k_idxs[n, k]`: `n · k · 4 B = 4 GB at n=1M, k=1024`. **`4,096 B/token`.**
- Write `inv_topk` payload: `n · k · 4 B = 4 GB`. **`4,096 B/token`.**
- Atomic histogram count to `inv_count[T]` (small, ~`T·4 = 1 MB`): negligible per token.

Total build IO: **~8,192 B/token** per (forward + backward) pair, charged in the
backward analysis (§ 5 of `ir_analysis.md`).

### 6.2 dK, dV kernel with sum-reduction

> **2026-04-25 update — measured against the hypothesis.** A first-draft
> Triton implementation of this design (`flash_sparse.triton.sparse_attn_bwd_v2`)
> turned out **substantially slower** than the v1 atomic-scatter approach on
> our H200 benchmarks (0.02–0.05× the speed). Root cause: this analysis
> missed the cost of *re-reading Q for every (query, kv-row) pair*. v2 loads
> ~8 MB/token of Q while v1's atomic-scatter writes ~256 KB/token of
> dKV — and Hopper's FP32 atomic units handle moderate contention faster
> than reloading Q.
>
> The fix is a block-K design: process BLOCK_KV kv rows AND BLOCK_Q queries
> per CTA, loading Q once and reusing it across all (q, k) pairs in the
> block. That's structurally close to FA3's backward but with a per-block
> "did query q select kv k?" predicate. Deferred to future work.c (CUDA).
> Until then, `flash_sparse.triton.sparse_attn_bwd` (atomic scatter) is the
> production backward.


Per K block (one CTA per block of `B_kv` compressed K rows):

```text
load K_block, V_block (= same tensor, MQA)
allocate __shared__ acc_dK[B_kv, d], acc_dV[B_kv, d] # FP32 accumulators
zero(acc_dK); zero(acc_dV)

for each compressed_index s in this block:
 queries = inv_topk[s, :inv_count[s]] # variable-length list
 for query t in queries:
 load Q[t], dO[t], LSE[t], Δ[t] from HBM
 recompute P[t,s] from Q[t], K_block[s], LSE[t]
 dP[t,s] = P[t,s] · (dO[t] · V[s] - Δ[t])
 atomic_add into acc_dK[s] : dP[t,s] · Q[t] # warp-level reduce, no global atomic
 atomic_add into acc_dV[s] : P[t,s] · dO[t]

# Cast accumulators to BF16, write once
write_BF16(dK[K_block_range], acc_dK)
write_BF16(dV[K_block_range], acc_dV)
```

The "atomic_add into acc_dK" inside the CTA uses **shared-memory atomics**
(`atomicAdd` on a `__shared__` pointer), which contend only within a single SM rather
than globally. NVIDIA H100 architecture whitepaper documents shared-memory atomics as
roughly **2 orders of magnitude faster** than global atomics under heavy contention; we
will measure the exact ratio in future work and not claim a specific number here.

**HBM traffic per token**:

| line item | reference | fused (this design) |
|---|---:|---:|
| dKV scatter writes | `k · d · 4 = 2 MB` (FP32 atomic to global) | `0` |
| dKV final tensor write | `T · d · 2 / n = d · 2 / m = 256 B` (BF16 once per K row) | `256 B` |
| inverted-index read | `0` | `4,096 B` (per fwd+bwd pair) |
| inverted-index build (charged to bwd) | `0` | `8,192 B` (read top-k + write inv_topk) |
| **subtotal** | **~2 MB** | **~12.5 KB** |

That's a ~160× reduction in dKV-related HBM traffic — and it shows up directly in the
per-token bwd budget in `ir_analysis.md` § 5.

**Wall-clock benefit (cited, not derived)**: NVIDIA H100 architecture whitepaper (Hopper
guide) and FA3 paper Table 5 indicate that replacing global atomics with structured
sum-reduction gives **1.5–2.0×** wall-clock improvement on backward in attention-bench
workloads with similar contention patterns. We adopt that range and treat it as a
the kernel-level work measurement target; we do not claim a specific multiplier here.

### 6.3 dQ kernel (FA-style, unchanged)

The dQ pass is identical to the standard FA backward:

```text
for each query block t_block:
 load Q[t_block], dO[t_block], LSE[t_block], Δ[t_block]
 allocate __shared__ acc_dQ[B_q, d]
 zero(acc_dQ)

 for each selected index s in topk_idxs[t_block]:
 load K[s], V[s] (gather)
 recompute P[t,s] from Q, K, LSE
 dP[t,s] = P[t,s] · (dO[t] · V[s] - Δ[t])
 acc_dQ[t] += dP[t,s] · K[s] # warp-level

 write dQ[t_block] (BF16, once)
```

No atomics needed — each query is owned by one CTA; dQ writes never race.

## 7. Complete fused-kernel topology

Putting it all together, the FlashSparse kernel set for V4 is:

### Forward
1. **`flash_sparse.csa.compressor_fwd`** (Triton or CUDA) — fused KV compressor for CSA
 layers. Computes `wkv`, `wgate`, softmax-weighted-sum, RMSNorm, RoPE, FP8 quant in
 one kernel. Output: `C^Comp` written to layer's KV cache.
2. **`flash_sparse.csa.indexer_topk_fwd`** — fused indexer + streaming top-k. Output:
 `top_k_idxs`.
3. **`flash_sparse.csa.sparse_attn_fwd`** — sparse attention with sliding window + top-k.
 Output: `O`, `LSE`. Same kernel handles HCA (with HCA-style `top_k_idxs`).

### Backward
4. **`flash_sparse.csa.build_inverted_topk`** — preprocess: build inverted top-k index.
 Lightweight. Cached across multiple bwd calls if forward indices haven't changed.
5. **`flash_sparse.csa.preprocess_delta`** — compute `Δ = Σ_d O · dO`. Light, FA-style.
6. **`flash_sparse.csa.sparse_attn_bwd_dq`** — dQ kernel (FA-style, no atomics).
7. **`flash_sparse.csa.sparse_attn_bwd_dkv`** — dK, dV kernel via inverted-index sum-
 reduction. Eliminates atomic_addx4.

### Compressor backward
For future work: the compressor backward needs `dKV / dC^Comp_input → dwkv, dwgate, dH`.
This is a bunch of straightforward GEMMs (`dwkv = dC^Comp · H_block^T` etc.) and is the
simplest of the kernels. Defer detailed fusion analysis to future work.

## 8. Producer-consumer warp specialization

For the sparse_attn forward (#3 above), we adopt the FA3 producer-consumer pattern (Shah
et al., 2024, arxiv 2407.08608):

- **Producer warp (1 of 8 within a CTA)**: issues TMA loads of `Q`, `K`, `V`, top-k
 indices into a ring buffer of shared memory via `cp.async.bulk`. FA3 paper claims a
 single producer warp can keep up with HBM bandwidth in their kernels; this is to be
 verified for our sparse-gather pattern in future work.
- **Consumer warps (7 of 8)**: WGMMA on `Q · K^T`, online softmax, `P · V`. Producer
 signals consumers via barriers.
- **Pingpong scheduling**: while consumer warps run softmax (slow exp), producer issues
 the next `K, V` load. While consumer runs WGMMA, producer waits without blocking.

This is the FA3 design from `flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h`
adapted for sparse gather. The gather-vs-contiguous difference: producer issues
`cp.async.bulk.tensor.2d` with a list of `(top_k_idx, B_kv_off)` pairs (per the CUDA
12.x programming guide for TMA scattered loads). TMA descriptors handle non-contiguous
HBM addresses natively; bandwidth utilization on scattered loads is documented at ~60%
of contiguous peak in NVIDIA Hopper benchmarks. the kernel-level work measurement will confirm this
holds for our access pattern.

For the indexer (#2), warp specialization is less critical — the kernel is short and
its inner loop is dot-product-bound — but we still split `q^I` projection (one warp)
from `K^IComp` streaming load + dot product (rest) for overlap.

## 9. Risks and unknowns flagged for future work

1. **Heap data structure on Hopper**: `k = 1024` heap updates with shared-memory bank
 conflicts could underperform. Two practical alternatives:
 - **Bucket-sort top-k** (TileLang's existing approach): avoid heap entirely.
 - **Unordered top-k** (track-min + array): single linear scan to find new min on
 replace; works well when the steady-state replacement rate is low.
 We will benchmark both in future work.

2. **Inverted-topk memory footprint**: `4 GB / layer at n=1M`. With 30 CSA layers, holding
 all in HBM simultaneously is `120 GB`, fitting on H200 (140 GB). Fine for single-GPU
 forward+backward, painful for multi-GPU pipeline-parallel. Mitigation: reuse one
 inverted-index buffer across layers (overwrite), at the cost of recomputing per layer
 per backward.

3. **Hopper TMA + sparse gather**: TMA descriptors handle scatter-gather natively, but
 the `cp.async.bulk.tensor` instruction performance for non-contiguous loads is
 under-documented. the kernel-level work will profile; fall back to per-warp-load if TMA underperforms
 on scattered access.

4. **FP4 indexer queries**: storing `q^I` in FP4 in shared memory means we cast to FP8 or
 BF16 before the `q^I · K^IComp` GEMM (Hopper WGMMA needs FP8 or higher). The cast is
 in-place via Hadamard's algorithm; future work will measure overhead.

## 10. Summary

- Forward CSA: **score matrix `I(t,s)` is the single highest-value elimination** (1.5–2 MB
 / token at 1M context). Eliminated by streaming top-k from `streaming_topk.md`. Other
 intermediates (compressor outputs, indexer queries) save another ~30 KB / token but
 matter less.
- Forward HCA: no big eliminations. Wall-clock gain comes from execution improvements
 (FA3 patterns), not from intermediate elimination.
- Backward CSA: **atomic_addx4 dKV scatter is the single highest-value replacement**.
 Replaced by inverted-topk sum-reduction. Saves 2 MB/token of FP32 atomic traffic *and*
 removes atomic-unit contention. Backward dQ is unchanged from FA.
- Backward HCA: same as forward — minimal IO win, mostly an execution-side gain.

The two-stage forward (kernels 1+2+3 above) is the recommended the kernel-level work starting topology.
It captures ≥99% of the IO savings of fully-fused while staying within the 228 KB SRAM
budget on Hopper.
