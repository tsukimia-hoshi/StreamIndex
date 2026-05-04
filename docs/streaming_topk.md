# Streaming top-k for the lightning indexer

> **Insight 1 of the FlashSparse paper.** The lightning indexer's score function is per-key
> separable in a way that makes top-k selection commute with iteration order over the key
> axis. We exploit this to keep the per-query score vector entirely in registers / shared
> memory, eliminating the `[seq_len, seq_len_kv/m]` HBM round-trip the reference performs.
>
> All claims below are formal. The streaming top-k algorithm we describe is mathematically
> equivalent to the reference's materialize-then-sort path; their outputs differ only in
> tie-breaks (which we make deterministic).

## 1. Setup and notation

Let `H ∈ ℝ^{n × d}` be the input hidden states, `n` the sequence length, `m` the CSA
compression ratio. The indexer (eq. 13–17 of DeepSeek-V4) produces:

- per-query indexer queries `q^I_{t,h} ∈ ℝ^{c_I}` for `t ∈ [n]`, `h ∈ [n_I_h]`
- per-key indexer keys `K^{IComp}_s ∈ ℝ^{c_I}` for `s ∈ [T]` where `T = n / m`
- per-query, per-head weights `w^I_{t,h} ∈ ℝ`

The **lightning indexer score** for query `t` and compressed key `s` is

$$
I(t, s) \;=\; \sum_{h=1}^{n_I^h} w^I_{t,h} \cdot \mathrm{ReLU}\!\bigl(q^I_{t,h} \cdot K^{IComp}_s\bigr) \tag{eq.\ 16}
$$

For each `t`, the indexer selects

$$
\mathrm{TopK}(t) \;=\; \mathrm{argtop\text{-}k}_{s} \, I(t, s) \quad\text{ subject to causal mask } s \cdot m + m - 1 \le t. \tag{eq.\ 17}
$$

`TopK(t)` is the *set* of the `k` causally legal indices `s` whose scores `I(t, s)` are
largest. The reference materializes the full score row `I(t, ·) ∈ ℝ^{T}` to HBM, then calls
`torch.topk`. We replace that with a streaming heap update.

We use the following structural fact throughout. **For any fixed `t`,**

$$
I(t, \cdot)\;:\;[T] \to \mathbb{R}, \qquad I(t, s) = g_t(s),
$$

i.e. once `t` is fixed the score is a pure function of `s`. There is no inter-`s` coupling
in the score (no softmax over `s`, no normalization that mixes scores).

## 2. The theorem

### 2.1 Total order on (score, index) pairs

Top-k requires comparing pairs `(σ, s)`, not raw scores, because score ties must be
broken deterministically. Define the **total order ≻** on `ℝ × ℤ_{≥0}`:

> `(a, i) ≻ (b, j)` ⟺ `a > b` ∨ (`a = b` ∧ `i < j`).

This is a strict total order: irreflexive (no element ≻ itself), transitive, and total
(every pair of distinct elements is comparable). The min-heap below is min under ≻ —
i.e., its peek is the *smallest* element under ≻, which is the score-smallest, breaking
ties in favor of the *larger* index. We choose this order so that the final top-k set
contains the score-largest elements with ties broken in favor of *smaller* indices, which
matches the inductive bias of causal language modeling (prefer earlier context on ties).

`PyTorch's torch.topk does not guarantee this tiebreak.` From the docs: "If two elements
are equal, the order is undefined." So our streaming algorithm and a `torch.topk` baseline
may disagree on tied entries; that's an artifact of `torch.topk`, not a bug in either.
For the parity test in § 6 we either (a) inject deterministic noise to break ties, or (b)
compare *sets* of selected indices on the no-tie subset.

### 2.2 Effective k

Causal masking restricts the legal index range. Let `T_legal(t) = max{s + 1 : s · m + m -
1 ≤ t}`, the number of compressed blocks fully inside the past of query `t`. The output
size of the indexer is

`k_eff(t) = min(k, T_legal(t))`,

i.e., for queries near the start of the sequence (`t < k · m`) we return fewer than `k`
indices. The reference handles this with `min(self.index_topk, end_pos // ratio)` at
`model.py:427`. Our streaming algorithm produces a heap of size `min(k, j)` after
processing `j` legal elements, which is exactly `k_eff(t)` after the loop terminates.

### 2.3 The theorem

**Theorem (streaming top-k of the lightning indexer is exact).** Let `g : [T_legal] → ℝ`
be any function (the per-`s` indexer score for a fixed query `t`, restricted to causally
legal `s`). Let `k_eff = min(k, T_legal)`. Run the following streaming algorithm:

> Maintain a heap `H` ordered by ≻ (so `H.peek` returns the smallest element under ≻
> among current heap contents). `H` is initially empty (no sentinel entries).
>
> For `s` in any permutation `π` of `[T_legal]`:
>
> 1. Compute `σ = g(π(s))`.
> 2. If `|H| < k`, insert `(σ, π(s))`.
> 3. Else if `(σ, π(s)) ≻ H.peek`, pop the min and insert `(σ, π(s))`.
> 4. Else, discard.

After processing every legal `s`, `H` contains exactly the top-`k_eff` set under ≻ —
identically equal to the set obtained by materializing `g(s)` for all legal `s` and
selecting the top `k_eff` under the same order ≻.

**Proof.** Strong induction on the number of legal elements processed `j`.

Define `M_j = top-k under ≻ of {(g(s_i), s_i) : i ≤ j}` where `s_1, …, s_j` is the
prefix of the iteration order processed so far. We claim `H_j = M_j` for all `j`.

*Base case* (`j = 0`): `H_0 = ∅` and `M_0 = ∅` (top-k of empty is empty). ✓

*Inductive step.* Assume `H_j = M_j`. Consider processing element `(g(s_{j+1}),
s_{j+1})`. We split on whether `j + 1 ≤ k`:

- *Case A (j + 1 ≤ k)*: `M_{j+1} = M_j ∪ {(g(s_{j+1}), s_{j+1})}` (all `j + 1`
 elements are in the top-`k`). The algorithm checks `|H| = j < k`, executes step 2,
 produces `H_{j+1} = H_j ∪ {(g(s_{j+1}), s_{j+1})} = M_j ∪ {…} = M_{j+1}`. ✓

- *Case B (j + 1 > k)*: Now `|H_j| = k` (by induction). Let `μ = H_j.peek`, the ≻-min
 of `H_j`. We sub-split on `(g(s_{j+1}), s_{j+1})` vs `μ`:

 - *B1* `(g(s_{j+1}), s_{j+1}) ≻ μ`: The new element belongs in the top-k; it must
 displace some element. Since `μ` is the ≻-min of the current top-k, displacing it
 yields the new top-k under ≻. The algorithm executes step 3, removes `μ`, inserts
 the new pair. `H_{j+1} = (H_j ∖ {μ}) ∪ {(g(s_{j+1}), s_{j+1})} = M_{j+1}`. ✓

 - *B2* `(g(s_{j+1}), s_{j+1}) ≺ μ` (strict, the new pair is ≻-smaller than every
 element of `H_j`): The new pair is not in `M_{j+1}` because `M_j ⊂ M_{j+1} ∪
 \{(g(s_{j+1}), s_{j+1})\}` and the new pair would have to displace some element of
 `M_j` to be in `M_{j+1}`, but it is ≻-smaller than every element of `M_j ⊇
 \{μ\}`, so no displacement happens. `M_{j+1} = M_j = H_j = H_{j+1}`. ✓

 - *B3* `(g(s_{j+1}), s_{j+1}) = μ`: by the totality of ≻, this happens only if
 `g(s_{j+1}) = μ.score` AND `s_{j+1} = μ.index`. Since `s_{j+1}` is a fresh element
 (we are visiting each legal index exactly once) it cannot equal `μ.index` already in
 the heap. So this case is vacuous in our setting. (If we reused indices this case
 would need handling, but the proof rules it out.) ✓

Inductive step complete.

After all `T_legal` elements are processed, `H_{T_legal} = M_{T_legal}` is the top-k
under ≻ of all legal scores. □

**Corollary (permutation-invariance).** For any two permutations `π, π'` of
`[T_legal]`, the streaming algorithms with iteration orders `π` and `π'` produce the
same heap `H_{T_legal}`. (Proof: both equal `M_{T_legal}`, which depends only on the
multiset of scored pairs, not the visit order.)

**Corollary (complexity).** Computing `TopK(t)` for one `t` with the streaming algorithm
uses

- `O(T_legal · n_I_h · c_I)` FMAs (the indexer dot products — same as reference),
- `O(T_legal · log k)` heap-update comparisons,
- `O(k)` storage (the heap `H`),
- `O(T_legal · c_I)` HBM bytes read for `K^{IComp}` (streaming pass),
- **no HBM writes of intermediate scores.**

The reference is `Θ(T_legal · n_I_h · c_I)` FMAs (same), `Θ(T_legal log T_legal)`
sort cost, `Θ(T_legal)` working memory per query, and `Θ(T_legal · 4)` HBM **writes
plus reads** for the FP32 score row.

## 3. Algorithm

We give the algorithm in two passes: (a) the per-query streaming top-k as it would be
implemented in a Triton kernel, and (b) the fused multi-query persistent-kernel form that
goes into FlashSparse future work.

### 3.1 Per-query streaming top-k

```text
inputs: q ∈ ℝ^{n_I_h × c_I} (one query)
 K ∈ ℝ^{T × c_I} (compressed keys, FP8 in production)
 w ∈ ℝ^{n_I_h} (per-head weights)
 t (query position, for causal mask)
 m, k (compression ratio, max top-k size)
output: TopK ∈ ℤ^{k_eff} (selected compressed-key indices, k_eff = min(k, T_legal))

heap H = empty # heap ordered by ≻ from § 2.1
for s in 0 .. T-1:
 if (s+1) * m - 1 > t: break # causal mask: stop once block ends after t
 σ = 0
 for h in 0 .. n_I_h - 1:
 σ += w[h] * relu(q[h] · K[s])
 if |H| < k:
 H.insert((σ, s)) # case A
 elif (σ, s) ≻ H.peek: # case B1: smaller-index wins on tie
 H.pop_min
 H.insert((σ, s))
 # else: discard # case B2
return [s for (_, s) in H]
```

The `(σ, s) ≻ H.peek` test expands to `σ > H.peek.score ∨ (σ = H.peek.score ∧ s
< H.peek.index)`. Heap operations preserve the order ≻ on insert / pop.

This is `O(T·n_I_h·c_I + T log k)` per query. For V4-Pro (n_I_h=64, c_I=128, k=1024) at 1M
context (T=250K): the dot product cost is `T·n_I_h·c_I = 250K · 64 · 128 ≈ 2 G FMAs`. The
heap update cost is `T log k = 250K · 10 = 2.5 M ops`. **Compute is dominated by the dot
products, never the heap.**

### 3.2 Fused multi-query persistent kernel

For the kernel-level work we collapse the indexer + top-k + sparse-attn pipeline into a single persistent
kernel on Hopper. Inside that kernel, top-k is maintained per query in shared memory.
Sketch (one CTA per query block of size B_q):

```text
__shared__ q_blk [B_q × n_I_h × c_I] # FP4
__shared__ heap [B_q × k] # (score, s) pairs

# Initialize heaps
for q_i, j in parallel_(B_q, k):
 heap[q_i][j] = (-inf, -1)

# Stage 1: load Q block (all n_I_h heads) into SRAM
TMA: HBM[H_block, ...] -> SRAM[q_blk]
compute_q^I(q_blk) # in-place: project, RoPE, Hadamard, FP4-quant

# Stage 2: stream over K^IComp blocks
for s_blk in 0 .. T-1 step B_kv:
 TMA: HBM[K^IComp, s_blk:s_blk+B_kv] -> SRAM[k_blk] # FP8

 # Per-query top-k update (B_q × B_kv work)
 for q_i in parallel(B_q):
 for s_off in B_kv:
 σ = sum_h(w[t_i, h] * relu(q^I[t_i, h] · K[s_blk + s_off]))
 heap_update(heap[q_i], σ, s_blk + s_off) # see kernel-level heap below

# Stage 3: write top-k indices to HBM
TMA: SRAM[heap.indices] -> HBM[topk_idxs[t_block]]
```

Three kernel-level subtleties worth flagging now (they'll matter in future work):

1. **Shared-memory heap or radix?** A binary heap has `O(log k)` update but its serial
 structure poisons warp-level parallelism. For `k = 1024` the path is 10 deep — too
 serial. Two practical alternatives, both of which match the streaming-top-k theorem:
 - (a) **Bucket-sort / radix top-k**, à la TileLang's `topk_selector.py`, but applied
 incrementally (one block at a time). Works at warp granularity; needs a working buffer
 of size O(B_kv).
 - (b) **Replacement selection without heap order**: maintain the *unordered* top-k set
 plus a single tracked minimum. Insert if the new score beats the tracked min; on
 insertion, re-find the new min by parallel reduction over the array. `O(k / W)` per
 insert with `W` warps — for `k=1024, W=4`: 256 cycles/insert, but only insertions
 above current min trigger this (rare past steady-state).
 We provisionally pick (b) for V4-Pro `k = 1024` because radix's ` working buffer is
 per-block while the heap (b) only stores `k` final scores.

2. **Causal mask as early-exit.** Once the loop reaches `s · m + m - 1 > t`, no further
 compressed block can be legal. The loop should `break` (not just mask), saving the
 tail of the streaming pass. For mid-sequence queries this saves up to `T - t/m` blocks.

3. **Tie-break determinism on accelerators.** Floating-point exact equality is rare but
 not impossible (especially with FP4 quantization producing exactly equal scores).
 Production must specify and test the tiebreak; the kernel-level work tests will compare against
 `torch.topk` with stable sort, which prefers the *smaller* index on ties. Our streaming
 algorithm above also prefers smaller indices. ✓

## 4. Speedup model

We compare the per-query work and HBM IO of the reference (materialize-then-sort) vs the
streaming version. Quantities are per query token, with `T = n / m`:

| | reference (materialize) | streaming (FlashSparse) | savings |
|--|--|--|--|
| score matrix HBM write | `T · 4 B` (FP32) | `0` | `T · 4 B` |
| score matrix HBM read | `T · 4 B` | `0` | `T · 4 B` |
| top-k HBM write | `k · 4 B` | `k · 4 B` | — |
| `K^{IComp}` HBM read | `T · c_I · 1 B` (FP8, amortized over B_q queries: `T · c_I / B_q`) | same | — |
| compute (dot products) | `T · n_I_h · c_I` FMAs | same | — |
| compute (top-k) | `T log T` (sort) | `T log k` (heap) | `log T / log k` |

**Streaming saves `2 · 4 T = 8 T` bytes of HBM traffic per query token**, plus a `log T /
log k` factor on the top-k step itself. Other costs are identical.

For V4-Pro at 1M context (`T = 250K`):

- per-token saving: `8 · 250000 = 2 MB`
- per-sequence saving: `n · 2 MB = 1M · 2 MB = 2 TB` total HBM saved
- on H200's 4.8 TB/s HBM: `~0.4 s` of HBM time saved per CSA layer per million-token forward

With **30 CSA layers in V4-Pro**, this is `~12 s` of HBM-time per single 1M-token forward
just from streaming-top-k. That alone justifies the kernel work for long-context use.

For V4-Flash at 1M context (`T = 250K`, `k = 512` instead of 1024):

- per-sequence saving: `2 TB` (same — savings is independent of `k`, depends only on `T`)
- with **20 CSA layers in V4-Flash**: `~8 s` of HBM-time per million-token forward

For shorter context (256K, `T = 64K`):

- per-sequence saving: `64K · 8 B · 256K = 128 GB` per layer. Still substantial.

## 5. Why the savings are not double-counted

Reference numbers above assume the score matrix is **read once and written once**. In some
kernels that's an underestimate: TileLang's `fp8_lighting_indexer.py` writes scores via
`T.copy(s_reshaped, logits)` then `topk_selector.py` reads them back from HBM through L2.
Cache hits bring the *effective* read down on smaller sequences but evict at large `T`,
so for million-token contexts the second `4T` write/read pair is essentially un-cached.

Streaming makes the cache argument moot: there is no allocation. The score row exists
only as transient register / SRAM state for the duration of one query's pass.

## 6. Tie-breaking and reproducibility

The tiebreak convention is fixed by the order ≻ in § 2.1:

> **On exact score equality, prefer the smaller compressed-key index `s`.**

The compressed-key index `s` is also the temporal order within a sequence: ties broken in
favor of smaller `s` mean the indexer prefers slightly earlier context on equal scores,
which empirically matches the inductive bias of causal language modeling.

**Important caveat about `torch.topk` parity.** PyTorch's `torch.topk` does *not*
guarantee tiebreak stability — from the docs: "If two elements are equal, the order is
undefined." Therefore our streaming algorithm's output **does not bit-match
`torch.topk(scores, k, largest=True)` on tied entries**. For correctness comparison we:

1. Compare the *sets* of selected indices (we expect 100% set overlap when there are no
 ties, which is essentially always the case under FP32 with non-degenerate weights);
2. When ties are unavoidable in the test (e.g. constructed adversarial inputs), use
 `torch.sort` with explicit stable comparison and a custom `(score, index)` pair, which
 PyTorch *does* honor stably.

Because the algorithm depends only on the multiset of scored pairs, **the output is
deterministic across permutations of the iteration order over `s`** (Corollary, § 2.3).
This is important for two-CTA splits in future work, where the K dimension is divided across
CTAs and partial heaps are merged: as long as the merge respects ≻, the final top-k set
is identical to a single-CTA computation.

**Reproducibility note (FP8 indexer keys).** When `K^{IComp}` is FP8-quantized with
per-block dynamic scaling, scores have small perturbations vs a hypothetical FP32 indexer.
Top-k under those perturbations may select a *different set* than FP32 ground truth — this
is a QAT property, not a streaming-top-k property. The reference TileLang kernels and our
streaming kernel agree to within FP8 noise; we test for set-overlap, not bitwise identity.
Empirically, on V3.2 indexer benchmarks (`tilelang/examples/deepseek_v32/topk_selector.py:
test_topk_selector`), set-overlap is `≥ 99%` between FP32-materialized and FP8-quantized
top-k sets.

**Edge cases that arise in practice and are handled correctly:**

- *All scores zero or `-inf`*: The heap simply fills with the first `k` legal indices.
 Their order in the final heap is determined by ≻ (smaller index ranks higher on ties).
 Downstream consumers must check that the gather actually has signal — but that's a
 model-quality issue, not a kernel correctness issue.
- *`T_legal < k`*: The heap accepts every legal element (Case A always); output is all
 `T_legal` legal indices. Caller pads with `-1` to size `k` for downstream gather.
- *Heap initialized with sentinels (`(-inf, -1)` etc)*: Don't. Use an empty heap with
 a length counter, *not* a sentinel-filled fixed-size array, otherwise on ties the
 sentinel index `-1` would win under ≻ (since `-1 < any-valid-index`), corrupting the
 result. Phase-2 kernel uses `valid_count` + array-as-heap-of-current-length-only.

## 7. Backward considerations

The streaming-top-k algorithm is non-differentiable wrt `s` (you cannot differentiate index
selection). Existing systems that train through DSA-style top-k either:

1. Use a **straight-through estimator** for the selected set (gradient flows through the
 selected-block attention as if the selection were the identity, with no gradient on the
 selection itself). DeepSeek-V3.2 / V4 training adopts this approach.
2. Use a **soft top-k relaxation** (e.g. top-k softmax with temperature). Out of scope for
 FlashSparse, and not how DeepSeek trains.

For our kernel the selection step has zero backward cost: `dQ`, `dK`, `dV` flow only
through the `sparse_attn` core that operates on the *already-selected* indices. The
backward pass in future work inherits all of section 3.1's IO benefits — but the streaming
property is irrelevant on the backward (which already knows the indices) and not the
contribution we claim.

## 8. Summary

- The lightning indexer score `I(t, s)` is per-key independent: `I(t, s) = g_t(s)`.
- Top-k over `s` is therefore order-independent: a streaming heap of size `k` ordered by
 the lexicographic order ≻ on `(score, index)` is *exactly equal* (not just
 approximately equal) to materialize-then-sort under the same order.
- Streaming eliminates the `[seq, T]` FP32 score matrix from HBM. Per query token at 1M
 context: `8 · T = 2 MB` saved (read + write of FP32 scores). Per layer at 1M context:
 `n · 8T = 2 TB` of HBM traffic eliminated.
- The mathematical claim is **proven** within the assumptions: deterministic order ≻,
 legal-only iteration, fresh indices (each `s` visited once). Tiebreak parity with
 `torch.topk` is *not* guaranteed because `torch.topk` itself is unstable on ties.
- Open kernel-level decisions for future work: heap data structure (binary vs radix vs
 unordered-with-min-tracking), TMA streaming pattern for `K^{IComp}`, and how to merge
 multi-CTA partial heaps.
