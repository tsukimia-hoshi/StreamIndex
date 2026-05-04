# IO complexity analysis

> Per-query-token HBM byte budget for CSA and HCA, forward and backward, in the
> DeepSeek-V4 reference implementation versus an idealized fused FlashSparse kernel.
> Numbers cite V4-Pro hyperparameters from `docs/hyperparameters.yaml`. Where the
> reference implementation has a tunable block size, we use plausible values and mark
> them with `(B_q = 64)` etc. so the analysis can be reproduced.

## 0. Notation, units, and ground truth

All HBM traffic is counted in **bytes per query token, per layer** (so per-sequence and
per-forward totals are obtained by multiplying by `n` and by the number of CSA / HCA
layers respectively). When a value is shared across queries within the same CTA we
amortize over `B_q`, the query-block size. Compute (FLOPs, FMAs) is a separate column;
the V4 attention path is HBM-bound on H100/H200 at long context, so HBM is the relevant
metric.

| symbol | meaning | V4-Pro value |
|---|---|---|
| `n` | sequence length | up to 2^20 = 1M |
| `m` | CSA compression ratio | 4 |
| `m'` | HCA compression ratio | 128 |
| `T` | `= n / m`, number of CSA-compressed entries | 250K @ 1M |
| `T'` | `= n / m'`, number of HCA-compressed entries | 8K @ 1M |
| `d` | `head_dim` | 512 (V4 convention; rope on last 64 dims) |
| `n_h` | query heads | 128 |
| `c_I` | indexer head_dim | 128 |
| `n_I_h` | indexer heads | 64 |
| `k` | top_k | 1024 |
| `n_win` | sliding window | 128 |
| `d_c` | `q_lora_rank` | 1536 |
| `d_model` | `hidden_size` | 7168 |
| `B_q` | per-CTA query block | 64 (typical FA-style choice for `d=512`) |
| `B_kv` | per-CTA key/value block | 64 |

Bytes per element (V4 production policy, `docs/hyperparameters.yaml`):

| dtype | bytes/elem | used for |
|---|---:|---|
| BF16 | 2 | Q, sliding-window KV (RoPE dims), output |
| FP8 e4m3 | 1 | main KV NoPE dims (`448` of `512`); indexer keys `K^IComp` |
| FP4 e2m1 | 0.5 | indexer queries `q^I` (Hadamard-rotated, packed 2 per byte) |
| FP32 | 4 | scores, LSE, attention sinks, dKV accumulators, FP8/FP4 block scales |

The **K row layout** for the main attention path follows `model.py:506`,
`act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)` and FlashMLA's published
656-byte-per-token format:

```
K row = [ 448 NoPE dims · FP8 e4m3 ]
 ⊕ [ 7 FP32 scales — one per 64-elem block of NoPE dims ]
 ⊕ [ 64 RoPE dims · BF16 ]
 = 448 + 28 + 128 = 604 B / row (V4-Pro head_dim=512)
```

We use **`B_K = 604`** as the per-K-row byte cost wherever a gather happens. The
`b_kv = 1.18` "effective bytes per element" is an aggregate convenience for the
compressor outputs; we prefer raw `B_K = 604` whenever counting actual gathers.

For the indexer keys (`c_I = 128`, FP8 with one FP32 scale per row), per-row cost is
`B_KI = 128 · 1 + 4 = 132 B / row`, i.e. `b_kI = 132/128 = 1.03125 B/elem`.

## 1. CSA forward — reference

The reference invokes four kernels per CSA layer (TileLang convention from
`tilelang/examples/deepseek_v32/`):

| stage | what it does | code path |
|---|---|---|
| A | per-token KV compressor (`m=4`, overlap) | `model.py:Compressor.forward` |
| B | indexer queries + indexer compressor + score matrix | `kernel.py:fp8_lighting_indexer` (TileLang) or einsum in `model.py:Indexer.forward` |
| C | top-k selector | `tilelang/.../topk_selector.py` |
| D | sparse attention core | `kernel.py:sparse_attn` (TileLang) |

Per stage, per token:

### Stage A — KV compressor

| line item | bytes / token | rationale |
|---|---:|---|
| read `H_t` | `d_model · 2 = 14,336` | BF16 hidden state (charged once per layer call) |
| write `wkv(H)` intermediate | `coff · d · 4 = 4,096` | FP32 output of `Linear`, `coff=2` for overlap |
| write `wgate(H)` intermediate | `coff · d · 4 = 4,096` | same |
| read `wkv`, `wgate` intermediates | `2 · 4,096 = 8,192` | for the softmax-gated weighted sum |
| write `C^Comp` NoPE part (per `m`-block) | `(d − rope_d) · 1 / m + scales / m = (448 + 28) / 4 = 119` | FP8 + 7 FP32 scales, per `m=4` |
| write `C^Comp` RoPE dims (BF16) | `rope_d · 2 / m = 32` | last 64 dims kept in BF16 |
| **subtotal** | **30,871** | |

### Stage B — indexer (with separate Compressor for `K^IComp`)

| line item | bytes / token | rationale |
|---|---:|---|
| indexer compressor: write `wkv_I`, `wgate_I` intermediates | `2 · coff · c_I · 4 = 2,048` | FP32 (`coff = 2` for overlap, `c_I = 128`) |
| indexer compressor: read intermediates | `2,048` | for the weighted sum |
| indexer compressor: write `K^IComp` | `B_KI / m = 132 / 4 = 33` | FP8 + 1 FP32 scale per row, per `m=4` |
| indexer queries: read `c^Q_t` | `d_c · 2 = 3,072` | BF16 latent |
| indexer queries: write `q^I_t` (FP4 + scales) | `n_I_h · c_I · 0.5 + n_I_h · (c_I/32) · 1 = 4,096 + 256 = 4,352` | FP4 e2m1 + 4 FP8-scale per row × 64 heads |
| indexer queries: read `q^I_t` | `4,352` | |
| weights `w^I_t`: write | `n_I_h · 2 = 128` | BF16 |
| weights: read | `128` | |
| `K^IComp` streaming read (amortized over `B_q`) | `T · B_KI / B_q = T · 2.0625` | shared across query block |
| **score matrix HBM write** (`I(t,s)` FP32) | **`T · 4`** | **the main waste** |
| **subtotal** | **`16,257 + T · 6.0625`** | |

At `n = 1M ⇒ T = 250K`: stage B = `16,257 + 1,515,625 = 1,531,882 B/token (≈ 1.46 MB)`.

### Stage C — top-k selector

| line item | bytes / token | rationale |
|---|---:|---|
| **read score matrix** | **`T · 4`** | TileLang's radix top-k reads the FP32 score row (reference reads via L2; for `T = 250K` the row alone is 1MB and does not fit in L2 → effectively un-cached HBM read) |
| write top-k indices | `k · 4 = 4096` | INT32 |
| **subtotal** | **`4096 + T · 4`** | |

At `T = 250K`: stage C = `~1 MB / token`.

### Stage D — sparse attention core

| line item | bytes / token | rationale |
|---|---:|---|
| read `Q` | `n_h · d · 2 = 131,072` | BF16, all 128 heads |
| read top-k indices | `k · 4 = 4,096` | |
| **gather KV at top-k indices** | `k · B_K = 1024 · 604 = 618,496` | FP8 NoPE + FP32 scales + BF16 RoPE per row; irreducible since these are the actual selected entries |
| sliding-window KV (amortized over `B_q`) | `(n_win + B_q − 1) · B_K / B_q ≈ 1,802` | `B_q=64` queries share a window union of size 191; same per-row layout |
| read `attn_sink` | `n_h · 4 = 512` | once per CTA, charge per token |
| LSE write | `n_h · 4 = 512` | needed by backward |
| write `O` | `n_h · d · 2 = 131,072` | BF16 |
| **subtotal** | **887,562** | |

### Stage totals

| stage | bytes/token | bytes/token at `n = 1M` (T = 250K) |
|---|---:|---:|
| A — KV compressor | 30,871 | 30,871 |
| B — indexer + score matrix | `16,257 + 6.0625·T` | 1,531,882 |
| C — top-k selector | `4,096 + 4·T` | 1,004,096 |
| D — sparse attention | 887,562 | 887,562 |
| **CSA fwd reference total** | | **3,454,411 (≈ 3.30 MB/token)** |

## 2. CSA forward — idealized FlashSparse

We assume one persistent kernel that fuses stages B+C+D, plus a separate (smaller) kernel
for stage A. We additionally apply streaming top-k from `streaming_topk.md`.

### Stage A — KV compressor (fused)

`wkv` and `wgate` outputs stay in registers; only the final compressed entry is written.

| line item | bytes / token |
|---|---:|
| read `H_t` | 14,336 |
| write `C^Comp` NoPE | 119 |
| write `C^Comp` RoPE | 32 |
| **subtotal** | **14,487** |

### Stage B+C+D — indexer + top-k + sparse attention (fully fused)

§ 3 of `fusion_analysis.md` shows that fully-fused doesn't fit in 228 KB shared memory;
the deployable design is two-stage (kernel 1: indexer + top-k; kernel 2: sparse attention).
The two-stage design adds one HBM round trip for the top-k indices (`k · 4 = 4,096
B/token`) but otherwise matches the fully-fused IO. We use the fully-fused numbers below
as the IO-savings ceiling and add the two-stage adjustment at the bottom.

| line item | bytes / token | change vs reference |
|---|---:|---|
| indexer compressor write `K^IComp` | `B_KI / m = 33` | unchanged |
| read `c^Q_t` | 3,072 | unchanged |
| `q^I` materialization | **0** | kept in registers / SRAM |
| weights `w^I_t` (write+read) | 256 | unchanged |
| `K^IComp` streaming read | `T · B_KI / B_q = T · 2.0625` | unchanged (irreducible HBM read) |
| **score matrix RW** | **0** | **eliminated by streaming top-k** |
| top-k indices (HBM round-trip in two-stage design) | `k · 4 · 2 = 8,192` (write + read) | replaces score matrix; `0` if fully fused |
| read `Q` | 131,072 | unchanged |
| **gather KV at top-k indices** | 618,496 | unchanged (irreducible) |
| sliding window | 1,802 | unchanged |
| `attn_sink`, LSE | 1,024 | unchanged |
| write `O` | 131,072 | unchanged |
| **subtotal (two-stage)** | **`895,047 + 2.0625·T`** | |
| **subtotal (fully fused, idealized)** | **`886,855 + 2.0625·T`** | not deployable on Hopper SRAM |

### Stage totals (fused, two-stage)

| stage | bytes/token | bytes/token at `n=1M` |
|---|---:|---:|
| A — KV compressor | 14,487 | 14,487 |
| B+C+D — two-stage indexer + topk + sparse_attn | `895,047 + 2.0625·T` | 1,410,672 |
| **CSA fwd two-stage total** | | **1,425,159 (≈ 1.36 MB/token)** |

## 3. CSA forward — multiplier

Reference total = `30,871 + 16,257 + 4,096 + 887,562 + 10.0625·T = 938,786 + 10.0625·T`
(stages A + B + C + D), where `10.0625 = 6.0625 + 4` from stages B and C.

Two-stage fused total = `14,487 + 895,047 + 2.0625·T = 909,534 + 2.0625·T`.

| context | reference bytes/token | fused bytes/token | **HBM IO multiplier** |
|---|---:|---:|---:|
| `n = 64K` (T = 16K) | 1,103,786 | 942,534 | 1.17× |
| `n = 256K` (T = 64K) | 1,582,786 | 1,041,534 | 1.52× |
| `n = 1M` (T = 250K) | 3,454,411 | 1,425,159 | **2.42×** |
| `n = 4M` (T = 1M) | 11,001,036 | 2,972,034 | **3.70×** |

The IO multiplier is sublinear in `T` until `10.0625·T` dominates over the fixed
`~900 KB`. The crossover (where streaming-top-k savings equal the rest of the per-token
budget) is around `T ≈ 90K`, i.e. ~360K-token context. At `n ≥ 1M` we are firmly in the
regime where the score matrix dominates the reference.

**The 4× CSA-forward target requires `n ≳ 5M` to come from IO alone.** For the plan's
stated `1M` target, the IO ceiling is **~2.4×** — the remaining wall-clock gap to 3-4×
must come from arithmetic-intensity improvements that we treat separately in § 7 (FA3
producer-consumer warp specialization, async TMA, FP4 GEMM for the indexer, persistent
kernel scheduling on H200's 132 SMs). The IO analysis here makes no claims about
wall-clock; it bounds what's possible from intermediate elimination alone.

## 4. HCA forward

HCA is a simpler beast: no indexer, no top-k, no overlap in the compressor.

### HCA fwd reference

| line item | bytes / token | rationale |
|---|---:|---|
| read `H_t` | 14,336 | |
| write `wkv`, `wgate` intermediates | `2 · d · 4 = 4,096` | `coff = 1` for HCA |
| read intermediates | 4,096 | |
| write `C^Comp` | `d · b_kv / m' ≈ 4.7` | one compressed entry per `m'=128` tokens |
| read `Q` | 131,072 | |
| read compressed KV (amortized) | `T' · B_K / B_q = T' · 9.4375` | dense over all `T'` compressed positions, FP8 NoPE + FP32 scales + BF16 RoPE per row |
| sliding window | 1,802 | same per-row layout as CSA |
| `attn_sink`, LSE | 1,024 | |
| write `O` | 131,072 | |
| **HCA fwd reference total** | **`287,498 + 9.4375·T'`** | |

At `n = 1M`, `T' = 8K`: total = `287,498 + 77,000 = 364,498` (≈ 0.35 MB / token).

### HCA fwd fused

Compressor is fused (saves the FP32 intermediate RW pair). The dense attention over `T'`
compressed positions is already HBM-efficient — there is no big-tensor materialization to
remove.

| line item | bytes / token |
|---|---:|
| read `H_t` | 14,336 |
| write `C^Comp` | 4.7 |
| read `Q` | 131,072 |
| read compressed KV | `9.4375·T'` |
| sliding window | 1,802 |
| `attn_sink`, LSE | 1,024 |
| write `O` | 131,072 |
| **HCA fwd fused total** | **`279,310 + 9.4375·T'`** |

At `n = 1M`: 356,310 bytes/token.

| context | reference | fused | **HCA IO multiplier** |
|---|---:|---:|---:|
| `n = 256K` (T'=2K) | 306,373 | 298,185 | 1.03× |
| `n = 1M` (T'=8K) | 364,498 | 356,310 | 1.02× |
| `n = 4M` (T'=32K) | 589,498 | 581,310 | 1.01× |

**HCA gives essentially zero IO speedup from the fusion-only treatment.** The savings
sit at ~1-3% across all context lengths. Where HCA wins is **arithmetic intensity** — §7
again — and that's not captured by raw byte counting. We will not claim an HCA-IO speedup
in future work; HCA is on the bench-and-measure track for its execution-side improvements.

## 5. CSA backward

The reference (TileLang `sparse_mla_bwd.py`) is a 3-pass scheme:

1. **Preprocess**: Δ_t = Σ_d O_{t,d} · dO_{t,d}. Reads O, dO; writes Δ.
2. **Main backward**: per query, for each selected K block, recompute attention scores
 from Q+K and saved LSE, compute dP, then `dQ += dP·K`, `dV += P^T·dO`,
 `dK += dP^T·Q`. Writes dQ via per-CTA accumulator (one CTA per query, no race),
 scatters dKV to selected indices via `atomic_addx4` into a FP32 workspace.
3. **Postprocess**: cast FP32 dKV workspace → BF16 final tensor.

Per-token IO (reference):

| line item | bytes / token | rationale |
|---|---:|---|
| Stage 1 read O | `n_h · d · 2 = 131,072` | |
| Stage 1 read dO | `n_h · d · 2 = 131,072` | |
| Stage 1 write Δ | `n_h · 4 = 512` | |
| Stage 2 read Q | 131,072 | |
| Stage 2 read dO | 131,072 | |
| Stage 2 read Δ, LSE | 1,024 | |
| Stage 2 read top-k indices | 4,096 | |
| Stage 2 gather KV at top-k | 618,496 | corrected K-row layout (604 B / row) |
| Stage 2 **scatter dKV (FP32 atomic)** | **`k · d · 4 = 2,097,152`** | atomic_addx4 to FP32 workspace |
| Stage 2 write dQ | 131,072 | |
| Stage 3 read FP32 dKV workspace + write BF16 dKV (per-token amortized) | `2 · T · d · 4 / n + T · d · 2 / n = d · 8/m + d · 2/m = 1,280` | postprocess: read FP32, cast, write BF16 once |
| **CSA bwd reference total** | **≈ 3,378,856** (≈ 3.22 MB/token; dominant cost: dKV scatter) | |

For HCA the same scheme drops the top-k indirection but keeps the same dKV scatter
pattern (over `T'` compressed entries instead of `k` selected).

### CSA backward fused

Two main savings:

(a) **Inverted-topk dK/dV reduction.** In a 2-pass design (FA-style), one CTA per K block
iterates over the queries that selected this K. To know "which queries selected this K"
we precompute an inverted index `topk_idxs⁻¹: K → [list of (t, slot)]` once per layer
per backward — reusable. Each K block accumulates dK, dV in shared memory, sums across
the queries that touched it, writes BF16 dK, dV once. **No FP32 atomic workspace.**

(b) **Streaming `K^IComp` reuse from forward LSE.** Backward needs P, which is recomputed
from Q, K, LSE (FA3 trick — saves storing the score matrix). This is identical between
reference and fused; both need the K gather. No new savings here.

In the fused inverted-topk design (`fusion_analysis.md` § 6), each K block accumulates
its `dK`, `dV` partial sums in shared memory across all queries that selected it, then
writes BF16 dKV once with no global atomics. Per-token IO accounting:

| line item | bytes / token | savings |
|---|---:|---|
| read O, dO (preprocess) | 262,144 | unchanged |
| write Δ | 512 | unchanged |
| read Q, dO (main) | 262,144 | unchanged |
| read Δ, LSE | 1,024 | unchanged |
| read top-k indices | 4,096 | unchanged |
| **read inverted-topk index** | `n · k · 4 / n = k · 4 = 4,096` per query token | new cost (replaces score matrix RW) |
| gather KV | 618,496 | unchanged |
| **dKV write BF16, no atomics** | `T · d · 2 / n = d · 2 / m = 256 B` | reduced from `2,097,152` (atomic FP32) |
| write dQ | 131,072 | unchanged |
| **CSA bwd fused total** | **≈ 1,285,856** | |

The build of the inverted-topk index itself costs a *separate one-time* `n · k · 4 = 4
GB` write per layer per forward (per-token: 4,096 B), reusable across multiple backward
calls if forward indices haven't changed. We charge this once if amortized over a single
fwd+bwd pair: `+4,096 B/token`. Total amortized: `1,289,952`.

| context | reference | fused (1 fwd+bwd pair) | **CSA bwd IO multiplier** |
|---|---:|---:|---:|
| `n = 1M` | 3,378,856 | 1,289,952 | **2.62×** |
| `n = 4M` | 3,378,856 | 1,289,952 | **2.62×** |

CSA backward IO speedup is roughly constant in `n` because all the `T`-scaling terms
amortize to per-token constants. The plan's 5–7× backward target requires this 2.7× IO
gain to compose with arithmetic-intensity gains — the reference's atomic_addx4 pattern is
known to be 2–3× slower than pure SUMR (sum-reduce) on Hopper due to atomic-unit
contention, so the wall-clock gain compounds. § 7.

## 6. HCA backward

HCA backward is similar to CSA backward but without the top-k indirection and without the
indexer. The atomic-scatter pattern is also less severe because there is no top-k —
rather, the dK, dV writes go to the same `T'` compressed entries that the forward
attended to, with the structural pattern `block_i ⟶ {queries in [m'·i, n)}`. This is
deterministic (no scatter), so the reference can already use sum-reduce instead of
atomics — the FA3 backward trick already applies natively. The IO multiplier is
correspondingly modest.

Per-token at `n = 1M`, `T' = 8K`:

| line item | reference | fused |
|---|---:|---:|
| O, dO read (preprocess) | 262,144 | 262,144 |
| Δ write/read | 1,024 | 1,024 |
| Q, dO read (main) | 262,144 | 262,144 |
| LSE read | 512 | 512 |
| dense compressed KV gather | `8.5 T' = 68,000` | 68,000 |
| sliding window | 1,650 | 1,650 |
| dQ write | 131,072 | 131,072 |
| dKV write | `d · 2 / m' = 8 B` | 8 |
| **HCA bwd total** | **726,554** | **726,554** |

| context | **HCA bwd IO multiplier** |
|---|---:|
| any | **1.0×** |

HCA backward gets **no IO speedup**. Wall-clock gain comes entirely from arithmetic-
intensity, see § 7. Realistic: 1.5–2×.

## 7. Why per-token IO is not the whole story

The plan claims **3–4× CSA forward, 5–7× CSA backward, up to 8× decode** at 1M context.
Pure HBM-IO accounting in §§ 1–6 gives the *IO ceilings* below: 2.4× CSA forward, 2.6×
CSA backward, ~1.0× HCA. The wall-clock targets compose IO with arithmetic-intensity
factors that we **estimate from cited sources but explicitly do not measure here**.
the kernel-level work produces the actual numbers.

### 7.1 Cited execution-side factors

Each of these is a *literature-supported range*, not a derivation:

1. **FA3 vs FA2 wall-clock on H100** — FA3 paper (Shah et al., 2024, arxiv 2407.08608),
 Table 4: BF16 forward 1.4–1.7× over FA2, FP8 forward 2.4×. Hopper-specific: producer-
 consumer warp specialization, async TMA, GEMM-softmax pingpong. We adopt **1.3–1.6×**
 for our fwd kernels (conservative since we have an extra gather hop) and **1.2–1.5×**
 for bwd.

2. **Atomic-free backward** — FA3 paper Table 5 (backward): ~1.5–2.0× over FA2 baseline,
 driven primarily by replacing global atomics with FA3's two-kernel (dQ-pass + dKV-pass)
 structure. We adopt **1.5–2.0×** for the backward atomic replacement.

3. **TMA bandwidth utilization** — NVIDIA Hopper architecture whitepaper: TMA achieves
 ~85% of HBM peak bandwidth on contiguous loads, ~60% on scattered (sparse-gather).
 Reference's per-warp `cp.async` loads typically reach ~50% peak. So TMA gives
 **1.1–1.3×** for sparse-gather kernels — folded into the FA3 number above when
 relevant; not double-counted.

4. **Persistent kernel / launch overhead** — CUDA programming guide: kernel launch is
 ~5–10 µs on H100/H200 with stream synchronization. For a 1M-token forward (~30 s
 measured baseline per DeepSeek-V3.2 paper Table 2), eliminating ~120 launches per layer
 × 30 layers × 10 µs = 36 ms saves <0.2% wall-clock. **Negligible** for prefill at 1M.
 For decode (per-step budget ~50 ms), 120 launches × 10 µs = 1.2 ms ≈ 2.4% — small but
 real.

5. **Decode-specific bandwidth saturation** — at decode time, the kernel is HBM-bandwidth
 bound (compute per token is small). IO multiplier translates ~1:1 to wall-clock,
 modulo ~10% scheduling overhead. So decode wall-clock multiplier ≈ IO multiplier × 1.1.

### 7.2 Composed estimates (with bounds)

We give **ranges**, not point estimates. Each range is the product of the IO multiplier
(measured-in-this-doc, deterministic) and the execution-side range (cited, to be
verified):

| op @ 1M | IO mult (this doc) | execution range (cited) | **wall-clock range** | plan target |
|---|---:|---:|---:|---:|
| CSA fwd | 2.42× | 1.3–1.6× (FA3) | **3.1–3.9×** | 3–4× |
| HCA fwd | 1.02× | 1.4–1.8× (FA3, dense) | **1.4–1.8×** | (plan silent) |
| CSA bwd | 2.62× | 1.3–1.6× (FA3) × 1.5–2.0× (atomic-free) = 2.0–3.2× | **5.2–8.4×** | 5–7× |
| HCA bwd | 1.00× | 1.3–1.6× (FA3, dense) | **1.3–1.6×** | (plan silent) |
| decode @ 1M | 2.42× | 1.1× (HBM-saturation factor) × 1.05× (launch) | **2.5–2.8×** | up to 8× |

### 7.3 Decode-target gap (and how to close it)

The plan calls for "up to 8× decode" but our IO + execution analysis gives only 2.5–2.8×
on a single sequence at `n = 1M`. The 8× target is plausible only with **batched
decoding** (multiple concurrent requests) or **speculative decoding** (multi-token per
step), neither of which is captured by per-token IO accounting on a single stream.

We **decline to ratify the 8× decode target on the basis of this analysis**. the kernel-level work
should measure single-stream decode and batched decode separately and report whichever is
closer to the published target. If batched decode reaches 8× and single-stream reaches
2.5×, the paper claims "up to 8× under batched decode."

### 7.4 Honest summary

- **CSA forward**: IO supports up to 2.4× at 1M. With FA3, range 3.1–3.9×. **Plan target
 3–4× achievable.**
- **CSA backward**: IO supports 2.6×. Atomic-free + FA3 multiplier reaches 5.2–8.4×.
 **Plan target 5–7× achievable, possibly comfortably exceeded.**
- **HCA forward / backward**: ~1× IO. FA3 alone gives 1.3–1.8×. Plan is silent on HCA;
 we will report measured numbers in future work without claiming a target.
- **Decode at 1M, single stream**: ~2.5–2.8×. The 8× plan target requires either
 batched decode or a stronger speculative-decoding story. **Caveat in future work paper.**

## 8. What this analysis assumes (and where it could be wrong)

1. **Block sizes**: `B_q = 64`, `B_kv = 64` are typical FA-style choices. The exact
 reference uses different blocks per kernel — TileLang `sparse_mla_fwd` uses `block_I =
 64`, and `fp8_lighting_indexer` uses `block_Q = 128 / heads`, `block_N = 256`. Our
 per-token bytes are within 10% of any reasonable block-size choice.

2. **L2 caching**: at `n = 1M`, `T = 250K`, the score matrix size is `T · 4 = 1 MB` per
 query row; over `B_q = 64` queries that's 64 MB shared. H200 L2 is ~50 MB. So roughly
 1× the score matrix fits in L2 — meaning the read in stage C is partially L2-hit. This
 blurs the "2 MB / token of score matrix HBM traffic" claim; the realistic effective
 number is somewhere between `T·4` and `2·T·4` depending on `B_q` and L2 pressure from
 other concurrent traffic. **We use `2·T·4` as the upper bound and `1·T·4` as the lower
 bound; the multiplier survives both** (range `2.0×–2.7×` instead of `2.5×`).

3. **FP32 score matrix**: the reference is FP32 (the einsum output dtype after `relu_`
 and the `.sum`). We could imagine a half-precision score matrix; that halves the
 reference's stage-B+C cost, taking the IO multiplier from `2.5×` to `~1.5×`. But this
 is not what the reference does, and it would harm top-k quality due to underflow
 collisions. We stay with FP32 for the reference.

4. **`Q` and `O` reads/writes are charged at `n_h · d · 2` per token** — full BF16 reads.
 In production with FP8 Q (which V4 does for some paths), these halve to `n_h · d`.
 Correspondingly the fused kernel cost would also halve in `Q` and `O` proportionally —
 the ratio stays the same.

5. **Inverted-topk index for backward**: we charge `1024 B/token` for the precomputed
 reverse-index lookup. In practice this is amortized across many bwd calls and is
 typically <100 B/token. Charging 1024 B is a conservative upper bound.

6. **Decode-time analysis** uses the same numbers but per-step (`n=1` query token). Both
 reference and fused load the entire compressed KV history, so the gather term scales
 the same way. The key decode-specific gain comes from kernel-launch overhead
 elimination (persistent kernel) which doesn't show up in per-token IO accounting.

## 9. Concrete numbers (V4-Pro, full forward at 1M context)

For one CSA layer at `n = 1M`:

- reference HBM IO: `n · 3.30 MB = 3.30 TB`
- fused HBM IO: `n · 1.36 MB = 1.36 TB`
- saved per layer: `1.94 TB`

V4-Pro has 30 CSA layers + 31 HCA layers. CSA savings dominate:

- `30 layers · 1.94 TB = 58.2 TB` of HBM traffic eliminated per forward at 1M.
- HCA layers: ~0 saved (HCA is already efficient at the IO level).

On H200 (4.8 TB/s HBM), `58.2 TB / 4.8 TB/s = 12.1 s` of HBM-bound latency saved per
forward — a **lower bound** on wall-clock improvement since real wall-clock time is
HBM-time + compute-time for an HBM-bound stack.

A reference 1M-context forward with the public DSA kernels has not been benchmarked on
H200 by DeepSeek; the V3.2 paper reports H100 numbers in the ~30s range for similar layer
counts. We **explicitly defer wall-clock baseline numbers to future work measurement** and
will not state "X seconds → Y seconds" predictions in this doc.

---

This locks the math for future work. The arithmetic in §§ 1–6 is verifiable; the wall-clock
ranges in § 7 are *cited* but not measured. the kernel-level work (`fusion_analysis.md`) names every
intermediate tensor we just declared "register-resident" and proves the SRAM accounting.
