# V4-Flash end-to-end evaluation harness

Pre-staged for **addition 1** of the 14-day paper plan
(the project roadmap). All scripts here are CPU-only or
require an H200; nothing runs as part of `pip install`.

## Layout

| File | Purpose |
|---|---|
| `setup_v4_flash.sh` | Day-1 VM bring-up: deps, model download, eval-data gen, smoke test. |
| `needle_haystack.py` | Generate needle-in-a-haystack prompts at arbitrary token length. |
| `longbench_v2_loader.py` | Load LongBench-V2, bucket by tokenized length. |
| `swebench_loader.py` | Load SWE-bench-Verified, bucket by tokenized length. |
| `run_v4_inference.py` | Inference harness — record predictions + TTFT + decode tps + peak HBM. |

## Day-1 plan (8×H200 VM)

```bash
# On the GPU host:
git clone https://github.com/RightNow-AI/StreamIndex
cd StreamIndex
pip install -e .
bash eval/setup_v4_flash.sh 2>&1 | tee eval/setup_$(date +%Y%m%d_%H%M%S).log
```

Setup script will:
1. Create / activate `~/flash-sparse/.venv`.
2. Install torch, transformers, vllm, sglang, triton, tilelang, flash-attn.
3. Reinstall `flash-sparse` editable.
4. Pre-download V4-Flash weights to HF cache.
5. Generate eval data: needle at 64K/256K/1M, LongBench at 64K/256K, SWE-bench at 64K/256K.
6. Smoke-test: load model, run two needle prompts at 8K with stock (materialize)
 indexer, confirm output is finite.

After setup, the **chunked-indexer patch** (`_apply_chunked_indexer_patch`
in `run_v4_inference.py`) needs to be implemented based on the actual
V4-Flash modelling code. That's the day-1 surgery — see the function's
docstring for the recipe.

## Day-2 onward — running real benchmarks

```bash
# For each (workload, sequence_length, indexer):
python eval/run_v4_inference.py \
 --model deepseek-ai/DeepSeek-V4-Flash \
 --indexer {materialize|chunked} \
 --samples eval/data/needle_64K.jsonl \
 --out eval/runs/needle_64K_{indexer}.jsonl \
 --max-new-tokens 64 --tp-size 2
```

The 3×3 grid (3 workloads × 3 lengths) × 2 indexers = **18 runs**. Plan:
materialize first to establish the OOM threshold, then chunked through the
full grid. Expected outcome:

| Workload | 64K (mat) | 256K (mat) | 1M (mat) | 64K (chunked) | 256K (chunked) | 1M (chunked) |
|---|---|---|---|---|---|---|
| needle | runs | OOM | OOM | runs | runs | runs |
| longbench | runs | OOM | n/a | runs | runs | n/a |
| swebench | runs | OOM | n/a | runs | runs | n/a |

If the actual outcome differs, see the project roadmap "what could go
wrong" for the failure-mode handlers.

## Scoring

Each loader has a `score` function:

```python
from eval.needle_haystack import score as score_needle
res = score_needle("eval/data/needle_64K.jsonl",
 "eval/runs/needle_64K_chunked.jsonl")
print(f"accuracy={res['accuracy']:.3f} by_depth={res['by_depth']}")
```

For SWE-bench, scoring requires running the [SWE-bench harness](https://github.com/princeton-nlp/SWE-bench)
on the predicted patches — that's a docker-per-instance step done after
inference, not in the harness itself.

## Pass criteria (paper §Addition-1)

- **Quality**: chunked path within 1% of materialize path at S where both
 run.
- **TTFT**: chunked TTFT ≤ 1.2× materialize TTFT.
- **HBM**: chunked peak HBM bounded (ideally flat or sub-linear in S).

If quality regresses by > 1%, suspect dtype/shape mismatch in the patch —
not the algorithm. If TTFT regresses by > 20%, suspect Python overhead in
the chunked driver — profile and either fuse more or document the overhead.
