#!/usr/bin/env bash
# Paper-experiment driver — run ALL the benches needed for the
# Memory-Bounded CSA paper, in priority order.
#
# Usage on an H100/H200 with the package installed:
#   cd /path/to/StreamIndex
#   bash benchmarks/run_paper_benchmarks.sh 2>&1 | tee benchmarks/paper_run_$(date +%Y%m%d_%H%M%S).log
#
# Stops on the first non-zero exit so we don't burn GPU time after a failure.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==========================================================="
echo "FlashSparse paper-experiment run — $(date -Iseconds)"
echo "==========================================================="
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
echo

# 1. THE GATING EXPERIMENT — TileLang verification at long context.
# Outcome decides paper viability per user's critique.
echo
echo "### [1/4] TileLang verification bench (V4-Flash dims, S up to 256K) ###"
echo
python benchmarks/bench_vs_tilelang_pipelined.py

# 2. Chunked-indexer recall study.
echo
echo "### [2/4] Chunked-indexer recall study (V4-Flash dims) ###"
echo
python benchmarks/bench_chunked_indexer_recall.py

# 3. V4-Pro full-dim bench — own the Triton-vs-CUDA peak gap.
echo
echo "### [3/4] V4-Pro production-dim bench (H=128, D=512) ###"
echo
python benchmarks/bench_v4_pro_dims.py

# 4. Existing long-context fwd+bwd at toy dims (sanity).
echo
echo "### [4/4] sparse_attn fwd+bwd long-context (toy dims, sanity) ###"
echo
python benchmarks/bench_long_context_fwd_bwd.py

echo
echo "==========================================================="
echo "ALL paper benchmarks complete — $(date -Iseconds)"
echo "==========================================================="
