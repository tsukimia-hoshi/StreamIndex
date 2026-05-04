#!/usr/bin/env bash
# Day-1 setup script for the 8xH200 VM.
#
# Sequential checklist that brings the box from "fresh nvidia-cuda image" to
# "ready to run V4-Flash with our eval harness." Designed to run unattended
# in ~30 minutes; surface any failures loudly.
#
# Usage on the VM:
# bash eval/setup_v4_flash.sh 2>&1 | tee eval/setup_$(date +%Y%m%d_%H%M%S).log
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==========================================================="
echo "Day-1 V4-Flash setup — $(date -Iseconds)"
echo "==========================================================="
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
echo

# 1. Python deps (assumes ~/flash-sparse/.venv exists from prior session).
if [ ! -d ~/flash-sparse/.venv ]; then
 echo "Creating fresh venv at ~/flash-sparse/.venv ..."
 python3 -m venv ~/flash-sparse/.venv
fi
# shellcheck source=/dev/null
source ~/flash-sparse/.venv/bin/activate
echo "Using python: $(which python)"
python -V

pip install --upgrade pip
pip install \
 'torch>=2.5' \
 'transformers>=4.50' \
 'datasets>=2.20' \
 'accelerate>=0.30' \
 'sentencepiece' \
 'tiktoken' \
 'flash-attn>=2.7' \
 'triton>=3.7' \
 'tilelang' \
 'vllm>=0.7' \
 'sglang>=0.4' || true

# 2. Reinstall flash-sparse (this repo) in editable mode.
pip install -e .

# 3. Hugging Face login (required for V4-Flash if gated).
if [ -z "${HF_TOKEN:-}" ]; then
 echo "WARNING: HF_TOKEN not set in env. If V4-Flash is a gated repo,"
 echo " model download will fail. Set with:"
 echo " export HF_TOKEN=<your-token>"
 echo " and re-run."
fi

# 4. Pre-download V4-Flash weights to local cache (large).
MODEL_NAME="${V4_MODEL:-deepseek-ai/DeepSeek-V4-Flash}"
echo
echo "Pre-downloading $MODEL_NAME ..."
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tokenizer = AutoTokenizer.from_pretrained('$MODEL_NAME', trust_remote_code=True)
print(f'Tokenizer ok ({tokenizer.__class__.__name__}, vocab={tokenizer.vocab_size})')
# Don't actually load weights into VRAM here — we just want the cache populated.
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id='$MODEL_NAME', allow_patterns=['*.json', '*.safetensors', '*.py', 'tokenizer*'])
print(f'Weights cached to: {path}')
"

# 5. Generate eval data (CPU-only, ~5 min).
mkdir -p eval/data
echo
echo "Generating eval data (needle-haystack, LongBench-V2, SWE-bench)..."
for SLEN in 65536 262144 1048576; do
 SLEN_HUMAN=$((SLEN / 1024))K
 OUT="eval/data/needle_${SLEN_HUMAN}.jsonl"
 if [ ! -f "$OUT" ]; then
 python eval/needle_haystack.py \
 --max-seq-len "$SLEN" --n-samples 50 \
 --tokenizer "$MODEL_NAME" --out "$OUT"
 else
 echo " $OUT already exists; skipping."
 fi
done

for SLEN in 65536 262144; do
 SLEN_HUMAN=$((SLEN / 1024))K
 OUT="eval/data/longbench_${SLEN_HUMAN}.jsonl"
 if [ ! -f "$OUT" ]; then
 python eval/longbench_v2_loader.py \
 --target-len "$SLEN" --max-samples 50 \
 --tokenizer "$MODEL_NAME" --out "$OUT" || \
 echo " longbench_${SLEN_HUMAN} skipped (no records in size bucket)."
 else
 echo " $OUT already exists; skipping."
 fi
done

for SLEN in 65536 262144; do
 SLEN_HUMAN=$((SLEN / 1024))K
 OUT="eval/data/swebench_${SLEN_HUMAN}.jsonl"
 if [ ! -f "$OUT" ]; then
 python eval/swebench_loader.py \
 --target-len "$SLEN" --max-samples 30 \
 --tokenizer "$MODEL_NAME" --out "$OUT" || \
 echo " swebench_${SLEN_HUMAN} skipped (no records in size bucket)."
 else
 echo " $OUT already exists; skipping."
 fi
done

# 6. Sanity: try the materialize-baseline path on a tiny sample to validate
# that model loading + harness work end-to-end before doing real runs.
echo
echo "Smoke test: materialize indexer at S=8K with 2 needle samples..."
python eval/needle_haystack.py \
 --max-seq-len 8192 --n-samples 2 \
 --tokenizer "$MODEL_NAME" --out eval/data/_smoke_8K.jsonl
python eval/run_v4_inference.py \
 --model "$MODEL_NAME" \
 --indexer materialize \
 --samples eval/data/_smoke_8K.jsonl \
 --out eval/runs/_smoke_8K_materialize.jsonl \
 --max-new-tokens 16 \
 --tp-size 2 \
 --max-samples 2

echo
echo "==========================================================="
echo "Day-1 setup COMPLETE — $(date -Iseconds)"
echo "==========================================================="
echo
echo "Next steps:"
echo " 1. Inspect transformers_modules/<v4-flash>/modeling_*.py to find"
echo " the CSA attention class and its forward signature."
echo " 2. Implement _apply_chunked_indexer_patch in eval/run_v4_inference.py."
echo " 3. Parity check: smoke test with --indexer chunked at S=2048,"
echo " compare logits to materialize within 1e-3 BF16."
echo " 4. Real runs: needle 64K/256K/1M, longbench 64K/256K, swebench 64K/256K"
echo " for both materialize and chunked indexers."
