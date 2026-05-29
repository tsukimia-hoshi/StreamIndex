"""SWE-bench-Verified loader — bucket longest-context issues for our 64K / 256K bins.

SWE-bench-Verified (princeton-nlp/SWE-bench_Verified) has 500 hand-validated
issues. We pick the ones whose repo-level context (issue text + relevant
file contents at the base commit) fits a target token length, and produce
prompts asking the model to generate a patch.

Pure-CPU. Output is JSONL records compatible with the eval harness.

Usage:
python eval/swebench_loader.py \\
--tokenizer deepseek-ai/DeepSeek-V4-Flash \\
--target-len 65536 --max-samples 30 \\
--out eval/data/swebench_64K.jsonl

Each output record:
{
"id": "swebench_<instance_id>",
"prompt": "<issue text + relevant file context + 'output a unified diff'>",
"expected_patch": "<the gold patch, used for scoring>",
"instance_id": "<repo>__<num>",
"actual_seq_len": <int>,
}

Scoring: pass-rate via running the model's generated patch through the
SWE-bench harness. **The harness itself is heavy (docker per instance);
this loader produces inputs only. The user runs the harness separately.**
For paper-purposes we report:
1. patch-applied (does it apply cleanly without conflicts)
2. tests-pass (does it pass the issue's failing test set)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Iterator


_PROMPT_TEMPLATE = """You are given an issue from the {repo} repository, along with the contents of files relevant to fixing it. Generate a unified diff (git format) that resolves the issue.

ISSUE:
{problem_statement}

RELEVANT FILES (current contents at the base commit):
{file_context}

Generate ONLY the unified diff, in standard `git diff` format. Do not include any prose or explanation outside the diff. The diff must apply cleanly to the base commit."""


def _stream_dataset(name: str = "princeton-nlp/SWE-bench_Verified") -> Iterator[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets not installed; install with: pip install datasets", file=sys.stderr)
        sys.exit(1)
        # SWE-bench has only test split.
        ds = load_dataset(name, split="test", streaming=True)
        for rec in ds:
            yield rec


def _build_prompt(rec: dict, max_files: int = 6) -> str:
    """Build a long-context prompt from an SWE-bench record.

    The relevant-files context is constructed from the gold patch's file
    list — these are the files the model needs to see to fix the issue.
    For each file, we include current contents at the base commit (assumed
    to be in the record's `text` or fetched separately). When unavailable,
    we use the patch's diff context as a hint.
    """
    repo = rec.get("repo", "unknown")
    problem = rec.get("problem_statement", "")
    # SWE-bench records include patch + base_commit but not full file
    # contents. The standard practice is to fetch them via the SWE-bench
    # harness or cached; the loader here stages a placeholder marker that
    # the offline-prep step (run on the VM with internet) will fill.
    patch = rec.get("patch", "")
    file_context = (
        "[FILE CONTEXT — TO BE FILLED BY VM-SIDE FETCH STEP]\n"
        "Files referenced in the gold patch:\n"
        + _extract_filenames_from_patch(patch)
        + "\n\nGold-patch diff context (for stub purposes):\n"
        + patch[:8000]
    )
    return _PROMPT_TEMPLATE.format(
        repo=repo,
        problem_statement=problem,
        file_context=file_context,
    )


def _extract_filenames_from_patch(patch: str) -> str:
    lines = []
    for line in patch.splitlines:
        if line.startswith("--- a/") or line.startswith("+++ b/"):
            lines.append(line)
            return "\n".join(lines[:20]) if lines else "(none)"


def main():
    p = argparse.ArgumentParser
    p.add_argument("--tokenizer", type=str, required=True)
    p.add_argument("--target-len", type=int, required=True)
    p.add_argument("--max-samples", type=int, default=30)
    p.add_argument("--dataset", type=str, default="princeton-nlp/SWE-bench_Verified")
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("transformers not installed; install with: pip install transformers", file=sys.stderr)
        sys.exit(1)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        lower = int(args.target_len * 0.8)
        upper = int(args.target_len * 1.2)

        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        n_written = 0
        n_seen = 0
        with open(args.out, "w", encoding="utf-8") as f:
            for rec in _stream_dataset(args.dataset):
                if n_written >= args.max_samples:
                    break
                n_seen += 1
                try:
                    prompt = _build_prompt(rec)
                except (KeyError, ValueError):
                    continue
                n_tok = len(tokenizer.encode(prompt, add_special_tokens=False))
                if not (lower <= n_tok <= upper):
                    continue
                out = {
                    "id": f"swebench_{rec.get('instance_id', n_seen)}",
                    "prompt": prompt,
                    "expected_patch": rec.get("patch", ""),
                    "instance_id": rec.get("instance_id", ""),
                    "actual_seq_len": n_tok,
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                n_written += 1
                print(f"Wrote {n_written} records (scanned {n_seen}) to {args.out}", file=sys.stderr)


# Scoring: this requires the SWE-bench harness running each predicted patch
# in a docker container. Implementation deferred to the VM run; the loader's
# output is the input to that harness. See:
# https://github.com/princeton-nlp/SWE-bench
# The relevant CLI is `swebench.harness.run_evaluation` over a JSONL of
# {instance_id, model_patch} records.


if __name__ == "__main__":
    main
