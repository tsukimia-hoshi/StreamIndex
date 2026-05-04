"""LongBench-V2 loader — slice longest examples for our 64K / 256K bins.

LongBench-V2 (THUDM/LongBench-V2) is the standard long-document QA
benchmark. We use the longest examples and bucket them by tokenized length
to fill our S∈{64K, 256K} eval slots.

Pure-CPU. Output is JSONL records compatible with the eval harness.

Usage:
 python eval/longbench_v2_loader.py \\
 --tokenizer deepseek-ai/DeepSeek-V4-Flash \\
 --target-len 65536 --max-samples 50 \\
 --out eval/data/longbench_64K.jsonl

Each output record:
 {
 "id": "longbench_v2_<orig-id>",
 "prompt": "<context>\\n\\n<question>",
 "answer": "<expected-answer>",
 "category": "<single_doc_qa | multi_doc_qa | ...>",
 "actual_seq_len": <int>,
 }

Scoring: exact-match (and partial-match for free-form answers) — see
`score_longbench_v2` below.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Iterator


def _stream_dataset(name: str = "THUDM/LongBench-v2") -> Iterator[dict]:
 """Stream LongBench-V2 examples from HF datasets."""
 try:
 from datasets import load_dataset
 except ImportError:
 print("datasets not installed; install with: pip install datasets",
 file=sys.stderr)
 sys.exit(1)
 ds = load_dataset(name, split="train", streaming=True)
 for rec in ds:
 yield rec


def _build_prompt(rec: dict) -> str:
 """LongBench-V2 record → prompt string.

 Standard schema: {context, question, choice_A, choice_B, choice_C, choice_D, answer}.
 For multiple-choice we present the choices and ask for a letter.
 """
 if "context" in rec and "question" in rec:
 choices = ""
 for letter in ("A", "B", "C", "D"):
 key = f"choice_{letter}"
 if key in rec and rec[key]:
 choices += f"({letter}) {rec[key]}\n"
 if choices:
 return (
 rec["context"]
 + "\n\nQuestion: " + rec["question"]
 + "\n\nChoices:\n" + choices
 + "\nAnswer with the letter only ((A), (B), (C), or (D))."
 )
 return rec["context"] + "\n\nQuestion: " + rec["question"] + "\n\nAnswer:"
 raise ValueError(f"unexpected record schema: {list(rec.keys)[:8]}")


def main:
 p = argparse.ArgumentParser
 p.add_argument("--tokenizer", type=str, required=True)
 p.add_argument("--target-len", type=int, required=True,
 help="Target prompt token length (we bucket within ±20%).")
 p.add_argument("--max-samples", type=int, default=50)
 p.add_argument("--dataset", type=str, default="THUDM/LongBench-v2")
 p.add_argument("--out", type=str, required=True)
 args = p.parse_args

 try:
 from transformers import AutoTokenizer
 except ImportError:
 print("transformers not installed; install with: pip install transformers",
 file=sys.stderr)
 sys.exit(1)

 tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
 lower = int(args.target_len * 0.8)
 upper = int(args.target_len * 1.2)

 os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
 n_written = 0
 with open(args.out, "w", encoding="utf-8") as f:
 for i, rec in enumerate(_stream_dataset(args.dataset)):
 if n_written >= args.max_samples:
 break
 try:
 prompt = _build_prompt(rec)
 except ValueError:
 continue
 n_tok = len(tokenizer.encode(prompt, add_special_tokens=False))
 if not (lower <= n_tok <= upper):
 continue
 out = {
 "id": f"longbench_v2_{rec.get('_id', rec.get('id', i))}",
 "prompt": prompt,
 "answer": rec.get("answer", ""),
 "category": rec.get("category", "unknown"),
 "actual_seq_len": n_tok,
 }
 f.write(json.dumps(out, ensure_ascii=False) + "\n")
 n_written += 1
 print(f"Wrote {n_written} records to {args.out}", file=sys.stderr)


def score(samples_path: str, predictions_path: str) -> dict:
 """Score LongBench-V2 predictions.

 Multiple-choice: extract the letter (A/B/C/D) and exact-match.
 """
 truth = {rec["id"]: rec for rec in (json.loads(l) for l in open(samples_path))}
 preds = {rec["id"]: rec["prediction"]
 for rec in (json.loads(l) for l in open(predictions_path))}
 correct = 0
 n = 0
 by_category = {}
 for sid, rec in truth.items:
 if sid not in preds:
 continue
 n += 1
 pred = str(preds[sid]).strip
 m = re.search(r"\(?([A-D])\)?", pred.upper)
 pred_letter = m.group(1) if m else pred.upper[:1]
 ok = pred_letter == str(rec["answer"]).strip.upper
 correct += int(ok)
 cat = rec.get("category", "unknown")
 by_category.setdefault(cat, {"correct": 0, "n": 0})
 by_category[cat]["correct"] += int(ok)
 by_category[cat]["n"] += 1
 return {
 "accuracy": correct / n if n else 0.0,
 "by_category": {c: v["correct"]/v["n"] for c, v in by_category.items},
 "n": n,
 }


if __name__ == "__main__":
 main
