"""Needle-in-a-haystack generator for long-context retrieval evaluation.

Generates prompts of a target token length with a single magic-number
needle inserted at a configurable depth. Used in addition 1 (V4-Flash
end-to-end) at S ∈ {64K, 256K, 1M}.

Pure-CPU; no model loading. Output is JSONL with one record per prompt.

Usage:
 python eval/needle_haystack.py \\
 --max-seq-len 65536 --n-samples 50 \\
 --depths 0.05,0.25,0.5,0.75,0.95 \\
 --tokenizer deepseek-ai/DeepSeek-V4-Flash \\
 --out eval/data/needle_64K.jsonl

Each record:
 {
 "id": "needle_64K_d50_0",
 "prompt": "<long passage with magic number embedded at depth 0.50>",
 "needle": "magic number is 47294",
 "answer": "47294",
 "depth": 0.50,
 "target_seq_len": 65536,
 "actual_seq_len": 65414,
 }

Scoring: exact-match on extracted integer in the model's response.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import List

# Filler corpus — Project Gutenberg public-domain text (Pride and Prejudice).
# Inlined as a fallback so the generator works without internet.
_FALLBACK_FILLER = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or other of their daughters. My dear Mr. Bennet, said his lady to him one day, have you heard that Netherfield Park is let at last? Mr. Bennet replied that he had not. But it is, returned she; for Mrs. Long has just been here, and she told me all about it. Mr. Bennet made no answer. Do you not want to know who has taken it? cried his wife impatiently. You want to tell me, and I have no objection to hearing it. This was invitation enough. Why, my dear, you must know, Mrs. Long says that Netherfield is taken by a young man of large fortune from the north of England; that he came down on Monday in a chaise and four to see the place, and was so much delighted with it that he agreed with Mr. Morris immediately; that he is to take possession before Michaelmas, and some of his servants are to be in the house by the end of next week. What is his name? Bingley. Is he married or single? Oh! single, my dear, to be sure! A single man of large fortune; four or five thousand a year. What a fine thing for our girls!"""


def _load_filler_text -> str:
 """Return a long string of natural-language filler.

 Tries multiple sources in order:
 1. NLTK gutenberg corpus (richer if installed).
 2. local cached download.
 3. inlined fallback (always works).
 """
 try:
 import nltk
 from nltk.corpus import gutenberg
 try:
 text = " ".join(gutenberg.raw(f) for f in gutenberg.fileids[:5])
 if len(text) > 500_000:
 return text
 except LookupError:
 pass
 except ImportError:
 pass
 return _FALLBACK_FILLER * 1000 # ~1.4M chars, enough for any S


def _tokenize_len(tokenizer, text: str) -> int:
 return len(tokenizer.encode(text, add_special_tokens=False))


def _build_one(
 tokenizer,
 target_len: int,
 depth: float,
 needle_value: int,
 rng: random.Random,
 filler: str,
) -> dict:
 """Build a single needle prompt of approximately `target_len` tokens.

 The needle is inserted at character position `depth * len(prefix)`. We
 binary-search the filler character count to hit the target token length.
 """
 needle_text = (
 f"\n\nThe magic number for this document is {needle_value}. "
 f"Remember it.\n\n"
 )

 instruction = (
 "You are given a long passage. At some point inside the passage there is a "
 "sentence stating the 'magic number'. After reading the passage, output ONLY "
 "the magic number as a single integer, with no other text.\n\n"
 )
 question = (
 "\n\nWhat is the magic number? Output the integer only.\n"
 )

 # Binary search on filler-char count to hit target_len ± 200 tokens.
 lo, hi = 1000, len(filler)
 best = None
 for _ in range(20):
 mid = (lo + hi) // 2
 passage = filler[:mid]
 cut = int(len(passage) * depth)
 # Snap to nearest space to avoid mid-word.
 while cut < len(passage) and passage[cut] != " ":
 cut += 1
 prompt = (
 instruction + passage[:cut] + needle_text + passage[cut:] + question
 )
 n_tok = _tokenize_len(tokenizer, prompt)
 if abs(n_tok - target_len) <= 200:
 best = (prompt, n_tok)
 break
 if n_tok < target_len:
 lo = mid + 1
 else:
 hi = mid - 1
 if best is None:
 # Use whichever side of the search we ended on.
 best = (prompt, n_tok)
 return {
 "prompt": best[0],
 "needle": needle_text.strip,
 "answer": str(needle_value),
 "depth": depth,
 "actual_seq_len": best[1],
 }


def main:
 p = argparse.ArgumentParser
 p.add_argument("--max-seq-len", type=int, required=True,
 help="Target prompt length in tokens.")
 p.add_argument("--n-samples", type=int, default=50,
 help="Number of prompts to generate (split across depths).")
 p.add_argument("--depths", type=str, default="0.05,0.25,0.5,0.75,0.95",
 help="Comma-separated depth fractions for needle placement.")
 p.add_argument("--tokenizer", type=str, required=True,
 help="HF tokenizer name or path.")
 p.add_argument("--seed", type=int, default=2026)
 p.add_argument("--out", type=str, required=True,
 help="Output JSONL path.")
 args = p.parse_args

 try:
 from transformers import AutoTokenizer
 except ImportError:
 print("transformers not installed; install with: pip install transformers",
 file=sys.stderr)
 sys.exit(1)

 tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
 filler = _load_filler_text
 print(f"Loaded {len(filler):,} chars of filler text", file=sys.stderr)

 depths = [float(d) for d in args.depths.split(",")]
 rng = random.Random(args.seed)
 samples_per_depth = max(1, args.n_samples // len(depths))

 os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
 n_written = 0
 with open(args.out, "w", encoding="utf-8") as f:
 for depth in depths:
 for i in range(samples_per_depth):
 needle_value = rng.randint(10_000, 99_999)
 rec = _build_one(
 tokenizer, args.max_seq_len, depth, needle_value, rng, filler,
 )
 rec["id"] = f"needle_{args.max_seq_len}_d{int(depth*100):02d}_{i}"
 rec["target_seq_len"] = args.max_seq_len
 f.write(json.dumps(rec, ensure_ascii=False) + "\n")
 n_written += 1
 print(f"Wrote {n_written} prompts to {args.out}", file=sys.stderr)


def score(samples_path: str, predictions_path: str) -> dict:
 """Score model predictions against needle-haystack ground truth.

 samples_path: JSONL produced by main.
 predictions_path: JSONL with keys {id, prediction}.
 Returns {"accuracy": float, "by_depth": {depth -> accuracy}, "n": int}.
 """
 truth = {rec["id"]: rec for rec in (json.loads(l) for l in open(samples_path))}
 preds = {rec["id"]: rec["prediction"]
 for rec in (json.loads(l) for l in open(predictions_path))}
 by_depth = {}
 correct = 0
 n = 0
 for sid, rec in truth.items:
 if sid not in preds:
 continue
 n += 1
 # Extract first integer from prediction.
 import re
 m = re.search(r"-?\d+", str(preds[sid]))
 ok = bool(m and m.group(0) == rec["answer"])
 correct += int(ok)
 d = rec["depth"]
 by_depth.setdefault(d, {"correct": 0, "n": 0})
 by_depth[d]["correct"] += int(ok)
 by_depth[d]["n"] += 1
 return {
 "accuracy": correct / n if n else 0.0,
 "by_depth": {d: v["correct"]/v["n"] for d, v in by_depth.items},
 "n": n,
 }


if __name__ == "__main__":
 main
