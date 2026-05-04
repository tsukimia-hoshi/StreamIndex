"""Phase-2 placeholder. Decode-time TFLOPS / TB/s benchmark.

Decode is the regime where DeepSeek's biggest published wins live, and the one
SGLang's --nsa-decode-backend hooks land into. Targets:
 * Sparse decode at seq_len=1M (V4-Pro top_k=1024, V4-Flash top_k=512)
 * FlashMLA flash_mla_with_kvcache(is_fp8_kvcache=True, indices=...) baseline
"""
if __name__ == "__main__":
 raise SystemExit("This benchmark is a placeholder; not yet implemented.")
