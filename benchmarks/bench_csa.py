"""Phase-2 placeholder. Wall-clock + TFLOPS benchmark for CSA forward.

Will compare flash_csa_forward against:
* pytorch reference (this package, slow ground truth)
* DeepSeek FlashMLA sparse_prefill_fwd (production CUDA baseline)
* tilelang sparse_mla_fwd (TileLang reference baseline)
"""

if __name__ == "__main__":
    raise SystemExit("This benchmark is a placeholder; not yet implemented.")
