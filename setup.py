"""StreamIndex build script.

Builds CUDA extensions when CUDA + a CUDA-enabled torch build are available.
Falls back to a metadata-only build (Triton-only, plus the pytorch reference)
when CUDA is not present, so the package is still installable for CI / docs.

The Triton kernels in flash_sparse/triton/ are JIT-compiled by Triton at
import time and do not require a build step here.
"""
from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup

_ROOT = Path(__file__).parent
_CUDA_DIR = _ROOT / "flash_sparse" / "cuda"


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available()) and bool(torch.version.cuda)
    except Exception:
        return False


def _build_ext_modules():
    if os.environ.get("STREAMINDEX_NO_CUDA") == "1":
        return []
    if not _cuda_available():
        return []
    cu_files = sorted(_CUDA_DIR.glob("*.cu"))
    if not cu_files:
        return []
    from torch.utils.cpp_extension import CUDAExtension

    return [
        CUDAExtension(
            name="flash_sparse._cuda",
            sources=[str(p) for p in cu_files],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--use_fast_math",
                    "--threads=4",
                    "-gencode=arch=compute_90a,code=sm_90a",   # H100/H200 (Hopper)
                    "-gencode=arch=compute_100a,code=sm_100a", # B200 (Blackwell)
                ],
            },
        )
    ]


def _cmdclass():
    if not _build_ext_modules():
        return {}
    from torch.utils.cpp_extension import BuildExtension

    return {"build_ext": BuildExtension}


setup(
    ext_modules=_build_ext_modules(),
    cmdclass=_cmdclass(),
)
