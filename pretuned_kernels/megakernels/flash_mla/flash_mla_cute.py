"""Nightly B200 comparison of DeepSeek-V3 CuTe MLA, shipped Triton and FlashInfer."""

from __future__ import annotations

from pretuned_kernels.megakernels.flash_mla import flash_mla as mla
from pretuned_kernels.megakernels.flash_mla.flash_mla import (
    use_cudagraph as use_cudagraph,
)


def main(verbose: bool = True) -> dict:
    return mla.main(verbose=verbose, include_cute=True)


if __name__ == "__main__":
    main()
