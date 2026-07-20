"""
Hardware marker for kernel: nvfp4_gemv_cute (CuTe / tcgen05 backend)

The CuTe NVFP4 GEMV fast paths are hand-written ``@cute.kernel`` inline-PTX
kernels launched directly with a fixed block/grid (``default_cute_launcher``),
not Helion DSL kernels with autotuned configs -- so there is nothing to tune and
no heuristic table. This file exists only so ``pretuned_kernels/run.py`` gates
the kernel to the hardware it targets: its ``_cuda_sm100`` suffix marks NVIDIA
B200 (sm100), the Blackwell target the CuTe fast paths require (they need compute
capability >= 10.0 and k_bytes % 2048 == 0; see ``_can_use_fast_cute_path`` in
examples/nvfp4_gemv.py). run.py only reads this file's *name*, never its
contents, for a direct-launch kernel like this one.
"""
