#!/usr/bin/env python3
"""Compare existing pretuned-kernel Triton codegen byte-for-byte.

The comparison reuses ``test/test_pretuned_kernels.py`` plus focused B200
codegen cases for the registered Triton and CuTe kernels. Newly introduced
megakernels are excluded because they do not exist in the baseline checkout.
``grouped_gemm_deepgemm`` is also excluded because it currently fails to bind
on both ``origin/main`` and the candidate before code generation begins.
"""

from __future__ import annotations

from pathlib import Path
import sys

from compare_example_codegen import compare_main

_CORRECTNESS_CLASS = "test/test_pretuned_kernels.py::TestPretunedKernelsCorrectness"
_CODEGEN_CASES = str(Path(__file__).with_name("pretuned_codegen_cases.py"))
_MEGAKERNEL_TESTS = (
    f"{_CORRECTNESS_CLASS}::test_qwen3_decode_layer",
    f"{_CORRECTNESS_CLASS}::test_gemma4_a4b_moe",
)
_EXPECTED_KERNELS = (
    "cross_entropy",
    "dynamic_per_token_scaled_fp8_quant",
    "fused_qk_norm_rope",
    "layer_norm",
    "per_token_group_fp8_quant",
    "rms_norm",
    "rms_norm_dynamic_per_token_quant",
    "rms_norm_per_block_quant",
    "rope_bwd",
    "rope_fwd",
    "scaled_mm",
    "silu_and_mul_per_block_quant",
    "silu_mul_fp8",
    "softmax",
    "vector_add",
    "nvfp4_gemv_bf16in_kernel",
    "nvfp4_gemv_fp4in_kernel",
    "nvfp4_gemv_bf16in_rows2_kernel",
    "nvfp4_gemv_bf16in_rows4_kernel",
    "nvfp4_gemv_fp4in_rows2_kernel",
    "nvfp4_gemv_fp4in_rows4_kernel",
    "scale_mm_cute",
    "scale_mm_cute_skinny_m",
    "scale_mm_cute_swap_ab",
    "projection_rotary",
    "interleaved_swiglu",
    "grouped_gemm",
)


def main() -> int:
    return compare_main(
        sys.argv[1:],
        test_target=(_CORRECTNESS_CLASS, _CODEGEN_CASES),
        suite_name="pretuned-kernel",
        default_pytest_args=tuple(
            arg for nodeid in _MEGAKERNEL_TESTS for arg in ("--deselect", nodeid)
        ),
        description=__doc__,
        require_same_collection=False,
        capture_args=("--require-aot-heuristic",),
        required_kernels=_EXPECTED_KERNELS,
    )


if __name__ == "__main__":
    raise SystemExit(main())
