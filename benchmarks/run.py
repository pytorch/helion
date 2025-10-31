# pyright: reportMissingImports=false

"""Performance comparison between Helion, torch.compile, Triton, and PyTorch eager by leveraging TritonBench.

Currently supported kernels are listed in `KERNEL_MAPPINGS` in `benchmarks/run.py`.

Usage:
$ python benchmarks/run.py [tritonbench args...] [--kernel <kernel_name(s)>]

Example usage:
$ python benchmarks/run.py --metrics speedup,accuracy --kernel vector_add  # Runs vector_add kernel
$ python benchmarks/run.py --metrics speedup,accuracy --kernel vector_add,rms_norm  # Runs multiple kernels
$ python benchmarks/run.py --metrics speedup,accuracy  # Runs all kernels

# On GPU-1, run first 1/4 of inputs for all kernels and save results to CSV in the current directory
$ CUDA_VISIBLE_DEVICES=1 python benchmarks/run.py --input-shard 1/4 --metrics accuracy,tflops,gbps,speedup --csv --output-dir ./

# Equally-spaced-k mode: Select 5 equally spaced inputs from all available inputs
$ python benchmarks/run.py --metrics speedup,accuracy --kernel softmax --input-sample-mode equally-spaced-k --num-inputs 5
"""

from __future__ import annotations

import argparse
import collections
from contextlib import suppress
import dataclasses
import functools
import gc
import importlib.util
import json
import logging
import os
from pathlib import Path
from pprint import pformat
import shutil
import subprocess
import sys
import tempfile
from typing import Any
from typing import Callable
from typing import cast

import torch
from torch.utils._pytree import tree_leaves
from torch.utils._pytree import tree_map

from helion._testing import get_nvidia_gpu_model
from helion._utils import counters

logger: logging.Logger = logging.getLogger(__name__)

StrPath = str | os.PathLike[str]

if os.getenv("HELION_BENCHMARK_DISABLE_LOGGING", "0") == "1":
    logging.disable(logging.CRITICAL)


def is_cuda() -> bool:
    return torch.version.cuda is not None


IS_B200 = is_cuda() and get_nvidia_gpu_model() == "NVIDIA B200"


def log_tensor_metadata(args: tuple[object, ...], kwargs: dict[str, object]) -> None:
    structure = (args, kwargs)
    if not any(isinstance(leaf, torch.Tensor) for leaf in tree_leaves(structure)):
        return

    def describe_tensor(obj: object) -> object:
        if isinstance(obj, torch.Tensor):
            return {
                "shape": tuple(obj.shape),
                "stride": tuple(obj.stride()),
                "dtype": str(obj.dtype),
                "device": str(obj.device),
            }
        return obj

    described_args, described_kwargs = tree_map(describe_tensor, structure)

    logger.warning(
        "Input tensor metadata:\n%s",
        pformat({"args": described_args, "kwargs": described_kwargs}, indent=2),
    )


# Maximum number of inputs to use
MAX_NUM_INPUTS = 20


@dataclasses.dataclass
class RunResult:
    model: str
    device: str
    shape: list[str]
    metrics: dict[str, list[float]]


# Maps tritonbench op names to Helion kernel examples
# Can map to a single kernel or a list of kernel variants
# Format options:
#   - Single kernel: (tritonbench_module, helion_module, helion_func)
#   - Single kernel with args: (tritonbench_module, helion_module, helion_func, args_dict)
#   - Multiple kernels: (tritonbench_module, [(helion_module, helion_func), ...])
#   - Multiple kernels with args: (tritonbench_module, [(helion_module, helion_func), ...], args_dict)
KERNEL_MAPPINGS: dict[str, tuple[str, ...]] = {  # pyright: ignore[reportAssignmentType]
    # <tritonbench_op_name>: (<tritonbench_module_path>, <helion_kernel_module_path>, <helion_kernel_function_name>)
    "vector_add": ("tritonbench.operators.vector_add.operator", "examples.add", "add"),
    "addmm": (
        "tritonbench.operators.addmm.operator",
        "examples.matmul",
        "addmm_tritonbench",
    ),
    "addmm-bwd": (
        "tritonbench.operators.addmm.operator",
        "examples.matmul",
        "addmm_tritonbench",
    ),
    "geglu": (
        "tritonbench.operators.geglu.operator",
        "examples.geglu",
        "geglu_tritonbench",
    ),
    "swiglu": (
        "tritonbench.operators.swiglu.operator",
        "examples.swiglu",
        "swiglu_tritonbench",
    ),
    "swiglu-bwd": (
        "tritonbench.operators.swiglu.operator",
        "examples.swiglu",
        "swiglu_tritonbench",
    ),
    "jsd": (
        "tritonbench.operators.jsd.operator",
        "examples.jsd",
        "jsd_tritonbench",
    ),
    "kl_div": (
        "tritonbench.operators.kl_div.operator",
        "examples.kl_div",
        "kl_div_tritonbench",
    ),
    "ragged_attention": (
        "tritonbench.operators.ragged_attention.operator",
        "examples.jagged_hstu_attn",
        "ragged_attention_tritonbench",
        {"target_size": 0},
    ),
    "embedding": (
        "tritonbench.operators.embedding.operator",
        "examples.embedding",
        "embedding_tritonbench",
    ),
    "vector_exp": (
        "tritonbench.operators.vector_exp.operator",
        "examples.exp",
        "exp_tritonbench",
    ),
    "vector_exp-bwd": (
        "tritonbench.operators.vector_exp.operator",
        "examples.exp",
        "exp_tritonbench",
    ),
    "rms_norm": (
        "tritonbench.operators.rms_norm.operator",
        "examples.rms_norm",
        "rms_norm_tritonbench",
    ),
    "rms_norm-bwd": (
        "tritonbench.operators.rms_norm.operator",
        "examples.rms_norm",
        "rms_norm_tritonbench",
        {
            "num_inputs": 5,  # rms_norm-bwd has 6 inputs total but last input raises Triton OOM at default config: https://github.com/pytorch/helion/issues/711
        },
    ),
    "sum": ("tritonbench.operators.sum.operator", "examples.sum", "sum_tritonbench"),
    "softmax": (
        "tritonbench.operators.softmax.operator",
        "examples.softmax",
        "softmax_tritonbench",
    ),
    "softmax-bwd": (
        "tritonbench.operators.softmax.operator",
        "examples.softmax",
        "softmax_tritonbench",
    ),
    "jagged_mean": (
        "tritonbench.operators.jagged_mean.operator",
        "examples.jagged_mean",
        "jagged_mean_tritonbench",
        {},
    ),
    "fp8_gemm": (
        "tritonbench.operators.fp8_gemm.fp8_gemm",
        "examples.fp8_gemm",
        "fp8_gemm_tritonbench",
        {
            "num_inputs": 10,  # fp8_gemm takes long time on Benchmark CI, so use fewer inputs instead.
        },
    ),
    "flash_attention": (
        "tritonbench.operators.flash_attention.operator",
        "examples.attention",
        "attention",
        {
            "d_head": 128,  # Set default head dimension to 128 for TLX attention compatibility
            "num_inputs": 6,  # flash_attention takes long time on Benchmark CI, so use fewer inputs instead.
        },
    ),
    "cross_entropy": (
        "tritonbench.operators.cross_entropy.operator",
        "examples.cross_entropy",
        "cross_entropy",
        {},
    ),
    "fp8_attention": (
        "tritonbench.operators.fp8_attention.operator",
        "examples.fp8_attention",
        "fp8_attention_tritonbench",
        {
            "num_inputs": 10,  # fp8_attention takes long time on Benchmark CI, so use fewer inputs instead.
        },
    ),
    "layer_norm": (
        "tritonbench.operators.layer_norm.operator",
        "examples.layer_norm",
        "layer_norm_tritonbench",
    ),
    "layer_norm-bwd": (
        "tritonbench.operators.layer_norm.operator",
        "examples.layer_norm",
        "layer_norm_tritonbench",
        {
            "num_inputs": 10,  # layer_norm-bwd takes long time on Benchmark CI, so use fewer inputs instead.
        },
    ),
    "jagged_softmax": (
        "tritonbench.operators.jagged_softmax.operator",
        "examples.jagged_softmax",
        "jagged_softmax_tritonbench",
    ),
    "grouped_gemm": (
        "tritonbench.operators.grouped_gemm.operator",
        "examples.grouped_gemm",
        "grouped_gemm_jagged_persistent_tritonbench",
        {
            "num_inputs": 6,  # grouped_gemm takes long time on Benchmark CI, so use fewer inputs instead.
        },
    ),
    "fused_linear_jsd": (
        "tritonbench.operators.fused_linear_jsd.operator",
        "examples.fused_linear_jsd",
        "fused_linear_jsd_fwd_tritonbench",
    ),
    # Multiple kernel variants:
    "gemm": (
        "tritonbench.operators.gemm.operator",
        "examples.matmul",
        "matmul_tritonbench",
        {
            "num_inputs": 8,  # gemm takes long time on Benchmark CI, so use fewer inputs instead.
            "non_square": "",  # use --non-square shapes
            "rep": "3000",  # gemm b200 can have noisy results from throttling
        },
    ),
    "gemm-bwd": (
        "tritonbench.operators.gemm.operator",
        "examples.matmul",
        "matmul_tritonbench",
        {
            "num_inputs": 10,  # gemm-bwd takes long time on Benchmark CI, so use fewer inputs instead.
        },
    ),
    "welford": (
        "tritonbench.operators.welford.operator",
        "examples.welford",
        "welford",
        {
            "num_inputs": 6,  # welford takes long time on Benchmark CI, so use fewer inputs instead.
        },
    ),
    "gather_gemv": (
        "tritonbench.operators.gather_gemv.operator",
        "examples.gather_gemv",
        "gather_gemv_tritonbench",
    ),
    "int4_gemm": (
        "tritonbench.operators.int4_gemm.int4_gemm",
        "examples.int4_gemm",
        "int4_gemm_tritonbench",
        {
            "num_inputs": 10,  # int4_gemm takes long time on Benchmark CI, so use fewer inputs instead.
        },
    ),
    "jagged_layer_norm": (
        "tritonbench.operators.jagged_layer_norm.operator",
        "examples.jagged_layer_norm",
        "jagged_layer_norm_tritonbench",
    ),
    "jagged_sum": (
        "tritonbench.operators.jagged_sum.operator",
        "examples.jagged_sum",
        "jagged_sum_tritonbench",
    ),
    "low_mem_dropout": (
        "tritonbench.operators.low_mem_dropout.operator",
        "examples.low_mem_dropout",
        "low_mem_dropout_tritonbench",
    ),
    "bf16xint16_gemm": (
        "tritonbench.operators.bf16xint16_gemm.bf16xint16_gemm",
        "examples.bf16xint16_gemm",
        "bf16xint16_gemm_tritonbench",
    ),
    "blackwell_attentions": (
        "tritonbench.operators.blackwell_attentions.operator",
        "examples.blackwell_attention",
        "blackwell_attention_tritonbench",
        {
            "d_head": 128,  # Set default head dimension to 128 for TLX attention compatibility
            "num_inputs": 6,  # flash_attention takes long time on Benchmark CI, so use fewer inputs instead.
            "input_id": 1,
        },
    ),
    "mamba2_chunk_scan": (
        "tritonbench.operators.mamba2_chunk_scan.operator",
        "examples.mamba2_chunk_scan",
        "helion_mamba2_chunk_scan_kernel",
    ),
    "mamba2_chunk_state": (
        "tritonbench.operators.mamba2_chunk_state.operator",
        "examples.mamba2_chunk_state",
        "helion_mamba2_chunk_state_kernel",
    ),
}


KERNEL_METRIC_MAPPINGS: dict[str, dict[str, str]] = {
    "vector_add": {
        "torch_add": "baseline",
        "triton_add-speedup": "triton_speedup",
        "triton_add-accuracy": "triton_accuracy",
        "torch_compile_add-speedup": "torch_compile_speedup",
        "torch_compile_add-accuracy": "torch_compile_accuracy",
        "helion_add-speedup": "helion_speedup",
        "helion_add-accuracy": "helion_accuracy",
    },
    "vector_exp": {
        "torch_exp": "baseline",
        "triton_exp-speedup": "triton_speedup",
        "triton_exp-accuracy": "triton_accuracy",
        "torch_compile_exp-speedup": "torch_compile_speedup",
        "torch_compile_exp-accuracy": "torch_compile_accuracy",
        "helion_exp_tritonbench-speedup": "helion_speedup",
        "helion_exp_tritonbench-accuracy": "helion_accuracy",
    },
    "sum": {
        "torch_sum": "baseline",
        "triton_sum-speedup": "triton_speedup",
        "triton_sum-accuracy": "triton_accuracy",
        "torch_compile_sum-speedup": "torch_compile_speedup",
        "torch_compile_sum-accuracy": "torch_compile_accuracy",
        "helion_sum_tritonbench-speedup": "helion_speedup",
        "helion_sum_tritonbench-accuracy": "helion_accuracy",
    },
    "layer_norm": {
        "torch_layer_norm": "baseline",
        "liger_layer_norm-speedup": "triton_speedup",
        "liger_layer_norm-accuracy": "triton_accuracy",
        "torch_compile_layer_norm-speedup": "torch_compile_speedup",
        "torch_compile_layer_norm-accuracy": "torch_compile_accuracy",
        "helion_layer_norm_tritonbench-speedup": "helion_speedup",
        "helion_layer_norm_tritonbench-accuracy": "helion_accuracy",
    },
    "layer_norm-bwd": {
        "torch_layer_norm": "baseline",
        "liger_layer_norm-speedup": "triton_speedup",
        "liger_layer_norm-accuracy": "triton_accuracy",
        "torch_compile_layer_norm-speedup": "torch_compile_speedup",
        "torch_compile_layer_norm-accuracy": "torch_compile_accuracy",
        "helion_layer_norm_tritonbench-speedup": "helion_speedup",
        "helion_layer_norm_tritonbench-accuracy": "helion_accuracy",
    },
    "softmax": {
        "naive_softmax": "baseline",
        "triton_softmax-speedup": "triton_speedup",
        "triton_softmax-accuracy": "triton_accuracy",
        "torch_compile_softmax-speedup": "torch_compile_speedup",
        "torch_compile_softmax-accuracy": "torch_compile_accuracy",
        "helion_softmax_tritonbench-speedup": "helion_speedup",
        "helion_softmax_tritonbench-accuracy": "helion_accuracy",
    },
    "softmax-bwd": {
        "naive_softmax": "baseline",
        "triton_softmax-speedup": "triton_speedup",
        "triton_softmax-accuracy": "triton_accuracy",
        "torch_compile_softmax-speedup": "torch_compile_speedup",
        "torch_compile_softmax-accuracy": "torch_compile_accuracy",
        "helion_softmax_tritonbench-speedup": "helion_speedup",
        "helion_softmax_tritonbench-accuracy": "helion_accuracy",
    },
    "rms_norm": {
        "llama_rms": "baseline",
        "liger_rms-speedup": "triton_speedup",
        "liger_rms-accuracy": "triton_accuracy",
        "torch_compile_rms-speedup": "torch_compile_speedup",
        "torch_compile_rms-accuracy": "torch_compile_accuracy",
        "helion_rms_norm_tritonbench-speedup": "helion_speedup",
        "helion_rms_norm_tritonbench-accuracy": "helion_accuracy",
    },
    "rms_norm-bwd": {
        "llama_rms": "baseline",
        "liger_rms-speedup": "triton_speedup",
        "liger_rms-accuracy": "triton_accuracy",
        "torch_compile_rms-speedup": "torch_compile_speedup",
        "torch_compile_rms-accuracy": "torch_compile_accuracy",
        "helion_rms_norm_tritonbench-speedup": "helion_speedup",
        "helion_rms_norm_tritonbench-accuracy": "helion_accuracy",
    },
    "cross_entropy": {
        "cross_entropy_loss": "baseline",
        "liger_cross_entropy_loss-speedup": "triton_speedup",
        "liger_cross_entropy_loss-accuracy": "triton_accuracy",
        "torch_compile_cross_entropy_loss-speedup": "torch_compile_speedup",
        "torch_compile_cross_entropy_loss-accuracy": "torch_compile_accuracy",
        "helion_cross_entropy-speedup": "helion_speedup",
        "helion_cross_entropy-accuracy": "helion_accuracy",
    },
    "geglu": {
        "torch_geglu": "baseline",
        "liger_geglu-speedup": "triton_speedup",
        "liger_geglu-accuracy": "triton_accuracy",
        "torch_compile_geglu-speedup": "torch_compile_speedup",
        "torch_compile_geglu-accuracy": "torch_compile_accuracy",
        "helion_geglu_tritonbench-speedup": "helion_speedup",
        "helion_geglu_tritonbench-accuracy": "helion_accuracy",
    },
    "swiglu": {
        "torch_swiglu": "baseline",
        "liger_swiglu-speedup": "triton_speedup",
        "liger_swiglu-accuracy": "triton_accuracy",
        "torch_compile_swiglu-speedup": "torch_compile_speedup",
        "torch_compile_swiglu-accuracy": "torch_compile_accuracy",
        "helion_swiglu_tritonbench-speedup": "helion_speedup",
        "helion_swiglu_tritonbench-accuracy": "helion_accuracy",
    },
    "swiglu-bwd": {
        "torch_swiglu": "baseline",
        "liger_swiglu-speedup": "triton_speedup",
        "liger_swiglu-accuracy": "triton_accuracy",
        "torch_compile_swiglu-speedup": "torch_compile_speedup",
        "torch_compile_swiglu-accuracy": "torch_compile_accuracy",
        "helion_swiglu_tritonbench-speedup": "helion_speedup",
        "helion_swiglu_tritonbench-accuracy": "helion_accuracy",
    },
    "jsd": {
        "torch_jsd": "baseline",
        "liger_jsd-speedup": "triton_speedup",
        "liger_jsd-accuracy": "triton_accuracy",
        "torch_compile_jsd-speedup": "torch_compile_speedup",
        "torch_compile_jsd-accuracy": "torch_compile_accuracy",
        "helion_jsd_tritonbench-speedup": "helion_speedup",
        "helion_jsd_tritonbench-accuracy": "helion_accuracy",
    },
    "welford": {
        "eager_layer_norm": "baseline",
        "triton_welford-speedup": "triton_speedup",
        "triton_welford-accuracy": "triton_accuracy",
        "torch_compile_welford-speedup": "torch_compile_speedup",
        "torch_compile_welford-accuracy": "torch_compile_accuracy",
        "helion_welford-speedup": "helion_speedup",
        "helion_welford-accuracy": "helion_accuracy",
    },
    "kl_div": {
        "torch_kl_div": "baseline",
        "liger_kl_div-speedup": "triton_speedup",
        "liger_kl_div-accuracy": "triton_accuracy",
        "torch_compile_kl_div-speedup": "torch_compile_speedup",
        "torch_compile_kl_div-accuracy": "torch_compile_accuracy",
        "helion_kl_div_tritonbench-speedup": "helion_speedup",
        "helion_kl_div_tritonbench-accuracy": "helion_accuracy",
    },
    "gather_gemv": {
        "eager_gather_gemv": "baseline",
        "triton_gather_gemv-speedup": "triton_speedup",
        "triton_gather_gemv-accuracy": "triton_accuracy",
        "torch_compile_gather_gemv-speedup": "torch_compile_speedup",
        "torch_compile_gather_gemv-accuracy": "torch_compile_accuracy",
        "helion_gather_gemv_tritonbench-speedup": "helion_speedup",
        "helion_gather_gemv_tritonbench-accuracy": "helion_accuracy",
    },
    "int4_gemm": {
        "preprocessed_eager_int4_gemm": "baseline",
        "preprocessed_triton_int4_gemm-speedup": "triton_speedup",
        "preprocessed_triton_int4_gemm-accuracy": "triton_accuracy",
        "preprocessed_torch_compile_int4_gemm-speedup": "torch_compile_speedup",
        "preprocessed_torch_compile_int4_gemm-accuracy": "torch_compile_accuracy",
        "helion_int4_gemm_tritonbench-speedup": "helion_speedup",
        "helion_int4_gemm_tritonbench-accuracy": "helion_accuracy",
    },
    "grouped_gemm": {
        "aten_grouped_mm": "baseline",
        "triton_grouped_gemm-speedup": "triton_speedup",
        "triton_grouped_gemm-accuracy": "triton_accuracy",
        "torch_compile_grouped_gemm-speedup": "torch_compile_speedup",
        "torch_compile_grouped_gemm-accuracy": "torch_compile_accuracy",
        "helion_grouped_gemm_jagged_persistent_tritonbench-speedup": "helion_speedup",
        "helion_grouped_gemm_jagged_persistent_tritonbench-accuracy": "helion_accuracy",
    },
    "jagged_layer_norm": {
        "torch_compile_nested_tensor_integration-speedup": "torch_compile_speedup",
        "torch_compile_nested_tensor_integration-accuracy": "torch_compile_accuracy",
        "helion_jagged_layer_norm_tritonbench-speedup": "helion_speedup",
        "helion_jagged_layer_norm_tritonbench-accuracy": "helion_accuracy",
    },
    "jagged_sum": {
        "triton_jagged_sum_no_pad_simple_fused-speedup": "triton_speedup",
        "triton_jagged_sum_no_pad_simple_fused-accuracy": "triton_accuracy",
        "torch_compile_nested_tensor_integration-speedup": "torch_compile_speedup",
        "torch_compile_nested_tensor_integration-accuracy": "torch_compile_accuracy",
        "helion_jagged_sum_tritonbench-speedup": "helion_speedup",
        "helion_jagged_sum_tritonbench-accuracy": "helion_accuracy",
    },
    "addmm": {
        "aten_addmm": "baseline",
        "triton_addmm-speedup": "triton_speedup",
        "triton_addmm-accuracy": "triton_accuracy",
        "pt2_addmm_maxautotune-speedup": "torch_compile_speedup",
        "pt2_addmm_maxautotune-accuracy": "torch_compile_accuracy",
        "helion_addmm_tritonbench-speedup": "helion_speedup",
        "helion_addmm_tritonbench-accuracy": "helion_accuracy",
    },
    # "ragged_attention": {
    #     "triton_ragged_attention-speedup": "triton_speedup",
    #     "triton_ragged_attention-accuracy": "triton_accuracy",
    #     "torch_compile_ragged_attention-speedup": "torch_compile_speedup",
    #     "torch_compile_ragged_attention-accuracy": "torch_compile_accuracy",
    #     "helion_ragged_attention_tritonbench-speedup": "helion_speedup",
    #     "helion_ragged_attention_tritonbench-accuracy": "helion_accuracy",
    # },
    "embedding": {
        "torch_embedding": "baseline",
        "liger_embedding-speedup": "triton_speedup",
        "liger_embedding-accuracy": "triton_accuracy",
        "torch_compile_embedding-speedup": "torch_compile_speedup",
        "torch_compile_embedding-accuracy": "torch_compile_accuracy",
        "helion_embedding_tritonbench-speedup": "helion_speedup",
        "helion_embedding_tritonbench-accuracy": "helion_accuracy",
    },
    "jagged_mean": {
        "torch_jagged_mean_torch_sum": "baseline",
        "triton_jagged_mean_variable_length_loop-speedup": "triton_speedup",
        "triton_jagged_mean_variable_length_loop-accuracy": "triton_accuracy",
        "torch_compile_jagged_mean_torch_sum-speedup": "torch_compile_speedup",
        "torch_compile_jagged_mean_torch_sum-accuracy": "torch_compile_accuracy",
        "helion_jagged_mean_tritonbench-speedup": "helion_speedup",
        "helion_jagged_mean_tritonbench-accuracy": "helion_accuracy",
    },
    "flash_attention": {
        "aten": "baseline",
        "triton_tutorial_flash_v2_tma_ws_persistent-speedup": "triton_speedup",
        "triton_tutorial_flash_v2_tma_ws_persistent-accuracy": "triton_accuracy",
        "flex_attention-speedup": "torch_compile_speedup",
        "flex_attention-accuracy": "torch_compile_accuracy",
        "helion_attention-speedup": "helion_speedup",
        "helion_attention-accuracy": "helion_accuracy",
    },
    "fp8_attention": {
        "triton_flash_v2": "baseline",
        "triton_flash_v2_ws-speedup": "triton_speedup",
        "triton_flash_v2_ws-accuracy": "triton_accuracy",
        "helion_fp8_attention_tritonbench-speedup": "helion_speedup",
        "helion_fp8_attention_tritonbench-accuracy": "helion_accuracy",
    },
    "jagged_softmax": {
        "torch_jagged_softmax_torch_sum": "baseline",
        "triton_jagged_softmax_variable_length_loop-speedup": "triton_speedup",
        "triton_jagged_softmax_variable_length_loop-accuracy": "triton_accuracy",
        "torch_compile_jagged_softmax_torch_sum-speedup": "torch_compile_speedup",
        "torch_compile_jagged_softmax_torch_sum-accuracy": "torch_compile_accuracy",
        "helion_jagged_softmax_tritonbench-speedup": "helion_speedup",
        "helion_jagged_softmax_tritonbench-accuracy": "helion_accuracy",
    },
    "fused_linear_jsd": {
        "torch_lm_head_jsd": "baseline",
        "triton_fused_linear_jsd-speedup": "triton_speedup",
        "triton_fused_linear_jsd-accuracy": "triton_accuracy",
        "torch_compile_fused_linear_jsd-speedup": "torch_compile_speedup",
        "torch_compile_fused_linear_jsd-accuracy": "torch_compile_accuracy",
        "helion_fused_linear_jsd_fwd_tritonbench-speedup": "helion_speedup",
        "helion_fused_linear_jsd_fwd_tritonbench-accuracy": "helion_accuracy",
    },
    "gemm": {
        "aten_matmul": "baseline",
        "triton_tutorial_matmul-speedup": "triton_speedup",
        "triton_tutorial_matmul-accuracy": "triton_accuracy",
        "pt2_triton_matmul-speedup": "torch_compile_speedup",
        "pt2_triton_matmul-accuracy": "torch_compile_accuracy",
        "helion_matmul_tritonbench-speedup": "helion_speedup",
        "helion_matmul_tritonbench-accuracy": "helion_accuracy",
    },
    "fp8_gemm": {
        "torch_fp8_gemm": "baseline",
        f"{'blackwell_persistent_tma' if IS_B200 else 'triton_tma_persistent'}_fp8_gemm-speedup": "triton_speedup",
        f"{'blackwell_persistent_tma' if IS_B200 else 'triton_tma_persistent'}_fp8_gemm-accuracy": "triton_accuracy",
        f"{'blackwell_pt2' if IS_B200 else 'pt2'}_fp8_gemm-speedup": "torch_compile_speedup",
        f"{'blackwell_pt2' if IS_B200 else 'pt2'}_fp8_gemm-accuracy": "torch_compile_accuracy",
        "helion_fp8_gemm_tritonbench-speedup": "helion_speedup",
        "helion_fp8_gemm_tritonbench-accuracy": "helion_accuracy",
    },
    "low_mem_dropout": {
        "seeded_dropout-accuracy": "triton_accuracy",
        "seeded_dropout-speedup": "triton_speedup",
        "torch_compile_dropout-accuracy": "torch_compile_accuracy",
        "torch_compile_dropout-speedup": "torch_compile_speedup",
        "helion_low_mem_dropout_tritonbench-accuracy": "helion_accuracy",
        "helion_low_mem_dropout_tritonbench-speedup": "helion_speedup",
    },
    "bf16xint16_gemm": {
        "bf16xbf16": "baseline",
        "bf16xint16-speedup": "triton_speedup",
        "bf16xint16-accuracy": "triton_accuracy",
        "torch_compile_bf16xbf16-speedup": "torch_compile_speedup",
        "torch_compile_bf16xbf16-accuracy": "torch_compile_accuracy",
        "helion_bf16xint16_gemm_tritonbench-speedup": "helion_speedup",
        "helion_bf16xint16_gemm_tritonbench-accuracy": "helion_accuracy",
    },
    "blackwell_attentions": {
        "aten": "baseline",
        "triton_tutorial_flash_v2_tma_ws_persistent-speedup": "triton_speedup",
        "triton_tutorial_flash_v2_tma_ws_persistent-accuracy": "triton_accuracy",
        "flex_attention-speedup": "torch_compile_speedup",
        "flex_attention-accuracy": "torch_compile_accuracy",
        "helion_blackwell_attention_tritonbench-speedup": "helion_speedup",
        "helion_blackwell_attention_tritonbench-accuracy": "helion_accuracy",
    },
    "mamba2_chunk_scan": {
        "eager": "baseline",
        "compile_speedup": "torch_compile_speedup",
        "compile_accuracy": "torch_compile_accuracy",
        "helion_mamba2_chunk_scan_kernel_speedup": "helion_speedup",
        "helion_mamba2_chunk_scan_kernel_accuracy": "helion_accuracy",
    },
    "mamba2_chunk_state": {
        "eager": "baseline",
        "compile_speedup": "torch_compile_speedup",
        "compile_accuracy": "torch_compile_accuracy",
        "helion_mamba2_chunk_state_kernel_speedup": "helion_speedup",
        "helion_mamba2_chunk_state_kernel_accuracy": "helion_accuracy",
    },
}


def check_and_setup_tritonbench() -> None:
    """Ensure a usable tritonbench installation is available."""

    benchmarks_dir = Path(__file__).parent
    tritonbench_path = benchmarks_dir / "tritonbench"
    installing_marker = (benchmarks_dir / ".tritonbench_installing").resolve()

    try:
        import tritonbench  # pyright: ignore[reportMissingImports]

        module_file = getattr(tritonbench, "__file__", None)
        tb_repo_path = tritonbench_path.resolve()

        candidate_paths: list[Path] = []

        def add_candidate_path(entry: object) -> None:
            if not isinstance(entry, (str, os.PathLike)):
                return
            path_entry = cast("StrPath", entry)
            with suppress(TypeError, OSError, RuntimeError):
                candidate_paths.append(Path(path_entry))

        if module_file is not None:
            add_candidate_path(module_file)

        module_paths = getattr(tritonbench, "__path__", None)
        if module_paths is not None:
            for entry in module_paths:
                add_candidate_path(entry)

        def is_local(path: Path) -> bool:
            try:
                resolved_path = path.resolve()
            except (OSError, RuntimeError):
                return False
            return (
                resolved_path == tb_repo_path or tb_repo_path in resolved_path.parents
            )

        has_local_checkout = any(is_local(path) for path in candidate_paths)

        if candidate_paths and not has_local_checkout:
            # If tritonbench is not from local checkout, assume it's a proper installation
            return

        if has_local_checkout:
            if installing_marker.exists():
                print(
                    "Detected partially installed tritonbench; reinstalling local checkout.",
                    file=sys.stderr,
                )
            else:
                return
        else:
            print(
                "Unable to determine tritonbench import path; reinstalling local checkout.",
                file=sys.stderr,
            )

    except ImportError:
        pass

    print(
        "Installing tritonbench from source...",
        file=sys.stderr,
    )
    print(f"Using tritonbench path: {tritonbench_path}")

    if tritonbench_path.exists():
        print("Removing existing tritonbench checkout...", file=sys.stderr)
        if tritonbench_path.is_dir():
            shutil.rmtree(tritonbench_path)
        else:
            tritonbench_path.unlink()

    sys.modules.pop("tritonbench", None)

    installing_marker.touch()

    try:
        print("Cloning tritonbench repository...", file=sys.stderr)
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/meta-pytorch/tritonbench.git",
                str(tritonbench_path),
            ],
            cwd=benchmarks_dir,
            check=True,
        )

        print("Initializing tritonbench submodules...", file=sys.stderr)
        subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive"],
            cwd=tritonbench_path,
            check=True,
        )

        print("Installing tritonbench requirements...", file=sys.stderr)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                "requirements.txt",
            ],
            cwd=tritonbench_path,
            check=True,
        )

        print("Running install.py --liger...", file=sys.stderr)
        subprocess.run(
            [sys.executable, "install.py", "--liger"],
            cwd=tritonbench_path,
            check=True,
        )

        print("Installing tritonbench package...", file=sys.stderr)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            cwd=tritonbench_path,
            check=True,
        )

        importlib.invalidate_caches()

        try:
            import tritonbench  # pyright: ignore[reportMissingImports]

            print("Tritonbench installed successfully.", file=sys.stderr)
            if installing_marker.exists():
                installing_marker.unlink()
        except ImportError:
            print(
                "Error: Tritonbench package installation failed. The package cannot be imported.",
                file=sys.stderr,
            )
            sys.exit(1)

    except subprocess.CalledProcessError as e:
        print(f"Error installing tritonbench: {e}", file=sys.stderr)
        if e.stdout:
            print(f"stdout: {e.stdout}", file=sys.stderr)
        if e.stderr:
            print(f"stderr: {e.stderr}", file=sys.stderr)
        sys.exit(1)


def run_kernel(
    kernel_name: str,
    tritonbench_args: list[str],
    input_shard_info: tuple[int, int] | None,
    results: list[RunResult],
) -> None:
    """Run a kernel benchmark, handling both single and multiple variants."""
    # Check if kernel is in the mapping table
    if kernel_name not in KERNEL_MAPPINGS:
        print(f"Error: Unknown kernel '{kernel_name}'", file=sys.stderr)
        print(
            f"Available kernels: {', '.join(KERNEL_MAPPINGS.keys())}", file=sys.stderr
        )
        sys.exit(1)

    mapping = KERNEL_MAPPINGS[kernel_name]

    # Extract operator args if present
    operator_args = {}

    # Normalize to list of variants format
    if isinstance(mapping[1], list):
        # Multiple variants format
        tritonbench_module = mapping[0]
        variants = mapping[1]
        # Check if last element is args dict
        if len(mapping) > 2 and isinstance(mapping[2], dict):
            operator_args = mapping[2]
    else:
        # Single kernel format
        if len(mapping) == 4 and isinstance(mapping[3], dict):
            # With args
            tritonbench_module = mapping[0]
            module_path = mapping[1]
            func_name = mapping[2]
            operator_args = mapping[3]  # pyright: ignore[reportGeneralTypeIssues]
            variants = [(module_path, func_name)]
        else:
            # Without args
            assert len(mapping) == 3  # Type narrowing for pyright
            tritonbench_module, module_path, func_name = mapping
            variants = [(module_path, func_name)]

    # Run all variants in the same benchmark
    run_kernel_variants(
        kernel_name,
        tritonbench_module,
        variants,
        tritonbench_args,
        input_shard_info,
        operator_args,
        results,
    )


def run_kernel_variants(
    kernel_name: str,
    tritonbench_module: str,
    variants: list[tuple[str, str]],
    tritonbench_args: list[str],
    input_shard_info: tuple[int, int] | None,
    operator_args: dict[str, Any] | None,
    results: list[RunResult],
) -> None:
    """Run kernel variants in the same benchmark run."""

    # Import tritonbench components
    from tritonbench.utils.parser import (  # pyright: ignore[reportMissingImports]
        get_parser,
    )
    from tritonbench.utils.triton_op import BenchmarkOperator
    from tritonbench.utils.triton_op import BenchmarkOperatorMetrics

    # Get the tritonbench operator name, stripping -bwd suffix for backward operators
    operator_name = kernel_name.removesuffix("-bwd")

    # Parse tritonbench arguments
    tb_parser = get_parser()

    assert "--op" not in tritonbench_args
    tritonbench_args = ["--op", operator_name, *tritonbench_args]

    # If kernel name ends with `-bwd`, then add --bwd flag
    if kernel_name.endswith("-bwd") and "--bwd" not in tritonbench_args:
        tritonbench_args.append("--bwd")

    # Add operator-specific default args if provided
    if operator_args:
        operator_custom_args_applied = {}
        for arg_name, arg_value in operator_args.items():
            arg_flag = f"--{arg_name.replace('_', '-')}"
            # Only apply if not already specified on command line
            already_specified = any(
                arg == arg_flag or arg.startswith(f"{arg_flag}=")
                for arg in tritonbench_args
            )
            if not already_specified:
                if arg_value == "":
                    # Boolean flag - just add the flag, no value
                    tritonbench_args.append(arg_flag)
                else:
                    # Regular flag with value
                    tritonbench_args.extend([arg_flag, str(arg_value)])
                operator_custom_args_applied[arg_name] = arg_value
        print(
            f"Applying custom args for {operator_name}: {operator_custom_args_applied}",
            file=sys.stderr,
        )

    # Apply num_inputs if not specified in command line
    if "--num-inputs" not in tritonbench_args:
        # Get per-kernel num_inputs or use MAX_NUM_INPUTS as default
        per_kernel_num_inputs = (operator_args or {}).get("num_inputs", MAX_NUM_INPUTS)
        # Use the smaller of per_kernel_num_inputs and MAX_NUM_INPUTS
        num_inputs = min(per_kernel_num_inputs, MAX_NUM_INPUTS)
        tritonbench_args.extend(["--num-inputs", str(num_inputs)])
        print(
            f"Using num_inputs={num_inputs} for {operator_name}",
            file=sys.stderr,
        )

    # Parse known args and collect unknown ones for operator
    tb_args, unknown_args = tb_parser.parse_known_args(tritonbench_args)

    # Import and get the operator class
    try:
        operator_module = importlib.import_module(tritonbench_module)
        Operator = operator_module.Operator
    except ImportError as e:
        print(
            f"Error: Could not import operator '{operator_name}' from tritonbench",
            file=sys.stderr,
        )
        print(f"Tried: {tritonbench_module}", file=sys.stderr)
        print(f"Import error: {e}", file=sys.stderr)
        sys.exit(1)

    # Import register_benchmark API
    from tritonbench.utils.triton_op import (  # pyright: ignore[reportMissingImports]
        register_benchmark,
    )

    # Register all variants as separate methods
    for module_path, func_name in variants:
        # Import the kernel function
        try:
            module = importlib.import_module(module_path)
            if not hasattr(module, func_name):
                print(
                    f"Error: Module '{module_path}' does not have a function named '{func_name}'",
                    file=sys.stderr,
                )
                continue
            kernel_func = getattr(module, func_name)
        except ImportError as e:
            print(
                f"Error: Could not import {func_name} from {module_path}",
                file=sys.stderr,
            )
            print(f"Import error: {e}", file=sys.stderr)
            continue

        # Create the benchmark method closure to capture the correct module and function
        def create_helion_method(
            mod: Any,  # noqa: ANN401
            kfunc: Callable[..., Any],
        ) -> Callable[..., Any]:
            def helion_method(
                self: object,
                *args: object,
                **kwargs: object,
            ) -> Callable[..., object]:
                """Helion implementation."""

                log_tensor_metadata(args, kwargs)

                # Reset counters for each new input
                counters.clear()

                # Reset all Helion kernels before creating the benchmark function
                # so that each input size can go through its own autotuning.
                from helion.runtime.kernel import Kernel

                for attr_name in dir(mod):
                    attr = getattr(mod, attr_name)
                    if isinstance(attr, Kernel):
                        attr.reset()
                        # Force autotuning unless HELION_AUTOTUNE_EFFORT=none is set
                        # This ensures we run autotuning even if the kernel has pre-specified configs
                        if os.environ.get("HELION_AUTOTUNE_EFFORT", "") != "none":
                            # Only force full autotuning if no configs are provided
                            if not attr.configs:
                                attr.settings.force_autotune = True
                            attr.settings.static_shapes = True

                if isinstance(kfunc, Kernel):
                    # Helion kernel - we call it in a lambda to delay execution until measurement
                    measured_func_callable = lambda: kfunc(*args, **kwargs)  # noqa: E731
                else:
                    # tritonbench integration wrapper - pass tritonbench operator instance as first argument
                    # The wrapper must return a callable that does the actual computation, for delayed execution
                    measured_func_callable = kfunc(self, *args, **kwargs)

                assert callable(measured_func_callable)
                return measured_func_callable

            return helion_method

        # Method name for the benchmark
        variant_name = func_name
        helion_method_name = f"helion_{variant_name}"

        # Use register_benchmark decorator
        decorated_method = register_benchmark(
            operator_name=operator_name,
            func_name=helion_method_name,
            baseline=False,
            enabled=True,
            fwd_only=False,
            label=helion_method_name,
        )(create_helion_method(module, kernel_func))

        # Set the decorated method on the Operator class
        setattr(Operator, helion_method_name, decorated_method)

    def accuracy_fail_hook(
        self: BenchmarkOperator, fn_name: str, metrics: BenchmarkOperatorMetrics
    ) -> None:
        """Hook called after each input benchmark to print the kernel config that causes tritonbench accuracy check failure."""
        if hasattr(metrics, "accuracy") and metrics.accuracy is False:
            if fn_name.startswith("helion_"):
                best_config_decorator = next(
                    iter(counters["best_config_decorator"].keys())
                )
                print(
                    f"{'!' * 80}\n"
                    f"TritonBench accuracy check failed with Helion kernel config: {best_config_decorator}\n"
                    f"{'!' * 80}",
                    file=sys.stderr,
                )

    Operator.benchmark_post_hook = accuracy_fail_hook

    if len(variants) == 1:
        print(
            f"Running {operator_name} benchmark with Helion implementation...\n",
            file=sys.stderr,
        )
    else:
        print(
            f"Running {operator_name} benchmark with {len(variants)} Helion implementations...\n",
            file=sys.stderr,
        )

    # Handle input sharding if requested
    if input_shard_info:
        shard_idx, total_shards = input_shard_info

        # Get the actual number of inputs for this operator
        total_inputs = Operator(
            tb_args=tb_args, extra_args=unknown_args
        )._available_num_inputs

        # Calculate shard boundaries
        inputs_per_shard = total_inputs // total_shards
        extra_inputs = total_inputs % total_shards

        if shard_idx <= extra_inputs:
            start_idx = (shard_idx - 1) * (inputs_per_shard + 1)
            shard_size = inputs_per_shard + 1
        else:
            start_idx = (
                extra_inputs * (inputs_per_shard + 1)
                + (shard_idx - 1 - extra_inputs) * inputs_per_shard
            )
            shard_size = inputs_per_shard

        print(
            f"Running input shard {shard_idx}/{total_shards}: inputs {start_idx} to {start_idx + shard_size - 1} (of {total_inputs} total)",
            file=sys.stderr,
        )

        # Add input-id and num-inputs to the tritonbench args before re-parsing
        tritonbench_args.extend(
            ["--input-id", str(start_idx), "--num-inputs", str(shard_size)]
        )

    try:
        from tritonbench.run import run as tritonbench_run
    except ImportError:
        try:
            from tritonbench.utils.run_utils import tritonbench_run
        except ImportError:
            from pytorch.tritonbench.run import run as tritonbench_run

    with tempfile.NamedTemporaryFile(mode="w+t", suffix=".csv") as tmp:
        tritonbench_args.extend(["--output", tmp.name])
        tritonbench_run(tritonbench_args)
        tmp.seek(0)
        try:
            process_result(kernel_name, tmp.readlines(), results)
        except Exception:
            logger.exception("failed to process results")

    # Force garbage collection multiple times to ensure memory is freed
    for _ in range(3):
        gc.collect()


@functools.cache
def get_device_name() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        # Inconsistent name reporting, so lets fix H100 to report simple name
        if name.startswith("NVIDIA H100"):
            return "NVIDIA H100"
        return name
    return "unknown"


def process_result(
    kernel_name: str, lines: list[str], results: list[RunResult]
) -> None:
    assert kernel_name in KERNEL_METRIC_MAPPINGS
    names = lines[0].strip().split(";")

    shape = []
    metrics = collections.defaultdict(list)
    for row in lines[1:]:
        row_data = row.strip().split(";")
        if row_data[0] == "average" or len(row_data) == 1:
            continue
        for idx, (name, item) in enumerate(zip(names, row_data, strict=True)):
            if idx == 0:
                shape.append(item)
            else:
                if name not in KERNEL_METRIC_MAPPINGS[kernel_name]:
                    logger.info(f"ignoring {name}")
                else:
                    if item == "":
                        # if benchmark failed, tritonbench emits empty string
                        item = 0.0
                    metrics[KERNEL_METRIC_MAPPINGS[kernel_name][name]].append(
                        float(item)
                    )

    results.append(
        RunResult(
            model=kernel_name,
            device=get_device_name(),
            shape=shape,
            metrics=metrics,
        )
    )


def write_results_to_json(
    output: str, results: list[RunResult], append_to_output: bool = False
) -> None:
    if len(results) == 0:
        return

    records = []
    for result in results:
        for metric_name, values in result.metrics.items():
            if len(values) == 0:
                continue

            records.append(
                {
                    "benchmark": {
                        "name": "Helion Benchmark",
                        "extra_info": {
                            "device": result.device,
                        },
                    },
                    "model": {
                        "name": result.model,
                    },
                    "metric": {
                        "name": metric_name,
                        "benchmark_values": values,
                    },
                }
            )

    # If appending and file exists, read existing data first
    if append_to_output and os.path.exists(output):
        try:
            with open(output) as f:
                existing_records = json.load(f)
                if isinstance(existing_records, list):
                    records = existing_records + records
        except (OSError, json.JSONDecodeError):
            # If file is corrupted or can't be read, just overwrite
            pass

    with open(output, "w") as f:
        json.dump(records, f, indent=2)


def main() -> None:
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run Helion kernels with tritonbench",
        allow_abbrev=False,  # Disable prefix matching to prevent --k from matching --kernel
    )
    parser.add_argument(
        "--kernel",
        "--op",
        type=str,
        dest="kernel",
        help="Name(s) of the Helion kernel module(s) to run. Can be a single kernel or comma-separated list (e.g., vector_add or vector_add,rms_norm). If not specified, runs all kernels.",
    )
    parser.add_argument(
        "--input-shard",
        type=str,
        help="Run only a subset of inputs for each kernel. Format: M/N where M is the shard number (1-indexed) and N is the total number of shards. For example, --input-shard 1/3 runs the first third of inputs for each kernel.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="The output filename (json)",
    )
    parser.add_argument(
        "--append-to-output",
        action="store_true",
        help="Append results to existing output file instead of overwriting",
    )
    parser.add_argument(
        "--list-impls-for-benchmark-ci",
        action="store_true",
        help="List implementations to be run on Benchmark CI for specified kernel(s).",
    )

    # Parse known args to get the kernel name, pass rest to tritonbench
    args, tritonbench_args = parser.parse_known_args()

    # Add default tolerance values if not already specified
    if "--atol" not in tritonbench_args:
        tritonbench_args.extend(["--atol", "1e-2"])
    if "--rtol" not in tritonbench_args:
        tritonbench_args.extend(["--rtol", "1e-2"])

    # Check if --bwd flag is used directly and ban it
    if "--bwd" in tritonbench_args:
        print(
            "Error: Direct usage of --bwd flag is not allowed. Please use the -bwd suffix in the operator name instead.\n"
            "Example: Instead of 'python benchmarks/run.py --op layer_norm --bwd', use 'python benchmarks/run.py --op layer_norm-bwd'",
            file=sys.stderr,
        )
        sys.exit(1)

    # Handle --list-impls-for-benchmark-ci flag
    if args.list_impls_for_benchmark_ci:
        assert args.kernel, (
            "--op or --kernel must be specified with --list-impls-for-benchmark-ci"
        )
        # List implementations for specified kernels to be run on Benchmark CI
        kernel_names = [k.strip() for k in args.kernel.split(",")]
        for kernel in kernel_names:
            assert kernel in KERNEL_METRIC_MAPPINGS, (
                f"Unable to find kernel in KERNEL_METRIC_MAPPINGS: {kernel}"
            )

            # Extract implementation names that have speedup metrics
            implementations = []
            baseline_impl = ""

            for metric_key, metric_value in KERNEL_METRIC_MAPPINGS[kernel].items():
                # Find the baseline implementation
                if metric_value == "baseline":
                    baseline_impl = metric_key
                # Get keys ending with "-speedup"
                elif metric_key.endswith("-speedup"):
                    # Remove the "-speedup" suffix to get implementation name
                    impl_name = metric_key[: -len("-speedup")]
                    implementations.append(impl_name)

            implementations = sorted(implementations)
            assert implementations, f"No implementations found for kernel: {kernel}"
            print(
                f"{kernel}: impls={','.join(implementations)} baseline={baseline_impl}"
            )
        sys.exit(0)

    # Check and setup tritonbench if needed
    check_and_setup_tritonbench()

    # Store input-shard info for later processing
    input_shard_info = None
    if args.input_shard:
        try:
            shard_idx, total_shards = map(int, args.input_shard.split("/"))
            if shard_idx < 1 or shard_idx > total_shards:
                print(
                    f"Error: Shard number {shard_idx} must be between 1 and {total_shards}",
                    file=sys.stderr,
                )
                sys.exit(1)
            input_shard_info = (shard_idx, total_shards)
        except ValueError:
            print(
                f"Error: Invalid input-shard format '{args.input_shard}'. Expected format: M/N (e.g., 1/3)",
                file=sys.stderr,
            )
            sys.exit(1)

    results: list[RunResult] = []

    if args.kernel:
        # Parse comma-separated kernel names
        kernel_names = [k.strip() for k in args.kernel.split(",")]

        # Validate all kernel names first
        invalid_kernels = [k for k in kernel_names if k not in KERNEL_MAPPINGS]
        if invalid_kernels:
            print(
                f"Error: Unknown kernel(s): {', '.join(invalid_kernels)}",
                file=sys.stderr,
            )
            print(
                f"Available kernels: {', '.join(KERNEL_MAPPINGS.keys())}",
                file=sys.stderr,
            )
            sys.exit(1)

        # Run specified kernels
        if len(kernel_names) == 1:
            run_kernel(kernel_names[0], tritonbench_args, input_shard_info, results)
        else:
            print(
                f"Running {len(kernel_names)} kernels: {', '.join(kernel_names)}...\n",
                file=sys.stderr,
            )
            for kernel_name in kernel_names:
                print(f"\n{'=' * 60}", file=sys.stderr)
                print(f"Kernel: {kernel_name}", file=sys.stderr)
                print(f"{'=' * 60}\n", file=sys.stderr)
                run_kernel(
                    kernel_name, tritonbench_args.copy(), input_shard_info, results
                )
    else:
        # Run all kernels
        print(f"Running all {len(KERNEL_MAPPINGS)} kernels...\n", file=sys.stderr)
        for kernel_name in KERNEL_MAPPINGS:
            print(f"\n{'=' * 60}", file=sys.stderr)
            print(f"Kernel: {kernel_name}", file=sys.stderr)
            print(f"{'=' * 60}\n", file=sys.stderr)
            run_kernel(kernel_name, tritonbench_args.copy(), input_shard_info, results)

    if args.output:
        write_results_to_json(
            args.output, results, append_to_output=args.append_to_output
        )


if __name__ == "__main__":
    main()
