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
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import functools
import gc
import importlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
from typing import Callable

import torch

logger: logging.Logger = logging.getLogger(__name__)


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
    "rms_norm": (
        "tritonbench.operators.rms_norm.operator",
        "examples.rms_norm",
        "rms_norm_tritonbench",
    ),
    "sum": ("tritonbench.operators.sum.operator", "examples.sum", "sum_tritonbench"),
    "softmax": (
        "tritonbench.operators.softmax.operator",
        "examples.softmax",
        "softmax",
    ),
    "jagged_mean": (
        "tritonbench.operators.jagged_mean.operator",
        "examples.jagged_mean",
        "jagged_mean_tritonbench",
        {"B": 32, "M": 8, "seqlen": 64}
        if os.environ.get("HELION_DEV_LOW_VRAM", "0") == "1"
        else {},
    ),
    "fp8_gemm": (
        "tritonbench.operators.fp8_gemm.fp8_gemm",
        "examples.fp8_gemm",
        "fp8_gemm_tritonbench",
    ),
    "flash_attention": (
        "tritonbench.operators.flash_attention.operator",
        "examples.attention",
        "attention",
        {
            "d_head": 128
        },  # Set default head dimension to 128 for TLX attention compatibility
    ),
    "cross_entropy": (
        "tritonbench.operators.cross_entropy.operator",
        "examples.cross_entropy",
        "cross_entropy",
        {"B": 4, "T": 512, "v_range": "10,15"}
        if os.environ.get("HELION_DEV_LOW_VRAM", "0") == "1"
        else {},
    ),
    "fp8_attention": (
        "tritonbench.operators.fp8_attention.operator",
        "examples.fp8_attention",
        "fp8_attention_tritonbench",
    ),
    "layer_norm": (
        "tritonbench.operators.layer_norm.operator",
        "examples.layer_norm",
        "layer_norm_fwd",
    ),
    "jagged_softmax": (
        "tritonbench.operators.jagged_softmax.operator",
        "examples.jagged_softmax",
        "jagged_softmax_tritonbench",
    ),
    # Multiple kernel variants:
    "gemm": (
        "tritonbench.operators.gemm.operator",
        [
            ("examples.matmul", "matmul_tritonbench"),
            ("examples.matmul_split_k", "matmul_split_k_tritonbench"),
        ],
    ),
}


KERNEL_METRIC_MAPPINGS: dict[str, dict[str, str]] = {
    "vector_add": {
        "triton_add-speedup": "triton_speedup",
        "triton_add-accuracy": "triton_accuracy",
        "torch_compile_add-speedup": "torch_compile_speedup",
        "torch_compile_add-accuracy": "torch_compile_accuracy",
        "helion_add-speedup": "helion_speedup",
        "helion_add-accuracy": "helion_accuracy",
    },
    "vector_exp": {
        "triton_exp-speedup": "triton_speedup",
        "triton_exp-accuracy": "triton_accuracy",
        "torch_compile_exp-speedup": "torch_compile_speedup",
        "torch_compile_exp-accuracy": "torch_compile_accuracy",
        "helion_exp_tritonbench-speedup": "helion_speedup",
        "helion_exp_tritonbench-accuracy": "helion_accuracy",
    },
    "sum": {
        "triton_sum-speedup": "triton_speedup",
        "triton_sum-accuracy": "triton_accuracy",
        "torch_compile_sum-speedup": "torch_compile_speedup",
        "torch_compile_sum-accuracy": "torch_compile_accuracy",
        "helion_sum_tritonbench-speedup": "helion_speedup",
        "helion_sum_tritonbench-accuracy": "helion_accuracy",
    },
    "layer_norm": {
        "liger_layer_norm-speedup": "triton_speedup",
        "liger_layer_norm-accuracy": "triton_accuracy",
        "torch_compile_layer_norm-speedup": "torch_compile_speedup",
        "torch_compile_layer_norm-accuracy": "torch_compile_accuracy",
        "helion_layer_norm_fwd-speedup": "helion_speedup",
        "helion_layer_norm_fwd-accuracy": "helion_accuracy",
    },
    "softmax": {
        "triton_softmax-speedup": "triton_speedup",
        "triton_softmax-accuracy": "triton_accuracy",
        "torch_compile_softmax-speedup": "torch_compile_speedup",
        "torch_compile_softmax-accuracy": "torch_compile_accuracy",
        "helion_softmax-speedup": "helion_speedup",
        "helion_softmax-accuracy": "helion_accuracy",
    },
    "rms_norm": {
        "liger_rms-speedup": "triton_speedup",
        "liger_rms-accuracy": "triton_accuracy",
        "torch_compile_rms-speedup": "torch_compile_speedup",
        "torch_compile_rms-accuracy": "torch_compile_accuracy",
        "helion_rms_norm_tritonbench-speedup": "helion_speedup",
        "helion_rms_norm_tritonbench-accuracy": "helion_accuracy",
    },
    "cross_entropy": {
        "liger_cross_entropy_loss-speedup": "triton_speedup",
        "liger_cross_entropy_loss-accuracy": "triton_accuracy",
        "torch_compile_cross_entropy_loss-speedup": "torch_compile_speedup",
        "torch_compile_cross_entropy_loss-accuracy": "torch_compile_accuracy",
        "helion_cross_entropy-speedup": "helion_speedup",
        "helion_cross_entropy-accuracy": "helion_accuracy",
    },
}


def get_system_memory_gb() -> float:
    """Get system memory in GB."""
    try:
        # Try to read from /proc/meminfo on Linux
        meminfo_path = Path("/proc/meminfo")
        if meminfo_path.exists():
            with open(meminfo_path) as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        # Extract memory in kB and convert to GB
                        mem_kb = int(line.split()[1])
                        return mem_kb / (1024 * 1024)

        # Fallback: use psutil if available
        try:
            import psutil

            return psutil.virtual_memory().total / (1024**3)
        except ImportError:
            pass

    except Exception:
        pass

    # Default to assuming high memory if we can't detect
    return 32.0


def check_and_setup_tritonbench() -> None:
    """Check if tritonbench is installed and install it from GitHub if not."""
    # Check if tritonbench is already installed
    try:
        import tritonbench  # pyright: ignore[reportMissingImports]

        return  # Already installed
    except ImportError:
        pass

    print("Tritonbench not found. Installing...", file=sys.stderr)

    # Clone to benchmarks/tritonbench
    benchmarks_dir = Path(__file__).parent
    tritonbench_path = benchmarks_dir / "tritonbench"

    try:
        # Clone the repository if it doesn't exist
        if not tritonbench_path.exists():
            print("Cloning tritonbench repository...", file=sys.stderr)
            subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/meta-pytorch/tritonbench.git",
                    str(tritonbench_path),
                ],
                check=True,
            )

            # Initialize submodules
            print("Initializing tritonbench's submodules...", file=sys.stderr)
            subprocess.run(
                ["git", "submodule", "update", "--init", "--recursive"],
                cwd=tritonbench_path,
                check=True,
            )

        # Detect system memory and choose install flags.
        # Low-memory systems can freeze when building dependencies like flash-attn,
        # so we only install the Liger library in that case.
        memory_gb = get_system_memory_gb()
        install_flag = "--liger" if memory_gb < 16 else "--all"

        # Install optional dependencies for tritonbench
        print(
            f"Running install.py {install_flag} (detected {memory_gb:.1f}GB system RAM)...",
            file=sys.stderr,
        )
        env = os.environ.copy()
        if install_flag == "--all":
            # Set max jobs to 4 to avoid OOM
            env["MAX_JOBS"] = "4"
        subprocess.run(
            [sys.executable, "install.py", install_flag],
            cwd=tritonbench_path,
            check=True,
            env=env,
        )

        # Install tritonbench package
        print("Installing tritonbench package...", file=sys.stderr)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(tritonbench_path)],
            check=True,
        )

        # Invalidate import caches to recognize newly installed package
        importlib.invalidate_caches()

        # Verify installation worked
        try:
            import tritonbench  # noqa: F401  # pyright: ignore[reportMissingImports]

            print(
                f"Tritonbench installed successfully with {install_flag}.",
                file=sys.stderr,
            )
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
    try:
        from tritonbench.utils.parser import (  # pyright: ignore[reportMissingImports]
            get_parser,
        )
    except ImportError:
        print(
            "Error: Could not import tritonbench. Make sure it's in the path.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Get the tritonbench operator name
    operator_name = kernel_name

    # Parse tritonbench arguments
    tb_parser = get_parser()

    assert "--op" not in tritonbench_args
    tritonbench_args = ["--op", operator_name, *tritonbench_args]

    # Add operator-specific default args if provided
    if operator_args:
        print(
            f"Applying custom args for {operator_name}: {operator_args}",
            file=sys.stderr,
        )
        # First, remove any existing occurrences of these args
        for arg_name, arg_value in operator_args.items():
            arg_flag = f"--{arg_name.replace('_', '-')}"
            # Remove existing arg if present
            while arg_flag in tritonbench_args:
                idx = tritonbench_args.index(arg_flag)
                tritonbench_args.pop(idx)  # Remove flag
                if idx < len(tritonbench_args) and not tritonbench_args[idx].startswith(
                    "--"
                ):
                    tritonbench_args.pop(idx)  # Remove value
            # Add the custom arg
            tritonbench_args.extend([arg_flag, str(arg_value)])

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

                # Reset all Helion kernels before creating the benchmark function
                # so that each input size can go through its own autotuning.
                from helion.runtime.kernel import Kernel

                for attr_name in dir(mod):
                    attr = getattr(mod, attr_name)
                    if isinstance(attr, Kernel):
                        attr.reset()
                        # Force autotuning unless HELION_USE_DEFAULT_CONFIG=1 is set
                        # This ensures we run autotuning even if the kernel has pre-specified configs
                        if os.environ.get("HELION_USE_DEFAULT_CONFIG", "0") != "1":
                            attr.settings.force_autotune = True
                            attr.settings.static_shape = True  # pyright: ignore[reportAttributeAccessIssue]

                def _inner() -> Callable[..., Any] | object:
                    # BENCHMARK HOT PATH, do not add any new logic here
                    result = kfunc(*args, **kwargs)
                    if callable(result):
                        return result()
                    return result

                return _inner

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
        from pytorch.tritonbench.run import run as tritonbench_run

    with tempfile.NamedTemporaryFile(mode="w+t", suffix=".csv") as tmp:
        tritonbench_args.extend(["--output", tmp.name])
        tritonbench_run(tritonbench_args)
        tmp.seek(0)
        try:
            process_result(kernel_name, tmp.readlines(), results)
        except Exception:
            logger.info("fail", exc_info=True)

    # Force garbage collection multiple times to ensure memory is freed
    for _ in range(3):
        gc.collect()


@functools.cache
def get_device_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
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
        if row_data[0] == "average":
            continue
        for idx, (name, item) in enumerate(zip(names, row_data, strict=True)):
            if idx == 0:
                shape.append(item)
            else:
                if name not in KERNEL_METRIC_MAPPINGS[kernel_name]:
                    logger.info(f"ignoring {name}")
                else:
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


def write_results_to_json(output: str, results: list[RunResult]) -> None:
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
    with open(output, "w") as f:
        json.dump(records, f)


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

    # Parse known args to get the kernel name, pass rest to tritonbench
    args, tritonbench_args = parser.parse_known_args()

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
        write_results_to_json(args.output, results)


if __name__ == "__main__":
    main()
