# pyright: reportMissingImports=false

"""Performance comparison between Helion, torch.compile, Triton, and PyTorch eager by leveraging TritonBench.

Currently supported kernels are listed in `KERNEL_MAPPINGS` in `benchmarks/run.py`.

Usage:
$ python benchmarks/run.py [tritonbench args...] [--kernel <kernel_name(s)>]

Example usage:
$ python benchmarks/run.py --metrics speedup,accuracy --kernel vector_add  # Runs vector_add kernel
$ python benchmarks/run.py --metrics speedup,accuracy --kernel vector_add,rms_norm  # Runs multiple kernels
$ python benchmarks/run.py --metrics speedup,accuracy  # Runs all kernels
"""

from __future__ import annotations

import argparse
import gc
import importlib
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
from typing import Callable

# Import tritonbench's run module which applies the async_task patch
# This ensures the patch is applied before we need it
import tritonbench.run  # noqa: F401  # This applies the async_task patch

# Maps tritonbench op names to Helion kernel examples
KERNEL_MAPPINGS: dict[str, tuple[str, str, str]] = {
    # <tritonbench_op_name>: (<tritonbench_module_path>, <helion_kernel_module_path>, <helion_kernel_function_name>)
    "vector_add": ("tritonbench.operators.vector_add.operator", "examples.add", "add"),
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
    ),
    "fp8_attention": (
        "tritonbench.operators.fp8_attention.operator",
        "examples.fp8_attention",
        "fp8_attention_tritonbench",
    ),
    "cross_entropy": (
        "tritonbench.operators.cross_entropy.operator",
        "examples.cross_entropy",
        "cross_entropy",
    ),
    "fused_linear_cross_entropy": (
        "tritonbench.operators.fused_linear_cross_entropy.operator",
        "examples.fused_linear_cross_entropy",
        "fused_linear_cross_entropy",
    ),
    "ragged_attention": (
        "tritonbench.operators.ragged_attention.operator",
        "examples.ragged_attention",
        "ragged_attention_tritonbench",
    ),
    "grouped_gemm": (
        "tritonbench.operators.grouped_gemm.operator",
        "examples.grouped_gemm",
        "grouped_gemm_tritonbench",
    ),
    "welford": (
        "tritonbench.operators.welford.operator",
        "examples.welford",
        "welford_tritonbench",
    ),
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
                    "https://github.com/pytorch-labs/tritonbench.git",
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
        subprocess.run(
            [sys.executable, "install.py", install_flag],
            cwd=tritonbench_path,
            check=True,
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


def run_kernel(kernel_name: str, tritonbench_args: list[str]) -> None:
    """Run a single kernel benchmark."""
    # Check if kernel is in the mapping table
    if kernel_name not in KERNEL_MAPPINGS:
        print(f"Error: Unknown kernel '{kernel_name}'", file=sys.stderr)
        print(
            f"Available kernels: {', '.join(KERNEL_MAPPINGS.keys())}", file=sys.stderr
        )
        sys.exit(1)

    tritonbench_module, module_path, func_name = KERNEL_MAPPINGS[kernel_name]

    # Import from the mapped module
    try:
        module = importlib.import_module(module_path)
        if not hasattr(module, func_name):
            print(
                f"Error: Module '{module_path}' does not have a function named '{func_name}'",
                file=sys.stderr,
            )
            sys.exit(1)
        kernel_func = getattr(module, func_name)
    except ImportError as e:
        print(
            f"Error: Could not import {func_name} from {module_path}", file=sys.stderr
        )
        print(f"Import error: {e}", file=sys.stderr)
        sys.exit(1)
        return

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

    # Get module's TRITONBENCH_ARGS if any
    module_args = getattr(module, "TRITONBENCH_ARGS", {})

    # Add module args to tritonbench_args if not already present
    for arg_name, arg_value in module_args.items():
        arg_flag = f"--{arg_name.replace('_', '-')}"
        if arg_flag not in tritonbench_args:
            tritonbench_args.extend([arg_flag, str(arg_value)])

    # Parse known args and collect unknown ones for operator
    tb_args, unknown_args = tb_parser.parse_known_args(tritonbench_args)

    # Import and run the operator
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

    # Create the benchmark method
    def helion_method(
        self: Any,
        *args: Any,
    ) -> Callable[..., Any]:
        """Helion implementation."""

        # Reset all Helion kernels before creating the benchmark function
        # so that each input size can go through its own autotuning.
        from helion.runtime.kernel import Kernel

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, Kernel):
                attr.reset()

        def _inner() -> Callable[..., Any]:
            # Force autotuning unless HELION_USE_DEFAULT_CONFIG=1 is set
            # This ensures we run autotuning even if the kernel has pre-specified configs
            if os.environ.get("HELION_USE_DEFAULT_CONFIG", "0") != "1":
                # Find all Kernel objects in the module and force autotuning
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, Kernel):
                        attr.settings.force_autotune = True

            # Handle special case for ragged_attention which needs additional parameters
            if kernel_name == "ragged_attention" and len(args) == 6:
                # Extract the 6 arguments from tritonbench
                q, k, v, seq_offsets, num_targets, max_seq_len = args
                
                # Convert None num_targets to empty tensor
                if num_targets is None:
                    num_targets = torch.empty(0, dtype=torch.int32, device=q.device)
                
                # Call with the same parameter order and defaults as triton_hstu_mha
                # Values taken from the operator.py defaults
                return kernel_func(
                    max_seq_len,
                    1.0 / q.size(-1),  # alpha = 1.0 / attn_dim (from operator.py line 74)
                    q,
                    k,
                    v,
                    seq_offsets,
                    num_targets,
                    0,     # max_attn_len (default from operator.py)
                    0,     # contextual_seq_len (default from operator.py)
                    True,  # sort_by_length (default from triton_hstu_mha call)
                )
            
            return kernel_func(*args)

        return _inner

    # Method name for the benchmark
    helion_method_name = f"helion_{kernel_name}"

    # Import register_benchmark API
    from tritonbench.utils.triton_op import (  # pyright: ignore[reportMissingImports]
        register_benchmark,
    )

    # Use register_benchmark decorator
    decorated_method = register_benchmark(
        operator_name=operator_name,
        func_name=helion_method_name,
        baseline=False,
        enabled=True,
        fwd_only=False,
        label=helion_method_name,
    )(helion_method)

    # Set the decorated method on the Operator class
    setattr(Operator, helion_method_name, decorated_method)

    print(
        f"Running {operator_name} benchmark with Helion implementation...\n",
        file=sys.stderr,
    )

    # Create and run the operator with unknown args
    op = Operator(tb_args=tb_args, extra_args=unknown_args)

    # Run with proper parameters
    warmup = int(getattr(tb_args, "warmup", 25))
    rep = int(getattr(tb_args, "iter", 100))
    op.run(warmup=warmup, rep=rep)

    # Print results
    print("\nBenchmark Results:", file=sys.stderr)
    print(op.output, file=sys.stderr)

    # Clean up memory after running the kernel
    # Delete the operator instance which contains all allocated tensors
    del op

    # Force garbage collection multiple times to ensure memory is freed
    for _ in range(3):
        gc.collect()


def main() -> None:
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Helion kernels with tritonbench")
    parser.add_argument(
        "--kernel",
        type=str,
        help="Name(s) of the Helion kernel module(s) to run. Can be a single kernel or comma-separated list (e.g., vector_add or vector_add,rms_norm). If not specified, runs all kernels.",
    )

    # Parse known args to get the kernel name, pass rest to tritonbench
    args, tritonbench_args = parser.parse_known_args()

    # Check and setup tritonbench if needed
    check_and_setup_tritonbench()

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
            run_kernel(kernel_names[0], tritonbench_args)
        else:
            print(
                f"Running {len(kernel_names)} kernels: {', '.join(kernel_names)}...\n",
                file=sys.stderr,
            )
            for kernel_name in kernel_names:
                print(f"\n{'=' * 60}", file=sys.stderr)
                print(f"Kernel: {kernel_name}", file=sys.stderr)
                print(f"{'=' * 60}\n", file=sys.stderr)
                run_kernel(kernel_name, tritonbench_args.copy())
    else:
        # Run all kernels
        print(f"Running all {len(KERNEL_MAPPINGS)} kernels...\n", file=sys.stderr)
        for kernel_name in KERNEL_MAPPINGS:
            print(f"\n{'=' * 60}", file=sys.stderr)
            print(f"Kernel: {kernel_name}", file=sys.stderr)
            print(f"{'=' * 60}\n", file=sys.stderr)
            run_kernel(kernel_name, tritonbench_args.copy())


if __name__ == "__main__":
    main()
