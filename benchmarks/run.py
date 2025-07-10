"""Performance comparison between Helion, torch.compile, Triton, and PyTorch eager by leveraging TritonBench.

Currently supported kernels are listed in `KERNEL_MAPPINGS` in `benchmarks/run.py`.

Usage:
$ python benchmarks/run.py [tritonbench args...] --kernel <kernel_name>

Example usage:
$ python benchmarks/run.py --metrics speedup,accuracy --kernel vector_add
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import subprocess
import sys
from typing import Any
from typing import Callable

# Maps tritonbench op names to Helion kernel examples
KERNEL_MAPPINGS: dict[str, tuple[str, str] | tuple[str, str, dict[str, Any]]] = {
    # <tritonbench_op_name>: (<helion_kernel_module_path>, <helion_kernel_function_name>, <optional_extra_args>)
    "vector_add": ("examples.add", "add"),
    "embedding": ("examples.embedding", "embedding_tritonbench"),
    "vector_exp": ("examples.exp", "exp_tritonbench"),
    # TODO(yf225): reduction dim size = 8192 currently throws error. After it's fixed we can remove "num_inputs" extra arg.
    "rms_norm": ("examples.rms_norm", "rms_norm_tritonbench", {"num_inputs": 3}),
    "sum": ("examples.sum", "sum_tritonbench"),
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
        import tritonbench

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
            import tritonbench  # noqa: F401

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


def main() -> None:
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Helion kernels with tritonbench")
    parser.add_argument(
        "--kernel",
        type=str,
        required=True,
        help="Name of the Helion kernel module (e.g., vector_add)",
    )

    # Parse known args to get the kernel name, pass rest to tritonbench
    args, tritonbench_args = parser.parse_known_args()

    # Check and setup tritonbench if needed
    check_and_setup_tritonbench()

    kernel_name = args.kernel

    # Check if kernel is in the mapping table
    assert kernel_name in KERNEL_MAPPINGS
    mapping = KERNEL_MAPPINGS[kernel_name]

    # Parse mapping - can be (module, func) or (module, func, extra_args)
    if len(mapping) == 2:
        module_path, func_name = mapping
        kernel_extra_args = {}
    else:
        module_path, func_name, kernel_extra_args = mapping
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
        from tritonbench.utils.parser import get_parser  # pyre-ignore[21]
    except ImportError:
        print(
            "Error: Could not import tritonbench. Make sure it's in the path.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Get the tritonbench operator name (assume it's the same as the kernel name)
    operator_name = kernel_name

    # Parse tritonbench arguments
    tb_parser = get_parser()

    assert "--op" not in tritonbench_args
    tritonbench_args = ["--op", operator_name, *tritonbench_args]

    # Apply kernel-specific default arguments if not already specified by user
    for arg_name, arg_value in kernel_extra_args.items():
        # Convert underscore to hyphen for CLI args (e.g., num_inputs -> --num-inputs)
        cli_arg = f"--{arg_name.replace('_', '-')}"
        if cli_arg not in tritonbench_args:
            tritonbench_args.extend([cli_arg, str(arg_value)])

    tb_args = tb_parser.parse_args(tritonbench_args)

    # Register the Helion kernel with tritonbench BEFORE importing the operator
    from tritonbench.utils.triton_op import (  # pyre-ignore[21]
        register_benchmark,
    )

    # Create the benchmark method
    def create_helion_method(  # pyre-ignore[3]
        kernel_func: Callable[..., Any],  # pyre-ignore[2]
    ) -> Callable[..., Any]:
        def helion_method(  # pyre-ignore[3]
            self: Any,  # pyre-ignore[2]
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

            def _inner() -> Callable[..., Any]:  # pyre-ignore[3]
                return kernel_func(*args)

            return _inner

        return helion_method

    # Register it as a benchmark first
    helion_method_name = f"helion_{kernel_name}"
    register_benchmark(
        operator_name=operator_name,
        func_name=helion_method_name,
        baseline=False,
        enabled=True,
        label=helion_method_name,
    )

    # Import and run the operator
    operator_module_name = f"tritonbench.operators.{operator_name}.operator"
    try:
        operator_module = importlib.import_module(operator_module_name)
        Operator = operator_module.Operator
    except ImportError:
        print(
            f"Error: Could not import operator '{operator_name}' from tritonbench",
            file=sys.stderr,
        )
        sys.exit(1)
        return

    # Monkey-patch the Operator class after import
    setattr(Operator, helion_method_name, create_helion_method(kernel_func))

    print(
        f"Running {operator_name} benchmark with Helion implementation...\n",
        file=sys.stderr,
    )

    # Create and run the operator
    op = Operator(tb_args=tb_args, extra_args={})

    # Run with proper parameters
    warmup = getattr(tb_args, "warmup", 25)
    rep = getattr(tb_args, "iter", 100)
    op.run(warmup=warmup, rep=rep)

    # Print results
    print("\nBenchmark Results:", file=sys.stderr)
    print(op.output, file=sys.stderr)


if __name__ == "__main__":
    main()
