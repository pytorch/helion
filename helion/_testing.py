from __future__ import annotations

import importlib
from pathlib import Path
import sys
from typing import TYPE_CHECKING
from typing import Callable

import torch
from triton.testing import do_bench

from .runtime.config import Config

if TYPE_CHECKING:
    import types

    from .runtime.kernel import Kernel


DEVICE = torch.device("cuda")
EXAMPLES_DIR: Path = Path(__file__).parent.parent / "examples"


def import_path(filename: Path) -> types.ModuleType:
    module_name = f"{__name__}.{filename.stem}"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, filename)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        sys.modules[module_name] = module
    return sys.modules[module_name]


def code_and_output(
    # pyre-ignore[11]
    fn: Kernel,
    args: tuple[object, ...],
    **kwargs: object,
) -> tuple[str, object]:
    if kwargs:
        config = Config(**kwargs)  # pyre-ignore[6]
    elif fn.configs:
        (config,) = fn.configs
    else:
        config = fn.bind(args).config_spec.default_config()
    code = fn.bind(args).to_triton_code(config)
    compiled_kernel = fn.bind(args).compile_config(config)
    try:
        result = compiled_kernel(*args)
    except Exception:
        sys.stderr.write(f"Failed to run kernel:\n{code}\n")
        raise
    return code, result


def run_example(
    kernel_fn: Callable[..., torch.Tensor] | Kernel | dict[str, Kernel],
    baseline_fn: Callable[..., torch.Tensor] | dict[str, Callable[..., torch.Tensor]],
    args: tuple[object, ...],
    kernel_name: str = "helion",
    baseline_name: str = "torch",
    rtol: float = 1e-2,
    atol: float = 1e-1,
) -> None:
    """Run complete example: correctness check + benchmark.

    Args:
        kernel_fn: Single kernel function, or dict of {name: function} for multiple kernel variants
        baseline_fn: Single baseline function or dict of {name: function} for multiple baselines
        args: Arguments to pass to all functions
        kernel_name: Name for single kernel in output (default: "helion")
        baseline_name: Name for single baseline in output (default: "torch")
        rtol: Relative tolerance for correctness check (default: 1e-2)
        atol: Absolute tolerance for correctness check (default: 1e-1)
    """
    torch.set_float32_matmul_precision("high")

    # Normalize to dict format
    kernels = kernel_fn if isinstance(kernel_fn, dict) else {kernel_name: kernel_fn}
    baselines = (
        baseline_fn if isinstance(baseline_fn, dict) else {baseline_name: baseline_fn}
    )

    # Check correctness against first baseline
    first_baseline_name, first_baseline_func = next(iter(baselines.items()))
    expected = first_baseline_func(*args)

    for name, func in {**kernels, **baselines}.items():
        if name != first_baseline_name:
            print(f"Testing {name} correctness...", file=sys.stderr)
            torch.testing.assert_close(func(*args), expected, rtol=rtol, atol=atol)

    # Benchmark all functions
    all_times = {
        name: do_bench(lambda fn=fn: fn(*args))
        for name, fn in {**kernels, **baselines}.items()
    }

    best_baseline_time = min(all_times[name] for name in baselines)

    # Print results
    print(f"\n{'=' * 65}\nBenchmark Results\n{'=' * 65}", file=sys.stderr)
    print(
        f"{'Implementation':<20} {'Time (ms)':<12} {'Speedup':<15}\n{'-' * 65}",
        file=sys.stderr,
    )

    for name, time in all_times.items():
        is_best_baseline = name in baselines and time == best_baseline_time
        speedup_str = (
            "1.00x (ref)" if is_best_baseline else f"{best_baseline_time / time:.2f}x"
        )
        print(f"{name:<20} {time:<12.4f} {speedup_str:<15}", file=sys.stderr)

    print(f"{'=' * 65}\n", file=sys.stderr)


def check_example(
    name: str,
    args: tuple[torch.Tensor, ...],
    expected: torch.Tensor,
    fn_name: str | None = None,
    skip_accuracy: bool = False,
    static_shapes: bool | None = None,
    **kwargs: object,
) -> str:
    """Helper used in unit tests to run a single example kernel and check its output."""
    kernel_fn = getattr(import_path(EXAMPLES_DIR / f"{name}.py"), fn_name or name)
    if static_shapes is not None:
        assert static_shapes in (True, False)
        kernel_fn.settings.static_shapes = static_shapes

    code, result = code_and_output(
        kernel_fn,
        args,
        **kwargs,
    )
    skip_accuracy or torch.testing.assert_close(result, expected, atol=1e-1, rtol=1e-2)
    return code
