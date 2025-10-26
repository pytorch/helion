from __future__ import annotations

import collections
import contextlib
import functools
import importlib
import inspect
import operator
import os
from pathlib import Path
import re
import sys
from typing import TYPE_CHECKING
from typing import Callable
from typing import Generator
import unittest

import pytest
import torch
from torch.utils._pytree import tree_map
import triton

from ._compat import get_tensor_descriptor_fn_name
from ._utils import counters
from .autotuner.benchmarking import compute_repeat
from .autotuner.benchmarking import interleaved_bench
from .runtime.config import Config
from .runtime.ref_mode import is_ref_mode_enabled
from .runtime.settings import RefMode

if TYPE_CHECKING:
    import types

    from .runtime.kernel import Kernel


DEVICE = torch.device("xpu") if torch.xpu.is_available() else torch.device("cuda")
PROJECT_ROOT: Path = Path(__file__).parent.parent
EXAMPLES_DIR: Path = PROJECT_ROOT / "examples"


def is_cuda() -> bool:
    """Return True if running on CUDA (NVIDIA GPU)."""
    return (
        triton.runtime.driver.active.get_current_target().backend == "cuda"  # pyright: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]
        and DEVICE.type == "cuda"
    )


def get_nvidia_gpu_model() -> str:
    """
    Retrieves the model of the NVIDIA GPU being used.
    Will return the name of the current device.
    Returns:
        str: The model of the NVIDIA GPU or empty str if not found.
    """
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return getattr(props, "name", "")
    return ""


def skipIfRefEager(reason: str) -> Callable[[Callable], Callable]:
    """Skip test if running in ref eager mode (HELION_INTERPRET=1)."""
    return unittest.skipIf(os.environ.get("HELION_INTERPRET") == "1", reason)


def skipIfNormalMode(reason: str) -> Callable[[Callable], Callable]:
    """Skip test if running in normal mode (i.e. if HELION_INTERPRET=1 is not set)."""
    return unittest.skipIf(os.environ.get("HELION_INTERPRET") != "1", reason)


def skipIfRocm(reason: str) -> Callable[[Callable], Callable]:
    """Skip test if running with rocm"""
    return unittest.skipIf(torch.version.hip is not None, reason)  # pyright: ignore[reportAttributeAccessIssue]


def skipIfXPU(reason: str) -> Callable[[Callable], Callable]:
    """Skip test if running with Intel XPU"""
    return unittest.skipIf(torch.xpu.is_available(), reason)  # pyright: ignore[reportAttributeAccessIssue]


def skipIfA10G(reason: str) -> Callable[[Callable], Callable]:
    """Skip test if running on A10G GPU"""
    gpu_model = get_nvidia_gpu_model()
    is_a10g = "A10G" in gpu_model
    return unittest.skipIf(is_a10g, reason)


def skipIfNotCUDA() -> Callable[[Callable], Callable]:
    """Skip test if not running on CUDA (NVIDIA GPU)."""
    return unittest.skipIf(
        not is_cuda(), "Test skipped: CUDA (NVIDIA GPU) is not available."
    )


def skipIfLowVRAM(
    reason: str = "Test requires high VRAM",
) -> Callable[[Callable], Callable]:
    """Skip test on systems with low GPU VRAM."""

    threshold_bytes = int(30.0 * (1024**3))
    total_memory: int | None = None
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            total_memory = int(getattr(props, "total_memory", 0))
    except Exception:
        total_memory = None

    low_vram = total_memory is not None and total_memory < threshold_bytes
    return unittest.skipIf(low_vram, reason)


def skipIfPy314(reason: str) -> Callable[[Callable], Callable]:
    """Skip test if running on Python 3.14"""
    return unittest.skipIf(sys.version_info >= (3, 14), reason)


@contextlib.contextmanager
def track_run_ref_calls() -> Generator[list[int], None, None]:
    """Context manager that tracks BoundKernel.run_ref calls.

    Yields:
        A list that will contain the count of run_ref calls.
    """
    from .runtime.kernel import BoundKernel

    original_run_ref = BoundKernel.run_ref
    run_ref_count = [0]

    def tracked_run_ref(self: BoundKernel, *args: object) -> object:
        run_ref_count[0] += 1
        return original_run_ref(self, *args)

    BoundKernel.run_ref = tracked_run_ref

    try:
        yield run_ref_count
    finally:
        BoundKernel.run_ref = original_run_ref


@contextlib.contextmanager
def assert_helion_ref_mode(
    ref_mode: RefMode = RefMode.OFF,
) -> Generator[None, None, None]:
    """Context manager that asserts Helion compilation behavior based on RefMode.

    - RefMode.OFF: expects compilation (run_ref should not be called)
    - RefMode.EAGER: expects no compilation (run_ref should be called)
    """
    with track_run_ref_calls() as run_ref_count:
        yield

        if ref_mode == RefMode.OFF:
            # In normal mode (RefMode.OFF), run_ref should not be called
            assert run_ref_count[0] == 0, (
                f"Expected run_ref to not be called in normal mode (RefMode.OFF), but got: run_ref={run_ref_count[0]}"
            )
        elif ref_mode == RefMode.EAGER:
            # In ref eager mode (RefMode.EAGER), run_ref should be called
            assert run_ref_count[0] > 0, (
                f"Expected run_ref to be called in ref eager mode (RefMode.EAGER), but got: run_ref={run_ref_count[0]}"
            )
        else:
            raise ValueError(f"Unknown RefMode: {ref_mode}")


assert_helion_compilation = functools.partial(
    assert_helion_ref_mode, ref_mode=RefMode.OFF
)

assert_ref_eager_mode = functools.partial(
    assert_helion_ref_mode, ref_mode=RefMode.EAGER
)


class RefEagerTestBase:
    """Base class for all ref eager mode test shards of normal Helion unit test files."""

    # Class-level tracking for assert_close counting
    _assert_close_count = 0
    _original_assert_close_func = None
    # Class-level tracking for assertRaises counting
    _assert_raises_count = 0
    _original_assert_raises_func = None
    # Class-level tracking for skipTest counting
    _skip_test_count = 0
    _original_skip_test_func = None
    # Class-level tracking for pytest.raises patching
    _original_pytest_raises = None
    # Class-level tracking for assertTrue/assertFalse/assertGreater
    _assert_true_count = 0
    _original_assert_true_func = None
    _assert_false_count = 0
    _original_assert_false_func = None
    _assert_greater_count = 0
    _original_assert_greater_func = None

    def setUp(self) -> None:
        """Common setup for all ref eager tests."""
        super().setUp()  # type: ignore[misc]

        # Check if HELION_INTERPRET is already set
        self._in_ref_eager_mode = os.environ.get("HELION_INTERPRET") == "1"

        # If not in ref eager mode, skip the setup
        if not self._in_ref_eager_mode:
            return

        # Reset assert_close counter for this test
        RefEagerTestBase._assert_close_count = 0
        # Reset assertRaises counter for this test
        RefEagerTestBase._assert_raises_count = 0
        # Reset skipTest counter for this test
        RefEagerTestBase._skip_test_count = 0
        # Reset assertTrue/assertFalse/assertGreater counters
        RefEagerTestBase._assert_true_count = 0
        RefEagerTestBase._assert_false_count = 0
        RefEagerTestBase._assert_greater_count = 0

        # Patch torch.testing.assert_close to count calls
        if RefEagerTestBase._original_assert_close_func is None:
            RefEagerTestBase._original_assert_close_func = torch.testing.assert_close

        def counting_assert_close(*args: object, **kwargs: object) -> None:
            RefEagerTestBase._assert_close_count += 1
            return RefEagerTestBase._original_assert_close_func(*args, **kwargs)  # type: ignore[misc]

        torch.testing.assert_close = counting_assert_close

        # Patch self.assertRaises to count calls
        if RefEagerTestBase._original_assert_raises_func is None:
            RefEagerTestBase._original_assert_raises_func = self.assertRaises

        def counting_assert_raises(*args: object, **kwargs: object) -> object:
            RefEagerTestBase._assert_raises_count += 1
            return RefEagerTestBase._original_assert_raises_func(*args, **kwargs)  # type: ignore[misc]

        self.assertRaises = counting_assert_raises

        # Patch self.skipTest to count calls
        if RefEagerTestBase._original_skip_test_func is None:
            RefEagerTestBase._original_skip_test_func = self.skipTest

        def counting_skip_test(*args: object, **kwargs: object) -> object:
            RefEagerTestBase._skip_test_count += 1
            return RefEagerTestBase._original_skip_test_func(*args, **kwargs)  # type: ignore[misc]

        self.skipTest = counting_skip_test

        # Store the tracking context manager instance so we can check counts in tearDown
        self._run_ref_tracker = track_run_ref_calls()
        self._run_ref_count = self._run_ref_tracker.__enter__()

        # Patch pytest.raises to count calls
        if RefEagerTestBase._original_pytest_raises is None:  # pyright: ignore[reportAttributeAccessIssue]
            RefEagerTestBase._original_pytest_raises = pytest.raises

        def counting_pytest_raises(*args: object, **kwargs: object) -> object:
            """Wrapper for pytest.raises that counts calls but still runs the original logic."""
            RefEagerTestBase._assert_raises_count += 1
            assert RefEagerTestBase._original_pytest_raises is not None  # pyright: ignore[reportAttributeAccessIssue]
            return RefEagerTestBase._original_pytest_raises(*args, **kwargs)  # pyright: ignore[reportAttributeAccessIssue]

        pytest.raises = counting_pytest_raises  # type: ignore[assignment]

        # Patch self.assertTrue to count calls
        if RefEagerTestBase._original_assert_true_func is None:
            RefEagerTestBase._original_assert_true_func = self.assertTrue

        def counting_assert_true(*args: object, **kwargs: object) -> None:
            RefEagerTestBase._assert_true_count += 1
            return RefEagerTestBase._original_assert_true_func(*args, **kwargs)  # type: ignore[misc]

        self.assertTrue = counting_assert_true  # type: ignore[assignment]

        # Patch self.assertFalse to count calls
        if RefEagerTestBase._original_assert_false_func is None:
            RefEagerTestBase._original_assert_false_func = self.assertFalse

        def counting_assert_false(*args: object, **kwargs: object) -> None:
            RefEagerTestBase._assert_false_count += 1
            return RefEagerTestBase._original_assert_false_func(*args, **kwargs)  # type: ignore[misc]

        self.assertFalse = counting_assert_false  # type: ignore[assignment]

        # Patch self.assertGreater to count calls
        if RefEagerTestBase._original_assert_greater_func is None:
            RefEagerTestBase._original_assert_greater_func = self.assertGreater

        def counting_assert_greater(*args: object, **kwargs: object) -> None:
            RefEagerTestBase._assert_greater_count += 1
            return RefEagerTestBase._original_assert_greater_func(*args, **kwargs)  # type: ignore[misc]

        self.assertGreater = counting_assert_greater  # type: ignore[assignment]

    def tearDown(self) -> None:
        """Common teardown with assertion counting check."""
        # If not in ref eager mode, skip the teardown logic
        if not self._in_ref_eager_mode:
            super().tearDown()  # type: ignore[misc]
            return

        try:
            # Exit the run_ref tracker
            self._run_ref_tracker.__exit__(None, None, None)

            # Check if the test was skipped
            test_method = getattr(self, self._testMethodName, None)  # type: ignore[attr-defined]
            is_skipped = (
                test_method is not None
                and hasattr(test_method, "__unittest_skip__")
                and test_method.__unittest_skip__
            ) or RefEagerTestBase._skip_test_count > 0

            # Assert that either run_ref was called or the test was skipped
            if not is_skipped and self._run_ref_count[0] == 0:
                self.fail(  # type: ignore[attr-defined]
                    f"Test {self._testMethodName} did not call run_ref and was not skipped"  # pyright: ignore[reportAttributeAccessIssue]
                )

            if not is_skipped:
                # Check that either assert_close, assertRaises, skipTest, assertTrue, assertFalse, or assertGreater was called
                total_assertions = (
                    RefEagerTestBase._assert_close_count
                    + RefEagerTestBase._assert_raises_count
                    + RefEagerTestBase._skip_test_count
                    + RefEagerTestBase._assert_true_count
                    + RefEagerTestBase._assert_false_count
                    + RefEagerTestBase._assert_greater_count
                )
                # Need to use the original assertGreater to avoid recursion
                if RefEagerTestBase._original_assert_greater_func is not None:
                    RefEagerTestBase._original_assert_greater_func(  # type: ignore[misc]
                        total_assertions,
                        0,
                        f"Test {self._testMethodName} did not call torch.testing.assert_close, assertRaises, skipTest, assertTrue, assertFalse, or assertGreater",  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
                    )
                else:
                    # Fallback if original not available
                    assert total_assertions > 0, (
                        f"Test {self._testMethodName} did not call any assertion methods"  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
                    )
        finally:
            # Restore the original assert_close function
            if RefEagerTestBase._original_assert_close_func is not None:
                torch.testing.assert_close = (
                    RefEagerTestBase._original_assert_close_func
                )

            # Restore the original assertRaises function
            if RefEagerTestBase._original_assert_raises_func is not None:
                self.assertRaises = RefEagerTestBase._original_assert_raises_func

            # Restore the original skipTest function
            if RefEagerTestBase._original_skip_test_func is not None:
                self.skipTest = RefEagerTestBase._original_skip_test_func

            # Restore the original pytest.raises function
            if RefEagerTestBase._original_pytest_raises is not None:  # pyright: ignore[reportAttributeAccessIssue]
                pytest.raises = RefEagerTestBase._original_pytest_raises  # pyright: ignore[reportAttributeAccessIssue]

            # Restore the original assertTrue function
            if RefEagerTestBase._original_assert_true_func is not None:
                self.assertTrue = RefEagerTestBase._original_assert_true_func

            # Restore the original assertFalse function
            if RefEagerTestBase._original_assert_false_func is not None:
                self.assertFalse = RefEagerTestBase._original_assert_false_func

            # Restore the original assertGreater function
            if RefEagerTestBase._original_assert_greater_func is not None:
                self.assertGreater = RefEagerTestBase._original_assert_greater_func

            super().tearDown()  # type: ignore[misc]

    # NOTE: We no-op these methods because they commonly check behaviors that are not relevant in ref eager mode.
    # Instead, we solely rely on the unit test's `torch.testing.assert_close` and `assertRaises` checks to ensure ref eager mode's correctness.
    def assertExpectedJournal(self, value: str) -> None:
        if not self._in_ref_eager_mode:
            super().assertExpectedJournal(value)  # type: ignore[misc]

    def assertIn(
        self, member: object, container: object, msg: str | None = None
    ) -> None:
        if not self._in_ref_eager_mode:
            super().assertIn(member, container, msg)  # type: ignore[misc]

    def assertNotIn(
        self, member: object, container: object, msg: str | None = None
    ) -> None:
        if not self._in_ref_eager_mode:
            super().assertNotIn(member, container, msg)  # type: ignore[misc]

    def assertTrueIfInNormalMode(self, condition: bool, msg: str | None = None) -> None:
        if not self._in_ref_eager_mode:
            self.assertTrue(condition, msg)  # type: ignore[attr-defined]

    def assertEqualCode(self, first: str, second: str, msg: str | None = None) -> None:
        if not self._in_ref_eager_mode:
            super().assertEqual(first, second, msg)  # type: ignore[misc]

    def assertNotEqualCode(
        self, first: str, second: str, msg: str | None = None
    ) -> None:
        if not self._in_ref_eager_mode:
            super().assertNotEqual(first, second, msg)  # type: ignore[misc]

    def getUserDefinedTunable(
        self, user_defined_tunables: dict[str, object], key: str
    ) -> object | None:
        """Look up a specific value via key from user defined tunables. Returns None in ref mode."""
        if self._in_ref_eager_mode:
            return None
        return user_defined_tunables.get(key)

    def assertIsInstance(
        self, obj: object, cls: type | tuple[type, ...], msg: str | None = None
    ) -> None:
        if not self._in_ref_eager_mode:
            super().assertIsInstance(obj, cls, msg)  # type: ignore[misc]


def import_path(filename: Path) -> types.ModuleType:
    module_name = f"{__name__}.{filename.stem}"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, filename)  # pyright: ignore[reportAttributeAccessIssue]
        assert spec is not None
        module = importlib.util.module_from_spec(spec)  # pyright: ignore[reportAttributeAccessIssue]
        assert spec.loader is not None
        spec.loader.exec_module(module)
        sys.modules[module_name] = module
    return sys.modules[module_name]


def code_and_output(
    fn: Kernel,
    args: tuple[object, ...],
    **kwargs: object,
) -> tuple[str, object]:
    bound = fn.bind(args)
    if is_ref_mode_enabled(bound.kernel.settings):
        if kwargs:
            config = Config(**kwargs)  # pyright: ignore[reportArgumentType]
            bound._config = config
        result = fn(*args)
        # Return the original kernel source code
        code = inspect.getsource(fn.fn)
        return code, result

    if kwargs:
        config = Config(
            **kwargs  # pyright: ignore[reportArgumentType]
        )
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
    bwd: bool = False,
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
        bwd: Whether to also test backward pass (default: False)
    """
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"  # type: ignore[reportAttributeAccessIssue]
    except AttributeError:  # No cudnn available
        torch.set_float32_matmul_precision("high")  # older deprecated API

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
            torch.testing.assert_close(
                func(*args).to(torch.float32),
                expected.to(torch.float32),
                rtol=rtol,
                atol=atol,
            )

    # Test backward pass
    if bwd:
        # Find tensors that require gradients in args
        grad_tensors = []
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.requires_grad:
                grad_tensors.append(arg)

        # Ensure we have tensors to check
        assert len(grad_tensors) > 0, (
            "BWD: No tensors with requires_grad=True found. "
            "Check that input tensors have requires_grad set."
        )

        # Run baseline backward pass
        baseline_out = first_baseline_func(*args)
        grad_output = torch.randn_like(baseline_out)

        # Save original gradients
        baseline_out.backward(grad_output, retain_graph=True)
        baseline_grads = [
            t.grad.clone() if t.grad is not None else None for t in grad_tensors
        ]

        # Ensure at least one gradient was computed
        has_gradient = any(g is not None for g in baseline_grads)
        assert has_gradient, (
            "BWD: No gradients were computed. All tensors have grad=None. "
            "Check that backward was called and tensors require gradients."
        )

        # Clear gradients
        for t in grad_tensors:
            t.grad = None

        # Test each implementation
        for name, func in {**kernels, **baselines}.items():
            if name != first_baseline_name:
                # Run backward
                out = func(*args)
                out.backward(grad_output, retain_graph=True)

                # Collect implementation gradients
                impl_grads = [t.grad for t in grad_tensors]

                # Ensure same number of grad tensors
                assert len(impl_grads) == len(baseline_grads), (
                    f"BWD: Mismatch in number of grad tensors for {name}: "
                    f"{len(impl_grads)} vs {len(baseline_grads)}"
                )

                # Compare each tensor's gradient
                for i, (tensor, baseline_grad) in enumerate(
                    zip(grad_tensors, baseline_grads, strict=False)
                ):
                    # Check gradient existence
                    assert (baseline_grad is None) == (tensor.grad is None), (
                        f"BWD: Gradient existence mismatch for tensor {i} in {name}: "
                        f"baseline has grad={baseline_grad is not None}, "
                        f"impl has grad={tensor.grad is not None}"
                    )

                    if baseline_grad is not None:
                        torch.testing.assert_close(
                            tensor.grad.to(torch.float32),
                            baseline_grad.to(torch.float32),
                            rtol=rtol,
                            atol=atol,
                            msg=f"BWD: Gradient mismatch for tensor {i} with shape {tensor.shape} in {name}",
                        )

                # Clear gradients for next test
                for t in grad_tensors:
                    t.grad = None

    # Benchmark all functions
    all_benchmarks = {**kernels, **baselines}
    bench_fns = [functools.partial(fn, *args) for fn in all_benchmarks.values()]
    repeat = compute_repeat(bench_fns[0])
    timings = interleaved_bench(bench_fns, repeat=repeat, desc="Benchmarking")
    all_times = dict(zip(all_benchmarks.keys(), timings, strict=True))
    best_baseline_time = min(all_times[name] for name in baselines)  # pyright: ignore[reportArgumentType]

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
    expected: object,
    fn_name: str | None = None,
    skip_accuracy: bool = False,
    static_shapes: bool | None = None,
    atol: float = 1e-1,
    rtol: float = 1e-2,
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

    if not skip_accuracy:
        # Use tree_map to apply assert_close to all tensor pairs
        def assert_close_fn(got: object, exp: object) -> None:
            # Skip if expected is None (i.e. we don't care what the actual value is)
            if exp is None:
                return
            # Both None is OK
            if got is None and exp is None:
                return
            assert isinstance(got, torch.Tensor) and isinstance(exp, torch.Tensor), (
                f"Type mismatch: got {type(got)}, expected {type(exp)}"
            )
            torch.testing.assert_close(
                got.to(torch.float32),
                exp.to(torch.float32),
                atol=atol,
                rtol=rtol,
            )

        tree_map(assert_close_fn, result, expected)
    return code


class AssertExpectedJournal:
    """
    Manages a <testfile>.expected file that contains expected output for TestCase.assertExpectedJournal() calls.

    This replaces the previous `expecttest` assertExpectedInline approach by storing expected output
    in external .expected files rather than inline strings in test files. This provides better
    organization and avoids cluttering test files with large code blocks.

    The .expected file format uses sections like:
    --- assertExpectedJournal(TestClass.test_method)
    expected output here

    --- assertExpectedJournal(TestClass.test_method)
    second expected output for same test

    Environment variable EXPECTTEST_ACCEPT=1 can be used to update expected outputs.
    """

    def __init__(self, cls: type[TestCase]) -> None:
        pyfile = os.path.abspath(inspect.getfile(cls))
        assert "/test/" in pyfile
        assert pyfile.endswith(".py")
        self.filename: Path = Path(pyfile[:-3] + ".expected")
        self._cache: dict[str, list[str]] | None = None
        self._current_id: str | None = None
        self._current_index: int = 0

    @property
    def cache(self) -> dict[str, list[str]]:
        if self._cache is None:
            return self.reload()
        return self._cache

    def reload(self) -> dict[str, list[str]]:
        if self.filename.exists():
            data = self.filename.read_text()
        else:
            data = ""
        result = collections.defaultdict(list)
        for name, expected in re.findall(
            r"--- assertExpectedJournal\(([^)]*)\)\n(.*?)(?=^--- assertExpectedJournal\(|\Z)",
            data,
            re.MULTILINE | re.DOTALL,
        ):
            result[name].append(expected.strip())
        self._cache = result
        return result

    def save(self) -> None:
        tmp = f"{self.filename}.tmp{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(
                f"This file is automatically generated by assertExpectedJournal calls in {self.filename.stem}.py.\n"
                "Update expected outputs by running tests with the EXPECTTEST_ACCEPT=1 environment variable set.\n\n"
            )
            for name, expected_values in sorted(
                self.cache.items(), key=operator.itemgetter(0)
            ):
                f.writelines(
                    f"--- assertExpectedJournal({name})\n{expected}\n\n"
                    for expected in expected_values
                )
            # Remove the last newline to play nicer with some people's editors
            f.truncate(f.tell() - 1)
        os.rename(tmp, self.filename)

    @staticmethod
    def normalize_id(test_id: str) -> str:
        match = re.search(r"\b([^.]+\.[^.]+)$", test_id)
        assert match, f"Test ID '{test_id}' does not match expected format"
        return match.group(1)

    @staticmethod
    def normalize_tensor_descriptors(code: str) -> str:
        return code.replace(
            get_tensor_descriptor_fn_name(), "tl.make_tensor_descriptor"
        )

    @staticmethod
    def normalize_device_name(code: str) -> str:
        """
        convert device='cuda:0' or device(type='cuda', index=0) etc to device=DEVICE
        """
        # device='cuda:0'
        reg_pattern_for_device_str = r"device\s*=\s*['\"][^'\"]+['\"]"
        normalized_code = re.sub(reg_pattern_for_device_str, "device=DEVICE", code)
        # device(type='cuda', index=0)
        reg_pattern_for_torch_device = (
            r"device\s*\(type\s*=\s*['\"][^'\"]+['\"][^'\"\)]*\)"
        )
        return re.sub(reg_pattern_for_torch_device, "device=DEVICE", normalized_code)

    @staticmethod
    def normalize_codegen_variants(code: str) -> str:
        # TODO(oulgen): Remove when PyTorch 2.10 becomes stable

        # Remove libdevice import line if present
        code = re.sub(
            r"^\s*from torch\._inductor\.runtime\.triton_compat import libdevice\s*\n?",
            "",
            code,
            flags=re.MULTILINE,
        )

        # Normalize sqrt variants
        # libdevice.sqrt( -> tl.sqrt_rn(
        code = re.sub(r"\blibdevice\.sqrt\s*\(", "tl.sqrt_rn(", code)
        # tl.sqrt( -> tl.sqrt_rn(
        code = re.sub(r"\btl\.sqrt\s*\(", "tl.sqrt_rn(", code)

        # Normalize rsqrt variants
        # libdevice.rsqrt( -> tl.rsqrt(
        code = re.sub(r"\blibdevice\.rsqrt\s*\(", "tl.rsqrt(", code)

        total_num_triton_helpers_replacements = 0

        # Normalize maximum variants
        # tl.maximum(a, b, tl.PropagateNan.ALL) -> triton_helpers.maximum(a, b)
        code, num_replacements = re.subn(
            r"\btl\.maximum\s*\(([^,]+),\s*([^,]+),\s*tl\.PropagateNan\.ALL\s*\)",
            r"triton_helpers.maximum(\1, \2)",
            code,
        )
        total_num_triton_helpers_replacements += num_replacements

        # Normalize minimum variants
        # tl.minimum(a, b, tl.PropagateNan.ALL) -> triton_helpers.minimum(a, b)
        code, num_replacements = re.subn(
            r"\btl\.minimum\s*\(([^,]+),\s*([^,]+),\s*tl\.PropagateNan\.ALL\s*\)",
            r"triton_helpers.minimum(\1, \2)",
            code,
        )
        total_num_triton_helpers_replacements += num_replacements

        triton_helpers_import = "from torch._inductor.runtime import triton_helpers"
        if (
            total_num_triton_helpers_replacements > 0
            and triton_helpers_import not in code
        ):
            # Insert right after `import triton.language as tl`
            code = re.sub(
                r"(^import triton\.language as tl$)",
                rf"\1\n{triton_helpers_import}",
                code,
                count=1,
                flags=re.MULTILINE,
            )

        return code

    @staticmethod
    def normalize_source_comment_structure(code: str) -> str:
        pattern = re.compile(
            r"^(?P<indent>\s*)# src\[(?P<prefix>[^:\]]+:)(?P<start>\d+|N)(?:-(?P<end>\d+|N))?]: (?P<text>.*?)(?P<newline>\r?\n|$)",
            flags=re.MULTILINE,
        )

        def replacer(match: re.Match[str]) -> str:
            text = match.group("text").rstrip()
            if not text.strip():
                return ""
            indent = match.group("indent")
            prefix = match.group("prefix")
            suffix = "N-N" if match.group("end") is not None else "N"
            newline = match.group("newline")
            return f"{indent}# src[{prefix}{suffix}]: {text}{newline}"

        return pattern.sub(replacer, code)

    @classmethod
    def normalize_code(cls, code: str) -> str:
        code = cls.normalize_source_comment_structure(code)
        code = cls.normalize_tensor_descriptors(code)
        code = cls.normalize_device_name(code)
        code = cls.normalize_codegen_variants(code)
        return code.strip()

    def lookup(self, test_id: str, value: str) -> tuple[str, str]:
        test_id = self.normalize_id(test_id)
        if self._current_id != test_id:
            self._current_id = test_id
            self._current_index = 0

        expected_values = self.cache[test_id]
        if self._current_index < len(expected_values):
            expected = expected_values[self._current_index]
        else:
            assert self._current_index == len(expected_values)
            expected_values.append("")
            expected = ""

        # Normalize both actual and expected for robust comparisons
        value = self.normalize_code(value)
        expected = self.normalize_code(expected)

        if value != expected and os.environ.get("EXPECTTEST_ACCEPT", "0") not in {
            "0",
            "false",
            "False",
            "",
        }:
            expected_values[self._current_index] = value
            # Reload to play nicer with other processes
            self.reload()[test_id][:] = expected_values
            self.save()
            expected = value
            print(
                f"Expected output for {test_id} updated: {len(expected)} => {len(value)} bytes",
                file=sys.stderr,
            )
        self._current_index += 1
        return value, expected


class RefEagerTestDisabled:
    """Base class for test classes that should be skipped when ref eager mode is enabled."""

    def setUp(self) -> None:
        """Skip test if ref eager mode is enabled."""
        super().setUp()  # type: ignore[misc]
        if os.environ.get("HELION_INTERPRET") == "1":
            self.skipTest("Test class disabled in ref eager mode")  # type: ignore[attr-defined]


class TestCase(unittest.TestCase):
    maxDiff = 16384

    @classmethod
    def setUpClass(cls) -> None:
        cls._expected_journal = AssertExpectedJournal(cls)
        super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        del cls._expected_journal

    def setUp(self) -> None:
        super().setUp()
        self._test_stack = contextlib.ExitStack()

        from torch._inductor.utils import fresh_cache

        self._test_stack.enter_context(fresh_cache())

        counters.clear()

    def tearDown(self) -> None:
        super().tearDown()
        self._test_stack.close()

    def assertExpectedJournal(self, value: str) -> None:
        """
        Assert that the given value matches the expected output stored in <testfile>.expected.

        This method replaces assertExpectedInline for code generation tests. Instead of storing
        expected output as inline strings in test files, it uses external .expected files for
        better organization.

        Args:
            value: The actual output to compare (usually generated Triton code)

        Raises:
            AssertionError: If value doesn't match expected output

        Note:
            Use EXPECTTEST_ACCEPT=1 environment variable to update expected outputs.
        """
        value, expected = self._expected_journal.lookup(self.id(), value)
        self.assertMultiLineEqual(
            value,
            expected,
            msg="To accept the new output, re-run test with env EXPECTTEST_ACCEPT=1",
        )
