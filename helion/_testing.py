from __future__ import annotations

import collections
import importlib
import inspect
import operator
import os
from pathlib import Path
import re
import sys
from typing import TYPE_CHECKING
from typing import Callable
import unittest

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
        os.rename(tmp, self.filename)

    @staticmethod
    def normalize_id(test_id: str) -> str:
        match = re.search(r"\b([^.]+\.[^.]+)$", test_id)
        assert match, f"Test ID '{test_id}' does not match expected format"
        return match.group(1)

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

        value = value.strip()
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
