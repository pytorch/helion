"""Functions in autotune arguments cross the benchmark-subprocess boundary by code.

Standard pickling refuses lambdas and closures (``matmul``'s ``epilogue``
default, an ``addmm`` epilogue capturing the bias); the provider used to fall
back to benchmarking in-process, which leaves the process in a state where
every later CUDA graph runs the kernel's nodes slower.  ``make_portable_args``
serializes such functions by their code object and closure values and
``load_trusted_kernel_args`` rebuilds real functions in the worker.
"""

from __future__ import annotations

import dataclasses
import functools
import os
import pickle
import sys
import tempfile
import threading
import types
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

from helion.autotuner import benchmark_job
from helion.autotuner import kernel_args
from helion.autotuner.accuracy import assert_close
from helion.autotuner.benchmark_job import AccuracyCheckJob
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.kernel_args import PortableFunction
from helion.autotuner.kernel_args import _clone_args
from helion.autotuner.kernel_args import make_portable_args
from helion.autotuner.kernel_args import resolve_portable_args
from helion.runtime.settings import Settings

ALPHA = 2.0


def _module_level(acc: torch.Tensor, tile: tuple[object, ...]) -> torch.Tensor:
    return acc * ALPHA


def _make_closure(bias: torch.Tensor, alpha: float, beta: float):
    def addmm_epilogue(acc: torch.Tensor, tile: tuple[object, ...]) -> torch.Tensor:
        return alpha * acc + beta * bias[tile[0], tile[1]]

    return addmm_epilogue


def _decorated(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs) * ALPHA

    return wrapper


@dataclasses.dataclass
class _ScaleInPlace:
    """``b *= a`` with ``epilogue(b)`` as the output; ``extra`` corrupts the
    in-place result (a candidate whose output is right but whose mutation
    is wrong, which only the post-call argument comparison can catch)."""

    extra: float = 0.0

    def __call__(
        self, a: torch.Tensor, b: torch.Tensor, epilogue: object
    ) -> torch.Tensor:
        b.mul_(a)
        out = cast("torch.Tensor", epilogue(b.clone()))  # pyrefly: ignore
        b.add_(self.extra)
        return out


def _roundtrip(args: tuple[object, ...]) -> tuple[object, ...]:
    portable = make_portable_args(args)
    assert portable is not None
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "args.pt")
        torch.save(portable, path)
        kernel_args.load_trusted_kernel_args.cache_clear()
        loaded = kernel_args.load_trusted_kernel_args(path)
    return tuple(loaded)


def test_lambda_default_roundtrips() -> None:
    epilogue = lambda acc, tile: acc  # noqa: E731
    x = torch.ones(4, 4)
    args = (x, x, epilogue)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            torch.save(args, os.path.join(tmp, "plain.pt"))
        except (pickle.PicklingError, AttributeError, TypeError):
            pass
        else:
            raise AssertionError("a lambda in the args must defeat plain pickling")
    a, b, fn = _roundtrip(args)
    assert torch.equal(a, x) and torch.equal(b, x)
    assert fn is not epilogue and fn.__closure__ is None
    assert fn.__code__.co_code == epilogue.__code__.co_code
    assert fn(torch.tensor([3.0]), ()) == torch.tensor([3.0])


def test_closure_cells_defaults_and_nested_functions_roundtrip() -> None:
    bias = torch.arange(16.0).reshape(4, 4)
    epilogue = _make_closure(bias, 0.5, 2.0)
    inner = lambda v: v + 1  # noqa: E731

    def outer(acc: torch.Tensor, scale: float = 3.0, *, shift: float = 1.0):
        return inner(acc) * scale + shift

    (fn, outer_fn) = _roundtrip((epilogue, outer))
    # The generated host code reads ``epilogue.__closure__[i].cell_contents``;
    # the rebuilt function has the same cells in the same order.
    assert [type(c.cell_contents) for c in fn.__closure__] == [
        type(c.cell_contents) for c in epilogue.__closure__
    ]
    rebuilt_bias = next(
        c.cell_contents
        for c in fn.__closure__
        if isinstance(c.cell_contents, torch.Tensor)
    )
    assert torch.equal(rebuilt_bias, bias)
    acc = torch.ones(2, 2)
    expected = epilogue(acc, (slice(0, 2), slice(0, 2)))
    assert torch.equal(fn(acc, (slice(0, 2), slice(0, 2))), expected)
    assert fn.__qualname__ == epilogue.__qualname__
    # Defaults, keyword-only defaults and a function captured in a closure.
    assert outer_fn.__defaults__ == (3.0,) and outer_fn.__kwdefaults__ == {"shift": 1.0}
    assert torch.equal(outer_fn(torch.zeros(1)), torch.tensor([4.0]))
    assert torch.equal(outer_fn(torch.zeros(1), 2.0, shift=0.0), torch.tensor([2.0]))


def test_module_level_function_keeps_its_globals() -> None:
    (fn,) = _roundtrip((_module_level,))
    assert fn.__module__ == __name__ and fn.__globals__ is _module_level.__globals__
    assert torch.equal(fn(torch.ones(1), ()), torch.tensor([ALPHA]))


def test_arguments_without_functions_are_left_alone() -> None:
    x = torch.ones(2)
    assert make_portable_args((x, 3, "s")) is None
    args = (x, 3)
    assert resolve_portable_args(args) is args


def test_portable_function_is_a_plain_picklable_record() -> None:
    portable = make_portable_args((lambda a: a,))
    assert portable is not None and isinstance(portable[0], PortableFunction)
    assert pickle.loads(pickle.dumps(portable[0])) == portable[0]


def test_functions_whose_globals_are_not_their_module_are_refused() -> None:
    # Globals are not shipped: the worker rebuilds the function with the
    # namespace of ``fn.__module__``.  A ``functools.wraps`` wrapper carries
    # the wrapped function's module but the decorator's globals (``ALPHA``
    # here would resolve in the wrong module), an exec-created function has
    # its own dict; both keep the in-process fallback.
    foreign = types.ModuleType("test_kernel_args_portable_foreign")
    exec("def epilogue(acc, tile):\n    return acc", foreign.__dict__)
    sys.modules[foreign.__name__] = foreign
    try:
        wrapped = _decorated(foreign.epilogue)
        assert wrapped.__module__ == foreign.__name__
        assert wrapped.__globals__ is not vars(foreign)
        assert make_portable_args((torch.ones(1), wrapped)) is None
        # The module-level function itself is portable.
        assert make_portable_args((foreign.epilogue,)) is not None
    finally:
        del sys.modules[foreign.__name__]
    namespace: dict[str, object] = {"__name__": __name__}
    exec("def epilogue(acc, tile):\n    return acc", namespace)
    assert make_portable_args((namespace["epilogue"],)) is None


def test_private_finalist_copies_clone_closure_and_default_tensors() -> None:
    # Finalists are timed back to back on private argument copies so one
    # candidate's L2-resident input is not the next one's hit; a tensor an
    # epilogue captures (addmm's bias) or takes as a default is an operand
    # of the kernel too and gets its own copy per slot, aliasing preserved.
    bias = torch.arange(16.0).reshape(4, 4)
    alias_of_bias = bias.view(16)
    epilogue = _make_closure(bias, 0.5, 2.0)

    def with_default(acc: torch.Tensor, scale: torch.Tensor = bias) -> torch.Tensor:
        return acc * scale

    plain = lambda acc, tile: acc  # noqa: E731
    args = (torch.ones(4, 4), alias_of_bias, epilogue, with_default, plain)
    portable = make_portable_args(args)
    assert portable is not None
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "args.pt")
        torch.save(portable, path)
        kernel_args.load_trusted_kernel_args.cache_clear()
        kernel_args.load_private_kernel_args.cache_clear()
        try:
            shared = kernel_args.load_trusted_kernel_args(path)
            slots = [kernel_args.load_private_kernel_args(path, i) for i in range(2)]
        finally:
            kernel_args.load_private_kernel_args.cache_clear()

    def closure_bias(fn: object) -> torch.Tensor:
        return next(
            c.cell_contents
            for c in fn.__closure__  # pyrefly: ignore
            if isinstance(c.cell_contents, torch.Tensor)
        )

    biases = [closure_bias(slot[2]) for slot in slots]
    defaults = [slot[3].__defaults__[0] for slot in slots]  # pyrefly: ignore
    storages = {
        t.untyped_storage().data_ptr()
        for t in (closure_bias(shared[2]), *biases, *defaults)
    }
    assert len(storages) == 3, "each slot owns its own copy of the captured bias"
    for slot, slot_bias, slot_default in zip(slots, biases, defaults, strict=True):
        assert torch.equal(slot_bias, bias) and torch.equal(slot_default, bias)
        # The positional view of the bias aliases the slot's copy, as it did
        # in the parent, and the function with no tensor is left alone.
        assert (
            slot[1].untyped_storage().data_ptr()  # pyrefly: ignore
            == slot_bias.untyped_storage().data_ptr()
            == slot_default.untyped_storage().data_ptr()
        )
        assert slot[4] is shared[4]
        acc = torch.ones(2, 2)
        expected = epilogue(acc, (slice(0, 2), slice(0, 2)))
        assert torch.equal(slot[2](acc, (slice(0, 2), slice(0, 2))), expected)  # pyrefly: ignore
        assert torch.equal(slot[3](torch.ones(4, 4)), bias)  # pyrefly: ignore
        assert slot[2].__qualname__ == epilogue.__qualname__  # pyrefly: ignore


def test_accuracy_comparison_tolerates_function_leaves() -> None:
    # The post-call arguments of a mutating kernel are compared against the
    # baseline's; a callable argument is not something the kernel writes to
    # and torch.testing.assert_close has no comparison pair for functions.
    epilogue = _make_closure(torch.ones(1), 1.0, 1.0)
    (portable,) = make_portable_args((epilogue,))  # pyrefly: ignore
    for leaf in (epilogue, portable, _ScaleInPlace()):
        assert_close((torch.ones(2), leaf), (torch.ones(2), leaf), atol=0, rtol=0)
    assert_close((torch.ones(2), epilogue), (torch.ones(2), portable), atol=0, rtol=0)
    with pytest.raises(AssertionError, match="Tensor-likes are not"):
        assert_close(
            (torch.ones(2), epilogue), (torch.zeros(2), epilogue), atol=0, rtol=0
        )
    with pytest.raises(AssertionError, match="Leaf type mismatch"):
        assert_close((torch.ones(2), epilogue), (torch.ones(2), 3.0), atol=0, rtol=0)


def test_accuracy_job_checks_mutated_args_around_a_callable() -> None:
    # The whole worker-side path for an in-place kernel with an epilogue: the
    # arguments and the baseline's post-call arguments are both serialized by
    # code, rebuilt in the job, and the mutated tensor is still checked.
    scale = torch.tensor([3.0])

    def epilogue(acc: torch.Tensor) -> torch.Tensor:
        return acc * scale

    a, b = torch.tensor([2.0]), torch.tensor([5.0])
    args = (a, b, epilogue)
    baseline_output = _ScaleInPlace()(a, b.clone(), epilogue)
    post_args = _clone_args((a, b * a, epilogue), None, idx_to_clone=[1])

    def run(fn: object) -> tuple[object, torch.Tensor]:
        with tempfile.TemporaryDirectory() as tmp:
            paths = {
                name: os.path.join(tmp, f"{name}.pt")
                for name in ("args", "baseline", "post_args")
            }
            torch.save(make_portable_args(args), paths["args"])
            torch.save(baseline_output, paths["baseline"])
            torch.save(make_portable_args(post_args), paths["post_args"])
            kernel_args.load_trusted_kernel_args.cache_clear()
            benchmark_job._load_trusted_baseline_output.cache_clear()
            with patch.object(benchmark_job, "_load_compiled_fn", return_value=fn):
                result = AccuracyCheckJob(
                    fn_spec=cast("benchmark_job.SerializedCompiledFunction", object()),
                    args_path=paths["args"],
                    baseline_path=paths["baseline"],
                    atol=0.0,
                    rtol=0.0,
                    mutated_arg_indices=(1,),
                    baseline_post_args_path=paths["post_args"],
                )()
            cached_b = cast(
                "torch.Tensor", kernel_args.load_trusted_kernel_args(paths["args"])[1]
            )
            kernel_args.load_trusted_kernel_args.cache_clear()
            benchmark_job._load_trusted_baseline_output.cache_clear()
        return result, cached_b

    result, cached_b = run(_ScaleInPlace())
    assert result.ok, result.message  # pyrefly: ignore
    # The job ran on a clone: the worker's shared arguments keep their values.
    torch.testing.assert_close(cached_b, b)
    result, _ = run(_ScaleInPlace(extra=1.0))
    assert not result.ok and "Tensor-likes are not" in result.message  # pyrefly: ignore


def _provider_for_setup(args: tuple[object, ...]) -> LocalBenchmarkProvider:
    provider = LocalBenchmarkProvider.__new__(LocalBenchmarkProvider)
    provider.settings = Settings(
        autotune_benchmark_subprocess=True, autotune_precompile="fork"
    )
    provider.log = Mock()
    provider.args = args
    provider.kernel = SimpleNamespace(
        supports_subprocess_benchmark=lambda: True,
        env=SimpleNamespace(device=torch.device("cpu")),
    )
    provider.config_spec = SimpleNamespace(
        backend=SimpleNamespace(name="cute", get_do_bench=lambda: None)
    )
    provider.mutated_arg_indices = []
    provider._args_unpicklable = False
    provider._subprocess_wrapper_unloadable = False
    provider._precompile_tmpdir = None
    provider._precompile_args_path = None
    provider._precompile_baseline_path = None
    provider._precompile_baseline_post_args_path = None
    provider._baseline_output = torch.ones(4, 4)
    provider._baseline_post_args = None
    return provider


def test_setup_serializes_callable_args_for_the_subprocess() -> None:
    bias = torch.ones(4, 4)
    args = (torch.ones(4, 4), torch.ones(4, 4), _make_closure(bias, 1.0, 1.0))
    provider = _provider_for_setup(args)
    provider.setup()
    try:
        assert not provider._args_unpicklable
        assert provider._precompile_args_path is not None
        kernel_args.load_trusted_kernel_args.cache_clear()
        loaded = kernel_args.load_trusted_kernel_args(provider._precompile_args_path)
        assert callable(loaded[2]) and torch.equal(
            loaded[2](torch.ones(2, 2), (slice(0, 2), slice(0, 2))),
            torch.full((2, 2), 2.0),
        )
        provider.log.warning.assert_not_called()
    finally:
        provider._precompile_tmpdir.cleanup()


def test_setup_serializes_the_baseline_post_args_of_a_mutating_kernel() -> None:
    # An in-place kernel with an epilogue (``matmul(x, y, out, epilogue=...)``):
    # the baseline's post-call arguments hold the same function leaf as the
    # arguments and take the same path; with the default settings this used
    # to raise out of setup() and abort the autotune.
    bias = torch.ones(4, 4)
    args = (torch.ones(4, 4), torch.ones(4, 4), _make_closure(bias, 1.0, 1.0))
    provider = _provider_for_setup(args)
    provider.mutated_arg_indices = [0]
    provider._baseline_post_args = _clone_args(args, None, idx_to_clone=[0])
    assert provider._subprocess_accuracy_check_enabled()
    provider.setup()
    try:
        assert not provider._args_unpicklable
        assert provider._precompile_args_path is not None
        assert provider._precompile_baseline_path is not None
        assert provider._precompile_baseline_post_args_path is not None
        benchmark_job._load_trusted_baseline_output.cache_clear()
        post_args = benchmark_job._load_trusted_baseline_output(
            provider._precompile_baseline_post_args_path
        )
        benchmark_job._load_trusted_baseline_output.cache_clear()
        kernel_args.load_trusted_kernel_args.cache_clear()
        loaded = kernel_args.load_trusted_kernel_args(provider._precompile_args_path)
        kernel_args.load_trusted_kernel_args.cache_clear()
        assert callable(post_args[2]) and callable(loaded[2])  # pyrefly: ignore
        # What the worker's accuracy job does with the two files.
        assert_close(loaded, post_args, atol=0.0, rtol=0.0)
        provider.log.warning.assert_not_called()
    finally:
        provider._precompile_tmpdir.cleanup()


def test_setup_falls_back_in_process_when_the_post_args_cannot_be_saved() -> None:
    args = (
        torch.ones(4, 4),
        torch.ones(4, 4),
        _make_closure(torch.ones(4, 4), 1.0, 1.0),
    )
    provider = _provider_for_setup(args)
    provider.mutated_arg_indices = [0]
    provider._baseline_post_args = (torch.ones(4, 4), threading.Lock(), args[2])
    provider.setup()
    try:
        # The arguments file is written (spawn-mode precompile needs it); the
        # benchmark and accuracy check stay in this process, as for arguments
        # that cannot be written at all.
        assert provider._precompile_args_path is not None
        assert provider._precompile_baseline_post_args_path is None
        assert provider._args_unpicklable
        assert not provider._subprocess_accuracy_check_enabled()
        provider.log.warning.assert_called_once()
    finally:
        provider._precompile_tmpdir.cleanup()


def test_setup_keeps_the_in_process_fallback_for_other_unpicklable_args() -> None:
    args = (
        torch.ones(2),
        threading.Lock(),
    )  # not a function: nothing to serialize by code
    provider = _provider_for_setup(args)
    provider.setup()
    try:
        assert provider._args_unpicklable
        assert provider._precompile_args_path is None
        provider.log.warning.assert_called_once()
    finally:
        provider._precompile_tmpdir.cleanup()
