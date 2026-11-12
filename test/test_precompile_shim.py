from __future__ import annotations

import ast
from collections import OrderedDict
import contextlib
import functools
import gc
import importlib
import logging
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
import unittest
from unittest.mock import ANY
from unittest.mock import Mock
from unittest.mock import patch
import weakref

from packaging.version import Version
import pytest
import torch
from torch._inductor.runtime.triton_compat import PTXASError

from helion._argument_device import _find_argument_device
from helion._testing import onlyBackends
from helion._testing.search_harness import mock_search
from helion._utils import triton_is_available
from helion.autotuner import benchmark_provider as benchmark_provider_module
from helion.autotuner import precompile_future as precompile_future_module
from helion.autotuner.benchmark_provider import UnmeasuredConfigValue
from helion.autotuner.logger import SUPPRESSED_TRITON_CODE_MSG
from helion.autotuner.logger import AutotuningLogger
from helion.autotuner.logger import classify_triton_exception
from helion.autotuner.precompile_future import PrecompileContext
from helion.autotuner.precompile_future import PrecompileFuture
from helion.autotuner.precompile_future import SerializedCompiledFunction
from helion.autotuner.precompile_future import _ExtractedLaunchArgs
from helion.autotuner.precompile_future import _run_kernel_in_subprocess_fork
from helion.autotuner.precompile_future import _run_kernel_in_subprocess_spawn
from helion.runtime.config import Config
from helion.runtime.precompile_shim import _device_probe_values
from helion.runtime.precompile_shim import _set_helion_compilation_success
from helion.runtime.precompile_shim import make_precompiler
from helion.runtime.settings import Settings

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Sequence

    from helion.autotuner.benchmark_provider import BenchmarkResult

pytestmark = pytest.mark.cpu_only

_HAS_FORK = "fork" in mp.get_all_start_methods()


def test_precompile_warning_formats_message_and_preserves_exception(capsys) -> None:
    decorator = "@helion.kernel(config=Config(block_sizes=[32]))"
    settings = Settings(autotune_precompile="fork")
    kernel = Mock()
    kernel.format_kernel_decorator.return_value = decorator
    context = PrecompileContext(settings, AutotuningLogger(settings), kernel, (), 1)
    with (
        patch("helion.autotuner.precompile_future.maybe_dump_triton_failure"),
        patch("helion.autotuner.precompile_future.log_generated_triton_code_debug"),
        pytest.raises(RuntimeError, match="precompile failed"),
    ):
        PrecompileFuture.create(
            context,
            Config(),
            Mock(side_effect=RuntimeError("precompile failed")),
            (),
            "unused-result",
            None,
        )
    output = capsys.readouterr().err
    assert (
        f"Helion autotuner precompile error for {decorator}. "
        f"{SUPPRESSED_TRITON_CODE_MSG}\n"
    ) in output
    assert "%s" not in output
    assert "Traceback (most recent call last):" in output
    assert "RuntimeError: precompile failed" in output


@pytest.mark.parametrize(
    "error,ignore,level,visible",
    [
        (
            torch.cuda.OutOfMemoryError("allocation failed"),
            False,
            logging.WARNING,
            False,
        ),
        (torch.cuda.OutOfMemoryError("allocation failed"), False, logging.DEBUG, True),
        (RuntimeError("PassManager::run failed"), False, logging.WARNING, True),
        (RuntimeError("unexpected compiler bug"), True, logging.DEBUG, False),
        (torch.cuda.OutOfMemoryError("allocation failed"), True, logging.DEBUG, False),
    ],
)
def test_precompile_diagnostics_follow_error_policy(
    capsys, error, ignore, level, visible
) -> None:
    settings = Settings(
        autotune_precompile="fork",
        autotune_ignore_errors=ignore,
        autotune_log_level=level,
    )
    kernel = Mock()
    kernel.format_kernel_decorator.return_value = "@helion.kernel()"
    context = PrecompileContext(settings, AutotuningLogger(settings), kernel, (), 1)
    with (
        patch("helion.autotuner.precompile_future.maybe_dump_triton_failure"),
        patch("helion.autotuner.precompile_future.log_generated_triton_code_debug"),
    ):
        future = PrecompileFuture.create(
            context, Config(), Mock(side_effect=error), (), "unused-result", None
        )
    assert future.ok is False and future.process is None
    output = capsys.readouterr().err
    assert ("Helion autotuner precompile error" in output) is visible
    assert ("Traceback (most recent call last):" in output) is visible
    if visible:
        assert str(error) in output


def _jit_function(compile_kernel: object = None) -> Mock:
    """A Triton ``JITFunction`` stand-in launched on device 0, whose binder and
    ``_pack_args`` return distinct options, and whose ``compile`` is
    ``compile_kernel``: an exception to raise, a callable, or None to return
    ``compile.return_value``."""
    fn = Mock()
    fn.debug = False
    binder = Mock(
        return_value=(
            OrderedDict(x=object()),
            [("*fp32", "D")],
            SimpleNamespace(binder=True),
        )
    )
    fn.device_caches = {0: ({}, object(), object(), Mock(), binder)}
    fn._pack_args.return_value = (
        SimpleNamespace(packed=True),
        {"x": "*fp32"},
        {},
        {(0,): "tt.divisibility"},
    )
    fn.compile.side_effect = compile_kernel
    return fn


@contextlib.contextmanager
def _precompiler_launch() -> Iterator[Mock]:
    """Let ``make_precompiler`` bind a ``_jit_function`` (Triton 3.7 path);
    yields the patched ``compute_cache_key``, which returns ``"cache-key"``."""
    with (
        patch.object(
            importlib.import_module("helion.runtime.kernel"),
            "_find_device",
            return_value=0,
        ),
        patch(
            "helion.runtime.precompile_shim.get_triton_version",
            return_value=Version("3.7.0"),
        ),
        patch(
            "triton.runtime.jit.compute_cache_key", return_value="cache-key"
        ) as compute_cache_key,
    ):
        yield compute_cache_key


class _UnpicklableArgsError(Exception):
    """A compile error whose args pickle cannot serialize, as an error holding
    a compiler's native object can be."""

    def __init__(self, message: str) -> None:
        super().__init__(message, threading.Lock())

    def __str__(self) -> str:
        return str(self.args[0])


class _FailingCompileBatch:
    """A one-config benchmark batch, precompiled in a fork child held to
    ``compile_timeout`` seconds, of a stand-in kernel whose Triton compile
    raises ``error`` after ``compile_seconds``. Each compile appends the pid of
    the process that ran it to ``attempts``; ``futures`` collects the batch's
    precompile futures; the kernel's ``maybe_log_repro`` is a Mock. The
    config's ``indexing`` value puts its compile failure in the provider's
    unmeasured-config-values report."""

    def __init__(
        self,
        error: Exception,
        tmpdir: str,
        *,
        compile_seconds: float = 0.0,
        compile_timeout: int = 30,
    ) -> None:
        self.error = error
        self.compile_seconds = compile_seconds
        self.attempts = Path(tmpdir) / "compile_attempts"
        self.attempts.touch()
        self.futures: list[PrecompileFuture] = []
        self.jit_fn = _jit_function(self.compile_kernel)
        self.search = mock_search(
            Settings(
                autotune_precompile="fork",
                autotune_accuracy_check=False,
                autotune_compile_timeout=compile_timeout,
                autotune_log_level=logging.CRITICAL,
            ),
            device=torch.device("cpu"),
            compile_config=lambda config, allow_print=False: self.compiled_config,
            maybe_log_repro=Mock(),
        )

    def compile_kernel(self, *args: object, **kwargs: object) -> object:
        with self.attempts.open("a") as f:
            f.write(f"{os.getpid()}\n")
        time.sleep(self.compile_seconds)
        raise self.error

    def compiled_config(
        self, *args: object, _launcher: Callable[..., object] | None = None
    ) -> object:
        if _launcher is None:
            # Launching compiles a kernel that is not compiled yet.
            return self.compile_kernel()
        return _launcher(self.jit_fn, (1,), *args)

    def run(self) -> BenchmarkResult:
        provider = self.search.benchmark_provider
        create_future = provider._create_precompile_future

        def record_future(config: Config, fn: object) -> PrecompileFuture:
            self.futures.append(create_future(config, fn))
            return self.futures[-1]

        with (
            _precompiler_launch(),
            patch.object(
                provider, "_create_precompile_future", side_effect=record_future
            ),
        ):
            (result,) = provider.benchmark(
                [Config(block_sizes=[1], indexing="pointer")]
            )
        return result

    def compiled_by(self) -> list[int]:
        return [int(pid) for pid in self.attempts.read_text().split()]

    def unmeasured_config_values(self) -> list[UnmeasuredConfigValue]:
        return self.search.benchmark_provider.unmeasured_config_values()


def _benchmark_past_a_budget_cut(
    block_sizes: list[int],
    *,
    failed: int,
    peer: Callable[[dict[int, bool]], dict[int, bool]] | None = None,
    peer_codegen_failures: Sequence[int] = (),
) -> tuple[Any, list[BenchmarkResult], list[Config]]:
    """Benchmark one config per block size, with precompile futures that are
    finished at once (the one of block size ``failed``, whose ``indexing`` is
    "block_ptr", with a PTXAS error) and a budget that runs out with the
    first timing.  Returns the search, the results and the configs timed.
    With ``peer`` or ``peer_codegen_failures``, this is one of two ranks: the
    gather past the cut returns this rank's flags (each keyed by its config's
    index in the batch) and ``peer`` of them, the gather of the configs whose
    code generation failed returns the peer's ``peer_codegen_failures`` (batch
    indices), and every other gather agrees."""
    search = mock_search(
        Settings(
            autotune_precompile="fork",
            autotune_accuracy_check=False,
            autotune_log_level=logging.CRITICAL,
        ),
        device=torch.device("cpu"),
        compile_config=lambda config, allow_print=False: Mock(),
    )
    provider = search.benchmark_provider
    error = PTXASError("not compiled")
    timed: list[Config] = []

    def create(config: Config, fn: object) -> PrecompileFuture:
        failed_here = config.config["block_sizes"] == [failed]
        return PrecompileFuture.skip(
            provider._precompile_context(),
            config,
            not failed_here,
            prepare_error=error if failed_here else None,
        )

    def benchmark_function(config: Config, fn: object, **kwargs: object) -> float:
        timed.append(config)
        return 1.0

    def all_gather_object(obj: object, process_group_name: object) -> list[object]:
        if isinstance(obj, list):
            return [obj, list(peer_codegen_failures)]
        if isinstance(obj, dict) and peer is not None:
            return [obj, peer(obj)]
        return [obj, obj]

    provider.set_budget_exceeded_fn(lambda: bool(timed))
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch.object(provider, "_create_precompile_future", side_effect=create)
        )
        stack.enter_context(
            patch.object(
                provider, "_benchmark_function", side_effect=benchmark_function
            )
        )
        if peer is not None or peer_codegen_failures:
            stack.enter_context(
                patch.object(
                    benchmark_provider_module,
                    "all_gather_object",
                    side_effect=all_gather_object,
                )
            )
        results = provider.benchmark(
            [
                Config(
                    block_sizes=[size],
                    indexing="block_ptr" if size == failed else "pointer",
                )
                for size in block_sizes
            ]
        )
    return search, results, timed


@onlyBackends(["triton"])
@unittest.skipUnless(triton_is_available(), "requires Triton")
class TestPrecompileShim(unittest.TestCase):
    def test_matches_triton_runtime_cache_key_and_argument_packing(self) -> None:
        fn = _jit_function()
        kernel_cache, kernel_key_cache, target, backend, binder = fn.device_caches[0]
        bound_args, specialization, binder_options = binder.return_value
        packed_options, signature, constexprs, attrs = fn._pack_args.return_value

        with _precompiler_launch() as compute_cache_key:
            precompile = make_precompiler(fn, Config(), Mock())(object())
            self.assertTrue(precompile(in_child_process=False))

        compute_cache_key.assert_called_once_with(
            kernel_key_cache, specialization, binder_options
        )
        fn._pack_args.assert_called_once_with(
            backend, ANY, bound_args, specialization, binder_options
        )
        fn.ASTSource.assert_called_once_with(fn, signature, constexprs, attrs)
        fn.compile.assert_called_once_with(
            fn.ASTSource.return_value, target=target, options=vars(packed_options)
        )
        compiled_kernel = fn.compile.return_value
        self.assertIs(kernel_cache["cache-key"], compiled_kernel)
        compiled_kernel._init_handles.assert_called_once_with()

    def test_device_discovery_unwraps_tensor_descriptors(self) -> None:
        # A launch whose tensors all travel as tensor descriptors (every
        # indexing choice ``tensor_descriptor``, no bare tensor argument) used
        # to abort the whole autotune with NoTensorArgs in the fork precompile.
        from triton.tools.tensor_descriptor import TensorDescriptor

        base = torch.zeros(32, 64)
        descriptor = TensorDescriptor.from_tensor(base, [16, 16])
        probed = _device_probe_values([descriptor, 148])
        self.assertIs(probed[0], base)
        self.assertEqual(probed[1], 148)
        self.assertEqual(_find_argument_device(probed), base.device)


@onlyBackends(["triton"])
@unittest.skipUnless(triton_is_available(), "requires Triton")
class TestPrecompileFailures(unittest.TestCase):
    def test_child_compile_error_propagates(self) -> None:
        # In the precompile child the error reaches the handler that reports
        # it to the parent; sys.exit(1) used to slip past that handler.
        error = PTXASError("forced failure")
        with _precompiler_launch():
            precompile = make_precompiler(_jit_function(error), Config(), Mock())(
                object()
            )
        with self.assertRaises(PTXASError) as caught:
            precompile()
        self.assertIs(caught.exception, error)
        # A compile in the parent raises it too, for its caller
        # (_triton_compile) to return.
        with self.assertRaises(PTXASError) as caught:
            precompile(in_child_process=False)
        self.assertIs(caught.exception, error)

    @unittest.skipUnless(_HAS_FORK, "requires the fork start method")
    def test_fork_compile_failure_skips_config_without_compiling_again(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            batch = _FailingCompileBatch(PTXASError("forced failure"), tmpdir)
            result = batch.run()
            compiled_by = batch.compiled_by()

        self.assertEqual((result.status, result.perf), ("error", math.inf))
        (future,) = batch.futures
        assert future.process is not None and future.remote_error is not None
        self.assertEqual(future.process.exitcode, 1)
        self.assertEqual(future.failure_reason, "error")
        self.assertEqual(
            (future.remote_error.exc_type, future.remote_error.classification),
            ("PTXASError", "warn"),
        )
        # Compiled once, in the child: the parent did not compile it again,
        # and still counts the failure, with the child's error.
        self.assertEqual(len(compiled_by), 1)
        self.assertNotEqual(compiled_by[0], os.getpid())
        self.assertEqual(batch.search._autotune_metrics.num_compile_failures, 1)
        self.assertEqual(
            batch.unmeasured_config_values(),
            [
                UnmeasuredConfigValue(
                    "indexing", "pointer", 1, "PTXASError: PTXAS error: forced failure"
                )
            ],
        )
        # Its repro (HELION_PRINT_REPRO=1) is a warning, although the error is
        # logged at debug level, as the in-process benchmark logs them.
        batch.search.kernel.maybe_log_repro.assert_called_once_with(
            batch.search.benchmark_provider.log.warning, (), future.config
        )

    @unittest.skipUnless(_HAS_FORK, "requires the fork start method")
    def test_fork_compile_error_with_unpicklable_args_reaches_the_parent(
        self,
    ) -> None:
        # The child sends the error's str(), not its args: args that pickle
        # cannot serialize used to fail the child's result write, so the
        # parent saw only "No result received from subprocess." at debug
        # level, without the error or its classification.
        error = _UnpicklableArgsError("PassManager::run failed")
        with tempfile.TemporaryDirectory() as tmpdir:
            batch = _FailingCompileBatch(error, tmpdir)
            result = batch.run()

        self.assertEqual((result.status, result.perf), ("error", math.inf))
        (future,) = batch.futures
        assert future.remote_error is not None
        self.assertEqual(
            (
                future.remote_error.exc_type,
                future.remote_error.message,
                future.remote_error.classification,
            ),
            ("_UnpicklableArgsError", "PassManager::run failed", "warn"),
        )
        self.assertEqual(
            batch.unmeasured_config_values(),
            [
                UnmeasuredConfigValue(
                    "indexing",
                    "pointer",
                    1,
                    "_UnpicklableArgsError: PassManager::run failed",
                )
            ],
        )

    @unittest.skipUnless(_HAS_FORK, "requires the fork start method")
    def test_fork_preparation_error_counts_with_its_error(self) -> None:
        # An expected error from the host code the parent runs to find the
        # launch to precompile rejects the config before a child starts; the
        # compile failure is counted with that error, as a child's error is.
        error = torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        self.assertEqual(classify_triton_exception(error), "debug")
        with tempfile.TemporaryDirectory() as tmpdir:
            batch = _FailingCompileBatch(PTXASError("not compiled"), tmpdir)
            with patch.object(batch, "compiled_config", side_effect=error):
                result = batch.run()
            compiled_by = batch.compiled_by()

        self.assertEqual((result.status, result.perf), ("error", math.inf))
        (future,) = batch.futures
        self.assertIsNone(future.process)
        self.assertIs(future.failure_error(), error)
        self.assertEqual(compiled_by, [])
        self.assertEqual(batch.search._autotune_metrics.num_compile_failures, 1)
        self.assertEqual(
            batch.unmeasured_config_values(),
            [
                UnmeasuredConfigValue(
                    "indexing",
                    "pointer",
                    1,
                    "OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
                )
            ],
        )

    @unittest.skipUnless(_HAS_FORK, "requires the fork start method")
    def test_fork_frontend_compilation_error_skips_config_with_a_warning(
        self,
    ) -> None:
        # Triton's frontend CompilationError is an error the classifier does
        # not expect ("raise": its message names no known failure), yet it
        # fails only this config. Like the fork child's other compile errors
        # it skips the config and counts as a compile failure; it is logged as
        # a warning, with Triton's message rather than its arguments' tuple.
        from triton.compiler.errors import CompilationError

        source = "def kernel(x):\n    y = x + 1\n    return y\n"
        node = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.BinOp))
        error = CompilationError(source, node, "ValueError('unsupported')")
        self.assertEqual(classify_triton_exception(error), "raise")
        with tempfile.TemporaryDirectory() as tmpdir:
            batch = _FailingCompileBatch(error, tmpdir)
            with patch.object(batch.search.log, "warning") as warning:
                result = batch.run()
            compiled_by = batch.compiled_by()

        self.assertEqual((result.status, result.perf), ("error", math.inf))
        (future,) = batch.futures
        assert future.process is not None and future.remote_error is not None
        self.assertEqual(future.process.exitcode, 1)
        self.assertEqual(future.remote_error.classification, "raise")
        self.assertEqual(len(compiled_by), 1)
        self.assertNotEqual(compiled_by[0], os.getpid())
        self.assertEqual(batch.search._autotune_metrics.num_compile_failures, 1)
        (message,) = [
            str(call.args[0])
            for call in warning.call_args_list
            if "CompilationError" in str(call.args[0])
        ]
        self.assertIn(f"Error: CompilationError: {error}\n", message)
        self.assertEqual(
            batch.unmeasured_config_values(),
            [
                UnmeasuredConfigValue(
                    "indexing", "pointer", 1, "CompilationError: at 2:8:"
                )
            ],
        )

    @unittest.skipUnless(_HAS_FORK, "requires the fork start method")
    def test_fork_compile_timeout_counts_as_compile_failure(self) -> None:
        # A compile the deadline kills is a compile failure, as a compile past
        # the benchmark worker's deadline is (_record_compile_timeout).
        with tempfile.TemporaryDirectory() as tmpdir:
            batch = _FailingCompileBatch(
                PTXASError("not raised: killed at the deadline"),
                tmpdir,
                compile_seconds=30,
                compile_timeout=1,
            )
            result = batch.run()
            compiled_by = batch.compiled_by()

        self.assertEqual((result.status, result.perf), ("timeout", math.inf))
        (future,) = batch.futures
        self.assertEqual(future.failure_reason, "timeout")
        # Compiled at most once, in the child (on a loaded host the deadline
        # can kill the child before it records its compile), never here.
        self.assertLessEqual(len(compiled_by), 1)
        self.assertNotIn(os.getpid(), compiled_by)
        self.assertEqual(batch.search._autotune_metrics.num_compile_failures, 1)
        self.assertEqual(
            batch.unmeasured_config_values(),
            [
                UnmeasuredConfigValue(
                    "indexing",
                    "pointer",
                    1,
                    "TimeoutError: compile exceeded autotune_compile_timeout=1s "
                    "(HELION_AUTOTUNE_COMPILE_TIMEOUT) in the precompile subprocess",
                )
            ],
        )

    @unittest.skipUnless(_HAS_FORK, "requires the fork start method")
    def test_fork_cached_failed_compile_skips_config_without_a_child(self) -> None:
        # A kernel an earlier in-process compile (_triton_compile) cached but
        # could not load is a known failure: the config is skipped and counted
        # without a fork child, which used to report it compiled.
        with tempfile.TemporaryDirectory() as tmpdir:
            batch = _FailingCompileBatch(PTXASError("compiled again"), tmpdir)
            failed_kernel = Mock()
            _set_helion_compilation_success(failed_kernel, False)
            kernel_cache = batch.jit_fn.device_caches[0][0]
            kernel_cache["cache-key"] = failed_kernel
            result = batch.run()
            compiled_by = batch.compiled_by()

        self.assertEqual((result.status, result.perf), ("error", math.inf))
        (future,) = batch.futures
        self.assertIsNone(future.process)
        self.assertEqual(compiled_by, [])
        self.assertEqual(batch.search._autotune_metrics.num_compile_failures, 1)
        # Counted with the error the in-process compile reports for it.
        self.assertEqual(
            batch.unmeasured_config_values(),
            [
                UnmeasuredConfigValue(
                    "indexing",
                    "pointer",
                    1,
                    "RuntimeError: an earlier compile of this kernel in this "
                    "process could not load it",
                )
            ],
        )

    def test_distributed_inprocess_compile_failure_counts(self) -> None:
        # A distributed kernel compiles in-process before it runs
        # (_triton_compile): a compile that fails on this rank -- in Triton,
        # or in preparing the compile, which used to raise past the ranks'
        # collective -- counts as a compile failure with its error, logged
        # with its traceback and dumped, as a config the precompile pool
        # rejects does; one that fails only on a peer is the peer's to report.
        for case, gathered, error in (
            ("compile", [False, True], "PTXASError: PTXAS error: not compiled"),
            ("preparation", [False, True], "TypeError: bad launch argument"),
            ("on-a-peer", [True, False], None),
        ):
            with (
                self.subTest(case),
                tempfile.TemporaryDirectory() as tmpdir,
                _precompiler_launch(),
                patch.object(
                    benchmark_provider_module.dist, "is_initialized", return_value=True
                ),
                patch.object(
                    benchmark_provider_module,
                    "all_gather_object",
                    return_value=gathered,
                ) as all_gather_object,
                patch.object(
                    benchmark_provider_module, "maybe_dump_triton_failure"
                ) as dump,
            ):
                batch = _FailingCompileBatch(PTXASError("not compiled"), tmpdir)
                binder = batch.jit_fn.device_caches[0][4]
                if case == "preparation":
                    binder.side_effect = TypeError("bad launch argument")
                elif case == "on-a-peer":
                    batch.jit_fn.compile.side_effect = None
                provider = batch.search.benchmark_provider
                with patch.object(provider.log, "debug") as debug:
                    perf = provider._benchmark_function(
                        Config(block_sizes=[1], indexing="pointer"),
                        batch.compiled_config,
                    )
                self.assertEqual(perf, math.inf)
                (call,) = all_gather_object.call_args_list
                self.assertIs(call.args[0], error is None)
                self.assertEqual(
                    batch.search._autotune_metrics.num_compile_failures,
                    int(error is not None),
                )
                self.assertEqual(
                    batch.unmeasured_config_values(),
                    []
                    if error is None
                    else [UnmeasuredConfigValue("indexing", "pointer", 1, error)],
                )
                logged = [
                    call.kwargs["exc_info"]
                    for call in debug.call_args_list
                    if call.kwargs.get("exc_info") is not None
                ]
                dumped = [call.args[2] for call in dump.call_args_list]
                if error is None:
                    self.assertEqual((logged, dumped), ([], []))
                    # The autotune log writes on rank 0 only: a rank whose
                    # compile succeeded says a peer's failed.
                    self.assertIn(
                        "failed on a peer rank",
                        "".join(call.args[0]() for call in debug.call_args_list),
                    )
                else:
                    (logged_error,) = logged
                    self.assertEqual(
                        f"{type(logged_error).__name__}: {logged_error}", error
                    )
                    self.assertEqual(dumped, [logged_error])

    def test_a_failed_precompile_past_the_budget_cut_stays_a_compile_failure(
        self,
    ) -> None:
        # The budget is asked before each timing, once the whole batch has
        # precompiled: a config whose precompile failed is a compile failure
        # whether or not the timing reaches it -- counted, in the
        # unmeasured-config-values report, and not a never-attempted
        # "skipped" config, which a search would try again.
        search, results, timed = _benchmark_past_a_budget_cut([1, 2, 4], failed=2)
        self.assertEqual(len(timed), 1)
        self.assertEqual([r.status for r in results], ["ok", "error", "skipped"])
        self.assertEqual(search._autotune_metrics.num_compile_failures, 1)
        self.assertEqual(
            search.benchmark_provider.unmeasured_config_values(),
            [
                UnmeasuredConfigValue(
                    "indexing", "block_ptr", 1, "PTXASError: PTXAS error: not compiled"
                )
            ],
        )

    def test_a_precompile_failed_on_one_rank_past_the_budget_cut_fails_everywhere(
        self,
    ) -> None:
        # Past a budget cut the ranks still agree on which precompiles failed:
        # a config that failed on this rank is a counted compile failure here,
        # one that failed only on a peer is "peer_compilation_fail", and only a
        # config that compiled on every rank stays never attempted
        # ("skipped").  A search drops skipped configs and keeps failed ones,
        # so a rank-local verdict would make the ranks' populations diverge.
        # The peer compiled [2] and failed [4].
        search, results, timed = _benchmark_past_a_budget_cut(
            [1, 2, 4, 8], failed=2, peer=lambda pending: {**pending, 1: True, 2: False}
        )
        self.assertEqual(len(timed), 1)
        self.assertEqual(
            [r.status for r in results],
            ["ok", "error", "peer_compilation_fail", "skipped"],
        )
        self.assertEqual(search._autotune_metrics.num_compile_failures, 1)

    def test_a_config_whose_code_generation_failed_on_a_peer_is_dropped_here(
        self,
    ) -> None:
        # Code generation failed for [4] on the peer only: this rank drops it
        # before timing ("peer_compilation_fail"), so both ranks time the same
        # configs and pair them at each collective, where the ranks' units
        # used to differ by [4] and paired different configs.
        search, results, timed = _benchmark_past_a_budget_cut(
            [1, 2, 4, 8], failed=2, peer_codegen_failures=[2]
        )
        self.assertEqual(len(timed), 1)
        self.assertEqual(
            [r.status for r in results],
            ["ok", "error", "peer_compilation_fail", "skipped"],
        )
        self.assertEqual(search._autotune_metrics.num_compile_failures, 1)

    def test_a_preparation_error_keeps_no_launch_arguments_alive(self) -> None:
        # A bind error raised while handling the launch extraction chains the
        # _ExtractedLaunchArgs, whose arguments are the kernel's tensors, and
        # tracebacks whose frames hold them; the future keeps the error for
        # the whole batch.
        alive: list[weakref.ref[torch.Tensor]] = []

        def prepare(*args: object, **kwargs: object) -> object:
            output = torch.empty(4)
            alive.append(weakref.ref(output))
            try:
                raise _ExtractedLaunchArgs(None, (1,), (output,), {})
            except _ExtractedLaunchArgs:
                raise PTXASError("bad launch argument")  # noqa: B904

        search = mock_search(
            Settings(autotune_precompile="fork", autotune_log_level=logging.CRITICAL),
            device=torch.device("cpu"),
        )
        provider = search.benchmark_provider
        with patch.object(
            precompile_future_module, "_prepare_precompiler_for_fork", prepare
        ):
            future = PrecompileFuture.create(
                provider._precompile_context(), Config(), Mock(), (), "", None
            )
        gc.collect()
        self.assertIsNone(alive[0]())
        self.assertIn("bad launch argument", str(future.failure_error()))

    @unittest.skipUnless(_HAS_FORK, "requires the fork start method")
    def test_children_exit_nonzero_without_an_ok_result(self) -> None:
        # The parent takes exit status 0 for a compiled config, so an exception
        # the children's handler lets through must not exit 0.
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "result.pkl"
            args_path = Path(tmpdir) / "args.pt"
            torch.save((), args_path)
            fn_spec = SerializedCompiledFunction(
                function_name="call",
                source_code="import sys\ndef call():\n    sys.exit(0)\n",
                filename="<test_children_exit_nonzero_without_an_ok_result>",
                module_name=None,
            )
            children = {
                "fork": (
                    _run_kernel_in_subprocess_fork,
                    (
                        functools.partial(sys.exit, 0),
                        Config(),
                        Mock(),
                        str(result_path),
                        "@test",
                    ),
                ),
                "spawn": (
                    _run_kernel_in_subprocess_spawn,
                    (fn_spec, str(args_path), str(result_path), "@test"),
                ),
            }
            for name, (target, args) in children.items():
                with (
                    self.subTest(child=name),
                    patch("helion.autotuner.precompile_future.synchronize_device"),
                ):
                    # Either child runs in a fork here; neither cares how it
                    # was started.
                    process = mp.get_context("fork").Process(target=target, args=args)
                    process.start()
                    process.join(30)
                    if process.is_alive():
                        process.kill()
                        process.join()
                    self.assertEqual(process.exitcode, 1)
                    self.assertFalse(result_path.exists())


if __name__ == "__main__":
    unittest.main()
