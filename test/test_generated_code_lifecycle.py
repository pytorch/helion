from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import torch

from test.test_generated_code_binding import _GeneratedCodeBindingTestCase
from test.test_generated_code_cache import _bound
from test.test_generated_code_cache import _GeneratedCodeCacheTestCase
from test.test_generated_code_cache import _runner

import helion
from helion._compat import supports_torch_compile_template_lowering
from helion._testing import DEVICE
from helion._utils import counters
import helion.language as hl
from helion.runtime.cached_kernel import _CachedBoundKernel
from helion.runtime.kernel import BoundKernel
from helion.runtime.kernel import KernelCompiler


def _multiply(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] * y[tile]
    return out


def _add_one(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + 1
    return out


def _run_backward_lowering_race() -> None:
    from torch._dynamo.convert_frame import compile_lock
    import triton

    from helion._compiler._inductor import template_buffer

    expected_backend = "hip" if torch.version.hip is not None else "cuda"
    assert triton.runtime.driver.active.get_current_target().backend == expected_backend

    def make_kernel():
        return helion.kernel(
            backend="triton",
            config=helion.Config(block_sizes=[128]),
            static_shapes=True,
            cache_level="code",
            autotune_effort="none",
        )(_add_one)

    x = torch.ones(1024, device=DEVICE)
    make_kernel()(x)
    kernel = make_kernel()
    runner = _runner(kernel, (x,))
    assert isinstance(runner, _CachedBoundKernel)
    torch.testing.assert_close(kernel(x), x + 1)
    eager_started = threading.Event()
    eager_in_frontend = threading.Event()
    errors: list[BaseException] = []

    def eager() -> None:
        try:
            assert eager_started.wait(20)
            y = torch.ones(2048, device=DEVICE)
            torch.testing.assert_close(kernel(y), y + 1)
        except BaseException as error:
            errors.append(error)
            eager_in_frontend.set()

    original_propagate = KernelCompiler.propagate_types
    original_bind = template_buffer._bind_kernel_for_lowering
    lowered: list[BoundKernel] = []

    def propagate(compiler, host_function):
        if threading.current_thread().name == "eager-new-binding":
            # The eager thread owns _bind_lock and is about to wait for
            # Dynamo's compile_lock, held by the lazy backward compiler.
            assert kernel._bind_lock._is_owned(), "eager bind did not own its lock"
            eager_in_frontend.set()
        return original_propagate(compiler, host_function)

    def bind_for_lowering(target, args):
        if target is kernel:
            assert compile_lock._is_owned(), "lowering did not own the compile lock"
        bound = original_bind(target, args)
        if target is kernel and not lowered:
            lowered.append(bound)
            eager_started.set()
            assert eager_in_frontend.wait(20)
        return bound

    class BackwardKernel(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value * 2

        @staticmethod
        def backward(ctx, grad):
            return kernel(grad)

    with (
        patch.object(KernelCompiler, "propagate_types", propagate),
        patch.object(template_buffer, "_bind_kernel_for_lowering", bind_for_lowering),
    ):
        worker = threading.Thread(target=eager, name="eager-new-binding", daemon=True)
        worker.start()
        value = torch.ones(1024, device=DEVICE, requires_grad=True)
        result = torch.compile(BackwardKernel.apply, fullgraph=True)(value)
        result.backward(torch.ones_like(value))
        worker.join(timeout=20)
        assert not worker.is_alive(), "eager bind did not finish"
    assert not errors, errors
    assert lowered, "lazy backward did not lower the Helion kernel"
    assert all(isinstance(bound, BoundKernel) for bound in lowered)
    assert all(not bound._cache_managed for bound in lowered)
    assert runner in kernel._bound_kernels.values()
    torch.testing.assert_close(value.grad, torch.full_like(value, 2))


class TestGeneratedCodeLifecycle(_GeneratedCodeBindingTestCase):
    def _warm(self, *, launch: bool = True):
        x = torch.ones(16)
        config = helion.Config(block_sizes=[32])
        _bound(x).set_config(config)
        warm = _bound(x)
        self.assertIsInstance(warm, _CachedBoundKernel)
        if launch:
            warm.set_config(config)
        return x, config, warm

    def test_public_bind_preserves_selected_config_and_launcher(self) -> None:
        x, config, warm = self._warm()
        run = warm._run
        path = warm.get_cached_path(config)
        normal = warm.kernel.bind((x, 2))
        self.assertIsInstance(normal, BoundKernel)
        self.assertEqual(normal._config, config)
        self.assertIs(normal._run, run)
        self.assertIs(normal.compile_config(config), run)
        self.assertEqual(normal.get_cached_path(config), path)
        self.assertIsNone(normal(x, 2))
        self.assertIsNone(warm.kernel(x, 2))

    def test_warm_set_config_records_accuracy_report_config_without_frontend(
        self,
    ) -> None:
        x, config, warm = self._warm()
        counters["best_config_decorator"].clear()
        with patch.object(
            warm, "_materialize", side_effect=AssertionError("frontend required")
        ):
            warm.set_config(config)
        decorator = next(iter(counters["best_config_decorator"]))
        self.assertEqual(
            decorator,
            BoundKernel.format_kernel_decorator(warm, config, warm.settings),
        )
        self.assertIsNone(warm(x, 2))

    def test_explicit_empty_selected_config_is_not_replaced_by_saved_config(
        self,
    ) -> None:
        _, saved, warm = self._warm()
        empty = helion.Config()
        self.assertNotEqual(saved, empty)
        warm._config = empty
        self.assertIs(warm._requested_config(None), empty)

    def test_existing_delegate_does_not_wait_for_bind_lock(self) -> None:
        x, _, warm = self._warm()
        normal = warm._materialize((x, 2))
        held = threading.Event()
        release = threading.Event()

        def hold_lock() -> None:
            with warm.kernel._bind_lock:
                held.set()
                self.assertTrue(release.wait(5))

        with ThreadPoolExecutor(max_workers=2) as pool:
            holder = pool.submit(hold_lock)
            self.assertTrue(held.wait(5))
            try:
                self.assertIs(pool.submit(warm._materialize).result(timeout=2), normal)
            finally:
                release.set()
            holder.result(timeout=5)

    @unittest.skipUnless(
        supports_torch_compile_template_lowering(),
        "requires Helion's Inductor template lowering",
    )
    def test_lowering_uses_isolated_frontend_for_warm_runner(self) -> None:
        from helion._compiler._inductor.template_buffer import _bind_kernel_for_lowering

        for launched in (False, True):
            with self.subTest(launched=launched):
                x, _, warm = self._warm(launch=launched)
                isolated = warm.kernel._bind_isolated
                acquired: list[bool] = []

                def try_eager_lock(lock=warm.kernel._bind_lock) -> bool:
                    result = lock.acquire(blocking=False)
                    if result:
                        lock.release()
                    return result

                with ThreadPoolExecutor(max_workers=1) as pool:

                    def checked_isolated(
                        args,
                        _acquired=acquired,
                        _isolated=isolated,
                        _pool=pool,
                        _try_lock=try_eager_lock,
                    ):
                        _acquired.append(_pool.submit(_try_lock).result(timeout=2))
                        return _isolated(args)

                    with (
                        patch.object(
                            warm,
                            "_materialize",
                            side_effect=AssertionError("lowering materialized runner"),
                        ),
                        patch.object(
                            warm.kernel, "_bind_isolated", side_effect=checked_isolated
                        ),
                    ):
                        bound = _bind_kernel_for_lowering(warm.kernel, (x, 2))
                self.assertEqual(acquired, [False])
                self.assertIsInstance(bound, BoundKernel)
                self.assertFalse(bound._cache_managed)
                self.assertIn(warm, warm.kernel._bound_kernels.values())
                self.assertIsNone(warm._delegate)


@unittest.skipIf(not torch.cuda.is_available(), "requires CUDA and Triton")
class TestGeneratedCodeAutodiff(_GeneratedCodeCacheTestCase):
    @unittest.skipUnless(
        supports_torch_compile_template_lowering(),
        "requires Helion's Inductor template lowering",
    )
    def test_lazy_backward_lowering_races_eager_bind_without_deadlock(self) -> None:
        # A separate process makes a lock-order regression a bounded failure
        # rather than leaving blocked autograd threads in the test worker.
        with tempfile.TemporaryDirectory() as inductor_cache:
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from test.test_generated_code_lifecycle import "
                        "_run_backward_lowering_race; _run_backward_lowering_race()"
                    ),
                ],
                check=True,
                timeout=60,
                env={
                    **os.environ,
                    # This exercises the Triton cache even in a TileIR CI job.
                    "HELION_BACKEND": "triton",
                    "ENABLE_TILE": "0",
                    "TORCHINDUCTOR_CACHE_DIR": inductor_cache,
                    "TORCHINDUCTOR_COMPILE_THREADS": "1",
                },
            )

    def test_warm_forward_backward_forward_preserves_launcher(self) -> None:
        config = helion.Config(block_sizes=[128])

        def make_kernel():
            return helion.kernel(
                backend="triton",
                config=config,
                cache_level="code",
                autotune_effort="none",
            )(_multiply)

        x, y, grad = (torch.randn(1024, device=DEVICE) for _ in range(3))
        cold = make_kernel()
        torch.testing.assert_close(cold(x, y), x * y)
        warm = make_kernel()
        runner = _runner(warm, (x, y))
        self.assertIsInstance(runner, _CachedBoundKernel)
        torch.testing.assert_close(warm(x, y), x * y)
        grad_x, grad_y = helion.experimental.backward(warm, grad, x, y)
        torch.testing.assert_close(grad_x, grad * y)
        torch.testing.assert_close(grad_y, grad * x)
        torch.testing.assert_close(warm(x, y), x * y)
        self.assertEqual(warm.bind((x, y))._config, config)
        self.assertIsNotNone(warm.bind((x, y))._run)
