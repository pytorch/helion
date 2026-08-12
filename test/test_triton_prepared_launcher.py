"""Safety and integration tests for Triton's resolved launch cache."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
import unittest
from unittest.mock import Mock
from unittest.mock import patch

import torch

import helion
from helion._compiler.triton.backend import TritonBackend
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
import helion.language as hl
from helion.runtime.triton import launcher

if TYPE_CHECKING:
    from collections.abc import Callable

    from triton.runtime.jit import JITFunction


_TRITON_PREPARED_GLOBAL = 1


class _FakeDispatchRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def __call__(self, *args: object) -> None:
        self.calls.append(args)


class _FakeCompiled:
    def __init__(self) -> None:
        self.runner_calls: list[tuple[object, ...]] = []
        self.runner_error: BaseException | None = None

    def __getitem__(self, _grid: tuple[int, int, int]) -> Callable[..., None]:
        def run(*args: object) -> None:
            self.runner_calls.append(args)
            if self.runner_error is not None:
                raise self.runner_error

        return run


class _FakeJITFunction:
    def __init__(self) -> None:
        self.params = [
            SimpleNamespace(
                is_const=False,
                is_constexpr=False,
                annotation_type="",
                do_not_specialize=False,
                do_not_specialize_on_alignment=False,
            )
        ]
        self.pre_run_hooks: list[Callable[..., None]] = []
        self.debug = None
        self.launch_metadata = None
        self.used_global_vals: dict[object, object] = {}
        self.device_caches: dict[object, tuple[object, ...]] = {}
        self.run_calls = 0
        self.compiled: list[object] = []

    def _device_cache(self) -> tuple[object, ...]:
        cache = self.device_caches.get(0)
        if cache is None:
            cache = ({}, {}, object(), "cuda", object())
            self.device_caches[0] = cache
        return cache

    def run(self, *_args: object, **_kwargs: object) -> object:
        self.run_calls += 1
        compiled = _FakeCompiled()
        self.compiled.append(compiled)
        kernel_cache = self._device_cache()[0]
        kernel_cache[self.run_calls] = compiled  # type: ignore[index]
        return compiled


def _fake_triton() -> SimpleNamespace:
    active = SimpleNamespace(get_current_device=lambda: 0)
    runtime_knobs = SimpleNamespace(
        debug=False,
        add_stages_inspection_hook=None,
        launch_enter_hook=None,
        launch_exit_hook=None,
    )
    return SimpleNamespace(
        knobs=SimpleNamespace(
            runtime=runtime_knobs,
            compilation=SimpleNamespace(instrumentation_mode=""),
        ),
        runtime=SimpleNamespace(driver=SimpleNamespace(active=active)),
    )


def _launch(
    jit_fn: object,
    value: object = 1,
    *,
    grid: tuple[int, ...] = (1,),
    num_warps: int = 4,
    num_stages: int = 1,
    launch_cooperative_grid: bool = False,
    ptx_options: str | None = None,
    maxnreg: int | None = None,
) -> object:
    extra_options = {} if maxnreg is None else {"maxnreg": maxnreg}
    return launcher.default_launcher(
        jit_fn,
        grid,
        value,
        num_warps=num_warps,
        num_stages=num_stages,
        launch_cooperative_grid=launch_cooperative_grid,
        ptx_options=ptx_options,
        **extra_options,
    )


@onlyBackends(["triton"])
class TestPreparedTritonLauncherUnit(TestCase):
    def test_dispatch_runners_project_arguments(self) -> None:
        fake_triton = _fake_triton()
        active = fake_triton.runtime.driver.active
        compiled = SimpleNamespace(
            _dispatch_arg_indices=(0, 2),
            _dispatcher=object(),
            _num_kernel_args=2,
        )
        fallback_compiled = SimpleNamespace(_dispatcher=None)
        dispatch_runner = _FakeDispatchRunner()
        dispatcher_fallback_calls: list[tuple[object, ...]] = []
        fallback_calls: list[tuple[object, ...]] = []

        def dispatcher_fallback_runner(*args: object) -> None:
            dispatcher_fallback_calls.append(args)

        def fallback_runner(*args: object) -> None:
            fallback_calls.append(args)

        with patch.object(launcher, "triton", fake_triton):
            wrapped_dispatch = launcher._wrap_prepared_launch_runner(
                active,
                compiled,
                dispatch_runner,
                (1, 1, 1),
                3,
            )
            wrapped_dispatcher_fallback = launcher._wrap_prepared_launch_runner(
                active,
                compiled,
                dispatcher_fallback_runner,
                (1, 1, 1),
                3,
            )
            wrapped_fallback = launcher._wrap_prepared_launch_runner(
                active,
                fallback_compiled,
                fallback_runner,
                (1, 1, 1),
                3,
            )
            wrapped_dispatch("runtime_a", "constexpr", "runtime_b")
            wrapped_dispatcher_fallback("runtime_a", "constexpr", "runtime_b")
            wrapped_fallback("runtime_a", "constexpr", "runtime_b")

        self.assertEqual(dispatch_runner.calls, [("runtime_a", "runtime_b")])
        self.assertEqual(dispatcher_fallback_calls, [("runtime_a", "runtime_b")])
        self.assertEqual(fallback_calls, [("runtime_a", "constexpr", "runtime_b")])

    def test_dispatch_runner_observes_hooks_added_after_prepare(self) -> None:
        fake_triton = _fake_triton()
        active = fake_triton.runtime.driver.active
        active.get_current_stream = lambda _device: 17
        dispatch_runner = _FakeDispatchRunner()
        function = object()
        packed_metadata = object()
        launch_metadata = object()
        result = object()
        compiled = SimpleNamespace(
            _dispatch_arg_indices=(0, 2),
            _dispatcher=object(),
            _num_kernel_args=2,
            function=function,
            packed_metadata=packed_metadata,
            launch_metadata=Mock(return_value=launch_metadata),
            run=Mock(return_value=result),
        )

        with patch.object(launcher, "triton", fake_triton):
            wrapped = launcher._wrap_prepared_launch_runner(
                active,
                compiled,
                dispatch_runner,
                (2, 3),
                3,
            )
            enter_hook = object()
            exit_hook = object()
            fake_triton.knobs.runtime.launch_enter_hook = enter_hook
            fake_triton.knobs.runtime.launch_exit_hook = exit_hook
            self.assertIs(wrapped("runtime_a", "constexpr", "runtime_b"), result)

        self.assertFalse(dispatch_runner.calls)
        compiled.launch_metadata.assert_called_once_with(  # type: ignore[union-attr]
            (2, 3), 17, "runtime_a", "constexpr", "runtime_b"
        )
        compiled.run.assert_called_once_with(  # type: ignore[union-attr]
            2,
            3,
            1,
            17,
            function,
            packed_metadata,
            launch_metadata,
            enter_hook,
            exit_hook,
            "runtime_a",
            "constexpr",
            "runtime_b",
        )

    def test_kernel_cache_membership_invalidates_prepared_launch(self) -> None:
        jit_fn = _FakeJITFunction()
        fake_triton = _fake_triton()
        with patch.object(launcher, "triton", fake_triton):
            first = cast("_FakeCompiled", _launch(jit_fn))
            self.assertIs(_launch(jit_fn), first)
            self.assertEqual(jit_fn.run_calls, 1)
            self.assertEqual(first.runner_calls, [(1,)])

            kernel_cache = jit_fn.device_caches[0][0]
            kernel_cache.clear()  # type: ignore[union-attr]
            second = _launch(jit_fn)
            self.assertIsNot(second, first)
            self.assertEqual(jit_fn.run_calls, 2)
            self.assertEqual(first.runner_calls, [(1,)])

            jit_fn.device_caches.clear()
            third = _launch(jit_fn)
            self.assertIsNot(third, second)
            self.assertEqual(jit_fn.run_calls, 3)

    def test_grid_and_launch_options_are_guarded(self) -> None:
        cases: tuple[Callable[[_FakeJITFunction], object], ...] = (
            lambda jit_fn: _launch(jit_fn, grid=(2,)),
            lambda jit_fn: _launch(jit_fn, num_warps=8),
            lambda jit_fn: _launch(jit_fn, num_stages=2),
            lambda jit_fn: _launch(jit_fn, launch_cooperative_grid=True),
            lambda jit_fn: _launch(jit_fn, ptx_options="--test"),
            lambda jit_fn: _launch(jit_fn, maxnreg=64),
        )
        for changed_launch in cases:
            with self.subTest(changed_launch=changed_launch):
                jit_fn = _FakeJITFunction()
                with patch.object(launcher, "triton", _fake_triton()):
                    _launch(jit_fn)
                    changed_launch(jit_fn)
                self.assertEqual(jit_fn.run_calls, 2)

    def test_dynamic_jit_modes_use_slow_path(self) -> None:
        cases = (
            ("jit_debug", True),
            ("runtime_debug", True),
            ("instrumentation", "profile"),
            ("add_stages", lambda: ("key", "hash")),
        )
        for field, value in cases:
            with self.subTest(field=field, value=value):
                jit_fn = _FakeJITFunction()
                fake_triton = _fake_triton()
                with patch.object(launcher, "triton", fake_triton):
                    _launch(jit_fn)
                    if field == "jit_debug":
                        jit_fn.debug = value
                    elif field == "runtime_debug":
                        fake_triton.knobs.runtime.debug = value
                    elif field == "instrumentation":
                        fake_triton.knobs.compilation.instrumentation_mode = value
                    else:
                        fake_triton.knobs.runtime.add_stages_inspection_hook = value
                    _launch(jit_fn)
                self.assertEqual(jit_fn.run_calls, 2)

    def test_prepared_runner_exception_is_not_retried(self) -> None:
        jit_fn = _FakeJITFunction()
        with patch.object(launcher, "triton", _fake_triton()):
            compiled = cast("_FakeCompiled", _launch(jit_fn))
            compiled.runner_error = RuntimeError("launch failed")
            with self.assertRaisesRegex(RuntimeError, "launch failed"):
                _launch(jit_fn)
        self.assertEqual(jit_fn.run_calls, 1)
        self.assertEqual(compiled.runner_calls, [(1,)])

    def test_cache_is_bounded(self) -> None:
        jit_fn = _FakeJITFunction()
        with patch.object(launcher, "triton", _fake_triton()):
            for value in range(10):
                _launch(jit_fn, grid=(value + 1,))
        self.assertEqual(jit_fn.run_calls, 10)
        self.assertEqual(len(jit_fn._helion_prepared_launches), 8)  # type: ignore[attr-defined]

    def test_simple_options_can_prepare(self) -> None:
        jit_fn = _FakeJITFunction()
        with patch.object(launcher, "triton", _fake_triton()):
            first = _launch(
                jit_fn,
                ptx_options="--target-option",
                maxnreg=64,
            )
            self.assertIs(
                _launch(
                    jit_fn,
                    ptx_options="--target-option",
                    maxnreg=64,
                ),
                first,
            )
        self.assertEqual(jit_fn.run_calls, 1)

    def test_backend_tensor_specialization_is_guarded(self) -> None:
        jit_fn = _FakeJITFunction()

        def specialize(
            _backend: object,
            value: object,
            _is_const: bool,
            _specialize: bool,
            _align: bool,
        ) -> tuple[str, str]:
            assert isinstance(value, torch.Tensor)
            storage_class = (
                "small" if value.untyped_storage().nbytes() <= 64 else "large"
            )
            return str(value.dtype), storage_class

        with (
            patch.object(launcher, "triton", _fake_triton()),
            patch.object(launcher, "native_specialize_impl", side_effect=specialize),
        ):
            first = _launch(jit_fn, torch.empty(16))
            self.assertIs(_launch(jit_fn, torch.empty(16)), first)
            second = _launch(jit_fn, torch.empty(32))

        self.assertIsNot(second, first)
        self.assertEqual(jit_fn.run_calls, 2)

    def test_missing_native_specializer_uses_slow_path(self) -> None:
        jit_fn = _FakeJITFunction()
        with (
            patch.object(launcher, "triton", _fake_triton()),
            patch.object(launcher, "native_specialize_impl", None),
        ):
            _launch(jit_fn)
            _launch(jit_fn)
        self.assertEqual(jit_fn.run_calls, 2)
        self.assertFalse(getattr(jit_fn, "_helion_prepared_launches", None))

    def test_backend_cache_clear_removes_all_prepared_state(self) -> None:
        config = object()
        jit_fn = SimpleNamespace(
            device_caches={0: ({"kernel": object()}, {}, None, None, None)},
            _helion_prepared_launches=[object()],
        )

        def compiled_fn() -> None:
            return None

        key = "_helion_test_kernel"
        compiled_fn.__globals__[key] = jit_fn
        bound = SimpleNamespace(
            _compile_cache={config: compiled_fn},
            config_spec=Mock(),
            kernel=SimpleNamespace(name="test_kernel"),
        )
        try:
            TritonBackend()._clear_triton_jit_cache(bound, config)  # type: ignore[arg-type]
        finally:
            del compiled_fn.__globals__[key]

        self.assertFalse(jit_fn.device_caches)
        self.assertNotIn("_helion_prepared_launches", jit_fn.__dict__)


def _make_add_one() -> helion.Kernel:
    @helion.kernel(
        static_shapes=True,
        config=helion.Config(block_sizes=[64], num_warps=4, num_stages=1),
    )
    def add_one(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        for tile in hl.tile(x.size(0)):
            out[tile] = x[tile] + 1
        return out

    return add_one


def _get_triton_jit_function(kernel: helion.Kernel) -> JITFunction:
    from triton.runtime.jit import JITFunction

    bound = next(iter(kernel._bound_kernels.values()))  # type: ignore[attr-defined]
    run = bound._run
    assert run is not None
    jit_fn = run.__globals__[f"_helion_{kernel.name}"]
    assert isinstance(jit_fn, JITFunction)
    return jit_fn


@onlyBackends(["triton"])
@skipIfNotCUDA()
class TestPreparedTritonLauncherCuda(RefEagerTestDisabled, TestCase):
    def test_pointer_alignment_change_reprepares_launch(self) -> None:
        add_one = _make_add_one()
        aligned = torch.randn(64, device=DEVICE)
        torch.testing.assert_close(add_one(aligned), aligned + 1)
        jit_fn = _get_triton_jit_function(add_one)

        unaligned = torch.randn(65, device=DEVICE)[1:]
        self.assertEqual(aligned.shape, unaligned.shape)
        self.assertEqual(aligned.stride(), unaligned.stride())
        self.assertEqual(aligned.data_ptr() % 16, 0)
        self.assertNotEqual(unaligned.data_ptr() % 16, 0)
        with patch.object(jit_fn, "run", wraps=jit_fn.run) as jit_run:
            out = add_one(unaligned)
        torch.testing.assert_close(out, unaligned + 1)
        self.assertEqual(jit_run.call_count, 1)

        another_unaligned = torch.randn(65, device=DEVICE)[1:]
        with patch.object(
            jit_fn,
            "run",
            side_effect=AssertionError("prepared launch called JITFunction.run"),
        ):
            out = add_one(another_unaligned)
        torch.testing.assert_close(out, another_unaligned + 1)

    def test_dynamic_scalars_reuse_matching_triton_specialization(self) -> None:
        import triton
        import triton.language as tl

        @triton.jit
        def scale(x, out, value, block: tl.constexpr):
            offsets = tl.arange(0, block)
            tl.store(out + offsets, tl.load(x + offsets) * value)

        x = torch.randn(64, device=DEVICE)
        out = torch.empty_like(x)
        launcher.default_launcher(
            scale, (1,), x, out, 17, 64, num_warps=4, num_stages=1
        )
        with patch.object(
            scale,
            "run",
            side_effect=AssertionError("dynamic integer called JITFunction.run"),
        ):
            launcher.default_launcher(
                scale, (1,), x, out, 19, 64, num_warps=4, num_stages=1
            )
        torch.testing.assert_close(out, x * 19)

        with patch.object(scale, "run", wraps=scale.run) as jit_run:
            launcher.default_launcher(
                scale, (1,), x, out, 16, 64, num_warps=4, num_stages=1
            )
        self.assertEqual(jit_run.call_count, 1)
        torch.testing.assert_close(out, x * 16)

        launcher.default_launcher(
            scale, (1,), x, out, 1.5, 64, num_warps=4, num_stages=1
        )
        with patch.object(
            scale,
            "run",
            side_effect=AssertionError("dynamic float called JITFunction.run"),
        ):
            launcher.default_launcher(
                scale, (1,), x, out, 2.5, 64, num_warps=4, num_stages=1
            )
        torch.testing.assert_close(out, x * 2.5)

        with patch.object(scale, "run", wraps=scale.run) as jit_run:
            launcher.default_launcher(
                scale, (1,), x, out, 2.5, 32, num_warps=4, num_stages=1
            )
        self.assertEqual(jit_run.call_count, 1)
        torch.testing.assert_close(out[:32], x[:32] * 2.5)

    def test_specialized_scalar_uses_native_runner_abi(self) -> None:
        import triton
        import triton.language as tl

        @triton.jit
        def fill(out, stride, value, block: tl.constexpr):
            offsets = tl.arange(0, block)
            tl.store(out + offsets * stride, value)

        out = torch.empty(64, device=DEVICE)
        launcher.default_launcher(fill, (1,), out, 1, 17, 64, num_warps=4, num_stages=1)
        with patch.object(
            fill,
            "run",
            side_effect=AssertionError("prepared launch called JITFunction.run"),
        ):
            launcher.default_launcher(
                fill, (1,), out, 1, 19, 64, num_warps=4, num_stages=1
            )
        torch.testing.assert_close(out, torch.full_like(out, 19))

    def test_warmup_never_prepares_or_launches(self) -> None:
        import triton
        import triton.language as tl

        @triton.jit
        def fill(out, block: tl.constexpr):
            offsets = tl.arange(0, block)
            tl.store(out + offsets, 1.0)

        out = torch.zeros(64, device=DEVICE)
        for _ in range(2):
            launcher.default_launcher(
                fill,
                (1,),
                out,
                64,
                num_warps=4,
                num_stages=1,
                warmup=True,
            )
        torch.testing.assert_close(out, torch.zeros_like(out))
        self.assertFalse(getattr(fill, "_helion_prepared_launches", None))

    def test_global_mutation_preserves_triton_error(self) -> None:
        import triton
        import triton.language as tl

        global _TRITON_PREPARED_GLOBAL
        original = _TRITON_PREPARED_GLOBAL
        _TRITON_PREPARED_GLOBAL = tl.constexpr(1)

        @triton.jit
        def add_global(x, out, block: tl.constexpr):
            offsets = tl.arange(0, block)
            tl.store(
                out + offsets,
                tl.load(x + offsets) + _TRITON_PREPARED_GLOBAL,
            )

        x = torch.randn(64, device=DEVICE)
        out = torch.empty_like(x)
        try:
            launcher.default_launcher(
                add_global, (1,), x, out, 64, num_warps=4, num_stages=1
            )
            _TRITON_PREPARED_GLOBAL = tl.constexpr(2)
            with self.assertRaisesRegex(RuntimeError, "Global variable"):
                launcher.default_launcher(
                    add_global, (1,), x, out, 64, num_warps=4, num_stages=1
                )
        finally:
            _TRITON_PREPARED_GLOBAL = original

    def test_helion_default_and_keyword_args_share_prepared_layers(self) -> None:
        @helion.kernel(
            static_shapes=True,
            config=helion.Config(block_sizes=[64], num_warps=4, num_stages=1),
        )
        def scale(x: torch.Tensor, value: float = 2.0) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] * value
            return out

        x = torch.randn(64, device=DEVICE)
        torch.testing.assert_close(scale(x), x * 2.0)
        jit_fn = _get_triton_jit_function(scale)
        other = torch.randn_like(x)
        with (
            patch.object(
                scale,
                "_fast_dispatch_key",
                side_effect=AssertionError("omitted default rebuilt dispatch key"),
            ),
            patch.object(
                jit_fn,
                "run",
                side_effect=AssertionError("omitted default called JITFunction.run"),
            ),
        ):
            torch.testing.assert_close(scale(other), other * 2.0)

        torch.testing.assert_close(scale(other, 2.0), other * 2.0)
        with (
            patch.object(
                scale,
                "_fast_dispatch_key",
                side_effect=AssertionError("keyword call rebuilt dispatch key"),
            ),
            patch.object(
                jit_fn,
                "run",
                side_effect=AssertionError("keyword call called JITFunction.run"),
            ),
        ):
            torch.testing.assert_close(scale(x=other, value=3.0), other * 3.0)

    def test_pre_run_hook_added_after_prepare_fires(self) -> None:
        add_one = _make_add_one()
        x = torch.randn(64, device=DEVICE)
        add_one(x)
        jit_fn = _get_triton_jit_function(add_one)
        calls: list[tuple[object, ...]] = []

        def hook(*args: object, **_kwargs: object) -> None:
            calls.append(args)

        jit_fn.pre_run_hooks.append(hook)
        try:
            torch.testing.assert_close(add_one(x), x + 1)
        finally:
            jit_fn.pre_run_hooks.remove(hook)
        self.assertEqual(len(calls), 1)

    def test_launch_hooks_remain_live_on_prepared_path(self) -> None:
        import triton

        add_one = _make_add_one()
        x = torch.randn(64, device=DEVICE)
        add_one(x)
        jit_fn = _get_triton_jit_function(add_one)
        enter_calls: list[object] = []
        exit_calls: list[object] = []

        def enter(metadata: object) -> None:
            enter_calls.append(metadata)

        def exit_hook(metadata: object) -> None:
            exit_calls.append(metadata)

        triton.knobs.runtime.launch_enter_hook.add(enter)
        triton.knobs.runtime.launch_exit_hook.add(exit_hook)
        try:
            with patch.object(
                jit_fn,
                "run",
                side_effect=AssertionError("launch hook forced JITFunction.run"),
            ):
                torch.testing.assert_close(add_one(x), x + 1)
        finally:
            triton.knobs.runtime.launch_exit_hook.remove(exit_hook)
            triton.knobs.runtime.launch_enter_hook.remove(enter)
        self.assertEqual(len(enter_calls), 1)
        self.assertEqual(len(exit_calls), 1)

    def test_custom_launch_metadata_preserves_short_grid(self) -> None:
        import triton
        import triton.language as tl

        observed_grids: list[tuple[int, ...]] = []

        def metadata(
            grid: tuple[int, ...], _metadata: object, _args: object
        ) -> dict[str, object]:
            observed_grids.append(grid)
            return {}

        @triton.jit(launch_metadata=metadata)
        def copy(x, out, size: tl.constexpr):
            offsets = tl.arange(0, size)
            tl.store(out + offsets, tl.load(x + offsets))

        def evaluate(metadata: object) -> None:
            metadata.get()  # type: ignore[union-attr]

        x = torch.randn(64, device=DEVICE)
        out = torch.empty_like(x)
        triton.knobs.runtime.launch_enter_hook.add(evaluate)
        try:
            for _ in range(2):
                launcher.default_launcher(
                    copy,
                    (1,),
                    x,
                    out,
                    64,
                    num_warps=4,
                    num_stages=1,
                )
        finally:
            triton.knobs.runtime.launch_enter_hook.remove(evaluate)
        torch.testing.assert_close(out, x)
        self.assertEqual(observed_grids, [(1,), (1,)])
        self.assertFalse(getattr(copy, "_helion_prepared_launches", None))

    def test_prepared_launch_uses_current_stream(self) -> None:
        add_one = _make_add_one()
        x = torch.zeros(64, device=DEVICE)
        add_one(x)
        jit_fn = _get_triton_jit_function(add_one)
        stream = torch.cuda.Stream()
        with (
            patch.object(
                jit_fn,
                "run",
                side_effect=AssertionError("prepared launch called JITFunction.run"),
            ),
            torch.cuda.stream(stream),
        ):
            x.fill_(3)
            out = add_one(x)
        stream.synchronize()
        torch.testing.assert_close(out, torch.full_like(out, 4))

    def test_self_removing_hook_does_not_prepare_that_call(self) -> None:
        add_one = _make_add_one()
        x = torch.randn(64, device=DEVICE)
        add_one(x)
        jit_fn = _get_triton_jit_function(add_one)
        jit_fn.__dict__.pop("_helion_prepared_launches", None)
        calls = 0

        def remove_self(*_args: object, **_kwargs: object) -> None:
            nonlocal calls
            calls += 1
            jit_fn.pre_run_hooks.remove(remove_self)

        jit_fn.pre_run_hooks.append(remove_self)
        torch.testing.assert_close(add_one(x), x + 1)
        self.assertEqual(calls, 1)
        self.assertFalse(getattr(jit_fn, "_helion_prepared_launches", None))

        torch.testing.assert_close(add_one(x), x + 1)
        self.assertTrue(jit_fn._helion_prepared_launches)  # type: ignore[attr-defined]

    def test_defaulted_jit_parameter_uses_slow_launcher(self) -> None:
        import triton
        import triton.language as tl

        @triton.jit
        def add_one(
            x,
            out,
            block: tl.constexpr = 64,  # pyrefly: ignore [bad-function-definition]
        ):
            offsets = tl.arange(0, block)
            tl.store(out + offsets, tl.load(x + offsets) + 1)

        x = torch.randn(64, device=DEVICE)
        out = torch.empty_like(x)
        for _ in range(2):
            launcher.default_launcher(
                add_one,
                (1,),
                x,
                out,
                num_warps=4,
                num_stages=1,
            )
        torch.testing.assert_close(out, x + 1)
        self.assertFalse(getattr(add_one, "_helion_prepared_launches", None))


@onlyBackends(["triton"])
@unittest.skipUnless(torch.version.hip is not None, "requires ROCm")
class TestPreparedTritonLauncherRocm(RefEagerTestDisabled, TestCase):
    def test_matching_storage_specialization_reuses_prepared_launch(self) -> None:
        add_one = _make_add_one()
        first = torch.randn(16, device=DEVICE)
        torch.testing.assert_close(add_one(first), first + 1)
        jit_fn = _get_triton_jit_function(add_one)

        same_shape = torch.randn(32, device=DEVICE)[:16]
        with patch.object(
            jit_fn,
            "run",
            side_effect=AssertionError("matching HIP specialization used slow path"),
        ):
            out = add_one(same_shape)
        torch.testing.assert_close(out, same_shape + 1)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
