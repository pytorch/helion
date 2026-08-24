"""Tests for the shared Triton/CuTe monomorphic prepared-call path."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib
import threading
from typing import TYPE_CHECKING
from typing import cast
import unittest
from unittest.mock import patch

import torch

import helion
from helion._compat import supports_torch_compile_fusion
from helion._compiler._dynamo.variables import infer_output_spec
from helion._compiler.cute.backend import CuteBackend
from helion._compiler.pallas.backend import PallasBackend
from helion._compiler.triton.backend import TileIRBackend
from helion._compiler.triton.backend import TritonBackend
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import onlyBackends
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence
    from typing import Hashable
    from typing_extensions import Self

kernel_module = importlib.import_module("helion.runtime.kernel")

_RESET_SPECIALIZATION_USES_X = False


def _select_reset_specialization_tensor(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    return x if _RESET_SPECIALIZATION_USES_X else y


def _make_add_one(*, distributed: bool = False) -> helion.Kernel:
    @helion.kernel(
        static_shapes=True,
        config=helion.Config(block_sizes=[64]),
        distributed=distributed,
    )
    def add_one(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        for tile in hl.tile(x.size(0)):
            out[tile] = x[tile] + 1
        return out

    return add_one


def _make_aot_scale(*, distributed: bool = False) -> helion.Kernel:
    @helion.aot_kernel(
        config=helion.Config(block_sizes=[64]),
        distributed=distributed,
    )
    def aot_scale(x: torch.Tensor, value: int) -> torch.Tensor:
        value = hl.specialize(value)
        out = torch.empty_like(x)
        for tile in hl.tile(x.size(0)):
            out[tile] = x[tile] * value
        return out

    return aot_scale


def _make_scale() -> helion.Kernel:
    @helion.kernel(
        static_shapes=True,
        config=helion.Config(block_sizes=[64]),
    )
    def scale(x: torch.Tensor, value: int) -> torch.Tensor:
        value = hl.specialize(value)
        out = torch.empty_like(x)
        for tile in hl.tile(x.size(0)):
            out[tile] = x[tile] * value
        return out

    return scale


def _tensor_size(values: Sequence[object]) -> int:
    value = values[0]
    assert isinstance(value, torch.Tensor)
    return value.size(0)


@onlyBackends(["triton", "cute"])
class TestPreparedCall(RefEagerTestDisabled, TestCase):
    def _assert_aot_repeat_uses_direct_dispatch_cache(
        self,
        kernel: helion.Kernel,
        args: tuple[object, ...],
        expected: torch.Tensor,
    ) -> None:
        self.assertIsNone(kernel._prepared_call)  # type: ignore[attr-defined]
        with (
            patch.object(
                kernel,
                "_key_fn",
                wraps=kernel._key_fn,  # type: ignore[attr-defined]
            ) as key_fn,
            patch.object(
                kernel,
                "_prepare_dispatch_entry",
                side_effect=AssertionError("AOT cache hit tried to prepare"),
            ),
            patch.object(
                kernel,
                "_fast_dispatch_key_and_guards",
                side_effect=AssertionError("AOT cache hit built guard metadata"),
            ),
            patch.object(
                kernel,
                "_bind",
                side_effect=AssertionError("AOT cache hit rebound the kernel"),
            ),
        ):
            torch.testing.assert_close(kernel(*args), expected)
        key_fn.assert_called_once_with(*args)

    def test_fresh_tensor_bypasses_dispatch_key(self) -> None:
        add_one = _make_add_one()
        x = torch.randn(64, device=DEVICE)
        first = add_one(x)
        self.assertIsNotNone(add_one._prepared_call)  # type: ignore[attr-defined]

        other = torch.randn_like(x)
        with (
            patch.object(
                add_one,
                "_fast_dispatch_key_and_guards",
                side_effect=AssertionError("prepared call rebuilt its dispatch key"),
            ),
            patch.object(
                add_one,
                "bind",
                side_effect=AssertionError("prepared call rebound the kernel"),
            ),
        ):
            second = add_one(x=other)

        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        torch.testing.assert_close(first, x + 1)
        torch.testing.assert_close(second, other + 1)

    def test_tensor_metadata_changes_use_dispatch(self) -> None:
        add_one = _make_add_one()
        add_one(torch.randn(64, device=DEVICE))
        static_indices = torch.randn(64, device=DEVICE)
        static_indices._dynamo_static_indices = {0}  # type: ignore[attr-defined]
        changed_static_indices = torch.randn_like(static_indices)
        changed_static_indices._dynamo_static_indices = set()  # type: ignore[attr-defined]
        variants = (
            torch.randn(128, device=DEVICE)[::2],
            torch.randn(64, device=DEVICE, dtype=torch.float16),
            torch.randn(65, device=DEVICE),
            static_indices,
            changed_static_indices,
        )

        for value in variants:
            with (
                self.subTest(
                    shape=value.shape,
                    stride=value.stride(),
                    dtype=value.dtype,
                    static_indices=getattr(value, "_dynamo_static_indices", None),
                ),
                patch.object(
                    add_one,
                    "_fast_dispatch_key_and_guards",
                    wraps=add_one._fast_dispatch_key_and_guards,  # type: ignore[attr-defined]
                ) as fast_key,
            ):
                out = add_one(value)
                fast_key.assert_called()
                torch.testing.assert_close(out, value + 1)

    def test_runtime_scalars_reuse_preparation_but_specialized_values_do_not(
        self,
    ) -> None:
        @helion.kernel(
            static_shapes=True,
            config=helion.Config(block_sizes=[64]),
        )
        def scale(x: torch.Tensor, value: float) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] * value
            return out

        @helion.kernel(
            static_shapes=True,
            config=helion.Config(block_sizes=[64]),
        )
        def scale_constexpr(x: torch.Tensor, value: hl.constexpr) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] * value
            return out

        scale_specialized = _make_scale()

        x = torch.randn(64, device=DEVICE)
        torch.testing.assert_close(scale(x, 2.0), x * 2.0)
        with patch.object(
            scale,
            "_fast_dispatch_key_and_guards",
            side_effect=AssertionError("runtime scalar rebuilt its dispatch key"),
        ):
            torch.testing.assert_close(scale(x, 3.0), x * 3.0)
            self.assertTrue(torch.isnan(scale(x, float("nan"))).all())

        for kernel, first_value, second_value in (
            (scale_constexpr, 2.0, 3.0),
            (scale_specialized, 2, 3),
        ):
            torch.testing.assert_close(kernel(x, first_value), x * first_value)
            with patch.object(
                kernel,
                "_fast_dispatch_key_and_guards",
                wraps=kernel._fast_dispatch_key_and_guards,  # type: ignore[attr-defined]
            ) as fast_key:
                torch.testing.assert_close(kernel(x, second_value), x * second_value)
            fast_key.assert_called()

        constexpr_nan = float("nan")
        self.assertTrue(torch.isnan(scale_constexpr(x, constexpr_nan)).all())
        self.assertIsNone(scale_constexpr._prepared_call)  # type: ignore[attr-defined]

    def test_custom_key_is_evaluated_once_and_disables_preparation(self) -> None:
        state = {"key": 0}
        calls = 0

        def key(_x: torch.Tensor) -> int:
            nonlocal calls
            calls += 1
            return state["key"]

        @helion.kernel(
            static_shapes=True,
            config=helion.Config(block_sizes=[64]),
            key=key,
        )
        def add_one(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] + 1
            return out

        x = torch.randn(64, device=DEVICE)
        torch.testing.assert_close(add_one(x), x + 1)
        self.assertIsNone(add_one._prepared_call)  # type: ignore[attr-defined]
        bound = next(iter(add_one._dispatch_cache.values()))  # type: ignore[attr-defined]
        different_shape = torch.randn(65, device=DEVICE)
        calls = 0
        fast_entry = add_one._fast_dispatch_key_and_guards(  # type: ignore[attr-defined]
            (different_shape,)
        )
        self.assertIsNotNone(fast_entry)
        assert fast_entry is not None
        self.assertIsNone(  # type: ignore[attr-defined]
            add_one._prepare_dispatch_entry((different_shape,), bound, fast_entry)
        )
        self.assertEqual(calls, 1)

        calls = 0
        torch.testing.assert_close(add_one(x), x + 1)
        self.assertEqual(calls, 1)
        self.assertIsNone(add_one._prepared_call)  # type: ignore[attr-defined]

        state["key"] = 1
        torch.testing.assert_close(add_one(x), x + 1)
        self.assertIsNone(add_one._prepared_call)  # type: ignore[attr-defined]

    def test_default_aot_key_repeat_uses_direct_dispatch_cache(self) -> None:
        @helion.aot_kernel(
            config=helion.Config(block_sizes=[64]),
        )
        def aot_add_one(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] + 1
            return out

        x = torch.randn(64, device=DEVICE)
        torch.testing.assert_close(aot_add_one(x), x + 1)
        self._assert_aot_repeat_uses_direct_dispatch_cache(aot_add_one, (x,), x + 1)

    def test_omitted_default_aot_repeat_uses_direct_dispatch_cache(self) -> None:
        def key(x: torch.Tensor, _value: float = 1.0) -> int:
            return x.size(0)

        @helion.aot_kernel(
            config=helion.Config(block_sizes=[64]),
            key=key,
        )
        def aot_add_value(x: torch.Tensor, value: float = 1.0) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] + value
            return out

        x = torch.randn(64, device=DEVICE)
        torch.testing.assert_close(aot_add_value(x), x + 1)
        self._assert_aot_repeat_uses_direct_dispatch_cache(aot_add_value, (x,), x + 1)

    def test_omitted_specialized_default_uses_prepared_call(self) -> None:
        @helion.kernel(
            static_shapes=True,
            config=helion.Config(block_sizes=[64]),
        )
        def add_default(x: torch.Tensor, value: int = 2) -> torch.Tensor:
            value = hl.specialize(value)
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] + value
            return out

        x = torch.randn(64, device=DEVICE)
        torch.testing.assert_close(add_default(x), x + 2)
        self.assertIsNotNone(add_default._prepared_call)  # type: ignore[attr-defined]

        with patch.object(
            add_default,
            "_fast_dispatch_key_and_guards",
            side_effect=AssertionError("prepared call rebuilt its dispatch key"),
        ):
            torch.testing.assert_close(add_default(x), x + 2)

    def test_user_key_aot_repeat_uses_direct_dispatch_cache(self) -> None:
        calls = 0

        def key(x: torch.Tensor) -> int:
            nonlocal calls
            calls += 1
            return x.size(0)

        @helion.aot_kernel(
            config=helion.Config(block_sizes=[64]),
            key=key,
        )
        def user_key_aot_add_one(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] + 1
            return out

        x = torch.randn(64, device=DEVICE)
        torch.testing.assert_close(user_key_aot_add_one(x), x + 1)

        calls = 0
        self._assert_aot_repeat_uses_direct_dispatch_cache(
            user_key_aot_add_one, (x,), x + 1
        )
        self.assertEqual(calls, 1)

    def test_aot_late_specialization_repeat_uses_dispatch_generation(self) -> None:
        aot_scale = _make_aot_scale()

        class ForbiddenLock:
            def __enter__(self) -> None:
                raise AssertionError("stable AOT hit acquired the bind lock")

            def __exit__(self, *args: object) -> None:
                raise AssertionError("stable AOT hit exited the bind lock")

        x = torch.randn(64, device=DEVICE)
        torch.testing.assert_close(aot_scale(x, 2), x * 2)
        self.assertTrue(aot_scale._has_specialization_extras)  # type: ignore[attr-defined]

        bound = next(iter(aot_scale._dispatch_cache.values()))  # type: ignore[attr-defined]
        self.assertEqual(
            bound._dispatch_generation,
            aot_scale._specialization_generation,  # type: ignore[attr-defined]
        )
        calls = [0, 0]

        def tensor_size(values: Sequence[object]) -> int:
            calls[0] += 1
            return _tensor_size(values)

        def scalar_value(values: Sequence[object]) -> int:
            calls[1] += 1
            return cast("int", values[1])

        aot_scale._extend_bound_kernel_specializations(  # type: ignore[attr-defined]
            bound,
            bound._base_spec_key,
            [tensor_size, scalar_value],
            (x, 2),
        )
        torch.testing.assert_close(aot_scale(x, 2), x * 2)
        self.assertEqual(
            bound._dispatch_generation,
            aot_scale._specialization_generation,  # type: ignore[attr-defined]
        )
        calls[:] = [0, 0]
        with (
            patch.object(
                aot_scale,
                "_key_fn",
                wraps=aot_scale._key_fn,  # type: ignore[attr-defined]
            ) as key_fn,
            patch.object(
                aot_scale,
                "_prepare_dispatch_entry",
                side_effect=AssertionError("stable AOT hit re-prepared dispatch"),
            ),
            patch.object(
                aot_scale,
                "_bind",
                side_effect=AssertionError("stable AOT hit rebound the kernel"),
            ),
            patch.object(aot_scale, "_bind_lock", ForbiddenLock()),
        ):
            torch.testing.assert_close(aot_scale(x, 2), x * 2)

        key_fn.assert_called_once_with(x, 2)
        self.assertEqual(calls, [1, 1])

    def test_omitted_default_aot_specializations_follow_canonical_schema(
        self,
    ) -> None:
        @helion.aot_kernel(
            config=helion.Config(block_sizes=[64]),
        )
        def aot_add_default(x: torch.Tensor, value: int = 1) -> torch.Tensor:
            value = hl.specialize(value)
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] + value
            return out

        x = torch.randn(64, device=DEVICE)
        torch.testing.assert_close(aot_add_default(x, 1), x + 1)
        bound = next(iter(aot_add_default._dispatch_cache.values()))  # type: ignore[attr-defined]

        def default_value(values: Sequence[object]) -> int:
            return cast("int", values[1])

        # The raw omitted-default alias does not exist yet. Its first bind must
        # still include this extension and normalize values for every extractor.
        aot_add_default._extend_bound_kernel_specializations(  # type: ignore[attr-defined]
            bound,
            bound._base_spec_key,
            [default_value],
            (x, 1),
        )
        torch.testing.assert_close(aot_add_default(x), x + 1)

        with (
            patch.object(
                aot_add_default,
                "normalize_args",
                side_effect=AssertionError("stable AOT hit normalized defaults"),
            ),
            patch.object(
                aot_add_default,
                "_prepare_dispatch_entry",
                side_effect=AssertionError("stable AOT hit re-prepared dispatch"),
            ),
        ):
            torch.testing.assert_close(aot_add_default(x), x + 1)

        def first_value(values: Sequence[object]) -> float:
            return float(cast("torch.Tensor", values[0])[0].item())

        def last_value(values: Sequence[object]) -> float:
            return float(cast("torch.Tensor", values[0])[-1].item())

        # Consecutive extensions occur before the evicted raw alias is rebound.
        # The persistent proxy must pick up both additions.
        aot_add_default._extend_bound_kernel_specializations(  # type: ignore[attr-defined]
            bound,
            bound._base_spec_key,
            [first_value],
            (x, 1),
        )
        aot_add_default._extend_bound_kernel_specializations(  # type: ignore[attr-defined]
            bound,
            bound._base_spec_key,
            [last_value],
            (x, 1),
        )

        raw_bound = aot_add_default.bind((x,))
        x[0] = x[0] + 1
        first_changed_bound = aot_add_default.bind((x,))
        self.assertIsNot(first_changed_bound, raw_bound)
        x[-1] = x[-1] + 1
        self.assertIsNot(aot_add_default.bind((x,)), first_changed_bound)
        torch.testing.assert_close(aot_add_default(x), x + 1)
        with patch.object(
            aot_add_default,
            "_prepare_dispatch_entry",
            side_effect=AssertionError("extended omitted-default hit re-prepared"),
        ):
            torch.testing.assert_close(aot_add_default(x), x + 1)

    def test_compiler_binds_are_isolated_from_eager_caches(self) -> None:
        @helion.kernel(
            static_shapes=True,
            config=helion.Config(block_sizes=[64]),
        )
        def add_default(x: torch.Tensor, value: int = 1) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] + value
            return out

        x = torch.zeros(64, device=DEVICE)
        bound = add_default.bind((x, 1))
        shared_state = (
            tuple(add_default._bound_kernels.items()),  # type: ignore[attr-defined]
            tuple(add_default._specialization_aliases.items()),  # type: ignore[attr-defined]
            tuple(add_default._specialize_extra.items()),  # type: ignore[attr-defined]
        )
        with patch.object(torch.compiler, "is_compiling", return_value=True):
            isolated = add_default.bind((x,))
        self.assertIsNot(isolated, bound)
        self.assertFalse(isolated._cache_managed)
        output_spec = infer_output_spec(add_default, (x, 1))
        self.assertEqual(output_spec["leaf_specs"][0]["shape"], [64])
        self.assertEqual(
            (
                tuple(add_default._bound_kernels.items()),  # type: ignore[attr-defined]
                tuple(add_default._specialization_aliases.items()),  # type: ignore[attr-defined]
                tuple(add_default._specialize_extra.items()),  # type: ignore[attr-defined]
            ),
            shared_state,
        )

    def test_isolated_bind_cache_key_is_read_only(self) -> None:
        scale = _make_scale()
        x = torch.randn(64, device=DEVICE)

        bound = scale._bind_isolated((x, 2))  # type: ignore[attr-defined]
        self.assertFalse(bound._cache_managed)
        scale._create_bound_kernel_cache_key(  # type: ignore[attr-defined]
            bound,
            (x, 2),
            bound._base_spec_key,
        )
        torch.testing.assert_close(bound(x, 2), x * 2)

        self.assertFalse(scale._bound_kernels)  # type: ignore[attr-defined]
        self.assertFalse(scale._dispatch_cache)  # type: ignore[attr-defined]
        self.assertFalse(scale._specialize_extra)  # type: ignore[attr-defined]

    def test_reset_rediscovers_conditional_specialization_schema(self) -> None:
        global _RESET_SPECIALIZATION_USES_X

        @helion.kernel(
            static_shapes=False,
            config=helion.Config(block_sizes=[64]),
        )
        def conditional_scale(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            selected = _select_reset_specialization_tensor(x, y)
            value = hl.specialize(selected.size(0))
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] + value
            return out

        x32 = torch.randn(32, device=DEVICE)
        x48 = torch.randn(48, device=DEVICE)
        y64 = torch.randn(64, device=DEVICE)
        try:
            _RESET_SPECIALIZATION_USES_X = False
            old = conditional_scale.bind((x32, y64))
            signature = old._base_spec_key
            old_extractors = conditional_scale._specialize_extra[signature]  # type: ignore[attr-defined]
            self.assertEqual(tuple(fn((x32, y64)) for fn in old_extractors), (64,))

            conditional_scale.reset()
            _RESET_SPECIALIZATION_USES_X = True
            first = conditional_scale.bind((x32, y64))
            new_extractors = conditional_scale._specialize_extra[signature]  # type: ignore[attr-defined]
            self.assertEqual(tuple(fn((x32, y64)) for fn in new_extractors), (32,))

            second = conditional_scale.bind((x48, y64))
            self.assertEqual(first._base_spec_key, second._base_spec_key)
            self.assertIsNot(first, second)
            self.assertEqual(len(conditional_scale._bound_kernels), 2)  # type: ignore[attr-defined]
        finally:
            _RESET_SPECIALIZATION_USES_X = False
            conditional_scale.reset()

    def test_stale_bound_does_not_republish_schema_after_reset(self) -> None:
        scale = _make_scale()
        x = torch.randn(64, device=DEVICE)
        torch.testing.assert_close(scale(x, 2), x * 2)
        stale_bound = scale._prepared_call.bound  # type: ignore[attr-defined]
        signature = stale_bound._base_spec_key

        scale.reset()
        cache_key = scale._create_bound_kernel_cache_key(  # type: ignore[attr-defined]
            stale_bound,
            (x, 2),
            signature,
        )

        self.assertEqual(cache_key.extra_results, (2,))
        self.assertFalse(scale._specialize_extra)  # type: ignore[attr-defined]
        self.assertFalse(scale._has_specialization_extras)  # type: ignore[attr-defined]

    def test_reset_preserves_inflight_omitted_default_alias(self) -> None:
        @helion.kernel(
            static_shapes=True,
            config=helion.Config(block_sizes=[64]),
        )
        def add_default(x: torch.Tensor, value: int = 2) -> torch.Tensor:
            value = hl.specialize(value)
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] + value
            return out

        x = torch.randn(64, device=DEVICE)
        add_default.bind((x, 2))
        add_default.bind((x,))
        signature = add_default._base_specialization_key((x,))  # type: ignore[attr-defined]
        alias = add_default._specialization_aliases[signature]  # type: ignore[attr-defined]
        alias_type = type(alias)
        original_call = alias_type.__call__
        alias_entered = threading.Event()
        release_alias = threading.Event()

        def blocking_call(current, values):
            alias_entered.set()
            self.assertTrue(release_alias.wait(5))
            return original_call(current, values)

        with (
            patch.object(alias_type, "__call__", new=blocking_call),
            ThreadPoolExecutor(max_workers=1) as pool,
        ):
            key = pool.submit(
                add_default._get_bound_kernel_cache_key,  # type: ignore[attr-defined]
                (x,),
                signature,
            )
            try:
                self.assertTrue(alias_entered.wait(5))
                add_default.reset()
            finally:
                release_alias.set()
            self.assertIsNone(key.result(timeout=5))

        self.assertFalse(add_default._specialize_extra)  # type: ignore[attr-defined]
        self.assertFalse(add_default._specialization_aliases)  # type: ignore[attr-defined]

    def test_isolated_bind_rejects_too_many_args_without_caching(self) -> None:
        add_one = _make_add_one()
        x = torch.empty(1, device=DEVICE)

        with self.assertRaisesRegex(
            TypeError,
            r"Too many arguments passed to the kernel, expected: 1 got: 2\.",
        ):
            add_one._bind_isolated((x, x))

        self.assertFalse(add_one._bound_kernels)

    @unittest.skipUnless(
        supports_torch_compile_fusion(),
        "requires Helion's torch.compile fusion integration",
    )
    def test_inductor_lowering_bind_does_not_wait_for_bind_lock(self) -> None:
        from helion._compiler._inductor.template_buffer import _bind_kernel_for_lowering

        add_one = _make_add_one()
        x = torch.randn(64, device=DEVICE)
        isolated_bound = add_one._bind_isolated((x,))
        self.assertFalse(isolated_bound._cache_managed)
        lock_held = threading.Event()
        release_lock = threading.Event()

        def hold_bind_lock() -> None:
            with add_one._bind_lock:
                lock_held.set()
                self.assertTrue(release_lock.wait(5))

        with (
            patch.object(
                add_one,
                "_bind_isolated",
                return_value=isolated_bound,
            ) as isolated_bind,
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            holder = pool.submit(hold_bind_lock)
            self.assertTrue(lock_held.wait(5))
            binding = pool.submit(_bind_kernel_for_lowering, add_one, (x,))
            try:
                self.assertIs(binding.result(timeout=2), isolated_bound)
            finally:
                release_lock.set()
            holder.result(timeout=5)
            isolated_bind.assert_called_once_with((x,))

        self.assertFalse(add_one._bound_kernels)

        managed_bound = _bind_kernel_for_lowering(add_one, (x,))
        self.assertTrue(managed_bound._cache_managed)
        self.assertIn(managed_bound, add_one._bound_kernels.values())

    def test_cache_key_generation_preserves_late_specializations(self) -> None:
        scale = _make_scale()
        x = torch.randn(64, device=DEVICE)
        torch.testing.assert_close(scale(x, 2), x * 2)
        prepared = scale._prepared_call
        assert prepared is not None
        bound = prepared.bound
        signature = bound._base_spec_key
        scale._extend_bound_kernel_specializations(
            bound,
            signature,
            [_tensor_size],
            (x, 2),
        )
        schema = tuple(scale._specialize_extra[signature])
        self.assertLess(len(bound._specialize_extra()), len(schema))

        cache_key = scale._create_bound_kernel_cache_key(
            bound,
            (x, 2),
            signature,
        )

        self.assertEqual(tuple(scale._specialize_extra[signature]), schema)
        self.assertEqual(cache_key.extra_results, (2, 64))

    def test_cache_key_generation_waits_for_specialization_extension(self) -> None:
        scale = _make_scale()
        x = torch.randn(64, device=DEVICE)
        torch.testing.assert_close(scale(x, 2), x * 2)
        prepared = scale._prepared_call
        assert prepared is not None
        bound = prepared.bound
        signature = bound._base_spec_key
        extension_entered = threading.Event()
        release_extension = threading.Event()
        key_lock_attempted = threading.Event()
        lock = threading.Lock()

        class TrackedLock:
            def __enter__(self) -> Self:
                if lock.locked():
                    key_lock_attempted.set()
                lock.acquire()
                return self

            def __exit__(
                self,
                exc_type: object,
                exc_value: object,
                traceback: object,
            ) -> None:
                lock.release()

        def blocking_size(values: Sequence[object]) -> int:
            extension_entered.set()
            self.assertTrue(release_extension.wait(5))
            return _tensor_size(values)

        with (
            patch.object(scale, "_specialize_extra_lock", TrackedLock()),
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            extension = pool.submit(
                scale._extend_bound_kernel_specializations,
                bound,
                signature,
                [blocking_size],
                (x, 2),
            )
            cache_key = None
            try:
                self.assertTrue(extension_entered.wait(5))
                cache_key = pool.submit(
                    scale._create_bound_kernel_cache_key,
                    bound,
                    (x, 2),
                    signature,
                )
                self.assertTrue(key_lock_attempted.wait(5))
                self.assertFalse(cache_key.done())
            finally:
                release_extension.set()

            self.assertTrue(extension.result(timeout=5))
            assert cache_key is not None
            result = cache_key.result(timeout=5)

        self.assertEqual(result.extra_results, (2, 64))

    def test_cache_key_generation_retries_after_specialization_extension(self) -> None:
        scale = _make_scale()
        x = torch.randn(64, device=DEVICE)
        torch.testing.assert_close(scale(x, 2), x * 2)
        prepared = scale._prepared_call
        assert prepared is not None
        bound = prepared.bound
        signature = bound._base_spec_key
        key_entered = threading.Event()
        release_key = threading.Event()
        key_thread_id: int | None = None

        def blocking_value(values: Sequence[object]) -> int:
            nonlocal key_thread_id
            if key_thread_id is None:
                key_thread_id = threading.get_ident()
                key_entered.set()
            if threading.get_ident() == key_thread_id:
                self.assertTrue(release_key.wait(5))
            return cast("int", values[1])

        scale._specialize_extra[signature] = [blocking_value]
        with ThreadPoolExecutor(max_workers=2) as pool:
            cache_key = pool.submit(
                scale._create_bound_kernel_cache_key,
                bound,
                (x, 2),
                signature,
            )
            try:
                self.assertTrue(key_entered.wait(5))
                extension = pool.submit(
                    scale._extend_bound_kernel_specializations,
                    bound,
                    signature,
                    [_tensor_size],
                    (x, 2),
                )
                self.assertTrue(extension.result(timeout=5))
                self.assertFalse(cache_key.done())
            finally:
                release_key.set()

            result = cache_key.result(timeout=5)

        self.assertEqual(result.extra_results, (2, 64))

    def test_specialization_key_retries_after_specialization_extension(self) -> None:
        scale = _make_scale()
        x = torch.randn(64, device=DEVICE)
        torch.testing.assert_close(scale(x, 2), x * 2)
        prepared = scale._prepared_call
        assert prepared is not None
        bound = prepared.bound
        signature = bound._base_spec_key
        key_entered = threading.Event()
        release_key = threading.Event()
        key_thread_id: int | None = None

        def blocking_value(values: Sequence[object]) -> int:
            nonlocal key_thread_id
            if key_thread_id is None:
                key_thread_id = threading.get_ident()
                key_entered.set()
            if threading.get_ident() == key_thread_id:
                self.assertTrue(release_key.wait(5))
            return cast("int", values[1])

        scale._specialize_extra[signature] = [blocking_value]
        with ThreadPoolExecutor(max_workers=2) as pool:
            specialization_key = pool.submit(scale.specialization_key, (x, 2))
            try:
                self.assertTrue(key_entered.wait(5))
                extension = pool.submit(
                    scale._extend_bound_kernel_specializations,
                    bound,
                    signature,
                    [_tensor_size],
                    (x, 2),
                )
                self.assertTrue(extension.result(timeout=5))
                self.assertFalse(specialization_key.done())
            finally:
                release_key.set()

            result = specialization_key.result(timeout=5)

        self.assertEqual(result[-2:], (2, 64))

    def test_cache_key_generation_does_not_wait_for_bind_lock(self) -> None:
        scale = _make_scale()
        x = torch.randn(64, device=DEVICE)
        torch.testing.assert_close(scale(x, 2), x * 2)
        prepared = scale._prepared_call
        assert prepared is not None
        bound = prepared.bound
        lock_held = threading.Event()
        release_lock = threading.Event()

        def hold_bind_lock() -> None:
            with scale._bind_lock:
                lock_held.set()
                self.assertTrue(release_lock.wait(5))

        with ThreadPoolExecutor(max_workers=2) as pool:
            holder = pool.submit(hold_bind_lock)
            self.assertTrue(lock_held.wait(5))
            cache_key = pool.submit(
                scale._create_bound_kernel_cache_key,
                bound,
                (x, 2),
                bound._base_spec_key,
            )
            try:
                result = cache_key.result(timeout=2)
            finally:
                release_lock.set()
            holder.result(timeout=5)

        self.assertEqual(result.extra_results, (2,))

    def test_stale_keyed_dispatch_generation_is_rebuilt(self) -> None:
        aot_scale = _make_aot_scale()
        x = torch.randn(64, device=DEVICE)
        torch.testing.assert_close(aot_scale(x, 2), x * 2)
        bound = next(iter(aot_scale._dispatch_cache.values()))  # type: ignore[attr-defined]
        stale_entry = aot_scale._fast_dispatch_key_and_guards((x, 2))  # type: ignore[attr-defined]
        assert stale_entry is not None
        with aot_scale._bind_lock:  # type: ignore[attr-defined]
            aot_scale._specialization_generation += 1  # type: ignore[attr-defined]
        self.assertIsNone(  # type: ignore[attr-defined]
            aot_scale._prepare_dispatch_entry((x, 2), bound, stale_entry)
        )

        with patch.object(
            aot_scale,
            "_prepare_dispatch_entry",
            wraps=aot_scale._prepare_dispatch_entry,  # type: ignore[attr-defined]
        ) as prepare:
            torch.testing.assert_close(aot_scale(x, 2), x * 2)

        prepare.assert_called_once()
        self.assertEqual(  # type: ignore[attr-defined]
            bound._dispatch_generation,
            aot_scale._specialization_generation,
        )

        with patch.object(
            aot_scale,
            "_prepare_dispatch_entry",
            side_effect=AssertionError("rebuilt generation was not reused"),
        ):
            torch.testing.assert_close(aot_scale(x, 2), x * 2)

        aot_scale.reset()
        self.assertNotEqual(
            bound._dispatch_generation,
            aot_scale._specialization_generation,  # type: ignore[attr-defined]
        )

    def test_distributed_aot_late_specialization_stays_locked(self) -> None:
        aot_scale = _make_aot_scale(distributed=True)
        x = torch.randn(64, device=DEVICE)
        torch.testing.assert_close(aot_scale(x, 2), x * 2)
        bound = next(iter(aot_scale._dispatch_cache.values()))  # type: ignore[attr-defined]
        self.assertIsNone(bound._dispatch_generation)

        with patch.object(
            aot_scale,
            "_prepare_dispatch_entry",
            wraps=aot_scale._prepare_dispatch_entry,  # type: ignore[attr-defined]
        ) as prepare:
            torch.testing.assert_close(aot_scale(x, 2), x * 2)

        prepare.assert_called_once()
        self.assertIsNone(bound._dispatch_generation)

    def test_process_group_aot_dispatch_stays_locked(self) -> None:
        @helion.aot_kernel(
            config=helion.Config(block_sizes=[64]),
            key=lambda _x, _group_name: 0,
        )
        def add_one(x: torch.Tensor, group_name: hl.ProcessGroupName) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] + 1
            return out

        x = torch.randn(64, device=DEVICE)
        torch.testing.assert_close(add_one(x, "unused"), x + 1)
        bound = next(iter(add_one._dispatch_cache.values()))  # type: ignore[attr-defined]
        self.assertIsNone(bound._dispatch_generation)
        with patch.object(
            add_one,
            "_prepare_dispatch_entry",
            wraps=add_one._prepare_dispatch_entry,  # type: ignore[attr-defined]
        ) as prepare:
            torch.testing.assert_close(add_one(x, "unused"), x + 1)

        prepare.assert_called_once()

    def test_compiler_capture_does_not_inspect_prepared_guard(self) -> None:
        add_one = _make_add_one()
        x = torch.randn(64, device=DEVICE)
        add_one(x)

        with (
            patch.object(
                torch.compiler, "is_compiling", return_value=True
            ) as is_compiling,
            patch.object(
                kernel_module._PreparedCall,
                "matches",
                side_effect=AssertionError("capture inspected prepared state"),
            ),
        ):
            out = add_one(x)
        is_compiling.assert_called_once_with()
        torch.testing.assert_close(out, x + 1)

    @unittest.skipUnless(
        supports_torch_compile_fusion(),
        "requires Helion's torch.compile fusion integration",
    )
    def test_fullgraph_capture_preserves_eager_caches(self) -> None:
        @helion.kernel(
            static_shapes=True,
            config=helion.Config(block_sizes=[64]),
        )
        def add_default(x: torch.Tensor, value: int = 1) -> torch.Tensor:
            value = hl.specialize(value)
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] + value
            return out

        def snapshot() -> tuple[object, ...]:
            return (
                tuple(add_default._bound_kernels.items()),  # type: ignore[attr-defined]
                tuple(add_default._dispatch_cache.items()),  # type: ignore[attr-defined]
                tuple(add_default._specialization_aliases.items()),  # type: ignore[attr-defined]
                tuple(  # type: ignore[attr-defined]
                    (key, tuple(extractors))
                    for key, extractors in add_default._specialize_extra.items()
                ),
            )

        cold_states: list[tuple[object, ...]] = []

        def cold_backend(
            graph: torch.fx.GraphModule,
            _example_inputs: list[torch.Tensor],
        ) -> Callable[..., object]:
            cold_states.append(snapshot())
            return graph.forward

        def cold_call(x: torch.Tensor) -> torch.Tensor:
            return add_default(x)

        x = torch.randn(64, device=DEVICE)
        compiled_cold = torch.compile(
            cold_call,
            backend=cold_backend,
            fullgraph=True,
        )
        torch.testing.assert_close(compiled_cold(x), x + 1)
        empty_state = ((), (), (), ())
        self.assertEqual(cold_states, [empty_state])

        eager_state = snapshot()
        self.assertNotEqual(eager_state, empty_state)
        warm_states: list[tuple[object, ...]] = []

        def warm_backend(
            graph: torch.fx.GraphModule,
            _example_inputs: list[torch.Tensor],
        ) -> Callable[..., object]:
            warm_states.append(snapshot())
            return graph.forward

        def warm_call(x: torch.Tensor) -> torch.Tensor:
            return add_default(x)

        compiled_warm = torch.compile(
            warm_call,
            backend=warm_backend,
            fullgraph=True,
        )
        torch.testing.assert_close(compiled_warm(x), x + 1)
        self.assertEqual(warm_states, [eager_state])
        self.assertEqual(snapshot(), eager_state)

    def test_unsupported_signature_does_not_rebind_after_run(self) -> None:
        @helion.kernel(
            static_shapes=True,
            config=helion.Config(block_sizes=[64]),
        )
        def add_list(xs: list[torch.Tensor]) -> torch.Tensor:
            out = torch.empty_like(xs[0])
            for tile in hl.tile(xs[0].size(0)):
                out[tile] = xs[0][tile] + xs[1][tile]
            return out

        x = torch.randn(64, device=DEVICE)
        y = torch.randn_like(x)
        with patch.object(add_list, "_bind", wraps=add_list._bind) as bind:  # type: ignore[attr-defined]
            out = add_list([x, y])
        self.assertEqual(bind.call_count, 1)
        torch.testing.assert_close(out, x + y)

    def test_distributed_state_change_rechecks_dispatch(self) -> None:
        add_one = _make_add_one()
        x = torch.randn(64, device=DEVICE)
        add_one(x)

        with (
            patch.object(kernel_module.dist, "is_initialized", return_value=True),
            patch.object(kernel_module, "kernel_uses_symm_mem", return_value=False),
            patch.object(
                add_one,
                "_fast_dispatch_key_and_guards",
                wraps=add_one._fast_dispatch_key_and_guards,  # type: ignore[attr-defined]
            ) as fast_key,
        ):
            out = add_one(x)
        fast_key.assert_called()
        torch.testing.assert_close(out, x + 1)

    def test_direct_keyed_hit_rechecks_distributed_transition(self) -> None:
        @helion.aot_kernel(
            config=helion.Config(block_sizes=[64]),
            key=lambda _x: 0,
        )
        def add_one(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] + 1
            return out

        x = torch.randn(64, device=DEVICE)
        torch.testing.assert_close(add_one(x), x + 1)

        with (
            patch.object(
                kernel_module.dist,
                "is_initialized",
                side_effect=(False, True, True, True),
            ),
            patch.object(kernel_module, "kernel_uses_symm_mem", return_value=False),
            patch.object(
                add_one,
                "_prepare_dispatch_entry",
                wraps=add_one._prepare_dispatch_entry,  # type: ignore[attr-defined]
            ) as prepare,
        ):
            out = add_one(x)

        prepare.assert_called_once()
        torch.testing.assert_close(out, x + 1)

    def test_publication_rejects_distributed_state_transition(self) -> None:
        add_one = _make_add_one(distributed=True)
        x = torch.randn(64, device=DEVICE)
        add_one(x)
        bound = add_one._prepared_call.bound  # type: ignore[attr-defined]
        fast_entry = add_one._fast_dispatch_key_and_guards((x,))  # type: ignore[attr-defined]
        assert fast_entry is not None

        with patch.object(
            kernel_module.dist,
            "is_initialized",
            side_effect=(False, True),
        ) as is_initialized:
            self.assertIsNone(  # type: ignore[attr-defined]
                add_one._prepare_dispatch_entry((x,), bound, fast_entry)
            )
        self.assertEqual(is_initialized.call_count, 2)

        with patch.object(kernel_module.dist, "is_initialized", return_value=True):
            self.assertIsNone(  # type: ignore[attr-defined]
                add_one._prepare_dispatch_entry((x,), bound, fast_entry)
            )

    def test_omitted_default_publication_rechecks_raw_args(self) -> None:
        @helion.kernel(
            static_shapes=True,
            config=helion.Config(block_sizes=[64]),
        )
        def add_default(x: torch.Tensor, value: int = 1) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] + value
            return out

        x = torch.randn(64, device=DEVICE)
        torch.testing.assert_close(add_default(x), x + 1)
        prepared = add_default._prepared_call
        assert prepared is not None
        fast_entry = add_default._fast_dispatch_key_and_guards((x,))
        assert fast_entry is not None
        calls: list[tuple[tuple[object, ...], bool | None]] = []

        def require_raw_args(
            args: Sequence[object], *, dist_initialized: bool | None = None
        ) -> bool:
            calls.append((tuple(args), dist_initialized))
            self.assertEqual(len(args), 1)
            self.assertIs(args[0], x)
            return False

        with patch.object(
            add_default,
            "_compute_is_distributed",
            side_effect=require_raw_args,
        ):
            entry = add_default._prepare_dispatch_entry(
                (x,), prepared.bound, fast_entry
            )

        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertIsNotNone(entry[0])
        self.assertFalse(entry[1])
        self.assertEqual(len(calls), 1)

    def test_custom_key_hit_rejects_distributed_state_transition(self) -> None:
        calls = 0

        def key(_x: torch.Tensor) -> int:
            nonlocal calls
            calls += 1
            return 0

        @helion.kernel(
            static_shapes=True,
            config=helion.Config(block_sizes=[64]),
            distributed=True,
            key=key,
        )
        def add_one(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] + 1
            return out

        x = torch.randn(64, device=DEVICE)
        add_one(x)
        bound = next(iter(add_one._dispatch_cache.values()))  # type: ignore[attr-defined]
        calls = 0
        fast_entry = add_one._fast_dispatch_key_and_guards((x,))  # type: ignore[attr-defined]
        assert fast_entry is not None
        with patch.object(
            kernel_module.dist,
            "is_initialized",
            side_effect=(False, True),
        ):
            self.assertIsNone(  # type: ignore[attr-defined]
                add_one._prepare_dispatch_entry((x,), bound, fast_entry)
            )
        self.assertEqual(calls, 1)

    def test_custom_key_distributed_bound_rechecks_dispatch(self) -> None:
        @helion.kernel(
            static_shapes=True,
            config=helion.Config(block_sizes=[64]),
            key=lambda _x: 0,
        )
        def add_one(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] + 1
            return out

        x = torch.randn(64, device=DEVICE)
        with patch.object(kernel_module, "kernel_uses_symm_mem", return_value=True):
            torch.testing.assert_close(add_one(x), x + 1)
            bound = next(iter(add_one._dispatch_cache.values()))  # type: ignore[attr-defined]
            self.assertTrue(bound._env._is_distributed)
            self.assertIsNone(bound._dispatch_generation)
            with patch.object(
                add_one,
                "_prepare_dispatch_entry",
                wraps=add_one._prepare_dispatch_entry,  # type: ignore[attr-defined]
            ) as prepare:
                out = add_one(x)

        prepare.assert_called_once()
        torch.testing.assert_close(out, x + 1)

    def test_alternating_cached_specializations_prepare_latest(self) -> None:
        add_one = _make_add_one()
        x64 = torch.randn(64, device=DEVICE)
        x65 = torch.randn(65, device=DEVICE)
        add_one(x64)
        add_one(x65)

        first_64 = torch.randn_like(x64)
        with patch.object(
            add_one,
            "_fast_dispatch_key_and_guards",
            wraps=add_one._fast_dispatch_key_and_guards,  # type: ignore[attr-defined]
        ) as fast_key:
            torch.testing.assert_close(add_one(first_64), first_64 + 1)
        fast_key.assert_called()

        second_64 = torch.randn_like(x64)
        with patch.object(
            add_one,
            "_fast_dispatch_key_and_guards",
            side_effect=AssertionError("latest specialization was not prepared"),
        ):
            torch.testing.assert_close(add_one(second_64), second_64 + 1)

    def test_dispatch_hit_prepares_without_repeating_extractors(self) -> None:
        add_one = _make_add_one()
        x = torch.randn(64, device=DEVICE)
        add_one(x)
        bound = add_one._prepared_call.bound  # type: ignore[attr-defined]
        calls = 0

        def counting_extractor(_args: Sequence[object]) -> int:
            nonlocal calls
            calls += 1
            return 1

        add_one._extend_bound_kernel_specializations(  # type: ignore[attr-defined]
            bound,
            bound._base_spec_key,
            [counting_extractor],
            (x,),
        )
        torch.testing.assert_close(add_one(x), x + 1)
        add_one._prepared_call = None  # type: ignore[attr-defined]
        calls = 0

        torch.testing.assert_close(add_one(x), x + 1)
        self.assertEqual(calls, 1)
        self.assertIsNotNone(add_one._prepared_call)  # type: ignore[attr-defined]

    def test_specialization_extension_cannot_leave_stale_preparation(self) -> None:
        add_one = _make_add_one()
        x = torch.randn(64, device=DEVICE)
        add_one(x)
        bound = add_one._prepared_call.bound  # type: ignore[attr-defined]
        add_one._prepared_call = None  # type: ignore[attr-defined]

        build_entered = threading.Event()
        release_build = threading.Event()
        extension_started = threading.Event()
        extension_done = threading.Event()
        original_build = kernel_module._PreparedCall.build

        def blocking_build(*args: object, **kwargs: object) -> object:
            build_entered.set()
            self.assertTrue(release_build.wait(5))
            return original_build(*args, **kwargs)  # type: ignore[arg-type]

        def extend() -> None:
            extension_started.set()
            add_one._extend_bound_kernel_specializations(  # type: ignore[attr-defined]
                bound,
                bound._base_spec_key,
                [_tensor_size],
                (x,),
            )
            extension_done.set()

        with (
            patch.object(
                kernel_module._PreparedCall, "build", side_effect=blocking_build
            ),
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            call = pool.submit(add_one, x)
            self.assertTrue(build_entered.wait(5))
            extension = pool.submit(extend)
            self.assertTrue(extension_started.wait(5))
            self.assertFalse(extension_done.wait(0.05))
            release_build.set()
            torch.testing.assert_close(call.result(timeout=5), x + 1)
            extension.result(timeout=5)

        self.assertIsNone(add_one._prepared_call)  # type: ignore[attr-defined]
        self.assertFalse(add_one._dispatch_cache)  # type: ignore[attr-defined]

    def test_failing_specialization_extension_is_transactional(self) -> None:
        add_one = _make_add_one()
        x = torch.randn(64, device=DEVICE)
        add_one(x)
        prepared = add_one._prepared_call  # type: ignore[attr-defined]
        assert prepared is not None
        generation = add_one._specialization_generation  # type: ignore[attr-defined]
        signature = prepared.bound._base_spec_key
        original_extractors = tuple(  # type: ignore[attr-defined]
            add_one._specialize_extra[signature]
        )

        def fail(_args: Sequence[object]) -> int:
            raise RuntimeError("late specialization failed")

        with self.assertRaisesRegex(RuntimeError, "late specialization failed"):
            add_one._extend_bound_kernel_specializations(  # type: ignore[attr-defined]
                prepared.bound,
                signature,
                [fail],
                (x,),
            )

        def unhashable(_args: Sequence[object]) -> Hashable:
            return cast("Hashable", [])

        with self.assertRaisesRegex(TypeError, "unhashable"):
            add_one._extend_bound_kernel_specializations(  # type: ignore[attr-defined]
                prepared.bound,
                signature,
                [unhashable],
                (x,),
            )

        self.assertEqual(  # type: ignore[attr-defined]
            tuple(add_one._specialize_extra[signature]), original_extractors
        )
        self.assertEqual(add_one._specialization_generation, generation)  # type: ignore[attr-defined]
        self.assertIs(add_one._prepared_call, prepared)  # type: ignore[attr-defined]
        with patch.object(
            add_one,
            "_fast_dispatch_key_and_guards",
            side_effect=AssertionError("transaction invalidated prepared call"),
        ):
            torch.testing.assert_close(add_one(x), x + 1)

    def test_concurrent_reset_cannot_republish_old_bound(self) -> None:
        add_one = _make_add_one()
        x = torch.randn(64, device=DEVICE)
        add_one(x)
        bound = add_one._prepared_call.bound  # type: ignore[attr-defined]
        add_one._prepared_call = None  # type: ignore[attr-defined]
        add_one._dispatch_cache.clear()  # type: ignore[attr-defined]

        call_finished_kernel = threading.Event()
        release_call = threading.Event()
        bound_type = type(bound)
        original_call = bound_type.__call__

        def blocking_call(
            current: object,
            *args: object,
            **kwargs: object,
        ) -> object:
            result = original_call(current, *args, **kwargs)
            call_finished_kernel.set()
            self.assertTrue(release_call.wait(5))
            return result

        with (
            patch.object(bound_type, "__call__", new=blocking_call),
            ThreadPoolExecutor(max_workers=1) as pool,
        ):
            call = pool.submit(add_one, x)
            try:
                self.assertTrue(call_finished_kernel.wait(5))
                add_one.reset()
            finally:
                release_call.set()
            torch.testing.assert_close(call.result(timeout=5), x + 1)

        self.assertIsNone(add_one._prepared_call)  # type: ignore[attr-defined]
        self.assertFalse(add_one._dispatch_cache)  # type: ignore[attr-defined]
        self.assertFalse(add_one._bound_kernels)  # type: ignore[attr-defined]

    def test_reset_rejects_late_specialization_from_old_bound(self) -> None:
        add_one = _make_add_one()
        x = torch.randn(64, device=DEVICE)
        add_one(x)
        bound = add_one._prepared_call.bound  # type: ignore[attr-defined]

        add_one.reset()
        self.assertFalse(  # type: ignore[attr-defined]
            add_one._extend_bound_kernel_specializations(
                bound,
                bound._base_spec_key,
                [_tensor_size],
                (x,),
            )
        )
        self.assertFalse(add_one._bound_kernels)  # type: ignore[attr-defined]

    def test_backend_capability_is_explicit_and_inherited(self) -> None:
        self.assertTrue(TritonBackend().supports_eager_prepared_call)
        self.assertTrue(TileIRBackend().supports_eager_prepared_call)
        self.assertTrue(CuteBackend().supports_eager_prepared_call)
        self.assertFalse(PallasBackend().supports_eager_prepared_call)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
