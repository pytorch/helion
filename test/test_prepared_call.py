"""Tests for the shared Triton/CuTe monomorphic prepared-call path."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib
import threading
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import torch

import helion
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
    from collections.abc import Sequence
    from typing import Hashable

kernel_module = importlib.import_module("helion.runtime.kernel")


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


def _tensor_size(values: Sequence[object]) -> int:
    value = values[0]
    assert isinstance(value, torch.Tensor)
    return value.size(0)


@onlyBackends(["triton", "cute"])
class TestPreparedCall(RefEagerTestDisabled, TestCase):
    def test_fresh_tensor_bypasses_dispatch_key(self) -> None:
        add_one = _make_add_one()
        x = torch.randn(64, device=DEVICE)
        first = add_one(x)
        self.assertIsNotNone(add_one._prepared_call)  # type: ignore[attr-defined]

        other = torch.randn_like(x)
        with (
            patch.object(
                add_one,
                "_fast_dispatch_key",
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
                    "_fast_dispatch_key",
                    wraps=add_one._fast_dispatch_key,  # type: ignore[attr-defined]
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

        @helion.kernel(
            static_shapes=True,
            config=helion.Config(block_sizes=[64]),
        )
        def scale_specialized(x: torch.Tensor, value: int) -> torch.Tensor:
            value = hl.specialize(value)
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] * value
            return out

        x = torch.randn(64, device=DEVICE)
        torch.testing.assert_close(scale(x, 2.0), x * 2.0)
        with patch.object(
            scale,
            "_fast_dispatch_key",
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
                "_fast_dispatch_key",
                wraps=kernel._fast_dispatch_key,  # type: ignore[attr-defined]
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
        fast_key = add_one._fast_dispatch_key((different_shape,))  # type: ignore[attr-defined]
        self.assertIsNotNone(fast_key)
        self.assertIsNone(  # type: ignore[attr-defined]
            add_one._prepare_dispatch_entry(
                (different_shape,), bound, known_fast_key=fast_key
            )
        )
        self.assertEqual(calls, 1)

        calls = 0
        torch.testing.assert_close(add_one(x), x + 1)
        self.assertEqual(calls, 1)
        self.assertIsNone(add_one._prepared_call)  # type: ignore[attr-defined]

        state["key"] = 1
        torch.testing.assert_close(add_one(x), x + 1)
        self.assertIsNone(add_one._prepared_call)  # type: ignore[attr-defined]

    def test_compiler_capture_does_not_inspect_prepared_guard(self) -> None:
        add_one = _make_add_one()
        x = torch.randn(64, device=DEVICE)
        add_one(x)

        with (
            patch.object(torch.compiler, "is_compiling", return_value=True),
            patch.object(
                kernel_module._PreparedCall,
                "matches",
                side_effect=AssertionError("capture inspected prepared state"),
            ),
        ):
            out = add_one(x)
        torch.testing.assert_close(out, x + 1)

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
                "_fast_dispatch_key",
                wraps=add_one._fast_dispatch_key,  # type: ignore[attr-defined]
            ) as fast_key,
        ):
            out = add_one(x)
        fast_key.assert_called()
        torch.testing.assert_close(out, x + 1)

    def test_publication_rejects_distributed_state_transition(self) -> None:
        add_one = _make_add_one(distributed=True)
        x = torch.randn(64, device=DEVICE)
        add_one(x)
        bound = add_one._prepared_call.bound  # type: ignore[attr-defined]

        with patch.object(
            kernel_module.dist,
            "is_initialized",
            side_effect=(False, True),
        ) as is_initialized:
            self.assertIsNone(  # type: ignore[attr-defined]
                add_one._prepare_dispatch_entry((x,), bound)
            )
        self.assertEqual(is_initialized.call_count, 2)

        with patch.object(kernel_module.dist, "is_initialized", return_value=True):
            self.assertIsNone(  # type: ignore[attr-defined]
                add_one._prepare_dispatch_entry((x,), bound)
            )

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
        with patch.object(
            kernel_module.dist,
            "is_initialized",
            side_effect=(False, True),
        ):
            fast_key = add_one._fast_dispatch_key((x,))  # type: ignore[attr-defined]
            assert fast_key is not None
            self.assertIsNone(  # type: ignore[attr-defined]
                add_one._prepare_dispatch_entry((x,), bound, known_fast_key=fast_key)
            )
        self.assertEqual(calls, 1)

    def test_alternating_cached_specializations_prepare_latest(self) -> None:
        add_one = _make_add_one()
        x64 = torch.randn(64, device=DEVICE)
        x65 = torch.randn(65, device=DEVICE)
        add_one(x64)
        add_one(x65)

        first_64 = torch.randn_like(x64)
        with patch.object(
            add_one,
            "_fast_dispatch_key",
            wraps=add_one._fast_dispatch_key,  # type: ignore[attr-defined]
        ) as fast_key:
            torch.testing.assert_close(add_one(first_64), first_64 + 1)
        fast_key.assert_called()

        second_64 = torch.randn_like(x64)
        with patch.object(
            add_one,
            "_fast_dispatch_key",
            side_effect=AssertionError("latest specialization was not prepared"),
        ):
            torch.testing.assert_close(add_one(second_64), second_64 + 1)

    def test_preparation_failure_falls_back_after_dispatch_hit(self) -> None:
        add_one = _make_add_one()
        x = torch.randn(64, device=DEVICE)
        add_one(x)
        bound = add_one._prepared_call.bound  # type: ignore[attr-defined]
        calls = 0

        def flaky_extractor(_args: Sequence[object]) -> int:
            nonlocal calls
            calls += 1
            if calls % 2 == 0:
                raise RuntimeError("second evaluation failed")
            return 1

        add_one._specialize_extra[bound._base_spec_key] = [flaky_extractor]  # type: ignore[attr-defined]
        add_one._has_specialization_extras = True  # type: ignore[attr-defined]
        fast_key = add_one._fast_dispatch_key((x,))  # type: ignore[attr-defined]
        assert fast_key is not None
        add_one._dispatch_cache = {fast_key: bound}  # type: ignore[attr-defined]
        add_one._prepared_call = None  # type: ignore[attr-defined]
        calls = 0

        torch.testing.assert_close(add_one(x), x + 1)
        self.assertEqual(calls, 2)
        self.assertIsNone(add_one._prepared_call)  # type: ignore[attr-defined]

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
        self.assertIs(add_one._prepared_call, prepared)  # type: ignore[attr-defined]
        with patch.object(
            add_one,
            "_fast_dispatch_key",
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

        def blocking_call(current: object, *args: object, **kwargs: object) -> object:
            result = original_call(current, *args, **kwargs)
            call_finished_kernel.set()
            self.assertTrue(release_call.wait(5))
            return result

        with (
            patch.object(bound_type, "__call__", new=blocking_call),
            ThreadPoolExecutor(max_workers=1) as pool,
        ):
            call = pool.submit(add_one, x)
            self.assertTrue(call_finished_kernel.wait(5))
            add_one.reset()
            release_call.set()
            torch.testing.assert_close(call.result(timeout=5), x + 1)

        self.assertIsNone(add_one._prepared_call)  # type: ignore[attr-defined]
        self.assertFalse(add_one._dispatch_cache)  # type: ignore[attr-defined]

    def test_backend_capability_is_explicit_and_inherited(self) -> None:
        self.assertTrue(TritonBackend().supports_eager_prepared_call)
        self.assertTrue(TileIRBackend().supports_eager_prepared_call)
        self.assertTrue(CuteBackend().supports_eager_prepared_call)
        self.assertFalse(PallasBackend().supports_eager_prepared_call)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
