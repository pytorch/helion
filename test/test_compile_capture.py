"""Unit tests for the TPU compile-capture support logic.

Most cover the device-independent pieces (schema/eligibility derivation, cache-key
signature, scalar folding, default filling, and the mutation detector) so the
capture path's correctness does not rest solely on hardware runs; these require no
backend. One Pallas-gated test exercises ``register_decoration_op`` end to end; it
runs under the pallas-interpret backend, so no TPU hardware is needed.
"""

from __future__ import annotations

import ast
import unittest

import pytest
import torch

import helion
from helion._compiler._dynamo.variables import _detect_mutated_inputs
from helion._testing import DEVICE
from helion._testing import skipUnlessPallas
import helion.language as hl
from helion.runtime._tpu_compile_capture import _const_scalar
from helion.runtime._tpu_compile_capture import _decoration_schema
from helion.runtime._tpu_compile_capture import _freeze
from helion.runtime._tpu_compile_capture import _is_functional
from helion.runtime._tpu_compile_capture import _resolves_without_benchmark
from helion.runtime._tpu_compile_capture import _signature
from helion.runtime._tpu_compile_capture import _tensors
from helion.runtime._tpu_compile_capture import register_decoration_op


@helion.kernel(
    backend="pallas", static_shapes=True, config=helion.Config(block_sizes=[16, 16])
)
def _cap_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = torch.broadcast_tensors(x, y)  # rebind: also exercises inplace_writes fix
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


def _body(src: str) -> list[ast.stmt]:
    return ast.parse(src).body


class TestDetectMutatedInputs:
    def test_param_rebind_is_not_mutation(self) -> None:
        # The bug this guards: `x, y = ...` rebinds names but does not mutate the
        # caller's tensors, so it must not be reported as an input mutation.
        body = _body("x, y = torch.broadcast_tensors(x, y)\nout = x + y\n")
        assert _detect_mutated_inputs(body, {"x", "y"}) == []

    def test_subscript_store_is_mutation(self) -> None:
        body = _body("x[tile] = y[tile]\n")
        assert _detect_mutated_inputs(body, {"x", "y"}) == ["x"]

    def test_atomic_is_mutation(self) -> None:
        body = _body("hl.atomic_add(x, [i], v)\n")
        assert _detect_mutated_inputs(body, {"x"}) == ["x"]

    def test_store_to_local_not_flagged(self) -> None:
        # `out` is not a parameter, so storing into it is not an input mutation.
        body = _body("out[tile] = x[tile]\n")
        assert _detect_mutated_inputs(body, {"x"}) == []


class TestSignature:
    def test_bool_int_float_do_not_collide(self) -> None:
        # True == 1 == 1.0 hash-equal; the key must keep them distinct.
        kb = _signature((True,))
        ki = _signature((1,))
        kf = _signature((1.0,))
        assert kb != ki != kf and kb != kf
        assert _signature((1,)) == _signature((1,))

    def test_distinct_scalar_values_distinct_keys(self) -> None:
        assert _signature((1e-5,)) != _signature((1e-6,))

    def test_tensors_are_shape_and_dtype_generic(self) -> None:
        # The op is generic over shape/dtype (the fake infers per call), so two
        # tensors key the SAME op -- one op serves every shape.
        a = torch.empty(4, 8, dtype=torch.float32)
        b = torch.empty(16, 16, dtype=torch.float16)
        assert _signature((a,)) == _signature((b,))
        assert _signature((a,)) != _signature((1.0,))  # tensor vs scalar differ

    def test_list_arg_is_keyable(self) -> None:
        assert _signature(([8192],)) is not None
        assert _signature(([8192],)) == _signature(([8192],))

    def test_list_holding_tensor_is_rejected(self) -> None:
        assert _signature(([torch.empty(2)],)) is None


class TestFreeze:
    def test_nested_container_frozen(self) -> None:
        assert _freeze([1, [2, 3]]) == (1, (2, 3))

    def test_tensor_inside_container_raises(self) -> None:
        with pytest.raises(TypeError):
            _freeze([torch.empty(2)])


class TestConstScalar:
    def test_int_and_float_folded(self) -> None:
        assert isinstance(_const_scalar(1e-5), hl.constexpr)
        assert isinstance(_const_scalar(3), hl.constexpr)
        assert isinstance(_const_scalar(True), hl.constexpr)

    def test_non_scalar_passes_through(self) -> None:
        t = torch.empty(2)
        assert _const_scalar(t) is t
        assert _const_scalar(None) is None
        assert _const_scalar([1, 2]) == [1, 2]


class TestArgs:
    def test_normalize_args_fills_trailing_default(self) -> None:
        # Capture relies on normalize_args filling omitted scalar defaults (eps).
        @helion.kernel()
        def k(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
            return x

        x = torch.empty(4)
        assert k.normalize_args(x) == (x, 1e-5)

    def test_tensors_extracts_in_order(self) -> None:
        a, b = torch.empty(2), torch.empty(3)
        assert _tensors((a, 0.5, b)) == (a, b)


_C16 = helion.Config(block_sizes=[16])
_C32 = helion.Config(block_sizes=[32])


def _pallas_kernel(fn: object, **settings: object) -> object:
    return helion.kernel(fn, backend="pallas", **settings)  # type: ignore[arg-type]


class TestResolvesWithoutBenchmark:
    """The decoration path only registers kernels that resolve a config with NO
    runtime benchmark; an over-broad predicate would autotune at lowering (crash)."""

    def _k(self, **settings: object) -> object:
        def k(x: torch.Tensor) -> torch.Tensor:
            return x

        return _pallas_kernel(k, **settings)

    def test_single_config_is_benchmark_free(self) -> None:
        assert _resolves_without_benchmark(self._k(config=_C16))

    def test_multiple_configs_benchmark(self) -> None:
        assert not _resolves_without_benchmark(self._k(configs=[_C16, _C32]))

    def test_effort_none_at_most_one_config_is_benchmark_free(self) -> None:
        assert _resolves_without_benchmark(self._k(autotune_effort="none"))
        # effort="none" with >=2 configs still benchmarks (FiniteSearch).
        assert not _resolves_without_benchmark(
            self._k(autotune_effort="none", configs=[_C16, _C32])
        )

    def test_force_autotune_overrides(self) -> None:
        # force_autotune benchmarks regardless of a provided config.
        assert not _resolves_without_benchmark(
            self._k(config=_C16, force_autotune=True)
        )


class TestDecorationSchema:
    def test_all_tensor_single_output(self) -> None:
        def k(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x

        assert _decoration_schema(_pallas_kernel(k, config=_C16)) == ([None, None], 1)

    def test_typed_scalar_and_tuple_return(self) -> None:
        def k(x: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
            return x, x

        assert _decoration_schema(_pallas_kernel(k, config=_C16)) == ([None, float], 2)

    def test_unannotated_return_is_ineligible(self) -> None:
        def k(x: torch.Tensor):  # no return annotation
            return x

        assert _decoration_schema(_pallas_kernel(k, config=_C16)) is None


class TestIsFunctional:
    def test_pure_kernel(self) -> None:
        def k(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile]
            return out

        assert _is_functional(_pallas_kernel(k, config=_C16))

    def test_returning_an_input_is_rejected(self) -> None:
        def k(x: torch.Tensor) -> torch.Tensor:
            return x

        assert not _is_functional(_pallas_kernel(k, config=_C16))


@skipUnlessPallas("Pallas/TPU backend not available")
class TestCompileCaptureRoundtrip(unittest.TestCase):
    def test_captured_callable_matches_eager(self) -> None:
        # End-to-end: register_decoration_op registers ONE op (annotated, functional,
        # benchmark-free); the returned callable runs it on the Pallas backend and
        # must match the eager kernel (generic over shape -- 2nd shape reuses the op).
        captured = register_decoration_op(_cap_add)
        self.assertIsNotNone(captured)
        for m, n in [(64, 64), (32, 128)]:
            x = torch.randn(m, n, device=DEVICE)
            y = torch.randn(m, n, device=DEVICE)
            torch.testing.assert_close(captured((x, y)), _cap_add(x, y))
