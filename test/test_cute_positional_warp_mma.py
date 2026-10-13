from __future__ import annotations

import ast
from copy import deepcopy
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_positional_pointwise_cache import weighted_gram
from .test_cute_positional_pointwise_cache import weighted_gram_tail
import helion
from helion import exc
from helion._compiler.cute.positional_root import PositionalEmitter
from helion._compiler.cute.positional_root import find_axis_collisions
from helion.autotuner.config_spec import CUTE_POSITIONAL_MMA_KEY as MMA
from helion.autotuner.config_spec import CUTE_POSITIONAL_MMA_WARPS_M_KEY as WARPS_M
from helion.autotuner.config_spec import (
    CUTE_POSITIONAL_POINTWISE_CACHE_BYTES_KEY as CACHE,
)
import helion.language as hl

if TYPE_CHECKING:
    from torch.fx import Node

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def loop_exported_state(
    x: torch.Tensor, gate: torch.Tensor, state: torch.Tensor, outputs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, steps, _, _ = x.shape
    steps = hl.specialize(steps)
    for tb in hl.tile(batch, block_size=1):
        current = state[tb, :, :].float()
        for step in range(steps):
            values = x[tb, step, :, :]
            weight = gate[tb, step, :].float().exp().to(x.dtype)
            current = hl.dot(
                values * weight[:, None, :], values.transpose(-1, -2), current
            )
            # Both stores are observable during the lexical iteration. The
            # next iteration changes operands and consumes the current carry.
            outputs[tb, step, :, :] = current
            state[tb, :, :] = current
    return outputs, state


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def broadcast_seed(
    x: torch.Tensor, gate: torch.Tensor, seed: torch.Tensor, scalar: hl.constexpr
) -> torch.Tensor:
    batch, rows, _ = x.shape
    rows = hl.specialize(rows)
    out = torch.empty([batch, rows, rows], dtype=torch.float32, device=x.device)
    for tb in hl.tile(batch, block_size=1):
        values = x[tb, :, :].float()
        weight = gate[tb, :].float().exp()
        if scalar:
            initial = seed[0, 0]
        else:
            initial = seed[tb, :][:, None, :]
        out[tb, :, :] = hl.dot(
            values * weight[:, None, :],
            values.transpose(-1, -2),
            initial.broadcast_to([tb, rows, rows]),  # pyrefly: ignore [no-matching-overload]
        )
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def nonsquare_computed_product(
    lhs: torch.Tensor, rhs: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    m, _ = lhs.shape
    _, n = rhs.shape
    out = torch.empty((m, n), dtype=torch.float32, device=lhs.device)
    for tm, tn in hl.tile([m, n], block_size=[16, 4]):
        a = lhs[tm, :].float() * gate[:].float().exp()[None, :]
        out[tm, tn] = hl.dot(a, rhs[:, tn].float())
    return out


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("num_warps", [1, 4, 8])
def test_mma_uses_typed_cache_and_only_original_output_storage(
    dtype: torch.dtype, num_warps: int
) -> None:
    observed: list[tuple[int, int, int]] = []
    original = PositionalEmitter.emit

    def observe(self: PositionalEmitter) -> list[str]:
        source = original(self)
        observed.append(
            (self.allocated_bytes, self.plan.shared_bytes, self.pointwise_cache_bytes)
        )
        return source

    with _cpu_codegen(), patch.object(PositionalEmitter, "emit", observe):
        kernel = helion.kernel(
            weighted_gram.fn,
            backend="cute",
            static_shapes=True,
            autotune_effort="none",
            dot_precision="tf32",
        )
        bound = kernel._bind_isolated(
            (torch.empty(2, 32, 32, dtype=dtype), torch.empty(2, 32))
        )
        spec = bound.config_spec
        assert spec.cute_positional_mma_search_enabled
        config = spec.default_config()
        config.config.update({MMA: True, CACHE: 256, "num_warps": num_warps})
        source = bound.to_code(config)
    assert "cute.gemm" in source
    assert "ptp_cache_" in source
    assert source.index("ptp_cache_") < source.index("cute.gemm")
    assert (
        f"atom_layout_mnk=({2 if num_warps % 2 == 0 else 1}, {num_warps // (2 if num_warps % 2 == 0 else 1)}, 1)"
        in source
    )
    assert observed == [(4224, 4096, 128)]
    if dtype == torch.float32:
        assert "MmaTF32Op" in source and "cvt_f32_tf32" in source
    else:
        assert "MmaF16BF16Op" in source
        assert "cvt_f32_tf32" not in source


@pytest.mark.parametrize("precision", ["ieee", "tf32x3"])
def test_fp32_precision_does_not_offer_mma(precision: str) -> None:
    with _cpu_codegen():
        kernel = helion.kernel(
            weighted_gram.fn,
            backend="cute",
            static_shapes=True,
            autotune_effort="none",
            dot_precision=precision,
        )
        bound = kernel._bind_isolated((torch.empty(2, 32, 32), torch.empty(2, 32)))
        spec = bound.config_spec
        assert not spec.cute_positional_mma_search_enabled
        config = spec.default_config()
        config.config[CACHE] = 256
        assert "cute.gemm" not in bound.to_code(config)
        config.config[MMA] = True
        with pytest.raises(exc.InvalidConfig, match="positional MMA candidate"):
            bound.to_code(config)


def test_warp_grid_normalizes_defaults_and_checks_divisors() -> None:
    with _cpu_codegen():
        bound = weighted_gram._bind_isolated(
            (torch.empty(2, 32, 32, dtype=torch.bfloat16), torch.empty(2, 32))
        )
        spec = bound.config_spec
        base = spec.default_config()
        assert MMA not in base and WARPS_M not in base
        disabled = deepcopy(base)
        disabled.config.update({MMA: False, WARPS_M: 4})
        assert spec.normalized_config(disabled) == base
        for warps in (1, 2, 4, 8):
            default_m = 2 if warps % 2 == 0 else 1
            assert spec.flatten_missing_field_default(
                WARPS_M, {"num_warps": warps}
            ) == (True, default_m)
            for rows in (1, 2, 4, 8):
                config = deepcopy(base)
                config.config.update({MMA: True, WARPS_M: rows, "num_warps": warps})
                if warps % rows:
                    with pytest.raises(exc.InvalidConfig, match="positive divisor"):
                        spec.normalized_config(config)
                else:
                    normalized = spec.normalized_config(config)
                    assert normalized.config.get(WARPS_M, default_m) == rows
                    assert (WARPS_M in normalized) == (rows != default_m)
        for invalid in (True, 0, -1, 3, "2"):
            config = deepcopy(base)
            config.config.update({MMA: True, WARPS_M: invalid, "num_warps": 4})
            with pytest.raises(exc.InvalidConfig, match="positive divisor"):
                spec.normalized_config(config)


def test_tail_cache_publication_stays_inside_mma_loop_scope() -> None:
    original = PositionalEmitter.emit_contraction
    calls: list[int] = []

    def observe(self: PositionalEmitter, node: Node) -> list[str]:
        saved = self.boundaries.copy()
        source = original(self, node)
        assert self.boundaries.keys() == saved.keys() | {node}
        calls.append(self.pointwise_cache_bytes)
        return source

    with _cpu_codegen(), patch.object(PositionalEmitter, "emit_contraction", observe):
        bound = weighted_gram_tail._bind_isolated(
            (torch.empty(2, 32, 40, dtype=torch.bfloat16), torch.empty(2, 40))
        )
        config = bound.config_spec.default_config()
        config.config.update({MMA: True, CACHE: 256})
        source = bound.to_code(config)
    assert calls == [128]
    loops = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.For)]
    assert any(
        "exp" in ast.unparse(loop)
        and "cute.gemm" in ast.unparse(loop)
        and "ptp_cache_" in ast.unparse(loop)
        for loop in loops
    )
    assert "40" in source
    assert source.count("else cutlass.BFloat16(0)") >= 2


@pytest.mark.parametrize("scalar", [False, True])
def test_broadcast_seed_cpu_codegen(scalar: bool) -> None:
    with _cpu_codegen():
        kernel = helion.kernel(
            broadcast_seed.fn,
            backend="cute",
            static_shapes=True,
            autotune_effort="none",
            dot_precision="tf32",
        )
        bound = kernel._bind_isolated(
            (torch.empty(2, 8, 20), torch.empty(2, 20), torch.empty(2, 8), scalar)
        )
        config = bound.config_spec.default_config()
        config.config.update({MMA: True, CACHE: 256})
        source = bound.to_code(config)
    assert "cute.gemm" in source and "MmaTF32Op" in source


def test_loop_exports_cpu_codegen() -> None:
    with _cpu_codegen():
        bound = loop_exported_state._bind_isolated(
            (
                torch.empty(2, 3, 8, 20, dtype=torch.bfloat16),
                torch.empty(2, 3, 20),
                torch.empty(2, 8, 8),
                torch.empty(2, 3, 8, 8),
            )
        )
        config = bound.config_spec.default_config()
        config.config.update({MMA: True, CACHE: 256})
        source = bound.to_code(config)
    assert "cute.gemm" in source and "ptp_cache_" in source


def test_reduction_candidate_defers_route_proof_until_codegen() -> None:
    with _cpu_codegen():
        kernel = helion.kernel(
            nonsquare_computed_product.fn,
            backend="cute",
            static_shapes=True,
            autotune_effort="none",
            dot_precision="tf32",
            config=helion.Config(
                cute_positional_mma=True, cute_positional_pointwise_cache_bytes=1024
            ),
        )
        bound = kernel.bind(
            (torch.empty(48, 128), torch.empty(128, 12), torch.empty(128))
        )
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            assert not find_axis_collisions(bound.host_function.device_ir.graphs)
        assert bound.config_spec.cute_positional_mma_search_enabled
        assert bound.config_spec.cute_positional_pointwise_cache_search_enabled
        with pytest.raises(exc.BackendUnsupported, match="positional root lowering"):
            bound.to_code(bound.config_spec.normalized_config(kernel.configs[0]))


def test_lane_split_without_axis_collision_registers_fixed_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from .test_cute_positional_root import _delta_state_args
    from .test_cute_positional_root import _delta_state_kernel
    from helion._compiler.cute import chained_loop

    # Exercise fallback discovery even though the loop is now accepted by the
    # earlier chained planner through its useful in-loop exports.
    monkeypatch.setattr(chained_loop, "discover_chained_loop", lambda graphs: None)
    with _cpu_codegen():
        kernel = _delta_state_kernel()
        kernel.configs = [
            helion.Config(
                block_sizes=[32],
                cute_positional_mma=True,
                cute_positional_pointwise_cache_bytes=256,
            )
        ]
        bound = kernel.bind(_delta_state_args(128))
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            assert not find_axis_collisions(bound.host_function.device_ir.graphs)
        assert bound.config_spec.cute_positional_mma_search_enabled
        assert bound.config_spec.cute_positional_pointwise_cache_search_enabled
        source = bound.to_code(bound.config_spec.normalized_config(kernel.configs[0]))
    assert "ptp_thread" in source and source.count("cute.gemm(") == 2
    assert "ptp_cache_" in source


def _run_gpu(
    kernel: helion.Kernel, args: tuple[object, ...], mma: bool, budget: int
) -> object:
    bound = kernel._bind_isolated(args)
    config = bound.config_spec.default_config()
    config.config.update({MMA: mma, CACHE: budget})
    return bound.compile_config(config)(*args)


@requires_cuda
@pytest.mark.parametrize("mma", [False, True])
@pytest.mark.parametrize("budget", [0, 256])
def test_gpu_k_tail_and_singleton_batch(mma: bool, budget: int) -> None:
    torch.manual_seed(123)
    x = (torch.randn(2, 8, 20, device="cuda") * 0.125).bfloat16()
    gate = torch.randn(2, 20, device="cuda") * 0.05
    result = _run_gpu(weighted_gram_tail, (x, gate), mma, budget)
    weighted = (x * gate.exp().bfloat16()[:, None, :]).bfloat16()
    expected = (weighted.double() @ x.double().transpose(-1, -2)).float()
    torch.testing.assert_close(result, expected, rtol=3e-3, atol=3e-4)


@requires_cuda
@pytest.mark.parametrize("mma", [False, True])
@pytest.mark.parametrize("budget", [0, 256])
def test_gpu_loop_carry_changing_inputs_and_exports(mma: bool, budget: int) -> None:
    torch.manual_seed(234)
    x = (torch.randn(2, 3, 8, 20, device="cuda") * 0.125).bfloat16()
    gate = torch.randn(2, 3, 20, device="cuda") * 0.05
    state = torch.randn(2, 8, 8, device="cuda") * 0.1
    expected_state = state.clone()
    expected_outputs = []
    for step in range(3):
        values = x[:, step]
        weighted = (values * gate[:, step].exp().bfloat16()[:, None, :]).bfloat16()
        expected_state = (
            expected_state.double()
            + weighted.double() @ values.double().transpose(-1, -2)
        ).float()
        expected_outputs.append(expected_state.clone())
    outputs = torch.full((2, 3, 8, 8), torch.nan, device="cuda")
    result = _run_gpu(loop_exported_state, (x, gate, state, outputs), mma, budget)
    torch.testing.assert_close(
        result,
        (torch.stack(expected_outputs, dim=1), expected_state),
        rtol=3e-3,
        atol=3e-4,
    )


@requires_cuda
@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize("mma", [False, True])
@pytest.mark.parametrize("budget", [0, 256])
def test_gpu_seed_broadcasting(scalar: bool, mma: bool, budget: int) -> None:
    torch.manual_seed(345)
    x = torch.randn(2, 8, 20, device="cuda") * 0.125
    gate = torch.randn(2, 20, device="cuda") * 0.05
    seed = torch.randn(2, 8, device="cuda") * 0.1
    kernel = helion.kernel(
        broadcast_seed.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        dot_precision="tf32",
    )
    result = _run_gpu(kernel, (x, gate, seed, scalar), mma, budget)
    initial = seed[0, 0] if scalar else seed[:, None, :]
    expected = (
        initial.double()
        + (x * gate.exp()[:, None, :]).double() @ x.double().transpose(-1, -2)
    ).float()
    torch.testing.assert_close(result, expected, rtol=3e-3, atol=3e-4)
