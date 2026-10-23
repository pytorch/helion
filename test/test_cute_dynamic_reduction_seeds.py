from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from examples.softmax import softmax_two_pass
import pytest
import sympy
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._compiler.autotuner_heuristics.cute import CuteReductionTileHeuristic
from helion._compiler.autotuner_heuristics.cute import CuteRolledRowLadderHeuristic
from helion._compiler.autotuner_heuristics.cute import CuteTileVecHeuristic
from helion._compiler.autotuner_heuristics.cute import CuteTileVecWarpPerRowHeuristic
from helion._compiler.autotuner_heuristics.cute import CuteTileVecWarpReduceHeuristic
from helion._compiler.autotuner_heuristics.cute import _cute_tile_inner_block_dtype
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl
from helion.language._tracing_ops import _host_tensor

if TYPE_CHECKING:
    from collections.abc import Callable


def _softmax(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for rows in hl.tile(x.size(0)):
        out[rows, :] = torch.softmax(x[rows, :].float(), dim=-1).to(x.dtype)
    return out


def _masked_softmax(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for rows in hl.tile(x.size(0)):
        value = x[rows, :]
        mask = value > float("-inf")
        value = torch.where(mask, value, float("-inf"))
        out[rows, :] = torch.softmax(value.float(), dim=-1).to(x.dtype)
    return out


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@skipUnlessBackends(["cute"])
@skipIfRefEager("asserts generated CuTe source, which ref eager does not emit")
@pytest.mark.parametrize("fn", [_softmax, _masked_softmax])
def test_cute_dynamic_reduction_seeds_vector_memory(
    dtype: torch.dtype, fn: Callable[[torch.Tensor], torch.Tensor]
) -> None:
    softmax = helion.kernel(
        fn, backend="cute", static_shapes=False, autotune_effort="none"
    )
    with _mock_cuda_unavailable():
        x = torch.empty(64, 4096, dtype=dtype)
        bound = _cpu_bind(softmax, (x,))
        assert bound.host_function is not None
        spec = bound.config_spec
        seed = CuteRolledRowLadderHeuristic.get_seed_config(
            bound.env, bound.host_function.device_ir
        )
        assert seed is not None
        generation = ConfigGeneration(spec)
        _, seed = generation.canonicalize_flat(generation.flatten(seed))
        assert seed in [config for _, config in generation.seed_flat_config_pairs()]
        reduction_id = spec.reduction_loops[0].block_id
        vector_index = spec.cute_vector_widths.block_id_to_index(reduction_id)
        assert seed.config["cute_vector_widths"][vector_index] == 16 // dtype.itemsize
        assert "cute.arch.load(" in bound.to_code(seed)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layout", ["contiguous", "padded", "offset"])
@skipUnlessBackends(["cute"])
@skipIfNotCUDA()
@skipIfRefEager("checks compiled dynamic seed execution")
def test_dynamic_seed_handles_changed_extent_and_alignment(
    dtype: torch.dtype, layout: str
) -> None:
    kernel = helion.kernel(
        _softmax, backend="cute", static_shapes=False, autotune_effort="none"
    )
    first = torch.randn(5, 4096, device="cuda", dtype=dtype)
    bound = kernel.bind((first,))
    assert bound.host_function is not None
    seed = CuteRolledRowLadderHeuristic.get_seed_config(
        bound.env, bound.host_function.device_ir
    )
    assert seed is not None
    # Use public dispatch so a changed pointer/stride residue gets its own
    # binding rather than bypassing the runtime's alignment specialization.
    seeded = helion.kernel(
        _softmax,
        config=seed,
        backend="cute",
        static_shapes=False,
        autotune_effort="none",
    )
    aligned = seeded.bind((first,))
    for width in (4096, 4100):
        if layout == "padded":
            x = torch.randn(5, width + 3, device="cuda", dtype=dtype)[:, :width]
        elif layout == "offset":
            x = torch.randn(5, width + 1, device="cuda", dtype=dtype)[:, 1:]
        else:
            x = torch.randn(5, width, device="cuda", dtype=dtype)
        actual = seeded(x)
        expected = torch.softmax(x.float(), dim=-1).to(dtype)
        torch.testing.assert_close(actual, expected)
        if layout == "offset":
            assert seeded.bind((x,)) is not aligned
        if layout == "contiguous" and width == 4096:
            assert seeded.bind((x,)) is aligned


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@skipUnlessBackends(["cute"])
@skipIfRefEager("checks dynamic tile heuristic eligibility")
def test_dynamic_tile_heuristics_keep_their_existing_static_gate(
    dtype: torch.dtype,
) -> None:
    kernel = helion.kernel(
        softmax_two_pass.fn, backend="cute", static_shapes=False, autotune_effort="none"
    )
    with _mock_cuda_unavailable():
        bound = _cpu_bind(kernel, (torch.empty(5, 4100, dtype=dtype),))
        assert bound.host_function is not None
        for heuristic in (
            CuteTileVecHeuristic,
            CuteTileVecWarpReduceHeuristic,
            CuteTileVecWarpPerRowHeuristic,
        ):
            assert not heuristic.is_eligible(bound.env, bound.host_function.device_ir)
            assert (
                heuristic.get_seed_config(bound.env, bound.host_function.device_ir)
                is None
            )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@skipUnlessBackends(["cute"])
@skipIfRefEager("checks persistent dynamic seed settings")
def test_dynamic_persistent_seed_keeps_scalar_memory(dtype: torch.dtype) -> None:
    kernel = helion.kernel(
        _softmax, backend="cute", static_shapes=False, autotune_effort="none"
    )
    with _mock_cuda_unavailable():
        bound = _cpu_bind(kernel, (torch.empty(5, 1024, dtype=dtype),))
        assert bound.host_function is not None
        seed = CuteReductionTileHeuristic.get_seed_config(
            bound.env, bound.host_function.device_ir
        )
        assert seed is not None and seed.reduction_loops == [None]
        spec = bound.config_spec
        reduction_id = spec.reduction_loops[0].block_id
        assert (
            spec.cute_vector_widths.config_get(
                seed.get("cute_vector_widths", []), reduction_id, 1
            )
            == 1
        )


@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.int64])
def test_dtype_walk_ignores_non_memory_index_and_mask_nodes(
    mask_dtype: torch.dtype,
) -> None:
    graph = torch.fx.Graph()
    if mask_dtype == torch.bool:
        index = graph.call_function(
            torch.ops.aten.gt.Scalar, (graph.placeholder("source"), 0)
        )
    else:
        index = graph.call_function(torch.ops.aten.arange.default, (4096,))
    index.meta["val"] = torch.empty(4096, dtype=mask_dtype)
    value = graph.call_function(_host_tensor, ())
    value.meta["val"] = torch.empty(4096, dtype=torch.bfloat16)
    env = SimpleNamespace(
        block_sizes=[SimpleNamespace(numel=sympy.Integer(4096))],
        is_jagged_tile=lambda block_id: False,
    )
    ir = SimpleNamespace(graphs=[SimpleNamespace(graph=graph)])
    assert _cute_tile_inner_block_dtype(env, ir, 0) == torch.bfloat16


@pytest.mark.parametrize("dtype", [torch.int8, torch.float8_e4m3fn])
def test_dtype_walk_uses_the_memory_payload_before_a_conversion(
    dtype: torch.dtype,
) -> None:
    graph = torch.fx.Graph()
    value = graph.call_function(_host_tensor, ())
    value.meta["val"] = torch.empty(4096, dtype=dtype)
    converted = graph.call_function(torch.ops.aten._to_copy.default, (value,))
    converted.meta["val"] = torch.empty(4096, dtype=torch.float32)
    env = SimpleNamespace(
        block_sizes=[SimpleNamespace(numel=sympy.Integer(4096))],
        is_jagged_tile=lambda block_id: False,
    )
    ir = SimpleNamespace(graphs=[SimpleNamespace(graph=graph)])
    assert _cute_tile_inner_block_dtype(env, ir, 0) == dtype
