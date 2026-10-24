from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast

from examples.softmax import softmax_two_pass
import pytest
import sympy
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._compiler.autotuner_heuristics.cute import CuteReductionTileHeuristic
from helion._compiler.autotuner_heuristics.cute import CuteReductionWideChunkHeuristic
from helion._compiler.autotuner_heuristics.cute import CuteResidentMultiRowHeuristic
from helion._compiler.autotuner_heuristics.cute import CuteResidentRowHeuristic
from helion._compiler.autotuner_heuristics.cute import CuteRolledClusterLadderHeuristic
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
from helion.language.memory_ops import load as language_load

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.device_ir import DeviceIR


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


def _memory_masked_softmax(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for rows in hl.tile(x.size(0)):
        predicate = mask[rows, :]
        value = x[rows, :]
        value = torch.where(predicate, value, float("-inf"))
        out[rows, :] = torch.softmax(value.float(), dim=-1).to(x.dtype)
    return out


def _square_row_bias_softmax(x: torch.Tensor, row_bias: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for rows in hl.tile(x.size(0)):
        value = x[rows, :]
        bias = row_bias[rows]
        value = value.float() + bias[:, None]
        out[rows, :] = torch.softmax(value, dim=-1).to(x.dtype)
    return out


def _square_row_bias_softmax_bias_first(
    x: torch.Tensor, row_bias: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(x)
    for rows in hl.tile(x.size(0)):
        bias = row_bias[rows]
        value = x[rows, :]
        value = value.float() + bias[:, None]
        out[rows, :] = torch.softmax(value, dim=-1).to(x.dtype)
    return out


def _jagged_sum(x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    num_rows = offsets.numel() - 1
    out = torch.empty(num_rows, dtype=x.dtype, device=x.device)
    for rows in hl.tile(num_rows):
        starts = offsets[rows]
        ends = offsets[rows.index + 1]
        lengths = ends - starts
        acc = hl.zeros([rows], dtype=x.dtype)
        for cols in hl.jagged_tile(lengths):
            mask = cols.index[None, :] < lengths[:, None]
            value = hl.load(x, [starts[:, None] + cols.index[None, :]], extra_mask=mask)
            acc = acc + value.sum(dim=-1)
        out[rows] = acc
    return out


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@skipUnlessBackends(["cute"])
@skipIfRefEager("asserts generated CuTe source, which ref eager does not emit")
@pytest.mark.parametrize("fn", [_softmax, _masked_softmax, _memory_masked_softmax])
def test_cute_dynamic_reduction_seeds_vector_memory(
    dtype: torch.dtype, fn: Callable[..., torch.Tensor]
) -> None:
    softmax = helion.kernel(
        fn, backend="cute", static_shapes=False, autotune_effort="none"
    )
    with _mock_cuda_unavailable():
        x = torch.empty(64, 4096, dtype=dtype)
        args = (
            (x, torch.ones_like(x, dtype=torch.bool))
            if fn is _memory_masked_softmax
            else (x,)
        )
        bound = _cpu_bind(softmax, args)
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
        assert (
            spec.cute_vector_widths.config_get(
                seed.get("cute_vector_widths", []), reduction_id, 1
            )
            == 16 // dtype.itemsize
        )
        assert "cute.arch.load(" in bound.to_code(seed)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layout", ["contiguous", "padded", "offset"])
@pytest.mark.parametrize(
    ("heuristic", "hint"),
    [
        (CuteRolledRowLadderHeuristic, 4096),
        (CuteReductionWideChunkHeuristic, 65536),
        (CuteRolledClusterLadderHeuristic, 65536),
    ],
)
@skipUnlessBackends(["cute"])
@skipIfNotCUDA()
@skipIfRefEager("checks compiled dynamic seed execution")
def test_dynamic_seed_handles_changed_extent_and_alignment(
    dtype: torch.dtype, layout: str, heuristic: type, hint: int
) -> None:
    kernel = helion.kernel(
        _softmax, backend="cute", static_shapes=False, autotune_effort="none"
    )
    first = torch.randn(5, hint, device="cuda", dtype=dtype)
    bound = kernel.bind((first,))
    assert bound.host_function is not None
    seed = heuristic.get_seed_config(bound.env, bound.host_function.device_ir)
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
    for width in (hint, hint + 4):
        if layout == "padded":
            x = torch.randn(5, width + 3, device="cuda", dtype=dtype)[:, :width]
        elif layout == "offset":
            x = torch.randn(5, width + 1, device="cuda", dtype=dtype)[:, 1:]
        else:
            x = torch.randn(5, width, device="cuda", dtype=dtype)
        actual = seeded(x)
        expected = torch.softmax(x.float(), dim=-1).to(dtype)
        torch.testing.assert_close(actual, expected)
        binding = seeded.bind((x,))
        assert seeded.bind((x,)) is binding
        if layout == "offset":
            assert binding is not aligned
        if layout == "contiguous" and width == hint:
            assert binding is aligned


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
            CuteResidentRowHeuristic,
            CuteResidentMultiRowHeuristic,
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

        rolled = CuteRolledRowLadderHeuristic.get_seed_config(
            bound.env, bound.host_function.device_ir
        )
        assert rolled is not None and rolled.reduction_loops == [512]
        assert (
            spec.cute_vector_widths.config_get(
                rolled.get("cute_vector_widths", []), reduction_id, 1
            )
            == 16 // dtype.itemsize
        )


@pytest.mark.parametrize("source", [None, _host_tensor, language_load])
@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.int64])
def test_dtype_walk_ignores_index_and_mask_nodes_when_float_memory_is_available(
    mask_dtype: torch.dtype,
    source: Callable[..., object] | None,
) -> None:
    graph = torch.fx.Graph()
    if source is not None:
        index = graph.call_function(source, ())
    elif mask_dtype == torch.bool:
        index = graph.call_function(
            torch.ops.aten.gt.Scalar, (graph.placeholder("source"), 0)
        )
    else:
        index = graph.call_function(torch.ops.aten.arange.default, (4096,))
    index.meta["val"] = torch.empty(4096, dtype=mask_dtype)
    value = graph.call_function(_host_tensor, ())
    value.meta["val"] = torch.empty(4096, dtype=torch.bfloat16)
    env = SimpleNamespace(
        block_sizes=[
            SimpleNamespace(numel=sympy.Integer(4096), var=sympy.Symbol("tile_width"))
        ],
        is_jagged_tile=lambda block_id: False,
        shape_env=SimpleNamespace(replace=lambda expr: expr),
        specialize_expr=lambda expr: expr,
    )
    ir = SimpleNamespace(graphs=[SimpleNamespace(graph=graph)])
    assert (
        _cute_tile_inner_block_dtype(
            cast("CompileEnvironment", env), cast("DeviceIR", ir), 0
        )
        == torch.bfloat16
    )
    other = graph.call_function(_host_tensor, ())
    other.meta["val"] = torch.empty(4096, dtype=torch.float32)
    assert (
        _cute_tile_inner_block_dtype(
            cast("CompileEnvironment", env), cast("DeviceIR", ir), 0
        )
        == torch.float32
    )
    graph.erase_node(other)
    graph.erase_node(value)
    assert _cute_tile_inner_block_dtype(
        cast("CompileEnvironment", env), cast("DeviceIR", ir), 0
    ) == (mask_dtype if source is not None else None)


@pytest.mark.parametrize("jagged", [False, True])
@pytest.mark.parametrize("dtype", [torch.int8, torch.float8_e4m3fn])
def test_dtype_walk_uses_the_memory_payload_before_a_conversion(
    dtype: torch.dtype,
    jagged: bool,
) -> None:
    graph = torch.fx.Graph()
    value = graph.call_function(_host_tensor, ())
    value.meta["val"] = torch.empty(4096, dtype=dtype)
    converted = graph.call_function(torch.ops.aten._to_copy.default, (value,))
    converted.meta["val"] = torch.empty(4096, dtype=torch.float32)
    env = SimpleNamespace(
        block_sizes=[
            SimpleNamespace(numel=sympy.Integer(4096), var=sympy.Integer(4096))
        ],
        is_jagged_tile=lambda block_id: jagged,
        shape_env=SimpleNamespace(replace=lambda expr: expr),
        specialize_expr=lambda expr: expr,
    )
    ir = SimpleNamespace(graphs=[SimpleNamespace(graph=graph)])
    assert (
        _cute_tile_inner_block_dtype(
            cast("CompileEnvironment", env), cast("DeviceIR", ir), 0
        )
        == dtype
    )

    # Root promotion rewrites the val metadata on existing host tensor nodes.
    value.meta["val"] = torch.empty(4096, dtype=torch.float32)
    assert (
        _cute_tile_inner_block_dtype(
            cast("CompileEnvironment", env), cast("DeviceIR", ir), 0
        )
        == torch.float32
    )


@pytest.mark.parametrize("specialized", [False, True])
def test_dtype_walk_resolves_replaced_and_specialized_extents(
    specialized: bool,
) -> None:
    symbol = sympy.Symbol("width")

    def rewrite(expr: sympy.Expr) -> sympy.Expr:
        return expr.xreplace({symbol: sympy.Integer(4096)})

    env = SimpleNamespace(
        block_sizes=[SimpleNamespace(numel=symbol, var=sympy.Symbol("tile_width"))],
        is_jagged_tile=lambda block_id: False,
        shape_env=SimpleNamespace(
            replace=(lambda expr: expr) if specialized else rewrite
        ),
        specialize_expr=rewrite if specialized else (lambda expr: expr),
    )
    graph = torch.fx.Graph()
    value = graph.call_function(_host_tensor, ())
    value.meta["val"] = torch.empty(4096, dtype=torch.bfloat16)
    ir = SimpleNamespace(graphs=[SimpleNamespace(graph=graph)])
    assert (
        _cute_tile_inner_block_dtype(
            cast("CompileEnvironment", env), cast("DeviceIR", ir), 0
        )
        == torch.bfloat16
    )
    env.shape_env.replace = lambda expr: expr
    env.specialize_expr = lambda expr: expr
    env.block_sizes[0].var = sympy.Integer(4096)
    assert (
        _cute_tile_inner_block_dtype(
            cast("CompileEnvironment", env), cast("DeviceIR", ir), 0
        )
        is None
    )
    assert (
        _cute_tile_inner_block_dtype(
            cast("CompileEnvironment", env),
            cast("DeviceIR", ir),
            0,
            allow_symbolic=True,
        )
        == torch.bfloat16
    )


@pytest.mark.parametrize("first_dtype", [torch.bfloat16, torch.float32])
def test_jagged_dtype_walk_keeps_inner_axis_intermediates(
    first_dtype: torch.dtype,
) -> None:
    graph = torch.fx.Graph()
    value = graph.call_function(_host_tensor, ())
    value.meta["val"] = torch.empty((4096, 8), dtype=torch.bfloat16)
    reduced = graph.call_function(torch.ops.aten.sum.dim_IntList, (value, [-1]))
    reduced.meta["val"] = torch.empty(4096, dtype=first_dtype)
    converted = graph.call_function(torch.ops.aten._to_copy.default, (reduced,))
    converted.meta["val"] = torch.empty(
        4096,
        dtype=torch.float32 if first_dtype is torch.bfloat16 else torch.bfloat16,
    )
    env = SimpleNamespace(
        block_sizes=[SimpleNamespace(var=sympy.Integer(4096))],
        is_jagged_tile=lambda block_id: True,
        shape_env=SimpleNamespace(replace=lambda expr: expr),
        specialize_expr=lambda expr: expr,
    )
    ir = SimpleNamespace(graphs=[SimpleNamespace(graph=graph)])
    assert (
        _cute_tile_inner_block_dtype(
            cast("CompileEnvironment", env), cast("DeviceIR", ir), 0
        )
        == torch.float32
    )


@pytest.mark.parametrize(
    "fn", [_square_row_bias_softmax, _square_row_bias_softmax_bias_first]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.int8])
@skipUnlessBackends(["cute"])
@skipIfRefEager("checks reduction axis provenance")
def test_square_row_bias_does_not_override_reduction_payload(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    dtype: torch.dtype,
) -> None:
    kernel = helion.kernel(
        fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    with _mock_cuda_unavailable():
        bound = _cpu_bind(
            kernel,
            (
                torch.empty(4096, 4096, dtype=dtype),
                torch.empty(4096, dtype=torch.float32),
            ),
        )
        assert bound.host_function is not None
        reduction_id = bound.config_spec.reduction_loops[0].block_id
        assert (
            _cute_tile_inner_block_dtype(
                bound.env, bound.host_function.device_ir, reduction_id
            )
            == dtype
        )


@skipUnlessBackends(["cute"])
@skipIfRefEager("checks vector slots for a jagged axis")
def test_jagged_tile_vector_seed_admission_is_preserved() -> None:
    kernel = helion.kernel(_jagged_sum, backend="cute", autotune_effort="none")
    with _mock_cuda_unavailable():
        bound = _cpu_bind(
            kernel,
            (
                torch.empty(731),
                torch.empty(18, dtype=torch.int64),
            ),
        )
        assert bound.host_function is not None
        spec = bound.config_spec
        assert len(spec.block_sizes) == 2 and not spec.reduction_loops
        inner_id = spec.block_sizes[1].block_id
        assert bound.env.is_jagged_tile(inner_id)
        assert inner_id in spec.cute_vector_widths.valid_block_ids()
        for heuristic in (
            CuteTileVecHeuristic,
            CuteTileVecWarpReduceHeuristic,
            CuteTileVecWarpPerRowHeuristic,
        ):
            assert heuristic.is_eligible(bound.env, bound.host_function.device_ir)
            seed = heuristic.get_seed_config(bound.env, bound.host_function.device_ir)
            assert seed is not None
            assert (
                spec.cute_vector_widths.config_get(
                    seed.get("cute_vector_widths", []), inner_id, 1
                )
                == 4
            )
