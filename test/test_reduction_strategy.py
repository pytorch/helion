from __future__ import annotations

import ast
import operator
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch
from torch.fx.experimental.symbolic_shapes import ShapeEnv

import helion
from helion import exc
from helion._compiler.compile_environment import BlockSizeInfo
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.compile_environment import LoopSpecBlockSizeSource
from helion._compiler.device_function import DeviceFunction
from helion._compiler.device_ir import ReductionLoopGraphInfo
from helion._compiler.flydsl.backend import FlyDSLBackend
from helion._compiler.metal.backend import MetalBackend
from helion._compiler.reduction_strategy import PersistentReductionStrategy
from helion._testing import cpu_render_target
from helion._testing import import_example
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
from helion._testing.triton_codegen import interpret_triton_selection
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
from helion.exc import ReductionOnNonTile
import helion.language as hl

pytestmark = pytest.mark.cpu_only

# Settings for kernels these tests only bind, on CPU under any HELION_BACKEND,
# to see which block sizes become reductions. Pallas refuses
# debug_dtype_asserts (it emits Triton checks), which the dtype-asserts CI job
# turns on through HELION_DEBUG_DTYPE_ASSERTS, so it is pinned off.
_BIND_ONLY: dict[str, Any] = {
    "static_shapes": True,
    "ref_mode": helion.RefMode.OFF,
    "debug_dtype_asserts": False,
}

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion._compiler.backend import Backend


@pytest.mark.parametrize("backend_type", [MetalBackend, FlyDSLBackend])
@pytest.mark.parametrize("tile_size", [4, 8, 16])
@pytest.mark.parametrize("graph_reduction", [False, True])
def test_merged_reduction_preserves_backend_thread_coverage(
    backend_type: type[Backend], tile_size: int, graph_reduction: bool
) -> None:
    # Exercise the real backend hooks without requiring Metal or ROCm devices.
    # Small merged reductions must not acquire CuTe lanes; larger reductions
    # must retain their backend's thread count instead of being capped at 32.
    backend = backend_type()
    shape_env = ShapeEnv()
    first_var = shape_env.create_unbacked_symint()
    second_var = shape_env.create_unbacked_symint()
    first, second = first_var.node.expr, second_var.node.expr
    block_ids = {first: 1, second: 2}
    config = helion.Config(block_sizes=[tile_size, tile_size])
    config_spec = ConfigSpec(
        backend=backend, device=torch.device("cpu"), target_device_capability=None
    )
    # The trace hint differs from every selected size, so only actual config
    # resolution can produce the expected thread coverage below.
    config_spec.block_sizes.extend(
        BlockSizeSpec(block_id=block_id, size_hint=64) for block_id in (1, 2)
    )
    tile_infos = [
        BlockSizeInfo(
            block_id=block_id,
            size=64,
            var=var,
            reduction=False,
            block_size_source=LoopSpecBlockSizeSource(),
        )
        for block_id, var in ((1, first_var), (2, second_var))
    ]
    env = SimpleNamespace(
        backend=backend,
        block_sizes=[SimpleNamespace(numel=first * second), *tile_infos],
        config_spec=config_spec,
        specialize_expr=lambda expr: expr,
        shape_env=shape_env,
        get_block_id=block_ids.get,
        known_equal=operator.eq,
    )
    graph = Mock(spec=ReductionLoopGraphInfo, block_ids=[0])
    fn = Mock(
        spec=DeviceFunction,
        config=config,
        block_size_var_cache={},
        tile_strategy=None,
        codegen=SimpleNamespace(codegen_graphs=[graph] if graph_reduction else []),
        new_var=Mock(side_effect=lambda name, **kwargs: name),
    )
    with (
        patch.object(CompileEnvironment, "current", return_value=env),
        patch(
            "helion._compiler.device_function.find_block_size_symbols",
            return_value=(block_ids, set()),
        ),
    ):
        strategy = PersistentReductionStrategy(fn, 0)

    expected_threads = min(tile_size**2, 64 if backend.name == "flydsl" else 1024)
    assert strategy._reduction_thread_count() == expected_threads
    assert strategy._synthetic_cute_lane_var is None


def test_split_k_barrier_binds_on_pallas() -> None:
    # The full slice over the hl.register_tunable-sized split dim needs a
    # reduction dim even though Pallas does not pad factory tensors.
    kernel = helion.kernel(
        import_example("split_k_barrier").split_k_matmul.fn,
        backend="pallas",
        **_BIND_ONLY,
    )
    bound = kernel.bind((torch.randn(150, 200), torch.randn(200, 6)))
    assert any(info.reduction for info in bound.env.block_sizes)


@pytest.mark.parametrize("backend", ["pallas", "triton"])
def test_reduction_over_runtime_extent_is_rejected(backend: str) -> None:
    @helion.kernel(backend=backend, **_BIND_ONLY)
    def kernel(x: torch.Tensor, extent: int) -> torch.Tensor:
        out = torch.empty_like(x)
        for tile in hl.tile(x.size(0)):
            values = hl.full([tile, extent], 1.0, x.dtype)
            out[tile] = values.sum(-1)
        return out

    with pytest.raises(ReductionOnNonTile):
        kernel.bind((torch.randn(17), 32))


def test_pallas_block_size_slice_keeps_its_extent() -> None:
    # Guards the scope of the tunable-extent fix: only tunable extents take
    # a reduction dim on Pallas; a full slice of a block-size-sized buffer
    # keeps its extent as before.
    @helion.kernel(backend="pallas", **_BIND_ONLY)
    def kernel(x: torch.Tensor) -> torch.Tensor:
        m, n = x.size()
        bm = hl.register_block_size(m)
        scratch = torch.zeros([bm, n], dtype=x.dtype, device=x.device)
        out = torch.empty_like(x)
        for tile in hl.tile(m, block_size=bm):
            out[tile, :] = x[tile, :] + scratch[:, :]
        return out

    bound = kernel.bind((torch.randn(32, 16),))
    assert not any(info.reduction for info in bound.env.block_sizes)


def _add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The combine function of a sum for ``hl.reduce``."""
    return a + b


def _flat_tile_row_ops(
    x: torch.Tensor, index: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    a, b, _c = x.shape
    sums = torch.empty([a, b], device=x.device)
    argmaxes = torch.empty([a, b], dtype=torch.int64, device=x.device)
    reduced = torch.empty([a, b], device=x.device)
    scans = torch.empty_like(x)
    gathered = torch.empty_like(x)
    for tile_a, tile_b in hl.tile([a, b]):
        row = x[tile_a, tile_b, :]
        sums[tile_a, tile_b] = row.sum(-1)
        argmaxes[tile_a, tile_b] = row.argmax(-1)
        reduced[tile_a, tile_b] = hl.reduce(_add, row, dim=2)
        scans[tile_a, tile_b, :] = torch.cumsum(row, dim=2)
        gathered[tile_a, tile_b, :] = torch.gather(row, 2, index[tile_a, tile_b, :])
    return sums, argmaxes, reduced, scans, gathered


def _flat_tile_column_ops(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _c, a, b = x.shape
    reduced = torch.empty([a, b], device=x.device)
    scans = torch.empty_like(x)
    for tile_a, tile_b in hl.tile([a, b]):
        column = x[:, tile_a, tile_b]
        reduced[tile_a, tile_b] = hl.reduce(_add, column, dim=-3)
        scans[:, tile_a, tile_b] = torch.cumsum(column, dim=-3)
    return reduced, scans


def _tile_cumsum(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_a, tile_b in hl.tile(x.size()):
        out[tile_a, tile_b] = torch.cumsum(x[tile_a, tile_b], dim=1)
    return out


def _tile_total(x: torch.Tensor) -> torch.Tensor:
    # Without a dim, hl.reduce reduces along both tile dims.
    out = torch.zeros([1], device=x.device)
    for tile_a, tile_b in hl.tile(x.size()):
        hl.atomic_add(out, [0], hl.reduce(_add, x[tile_a, tile_b]))
    return out


def _tile_softmax(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_a, tile_b in hl.tile(x.size()):
        out[tile_a, tile_b] = torch.softmax(x[tile_a, tile_b], dim=-1)
    return out


def _tile_column_max_offsets(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_a, tile_b in hl.tile(x.size()):
        tile = x[tile_a, tile_b]
        out[tile_a, tile_b] = tile - tile.amax(0, keepdim=True)
    return out


def _tile_sort(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_a, tile_b in hl.tile(x.size()):
        values, _ = torch.sort(x[tile_a, tile_b], dim=-1)
        out[tile_a, tile_b] = values
    return out


def _tile_gather(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_a, tile_b in hl.tile(x.size()):
        tile = x[tile_a, tile_b]
        out[tile_a, tile_b] = torch.gather(tile, 1, (tile > 0).long())
    return out


def _tile_gather_then_row_shift(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    gathered = torch.empty_like(x)
    shifted = torch.empty_like(x)
    for tile_a, tile_b in hl.tile(x.size()):
        tile = x[tile_a, tile_b]
        gathered[tile_a, tile_b] = (
            torch.gather(tile, 1, (tile > 0).long()) + x[0, tile_b]
        )
    for tile_c, tile_d in hl.tile(x.size()):
        shifted[tile_c, tile_d] = x[tile_c, tile_d] + x[0, tile_d]
    return gathered, shifted


def _tile_offsets_from_total(x: torch.Tensor) -> torch.Tensor:
    # Keeping the reduced dims, the total has the tile's rank.
    out = torch.empty_like(x)
    for tile_a, tile_b in hl.tile(x.size()):
        tile = x[tile_a, tile_b]
        out[tile_a, tile_b] = tile - hl.reduce(_add, tile, keep_dims=True)
    return out


def _tile_stable_sort(x: torch.Tensor) -> torch.Tensor:
    # aten.sort.stable, which CuTe lowers itself (pre_inductor_lowering).
    out = torch.empty_like(x)
    for tile_a, tile_b in hl.tile(x.size()):
        values, _ = torch.sort(x[tile_a, tile_b], dim=-1, stable=True)
        out[tile_a, tile_b] = values
    return out


def _tile_argmax_over_one(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty(x.shape[:2], dtype=torch.int64, device=x.device)
    for tile_a, tile_b in hl.tile(x.shape[:2]):
        out[tile_a, tile_b] = torch.argmax(x[tile_a, tile_b, :], dim=-1)
    return out


def _two_root_tile_totals(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    first = torch.empty_like(x)
    second = torch.empty_like(x)
    for tile_a, tile_b in hl.tile(x.size()):
        tile = x[tile_a, tile_b]
        first[tile_a, tile_b] = tile * hl.reduce(_add, tile)
    for tile_c, tile_d in hl.tile(x.size()):
        tile = x[tile_c, tile_d]
        second[tile_c, tile_d] = tile * hl.reduce(_add, tile)
    return first, second


def _dim_operand(call: ast.Call) -> object:
    """The dim a Triton reduction, scan or gather call runs along."""
    assert isinstance(call.func, ast.Attribute)
    if call.func.attr == "gather":
        (axis,) = [keyword.value for keyword in call.keywords if keyword.arg == "axis"]
        return ast.literal_eval(axis)
    position = 2 if call.func.attr == "max_with_index" else 1
    return ast.literal_eval(call.args[position])


@skipUnlessBackends(["triton"])
@helion._testing.skipIfRefEager("requires compiler IR")
def test_flattened_tile_ops_use_physical_dims() -> None:
    # The flattened [tile_a, tile_b] is physical dim 0 of a row, so ops along
    # the row's dim 2 run along physical dim 1; it is the last physical dim
    # of a column, so dim -3 of the column is physical dim -2.
    flat = helion.Config(block_sizes=[2, 4], flatten_loops=[True])
    x = torch.randn(4, 8, 32)
    cases = [
        (
            _flat_tile_row_ops,
            (x, torch.randint(0, 32, x.shape)),
            {
                "sum": 1,
                "max_with_index": 1,
                "reduce": 1,
                "associative_scan": 1,
                "gather": 1,
            },
        ),
        (
            _flat_tile_column_ops,
            (x.permute(2, 0, 1),),
            {"reduce": -2, "associative_scan": -2},
        ),
    ]
    for fn, args, expected in cases:
        with cpu_target():
            bound = cpu_bind(
                helion.kernel(fn, backend="triton", static_shapes=True), args
            )
            code = bound.to_triton_code(flat)
        dims: dict[str, set[object]] = {}
        for node in ast.walk(ast.parse(code)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in expected
            ):
                dims.setdefault(node.func.attr, set()).add(_dim_operand(node))
        assert dims == {name: {dim} for name, dim in expected.items()}


@skipUnlessBackends(["triton", "cute"])
@helion._testing.skipIfRefEager("requires compiler IR")
def test_a_reduction_along_a_grid_tile_is_not_pointwise() -> None:
    # The reduction seeds do not size a partial reduction along a grid tile,
    # but the kernel depends on the tile: no pointwise fact, so CuTe does not
    # re-offer the flatten choice type propagation dropped.
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(_tile_column_max_offsets, static_shapes=True),
            (torch.randn(8, 32),),
        )
    assert not bound.config_spec.pointwise_facts
    assert not bound.config_spec.flatten_loops


@skipUnlessBackends(["triton"])
@helion._testing.skipIfRefEager("requires compiler IR")
def test_an_unpinned_scan_along_a_flattened_tile_is_refused() -> None:
    # Without the pin, the scan along a dim of the flattened tile reaches
    # codegen, which refuses it rather than scanning the whole tile.
    with (
        cpu_target(),
        patch("helion._compiler.compile_environment.BlockSizeInfo.pin_unflattened"),
    ):
        bound = cpu_bind(
            helion.kernel(_tile_cumsum, backend="triton", static_shapes=True),
            (torch.randn(8, 32),),
        )
        with pytest.raises(exc.InvalidConfig, match="dim 1 of a flattened tile"):
            bound.to_triton_code(
                helion.Config(block_sizes=[2, 32], flatten_loops=[True])
            )


@skipUnlessBackends(["triton", "cute"])
@helion._testing.skipIfRefEager("requires compiler IR")
def test_an_argmax_over_a_dim_of_one_pins_nothing() -> None:
    # Over a dim of size 1 the argmax lowers to a constant, whose operand
    # strip_unused_inputs replaces with None: it runs along nothing, so the
    # tile stays flattenable, and the flattened tile renders.
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(_tile_argmax_over_one, static_shapes=True),
            (torch.randn(8, 32, 1),),
        )
        assert [spec.pinned for spec in bound.config_spec.flatten_loops] == [False]
        bound.to_code(helion.Config(block_sizes=[2, 32], flatten_loops=[True]))


@skipUnlessBackends(["triton"])
@helion._testing.skipIfRefEager("requires compiler IR")
def test_a_total_in_a_kernel_of_two_roots_pins_flatten_loops(
    tmp_path, monkeypatch
) -> None:
    # In a kernel of two roots a flattened tile keeps a dim per block over its
    # one index (FlattenedTileStrategy.compacts_tiles), so a total over the
    # value would count each element once per lane of the other dim: both
    # loops stay unflattened, and every 2x32 tile of ones totals 64.
    x = torch.ones(8, 32)
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(_two_root_tile_totals, backend="triton", static_shapes=True),
            (x,),
        )
        assert [spec.pinned for spec in bound.config_spec.flatten_loops] == [
            True,
            True,
        ]
        config = helion.Config(block_sizes=[2, 32, 2, 32], flatten_loops=[True, True])
        bound.config_spec.normalize(config)
        code = bound.to_triton_code(config)
    for total in interpret_triton_selection(
        code, (x,), tmp_path / "two_roots.py", monkeypatch
    ):
        torch.testing.assert_close(total, torch.full_like(x, 64.0))


_TRITON = skipUnlessBackends(["triton"])
_CUTE = skipUnlessBackends(["cute"])


@pytest.mark.parametrize(
    ("backend", "fn", "pinned"),
    # One kernel per kind of op on each backend; a lane runs only the cases
    # of its own backend.
    [
        pytest.param("triton", _tile_softmax, [True], marks=_TRITON),
        pytest.param("triton", _tile_cumsum, [True], marks=_TRITON),
        pytest.param("triton", _tile_sort, [True], marks=_TRITON),
        # tl.reduce(x, None, fn) reduces a flattened tile at once.
        pytest.param("triton", _tile_total, [False], marks=_TRITON),
        pytest.param("triton", _tile_offsets_from_total, [True], marks=_TRITON),
        pytest.param("cute", _tile_softmax, [True], marks=_CUTE),
        pytest.param("cute", _tile_cumsum, [True], marks=_CUTE),
        pytest.param("cute", _tile_total, [True], marks=_CUTE),
        pytest.param("cute", _tile_gather, [True], marks=_CUTE),
        pytest.param("cute", _tile_stable_sort, [True], marks=_CUTE),
        # Type propagation drops both choices (``x[0, tile]`` reads part of a
        # tile); CuTe re-offers them to a pointwise kernel, each in its slot so
        # a saved config's choices still bind to the loops they were tuned
        # for, pinned where the gather runs along a tile dim.
        pytest.param("cute", _tile_gather_then_row_shift, [True, False], marks=_CUTE),
    ],
)
@helion._testing.skipIfRefEager("requires compiler IR")
def test_op_along_a_tile_dim_pins_flatten_loops(
    backend: str, fn: Callable[[torch.Tensor], object], pinned: list[bool]
) -> None:
    # A flattened tile's dims share one index, so a reduction, scan, sort or
    # gather along one of them pins its loop unflattened.
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(fn, backend=backend, static_shapes=True),
            (torch.randn(8, 32),),
        )
        assert [spec.pinned for spec in bound.config_spec.flatten_loops] == pinned
        config = helion.Config(
            block_sizes=[2, 32] * len(pinned), flatten_loops=[True] * len(pinned)
        )
        bound.config_spec.normalize(config)
        assert config.flatten_loops == [not p for p in pinned]


def _consumer_row_first(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        later = y[row, :]
        total = x[row, :].to(torch.float32).sum(-1, keepdim=True)
        out[row, :] = (later.to(torch.float32) * total).to(x.dtype)
    return out


def _consumer_row_before_store(
    x: torch.Tensor, y: torch.Tensor, flags: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        later = y[row, :]
        flags[row] = 1.0
        total = x[row, :].to(torch.float32).sum(-1, keepdim=True)
        out[row, :] = (later.to(torch.float32) * total).to(x.dtype)
    return out


@skipUnlessBackends(["triton", "cute"])
@helion._testing.skipIfRefEager("requires compiler IR")
@pytest.mark.parametrize("split", [False, True])
def test_consumer_only_row_loaded_first_stays_in_the_consumer_sweep(
    split: bool,
) -> None:
    # ``y`` is loaded ahead of the reduction but read only after it, so the
    # reduction sweep loads only ``x`` and the consumer sweep only ``y``, and
    # with ``split`` the store between the two loads adds no sweep of its
    # own.  On CuTe the config turns the loads into cache-hint loads, inline
    # asm the DSL keeps even when dead; the last check confirms it did.
    rows = torch.empty((4, 4096), dtype=torch.bfloat16)
    args = (rows, torch.empty_like(rows))
    if split:
        kernel = helion.kernel(_consumer_row_before_store, static_shapes=True)
        args = (*args, torch.empty(4))
    else:
        kernel = helion.kernel(_consumer_row_first, static_shapes=True)
    cute = kernel.settings.backend == "cute"
    with cpu_render_target():
        bound = cpu_bind(kernel, args)
        spec = bound.config_spec
        extra = {}
        if cute:
            owner = spec.reduction_loops[0].block_id
            extra = {
                "num_threads": [
                    128 if block == owner else 1
                    for block in spec.num_threads.valid_block_ids()
                ],
                "cute_vector_widths": [
                    4 if block == owner else 1
                    for block in spec.cute_vector_widths.valid_block_ids()
                ],
                "load_eviction_policies": ["l2_last"]
                * spec.load_eviction_policies.length,
            }
        code = bound.to_code(
            helion.Config(block_sizes=[1], reduction_loops=[1024], **extra)
        )
    decorator = "cute.kernel" if cute else "triton.jit"
    body = next(
        node.body
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef)
        and any(ast.unparse(item) == decorator for item in node.decorator_list)
    )
    sweeps = [statement for statement in body if isinstance(statement, ast.For)]
    assert [
        {node.id for node in ast.walk(sweep) if isinstance(node, ast.Name)} & {"x", "y"}
        for sweep in sweeps
    ] == [{"x"}, {"y"}]
    if cute:
        assert "_cute_load_l2_evict_last" in ast.unparse(sweeps[0])


def _load_eviction_policies(code: str) -> dict[str, list[str]]:
    """The eviction policy of each Triton ``tl.load``, by the tensor it reads."""
    policies: dict[str, list[str]] = {}
    for node in ast.walk(ast.parse(code)):
        if not (
            isinstance(node, ast.Call)
            and ast.unparse(node.func) == "tl.load"
            and isinstance(node.args[0], ast.BinOp)
            and isinstance(node.args[0].left, ast.Name)
        ):
            continue
        (policy,) = [k.value for k in node.keywords if k.arg == "eviction_policy"]
        assert isinstance(policy, ast.Constant)
        policies.setdefault(node.args[0].left.id, []).append(policy.value)
    return policies


@skipUnlessBackends(["triton"])
@helion._testing.skipIfRefEager("requires compiler IR")
def test_saved_per_load_lists_follow_the_loads_a_config_emits() -> None:
    # Per-load config lists are positional over the loads a config emits.
    # While the reduction sweep still loaded the consumer-only ``y`` (dead),
    # this kernel had 5 eviction and 7 indexing slots, and a rolled config
    # emitted y, x, then y and the store: this saved list gave x "last" and
    # the consumer's y "first".  It still loads, and now gives x "first" and
    # y "last" (indexing entries move the same way).
    rows = torch.empty((4, 4096), dtype=torch.bfloat16)
    kernel = helion.kernel(_consumer_row_first, static_shapes=True)
    with cpu_render_target():
        bound = cpu_bind(kernel, (rows, torch.empty_like(rows)))
        spec = bound.config_spec
        assert (spec.load_eviction_policies.length, spec.indexing.length) == (4, 6)
        code = bound.to_code(
            helion.Config(
                block_sizes=[1],
                reduction_loops=[1024],
                load_eviction_policies=["first", "last", "first", "last", "first"],
                indexing=["pointer"] * 7,
            )
        )
    assert _load_eviction_policies(code) == {
        "x": ["evict_first"],
        "y": ["evict_last"],
    }


def _scaled_cumsum_of_row(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        r = x[row, :]
        scale = r.sum(-1, keepdim=True) * 2
        out[row, :] = hl.cumsum(r, 1) * scale
    return out


def _cumsum_of_scaled_row(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        r = x[row, :]
        total = r.sum(-1, keepdim=True)
        out[row, :] = hl.cumsum(r * total, 1)
    return out


def _row_sums_then_cumsum_of_scaled_row(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sums = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        sums[row] = x[row, :].sum(-1)
    for row in hl.tile(x.size(0)):
        r = x[row, :]
        total = r.sum(-1, keepdim=True)
        out[row, :] = hl.cumsum(r * total, 1)
    return sums, out


@skipUnlessBackends(["triton", "cute"])
@helion._testing.skipIfRefEager("requires compiler IR")
@pytest.mark.parametrize(
    ("fn", "loads", "stores"),
    [
        (_scaled_cumsum_of_row, 1, 1),
        (_cumsum_of_scaled_row, 1, 1),
        (_row_sums_then_cumsum_of_scaled_row, 2, 2),
    ],
)
def test_refused_roll_leaves_no_loop_graphs(
    fn: Callable[[torch.Tensor], object], loads: int, stores: int
) -> None:
    # An hl.cumsum over the rdim refuses its roll: of the row itself, of a
    # value computed from the row after the roller registered the sum's
    # loop, and in a second loop after the first one rolled.  The rdim stays
    # persistent, and none of the refused roll's loop graphs remain to give
    # their loads config slots.
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(fn, static_shapes=True), (torch.empty((4, 4096)),)
        )
    spec = bound.config_spec
    device_ir = bound.host_function.device_ir
    assert not any(
        isinstance(graph, ReductionLoopGraphInfo) for graph in device_ir.graphs
    )
    assert not device_ir.rolled_reductions
    assert len(spec.reduction_loops) == 0
    assert spec.load_eviction_policies.length == loads
    assert spec.indexing.length == loads + stores


@skipUnlessBackends(["triton", "cute"])
@helion._testing.skipIfRefEager("requires compiler IR")
def test_a_config_saved_with_the_dropped_slots_seeds_a_search() -> None:
    # Older versions gave the refused roll's loads slots too: a config they
    # saved lists 3 indexing and 2 load eviction entries, where the kernel
    # has 2 and 1 now.  As a search seed it flattens to the kernel's lengths,
    # its trailing entries cut, and pattern search moves from it.
    from helion.autotuner.config_generation import ConfigGeneration

    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(_cumsum_of_scaled_row, static_shapes=True),
            (torch.empty((4, 4096)),),
        )
        spec = bound.config_spec
        saved = helion.Config(
            **{
                **spec.default_config().config,
                "indexing": ["pointer"] * 3,
                "load_eviction_policies": [""] * 2,
            }
        )
        generation = ConfigGeneration(spec)
        ((flat, config),) = generation.user_seed_flat_config_pairs([saved])
        for index, fragment in enumerate(generation.flat_spec):
            fragment.pattern_neighbors(flat[index])
    # CuTe has no indexing slots.
    assert config.config.get("indexing", ["pointer"] * 2) == ["pointer"] * 2
    assert config.config["load_eviction_policies"] == [""]


def _row_stats_then_cumsum(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    stats = torch.empty([x.size(0), 2], dtype=x.dtype, device=x.device)
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        stats[row, 0] = x[row, :].sum(-1)
        stats[row, 1] = x[row, :].amax(-1)
    for row in hl.tile(x.size(0)):
        out[row, :] = hl.cumsum(x[row, :], 1)
    return stats, out


@skipUnlessBackends(["triton"])
@helion._testing.skipIfRefEager("requires compiler IR")
def test_a_refused_roll_offers_no_reduction_fusion() -> None:
    # The first loop's two reductions could fuse, but the hl.cumsum of the
    # second loop keeps the dim persistent: no config rolls either loop.
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(_row_stats_then_cumsum, static_shapes=True),
            (torch.randn(4, 64),),
        )
    spec = bound.config_spec
    assert len(spec.reduction_loops) == 0
    assert not spec.reduction_fusion_available
