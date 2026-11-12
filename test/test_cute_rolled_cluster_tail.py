"""CPU codegen coverage for masked, disjoint cluster reduction slices and for
the race check that keeps the cluster's CTAs off each other's memory."""

from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import sympy
import torch

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute import lane_cluster_split
from helion._compiler.cute.reduction_emitter import CuteLoopedReductionEmitter
from helion._compiler.host_function import HostFunction
from helion._compiler.reduction_strategy import LoopedReductionStrategy
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.cpu_render(reduction_smem_budget_bytes=232448),
]


def _normalize_rows(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    out = torch.empty_like(x)
    for row in hl.tile(rows):
        values = x[row, :].to(torch.float32)
        scale = torch.rsqrt(torch.mean(values * values, dim=-1) + 1e-5)
        out[row, :] = (values * scale[:, None] * weight[:]).to(x.dtype)
    return out


def _source(
    columns: int,
    chunk: int,
    cluster: int,
    *,
    threads: int = 128,
    vector: int = 8,
    reload: str = "gmem",
    static: bool = True,
) -> tuple[str, helion.Config]:
    kernel = helion.kernel(
        _normalize_rows, backend="cute", static_shapes=static, autotune_effort="none"
    )
    # Padded row strides keep vector input packets aligned for odd tails.
    stride = -(-columns // 8) * 8
    x = torch.empty((4, stride), dtype=torch.bfloat16)[:, :columns]
    weight = torch.empty(columns, dtype=torch.bfloat16)
    bound = cpu_bind(kernel, (x, weight))
    config = bound.config_spec.normalized_config(
        helion.Config(
            block_sizes=[1],
            num_threads=[1, threads],
            reduction_loops=[chunk],
            cute_vector_widths=[vector, 1],
            cute_reduction_reloads=[reload],
            cute_cluster_n=cluster,
        )
    )
    return bound.to_code(config), config


def _rolls(source: str) -> list[ast.For]:
    return [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id.startswith("roffset_")
    ]


def _assert_disjoint_ranges(
    source: str,
    *,
    columns: int,
    chunk: int,
    cluster: int,
) -> None:
    rolls = _rolls(source)
    assert len(rolls) == 2  # producer and consume sweeps
    for roll in rolls:
        assert isinstance(roll.iter, ast.Call)
        covered = []
        for rank in range(cluster):
            scope = {
                "__builtins__": {},
                "cutlass": SimpleNamespace(Int32=int),
                "cute": SimpleNamespace(
                    arch=SimpleNamespace(block_idx=lambda rank=rank: (0, rank, 0))
                ),
                "_REDUCTION_BLOCK_1": chunk,
            }
            bounds = [
                eval(compile(ast.Expression(arg), "<roll>", "eval"), scope)
                for arg in roll.iter.args
            ]
            for offset in range(*bounds):
                covered.extend(range(offset, offset + chunk))
        assert len(covered) == len(set(covered))
        active = sorted(index for index in covered if index < columns)
        assert active == list(range(columns))
        if columns % (cluster * chunk):
            # The memory/reduction mask must use the original logical extent,
            # on both sweeps; padding and the completely inactive CTA stay out.
            assert any(
                isinstance(node, ast.Compare)
                and any(
                    isinstance(value, ast.Constant) and value.value == columns
                    for value in node.comparators
                )
                for node in ast.walk(roll)
            )


@pytest.mark.parametrize(
    "columns,chunk,cluster,threads,vector",
    [
        (7168, 1024, 8, 128, 8),  # one whole chunk short: an entirely inactive CTA
        (6144, 2048, 4, 128, 8),
        (8196, 2048, 8, 128, 8),  # a straddling vector packet plus inactive CTAs
        (255, 128, 4, 64, 1),
    ],
)
def test_static_one_trip_cluster_keeps_tail_masks(
    columns: int,
    chunk: int,
    cluster: int,
    threads: int,
    vector: int,
) -> None:
    source, config = _source(columns, chunk, cluster, threads=threads, vector=vector)
    assert config["reduction_loops"] == [chunk]
    assert "_cute_grouped_reduce_cluster(" in source
    assert f"_helion_cute_cluster_shape = (1, {cluster}, 1)" in source
    assert "mask_1" in source
    _assert_disjoint_ranges(source, columns=columns, chunk=chunk, cluster=cluster)


@pytest.mark.parametrize("columns,chunk,cluster", [(8192, 1024, 8), (16384, 2048, 4)])
def test_exact_cluster_bounds_stay_unmasked(
    columns: int, chunk: int, cluster: int
) -> None:
    source, _ = _source(columns, chunk, cluster)
    assert "_cute_grouped_reduce_cluster(" in source
    assert "mask_1" not in source
    _assert_disjoint_ranges(source, columns=columns, chunk=chunk, cluster=cluster)


def test_partial_cluster_register_prefetch_mode_keeps_cache() -> None:
    source, _ = _source(7168, 1024, 8, reload="register_prefetch")
    assert "_cute_grouped_reduce_cluster(" in source
    assert "_fuse_cache_" in source
    _assert_disjoint_ranges(source, columns=7168, chunk=1024, cluster=8)


@pytest.mark.parametrize("static", [False, True])
def test_partial_multiple_trip_extent_keeps_main_guarded_behavior(static: bool) -> None:
    source, _ = _source(8196, 2048, 2, static=static)
    if static:
        assert "_cute_grouped_reduce_cluster(" in source
        _assert_disjoint_ranges(source, columns=8196, chunk=2048, cluster=2)
    else:
        assert "_cute_grouped_reduce_cluster(" not in source
        assert "_helion_cute_cluster_shape" not in source
    assert "mask_1" in source


@pytest.mark.parametrize(
    "reason",
    [
        "warp",
        "atomic",
        "multirow",
        "indexed",
        "claimed",
        "unbacked",
        "unguarded",
        "empty",
    ],
)
def test_partial_cluster_keeps_legality_guards(reason: str) -> None:
    extent = sympy.Integer(7168)
    if reason in {"unbacked", "unguarded"}:
        extent = sympy.Symbol("u0" if reason == "unbacked" else "s0", integer=True)
    elif reason == "empty":
        extent = sympy.Integer(0)
    env = SimpleNamespace(
        backend=SimpleNamespace(name="cute"),
        shape_env=SimpleNamespace(size_hint=lambda expr: int(expr)),
        block_sizes=[SimpleNamespace(numel=extent)],
        config_spec=SimpleNamespace(
            cute_indexed_reduction_block_ids={0} if reason == "indexed" else set()
        ),
        specialize_expr=lambda expr: expr,
        specialized_multiple=lambda expr, multiple: (
            isinstance(expr, sympy.Integer) and expr % multiple == 0
        ),
    )
    fn = SimpleNamespace(
        config=helion.Config(cute_cluster_n=8),
        cute_state=SimpleNamespace(
            lanes=SimpleNamespace(simt_cluster_n=2 if reason == "claimed" else 1)
        ),
        new_var=lambda name, **kwargs: name,
    )
    strategy: Any = object.__new__(LoopedReductionStrategy)
    strategy._fn = lambda: fn
    strategy.block_ids = [0]
    strategy._thread_count = 32 if reason == "warp" else 128
    strategy._loop_block_size = 1024
    strategy._mask_var = "mask_0"
    strategy._cute_reduction_vec_width = 1
    strategy._cute_rolled_cluster_n = 1
    strategy._cute_rolled_cluster_slice_len = 0
    strategy._cute_rolled_cluster_checked = False
    state = SimpleNamespace(
        codegen=SimpleNamespace(
            current_grid_state=SimpleNamespace(
                thread_axis_sizes={0: 2} if reason == "multirow" else {},
                lane_loops=[],
            )
        )
    )
    host = SimpleNamespace(
        device_ir=SimpleNamespace(has_atomic_ops=lambda: reason == "atomic")
    )
    with (
        patch.object(CompileEnvironment, "current", return_value=env),
        patch.object(HostFunction, "current", return_value=host),
    ):
        CuteLoopedReductionEmitter(strategy)._maybe_apply_cute_rolled_cluster(state)
    assert strategy._cute_rolled_cluster_n == 1
    assert strategy._mask_var == "mask_0"


def _column_running_update(x: torch.Tensor, running: torch.Tensor) -> torch.Tensor:
    # BatchNorm's running-mean update (better-benchmark var_mean_42fad1ece813).
    rows, columns = x.shape
    mean = torch.empty([columns], dtype=torch.float32, device=x.device)
    for column in hl.tile(columns):
        column_mean = torch.sum(x[:, column], dim=0) / rows
        running[column] = column_mean * 0.1 + running[column] * 0.9
        mean[column] = column_mean
    return mean


def _normalize_rows_in_place(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    inv_rms = torch.empty([rows], dtype=torch.float32, device=x.device)
    for row in hl.tile(rows):
        values = x[row, :].to(torch.float32)
        scale = torch.rsqrt(torch.mean(values * values, dim=-1) + 1e-5)
        x[row, :] = (values * scale[:, None] * weight[:]).to(x.dtype)
        inv_rms[row] = scale
    return inv_rms


def _normalize_rows_read_back(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    first = torch.empty([rows], dtype=x.dtype, device=x.device)
    for row in hl.tile(rows):
        values = x[row, :].to(torch.float32)
        scale = torch.rsqrt(torch.mean(values * values, dim=-1) + 1e-5)
        x[row, :] = (values * scale[:, None] * weight[:]).to(x.dtype)
        first[row] = x[row, 0]
    return first


def _normalize_rows_scattered(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    out = torch.empty_like(x)
    for row in hl.tile(rows):
        values = x[row, :].to(torch.float32)
        scale = torch.rsqrt(torch.mean(values * values, dim=-1) + 1e-5)
        out[row, 0] = scale.to(x.dtype)
        out[row, perm[:]] = (values * scale[:, None]).to(x.dtype)
    return out


def _normalize_rows_gathered(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    out = torch.empty_like(x)
    for row in hl.tile(rows):
        values = x[row, :].to(torch.float32)
        scale = torch.rsqrt(torch.mean(values * values, dim=-1) + 1e-5)
        x[row, :] = (values * scale[:, None]).to(x.dtype)
        out[row, :] = x[row, perm[:]]
    return out


def _shift_rows_in_place(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    inv_rms = torch.empty([rows], dtype=torch.float32, device=x.device)
    for row in hl.tile(rows):
        values = x[row, 1024:].to(torch.float32)
        scale = torch.rsqrt(torch.mean(values * values, dim=-1) + 1e-5)
        x[row, :-1024] = (values * scale[:, None]).to(x.dtype)
        inv_rms[row] = scale
    return inv_rms


def _packed_running_update(x: torch.Tensor, running: torch.Tensor) -> torch.Tensor:
    # The running update again, its old value read by a packed load.
    rows, columns = x.shape
    out = torch.empty_like(x)
    for row in hl.tile(rows):
        values = x[row, :].to(torch.float32)
        scale = torch.rsqrt(torch.mean(values * values, dim=-1) + 1e-5)
        out[row, :] = (values * scale[:, None]).to(x.dtype)
        old = hl.load_bfloat16_x16_to_float16(running, row.index)
        running[row, 0] = (old[0].to(torch.float32) * 0.9 + scale * 0.1).to(
            running.dtype
        )
    return out


def _normalize_rows_packed_shift(x: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    out = torch.empty_like(x)
    for row in hl.tile(rows):
        values = x[row, :].to(torch.float32)
        scale = torch.rsqrt(torch.mean(values * values, dim=-1) + 1e-5)
        lanes = hl.load_bfloat16_x16_to_float16(shift, row.index)
        out[row, :] = (
            values * scale[:, None] + lanes[0].to(torch.float32)[:, None]
        ).to(x.dtype)
    return out


def _normalize_rows_prefetched(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    out = torch.empty_like(x)
    for row in hl.tile(rows):
        hl.prefetch(weight, [slice(None)])
        values = x[row, :].to(torch.float32)
        scale = torch.rsqrt(torch.mean(values * values, dim=-1) + 1e-5)
        out[row, :] = (values * scale[:, None] * weight[:]).to(x.dtype)
    return out


def _normalize_into_half_view(x: torch.Tensor) -> torch.Tensor:
    half = x.view(torch.float16)
    (columns,) = x.shape
    inv_rms = torch.empty([1], dtype=torch.float32, device=x.device)
    for tile in hl.tile(1):
        values = x[:].to(torch.float32)
        scale = torch.rsqrt(torch.mean(values * values, dim=-1) + 1e-5)
        half[:columns] = (values * scale).to(torch.float16)
        inv_rms[tile] = scale
    return inv_rms


_COLUMN_CLUSTER = helion.Config(
    block_sizes=[1], num_threads=[1, 64], reduction_loops=[64], cute_cluster_n=4
)
_ROLLED_CLUSTER = helion.Config(
    block_sizes=[1],
    num_threads=[1, 128],
    reduction_loops=[1024],
    cute_vector_widths=[1, 1],
    cute_cluster_n=4,
)
# ``gmem`` reloads re-read the input in the consume sweep, after the peers'
# slices may have stored over it.
_RELOADED_CLUSTER = helion.Config(
    block_sizes=[1],
    num_threads=[1, 128],
    reduction_loops=[1024],
    cute_vector_widths=[1, 1],
    cute_reduction_reloads=["gmem"],
    cute_cluster_n=4,
)
_BF16_ROWS = (
    torch.empty(8, 4096, dtype=torch.bfloat16),
    torch.empty(4096, dtype=torch.bfloat16),
)


@pytest.mark.parametrize(
    "kernel,args,config,clustered",
    [
        # A running statistic updated after the roll: each CTA would load
        # it, maybe after a peer stored its update, and apply the update
        # again.
        pytest.param(
            _column_running_update,
            (torch.empty(64, 384), torch.empty(384)),
            _COLUMN_CLUSTER,
            False,
            id="read-modify-write",
        ),
        # An element read back after the slices stored it, by every CTA
        # rather than only the one whose slice holds it.
        pytest.param(
            _normalize_rows_read_back,
            _BF16_ROWS,
            _ROLLED_CLUSTER,
            False,
            id="read-back",
        ),
        # A gather inside the sweep reads elements of the peers' slices.
        pytest.param(
            _normalize_rows_gathered,
            (
                torch.empty(8, 4096, dtype=torch.bfloat16),
                torch.empty(4096, dtype=torch.int64),
            ),
            _ROLLED_CLUSTER,
            False,
            id="gather",
        ),
        # A scatter fills the CTA's slice after every CTA stored the row's
        # first element: a peer's late copy can land over the scattered one.
        pytest.param(
            _normalize_rows_scattered,
            (
                torch.empty(8, 4096, dtype=torch.bfloat16),
                torch.empty(4096, dtype=torch.int64),
            ),
            _ROLLED_CLUSTER,
            False,
            id="scatter-after-slot",
        ),
        # A shifted copy within each row: a CTA's slice of the store is the
        # next CTA's slice of the reloaded input.
        pytest.param(
            _shift_rows_in_place,
            (torch.empty(1, 5120),),
            _RELOADED_CLUSTER,
            False,
            id="shifted-slices",
        ),
        # A float16 view of float32 data: equal element offsets, other bytes.
        pytest.param(
            _normalize_into_half_view,
            (torch.empty(4096),),
            _RELOADED_CLUSTER,
            False,
            id="dtype-view",
        ),
        # A packed load reads memory like a load: the running update through
        # one races as above.
        pytest.param(
            _packed_running_update,
            (
                torch.empty(8, 4096, dtype=torch.bfloat16),
                torch.empty(8, 16, dtype=torch.bfloat16),
            ),
            _ROLLED_CLUSTER,
            False,
            id="packed-read-modify-write",
        ),
        # A packed load of a tensor nothing writes stays split.
        pytest.param(
            _normalize_rows_packed_shift,
            (
                torch.empty(8, 4096, dtype=torch.bfloat16),
                torch.empty(8, 16, dtype=torch.bfloat16),
            ),
            _ROLLED_CLUSTER,
            True,
            id="packed-read-only",
        ),
        # Slices updated in place and a row scalar nothing reads stay split.
        pytest.param(
            _normalize_rows_in_place,
            _BF16_ROWS,
            _ROLLED_CLUSTER,
            True,
            id="in-place",
        ),
        # A prefetch touches no memory, so it keeps the split.
        pytest.param(
            _normalize_rows_prefetched,
            _BF16_ROWS,
            _ROLLED_CLUSTER,
            True,
            id="prefetch",
        ),
    ],
)
def test_rolled_cluster_declines_cross_cta_races(
    kernel: Any, args: tuple[torch.Tensor, ...], config: helion.Config, clustered: bool
) -> None:
    """Every cluster CTA replays the accesses outside the split loop, so a
    tensor the kernel writes may be accessed there only by stores, and then
    nowhere else; its sliced accesses must agree on the element each CTA's
    slice reaches."""
    _assert_cluster_split(kernel, args, config, clustered)


def _tile_loop_running_update(x: torch.Tensor, running: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    total = torch.empty([rows], dtype=torch.float32, device=x.device)
    block_rows = hl.register_block_size(rows)
    block_columns = hl.register_block_size(columns)
    for row in hl.tile(rows, block_size=block_rows):
        acc = hl.zeros([row], dtype=torch.float32)
        for column in hl.tile(columns, block_size=block_columns):
            acc = acc + x[row, column].sum(dim=1)
        running[row] = acc * 0.1 + running[row] * 0.9
        total[row] = acc
    return total


def _tile_loop_shifted_copy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    block_rows = hl.register_block_size(rows)
    block_columns = hl.register_block_size(columns - 1024)
    for row in hl.tile(rows, block_size=block_rows):
        for column in hl.tile(0, columns - 1024, block_size=block_columns):
            x[row, column] = x[row, column] * 2.0
        for column in hl.tile(1024, columns, block_size=block_columns):
            y[row, column] = x[row, column]
    return y


def _tile_loop_normalize_shifted_in_place(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    total = torch.empty([rows], dtype=torch.float32, device=x.device)
    block_rows = hl.register_block_size(rows)
    block_columns = hl.register_block_size(columns - 1024)
    for row in hl.tile(rows, block_size=block_rows):
        acc = hl.zeros([row], dtype=torch.float32)
        for column in hl.tile(1024, columns, block_size=block_columns):
            acc = acc + x[row, column].sum(dim=1)
        for column in hl.tile(1024, columns, block_size=block_columns):
            x[row, column] = x[row, column] / acc[:, None]
        total[row] = acc
    return total


def _tile_loop_shifted_per_step(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    total = torch.empty([rows], dtype=torch.float32, device=x.device)
    block_rows = hl.register_block_size(rows)
    block_columns = hl.register_block_size(columns - 1024)
    for row in hl.tile(rows, block_size=block_rows):
        for step in hl.grid(2):
            begin = step * 1024
            for column in hl.tile(
                begin, begin + columns - 1024, block_size=block_columns
            ):
                x[row, column] = x[row, column] * 2.0
        total[row] = 0.0
    return total


def _tile_loop_doubled_in_while(x: torch.Tensor, limits: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    total = torch.empty([rows], dtype=torch.float32, device=x.device)
    block_rows = hl.register_block_size(rows)
    block_columns = hl.register_block_size(columns)
    for row in hl.tile(rows, block_size=block_rows):
        steps = hl.zeros([row], dtype=torch.int32)
        limit = limits[row]
        while torch.any(steps < limit):
            for column in hl.tile(columns, block_size=block_columns):
                x[row, column] = x[row, column] * 2.0
            steps = steps + 1
        total[row] = 0.0
    return total


def _tile_loop_update_below(
    x: torch.Tensor, running: torch.Tensor, width: hl.constexpr
) -> torch.Tensor:
    rows, columns = x.shape
    total = torch.empty([rows], dtype=torch.float32, device=x.device)
    block_rows = hl.register_block_size(rows)
    block_columns = hl.register_block_size(columns)
    update = hl.constexpr(block_columns < width)
    for row in hl.tile(rows, block_size=block_rows):
        acc = hl.zeros([row], dtype=torch.float32)
        for column in hl.tile(columns, block_size=block_columns):
            acc = acc + x[row, column].sum(dim=1)
        if update:
            running[row] = acc * 0.1 + running[row] * 0.9
        total[row] = acc
    return total


def _tile_loop_fill_after_slot(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    out = torch.empty_like(x)
    block_rows = hl.register_block_size(rows)
    block_columns = hl.register_block_size(columns)
    for row in hl.tile(rows, block_size=block_rows):
        acc = hl.zeros([row], dtype=torch.float32)
        for column in hl.tile(columns, block_size=block_columns):
            acc = acc + x[row, column].sum(dim=1)
        out[row, 0] = acc * 0.0 - 1.0
        for column in hl.tile(columns, block_size=block_columns):
            out[row, column] = acc[:, None]
    return out


def _tile_loop_masked_flag(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    out = torch.empty([rows], dtype=x.dtype, device=x.device)
    block_rows = hl.register_block_size(rows)
    block_columns = hl.register_block_size(columns)
    for row in hl.tile(rows, block_size=block_rows):
        out[row] = x[row, 0] * 0.0
        for column in hl.tile(columns, block_size=block_columns):
            hl.store(out, [row], 1.0, extra_mask=column.index == 0)
    return out


def _tile_loop_valued_flags(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    out = torch.empty([rows], dtype=x.dtype, device=x.device)
    block_rows = hl.register_block_size(rows)
    block_columns = hl.register_block_size(columns)
    for row in hl.tile(rows, block_size=block_rows):
        for column in hl.tile(columns, block_size=block_columns):
            flag = column.index * 0
            hl.store(out, [row], flag + 1.0, extra_mask=column.index == 0)
            hl.store(out, [row], flag + 2.0, extra_mask=column.index == 4096)
    return out


def _tile_loop_masked_flag_per_step(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    out = torch.empty([rows], dtype=x.dtype, device=x.device)
    block_rows = hl.register_block_size(rows)
    block_columns = hl.register_block_size(columns)
    for row in hl.tile(rows, block_size=block_rows):
        for column in hl.tile(columns, block_size=block_columns):
            for step in hl.grid(2):
                hl.store(out, [row], step, extra_mask=column.index == step * 4096)
    return out


def _tile_loop_valued_flag_per_step(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    out = torch.empty([rows], dtype=x.dtype, device=x.device)
    block_rows = hl.register_block_size(rows)
    block_columns = hl.register_block_size(columns)
    for row in hl.tile(rows, block_size=block_rows):
        for column in hl.tile(columns, block_size=block_columns):
            for step in hl.grid(2):
                out[row] = column.index * 0 + step
    return out


def _tile_loop_masked_valued_flag_per_step(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    out = torch.empty([rows], dtype=x.dtype, device=x.device)
    block_rows = hl.register_block_size(rows)
    block_columns = hl.register_block_size(columns)
    for row in hl.tile(rows, block_size=block_rows):
        for column in hl.tile(columns, block_size=block_columns):
            for step in hl.grid(2):
                hl.store(
                    out,
                    [row],
                    column.index * 0 + step,
                    extra_mask=column.index == step * 4096,
                )
    return out


def _tile_loop_first_column_flag_per_step(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    out = torch.empty([rows], dtype=x.dtype, device=x.device)
    block_rows = hl.register_block_size(rows)
    block_columns = hl.register_block_size(columns)
    for row in hl.tile(rows, block_size=block_rows):
        for column in hl.tile(columns, block_size=block_columns):
            for step in hl.grid(2):
                if column.index == step * 4096:
                    out[row] = x[row, 0] + step
    return out


def _tile_loop_scaled_per_step(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    out = torch.empty_like(x)
    block_rows = hl.register_block_size(rows)
    block_columns = hl.register_block_size(columns)
    for row in hl.tile(rows, block_size=block_rows):
        for column in hl.tile(columns, block_size=block_columns):
            for step in hl.grid(2):
                out[row, column] = x[row, column] * step
    return out


def _tile_loop_sum_row_masked_flag(x: torch.Tensor, flag: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    total = torch.empty([rows], dtype=torch.float32, device=x.device)
    block_rows = hl.register_block_size(rows)
    block_columns = hl.register_block_size(columns)
    for row in hl.tile(rows, block_size=block_rows):
        acc = hl.zeros([row], dtype=torch.float32)
        flag[row] = x[row, 0] * 0.0
        for column in hl.tile(columns, block_size=block_columns):
            acc = acc + x[row, column].sum(dim=1)
            hl.store(flag, [row], x[row, 0], extra_mask=x[row, 0] > 0)
        total[row] = acc
    return total


def _tile_loop_first_column_flag(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    out = torch.empty([rows], dtype=x.dtype, device=x.device)
    block_rows = hl.register_block_size(rows)
    block_columns = hl.register_block_size(columns)
    for row in hl.tile(rows, block_size=block_rows):
        out[row] = x[row, 0] * 0.0
        for column in hl.tile(columns, block_size=block_columns):
            if column.index == 0:
                out[row] = x[row, 0] + 1.0
    return out


def _tile_loop_first_column_flag_while(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    out = torch.empty([rows], dtype=x.dtype, device=x.device)
    block_rows = hl.register_block_size(rows)
    block_columns = hl.register_block_size(columns)
    for row in hl.tile(rows, block_size=block_rows):
        out[row] = x[row, 0] * 0.0
        for column in hl.tile(columns, block_size=block_columns):
            i = column.index * 0
            while i + column.index < 1:
                out[row] = x[row, 0] + 1.0
                i = i + 1
    return out


def _tile_loop_nested_first_column_flag(x: torch.Tensor, flag: int) -> torch.Tensor:
    rows, columns = x.shape
    out = torch.empty([rows], dtype=x.dtype, device=x.device)
    block_rows = hl.register_block_size(rows)
    block_columns = hl.register_block_size(columns)
    for row in hl.tile(rows, block_size=block_rows):
        out[row] = x[row, 0] * 0.0
        for column in hl.tile(columns, block_size=block_columns):
            if column.index == 0:
                if flag > 0:
                    out[row] = x[row, 0] + 1.0
    return out


def _tile_loop_first_column_flag_slice(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    out = torch.empty([rows, 1], dtype=x.dtype, device=x.device)
    block_rows = hl.register_block_size(rows)
    block_columns = hl.register_block_size(columns)
    for row in hl.tile(rows, block_size=block_rows):
        for column in hl.tile(columns, block_size=block_columns):
            out[row, :] = 0.0
            if column.index == 0:
                out[row, :] = 1.0
    return out


def _tile_loop_masked_flag_slice(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    out = torch.empty([rows, 1], dtype=x.dtype, device=x.device)
    block_rows = hl.register_block_size(rows)
    block_columns = hl.register_block_size(columns)
    for row in hl.tile(rows, block_size=block_rows):
        for column in hl.tile(columns, block_size=block_columns):
            out[row, :] = 0.0
            hl.store(out, [row, slice(None)], 1.0, extra_mask=column.index == 0)
    return out


def _tile_loop_valued_flag_slice(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    out = torch.empty([rows, 1], dtype=x.dtype, device=x.device)
    block_rows = hl.register_block_size(rows)
    block_columns = hl.register_block_size(columns)
    for row in hl.tile(rows, block_size=block_rows):
        for column in hl.tile(columns, block_size=block_columns):
            out[row, :] = 0.0
            hl.store(
                out,
                [row, slice(None)],
                column.index * 0 + 1.0,
                extra_mask=column.index == 0,
            )
    return out


def _tile_loop_sum_first_tile_flag(x: torch.Tensor, flag: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    total = torch.empty([rows], dtype=torch.float32, device=x.device)
    block_rows = hl.register_block_size(rows)
    block_columns = hl.register_block_size(columns)
    for row in hl.tile(rows, block_size=block_rows):
        acc = hl.zeros([row], dtype=torch.float32)
        flag[row] = x[row, 0] * 0.0
        for column in hl.tile(columns, block_size=block_columns):
            acc = acc + x[row, column].sum(dim=1)
            if column.begin == 0:
                flag[row] = x[row, 0] + 1.0
        total[row] = acc
    return total


def _tile_loop_sum_first_tile_flag_while(
    x: torch.Tensor, flag: torch.Tensor
) -> torch.Tensor:
    rows, columns = x.shape
    total = torch.empty([rows], dtype=torch.float32, device=x.device)
    block_rows = hl.register_block_size(rows)
    block_columns = hl.register_block_size(columns)
    for row in hl.tile(rows, block_size=block_rows):
        acc = hl.zeros([row], dtype=torch.float32)
        flag[row] = x[row, 0] * 0.0
        for column in hl.tile(columns, block_size=block_columns):
            acc = acc + x[row, column].sum(dim=1)
            i = hl.zeros([], dtype=torch.int32)
            while i < 1:
                flag[row] = x[row, 0] + 1.0
                i = i + 1
        total[row] = acc
    return total


def _tile_loop_scaled_overlapping_steps(x: torch.Tensor) -> torch.Tensor:
    steps, rows, columns = x.shape
    block_rows = hl.register_block_size(rows)
    block_columns = hl.register_block_size(columns)
    for row in hl.tile(rows, block_size=block_rows):
        for column in hl.tile(columns, block_size=block_columns):
            for step in hl.grid(steps):
                x[step, row, column] = x[step, row, column] * 2.0
    return x


def _tile_loop_fill_under_if(x: torch.Tensor, flag: int) -> torch.Tensor:
    rows, columns = x.shape
    out = torch.empty_like(x)
    block_rows = hl.register_block_size(rows)
    block_columns = hl.register_block_size(columns)
    for row in hl.tile(rows, block_size=block_rows):
        out[row, 0] = x[row, 0] * 0.0 - 1.0
        for column in hl.tile(columns, block_size=block_columns):
            if flag > 0:
                out[row, column] = x[row, column]
    return out


def _tile_loop_sum_into_buffers(
    x: torch.Tensor, template: torch.Tensor, pointers: torch.Tensor
) -> torch.Tensor:
    rows, columns = x.shape
    total = torch.empty([rows], dtype=torch.float32, device=x.device)
    block_rows = hl.register_block_size(rows)
    block_columns = hl.register_block_size(columns)
    for row in hl.tile(rows, block_size=block_rows):
        acc = hl.zeros([row], dtype=torch.float32)
        for column in hl.tile(columns, block_size=block_columns):
            acc = acc + x[row, column].sum(dim=1)
        buffers = hl.stacktensor_like(template, pointers[:])
        buffers[row] = 0.0
        total[row] = acc
    return total


def _tile_loop_normalize_in_place(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    total = torch.empty([rows], dtype=torch.float32, device=x.device)
    block_rows = hl.register_block_size(rows)
    block_columns = hl.register_block_size(columns)
    for row in hl.tile(rows, block_size=block_rows):
        acc = hl.zeros([row], dtype=torch.float32)
        for column in hl.tile(columns, block_size=block_columns):
            acc = acc + x[row, column].sum(dim=1)
        for column in hl.tile(columns, block_size=block_columns):
            x[row, column] = x[row, column] / acc[:, None]
        total[row] = acc
    return total


_TILE_LOOP_CLUSTER = helion.Config(
    block_sizes=[1, 8192],
    num_threads=[0, 256],
    cute_vector_widths=[1, 8],
    cute_lane_layouts=["blocked", "strided"],
    cute_cluster_n=2,
)


@pytest.mark.parametrize(
    "kernel,args,clustered",
    [
        pytest.param(
            _tile_loop_running_update,
            (torch.empty(8, 8192), torch.empty(8)),
            False,
            id="read-modify-write",
        ),
        # Two loops over one block size from different begins: a CTA's slice
        # of the second is partly the peer's slice of the first.
        pytest.param(
            _tile_loop_shifted_copy,
            (torch.empty(8, 8192), torch.empty(8, 8192)),
            False,
            id="shifted-loops",
        ),
        # A broadcast fill is sliced by what its subscript spans; every CTA
        # also replays the earlier store of the row's first element.
        pytest.param(
            _tile_loop_fill_after_slot,
            (torch.empty(8, 8192),),
            False,
            id="fill-after-slot",
        ),
        # The same fill in a branch inside the loop is still sliced.
        pytest.param(
            _tile_loop_fill_under_if,
            (torch.empty(8, 8192), 1),
            False,
            id="fill-under-if",
        ),
        # A row store masked along the split axis lands in the CTA whose
        # slice holds column 0 only, after or before the peers' replayed
        # store of the row.
        pytest.param(
            _tile_loop_masked_flag,
            (torch.empty(8, 8192),),
            False,
            id="split-masked-store",
        ),
        # So do row stores of values that span the split axis: their
        # subscript, not their value, says which elements they reach.
        pytest.param(
            _tile_loop_valued_flags,
            (torch.empty(8, 8192),),
            False,
            id="split-masked-valued-stores",
        ),
        # A mask of the row alone is the same in every CTA: identical copies.
        pytest.param(
            _tile_loop_sum_row_masked_flag,
            (torch.empty(8, 8192), torch.empty(8)),
            True,
            id="row-masked-store",
        ),
        # A branch around the row store decides alike: on the split axis
        # (``column.index == 0``) the store runs in one CTA only, on the
        # tile's begin in every CTA.
        pytest.param(
            _tile_loop_first_column_flag,
            (torch.empty(8, 8192),),
            False,
            id="split-branch-store",
        ),
        # A device while whose test varies along the split axis is such a
        # branch: only CTA 0 runs the store.
        pytest.param(
            _tile_loop_first_column_flag_while,
            (torch.empty(8, 8192),),
            False,
            id="split-while-store",
        ),
        pytest.param(
            _tile_loop_nested_first_column_flag,
            (torch.empty(8, 8192), 1),
            False,
            id="split-branch-nested-store",
        ),
        # The same row store inside the loop, through a slice of another
        # axis (a dim of size 1), under the branch or the mask alike.
        pytest.param(
            _tile_loop_first_column_flag_slice,
            (torch.empty(8, 8192),),
            False,
            id="split-branch-sliced-store",
        ),
        pytest.param(
            _tile_loop_masked_flag_slice,
            (torch.empty(8, 8192),),
            False,
            id="split-masked-sliced-store",
        ),
        pytest.param(
            _tile_loop_valued_flag_slice,
            (torch.empty(8, 8192),),
            False,
            id="split-masked-valued-sliced-store",
        ),
        pytest.param(
            _tile_loop_sum_first_tile_flag,
            (torch.empty(8, 8192), torch.empty(8)),
            True,
            id="uniform-branch-store",
        ),
        pytest.param(
            _tile_loop_sum_first_tile_flag_while,
            (torch.empty(8, 8192), torch.empty(8)),
            True,
            id="uniform-while-store",
        ),
        # Strides that let two indices name one element: x[1, row, j] is
        # x[0, row, 4096 + j], which the steps update in different CTAs.
        pytest.param(
            _tile_loop_scaled_overlapping_steps,
            (torch.empty(8 * 12288).as_strided((2, 8, 8192), (4096, 12288, 1)),),
            False,
            id="overlapping-layout",
        ),
        # A device loop inside the split loop runs such a store once per
        # step, each time in the CTA whose slice the mask or the test selects
        # then: CTA 0 stores step 0's value, CTA 1 step 1's, unordered.
        pytest.param(
            _tile_loop_masked_flag_per_step,
            (torch.empty(8, 8192),),
            False,
            id="split-masked-store-per-step",
        ),
        pytest.param(
            _tile_loop_first_column_flag_per_step,
            (torch.empty(8, 8192),),
            False,
            id="split-branch-store-per-step",
        ),
        # Unmasked, a row store of a value that spans the split axis runs
        # in every CTA once per step: CTA 0's step 1 can land before CTA 1's
        # step 0.
        pytest.param(
            _tile_loop_valued_flag_per_step,
            (torch.empty(8, 8192),),
            False,
            id="split-valued-store-per-step",
        ),
        # The masked per-step store with such a value: CTA 0 stores step 0,
        # CTA 1 step 1, unordered.
        pytest.param(
            _tile_loop_masked_valued_flag_per_step,
            (torch.empty(8, 8192),),
            False,
            id="split-masked-valued-store-per-step",
        ),
        # The split loop inside a device loop whose index moves its begin: a
        # CTA's slice in one step is a peer's slice in the other.
        pytest.param(
            _tile_loop_shifted_per_step,
            (torch.empty(8, 9216),),
            False,
            id="loop-in-device-loop",
        ),
        # A device ``while`` is such a loop too.
        pytest.param(
            _tile_loop_doubled_in_while,
            (torch.empty(8, 8192), torch.empty(8, dtype=torch.int32)),
            False,
            id="loop-in-while",
        ),
        # A read-modify-write on the taken side of a branch on block sizes
        # races; on the side the config never emits, it keeps the split.
        pytest.param(
            _tile_loop_update_below,
            (torch.empty(8, 8192), torch.empty(8), 16384),
            False,
            id="read-modify-write-taken",
        ),
        pytest.param(
            _tile_loop_update_below,
            (torch.empty(8, 8192), torch.empty(8), 8192),
            True,
            id="read-modify-write-untaken",
        ),
        # A stack tensor's buffers may be any tensor's memory.
        pytest.param(
            _tile_loop_sum_into_buffers,
            (torch.empty(8, 8192), torch.empty(8), torch.empty(2, dtype=torch.uint64)),
            False,
            id="stack-tensor-store",
        ),
        # Loops from one fixed shifted begin stay split.
        pytest.param(
            _tile_loop_normalize_shifted_in_place,
            (torch.empty(8, 9216),),
            True,
            id="shifted-begin",
        ),
        pytest.param(
            _tile_loop_normalize_in_place,
            (torch.empty(8, 8192),),
            True,
            id="in-place",
        ),
    ],
)
def test_tile_loop_cluster_declines_cross_cta_races(
    kernel: Any, args: tuple[torch.Tensor, ...], clustered: bool
) -> None:
    """The cluster split of a lane-looped tile axis shares the check."""
    _assert_cluster_split(kernel, args, _TILE_LOOP_CLUSTER, clustered)


def test_a_sliced_store_in_a_nested_loop_keeps_the_tile_loop_split() -> None:
    """A device loop inside the split loop runs a sliced store once per step,
    each time into the CTA's own slice: the split stays (no reduction, so no
    cluster reduce either)."""
    bound = cpu_bind(
        helion.kernel(
            _tile_loop_scaled_per_step,
            backend="cute",
            static_shapes=True,
            autotune_effort="none",
        ),
        (torch.empty(8, 8192),),
    )
    source = bound.to_code(bound.config_spec.normalized_config(_TILE_LOOP_CLUSTER))
    assert "_helion_cute_cluster_shape" in source


def _tile_loop_window_add(
    x: torch.Tensor, start: torch.Tensor, out: torch.Tensor, atomic: hl.constexpr
) -> torch.Tensor:
    rows, columns = x.shape
    block_rows = hl.register_block_size(rows)
    for row in hl.tile(rows, block_size=block_rows):
        begin = start[0]
        for column in hl.tile(begin, begin + 4096):
            if atomic:
                hl.atomic_add(out, [row, column], x[row, column])
            else:
                out[row, column] = x[row, column]
    return out


@pytest.mark.parametrize("atomic", [False, True])
def test_tile_loop_cluster_skips_loaded_extents(atomic: bool) -> None:
    """A tile loop whose begin is read from a tensor has no extent to slice."""
    rows = torch.empty(8, 8192)
    start = torch.empty(1, dtype=torch.int64)
    _assert_cluster_split(
        _tile_loop_window_add,
        (rows, start, torch.empty(8, 8192), atomic),
        _TILE_LOOP_CLUSTER,
        False,
        checked=False,
    )


def _column_running_update_into(
    x: torch.Tensor, old_running: torch.Tensor, running: torch.Tensor
) -> torch.Tensor:
    rows, columns = x.shape
    mean = torch.empty([columns], dtype=torch.float32, device=x.device)
    for column in hl.tile(columns):
        column_mean = torch.sum(x[:, column], dim=0) / rows
        running[column] = column_mean * 0.1 + old_running[column] * 0.9
        mean[column] = column_mean
    return mean


def _normalize_rows_into(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    rows, columns = src.shape
    inv_rms = torch.empty([rows], dtype=torch.float32, device=src.device)
    for row in hl.tile(rows):
        values = src[row, :].to(torch.float32)
        scale = torch.rsqrt(torch.mean(values * values, dim=-1) + 1e-5)
        dst[row, :] = (values * scale[:, None]).to(dst.dtype)
        inv_rms[row] = scale
    return inv_rms


def _tile_loop_fill(
    x: torch.Tensor, left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    rows, columns = x.shape
    total = torch.empty([rows], dtype=torch.float32, device=x.device)
    block_rows = hl.register_block_size(rows)
    block_columns = hl.register_block_size(columns)
    for row in hl.tile(rows, block_size=block_rows):
        acc = hl.zeros([row], dtype=torch.float32)
        for column in hl.tile(columns, block_size=block_columns):
            acc = acc + x[row, column].sum(dim=1)
        for column in hl.tile(columns, block_size=block_columns):
            left[row, column] = acc[:, None]
            right[row, column] = 0.0
        total[row] = acc
    return total


_RUNNING = torch.empty(384)
_ROW = torch.empty(1, 5120)
_ROWS = torch.empty(8, 8192)


@pytest.mark.parametrize(
    "kernel,args,config,clustered",
    [
        # One running statistic passed as the read and the written argument.
        pytest.param(
            _column_running_update_into,
            (torch.empty(64, 384), _RUNNING, _RUNNING),
            _COLUMN_CLUSTER,
            False,
            id="aliased-read-modify-write",
        ),
        pytest.param(
            _column_running_update_into,
            (torch.empty(64, 384), _RUNNING, torch.empty(384)),
            _COLUMN_CLUSTER,
            True,
            id="distinct-read-modify-write",
        ),
        # The shifted copy through overlapping argument views.
        pytest.param(
            _normalize_rows_into,
            (_ROW[:, 1024:], _ROW[:, :-1024]),
            _RELOADED_CLUSTER,
            False,
            id="shifted-views",
        ),
        pytest.param(
            _normalize_rows_into,
            (torch.empty(1, 4096), torch.empty(1, 4096)),
            _RELOADED_CLUSTER,
            True,
            id="distinct-views",
        ),
        # Broadcast stores fill the slices their subscripts span, so two
        # overlapping views of one output race across the CTAs.
        pytest.param(
            _tile_loop_fill,
            (torch.empty(8, 7168), _ROWS[:, :7168], _ROWS[:, 1024:]),
            _TILE_LOOP_CLUSTER,
            False,
            id="broadcast-overlap",
        ),
        pytest.param(
            _tile_loop_fill,
            (torch.empty(8, 7168), torch.empty(8, 7168), torch.empty(8, 7168)),
            _TILE_LOOP_CLUSTER,
            True,
            id="broadcast-distinct",
        ),
    ],
)
def test_cluster_split_declines_overlapping_arguments(
    kernel: Any, args: tuple[torch.Tensor, ...], config: helion.Config, clustered: bool
) -> None:
    """With static shapes every argument has its own fake storage, even when
    the caller passes overlapping tensors: the bound alias facts decide
    which may share memory, and accesses through different views of it need
    not slice alike."""
    _assert_cluster_split(kernel, args, config, clustered)


def test_cluster_split_declines_other_host_tensor_calls() -> None:
    """A call that takes a host tensor may access any of its memory, unless
    it is a load, a store or an op that touches none.  No other op CuTe
    lowers takes one today (an atomic declines the split first), so the
    prefetch, counted as such an op, stands in for one."""
    routing = tuple(
        op for op in lane_cluster_split.TENSOR_ROUTING_OPS if op is not hl.prefetch
    )
    with patch.object(lane_cluster_split, "TENSOR_ROUTING_OPS", routing):
        _assert_cluster_split(
            _normalize_rows_prefetched, _BF16_ROWS, _ROLLED_CLUSTER, False
        )


def _assert_cluster_split(
    kernel: Any,
    args: tuple[torch.Tensor, ...],
    config: helion.Config,
    clustered: bool,
    *,
    checked: bool = True,
) -> None:
    """With ``checked``, the race check also runs and decides: a decline by
    an earlier legality check does not pass for one by the race check."""
    bound = cpu_bind(
        helion.kernel(
            kernel, backend="cute", static_shapes=True, autotune_effort="none"
        ),
        args,
    )
    races = lane_cluster_split.cluster_split_races
    verdicts: list[bool] = []

    def recorded(strategy: Any, block_id: int) -> bool:
        verdicts.append(races(strategy, block_id))
        return verdicts[-1]

    with (
        patch.object(lane_cluster_split, "cluster_split_races", recorded),
        patch(f"{CuteLoopedReductionEmitter.__module__}.cluster_split_races", recorded),
    ):
        source = bound.to_code(bound.config_spec.normalized_config(config))
    assert ("_cute_grouped_reduce_cluster(" in source) is clustered
    assert ("_helion_cute_cluster_shape" in source) is clustered
    if checked:
        assert verdicts, "the race check did not run"
        assert any(verdicts) is not clustered, verdicts
