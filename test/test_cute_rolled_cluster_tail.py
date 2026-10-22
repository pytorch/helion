"""CPU codegen coverage for masked, disjoint cluster reduction slices."""

from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import patch

import pytest
import sympy
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.host_function import HostFunction
from helion._compiler.reduction_strategy import LoopedReductionStrategy
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _cpu_only() -> Iterator[None]:
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU-only test")),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
        _forbid_native_compile() as compiles,
    ):
        yield
        for compile_mock in compiles:
            compile_mock.assert_not_called()


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
    bound = _cpu_bind(kernel, (x, weight))
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
        cute_state=SimpleNamespace(simt_cluster_n=2 if reason == "claimed" else 1),
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
                thread_axis_sizes={0: 2} if reason == "multirow" else {}
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
        strategy._maybe_apply_cute_rolled_cluster(state)
    assert strategy._cute_rolled_cluster_n == 1
    assert strategy._mask_var == "mask_0"
