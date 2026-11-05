"""CPU-only proofs and source pins for bounded MMA-DAG split stores."""

from __future__ import annotations

import ast
import itertools
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast

import pytest
import sympy
import torch

from test.test_cute_mma_dag import _kernel_body
from test.test_cute_mma_dag import _program
from test.test_cute_mma_dag import _source

import helion
from helion._compiler.cute import cute_mma_dag as dag
from helion._compiler.cute.cute_mma_dag_tcgen05 import tcgen05_facts
from helion._compiler.cute.split_store import integer_bounds
from helion._compiler.cute.split_store import prove_split_store_map
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig
import helion.language as hl
from helion.runtime.cute.launcher import _MmaDagAliasGuard

if TYPE_CHECKING:
    from collections.abc import Generator

    from helion.runtime.kernel import BoundKernel


@pytest.fixture(autouse=True)
def _cpu_only() -> Generator[None, None, None]:
    # Real CPU tensors retain pointer/alias metadata; cpu_bind isolates
    # the compiler's own fake mode without permitting device work.
    with mock_cuda_unavailable(), forbid_cuda_work():
        yield


def _bind_fn(fn: object, args: tuple[torch.Tensor, ...]) -> BoundKernel:
    kernel = helion.kernel(fn, backend="cute", static_shapes=True)  # type: ignore[arg-type]
    return cpu_bind(kernel, args)


ROW, COL = sympy.symbols("row col", integer=True, nonnegative=True)


def _indices(n: int, row_period: int, col_period: int) -> tuple[sympy.Expr, ...]:
    flat = COL + n * ROW
    return (
        sympy.floor(ROW / row_period),
        sympy.Mod(sympy.floor(flat / col_period), n // col_period),
        sympy.Mod(ROW, row_period),
        sympy.Mod(flat, col_period),
    )


def _strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = [1] * len(shape)
    for dim in range(len(shape) - 2, -1, -1):
        strides[dim] = strides[dim + 1] * shape[dim + 1]
    return tuple(strides)


@pytest.mark.parametrize(
    ("expr", "expected"),
    (
        (sympy.floor(sympy.Mul(ROW, sympy.Rational(1, 4))), (0, 11)),
        (sympy.floor(sympy.Mul(ROW, sympy.Rational(-1, 4))), (-12, 0)),
        (sympy.Add(sympy.Mul(ROW, 40), COL), (0, 1919)),
        (sympy.Mod(ROW, 16), (0, 15)),
        (sympy.Mul(ROW, sympy.Rational(1, 4)), None),
        (sympy.Mul(ROW, COL), None),
        (sympy.Symbol("unbounded", integer=True), None),
        (sympy.oo, None),
        (-sympy.oo, None),
        (sympy.nan, None),
    ),
)
def test_integer_bounds_are_finite_exact_endpoints_not_truncated(expr, expected):
    interval = integer_bounds(expr, {ROW: (0, 47), COL: (0, 39)})
    assert interval == expected
    if interval is not None:
        assert all(type(endpoint) is int for endpoint in interval)


def test_split_map_rejects_symbols_without_integer_proof():
    row, col = sympy.symbols("unproved_row unproved_col")
    assert (
        prove_split_store_map(
            (
                sympy.floor(row / 16),
                sympy.floor(col / 8),
                sympy.Mod(row, 16),
                sympy.Mod(col, 8),
            ),
            (row, col),
            (48, 40),
            (3, 5, 16, 8),
            (640, 128, 8, 1),
        )
        is None
    )


@pytest.mark.parametrize("permutation", tuple(itertools.permutations(range(4))))
def test_split_store_map_is_bijective_for_every_dimension_permutation(
    permutation: tuple[int, ...],
) -> None:
    m, n, row_period, col_period = 48, 40, 16, 8
    original_shape = (m // row_period, n // col_period, row_period, col_period)
    shape = tuple(original_shape[dim] for dim in permutation)
    strides = _strides(shape)
    indices = _indices(n, row_period, col_period)
    store_map = prove_split_store_map(
        tuple(indices[dim] for dim in permutation), (ROW, COL), (m, n), shape, strides
    )
    assert store_map is not None
    assert store_map.periods == (row_period, col_period)
    seen: set[int] = set()
    for row in range(m):
        for col in range(n):
            flat = col + n * row
            digits = (
                row // row_period,
                (flat // col_period) % (n // col_period),
                row % row_period,
                flat % col_period,
            )
            expected = sum(
                digits[dim] * stride
                for dim, stride in zip(permutation, strides, strict=True)
            )
            assert store_map.offset(row, col) == expected
            seen.add(expected)
    assert seen == set(range(m * n))


@pytest.mark.parametrize(
    "bad_strides",
    ((640, 128, 8, 0), (640, 128, 8, 8), (640, 128, -8, 1), (1280, 256, 16, 2)),
)
def test_split_store_map_rejects_overlapping_reversed_and_non_dense_layouts(
    bad_strides: tuple[int, ...],
) -> None:
    assert (
        prove_split_store_map(
            _indices(40, 16, 8), (ROW, COL), (48, 40), (3, 5, 16, 8), bad_strides
        )
        is None
    )


@pytest.mark.parametrize(
    "bad_index", (ROW + COL, ROW * COL, ROW + 1, sympy.Symbol("gathered"))
)
def test_split_store_map_rejects_unknown_or_incomplete_digit_maps(
    bad_index: sympy.Expr,
) -> None:
    indices = list(_indices(40, 16, 8))
    indices[0] = bad_index
    assert (
        prove_split_store_map(
            indices, (ROW, COL), (48, 40), (3, 5, 16, 8), (640, 128, 8, 1)
        )
        is None
    )
    assert (
        prove_split_store_map(
            _indices(40, 16, 8), (ROW, COL), (48, 40), (3, 5, 8, 8), (320, 64, 8, 1)
        )
        is None
    )


def test_split_store_tile_bases_and_local_pairs_match_the_global_map() -> None:
    store_map = prove_split_store_map(
        _indices(40, 16, 8), (ROW, COL), (48, 40), (3, 5, 16, 8), (640, 128, 8, 1)
    )
    assert store_map is not None
    assert store_map.tile_reason(16, 8) is None
    for tiles in ((32, 8), (16, 16), (12, 8)):
        assert store_map.tile_reason(*tiles) is not None
    row_stride, col_stride = store_map.inner_strides
    # Evaluate exactly the byte-base terms used by the warp emitter; a
    # nonzero caller pointer offset is carried once, not dropped or doubled.
    pointer, storage_offset, elem_bytes = 4096, 7, 2
    expr = " + ".join(store_map.base_terms("row", "col", elem_bytes))
    for row in range(0, 48, 16):
        for col in range(0, 40, 8):
            base = (
                pointer
                + storage_offset * elem_bytes
                + eval(
                    expr,
                    {"cutlass": SimpleNamespace(Int64=int), "row": row, "col": col},
                )
            )
            for local_row in range(16):
                for local_col in range(0, 8, 2):
                    for bit in (0, 1):
                        got = (
                            base
                            + (local_row * row_stride + (local_col + bit) * col_stride)
                            * elem_bytes
                        )
                        expected = (
                            pointer
                            + (
                                storage_offset
                                + store_map.offset(
                                    row + local_row, col + local_col + bit
                                )
                            )
                            * elem_bytes
                        )
                        assert got == expected


def _three_shared_a_split_stores(
    a: torch.Tensor,
    b: torch.Tensor,
    b1: torch.Tensor,
    b2: torch.Tensor,
    out0: torch.Tensor,
    out1: torch.Tensor,
    out2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, k = a.shape
    n = b.size(1)
    row_period = hl.specialize(out2.size(2))
    col_period = hl.specialize(out2.size(3))
    for tile_m, tile_n in hl.tile((m, n)):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        acc1 = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        acc2 = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            shared_a = a[tile_m, tile_k].to(torch.float32).to(a.dtype)
            rhs = b[tile_k, tile_n].to(torch.float32).to(b.dtype)
            rhs1 = b1[tile_k, tile_n].to(torch.float32).to(b1.dtype)
            rhs2 = b2[tile_k, tile_n].to(torch.float32).to(b2.dtype)
            acc = torch.addmm(acc, shared_a, rhs)
            acc1 = torch.addmm(acc1, shared_a, rhs1)
            acc2 = torch.addmm(acc2, shared_a, rhs2)
        out0[tile_m, tile_n] = acc1.to(out0.dtype)
        out1[tile_m, tile_n] = acc2.to(out1.dtype)
        row = tile_m.index.reshape([tile_m, 1])
        col = tile_n.index.reshape([1, tile_n])
        flat = col + n * row
        out2[
            row // row_period,
            (flat // col_period) % (n // col_period),
            row % row_period,
            flat % col_period,
        ] = acc.to(torch.bfloat16).to(out2.dtype)
    return out0, out1, out2


def _inputs(
    dtype: torch.dtype = torch.float32, offset: int = 0
) -> tuple[torch.Tensor, ...]:
    m, n, k = 96, 96, 32
    shape = (3, 6, 32, 16)
    base = torch.empty(m * n + offset, dtype=dtype)
    return (
        torch.empty((m, k), dtype=torch.bfloat16),
        torch.empty((k, n), dtype=torch.bfloat16),
        torch.empty((k, n), dtype=torch.bfloat16),
        torch.empty((k, n), dtype=torch.bfloat16),
        torch.empty((m, n), dtype=torch.bfloat16),
        torch.empty((m, n), dtype=torch.bfloat16),
        base.as_strided(shape, _strides(shape), offset),
    )


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
@pytest.mark.parametrize("offset", (0, 7))
def test_three_carries_keep_shared_a_rounding_and_original_split_store_pointer(
    dtype: torch.dtype,
    offset: int,
) -> None:
    with forbid_cuda_work():
        args = _inputs(dtype, offset)
        bound = _bind_fn(_three_shared_a_split_stores, args)
        graphs = bound.host_function.device_ir.graphs
        before = tuple(
            tuple(node.format_node() for node in graph.graph.nodes) for graph in graphs
        )
        program = _program(bound)
        assert before == tuple(
            tuple(node.format_node() for node in graph.graph.nodes) for graph in graphs
        )
        assert len(program.carries) == len(program.dots) == len(program.stores) == 3
        assert len({dot.slot for dot in program.dots}) == 3
        assert len({dot.a.chain_key for dot in program.dots}) == 1
        assert all(
            not operand.pointwise for dot in program.dots for operand in (dot.a, dot.b)
        )
        access = program.stores[2].access
        store_map = access.store_map
        assert store_map is not None and store_map.periods == (32, 16)
        assert access.fake.shape == (3, 6, 32, 16)
        assert access.fake.stride() == _strides((3, 6, 32, 16))
        # static_shapes recreates normalized fake input metadata.  The
        # access retains that exact host reference; the runtime argument's
        # iterator supplies its original, possibly offset, pointer.
        host = cast("torch.fx.Node", access.node.args[0])
        assert access.fake is host.meta["val"]
        with bound._runtime_arg_values_for_codegen():
            assert bound.env.runtime_arg_values_by_name["out2"] is args[-1]
        assert args[-1].storage_offset() == offset
        assert access.dtype == dtype
        assert program.stores[0].frag is program.carries[1].phi
        assert program.stores[1].frag is program.carries[2].phi
        assert program.stores[2].frag is program.carries[0].phi
        if dtype == torch.float32:
            rounded = cast("torch.fx.Node", program.stores[2].value.args[0])
            assert rounded.meta["val"].dtype == torch.bfloat16
        facts = bound.config_spec.cute_mma_dag_facts
        assert facts is not None and facts.mapped_stores
        with bound.env, bound.host_function:
            assert tcgen05_facts(program) is None
        seeds = [
            c.config
            for c in bound.config_spec.compiler_seed_configs
            if c.config.get("cute_mma_dag") == "warp"
        ]
        assert seeds
        assert bound.config_spec.cute_mma_dag.choices == ("off", "warp")
        for sizes, warps, stages in dag.mma_dag_seed_configs(facts):
            assert dag.mma_dag_config_reason(facts, sizes, warps, stages) is None
            rows, cols, proof = facts.mapped_stores[0]
            assert (
                proof.tile_reason(
                    facts.block_size(rows, sizes), facts.block_size(cols, sizes)
                )
                is None
            )
        source = _source(bound, helion.Config.from_dict(seeds[0]))
        body = _kernel_body(source)
        assert ".mma_bf16(" in body and "tcgen05_mma(" not in body
        mapped_base = next(
            line for line in body.splitlines() if "out2.iterator.toint()" in line
        )
        terms = store_map.base_terms(
            f"md_a{access.dims[0].index}", f"md_a{access.dims[1].index}", dtype.itemsize
        )
        assert all(term in mapped_base for term in terms)
        assert "helion_mma_dag_alias" in source


def test_split_store_config_rejects_crossing_tiles_and_tcgen05() -> None:
    with forbid_cuda_work():
        bound = _bind_fn(_three_shared_a_split_stores, _inputs())
        facts = bound.config_spec.cute_mma_dag_facts
        assert facts is not None
        sizes = {axis.block_id: 32 for axis in facts.axes if axis.tunable}
        rows, cols, _proof = facts.mapped_stores[0]
        sizes[facts.axes[cols].block_id] = 16
        assert dag.mma_dag_config_reason(facts, sizes, 4, 2) is None
        for axis_index, too_wide in ((rows, 64), (cols, 32)):
            bad = dict(sizes)
            bad[facts.axes[axis_index].block_id] = too_wide
            reason = dag.mma_dag_config_reason(facts, bad, 4, 2)
            assert reason is not None and (
                "split period" in reason or "must divide the extent" in reason
            )
        reason = dag.mma_dag_config_reason(facts, sizes, 4, 2, "tcgen05")
        assert reason is not None and "no tcgen05 region-store proof" in reason
        with bound.env, pytest.raises(InvalidConfig, match="cute_mma_dag"):
            bound.config_spec.normalized_config(
                helion.Config(block_sizes=[32, 16, 32], cute_mma_dag="tcgen05")
            )


def test_split_store_runtime_alias_guard_and_overlapping_stride_rejection() -> None:
    with forbid_cuda_work():
        args = _inputs(torch.bfloat16)
        bound = _bind_fn(_three_shared_a_split_stores, args)
        source = _source(bound, bound.config_spec.compiler_seed_configs[0])
        plans = next(
            ast.literal_eval(node.value)
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute)
                and target.attr == "_helion_cute_wrapper_plans"
                for target in node.targets
            )
        )
        plan = next(plan for plan in plans if plan["kind"] == "helion_mma_dag_alias")
        guard = _MmaDagAliasGuard(plan, args)
        guard.check(args)
        aliased = list(args)
        aliased[-1] = args[-3].as_strided((3, 6, 32, 16), _strides((3, 6, 32, 16)))
        with pytest.raises(BackendUnsupported, match="alias"):
            guard.check(tuple(aliased))
        overlapping = list(_inputs())
        overlapping[-1] = overlapping[-1].as_strided((3, 6, 32, 16), (3072, 512, 16, 0))
        assert (
            _bind_fn(
                _three_shared_a_split_stores, tuple(overlapping)
            ).config_spec.cute_mma_dag
            is None
        )


@pytest.mark.parametrize("invalid", ("truncation", "overflow", "mask", "known_alias"))
def test_split_store_fx_proof_fails_closed_on_invalid_metadata(invalid: str) -> None:
    with forbid_cuda_work():
        bound = _bind_fn(_three_shared_a_split_stores, _inputs(torch.bfloat16))
        program = _program(bound)
        mapped = program.stores[2].access
        indices = cast("list[torch.fx.Node]", mapped.node.args[1])
        if invalid == "truncation":
            indices[0].kwargs = {"rounding_mode": "trunc"}
        elif invalid == "overflow":
            product = next(
                node
                for node in mapped.node.graph.nodes
                if node.target is torch.ops.aten.mul.Tensor
                and isinstance(node.meta.get("val"), torch.Tensor)
                and node.meta["val"].dtype == torch.int32
            )
            product.args = (product.args[0], 2**31)
        elif invalid == "mask":
            mapped.node.args = (*mapped.node.args[:3], indices[0])
        else:
            # Emulate known fake-storage alias metadata.  Static input
            # normalization may not carry runtime aliases, which are tested
            # separately against the emitted launch guard above.
            out0 = program.stores[0].access.fake
            host = cast("torch.fx.Node", mapped.node.args[0])
            host.meta["val"] = out0.view(mapped.fake.shape)
        with bound.env, bound.host_function:
            assert dag.match_mma_dag(bound.host_function.device_ir.graphs) is None


def test_split_store_proof_rejects_an_extra_root_axis() -> None:
    with forbid_cuda_work():
        bound = _bind_fn(_three_shared_a_split_stores, _inputs())
        with bound.env, bound.host_function:
            matcher = dag._Matcher(bound.host_function.device_ir.graphs)
            program = matcher.match()
            assert program is not None
            access = program.stores[2].access
            # The otherwise identical access would be written once per
            # additional root coordinate: its two digit pairs do not own it.
            matcher.root_block_ids = (*matcher.root_block_ids, -1)
            with pytest.raises(dag._NotMine):
                matcher.split_store_access(
                    access.node, access.fake, access.name, access.node.args[1]
                )


def test_split_store_with_noncontiguous_columns_uses_scalar_addresses() -> None:
    with forbid_cuda_work():
        args = list(_inputs())
        args[-1] = args[-1].as_strided((3, 6, 32, 16), (3072, 512, 1, 32))
        bound = _bind_fn(_three_shared_a_split_stores, tuple(args))
        program = _program(bound)
        access = program.stores[2].access
        assert access.store_map is not None
        assert access.store_map.inner_strides == (1, 32)
        source = _source(bound, bound.config_spec.compiler_seed_configs[0])
        body = _kernel_body(source)
        assert ".mma_bf16(" in body and ".stg_f32(" in body
        base = next(
            line for line in body.splitlines() if "out2.iterator.toint()" in line
        )
        assert all(
            term in base
            for term in access.store_map.base_terms(
                f"md_a{access.dims[0].index}", f"md_a{access.dims[1].index}", 4
            )
        )
