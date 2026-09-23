from __future__ import annotations

import ast
from dataclasses import replace
from itertools import starmap
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast

import pytest

from helion import exc
from helion._compiler.ast_extension import ExtendedAST
from helion._compiler.ast_extension import convert
from helion._compiler.ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from helion._compiler.cute.collective_matmul import MARKER
from helion._compiler.cute.collective_matmul import CollectiveMmaSite
from helion._compiler.cute.collective_matmul import CollectiveSiteAnalysis
from helion._compiler.cute.collective_matmul import _remove_dead_staged_operands
from helion._compiler.cute.collective_register_chain_lowering import _clone_statement
from helion._compiler.cute.collective_register_chain_lowering import _constant_bound
from helion._compiler.cute.collective_register_chain_lowering import _packet_extent
from helion._compiler.cute.collective_register_chain_lowering import (
    _terminal_arena_dead,
)
from helion._compiler.cute.collective_register_chain_lowering import (
    lower_register_chain,
)
from helion._compiler.cute.scalar_recipe import build_recipe

if TYPE_CHECKING:
    from helion._compiler.device_function import DeviceFunction


_THREADS = (1, 4, 16)
_TILE_STOP = """
tile = cutlass.Int32(cute.arch.block_idx()[0]) % 4
stop = 16 * (tile + 1)
"""


def _expr(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


def _site(identity: int = 0, bk: int = 64) -> CollectiveMmaSite:
    return CollectiveMmaSite(
        identity=identity,
        m_index="m",
        n_index="n",
        k_index=f"ki{identity}",
        m_offset="m0",
        n_offset="n0",
        k_offset=f"k{identity}",
        bm=16,
        bn=16,
        bk=bk,
        m_axis=1,
        n_axis=2,
        dtype="cutlass.BFloat16",
        grid_row_lane="row_lane",
    )


def _analysis(
    iterator: str, *, prefix: str = _TILE_STOP, bk: int = 64
) -> CollectiveSiteAnalysis:
    reduction = cast(
        "ast.For",
        ast.parse(
            f"""
for k0 in {iterator}:
    for ki0 in range(k0, k0 + {bk}):
        if ki0 < stop:
            record_left(ki0)
        record_right(ki0)
"""
        ).body[0],
    )
    row = ast.For(ast.Name("row_lane", ast.Store()), _expr("range(4)"), [reduction], [])
    left = build_recipe(_expr("left if ki0 < stop else 0"), [], {"left", "ki0", "stop"})
    right = build_recipe(_expr("right"), [], {"right"})
    assert left is not None and right is not None
    dominating = ast.parse(prefix).body
    return CollectiveSiteAnalysis(
        site=_site(bk=bk),
        row_loop=row,
        reduction_loop=reduction,
        path=[([row], 0), (row.body, 0)],
        row_depth=0,
        reduction_depth=1,
        result_name="acc",
        reduction_iterator=cast("ast.Call", reduction.iter),
        a_recipe=left,
        b_recipe=right,
        seed_recipe=None,
        dominating=dominating,
        seed_dominating=dominating,
        pure_statement_ids=frozenset(),
    )


def _canonical_loop(analysis: CollectiveSiteAnalysis, extent: int) -> ast.For:
    loop = cast("ast.For", _clone_statement(analysis.reduction_loop))
    assert isinstance(loop.iter, ast.Call)
    loop.iter.args = list(
        starmap(
            _constant_bound,
            zip(loop.iter.args, (0, extent, analysis.site.bk), strict=True),
        )
    )
    return loop


def _read_indices(
    prefix: list[ast.stmt], loop: ast.For, block: int
) -> tuple[list[int], list[int]]:
    left: list[int] = []
    right: list[int] = []
    namespace = {
        "cute": SimpleNamespace(arch=SimpleNamespace(block_idx=lambda: (block, 0, 0))),
        "cutlass": SimpleNamespace(Int32=int, Int64=int, Uint32=int, Uint64=int),
        "record_left": left.append,
        "record_right": right.append,
    }
    module = ast.Module([*prefix, loop], [])
    exec(
        compile(ast.fix_missing_locations(module), "<packet-read-indices>", "exec"),
        namespace,
    )
    return left, right


def test_one_packet_across_tile_stops_preserves_masks_and_full_right_domain() -> None:
    analysis = _analysis(
        "range(cutlass.Int32(0), cutlass.Int32(stop), cutlass.Int32(64))"
    )
    original = ast.dump(analysis.reduction_loop, include_attributes=True)
    extent = _packet_extent(analysis, {}, _THREADS)
    assert extent == 64
    canonical = _canonical_loop(analysis, extent)
    assert [ast.dump(node) for node in canonical.body] == [
        ast.dump(node) for node in analysis.reduction_loop.body
    ]
    for block, stop in enumerate((16, 32, 48, 64)):
        before = _read_indices(
            analysis.seed_dominating,
            cast("ast.For", _clone_statement(analysis.reduction_loop)),
            block,
        )
        after = _read_indices(
            analysis.seed_dominating,
            cast("ast.For", _clone_statement(canonical)),
            block,
        )
        assert before == after == (list(range(stop)), list(range(64)))
    assert ast.dump(analysis.reduction_loop, include_attributes=True) == original


def test_packet_count_that_varies_across_tiles_is_rejected() -> None:
    analysis = _analysis("range(0, stop, 32)", bk=32)
    original = ast.dump(analysis.reduction_loop, include_attributes=True)
    with pytest.raises(exc.BackendUnsupported, match="number of K packets varies"):
        _packet_extent(analysis, {}, _THREADS)
    assert ast.dump(analysis.reduction_loop, include_attributes=True) == original


@pytest.mark.parametrize(
    "bound",
    [
        "runtime_stop",
        "0",
        "cutlass.Int32(2147483647)",
        "cutlass.Int32(2147483648)",
        "cutlass.Uint32(4294967360)",
        "cutlass.Int32(cutlass.Uint64(4294967360))",
        "cutlass.Float32(64)",
        "cutlass.Int32(32.0)",
        "64 + cutlass.Int32(cute.arch.block_idx()[0])",
    ],
)
def test_unknown_overflowing_or_wrapping_packet_bound_is_rejected(bound: str) -> None:
    analysis = _analysis(f"range(0, {bound}, 64)", prefix="")
    with pytest.raises(exc.BackendUnsupported):
        _packet_extent(analysis, {}, _THREADS)


def test_unknown_rebinding_invalidates_an_earlier_constant_packet_bound() -> None:
    analysis = _analysis(
        "range(0, stop, 64)", prefix="stop = 64\nstop = runtime_stop\n"
    )
    with pytest.raises(exc.BackendUnsupported):
        _packet_extent(analysis, {}, _THREADS)


@pytest.mark.parametrize(
    "bound,expected",
    [
        ("cutlass.Int32(stop)", "cutlass.Int32(64)"),
        ("cutlass.Int64(cutlass.Int32(stop))", "cutlass.Int64(cutlass.Int32(64))"),
        ("cutlass.Uint32(stop)", "cutlass.Uint32(64)"),
        ("cutlass.Uint64(cutlass.Uint32(stop))", "cutlass.Uint64(cutlass.Uint32(64))"),
    ],
)
def test_proven_bound_retains_integer_casts_iterator_and_unroll(
    bound: str, expected: str
) -> None:
    analysis = _analysis(
        f"cutlass.range(cutlass.Int32(0), {bound}, cutlass.Int32(64), unroll=2)"
    )
    original = ast.dump(analysis.reduction_iterator, include_attributes=True)
    canonical = _canonical_loop(analysis, _packet_extent(analysis, {}, _THREADS))
    assert isinstance(canonical.iter, ast.Call)
    assert ast.dump(canonical.iter) == ast.dump(
        _expr(
            f"cutlass.range(cutlass.Int32(0), {expected}, cutlass.Int32(64), unroll=2)"
        )
    )
    assert ast.dump(analysis.reduction_iterator, include_attributes=True) == original


@pytest.mark.parametrize("extended", [False, True])
def test_cloned_dce_retains_annotations_and_leaves_original_ast_unchanged(
    extended: bool,
) -> None:
    parsed = ast.parse("for lane in range(32):\n    discarded = lane\n    record(17)\n")
    module = cast("ast.Module", convert(parsed)) if extended else parsed
    loop = cast("ast.For", module.body[0])
    metadata = {"owner": "scalarization"}
    setattr(loop, HELION_LANE_LOOP_VAR_ATTR, "lane")
    vars(loop.body[-1])["_test_annotation"] = metadata
    original = ast.dump(module, include_attributes=True)
    copy = cast("ast.For", _clone_statement(loop))
    assert isinstance(copy, ExtendedAST) is extended
    assert getattr(copy, HELION_LANE_LOOP_VAR_ATTR) == "lane"
    assert vars(copy.body[-1])["_test_annotation"] is metadata
    assert not {id(node) for node in ast.walk(loop)} & {
        id(node) for node in ast.walk(copy)
    }
    if extended:
        assert isinstance(loop, ExtendedAST) and isinstance(copy, ExtendedAST)
        assert copy._location is loop._location
        assert copy._loop_type is loop._loop_type
    state = SimpleNamespace(
        collective_mma_sites=[replace(_site(), synthetic_k_lane="lane")],
        collective_mma_pure_stmt_ids={id(loop.body[0])},
    )
    df = cast("DeviceFunction", SimpleNamespace(cute_state=state, dce_vars=[]))
    copied_body: list[ast.stmt] = [copy]
    _remove_dead_staged_operands(copied_body, df, pure_statement_ids={id(copy.body[0])})
    assert len(copied_body) == 1 and isinstance(copied_body[0], ast.Expr)
    assert ast.unparse(copied_body[0]) == "record(17)"
    assert vars(copied_body[0])["_test_annotation"] is metadata
    assert ast.dump(module, include_attributes=True) == original
    assert getattr(loop, HELION_LANE_LOOP_VAR_ATTR) == "lane"
    assert metadata == {"owner": "scalarization"}
    assert state.collective_mma_pure_stmt_ids == {id(loop.body[0])}


def test_late_rejection_does_not_change_source_namespace_or_owned_pure_ids() -> None:
    body = ast.parse(
        f"""
n = n0 + cutlass.Int32(cute.arch.thread_idx()[2])
for row_lane in range(4):
    m = m0 + cutlass.Int32(cute.arch.thread_idx()[1]) * 4 + row_lane
    acc0 = cutlass.Float32(0)
    for k0 in range(0, 32, 32):
        for kl0 in range(32):
            ki0 = k0 + kl0
            a0 = cutlass.BFloat16(1)
            b0 = cutlass.BFloat16(2)
            acc0 = {MARKER}(0, a0, b0, acc0)
    acc1 = cutlass.Float32(0)
    for k1 in range(0, 64, 64):
        for kl1 in range(64):
            ki1 = k1 + kl1
            a1 = cutlass.BFloat16(3)
            b1 = cutlass.BFloat16(4)
            acc1 = {MARKER}(1, a1, b1, acc1)
    pass
"""
    ).body
    module = ast.Module(body, [])
    original = ast.dump(module, include_attributes=True)
    original_nodes = {id(node) for node in ast.walk(module)}
    namespace_calls: list[str] = []

    def new_var(hint: str) -> str:
        namespace_calls.append(hint)
        return f"allocated_{len(namespace_calls)}_{hint}"

    state = SimpleNamespace(
        collective_mma_sites=[_site(0, 32), _site(1, 64)],
        collective_mma_pure_stmt_ids={31},
        collective_mma_emitted_stmt_ids=set(),
        collective_mma_shared_results=set(),
    )
    df = cast(
        "DeviceFunction",
        SimpleNamespace(
            cute_state=state,
            arguments=[],
            dce_vars=["old_dead_value"],
            tile_strategy=SimpleNamespace(thread_block_dims=lambda: _THREADS),
            codegen=SimpleNamespace(max_thread_block_dims=(1, 1, 1)),
            new_var=new_var,
        ),
    )
    # Both complete site proofs run first. The independent zero seed at the
    # second contraction then rejects the whole chain, after local staging.
    with pytest.raises(exc.BackendUnsupported, match="does not consume its preceding"):
        lower_register_chain(
            body,
            df,
            boundary_names={"m0", "n0"},
            disjoint_pairs=set(),
            rename_groups={"acc0": "canonical_acc0"},
        )
    assert ast.dump(module, include_attributes=True) == original
    assert {id(node) for node in ast.walk(module)} == original_nodes
    assert namespace_calls == []
    assert state.collective_mma_pure_stmt_ids == {31}
    assert state.collective_mma_emitted_stmt_ids == set()
    assert state.collective_mma_shared_results == set()
    assert df.dce_vars == ["old_dead_value"]


@pytest.mark.parametrize(
    "suffix",
    [
        "later_shared.store(value)",
        "escape(pointer)",
        "for repeat in range(2):\n    use_shared()",
        "return",
    ],
)
def test_terminal_retirement_proof_rejects_every_later_use_or_unknown_suffix(
    suffix: str,
) -> None:
    region = cast(
        "ast.For", ast.parse("for row in range(4):\n    output.store(value)").body[0]
    )
    body = [region, *ast.parse(suffix).body]
    before = ast.dump(ast.Module(body, []), include_attributes=True)
    assert not _terminal_arena_dead(body, region)
    assert ast.dump(ast.Module(body, []), include_attributes=True) == before


@pytest.mark.parametrize(
    "container", ["for repeat in range(2):", "while active:", "if active:"]
)
def test_terminal_retirement_proof_rejects_an_enclosing_backedge_or_branch(
    container: str,
) -> None:
    outer = ast.parse(
        f"{container}\n    for row in range(4):\n        output.store(value)"
    ).body[0]
    region = cast("ast.For", cast("ast.For", outer).body[0])
    assert not _terminal_arena_dead([outer], region)


def test_terminal_retirement_proof_requires_one_complete_root_region() -> None:
    body = ast.parse(
        "offset = cutlass.Int32(0)\nfor row in range(4):\n    output.store(value)"
    ).body
    region = cast("ast.For", body[-1])
    before = ast.dump(ast.Module(body, []), include_attributes=True)
    assert _terminal_arena_dead(body, region)
    assert not _terminal_arena_dead([region, region], region)
    region.orelse = [ast.Pass()]
    assert not _terminal_arena_dead(body, region)
    region.orelse = []
    assert ast.dump(ast.Module(body, []), include_attributes=True) == before
