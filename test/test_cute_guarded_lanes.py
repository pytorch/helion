"""GPU-free coverage of ``cute_guarded_lanes``: the pass that runs a lane's
loads and arithmetic under the guard of the stores they feed
(``helion/_compiler/cute/guarded_lanes.py``), the predicated packet loads of
the lane-packet emitter and the gathered-coordinate bounds the tile masks
imply (``gathered_bounds.py``)."""

from __future__ import annotations

import ast
import random
import re
import textwrap

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute.autotune.search_surface import extremum_packet_loads
from helion._compiler.cute.emitters import lane_packets
from helion._compiler.cute.guarded_lanes import guard_lanes
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion._testing.guarded_lane_kernels import flags_reread_after_a_store
from helion._testing.guarded_lane_kernels import neighbor_max_into
from helion._testing.guarded_lane_kernels import packet_tile_seeds
from helion._testing.guarded_lane_kernels import padded_and_transposed
from helion._testing.guarded_lane_kernels import padded_sum
from helion._testing.guarded_lane_kernels import pooled_max
from helion._testing.guarded_lane_kernels import sorted_kept_rows
from helion._testing.guarded_lane_kernels import window_max
from helion._testing.guarded_lane_kernels import window_max_into
from helion._testing.unfold_kernels import sliding_windows
import helion.language as hl

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]

_VT = "ir.VectorType.get([8], cutlass.Uint16.mlir_type)"


def _guard(source: str, plain_loads: dict[str, str] | None = None) -> str:
    body = ast.parse(textwrap.dedent(source)).body
    return ast.unparse(
        ast.Module(body=guard_lanes(body, plain_loads or {}), type_ignores=[])
    )


def _same(source: str) -> None:
    """The pass leaves ``source`` as it is."""
    expected = ast.unparse(ast.parse(textwrap.dedent(source)))
    assert _guard(source) == expected


def _guard_without_store_targets(source: str) -> str:
    """``_guard`` of ``source`` with its loop targets built as a generator
    that sets no context builds them: plain names (``ast.Load``)."""
    body = ast.parse(textwrap.dedent(source)).body
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.For):
            for name in ast.walk(node.target):
                if isinstance(name, ast.Name):
                    name.ctx = ast.Load()
    return ast.unparse(ast.Module(body=guard_lanes(body, {}), type_ignores=[]))


# ---------------------------------------------------------------------------
# The pass on its own
# ---------------------------------------------------------------------------


def test_packet_region_moves_under_its_store_guard() -> None:
    """Anchored packets, their lane selects and the lane arithmetic move
    under the flush's guard; a select the guard implies is dropped (a tile
    mask, and the lane mask through its single definition) and one it does
    not imply keeps its remaining conjunct."""
    code = _guard(
        f"""
        mask_1 = indices_1 < 73
        for lane_3 in range(1):
            lane_base_3 = tile_offset_3 + thread * 8
            p0 = cute.arch.load(x.iterator + lane_base_3 if mask_1 and lane_base_3 < 64 else x.iterator, {_VT})
            p1 = cute.arch.load(y.iterator + lane_base_3 if pad and mask_1 else y.iterator, {_VT})
            vals = []
            for vec_lane_3 in cutlass.range_constexpr(8):
                mask_3 = lane_base_3 < 64
                a = cutlass.Uint16(p0[vec_lane_3]).bitcast(cutlass.BFloat16) if mask_1 and mask_3 else cutlass.BFloat16(0)
                b = cutlass.Uint16(p1[vec_lane_3]).bitcast(cutlass.BFloat16) if pad and (mask_1 and mask_3) else cutlass.BFloat16(0)
                vals.append((cutlass.Float32(a) + cutlass.Float32(b)).bitcast(cutlass.Uint16))
            if mask_1 and mask_3:
                _cute_store_u16_vec(out.iterator + lane_base_3, vals)
        """
    )
    expected = ast.unparse(
        ast.parse(
            textwrap.dedent(
                f"""
                mask_1 = indices_1 < 73
                for lane_3 in range(1):
                    lane_base_3 = tile_offset_3 + thread * 8
                    mask_3 = lane_base_3 < 64
                    if mask_1 and mask_3:
                        p0 = cute.arch.load(x.iterator + lane_base_3, {_VT})
                        p1 = cute.arch.load(y.iterator + lane_base_3 if pad else y.iterator, {_VT})
                        vals = []
                        for vec_lane_3 in cutlass.range_constexpr(8):
                            a = cutlass.Uint16(p0[vec_lane_3]).bitcast(cutlass.BFloat16)
                            b = cutlass.Uint16(p1[vec_lane_3]).bitcast(cutlass.BFloat16) if pad else cutlass.BFloat16(0)
                            vals.append((cutlass.Float32(a) + cutlass.Float32(b)).bitcast(cutlass.Uint16))
                        _cute_store_u16_vec(out.iterator + lane_base_3, vals)
                """
            )
        )
    )
    assert code == expected


def test_predicated_packets_keep_what_the_guard_does_not_imply() -> None:
    """A predicated packet loses the conjuncts the guard implies; one the
    lane-packet emitter recorded becomes its unpredicated load when the
    guard implies the whole predicate, and any other keeps its predicate."""
    source = f"""
        p0 = _cute_load_v16b_if(x.iterator + base, mask_1 and pad, {_VT}, 'evict_last')
        p1 = _cute_load_v16b_if(x.iterator + base + 8, mask_1, {_VT}, 'l1_l2_last')
        p2 = _cute_load_v16b_if(x.iterator + base + 16, mask_1, {_VT})
        vals = []
        for v in cutlass.range_constexpr(8):
            vals.append(p0[v] + p1[v] + p2[v])
        if mask_1:
            _cute_store_u16_vec(out.iterator + base, vals)
        """
    code = _guard(source, {"p1": f"_cute_load_l1_l2_evict_last({{pointer}}, {_VT})"})
    assert f"p0 = _cute_load_v16b_if(x.iterator + base, pad, {_VT}, 'evict_last')" in (
        code
    )
    assert f"p1 = _cute_load_l1_l2_evict_last(x.iterator + base + 8, {_VT})" in code
    # No record of an unpredicated form: the predicated load stays.
    assert f"p2 = _cute_load_v16b_if(x.iterator + base + 16, mask_1, {_VT})" in code
    assert code.index("if mask_1:") < code.index("p0 = ")


def test_scalar_lane_moves_under_its_store_guard() -> None:
    """The scalar form: masked loads and arithmetic of a lane loop's body
    ahead of a guarded plain store; the masks the guard reads stay ahead."""
    code = _guard(
        """
        for lane_2 in range(2):
            indices_2 = tile_offset_2 + lane_2
            mask_2 = indices_2 < 27
            col = indices_2 * 2
            a = (x.iterator + col).load() if mask_1 and mask_2 else cutlass.BFloat16(0)
            b = (x.iterator + col + 1).load() if mask_1 and (mask_2 and True) else cutlass.BFloat16(0)
            m = cute.arch.fmax(cutlass.Float32(a), cutlass.Float32(b), nan=True)
            if mask_1 and mask_2:
                (out.iterator + indices_2).store(cutlass.BFloat16(m))
        """
    )
    expected = ast.unparse(
        ast.parse(
            textwrap.dedent(
                """
                for lane_2 in range(2):
                    indices_2 = tile_offset_2 + lane_2
                    mask_2 = indices_2 < 27
                    if mask_1 and mask_2:
                        col = indices_2 * 2
                        a = (x.iterator + col).load()
                        b = (x.iterator + col + 1).load()
                        m = cute.arch.fmax(cutlass.Float32(a), cutlass.Float32(b), nan=True)
                        (out.iterator + indices_2).store(cutlass.BFloat16(m))
                """
            )
        )
    )
    assert code == expected


def test_stores_under_one_guard_share_the_region() -> None:
    code = _guard(
        """
        a = (x.iterator + i).load() if mask else cutlass.Float32(0)
        b = a * 2.0
        if mask:
            (out.iterator + i).store(a)
        if mask:
            (out2.iterator + i).store(b)
        """
    )
    expected = ast.unparse(
        ast.parse(
            textwrap.dedent(
                """
                if mask:
                    a = (x.iterator + i).load()
                    b = a * 2.0
                    (out.iterator + i).store(a)
                    (out2.iterator + i).store(b)
                """
            )
        )
    )
    assert code == expected


def test_stores_under_different_guards_keep_the_shared_lanes() -> None:
    """A value that reaches a store under another guard is read outside the
    first store's region, so the statements computing it stay where they
    are; what only the second store reads moves under its guard."""
    code = _guard(
        """
        a = (x.iterator + i).load() if mask_1 else cutlass.Float32(0)
        if mask_1 and mask_2:
            (out.iterator + i).store(a)
        c = a + 1.0
        if mask_1:
            (out2.iterator + i).store(c)
        """
    )
    expected = ast.unparse(
        ast.parse(
            textwrap.dedent(
                """
                a = (x.iterator + i).load() if mask_1 else cutlass.Float32(0)
                if mask_1 and mask_2:
                    (out.iterator + i).store(a)
                if mask_1:
                    c = a + 1.0
                    (out2.iterator + i).store(c)
                """
            )
        )
    )
    assert code == expected


# One nest per decline of the pass (the module docstring of
# ``guarded_lanes``): each has to come out as it went in.
@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            """
            a = (x.iterator + i).load() if mask else cutlass.Float32(0)
            if mask:
                (out.iterator + i).store(a)
            total = total + a
            """,
            id="value-read-after-the-store",
        ),
        pytest.param(
            """
            acc = acc + (x.iterator + i).load()
            if mask:
                (out.iterator + i).store(acc)
            """,
            id="value-carried-into-the-region",
        ),
        pytest.param(
            """
            (scratch.iterator + i).store(cutlass.Float32(1.0))
            if mask:
                (out.iterator + i).store(cutlass.Float32(2.0))
            """,
            id="store-ahead-of-the-guarded-store",
        ),
        pytest.param(
            """
            a = _cute_unknown_helper(x, i)
            if mask:
                (out.iterator + i).store(a)
            """,
            id="call-with-unknown-effects",
        ),
        pytest.param(
            """
            a = (x.iterator + i).load()
            if mask:
                (out.iterator + i).store(a)
            else:
                (out.iterator + i).store(cutlass.Float32(0))
            """,
            id="store-with-an-else-branch",
        ),
        pytest.param(
            """
            a = (x.iterator + i).load()
            if mask:
                cute.arch.atomic_add(out.iterator + i, a)
            """,
            id="guarded-atomic",
        ),
        pytest.param(
            """
            a = (x.iterator + i).load() if mask else cutlass.Float32(0)
            if mask:
                (out.iterator + i).store(a)
            a = cutlass.Float32(0)
            """,
            id="name-also-assigned-outside-the-region",
        ),
        pytest.param(
            """
            for lane in range(4):
                a = (x.iterator + lane).load()
            if mask:
                (out.iterator + i).store(a)
            """,
            id="runtime-loop-ahead-of-the-store",
        ),
        pytest.param(
            """
            a = (x.iterator + i).load() if mask_1 else cutlass.Float32(0)
            if mask_1 and mask_2:
                (out.iterator + i).store(a)
            if mask_1:
                (out2.iterator + i).store(a)
            """,
            id="adjacent-stores-under-different-guards",
        ),
        pytest.param(
            f"""
            p0 = cute.arch.load(x.iterator + base, {_VT})
            vals = []
            for v in cutlass.range_constexpr(8):
                vals.append(p0[v])
            if mask_1:
                _cute_store_u16_vec(out1.iterator + base, vals)
            if mask_2:
                _cute_store_u16_vec(out2.iterator + base, vals)
            """,
            id="list-flushed-under-two-guards",
        ),
        pytest.param(
            """
            vals = []
            cute.arch.sync_threads()
            for v in cutlass.range_constexpr(8):
                vals.append(p0[v])
            if mask:
                _cute_store_u16_vec(out.iterator + base, vals)
            """,
            id="list-created-ahead-of-a-barrier",
        ),
        pytest.param(
            """
            vals = []
            for v in cutlass.range_constexpr(8):
                vals.append(p0[v])
            if mask:
                _cute_store_u16_vec(out.iterator + base, vals)
            total = vals[0] + q
            """,
            id="list-read-after-the-stores",
        ),
        pytest.param(
            f"""
            p0 = cute.arch.load(x.iterator + base, {_VT})
            vals = []
            for v in cutlass.range_constexpr(4):
                mask = base < 13
                vals.append(p0[v] if mask else cutlass.Uint16(0))
            for v in cutlass.range_constexpr(4):
                mask = base < 9
                vals.append(p0[v + 4] if mask else cutlass.Uint16(0))
            if mask:
                _cute_store_u16_vec(out.iterator + base, vals)
            """,
            id="guard-definition-assigned-in-two-loops",
        ),
        pytest.param(
            """
            a = (x.iterator + i).load() if mask else cutlass.Float32(0)
            if cute.arch.atomic_add(x.iterator + i, 1) > 0:
                (out.iterator + i).store(a)
            """,
            id="guard-with-effects",
        ),
        pytest.param(
            f"""
            p0 = cute.arch.load(x.iterator + base, {_VT})
            vals = []
            for v in cutlass.range_constexpr(_cute_unknown_helper(x)):
                vals.append(p0[v])
            if mask:
                _cute_store_u16_vec(out.iterator + base, vals)
            """,
            id="loop-bound-with-unknown-effects",
        ),
    ],
)
def test_region_is_left_alone(source: str) -> None:
    _same(source)


def test_definition_is_not_expanded_past_a_rebinding() -> None:
    """``mask`` was computed from the row of the previous trip (``row`` is
    bound after it), so the guard does not imply ``row < 8`` as the select
    evaluates it; and a name bound twice is never expanded."""
    code = _guard(
        """
        for lane in range(4):
            mask = row < 8
            row = base + lane
            a = (x.iterator + row).load() if row < 8 else cutlass.Float32(0)
            if mask:
                (out.iterator + lane).store(a)
        """
    )
    assert "a = (x.iterator + row).load() if row < 8 else cutlass.Float32(0)" in code
    code = _guard(
        """
        mask = row < 8
        mask = mask and other
        a = (x.iterator + row).load() if row < 8 else cutlass.Float32(0)
        if mask:
            (out.iterator + row).store(a)
        """
    )
    assert "a = (x.iterator + row).load() if row < 8 else cutlass.Float32(0)" in code


def test_definition_is_not_expanded_over_an_operand_bound_twice() -> None:
    """``row`` moved on after ``mask`` was computed from it, so the guard
    says nothing about ``row < 8`` as the select evaluates it."""
    code = _guard(
        """
        row = base
        mask = row < 8
        row = row + 1
        a = (x.iterator + row).load() if row < 8 else cutlass.Float32(0)
        if mask:
            (out.iterator + i).store(a)
        """
    )
    assert "a = (x.iterator + row).load() if row < 8 else cutlass.Float32(0)" in code
    assert code.index("if mask:") < code.index("a = ")


def test_definition_inside_a_branch_is_not_expanded() -> None:
    """A definition under a branch the guard is outside of need not have run
    when the guard does."""
    code = _guard(
        """
        if flag:
            mask = row < 8
        a = (x.iterator + row).load() if row < 8 else cutlass.Float32(0)
        if mask:
            (out.iterator + row).store(a)
        """
    )
    assert "a = (x.iterator + row).load() if row < 8 else cutlass.Float32(0)" in code


# The mask tensor is stored between two reads of it, each spelled as a load
# through a pointer and as an element of the tensor (subscript, getitem).
@pytest.mark.parametrize(
    ("read", "store"),
    [
        (
            "(flags.iterator + row).load() != cutlass.Uint8(0)",
            "(flags.iterator + row).store(cutlass.Boolean(old == 0))",
        ),
        ("flags[row] != cutlass.Uint8(0)", "flags[row] = cutlass.Boolean(old == 0)"),
        (
            "operator.getitem(flags, row) != cutlass.Uint8(0)",
            "flags[row] = cutlass.Boolean(old == 0)",
        ),
    ],
)
def test_read_of_memory_is_not_implied_through_a_definition(
    read: str, store: str
) -> None:
    """``old`` and ``new`` are one expression on either side of a store into
    the ``flags`` it reads, so they are two values: the guard's ``old`` says
    nothing about ``new``, whose select stays, while the conjunct of its
    definition that reads no memory still holds and the region still moves
    under the guard.  Definitions that read no memory are one value (an
    element of a thread index or of a shape is no read of memory), and both
    selects go."""
    source = """
        old = {read} and row < limit
        {store}
        new = {read} and row < limit
        if other:
            (seen.iterator + row).store(new)
        a = (x.iterator + row).load() if new else cutlass.Float32(0)
        b = (y.iterator + row).load() if row < limit else cutlass.Float32(0)
        if old:
            (out.iterator + row).store(a + b)
        """
    code = _guard(source.format(read=read, store=store))
    assert "a = (x.iterator + row).load() if new else cutlass.Float32(0)" in code
    assert "b = (y.iterator + row).load()\n" in code
    assert code.index("if old:") < code.index("a = ")
    immutable = "cute.arch.thread_idx()[0] < x.shape[0]"
    code = _guard(source.format(read=immutable, store=store))
    assert "a = (x.iterator + row).load()\n" in code
    assert "b = (y.iterator + row).load()\n" in code


def test_conjunct_that_reads_memory_is_kept() -> None:
    """The guard's own conjunct, a load, is read again by a select of the
    second store, after the first store changed what it reads: the select
    is kept although it is the guard's expression, and so is a select
    ahead of the stores (a conjunct that reads memory is never matched)."""
    flag = "(flags.iterator + row).load() != cutlass.Uint8(0)"
    code = _guard(
        f"""
        a = (x.iterator + row).load() if {flag} else cutlass.Float32(0)
        if {flag}:
            (flags.iterator + row).store(cutlass.Boolean(0))
            (out.iterator + row).store(a if {flag} else cutlass.Float32(0))
        """
    )
    assert code.count(f"if {flag} else cutlass.Float32(0)") == 2
    assert code.index(f"if {flag}:") < code.index("a = ")


def test_stores_under_a_guard_that_reads_memory_stay_apart() -> None:
    """Two stores under one expression that reads memory are not under one
    guard: the second test is evaluated after the first store, which here
    clears the flag it reads.  Each is a region of its own, so only the
    statement ahead of the first moves, under the first."""
    flag = "(flags.iterator + row).load() != cutlass.Uint8(0)"
    code = _guard(
        f"""
        a = (x.iterator + row).load()
        if {flag}:
            (flags.iterator + row).store(cutlass.Boolean(0))
        if {flag}:
            (out.iterator + row).store(a + a)
        """
    )
    assert code.count(f"if {flag}:") == 2
    code = _guard(
        f"""
        a = (x.iterator + row).load()
        if {flag}:
            (seen.iterator + row).store(a)
        if {flag}:
            (out.iterator + row).store(cutlass.Float32(1))
        """
    )
    assert code.count(f"if {flag}:") == 2
    assert code.index(f"if {flag}:") < code.index("a = ")


def test_guard_definition_stays_in_a_loop_it_depends_on() -> None:
    """A mask the constexpr loop computes per lane cannot move ahead of the
    region, so the loop stays ahead of the guard."""
    _same(
        f"""
        p0 = cute.arch.load(x.iterator + base, {_VT})
        vals = []
        for v in cutlass.range_constexpr(8):
            mask = base + v < 13
            vals.append(p0[v])
        if mask:
            _cute_store_u16_vec(out.iterator + base, vals)
        """
    )


def test_loop_targets_without_a_store_context_are_bound() -> None:
    """A loop's target is bound by its loop whatever context the node
    carries: a mask computed from the index stays in its loop (and the loop
    ahead of the guard), a definition over the index is looked up without
    a missing binding, and a region whose loop computes nothing the guard
    reads moves as it does with ``Store`` targets."""
    stays = f"""
        p0 = cute.arch.load(x.iterator + base, {_VT})
        vals = []
        for v in cutlass.range_constexpr(8):
            mask = base + v < 13
            vals.append(p0[v])
        if mask:
            _cute_store_u16_vec(out.iterator + base, vals)
        """
    expanded = """
        for lane in range(4):
            mask = base + lane < 8
            a = (x.iterator + lane).load() if mask else cutlass.Float32(0)
            if mask:
                (out.iterator + lane).store(a)
        """
    moves = f"""
        mask = base + 8 < 64
        p0 = cute.arch.load(x.iterator + base if mask else x.iterator, {_VT})
        vals = []
        for v in cutlass.range_constexpr(8):
            vals.append(p0[v])
        if mask:
            _cute_store_u16_vec(out.iterator + base, vals)
        """
    for source in (stays, expanded, moves):
        assert _guard_without_store_targets(source) == _guard(source)
    assert _guard(stays) == ast.unparse(ast.parse(textwrap.dedent(stays)))
    assert "a = (x.iterator + lane).load()\n" in _guard(expanded)
    assert _guard(moves).index("if mask:") < _guard(moves).index("for v in")


def test_guard_definition_leaves_what_it_reads_ahead_of_the_region() -> None:
    """The mask the guard reads moves out of the constexpr loop; the value
    it is computed from is then read outside the region and stays ahead of
    it, where the mask finds it."""
    code = _guard(
        f"""
        p0 = cute.arch.load(x.iterator + base, {_VT})
        limit = base + 8
        vals = []
        for v in cutlass.range_constexpr(8):
            mask = limit < 64
            vals.append(p0[v])
        if mask:
            _cute_store_u16_vec(out.iterator + base, vals)
        """
    )
    expected = ast.unparse(
        ast.parse(
            textwrap.dedent(
                f"""
                p0 = cute.arch.load(x.iterator + base, {_VT})
                limit = base + 8
                mask = limit < 64
                if mask:
                    vals = []
                    for v in cutlass.range_constexpr(8):
                        vals.append(p0[v])
                    _cute_store_u16_vec(out.iterator + base, vals)
                """
            )
        )
    )
    assert code == expected


# ---------------------------------------------------------------------------
# Generated kernels
# ---------------------------------------------------------------------------


def _code(
    kernel: object,
    args: torch.Tensor | tuple[torch.Tensor, ...],
    *,
    guarded: bool,
    **config: object,
) -> str:
    values: dict[str, object] = dict(config)
    if guarded:
        values["cute_guarded_lanes"] = True
    with mock_cuda_unavailable():
        bound = cpu_bind(kernel, args if isinstance(args, tuple) else (args,))
        return bound.to_code(helion.Config.from_dict(values))


def _kernel_body(code: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )


def _guards_with_loads(code: str) -> list[ast.If]:
    """The ``if`` statements of the kernel that hold a packet or scalar
    load and a store."""
    found = []
    for node in ast.walk(_kernel_body(code)):
        if not isinstance(node, ast.If):
            continue
        text = ast.unparse(ast.Module(body=node.body, type_ignores=[]))
        if ("load(" in text or "_cute_load_" in text) and "store" in text:
            found.append(node)
    return found


# Partial tiles on H and W (9 and 7 outputs in blocks of 4), whole 16-byte
# packets on C.
_POOL_CONFIG = {
    "block_sizes": [4, 4, 16],
    "num_threads": [4, 4, 2],
    "cute_vector_widths": [1, 1, 8],
}


def test_window_packets_run_under_the_store_guard() -> None:
    x = torch.zeros([18, 14, 16], dtype=torch.bfloat16)
    before = _code(window_max, x, guarded=False, **_POOL_CONFIG)
    # Knob off: anchored packets with a select per lane, as before.
    assert before.count(" else x.iterator") == 4
    assert "cutlass.BFloat16(0)" in before
    assert not _guards_with_loads(before)
    after = _code(window_max, x, guarded=True, **_POOL_CONFIG)
    (guard,) = _guards_with_loads(after)
    body = ast.unparse(ast.Module(body=guard.body, type_ignores=[]))
    # Four plain packets, no anchor, no select, no bound of a coordinate.
    assert body.count("cute.arch.load(x.iterator") == 4
    assert " else " not in body
    assert "_cute_load_v16b_if" not in after
    assert "< 18" not in after and "< 14" not in after
    assert "cute.math.max(" in body
    assert "_cute_store_u16_vec(" in body


def test_padded_packets_load_predicated() -> None:
    """The lanes of a padded load take no select: the packet loads
    predicated on what the store's guard does not imply (the padding mask,
    and the bound of the coordinate that can pass the image), with the
    site's cache hint; the 32-byte packet too."""
    x = torch.zeros([9, 7, 32], dtype=torch.bfloat16)
    config = {
        "block_sizes": [4, 4, 32],
        "num_threads": [4, 4, 2],
        "cute_vector_widths": [1, 1, 16],
        "load_eviction_policies": ["l1_l2_last", "last", ""],
    }
    before = _code(padded_sum, x, guarded=False, **config)
    assert "_cute_load_v32b_if" not in before
    after = _code(padded_sum, x, guarded=True, **config)
    (guard,) = _guards_with_loads(after)
    body = ast.unparse(ast.Module(body=guard.body, type_ignores=[]))
    predicated = [
        node
        for node in ast.walk(guard)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "_cute_load_v32b_if"
    ]
    assert len(predicated) == 2
    up, down = (ast.unparse(node.args[1]) for node in predicated)
    # The bound of ``h - 1`` follows from the tile mask and is gone (the
    # coordinate can be negative, which the bound never tested); that of
    # ``h + 1``, which can pass the image, stays.
    assert "cutlass.Int32(" not in up
    assert "cutlass.Int32(" in down and "< 9" in down
    assert ast.unparse(predicated[0].args[3]) == "'l1_l2_last'"
    assert ast.unparse(predicated[1].args[3]) == "'evict_last'"
    # The tile's own packet has a dead fill: a plain load under the guard.
    assert body.count("cute.arch.load(x.iterator") == 1
    assert "cutlass.BFloat16(0)" not in after


# The lane tile is partial (48 channels in tiles of 32), so the flush is
# guarded by the mask of its last lane, which the V-loop computes: the pass
# leaves the nest alone and the packets keep the form the emitter gave them.
_PARTIAL_LANE_TILE = {
    "block_sizes": [4, 4, 32],
    "num_threads": [4, 4, 2],
    "cute_vector_widths": [1, 1, 16],
}


def _calls(code: str, name: str) -> int:
    return sum(
        isinstance(node, ast.Call) and ast.unparse(node.func) == name
        for node in ast.walk(_kernel_body(code))
    )


def test_fp16_packets_keep_the_anchored_form() -> None:
    """ptxas widens the wrong half of a predicated packet's word for an
    fp16 lane without a select (``PREDICATED_LANE_DTYPES``): an fp16 packet
    stays an anchored load with its lane selects where a bf16 one loads
    predicated.  Where the stores' guard implies its whole mask, the pass
    still moves it under the guard: a plain load, its lanes without a
    select, as for bf16."""
    halves = {
        dtype: _code(
            window_max,
            torch.zeros([18, 14, 48], dtype=dtype),
            guarded=True,
            **_PARTIAL_LANE_TILE,
        )
        for dtype in (torch.bfloat16, torch.float16)
    }
    assert _calls(halves[torch.bfloat16], "_cute_load_v32b_if") == 4
    assert "cutlass.BFloat16(0)" not in halves[torch.bfloat16]
    assert _calls(halves[torch.float16], "_cute_load_v32b_if") == 0
    assert halves[torch.float16].count(" else x.iterator") == 4
    assert halves[torch.float16].count(" else cutlass.Float16(0)") == 4
    # The padded loads of a nest the pass does move: anchored, under the
    # guard, each lane selecting on the padding mask.
    padded = _code(
        padded_sum,
        torch.zeros([9, 7, 32], dtype=torch.float16),
        guarded=True,
        **_PARTIAL_LANE_TILE,
    )
    assert "_cute_load_v32b_if" not in padded
    (guard,) = _guards_with_loads(padded)
    body = ast.unparse(ast.Module(body=guard.body, type_ignores=[]))
    assert body.count(" else x.iterator") == 2
    assert body.count(" else cutlass.Float16(0)") == 2
    # Whole packets under a guard that implies their mask.
    whole = _code(
        window_max,
        torch.zeros([18, 14, 16], dtype=torch.float16),
        guarded=True,
        **_POOL_CONFIG,
    )
    assert "_cute_load_v16b_if" not in whole
    (guard,) = _guards_with_loads(whole)
    body = ast.unparse(ast.Module(body=guard.body, type_ignores=[]))
    assert body.count("cute.arch.load(x.iterator") == 4
    assert body.count(".bitcast(cutlass.Float16)") == 4
    assert " else " not in body


def test_sorted_packets_keep_the_anchored_form() -> None:
    """A sort (or a top-k) lowers the masked value its load records for it,
    which a predicated packet would replace: the load keeps the anchored
    packet and the lane selects it has without the knob, so the kernel
    renders the knob-off code, and its packet-tile seeds render too."""
    keep = torch.zeros([9], dtype=torch.bool)
    config = {
        "block_sizes": [4, 32],
        "num_threads": [4, 4],
        "cute_vector_widths": [1, 8],
    }
    for dtype in (torch.bfloat16, torch.float32):
        args = (torch.zeros([9, 32], dtype=dtype), keep)
        before = _code(sorted_kept_rows, args, guarded=False, **config)
        after = _code(sorted_kept_rows, args, guarded=True, **config)
        assert before.count(" else x.iterator") == 1
        assert "_if(x.iterator" not in after
        assert after == before
    kernel = helion.kernel(
        sorted_kept_rows.fn, backend="cute", static_shapes=True, autotune_effort="full"
    )
    with mock_cuda_unavailable():
        bound = cpu_bind(
            kernel,
            (
                torch.zeros([64, 64], dtype=torch.bfloat16),
                torch.zeros([64], dtype=torch.bool),
            ),
        )
        seeds = [
            seed
            for seed in bound.config_spec.compiler_seed_configs
            if seed.config.get("cute_guarded_lanes") is True
        ]
        assert [seed.config["block_sizes"] for seed in seeds] == [[16, 64], [16, 64]]
        for seed in seeds:
            assert "torch.sort" in bound.to_code(seed)


def test_byte_packets_keep_the_anchored_form() -> None:
    """An int8 lane is a byte of a packed integer, which has no predicated
    load: the padded packets stay anchored and their lanes keep the select."""
    code = _code(
        padded_sum,
        torch.zeros([9, 7, 32], dtype=torch.int8),
        guarded=True,
        block_sizes=[4, 4, 32],
        num_threads=[4, 4, 4],
        cute_vector_widths=[1, 1, 8],
    )
    assert "_cute_load_v" not in code
    assert code.count(" else cutlass.Int8(0)") == 2


def test_dead_fill_packet_stays_anchored_where_the_pass_declines() -> None:
    """A packet whose fill no store observes needs neither a predicate nor a
    select: it is the anchored load it was, next to the predicated packets
    of the padded loads."""
    code = _code(
        padded_sum,
        torch.zeros([9, 7, 48], dtype=torch.bfloat16),
        guarded=True,
        **_PARTIAL_LANE_TILE,
    )
    assert not _guards_with_loads(code)
    assert _calls(code, "_cute_load_v32b_if") == 2
    assert _calls(code, "cute.arch.load") == 1
    assert code.count(" else x.iterator") == 1
    assert "cutlass.BFloat16(0)" not in code


def test_packet_without_a_guard_keeps_its_lane_selects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A packet every thread loads in bounds has no predicate to load under,
    so it cannot leave zeros: lanes that asked for a predicated packet take
    their fill from the select again."""
    monkeypatch.setattr(
        lane_packets.TilePacketGuard,
        "build",
        lambda self: lane_packets.PacketGuard(None, ""),
    )
    code = _code(
        padded_sum,
        torch.zeros([9, 7, 32], dtype=torch.bfloat16),
        guarded=True,
        **_PARTIAL_LANE_TILE,
    )
    assert "_cute_load_v32b_if" not in code
    assert _calls(code, "cute.arch.load") == 3
    assert " else x.iterator" not in code
    assert code.count(" else cutlass.BFloat16(0)") == 2


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _mirrored_rows(
    x: torch.Tensor, y: torch.Tensor, keep: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # One packet of ``x`` read twice: by a store of its own tiles (a dead
    # fill) and by a store at mirrored rows (which sees the fill).
    rows, columns = x.shape
    out = torch.empty_like(x)
    mirrored = torch.zeros_like(x)
    for tile_r, tile_c in hl.tile([rows, columns]):
        kept = hl.load(y, [tile_r, tile_c], extra_mask=keep[tile_r][:, None])
        out[tile_r, tile_c] = x[tile_r, tile_c] + kept
        mirrored[(rows - 1) - tile_r.index, tile_c] = x[tile_r, tile_c] * 2.0
    return out, mirrored


def test_one_packet_loads_in_both_forms() -> None:
    """The dead-fill read of a packet and the read whose fill is observed
    are two loads, anchored and predicated: neither takes a select."""
    x = torch.zeros([9, 48], dtype=torch.bfloat16)
    with mock_cuda_unavailable():
        bound = cpu_bind(_mirrored_rows, (x, x, torch.zeros([9], dtype=torch.bool)))
        code = bound.to_code(
            helion.Config.from_dict(
                {
                    "block_sizes": [4, 32],
                    "num_threads": [4, 4],
                    "cute_vector_widths": [1, 8],
                    "cute_guarded_lanes": True,
                }
            )
        )
    loads = [
        ast.unparse(node)
        for node in ast.walk(_kernel_body(code))
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) in ("cute.arch.load", "_cute_load_v16b_if")
        and ast.unparse(node.args[0]).startswith("x.iterator")
    ]
    assert sorted(text.split("(", 1)[0] for text in loads) == [
        "_cute_load_v16b_if",
        "cute.arch.load",
    ]
    assert " else cutlass.BFloat16(0)" not in code


def test_lane_values_of_a_packet_stay_ahead_of_a_per_lane_store() -> None:
    """A lane's value goes to the packet flushed after the V-loop and to a
    store of its own inside it.  The guard of the per-lane store takes
    nothing with it: the append that hands the value to the packet stays in
    the loop, with the load and the select it reads."""
    args = tuple(
        torch.zeros(shape, dtype=torch.bfloat16)
        for shape in ([9, 41], [9, 48], [48, 9])
    )
    config = {
        "block_sizes": [1, 32],
        "num_threads": [0, 4],
        "cute_vector_widths": [1, 8],
    }
    after = _kernel_body(_code(padded_and_transposed, args, guarded=True, **config))
    (vloop,) = [
        node
        for node in ast.walk(after)
        if isinstance(node, ast.For) and "range_constexpr" in ast.unparse(node.iter)
    ]
    assert sum(".append(" in ast.unparse(stmt) for stmt in vloop.body) == 1
    (guard,) = [stmt for stmt in vloop.body if isinstance(stmt, ast.If)]
    assert [ast.unparse(stmt).rpartition(".store(")[1] for stmt in guard.body] == [
        ".store("
    ]
    # Nothing else of the kernel moves either.
    before = _kernel_body(_code(padded_and_transposed, args, guarded=False, **config))
    assert ast.unparse(after) == ast.unparse(before)


def test_scalar_window_loads_run_under_the_store_guard() -> None:
    """Without vector lanes the loads are scalar: they lose their selects
    under the guard the same way."""
    x = torch.zeros([18, 14, 16], dtype=torch.bfloat16)
    config = {"block_sizes": [4, 4, 16], "num_threads": [4, 4, 16]}
    before = _code(window_max, x, guarded=False, **config)
    assert "cutlass.BFloat16(0)" in before
    after = _code(window_max, x, guarded=True, **config)
    (guard,) = _guards_with_loads(after)
    body = ast.unparse(ast.Module(body=guard.body, type_ignores=[]))
    assert body.count(".load()") == 4
    assert " else " not in body


def test_mask_reread_after_a_store_keeps_its_select() -> None:
    """The load is masked by the flag as read after the store that inverts
    it, and its value is stored under the flag as read before: two loads of
    one address, so the guard does not imply the load's mask and the lane
    keeps that select under it."""
    args = (torch.zeros([37]), torch.zeros([37], dtype=torch.bool))
    config = {"block_sizes": [8], "num_threads": [8]}
    before = _code(flags_reread_after_a_store, args, guarded=False, **config)
    assert ".load() if new and mask_0 else cutlass.Float32(0)" in before
    after = _code(flags_reread_after_a_store, args, guarded=True, **config)
    (guard,) = (
        node
        for node in ast.walk(_kernel_body(after))
        if isinstance(node, ast.If) and ast.unparse(node.test) == "old and mask_0"
    )
    body = ast.unparse(ast.Module(body=guard.body, type_ignores=[]))
    # The tile mask goes, the reread flag stays.
    assert ".load() if new else cutlass.Float32(0)" in body
    assert "mask_0" not in body


def test_bound_is_kept_where_the_masks_do_not_imply_it() -> None:
    """A dynamic extent has no static range, so every coordinate keeps its
    bound against the tensor's size."""
    kernel = helion.kernel(
        window_max.fn, backend="cute", static_shapes=False, autotune_effort="none"
    )
    x = torch.zeros([18, 14, 16], dtype=torch.bfloat16)
    before = _code(kernel, x, guarded=False, **_POOL_CONFIG)
    after = _code(kernel, x, guarded=True, **_POOL_CONFIG)
    for code in (before, after):
        assert "< x_size_0" in code and "< x_size_1" in code
    assert after.count("< x_size_0") == before.count("< x_size_0")


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _gathered_rows(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    e: torch.Tensor,
    f: torch.Tensor,
    g: torch.Tensor,
) -> torch.Tensor:
    # Rows of seven sources (each with its own number of rows, which names
    # its bound in the code) gathered over a 9 x 7 tile of rows.
    columns = a.size(1)
    out = torch.empty([9, 7, columns], dtype=a.dtype, device=a.device)
    for tile_h, tile_w, tile_c in hl.tile([9, 7, columns]):
        h = tile_h.index
        w = tile_w.index
        out[tile_h, tile_w, tile_c] = (
            a[(h * 2 + 1)[:, None], tile_c]
            + b[(h * 2 + 1)[:, None], tile_c]
            + c[(8 - h)[:, None], tile_c]
            + d[(h * 1)[:, None] - (w * 1)[None, :] + 3, tile_c]
            + e[((h + 2147483647) - 2147483647)[:, None], tile_c]
            + f[(h * -1 + 8)[:, None], tile_c]
            + g[(h % 5)[:, None], tile_c]
        )
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _rows_of_a_later_tile(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # The row tile starts at row 2 of 9.
    out = torch.zeros([9, x.size(1)], dtype=x.dtype, device=x.device)
    for tile_h, tile_c in hl.tile([2, 0], [9, x.size(1)]):
        h = tile_h.index
        out[tile_h, tile_c] = x[h - 1, tile_c] + y[h + 1, tile_c]
    return out


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _static_rows_of_a_dynamic_tensor(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([8, x.size(1)], dtype=x.dtype, device=x.device)
    for tile_h, tile_c in hl.tile([8, x.size(1)]):
        out[tile_h, tile_c] = x[tile_h.index * 2, tile_c]
    return out


def _coordinate_bounds(code: str) -> list[str]:
    """What the gathered coordinates of the kernel are compared against:
    the right-hand sides of its ``cutlass.Int32(v_N) < SIZE`` terms."""
    return sorted(
        set(re.findall(r"cutlass\.Int32\(v_\d+\) < (\w+)", code)),
        key=lambda size: (len(size), size),
    )


def test_coordinate_bound_follows_the_range_of_the_coordinate() -> None:
    """Signed interval arithmetic over the tile's rows 0..8 and columns
    0..6: a bound goes where the top of the coordinate's range is below the
    number of rows, and stays where it is not or the range is not modeled."""

    def rows(count: int) -> torch.Tensor:
        return torch.zeros([count, 16])

    # In the order of ``_gathered_rows``' sources:
    #   2h + 1 in [1, 17] against 18 rows: implied by the tile mask;
    #   the same against 17 rows: the last row is past the tensor;
    #   8 - h in [0, 8] against 10 rows: implied (the subtrahend's range is
    #     reversed);
    #   h - w + 3 in [-3, 11] against 11 rows: the top is h's top less w's
    #     bottom, past the tensor;
    #   (h + 2**31 - 1) - (2**31 - 1): an intermediate leaves the int32
    #     range, where the generated arithmetic wraps;
    #   -h + 8 in [0, 8] against 13 rows: implied (a negative factor swaps
    #     the ends of the range);
    #   h % 5 against 50 rows: not an operation the analysis models.
    args = (rows(18), rows(17), rows(10), rows(11), rows(12), rows(13), rows(50))
    config = {"block_sizes": [4, 4, 16], "num_threads": [4, 4, 16]}
    before = _code(_gathered_rows, args, guarded=False, **config)
    assert _coordinate_bounds(before) == ["10", "11", "12", "13", "17", "18", "50"]
    after = _code(_gathered_rows, args, guarded=True, **config)
    assert _coordinate_bounds(after) == ["11", "12", "17", "50"]


def test_coordinate_bound_counts_from_the_start_of_the_tile() -> None:
    """Rows 2..8 of a tile that starts at row 2: ``h - 1`` stays below the
    12 rows of its tensor and ``h + 1`` reaches the 9th row of a tensor of
    9."""
    args = (torch.zeros([12, 16]), torch.zeros([9, 16]))
    config = {"block_sizes": [4, 16], "num_threads": [4, 16]}
    before = _code(_rows_of_a_later_tile, args, guarded=False, **config)
    assert _coordinate_bounds(before) == ["9", "12"]
    after = _code(_rows_of_a_later_tile, args, guarded=True, **config)
    assert _coordinate_bounds(after) == ["9"]


def test_coordinate_bound_is_kept_against_a_symbolic_size() -> None:
    """A static tile over a tensor of dynamic shape: the coordinate's range
    is known, the number of rows it must stay below is not."""
    args = (torch.zeros([18, 16]),)
    config = {"block_sizes": [4, 16], "num_threads": [4, 16]}
    for guarded in (False, True):
        code = _code(_static_rows_of_a_dynamic_tensor, args, guarded=guarded, **config)
        assert _coordinate_bounds(code) == ["x_size_0"]


def test_knob_is_offered_on_gathered_and_padded_loads_only() -> None:
    @helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
    def plain(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        for tile_r, tile_c in hl.tile(x.shape):
            out[tile_r, tile_c] = x[tile_r, tile_c] * 2.0
        return out

    @helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
    def padded_row_sums(x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        # A padded load, but in a kernel with a reduction.
        out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
        for tile_r in hl.tile(x.size(0)):
            kept = hl.load(x, [tile_r, slice(None)], extra_mask=keep[tile_r][:, None])
            out[tile_r] = kept.sum(-1)
        return out

    with mock_cuda_unavailable():
        for kernel, args in (
            (plain, (torch.zeros([9, 16]),)),
            (padded_row_sums, (torch.zeros([9, 16]), torch.zeros([9]).bool())),
        ):
            spec = cpu_bind(kernel, args).config_spec
            assert "cute_guarded_lanes" not in spec._flat_fields()
            with pytest.raises(exc.InvalidConfig, match="cute_guarded_lanes requires"):
                spec.normalize(
                    spec.default_config().config | {"cute_guarded_lanes": True}
                )
            # The knob-off value a search writes is dropped where it is not
            # offered.
            config = spec.default_config().config | {"cute_guarded_lanes": False}
            spec.normalize(config)
            assert "cute_guarded_lanes" not in config
        for kernel in (window_max, padded_sum):
            fingerprints = set()
            for shape in ([18, 14, 16], [20, 16, 16], [34, 6, 64]):
                spec = cpu_bind(
                    kernel, (torch.zeros(shape, dtype=torch.bfloat16),)
                ).config_spec
                assert "cute_guarded_lanes" in spec._flat_fields()
                fingerprints.add(spec.structural_fingerprint())
            # One search space at every shape.
            assert len(fingerprints) == 1


# ---------------------------------------------------------------------------
# Seeds and the search surface
# ---------------------------------------------------------------------------


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def _plane_window_max(x: torch.Tensor) -> torch.Tensor:
    # x: [P, H, W]: the window gathers along the contiguous dim.
    planes, height, width = x.shape
    out = torch.empty([planes, height // 2, width // 2], dtype=x.dtype, device=x.device)
    for tile_p, tile_h, tile_w in hl.tile([planes, height // 2, width // 2]):
        row = (tile_h.index * 2)[:, None]
        col = (tile_w.index * 2)[None, :]
        best = x[tile_p, row, col].to(torch.float32)
        value = x[tile_p, row + 1, col + 1].to(torch.float32)
        out[tile_p, tile_h, tile_w] = torch.maximum(best, value).to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def _row_and_every_other_max(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # y: [P, H, W], read by rows; x: [P, H, 2 * W], read at every other
    # column: one load takes the row's tile, the other gathers along it.
    planes, height, width = y.shape
    out = torch.empty_like(y)
    for tile_p, tile_h, tile_w in hl.tile([planes, height, width]):
        plane = tile_p.index[:, None, None]
        row = tile_h.index[None, :, None]
        col = (tile_w.index * 2)[None, None, :]
        out[tile_p, tile_h, tile_w] = torch.maximum(
            y[tile_p, tile_h, tile_w], x[plane, row, col]
        )
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def _pooled_mean_leaky(x: torch.Tensor) -> torch.Tensor:
    # 3x3 average pooling, stride 2, of a [H, W, C] activation and a leaky
    # ReLU written as a maximum: a lane adds its nine packets up, and the
    # extremum's operands are computed values.
    height, width, channels = x.shape
    out_h = (height - 3) // 2 + 1
    out_w = (width - 3) // 2 + 1
    out = torch.empty([out_h, out_w, channels], dtype=x.dtype, device=x.device)
    for tile_h, tile_w, tile_c in hl.tile([out_h, out_w, channels]):
        row = (tile_h.index * 2)[:, None]
        col = (tile_w.index * 2)[None, :]
        total = x[row, col, tile_c].to(torch.float32)
        for offset in hl.static_range(1, 9):
            tap = x[row + offset // 3, col + offset % 3, tile_c]
            total = total + tap.to(torch.float32)
        mean = total * (1.0 / 9.0)
        out[tile_h, tile_w, tile_c] = torch.maximum(mean, mean * 0.1).to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def _two_row_maxima(x: torch.Tensor) -> torch.Tensor:
    # The sum of the maxima of two neighbouring 1x5 windows of a [H, W, C]
    # activation: ten packet loads per lane, five to each extremum.
    height, width, channels = x.shape
    out_w = (width - 10) // 2 + 1
    out = torch.empty([height, out_w, channels], dtype=x.dtype, device=x.device)
    for tile_h, tile_w, tile_c in hl.tile([height, out_w, channels]):
        row = tile_h.index[:, None]
        col = (tile_w.index * 2)[None, :]
        left = x[row, col, tile_c].to(torch.float32)
        right = x[row, col + 5, tile_c].to(torch.float32)
        for offset in hl.static_range(1, 5):
            left = torch.maximum(left, x[row, col + offset, tile_c].to(torch.float32))
            right = torch.maximum(
                right, x[row, col + 5 + offset, tile_c].to(torch.float32)
            )
        out[tile_h, tile_w, tile_c] = (left + right).to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def _row_maxima_floored(x: torch.Tensor, floor: torch.Tensor) -> torch.Tensor:
    # The maxima of the 1x7 windows of a [H, W, C] activation and of a floor
    # per output pixel: eight loads to one extremum, seven of them packets
    # (the floor is one value for all of a pixel's channels).
    height, width, channels = x.shape
    out_w = (width - 7) // 2 + 1
    out = torch.empty([height, out_w, channels], dtype=x.dtype, device=x.device)
    for tile_h, tile_w, tile_c in hl.tile([height, out_w, channels]):
        row = tile_h.index[:, None]
        col = (tile_w.index * 2)[None, :]
        best = x[row, col, tile_c].to(torch.float32)
        for offset in hl.static_range(1, 7):
            best = torch.maximum(best, x[row, col + offset, tile_c].to(torch.float32))
        low = floor[tile_h.index[:, None, None], tile_w.index[None, :, None]]
        best = torch.maximum(best, low.to(torch.float32))
        out[tile_h, tile_w, tile_c] = best.to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def _tap_floor(x: torch.Tensor, floor: torch.Tensor) -> torch.Tensor:
    # One strided tap of a [H, W, C] activation against a per-channel floor:
    # two packet loads per lane, the floor's through a broadcast view.
    height, width, channels = x.shape
    out = torch.empty(
        [height // 2, width // 2, channels], dtype=x.dtype, device=x.device
    )
    for tile_h, tile_w, tile_c in hl.tile([height // 2, width // 2, channels]):
        row = (tile_h.index * 2)[:, None]
        col = (tile_w.index * 2)[None, :]
        value = x[row, col, tile_c].to(torch.float32)
        out[tile_h, tile_w, tile_c] = torch.maximum(
            value, floor[tile_c].to(torch.float32)[None, None, :]
        ).to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _transposed_kept_rows(x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    # The loads take packets along the columns of x, which the store lays
    # out at its largest stride.
    rows, columns = x.shape
    out_t = torch.empty([columns, rows], dtype=x.dtype, device=x.device)
    for tile_r, tile_c in hl.tile([rows, columns]):
        kept = hl.load(x, [tile_r, tile_c], extra_mask=keep[tile_r][:, None])
        out_t[tile_c, tile_r] = kept.permute(1, 0)
    return out_t


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def _transposed_and_spread_rows(
    x: torch.Tensor, keep: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # ``_transposed_kept_rows`` beside a store of the rows to every other
    # row of a second output: a computed coordinate of the row tile, but a
    # store's.
    rows, columns = x.shape
    out_t = torch.empty([columns, rows], dtype=x.dtype, device=x.device)
    spread = torch.zeros([2 * rows, columns], dtype=x.dtype, device=x.device)
    for tile_r, tile_c in hl.tile([rows, columns]):
        kept = hl.load(x, [tile_r, tile_c], extra_mask=keep[tile_r][:, None])
        out_t[tile_c, tile_r] = kept.permute(1, 0)
        spread[tile_r.index * 2, tile_c] = kept
    return out_t, spread


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def _window_max_into_both(
    x: torch.Tensor, out: torch.Tensor, other: torch.Tensor
) -> None:
    # The maxima of two taps of a [H, W, C] activation, stored into two
    # [H // 2, W // 2, C] views.
    out_h, out_w, channels = out.shape
    for tile_h, tile_w, tile_c in hl.tile([out_h, out_w, channels]):
        row = (tile_h.index * 2)[:, None]
        col = (tile_w.index * 2)[None, :]
        best = torch.maximum(x[row, col, tile_c], x[row + 1, col + 1, tile_c])
        out[tile_h, tile_w, tile_c] = best
        other[tile_h, tile_w, tile_c] = best


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def _every_other_row_into(x: torch.Tensor, out: torch.Tensor) -> None:
    # Every other row of a [2 * H, W, C] activation: one load per stored
    # element, through a computed coordinate along H alone.
    out_h, out_w, channels = out.shape
    for tile_h, tile_w, tile_c in hl.tile([out_h, out_w, channels]):
        out[tile_h, tile_w, tile_c] = x[tile_h.index * 2, tile_w, tile_c]


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def _shifted_floor_into(
    x: torch.Tensor, floor: torch.Tensor, out: torch.Tensor
) -> None:
    # A [H, W + 1, C] activation one column over, against a per-channel
    # floor: one load of the tiles per stored element (the tile's index plus
    # one keeps its tile) beside a packet load along the channels alone.
    out_h, out_w, channels = out.shape
    for tile_h, tile_w, tile_c in hl.tile([out_h, out_w, channels]):
        value = x[tile_h, tile_w.index + 1, tile_c].to(torch.float32)
        out[tile_h, tile_w, tile_c] = torch.maximum(
            value, floor[tile_c].to(torch.float32)[None, None, :]
        ).to(out.dtype)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def _both_columns_into(x: torch.Tensor, out: torch.Tensor, right: torch.Tensor) -> None:
    # The two taps of ``neighbor_max_into``, each copied to a view of its
    # own: two loads and two stores.
    out_h, out_w, channels = out.shape
    for tile_h, tile_w, tile_c in hl.tile([out_h, out_w, channels]):
        out[tile_h, tile_w, tile_c] = x[tile_h, tile_w, tile_c]
        right[tile_h, tile_w, tile_c] = x[tile_h, tile_w.index + 1, tile_c]


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def _kept_copies_into(
    x: torch.Tensor,
    y: torch.Tensor,
    keep: torch.Tensor,
    out: torch.Tensor,
    spread: torch.Tensor,
) -> None:
    # Two copies under a row mask, the second stored to every other row of
    # ``spread``: a store through a computed coordinate.
    out_h, out_w, channels = out.shape
    for tile_h, tile_w, tile_c in hl.tile([out_h, out_w, channels]):
        mask = keep[tile_h][:, None, None]
        out[tile_h, tile_w, tile_c] = hl.load(
            x, [tile_h, tile_w, tile_c], extra_mask=mask
        )
        spread[tile_h.index * 2, tile_w, tile_c] = hl.load(
            y, [tile_h, tile_w, tile_c], extra_mask=mask
        )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def _kept_and_scattered_copies_into(
    x: torch.Tensor,
    y: torch.Tensor,
    keep: torch.Tensor,
    rows: torch.Tensor,
    out: torch.Tensor,
    scattered: torch.Tensor,
) -> None:
    # ``_kept_copies_into`` with the second copy stored to the rows that
    # ``rows`` names: a store through a loaded index.
    out_h, out_w, channels = out.shape
    for tile_h, tile_w, tile_c in hl.tile([out_h, out_w, channels]):
        mask = keep[tile_h][:, None, None]
        out[tile_h, tile_w, tile_c] = hl.load(
            x, [tile_h, tile_w, tile_c], extra_mask=mask
        )
        scattered[rows[tile_h], tile_w, tile_c] = hl.load(
            y, [tile_h, tile_w, tile_c], extra_mask=mask
        )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def _window_max_and_spread_into(
    x: torch.Tensor, out: torch.Tensor, spread: torch.Tensor
) -> None:
    # The window of ``window_max_into``, stored to ``out`` and to every other
    # row of ``spread``.
    out_h, out_w, channels = out.shape
    for tile_h, tile_w, tile_c in hl.tile([out_h, out_w, channels]):
        row = (tile_h.index * 2)[:, None]
        col = (tile_w.index * 2)[None, :]
        best = x[row, col, tile_c].to(torch.float32)
        for offset in hl.static_range(1, 4):
            value = x[row + offset // 2, col + offset % 2, tile_c].to(torch.float32)
            best = torch.maximum(best, value)
        out[tile_h, tile_w, tile_c] = best.to(out.dtype)
        spread[tile_h.index * 2, tile_w, tile_c] = best.to(spread.dtype)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def _neighbor_extrema_into(
    x: torch.Tensor, high: torch.Tensor, low: torch.Tensor
) -> None:
    # The two taps of ``neighbor_max_into``, their maximum and their minimum
    # stored to a view each: a window with as many outputs as taps.
    out_h, out_w, channels = high.shape
    for tile_h, tile_w, tile_c in hl.tile([out_h, out_w, channels]):
        left = x[tile_h, tile_w, tile_c].to(torch.float32)
        right = x[tile_h, tile_w.index + 1, tile_c].to(torch.float32)
        high[tile_h, tile_w, tile_c] = torch.maximum(left, right).to(high.dtype)
        low[tile_h, tile_w, tile_c] = torch.minimum(left, right).to(low.dtype)


def _folded_packets(bound: object) -> int:
    """The packets one fp32 extremum of ``bound`` folds along its last tile."""
    return extremum_packet_loads(
        bound.host_function.device_ir,  # type: ignore[attr-defined]
        bound.config_spec.block_sizes[-1].block_id,  # type: ignore[attr-defined]
    )


def _folded_packets_and_minimums(
    kernel: object, args: tuple[object, ...]
) -> tuple[int, list[object]]:
    """``_folded_packets`` of ``kernel`` bound to ``args``, and the
    ``cute_min_blocks_per_mp`` each of its packet tile seeds names."""
    bound = cpu_bind(kernel, args)
    return _folded_packets(bound), [
        seed.get("cute_min_blocks_per_mp")
        for seed in packet_tile_seeds(bound.config_spec)
    ]


def test_packet_tile_seeds_put_the_packet_threads_on_thread_x() -> None:
    """A window over a channels-last activation: 64 channels are eight
    16-byte packets, so a CTA of 128 threads takes 4 x 4 pixels of them (a
    warp holds four pixels' whole rows), in storage order; the sibling takes
    16 pixels of one image row.  The window's four packets are too few for
    a twin with a CTAs-per-SM minimum (the test below)."""
    kernel = helion.kernel(
        window_max.fn, backend="cute", static_shapes=True, autotune_effort="full"
    )
    with mock_cuda_unavailable():
        bound = cpu_bind(kernel, (torch.zeros([146, 146, 64], dtype=torch.bfloat16),))
        spec = bound.config_spec
        assert _folded_packets(bound) == 4
        seeds = packet_tile_seeds(spec)
        assert [
            (seed["block_sizes"], seed["num_threads"], "load_eviction_policies" in seed)
            for seed in seeds
        ] == [
            ([4, 4, 64], [4, 4, 8], False),
            ([4, 4, 64], [4, 4, 8], True),
            ([1, 16, 64], [1, 16, 8], False),
        ]
        for seed in seeds:
            assert seed["loop_orders"] == [[2, 1, 0]]
            assert seed["cute_vector_widths"] == [1, 1, 8]
            assert seed["cute_fmnmx_extrema"] is True
            assert "cute_min_blocks_per_mp" not in seed
            # The seed is a config of the search space as it stands.
            normalized = dict(seed)
            spec.normalize(normalized)
            assert normalized["block_sizes"] == seed["block_sizes"]
            assert normalized["loop_orders"] == seed["loop_orders"]
        # 192 channels: tiles of 64, the largest power of two dividing them.
        spec = cpu_bind(
            kernel, (torch.zeros([34, 34, 192], dtype=torch.bfloat16),)
        ).config_spec
        assert [seed["block_sizes"] for seed in packet_tile_seeds(spec)] == [
            [4, 4, 64],
            [4, 4, 64],
            [1, 16, 64],
        ]
        # A window along the contiguous dim loads scalars there: no packet
        # tile, although its 64 outputs per row would take whole packets.
        spec = cpu_bind(
            _plane_window_max, (torch.zeros([64, 56, 128], dtype=torch.bfloat16),)
        ).config_spec
        assert "cute_guarded_lanes" in spec._flat_fields()
        assert packet_tile_seeds(spec) == []
        # One gathering load is enough: the lane it shares with a load of
        # the row's own tile reads that row element by element too.
        spec = cpu_bind(
            _row_and_every_other_max,
            (
                torch.zeros([8, 16, 128], dtype=torch.bfloat16),
                torch.zeros([8, 16, 64], dtype=torch.bfloat16),
            ),
        ).config_spec
        assert "cute_guarded_lanes" in spec._flat_fields()
        assert packet_tile_seeds(spec) == []


def test_packet_tile_seeds_name_a_cta_minimum_where_an_extremum_folds_eight_packets() -> (
    None
):
    """Where one fp32 extremum folds eight or more of a lane's packets (the
    nine taps of a 3x3 max pooling) the first packet tile is seeded once
    more with ``cute_min_blocks_per_mp``: last, after the seeds a kernel has
    without it, and the option changes the launch alone, not the kernel's
    text.  Seven packets get no twin (``_packet_tile_seeds`` says where the
    count comes from), and neither do packets that reach the extremum
    through other arithmetic or a view, ten packets that two extrema fold
    five apiece, nor seven packets folded with a load that is not one."""
    kernel = helion.kernel(
        pooled_max.fn, backend="cute", static_shapes=True, autotune_effort="full"
    )
    x = torch.zeros([35, 39, 64], dtype=torch.bfloat16)
    with mock_cuda_unavailable():
        bound = cpu_bind(kernel, (x, 3, 3))
        spec = bound.config_spec
        assert _folded_packets(bound) == 9
        seeds = packet_tile_seeds(spec)
        assert [
            (seed["block_sizes"], seed.get("cute_min_blocks_per_mp")) for seed in seeds
        ] == [
            ([4, 4, 64], None),
            ([4, 4, 64], None),
            ([1, 16, 64], None),
            ([4, 4, 64], 1),
        ]
        twin = dict(seeds[-1])
        del twin["cute_min_blocks_per_mp"]
        assert twin == seeds[0]
        # A value of the option's search choices: the search moves both ways.
        assert spec._flat_fields()["cute_min_blocks_per_mp"].choices[:2] == (0, 1)
        with_minimum = bound.to_code(helion.Config.from_dict(seeds[-1]))
        attribute = "_helion_pooled_max._helion_cute_min_blocks_per_mp = 1"
        assert with_minimum.count(attribute) == 1
        assert bound.to_code(helion.Config.from_dict(seeds[0])).splitlines() == [
            line for line in with_minimum.splitlines() if line != attribute
        ]

        # Eight packets (a 2x4 window) get the twin, seven (1x7) do not.
        assert _folded_packets_and_minimums(kernel, (x, 2, 4)) == (
            8,
            [None, None, None, 1],
        )
        assert _folded_packets_and_minimums(kernel, (x, 1, 7)) == (7, [None] * 3)
        # Nine packets a lane adds up before its one extremum.
        assert _folded_packets_and_minimums(_pooled_mean_leaky, (x,)) == (
            0,
            [None] * 3,
        )
        # Ten packets, five to each of two extrema.
        assert _folded_packets_and_minimums(_two_row_maxima, (x,)) == (5, [None] * 3)
        # Eight loads to one extremum, one of them not a packet.
        pixel_floor = torch.zeros([35, 17], dtype=torch.bfloat16)
        assert _folded_packets_and_minimums(_row_maxima_floored, (x, pixel_floor)) == (
            7,
            [None] * 3,
        )
        # Two packets, one of them behind a broadcast view.
        floor = torch.zeros([64], dtype=torch.bfloat16)
        assert _folded_packets_and_minimums(_tap_floor, (x, floor)) == (1, [None] * 3)


def test_packet_tile_seeds_follow_the_stores_of_a_window_that_changes_the_layout() -> (
    None
):
    """A window over a channels-last activation stored into a view of another
    layout: the loads take their packets along the channels, which no store
    has at stride 1, so the tiles have no storage order that begins with the
    packet block.  The packet tile is seeded all the same, the packet threads
    on thread x and the other tiles behind them as the store lays them out,
    so the threads that fill the warp run along the store's stride-1 axis:
    the image row of a channel-first buffer (the order a channels-last
    output has), the image column of a buffer that stores its planes column
    by column.  The seeds are configs of the search space as it stands, and
    the lanes keep their packets and store one element each.

    No seed where the stores disagree on the layout, and none for a copy,
    whose loads index the tiles themselves (a computed coordinate of a store
    does not make it a window)."""
    kernel = helion.kernel(
        window_max_into.fn, backend="cute", static_shapes=True, autotune_effort="full"
    )
    bf16 = torch.bfloat16
    x = torch.zeros([146, 146, 64], dtype=bf16)
    channels_last = torch.zeros([73, 73, 64], dtype=bf16)
    # [C, H, W] and [C, W, H] buffers as [H, W, C] views.
    channel_first = torch.zeros([64, 73, 73], dtype=bf16).permute(1, 2, 0)
    by_columns = torch.zeros([64, 73, 73], dtype=bf16).permute(2, 1, 0)
    with mock_cuda_unavailable():
        for out, order, sibling in (
            (channels_last, [2, 1, 0], ([1, 16, 64], [1, 16, 8])),
            (channel_first, [2, 1, 0], ([1, 16, 64], [1, 16, 8])),
            (by_columns, [2, 0, 1], ([16, 1, 64], [16, 1, 8])),
        ):
            bound = cpu_bind(kernel, (x, out))
            spec = bound.config_spec
            seeds = packet_tile_seeds(spec)
            assert [
                (
                    seed["block_sizes"],
                    seed["num_threads"],
                    "load_eviction_policies" in seed,
                )
                for seed in seeds
            ] == [
                ([4, 4, 64], [4, 4, 8], False),
                ([4, 4, 64], [4, 4, 8], True),
                (*sibling, False),
            ]
            for seed in seeds:
                assert seed["loop_orders"] == [order]
                assert seed["cute_vector_widths"] == [1, 1, 8]
                assert seed["cute_fmnmx_extrema"] is True
                normalized = dict(seed)
                spec.normalize(normalized)
                assert all(normalized[key] == value for key, value in seed.items())
        # The last binding's sibling: eight packet threads on thread x and 16
        # positions of an image column on thread y, four packets a thread
        # and a store per lane value.
        code = bound.to_code(helion.Config.from_dict(seeds[-1]))
        assert "block=(8, 16, 1)" in code
        assert code.count("_cute_load_v16b_if(x.iterator") == 4
        assert code.count(").store(cutlass.BFloat16(") == 1
        assert "_cute_store_u16_vec(" not in code

        # Two stores, two layouts: no order to follow.
        spec = cpu_bind(
            _window_max_into_both, (x, channel_first, channels_last)
        ).config_spec
        assert "cute_guarded_lanes" in spec._flat_fields()
        assert packet_tile_seeds(spec) == []
        # One layout again: the seeds are back.
        spec = cpu_bind(
            _window_max_into_both, (x, channel_first, channel_first)
        ).config_spec
        assert [seed["loop_orders"] for seed in packet_tile_seeds(spec)] == [
            [[2, 1, 0]]
        ] * 3
        # A transposed copy: the load reads both tiles themselves.
        rows = (torch.zeros([64, 128], dtype=bf16), torch.zeros([64], dtype=torch.bool))
        copy = helion.kernel(
            _transposed_kept_rows.fn,
            backend="cute",
            static_shapes=True,
            autotune_effort="full",
        )
        for transposed in (copy, _transposed_and_spread_rows):
            spec = cpu_bind(transposed, rows).config_spec
            assert "cute_guarded_lanes" in spec._flat_fields()
            assert packet_tile_seeds(spec) == []


def test_packet_tile_seeds_tell_a_gather_from_a_copy_by_its_loads() -> None:
    """Which layout changes get the packet tile is read from the facts of
    the loads and stores that run along the tiles (``gathers_for_stores``):
    how many there are of each, and whether a load has a coordinate without
    a tile, whichever tile a tap moves along.  Into a channel-first buffer,
    the image row at stride 1:

    - two taps a column apart, the second written as the tile's index plus
      one, which keeps its tile in the load's fact: two loads for one store;
    - every other row, one load whose only computed coordinate is the image
      column's, a tile the stores do not have at stride 1.

    Both are seeded as the 2x2 window is.  No seed for one load of the tiles
    themselves per store: a shifted copy, although a second packet load
    along the channels alone (a floor) sits beside it, and the two taps
    copied to a view each.  The loads are counted against the stores over
    the kernel, so the maximum and the minimum of the two taps, stored to a
    view each, get none either: two loads for two stores."""
    bf16 = torch.bfloat16
    neighbors = helion.kernel(
        neighbor_max_into.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="full",
    )
    shifted = torch.zeros([73, 74, 64], dtype=bf16)

    def channel_first() -> torch.Tensor:
        return torch.zeros([64, 73, 73], dtype=bf16).permute(1, 2, 0)

    with mock_cuda_unavailable():
        for kernel, args in (
            (neighbors, (shifted, channel_first())),
            (
                _every_other_row_into,
                (torch.zeros([146, 73, 64], dtype=bf16), channel_first()),
            ),
        ):
            spec = cpu_bind(kernel, args).config_spec
            assert [
                (seed["block_sizes"], seed["num_threads"], seed["loop_orders"])
                for seed in packet_tile_seeds(spec)
            ] == [
                ([4, 4, 64], [4, 4, 8], [[2, 1, 0]]),
                ([4, 4, 64], [4, 4, 8], [[2, 1, 0]]),
                ([1, 16, 64], [1, 16, 8], [[2, 1, 0]]),
            ]
        # The neighbors again, into a channels-last view: the strict order.
        spec = cpu_bind(neighbors, (shifted, torch.zeros([73, 73, 64], dtype=bf16)))
        assert len(packet_tile_seeds(spec.config_spec)) == 3
        for kernel, args in (
            (
                _shifted_floor_into,
                (shifted, torch.zeros([64], dtype=bf16), channel_first()),
            ),
            (_both_columns_into, (shifted, channel_first(), channel_first())),
            (_neighbor_extrema_into, (shifted, channel_first(), channel_first())),
        ):
            spec = cpu_bind(kernel, args).config_spec
            assert "cute_guarded_lanes" in spec._flat_fields()
            assert packet_tile_seeds(spec) == []


def test_packet_tile_seeds_count_a_store_through_a_computed_coordinate() -> None:
    """A store to every other row of a second output runs along the tiles as
    the loads do, through a computed coordinate, so ``gathers_for_stores``
    takes it for a store like the others: it counts against the loads, and
    its layout has to be the other stores'.

    - Two copies under a row mask, one of them stored that way: two loads
      for two stores and no seed, whichever layout the second output has.
    - A 2x2 window stored to a channel-first buffer and, that way, to a
      second one: seeded as the window alone is where the second buffer is
      channel-first too, and not where it is channels-last, the stores
      having no one layout for the threads to follow.

    A store through a loaded index has no affine block and is not counted:
    the two copies with the second one scattered to the rows an index
    tensor names read as two loads for one store and are seeded.  That is a
    known over-seed of ``gathers_for_stores``, pinned here so that a change
    of what it counts is a deliberate one."""
    bf16 = torch.bfloat16

    def channel_first(rows: int) -> torch.Tensor:
        return torch.zeros([64, rows, 73], dtype=bf16).permute(1, 2, 0)

    def channels_last(rows: int) -> torch.Tensor:
        return torch.zeros([rows, 73, 64], dtype=bf16)

    keep = torch.zeros([73], dtype=torch.bool)
    with mock_cuda_unavailable():
        for spread in (channel_first(146), channels_last(146)):
            copies = (channels_last(73), channels_last(73), keep, channel_first(73))
            spec = cpu_bind(_kept_copies_into, (*copies, spread)).config_spec
            assert "cute_guarded_lanes" in spec._flat_fields()
            assert packet_tile_seeds(spec) == []
        # Known over-seed: the scattered store is not counted, its load is.
        rows = torch.zeros([73], dtype=torch.int64)
        spec = cpu_bind(
            _kept_and_scattered_copies_into,
            (*copies[:3], rows, channel_first(73), channel_first(73)),
        ).config_spec
        assert [seed["loop_orders"] for seed in packet_tile_seeds(spec)] == [
            [[2, 1, 0]]
        ] * 3
        x = torch.zeros([146, 146, 64], dtype=bf16)
        spec = cpu_bind(
            _window_max_and_spread_into, (x, channel_first(73), channel_first(146))
        ).config_spec
        assert [seed["loop_orders"] for seed in packet_tile_seeds(spec)] == [
            [[2, 1, 0]]
        ] * 3
        spec = cpu_bind(
            _window_max_and_spread_into, (x, channel_first(73), channels_last(146))
        ).config_spec
        assert "cute_guarded_lanes" in spec._flat_fields()
        assert packet_tile_seeds(spec) == []


def _witness_carriers(spec: object, key: str) -> list[tuple[object, object]]:
    return [
        (
            witness.carrier.config["block_sizes"],
            witness.carrier.config["cute_vector_widths"],
        )
        for group in spec.compiler_coverage_groups  # type: ignore[attr-defined]
        if group.key == key
        for witness in group.witnesses
    ]


def test_guarded_lanes_witnesses_ride_on_seeds_with_a_partial_tile() -> None:
    """A seed whose tiles all divide their extents masks nothing, so it
    carries no ``cute_guarded_lanes`` witness (it does carry one of
    ``cute_fmnmx_extrema``, which changes its maxima whatever the tiles);
    with 48 channels the seeds' 64-wide lane tile is partial and they do."""
    kernel = helion.kernel(
        window_max.fn, backend="cute", static_shapes=True, autotune_effort="full"
    )

    def spec_of(channels: int) -> object:
        return cpu_bind(
            kernel, (torch.zeros([146, 146, channels], dtype=torch.bfloat16),)
        ).config_spec

    with mock_cuda_unavailable():
        whole = spec_of(64)
        assert _witness_carriers(whole, "cute_guarded_lanes") == []
        assert (
            _witness_carriers(whole, "cute_fmnmx_extrema")
            == [([1, 1, 64], [1, 1, 8])] * 4
        )
        assert (
            _witness_carriers(spec_of(48), "cute_guarded_lanes")
            == [([1, 1, 64], [1, 1, 8])] * 4
        )


def test_a_window_gathers_witnesses_ride_on_its_staged_store_seeds() -> None:
    """A window gather (a ``[n, window]`` output, one tile across the
    window) gets its seeds with a partial tile from the staged-store family,
    so the ``cute_guarded_lanes`` witnesses ride on those: staged store,
    sink and guarded lanes together.  The group then owns the knob, as on
    every kernel with such a seed: the random initial population keeps it
    off, where without a carrier half its rows would draw it, and goes on
    drawing ``cute_staged_store``."""
    kernel = helion.kernel(
        sliding_windows.fn, backend="cute", static_shapes=True, autotune_effort="full"
    )
    with mock_cuda_unavailable():
        bound = cpu_bind(kernel, (torch.zeros([3, 40, 16]), 4, 1))
        spec = bound.config_spec
        (group,) = (
            group
            for group in spec.compiler_coverage_groups
            if group.key == "cute_guarded_lanes"
        )
        assert len(group.witnesses) == 2
        for witness in group.witnesses:
            carrier = witness.carrier.config
            assert witness.value is True
            assert carrier["cute_staged_store"] is True
            assert carrier["cute_vloop_sink"] is True
            assert carrier.get("cute_guarded_lanes") is not True
            # 1920 outputs in tiles that do not divide them.
            assert 1920 % carrier["block_sizes"][0]
        with bound.env:
            generation = spec.create_config_generation()
            random.seed(99)
            rows = generation.initial_population_view().random_population_flat(60)
            configs = [generation.unflatten(row).config for row in rows]
    assert not any(config.get("cute_guarded_lanes") for config in configs)
    assert 20 <= sum(bool(config.get("cute_staged_store")) for config in configs) <= 40


def test_generated_configs_keep_the_packet_threads_on_thread_x() -> None:
    """Under the knob a config the search generates moves the packet block
    ahead of the first threaded block of its loop order; a pinned config,
    and one without the knob, keeps its order."""
    with mock_cuda_unavailable():
        spec = cpu_bind(
            window_max, (torch.zeros([146, 146, 64], dtype=torch.bfloat16),)
        ).config_spec

        def normalized(fix_invalid: bool, **values: object) -> dict[str, object]:
            config = spec.default_config().config | {
                "block_sizes": [4, 4, 64],
                "num_threads": [4, 4, 8],
                "cute_vector_widths": [1, 1, 8],
                **values,
            }
            spec.normalize(config, _fix_invalid=fix_invalid)
            return config

        for order, expected in (
            ([0, 1, 2], [2, 0, 1]),
            ([1, 0, 2], [2, 1, 0]),
            ([1, 2, 0], [2, 1, 0]),
            ([2, 0, 1], [2, 0, 1]),
        ):
            config = normalized(True, loop_orders=[order], cute_guarded_lanes=True)
            assert config["loop_orders"] == [expected]
            assert normalized(True, loop_orders=[order])["loop_orders"] == [order]
            pinned = normalized(False, loop_orders=[order], cute_guarded_lanes=True)
            assert pinned["loop_orders"] == [order]
        # One thread on H: W is the first threaded block the packets pass.
        config = normalized(
            True,
            block_sizes=[1, 16, 64],
            num_threads=[1, 16, 8],
            loop_orders=[[0, 1, 2]],
            cute_guarded_lanes=True,
        )
        assert config["loop_orders"] == [[0, 2, 1]]
        # Scalar lanes have no packet to keep together.
        config = normalized(
            True,
            cute_vector_widths=[1, 1, 1],
            loop_orders=[[0, 1, 2]],
            cute_guarded_lanes=True,
        )
        assert config["loop_orders"] == [[0, 1, 2]]
        # A window along the contiguous dim gathers scalars there: not the
        # packet tile the rule was measured on, so every loop order stays
        # in the search under the knob.
        spec = cpu_bind(
            _plane_window_max, (torch.zeros([64, 56, 128], dtype=torch.bfloat16),)
        ).config_spec
        for order in ([0, 1, 2], [1, 0, 2], [2, 1, 0]):
            config = normalized(True, loop_orders=[order], cute_guarded_lanes=True)
            assert config["loop_orders"] == [order]
            assert config["cute_vector_widths"] == [1, 1, 8]
        # Packets along the columns of the source, but a transposed store:
        # no storage order for the CTAs to walk, so the rows may lead.
        spec = cpu_bind(
            _transposed_kept_rows,
            (
                torch.zeros([64, 128], dtype=torch.bfloat16),
                torch.zeros([64], dtype=torch.bool),
            ),
        ).config_spec
        config = normalized(
            True,
            block_sizes=[4, 64],
            num_threads=[4, 8],
            cute_vector_widths=[1, 8],
            loop_orders=[[0, 1]],
            cute_guarded_lanes=True,
        )
        assert config["loop_orders"] == [[0, 1]]
        assert config["cute_vector_widths"] == [1, 8]
        # A window that changes the layout: its seeds put the packet threads
        # on thread x, and the search keeps every order.
        spec = cpu_bind(
            window_max_into,
            (
                torch.zeros([146, 146, 64], dtype=torch.bfloat16),
                torch.zeros([64, 73, 73], dtype=torch.bfloat16).permute(1, 2, 0),
            ),
        ).config_spec
        for order in ([0, 1, 2], [1, 0, 2], [2, 1, 0]):
            config = normalized(True, loop_orders=[order], cute_guarded_lanes=True)
            assert config["loop_orders"] == [order]
            assert config["cute_vector_widths"] == [1, 1, 8]


def test_generated_configs_keep_an_order_the_move_could_not_launch() -> None:
    """The move shifts the threaded blocks ahead of the packet block one
    thread axis up.  With 128 threads on W that puts W on thread z, which
    holds 64: the config keeps its loop order, which launches."""
    with mock_cuda_unavailable():
        bound = cpu_bind(
            window_max, (torch.zeros([146, 146, 64], dtype=torch.bfloat16),)
        )
        spec = bound.config_spec

        def normalized(order: list[int]) -> dict[str, object]:
            config = spec.default_config().config | {
                "block_sizes": [2, 128, 32],
                "num_threads": [2, 128, 4],
                "cute_vector_widths": [1, 1, 8],
                "loop_orders": [order],
                "cute_guarded_lanes": True,
            }
            spec.normalize(config, _fix_invalid=True)
            return config

        # H, W, C as threads (2, 128, 4): C first would launch (4, 2, 128).
        kept = normalized([0, 1, 2])
        assert kept["loop_orders"] == [[0, 1, 2]]
        code = bound.to_code(helion.Config.from_dict(kept))
        assert "block=(2, 128, 4)" in code
        # W, H, C: C first leaves W on thread y, so the packet block moves.
        assert normalized([1, 0, 2])["loop_orders"] == [[2, 1, 0]]
