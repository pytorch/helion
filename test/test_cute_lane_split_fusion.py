"""The lane split gives each sweep its own statements, so later sweeps fuse.

``split_lane_loop_reductions`` turns a persistent reduction over synthetic
lanes into an accumulate loop and a consume loop.  Both re-run the reduction
input's producers, the consume loop on statement objects of its own: the
device-body passes key facts by node identity, and ``fuse_two_pass_loads``
orders the consume loop's copy of the input load after its producer and
serves it from the accumulate loop's register fragment instead of reloading.
With its own statements, the consume loop's copy of a
persistent-branch packet load is reused from the accumulate loop's hoisted
packet only while nothing between them may write it or rebind a name of its
address (an alias the final rename merges into it included), and the exact
division still takes a dividend the consume loop now reads from an fp32 load
cache.  A write the vectorizer cannot resolve, or a call whose memory effects
are unknown (impure inline asm), may write any packet: between the loops,
and inside a lane loop, whose loads and stores it then keeps per lane.  The
kernels render on CPU.
"""

from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute import fuse_two_pass_loads as fuse_module
from helion._compiler.cute.hoist_loop_invariant_recip import _picks_one_element
from helion._compiler.cute.hoist_loop_invariant_recip import (
    hoist_exact_invariant_divisions,
)
from helion._compiler.cute.persistent_branch_vec import (
    vectorize_branch_local_persistent_fragments,
)
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

pytestmark = pytest.mark.cpu_only


def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    rows, n = x.shape
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        row = x[tile_r, :].to(torch.float32)
        mean_square = torch.mean(row * row, dim=-1, keepdim=True)
        normalized = row * torch.rsqrt(mean_square + 1e-6) * weight[None, :]
        out[tile_r, :] = normalized.to(out.dtype)
    return out


def _repeated_statements(body: list[ast.stmt]) -> list[str]:
    seen: set[int] = set()
    repeated: list[str] = []
    for statement in (
        node for root in body for node in ast.walk(root) if isinstance(node, ast.stmt)
    ):
        if id(statement) in seen:
            repeated.append(ast.unparse(statement))
        seen.add(id(statement))
    return repeated


def test_rms_norm_store_sweep_reads_the_fragment() -> None:
    # Statements found at two positions of each body the load fusion gets
    # (checked on entry: a later pass dealiases the same loops in place).
    repeated: list[list[str]] = []
    original = fuse_module.fuse_two_pass_loads

    def record(body: list[ast.stmt], *args: object, **kwargs: object) -> object:
        repeated.append(_repeated_statements(body))
        return original(body, *args, **kwargs)

    with (
        cpu_render_target(reduction_smem_budget_bytes=232448),
        patch.object(fuse_module, "fuse_two_pass_loads", record),
    ):
        kernel = helion.kernel(rms_norm, backend="cute", static_shapes=True)
        bound = cpu_bind(
            kernel,
            (
                torch.empty(64, 256, dtype=torch.bfloat16),
                torch.empty(256, dtype=torch.bfloat16),
            ),
        )
        config = helion.Config(
            block_sizes=[4], num_threads=[4, 32], cute_reduction_reloads=["register"]
        )
        code = bound.to_code(bound.config_spec.normalized_config(config))
    # Every position of the body the device-body passes see is its own node.
    assert repeated
    assert all(statements == [] for statements in repeated)
    device = next(
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef) and node.name == "_helion_rms_norm"
    )
    sweeps = [ast.unparse(node) for node in device.body if isinstance(node, ast.For)]
    # The sum-of-squares sweep fills the fragment; the normalize sweep reads
    # it instead of loading x again.
    assert len(sweeps) == 2
    assert "x.iterator" in sweeps[0]
    assert "_fuse_cache_0[" in sweeps[0]
    assert "x.iterator" not in sweeps[1]
    assert "= _fuse_cache_0[" in sweeps[1]


PACKET_LOAD = (
    "_helion_persistent_branch_vec_load(7, 8, 'cutlass.BFloat16', '', "
    "x.iterator + lane_index, (x.iterator + lane_index).load())"
)


def _inline_asm(pure: bool) -> str:
    return (
        "_cute_inline_asm_elementwise((address, total), "
        "asm='atom.global.add.f32 $0, [$1], $2;', constraints='=f,l,f', "
        f"dtype=cutlass.Float32, is_pure={pure})"
    )


def _vectorize(source: str, rename_groups: dict[str, str] | None = None) -> str:
    result = vectorize_branch_local_persistent_fragments(
        ast.parse(source).body,
        proven_disjoint_tensor_pairs={frozenset(("x", "y"))},
        rename_groups=rename_groups,
    )
    return ast.unparse(ast.Module(body=result, type_ignores=[]))


# One axis, the statement between the loops; each case is one kind of
# statement the reuse has to see.
@pytest.mark.parametrize(
    ("between", "loads"),
    [
        ("", 1),
        # A store to a tensor proven disjoint from the packet's keeps it, as
        # do inline asm declared pure (lowered without side effects) and a
        # cluster reduction, whose scratch is shared memory.
        ("(y.iterator + 3).store(total)", 1),
        (_inline_asm(pure=True), 1),
        (
            (
                "total = _cute_grouped_reduce_cluster("
                "total, 'sum', cutlass.Float32(0), 0, buffer, barrier)"
            ),
            1,
        ),
        # A write the packet's tensor may alias, a store no recognizer
        # resolves, a call whose effects are unknown (impure asm, the wait for
        # the previous grid's writes) or a rebound address name ends the
        # reuse: the consume loop loads its own packet.
        ("(x.iterator + 3).store(total)", 2),
        ("cute.copy(atom, source, target)", 2),
        (_inline_asm(pure=False), 2),
        ("cute.arch.griddepcontrol_wait()", 2),
        ("lane_base = lane_base + 8", 2),
        # ``lane_base_1`` is ``lane_base`` once the final rename runs.
        ("lane_base_1 = lane_base + 8", 2),
    ],
)
def test_consume_loop_reuses_the_accumulate_loops_packet_while_it_holds(
    between: str, loads: int
) -> None:
    code = _vectorize(
        f"""
for synthetic_lane_7 in cutlass.range_constexpr(8):
    lane_index = lane_base + cutlass.Int32(synthetic_lane_7)
    value = {PACKET_LOAD}
    total = total + value
{between}
for synthetic_lane_7 in cutlass.range_constexpr(8):
    lane_index = lane_base + cutlass.Int32(synthetic_lane_7)
    again = {PACKET_LOAD}
    (y.iterator + lane_index).store(again)
""",
        rename_groups={"lane_base_1": "lane_base"},
    )

    assert code.count("cute.arch.load(x.iterator") == loads
    assert "_helion_persistent_branch_vec_" not in code


ELEMENT_PACKET_LOAD = (
    "_helion_persistent_branch_vec_load(7, 8, 'cutlass.BFloat16', '', "
    "x.iterator + offs[0] + lane_index, (x.iterator + offs[0] + lane_index).load())"
)
FRAGMENT_REDUCE_INTO_OFFS = (
    "_cute_grouped_reduce_shared_two_stage_fragment(values, offs, 'sum', "
    "cutlass.Float32(0), lane, lane_in_group, lane_mod_pre, count=1, pre=1, "
    "group_span=8, group_count=1)"
)


@pytest.mark.parametrize(
    ("between", "loads"),
    [
        ("", 1),
        # ``alias`` holds the fragment ``offs``: the store into its element
        # changes the ``offs[0]`` the packet's address reads.
        ("alias = offs\nalias[0] = 8", 2),
        # The fragment grouped reduce stores its results into ``offs``.
        (FRAGMENT_REDUCE_INTO_OFFS, 2),
    ],
    ids=["nothing", "alias", "fragment-reduce"],
)
def test_a_store_into_an_element_ends_a_packet_whose_address_reads_one(
    between: str, loads: int
) -> None:
    code = _vectorize(
        f"""
for synthetic_lane_7 in cutlass.range_constexpr(8):
    lane_index = lane_base + cutlass.Int32(synthetic_lane_7)
    value = {ELEMENT_PACKET_LOAD}
    total = total + value
{between}
for synthetic_lane_7 in cutlass.range_constexpr(8):
    lane_index = lane_base + cutlass.Int32(synthetic_lane_7)
    again = {ELEMENT_PACKET_LOAD}
    (y.iterator + lane_index).store(again)
"""
    )

    assert code.count("cute.arch.load(x.iterator") == loads


@pytest.mark.parametrize(
    ("between", "fuses"),
    [
        ("", True),
        # A store into the fragment's element, through another name or by the
        # fragment grouped reduce, changes the ``offs[0]`` both sweeps' loads
        # read.
        ("alias = offs\nalias[0] = 8", False),
        (FRAGMENT_REDUCE_INTO_OFFS, False),
    ],
    ids=["nothing", "alias", "fragment-reduce"],
)
def test_a_store_into_an_element_ends_a_fused_load_whose_address_reads_one(
    between: str, fuses: bool
) -> None:
    source = f"""
for synthetic_lane_7 in range(8):
    value = (x.iterator + offs[0] + synthetic_lane_7).load()
    total = total + value
{between}
for synthetic_lane_7 in range(8):
    again = (x.iterator + offs[0] + synthetic_lane_7).load()
    (y.iterator + synthetic_lane_7).store(again)
"""
    result = fuse_module.fuse_two_pass_loads(
        ast.parse(source).body,
        tensor_dtypes={"x": "cutlass.Float32", "y": "cutlass.Float32"},
        reload_modes={7: "register"},
        proven_disjoint_tensor_pairs={frozenset(("x", "y"))},
    )
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))
    assert ("_fuse_cache_" in code) is fuses


def test_two_loads_of_one_packet_in_a_loop_share_its_load() -> None:
    code = _vectorize(
        f"""
for synthetic_lane_7 in cutlass.range_constexpr(8):
    lane_index = lane_base + cutlass.Int32(synthetic_lane_7)
    value = {PACKET_LOAD}
    again = {PACKET_LOAD}
    total = total + value + again
"""
    )

    assert code.count("cute.arch.load(x.iterator") == 1
    assert "_helion_persistent_branch_vec_" not in code


@pytest.mark.parametrize(
    ("write", "hoisted"),
    [
        # A store to a tensor proven disjoint from the packet's.
        ("(y.iterator + lane_index).store(total)", True),
        # Before the load, a write that may reach the packet's tensor runs
        # ahead of the next lane's load, so the loop keeps its scalar loads:
        # a resolved store, a vector reduction into memory, a bulk copy, and
        # the calls whose effects are unknown.
        ("(x.iterator + 3).store(total)", False),
        ("_cute_red_add_bf16_vec(x.iterator + lane_index, values)", False),
        ("cute.copy(atom, source, target)", False),
        (_inline_asm(pure=False), False),
        ("cute.arch.griddepcontrol_wait()", False),
    ],
)
def test_a_write_in_the_lane_loop_keeps_the_packet_load_in_it(
    write: str, hoisted: bool
) -> None:
    code = _vectorize(
        f"""
for synthetic_lane_7 in cutlass.range_constexpr(8):
    lane_index = lane_base + cutlass.Int32(synthetic_lane_7)
    {write}
    value = {PACKET_LOAD}
    total = total + value
"""
    )

    assert ("cute.arch.load(x.iterator" in code) is hoisted
    assert ("(x.iterator + lane_index).load()" in code) is not hoisted


@pytest.mark.parametrize(
    ("write", "load", "hoisted"),
    [
        # A store into an element changes the ``offs[0]`` the next lane's
        # address reads, directly or through ``alias``, which holds ``offs``.
        ("offs[0] = offs[0] + 1", ELEMENT_PACKET_LOAD, False),
        ("alias[0] = alias[0] + 1", ELEMENT_PACKET_LOAD, False),
        (FRAGMENT_REDUCE_INTO_OFFS, ELEMENT_PACKET_LOAD, False),
        # An address that reads no element keeps its packet.
        ("alias[0] = alias[0] + 1", PACKET_LOAD, True),
    ],
    ids=["direct", "alias", "fragment-reduce", "no-element"],
)
def test_a_store_into_an_element_in_the_lane_loop_keeps_an_element_address_load(
    write: str, load: str, hoisted: bool
) -> None:
    code = _vectorize(
        f"""
alias = offs
for synthetic_lane_7 in cutlass.range_constexpr(8):
    lane_index = lane_base + cutlass.Int32(synthetic_lane_7)
    value = {load}
    total = total + value
    {write}
"""
    )

    assert ("cute.arch.load(x.iterator" in code) is hoisted


@pytest.mark.parametrize(
    ("address", "hoisted"),
    [
        # Every iteration of the inner loop loads the lane's element.
        ("x.iterator + lane_index", True),
        # The inner loop's index moves the address and is unbound above the
        # lane loop.
        ("x.iterator + inner * 8 + lane_index", False),
    ],
    ids=["invariant", "inner-index"],
)
def test_a_load_reading_an_inner_loops_index_stays_in_the_lane_loop(
    address: str, hoisted: bool
) -> None:
    load = (
        "_helion_persistent_branch_vec_load(7, 8, 'cutlass.BFloat16', '', "
        f"{address}, ({address}).load())"
    )
    code = _vectorize(
        f"""
for synthetic_lane_7 in cutlass.range_constexpr(8):
    lane_index = lane_base + cutlass.Int32(synthetic_lane_7)
    for inner in cutlass.range_constexpr(2):
        value = {load}
        total = total + value
"""
    )

    assert ("cute.arch.load(x.iterator" in code) is hoisted
    assert (f"({address}).load()" in code) is not hoisted


@pytest.mark.parametrize(
    ("write", "sunk"),
    [
        ("", True),
        # A reduction into a tensor proven disjoint from the stored one.
        ("_cute_red_add_bf16_vec(x.iterator + 3, values)", True),
        # Writes that may reach the stored tensor: a reduction into it, a
        # bulk copy, impure asm.  The stores stay in the loop.
        ("_cute_red_add_bf16_vec(y.iterator + 3, values)", False),
        ("cute.copy(atom, source, target)", False),
        (_inline_asm(pure=False), False),
    ],
)
def test_a_write_in_the_lane_loop_keeps_the_packet_store_in_it(
    write: str, sunk: bool
) -> None:
    code = _vectorize(
        f"""
for synthetic_lane_7 in cutlass.range_constexpr(8):
    lane_index = lane_base + cutlass.Int32(synthetic_lane_7)
    {write}
    _helion_persistent_branch_vec_store(7, 8, 'cutlass.BFloat16', y.iterator + lane_index, total, None)
"""
    )

    assert ("_cute_store_u16_vec(y.iterator" in code) is sunk
    assert ("(y.iterator + lane_index).store(total)" in code) is not sunk


@pytest.mark.parametrize(
    ("rebound", "renamed"),
    [
        # The loop advances a name of the packet's address, or of the mask
        # deciding it, under the name the final rename gives it: the packet
        # differs per lane, so the loads stay scalar.  Without the rename
        # groups the names look unrelated and the loop is vectorized.
        ("lane_base_1 = lane_base + 8", True),
        ("lane_base_1 = lane_base + 8", False),
        ("n_1 = n - 8", True),
        ("n_1 = n - 8", False),
    ],
)
def test_a_name_the_lane_loop_rebinds_under_its_final_name_keeps_the_loads(
    rebound: str, renamed: bool
) -> None:
    masked = (
        "_helion_persistent_branch_vec_load(7, 8, 'cutlass.BFloat16', '', "
        "x.iterator + lane_index, (x.iterator + lane_index).load() "
        "if lane_index < n else cutlass.BFloat16(0))"
    )
    code = _vectorize(
        f"""
for synthetic_lane_7 in cutlass.range_constexpr(8):
    lane_index = lane_base + cutlass.Int32(synthetic_lane_7)
    value = {masked}
    total = total + value
    {rebound}
""",
        rename_groups={"lane_base_1": "lane_base", "n_1": "n"} if renamed else None,
    )

    assert ("cute.arch.load(x.iterator" in code) is not renamed


def test_a_fragment_view_anywhere_in_the_index_picks_no_single_element() -> None:
    def index(source: str) -> ast.expr:
        node = ast.parse(f"fragment[{source}]", mode="eval").body
        assert isinstance(node, ast.Subscript)
        return node.slice

    assert _picks_one_element(index("lane"))
    assert _picks_one_element(index("lane, 1"))
    # A slice, a ``None`` mode or an ellipsis, in a nested coordinate too.
    for source in (":", "None", "(None, 0), 1", "lane, (0, ...)", "lane, 1:"):
        assert not _picks_one_element(index(source))


@pytest.mark.parametrize(
    "alias", ["_fuse_cache_0", "_fuse_cache_0[None]"], ids=["name", "view"]
)
def test_exact_division_keeps_the_plain_form_for_a_cache_written_through_an_alias(
    alias: str,
) -> None:
    # ``alias`` holds ``_fuse_cache_0`` or a view of it: the masked zero of a
    # ragged row it stores reaches the cache's elements, which are then no
    # fp32 loads.  The divisor is fp32, so the dividend alone decides.
    body = ast.parse(
        f"""
_fuse_cache_0 = cute.make_rmem_tensor(4, cutlass.Float32)
alias = {alias}
total = cutlass.Float32(0)
for lane in range(4):
    value = (x.iterator + lane).load() if lane < n else cutlass.Float32(0)
    alias[lane] = value
    total = total + cutlass.Float32(1)
for lane in range(4):
    cached = _fuse_cache_0[lane]
    v_0 = cached / total
    (out.iterator + lane).store(cutlass.Float32(v_0))
"""
    ).body
    result = hoist_exact_invariant_divisions(
        body,
        tensor_dtypes={"x": "cutlass.Float32", "out": "cutlass.Float32"},
        argument_names=("x", "out", "n"),
        proven_disjoint_tensor_pairs=(frozenset(("x", "out")),),
    )
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))

    assert "_cute_exact_div_redo_f32(" not in code


@pytest.mark.parametrize(
    ("masked", "index", "exact"),
    [
        (False, "lane", True),
        # A cache the masked zero of a ragged row reaches stays a select's
        # value, which keeps the plain form.
        (True, "lane", False),
        # A view of the fragment (a slice) is no fp32 scalar.
        (False, ":", False),
    ],
)
def test_exact_division_takes_an_fp32_load_cache_read_as_its_dividend(
    masked: bool, index: str, exact: bool
) -> None:
    load = "(x.iterator + lane).load()" + (
        " if lane < n else cutlass.Float32(0)" if masked else ""
    )
    body = ast.parse(
        f"""
_fuse_cache_0 = cute.make_rmem_tensor(4, cutlass.Float32)
total = cutlass.Float32(0)
for lane in range(4):
    value = {load}
    _fuse_cache_0[lane] = value
    total = total + cutlass.Float32(1)
for lane in range(4):
    cached = _fuse_cache_0[{index}]
    v_0 = cached / total
    (out.iterator + lane).store(cutlass.Float32(v_0))
"""
    ).body
    result = hoist_exact_invariant_divisions(
        body,
        tensor_dtypes={"x": "cutlass.Float32", "out": "cutlass.Float32"},
        argument_names=("x", "out", "n"),
        proven_disjoint_tensor_pairs=(frozenset(("x", "out")),),
    )
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))

    # An fp32 fragment holds fp32 elements; the divisor is fp32, so the
    # dividend alone decides.
    assert ("_cute_exact_div_fast_f32(cached, total" in code) is exact
    assert ("if _cute_exact_div_redo_f32(" in code) is exact
