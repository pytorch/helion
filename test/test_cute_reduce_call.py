"""Pins for ``helion._compiler.cute.reduce_call``.

The registry names exactly the grouped-reduce dispatchers of
``reduce_helpers`` that generated code imports, the role sets the passes
consume match the tables they used to spell out name for name, and
``ReduceCall`` renders every emitter form back byte for byte.  CPU only: the
checks read ``reduce_helpers.py`` as text, so they never import the DSL.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from helion._compiler.cute import reduce_call
from helion._compiler.cute.backend import CuteBackend
from helion._compiler.cute.reduce_call import CLUSTER_REDUCES
from helion._compiler.cute.reduce_call import COLLECTIVE_REDUCES
from helion._compiler.cute.reduce_call import CTA_REDUCES
from helion._compiler.cute.reduce_call import FP32_REDUX_REDUCES
from helion._compiler.cute.reduce_call import GROUPED_REDUCE_HELPERS
from helion._compiler.cute.reduce_call import GROUPED_SHARED_REDUCES
from helion._compiler.cute.reduce_call import LANE_FOLD_REDUCES
from helion._compiler.cute.reduce_call import ONLINE_PAIR_REDUCES
from helion._compiler.cute.reduce_call import ReduceCall
from helion._compiler.cute.reduce_call import ReduceScope
from helion._compiler.cute.reduce_call import runtime_lane_expr

# AST and table pins (no render, no device): they run on every backend row.

WARP = "_cute_grouped_reduce_warp"
TWO_STAGE = "_cute_grouped_reduce_shared_two_stage"
TREE = "_cute_grouped_reduce_shared_tree"
CLUSTER = "_cute_grouped_reduce_cluster"
BLOCK = "_cute_grouped_reduce_block"
COLUMNS = "_cute_grouped_reduce_shared_columns"
FRAGMENT = "_cute_grouped_reduce_shared_two_stage_fragment"
PAIR = "_cute_grouped_reduce_cluster_online_pair"
WARP_PAIR = "_cute_grouped_reduce_cluster_warp_online_pair"


def _dispatchers() -> dict[str, ast.FunctionDef]:
    """The undecorated module-level ``_cute_grouped_reduce_*`` functions of
    ``reduce_helpers`` (the ``@cute.jit`` bodies are not dispatchers)."""
    source = Path(reduce_call.__file__).with_name("reduce_helpers.py").read_text()
    return {
        node.name: node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and node.name.startswith("_cute_grouped_reduce_")
        and not node.decorator_list
    }


def test_registry_names_the_emitted_dispatchers() -> None:
    dispatchers = _dispatchers()
    assert set(dispatchers) >= GROUPED_REDUCE_HELPERS
    # The serial and interleaved forms are selected inside the two-stage
    # dispatcher by its group shape; no emitter names them.
    assert set(dispatchers) - GROUPED_REDUCE_HELPERS == {
        "_cute_grouped_reduce_shared_serial",
        "_cute_grouped_reduce_shared_interleaved",
    }
    assert {scope.helper for scope in ReduceScope} == GROUPED_REDUCE_HELPERS


def test_registry_matches_the_generated_code_imports() -> None:
    imported = {
        name
        for name, statement in CuteBackend().library_imports.items()
        if "reduce_helpers import _cute_grouped_reduce" in statement
    }
    assert imported == GROUPED_REDUCE_HELPERS


def test_fp32_redux_set_is_the_dispatchers_that_accept_the_keyword() -> None:
    accepting = {
        name
        for name, function in _dispatchers().items()
        if any(arg.arg == "fp32_redux" for arg in function.args.kwonlyargs)
    }
    # ``_shared_serial`` accepts it too but is never emitted (see above).
    assert accepting - {"_cute_grouped_reduce_shared_serial"} == FP32_REDUX_REDUCES


def test_role_sets_pin_the_tables_the_passes_used_to_spell_out() -> None:
    # Each role set replaces a per-pass helper-name table that spelled out the
    # same reduce kinds; the comments name the tables as the passes had them
    # before they were deleted in favour of these sets.
    # hoist_warp_reduce / fold_lane_reductions: the grouped reduce funcs and
    # the reduce helpers.
    assert {TWO_STAGE, TREE, WARP, CLUSTER} == LANE_FOLD_REDUCES
    # _ast_pass_utils' chunk-reduction helpers, online_chunk_reduction,
    # duplicate_tile_maxima's max helpers and bounded_loop_cache's reductions
    # (the latter two minus the never-emitted ``_shared_serial`` entry).
    assert {TREE, TWO_STAGE, WARP} == CTA_REDUCES
    # native_warp_reduction's helper table minus ``_shared_serial``.
    assert {WARP, TWO_STAGE, CLUSTER, BLOCK, PAIR, WARP_PAIR} == FP32_REDUX_REDUCES
    # scalar_epilogue._COLLECTIVES.
    assert {CLUSTER, PAIR, WARP_PAIR, TWO_STAGE, TREE, WARP} == COLLECTIVE_REDUCES
    # hoist_warp_reduce.validate_cluster_reduce_placement's trio.
    assert {CLUSTER, PAIR, WARP_PAIR} == CLUSTER_REDUCES
    assert {PAIR, WARP_PAIR} == ONLINE_PAIR_REDUCES
    # finalize_reduce_groups._GROUPED_SHARED_REDUCES.
    assert {TWO_STAGE, COLUMNS} == GROUPED_SHARED_REDUCES
    assert FRAGMENT in GROUPED_REDUCE_HELPERS


# The exact texts the emitters write (reduction_emitter, matmul_fallback,
# lane_loops.markers, integer_loop_reduction, cluster_online_pair and the
# ``fp32_redux`` keyword native_warp_reduction appends).
_FORMS = [
    f"{WARP}(v, 'sum', cutlass.Float32(0), lane, pre=2, group_span=16)",
    (
        f"{TWO_STAGE}(v, 'max', cutlass.Float32(float('-inf')), a, b, c, "
        "pre=1, group_span=128, group_count=8)"
    ),
    (
        f"{TWO_STAGE}(0, 'sum', cutlass.Int64(0), cutlass.Int32(cute.arch.thread_idx()[0]), "
        "cutlass.Int32(cute.arch.thread_idx()[0]) % 128, 0, "
        "pre=1, group_span=128, group_count=1)"
    ),
    (
        f"{TREE}(v, 'prod', cutlass.Float32(1.0), a, b, c, "
        "pre=3, group_span=48, num_threads=48, group_count=1)"
    ),
    (
        f"{CLUSTER}(v, 'sum', cutlass.Float32(0.0), lane, buf, mbar, "
        "group_span=256, cluster_n=4)"
    ),
    f"{BLOCK}(v, 'max', cutlass.Float32(float('-inf')), lane, group_span=256)",
    (
        f"{WARP}(v, 'max', cutlass.Float32(float('-inf')), lane, "
        "pre=1, group_span=32, fp32_redux=True)"
    ),
]


@pytest.mark.parametrize("text", _FORMS)
def test_parse_render_round_trips_every_emitter_form(text: str) -> None:
    node = ast.parse(text, mode="eval").body
    reduce = ReduceCall.parse(node)
    assert reduce is not None
    assert reduce.render() == text
    assert ast.unparse(node) == text
    assert reduce.helper == text.split("(")[0]


def test_render_spells_the_emitter_forms() -> None:
    warp = ReduceCall(
        scope=ReduceScope.WARP,
        op="sum",
        input="acc",
        identity="cutlass.Float32(0)",
        lane_expr="cutlass.Int32(cute.arch.thread_idx()[0])",
        pre=4,
        group_span=32,
    )
    assert warp.render() == (
        f"{WARP}(acc, 'sum', cutlass.Float32(0), "
        "cutlass.Int32(cute.arch.thread_idx()[0]), pre=4, group_span=32)"
    )
    two_stage = ReduceCall(
        scope=ReduceScope.SHARED_TWO_STAGE,
        op="max",
        input="part",
        identity="cutlass.Float32(float('-inf'))",
        lane_expr="lane",
        lane_in_group="lane_in_group",
        lane_mod_pre="lane_mod_pre",
        pre=1,
        group_span=128,
        group_count=8,
    )
    assert two_stage.render() == (
        f"{TWO_STAGE}(part, 'max', cutlass.Float32(float('-inf')), "
        "lane, lane_in_group, lane_mod_pre, pre=1, group_span=128, group_count=8)"
    )
    cluster = ReduceCall(
        scope=ReduceScope.CLUSTER,
        op="sum",
        input="part",
        identity="cutlass.Float32(0.0)",
        lane_expr="lane",
        buffers=("buf", "mbar"),
        group_span=256,
        cluster_n=2,
    )
    assert cluster.render() == (
        f"{CLUSTER}(part, 'sum', cutlass.Float32(0.0), lane, buf, mbar, "
        "group_span=256, cluster_n=2)"
    )


@pytest.mark.parametrize(
    "text",
    [
        # Not of the scalar family.
        (
            f"{COLUMNS}([a], 'sum', cutlass.Float32(0), l, g, pre=1, group_span=64, "
            "group_count=1)"
        ),
        (
            f"{PAIR}(s, m, lane, buf, mbar, group_span=128, cluster_n=2, scale=1.0, "
            "fastmath=False)"
        ),
        # The op is not a literal.
        f"{WARP}(v, op, cutlass.Float32(0), lane, pre=2, group_span=16)",
        # Positional count / keyword set of another dispatcher.
        f"{WARP}(v, 'sum', cutlass.Float32(0), pre=2, group_span=16)",
        f"{TWO_STAGE}(v, 'sum', cutlass.Float32(0), a, b, c, pre=1, group_span=64)",
        # A keyword that is not an integer literal.
        f"{WARP}(v, 'sum', cutlass.Float32(0), lane, pre=2, group_span=span)",
        "cute.arch.warp_reduction_sum(v, threads_in_group=32)",
    ],
)
def test_parse_declines_other_shapes(text: str) -> None:
    assert ReduceCall.parse(ast.parse(text, mode="eval").body) is None


def test_runtime_lane_expr_is_the_flattened_thread_id() -> None:
    assert runtime_lane_expr("cutlass.Int32") == (
        "cutlass.Int32(cute.arch.thread_idx()[0])"
        " + (cutlass.Int32(cute.arch.thread_idx()[1]))"
        " * (cutlass.Int32(cute.arch.block_dim()[0]))"
        " + (cutlass.Int32(cute.arch.thread_idx()[2]))"
        " * (cutlass.Int32(cute.arch.block_dim()[0]))"
        " * (cutlass.Int32(cute.arch.block_dim()[1]))"
    )
