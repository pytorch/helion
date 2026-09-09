"""Static safety tests for profitable grid-lane invariant reduction hoisting."""

from __future__ import annotations

import ast
import textwrap
from typing import cast

from helion._compiler.ast_extension import ExtendedAST
from helion._compiler.ast_extension import convert
from helion._compiler.ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from helion._compiler.ast_read_writes import ast_rename
from helion._compiler.cute.hoist_lane_invariant_reductions import (
    hoist_lane_invariant_reductions,
)

_TENSORS = {"bias", "gate", "head", "indices", "key", "out", "state"}
_TENSOR_DTYPES = {
    "bias": "cutlass.Float32",
    "gate": "cutlass.BFloat16",
    "head": "cutlass.Float32",
    "indices": "cutlass.Int32",
    "key": "cutlass.BFloat16",
    "out": "cutlass.Float32",
    "state": "cutlass.BFloat16",
}
_DISJOINT = {
    frozenset((left, right)) for left in _TENSORS for right in _TENSORS if left != right
}


def _lane_loop(source: str) -> list[ast.stmt]:
    body = ast.parse(source).body
    assert len(body) == 1 and isinstance(body[0], ast.For)
    setattr(body[0], HELION_LANE_LOOP_VAR_ATTR, "row_lane")
    return body


def _rewrite(
    source: str,
    *,
    disjoint: set[frozenset[str]] = _DISJOINT,
    rename_groups: dict[str, str] | None = None,
    thread_block_dims: tuple[int, int, int] | None = None,
) -> list[ast.stmt]:
    return hoist_lane_invariant_reductions(
        _lane_loop(source),
        tensor_names=_TENSORS,
        tensor_dtypes=_TENSOR_DTYPES,
        proven_disjoint_tensor_pairs=disjoint,
        rename_groups=rename_groups,
        thread_block_dims=thread_block_dims,
        uniform_names={
            "base",
            "batch",
            "cond",
            "external",
            "other",
            "row_base",
            "seed",
            "seed_1",
            "seed_2",
            "vector_type",
        },
    )


def _rewrite_body(
    source: str,
    *,
    disjoint: set[frozenset[str]] = _DISJOINT,
    rename_groups: dict[str, str] | None = None,
    tensor_dtypes: dict[str, str] = _TENSOR_DTYPES,
    extended: bool = False,
) -> list[ast.stmt]:
    parsed = ast.parse(source)
    module = cast("ast.Module", convert(parsed)) if extended else parsed
    body = module.body
    lane_loops = [
        node
        for stmt in body
        for node in ast.walk(stmt)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "row_lane"
    ]
    assert len(lane_loops) == 1
    setattr(lane_loops[0], HELION_LANE_LOOP_VAR_ATTR, "row_lane")
    return hoist_lane_invariant_reductions(
        body,
        tensor_names=set(tensor_dtypes),
        tensor_dtypes=tensor_dtypes,
        proven_disjoint_tensor_pairs=disjoint,
        rename_groups=rename_groups,
        uniform_names={"base", "scale"},
        float_scalar_names={"scale"},
        thread_block_dims=(16, 8, 1),
    )


def _source(body: list[ast.stmt]) -> str:
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


_POSITIVE = """
for row_lane in range(16):
    row = row_base + row_lane
    state_index_load = (indices.iterator + batch).load()
    state_index = cutlass.Int32(state_index_load)
    invalid = operator.lt(state_index, 0)
    if invalid:
        (out.iterator + row).store(0)
    else:
        acc = cutlass.Float32(0)
        vec = cute.arch.load(key.iterator + base, vector_type)
        for k_lane in cutlass.range_constexpr(8):
            value = cutlass.Float32(vec[k_lane])
            acc = acc + value * value
        norm = cutlass.Float32(
            cute.arch.warp_reduction_sum(acc, threads_in_group=16)
        )
        state_value = (state.iterator + row).load()
        (out.iterator + row).store(norm + state_value)
"""


def test_wide_lane_loop_unswitches_and_hoists_invariant_reduction() -> None:
    result = _rewrite(_POSITIVE)
    assert len(result) == 4
    assert isinstance(result[-1], ast.If)
    lifted = result[-1]
    assert len(lifted.orelse) == 5
    assert isinstance(lifted.orelse[-1], ast.For)

    code = _source(result)
    assert code.count("warp_reduction_sum") == 1
    assert code.index("norm =") < code.rindex("for row_lane in range(16)")
    assert code.index("state_index_load =") < code.index("if invalid:")
    assert code.index("state_value =") > code.rindex("for row_lane in range(16)")


def test_output_lane_owner_does_not_block_uniform_unswitch() -> None:
    source = _POSITIVE.replace(
        "        (out.iterator + row).store(norm + state_value)",
        "        if cute.arch.lane_idx() % 16 == 0:\n"
        "            (out.iterator + row).store(norm + state_value)",
    )

    code = _source(_rewrite(source))

    assert code.count("warp_reduction_sum") == 1
    assert code.index("norm =") < code.rindex("for row_lane in range(16)")
    assert "if cute.arch.lane_idx() % 16 == 0:" in code


def test_full_lane_bound_is_removed_before_reduction_hoist() -> None:
    source = _POSITIVE.replace(
        "vec = cute.arch.load(key.iterator + base, vector_type)",
        "vec = (cute.arch.load(key.iterator + base, vector_type)\n"
        "               if cutlass.Int32(cute.arch.thread_idx()[1]) * 16\n"
        "               + cutlass.Int32(row_lane) < 128\n"
        "               else cutlass.Float32(0))",
    )
    code = _source(_rewrite(source, thread_block_dims=(16, 8, 1)))

    assert code.index("norm =") < code.rindex("for row_lane in range(16)")
    assert "thread_idx()[1]" not in code[: code.rindex("for row_lane in range(16)")]


def test_partial_lane_bound_is_not_removed() -> None:
    source = _POSITIVE.replace(
        "vec = cute.arch.load(key.iterator + base, vector_type)",
        "vec = (cute.arch.load(key.iterator + base, vector_type)\n"
        "               if cutlass.Int32(cute.arch.thread_idx()[1]) * 16\n"
        "               + cutlass.Int32(row_lane) < 127\n"
        "               else cutlass.Float32(0))",
    )
    code = _source(_rewrite(source, thread_block_dims=(16, 8, 1)))

    assert code.index("norm =") > code.rindex("for row_lane in range(16)")


def test_overflowing_lane_bound_is_not_simplified() -> None:
    source = _POSITIVE.replace(
        "vec = cute.arch.load(key.iterator + base, vector_type)",
        "vec = (cute.arch.load(key.iterator + base, vector_type)\n"
        "               if cutlass.Int32(row_lane) * 2147483648 >= 0\n"
        "               else cutlass.Float32(0))",
    )
    code = _source(_rewrite(source, thread_block_dims=(16, 8, 1)))

    assert code.index("norm =") > code.rindex("for row_lane in range(16)")


def test_two_lane_loop_hoists_expensive_invariant_reduction() -> None:
    result = _rewrite(_POSITIVE.replace("range(16)", "range(2)", 1))
    code = _source(result)

    assert code.count("warp_reduction_sum") == 1
    assert code.index("norm =") < code.rindex("for row_lane in range(2)")


def test_single_lane_loop_is_unchanged() -> None:
    result = _rewrite(_POSITIVE.replace("range(16)", "range(1)", 1))
    assert len(result) == 1 and isinstance(result[0], ast.For)
    assert _source(result).count("for row_lane in range(1)") == 1


def test_two_lane_loop_leaves_cheap_reduction_in_place() -> None:
    source = """
for row_lane in range(2):
    row = row_base + row_lane
    value = cutlass.Float32(seed)
    reduced = cutlass.Float32(
        cute.arch.warp_reduction_sum(value, threads_in_group=16)
    )
    (out.iterator + row).store(reduced)
"""
    result = _rewrite(source)

    assert len(result) == 1 and isinstance(result[0], ast.For)
    assert _source(result).index("reduced =") > _source(result).index(
        "for row_lane in range(2)"
    )


def test_row_dependent_work_does_not_make_cheap_reduction_profitable() -> None:
    source = """
for row_lane in range(2):
    row = row_base + row_lane
    row_0 = row + 1
    row_1 = row_0 + 1
    row_2 = row_1 + 1
    row_3 = row_2 + 1
    row_4 = row_3 + 1
    row_5 = row_4 + 1
    row_6 = row_5 + 1
    row_7 = row_6 + 1
    row_8 = row_7 + 1
    row_9 = row_8 + 1
    value = cutlass.Float32(seed)
    reduced = cutlass.Float32(
        cute.arch.warp_reduction_sum(value, threads_in_group=16)
    )
    (out.iterator + row).store(reduced + row_9)
"""
    code = _source(_rewrite(source))

    assert code.index("reduced =") > code.index("for row_lane in range(2)")


def test_overlapping_reduction_scores_only_marginal_work() -> None:
    source = """
for row_lane in range(2):
    row = row_base + row_lane
    acc = cutlass.Float32(0)
    for k_lane in cutlass.range_constexpr(8):
        value = (key.iterator + base + k_lane).load()
        acc = acc + cutlass.Float32(value) * cutlass.Float32(value)
    expensive = cutlass.Float32(
        cute.arch.warp_reduction_sum(acc, threads_in_group=16)
    )
    cheap = cutlass.Float32(
        cute.arch.warp_reduction_sum(expensive, threads_in_group=16)
    )
    (out.iterator + row).store(expensive + cheap)
"""
    code = _source(_rewrite(source))
    row_loop = code.index("for row_lane in range(2)")

    assert code.index("expensive =") < row_loop
    assert code.index("cheap =") > row_loop


def test_hoist_requires_storage_disjoint_proof() -> None:
    result = _rewrite(_POSITIVE, disjoint=set())
    assert len(result) == 1 and isinstance(result[0], ast.For)


def test_lane_dependent_reduction_stays_inside_loop() -> None:
    source = _POSITIVE.replace("key.iterator + base", "key.iterator + base + row")
    result = _rewrite(source)
    assert isinstance(result[-1], ast.If)
    lifted = result[-1]
    assert len(lifted.orelse) == 1
    assert isinstance(lifted.orelse[0], ast.For)
    assert "warp_reduction_sum" in ast.unparse(lifted.orelse[0])


def test_atomic_in_lane_loop_rejects_hoist() -> None:
    source = _POSITIVE.replace(
        "(out.iterator + row).store(norm + state_value)",
        "cute.arch.atomic_add(out.iterator + row, norm + state_value)",
    )
    result = _rewrite(source)
    assert len(result) == 1 and isinstance(result[0], ast.For)


def test_unknown_call_in_reduction_slice_rejects_hoist() -> None:
    source = _POSITIVE.replace(
        "value = cutlass.Float32(vec[k_lane])",
        "value = opaque_transform(vec[k_lane])",
    )
    result = _rewrite(source)
    assert len(result) == 1 and isinstance(result[0], ast.For)


def test_loop_carried_guard_is_not_unswitched() -> None:
    source = _POSITIVE.replace(
        "(out.iterator + row).store(0)",
        "invalid = False\n        (out.iterator + row).store(0)",
    )
    result = _rewrite(source)
    assert len(result) == 1
    loop = cast("ast.For", result[0])
    assert isinstance(loop, ast.For)
    assert isinstance(loop.body[-1], ast.If)


def test_effectful_guard_expression_is_not_unswitched() -> None:
    result = _rewrite(_POSITIVE.replace("if invalid:", "if predicate():"))
    assert len(result) == 1 and isinstance(result[0], ast.For)


def test_for_else_is_preserved_by_declining_rewrite() -> None:
    source = _POSITIVE + "else:\n    finalize()\n"
    result = _rewrite(source)
    assert len(result) == 1
    loop = cast("ast.For", result[0])
    assert len(loop.orelse) == 1
    assert "finalize()" in ast.unparse(loop.orelse[0])


def test_unknown_effect_in_remainder_rejects_load_motion() -> None:
    source = _POSITIVE.replace(
        "(out.iterator + row).store(norm + state_value)",
        "(out.iterator + row).store(norm + state_value)\n"
        "        opaque_sync_or_mutation()",
    )
    result = _rewrite(source)
    assert len(result) == 1 and isinstance(result[0], ast.For)


def test_selected_cache_backedge_rejects_reduction_hoist() -> None:
    source = _POSITIVE.replace(
        "value = cutlass.Float32(vec[k_lane])",
        "value = cutlass.Float32(vec[k_lane])\n"
        "            _fuse_cache_0[k_lane] = value",
    ).replace(
        "state_value = (state.iterator + row).load()",
        "_fuse_cache_0[0] = other\n        state_value = (state.iterator + row).load()",
    )
    result = _rewrite(source)
    assert isinstance(result[-1], ast.If)
    lifted = result[-1]
    assert len(lifted.orelse) == 1
    assert isinstance(lifted.orelse[0], ast.For)


def test_selected_result_backedge_rejects_reduction_hoist() -> None:
    source = _POSITIVE.replace(
        "(out.iterator + row).store(norm + state_value)",
        "(out.iterator + row).store(norm + state_value)\n"
        "        norm = cutlass.Float32(0)",
    )
    result = _rewrite(source)
    assert isinstance(result[-1], ast.If)
    lifted = result[-1]
    assert len(lifted.orelse) == 1
    assert isinstance(lifted.orelse[0], ast.For)


def test_inner_licm_rejects_late_reaching_definition() -> None:
    source = _POSITIVE.replace(
        "value = cutlass.Float32(vec[k_lane])",
        "use_before_definition = x + k_lane\n"
        "            x = cutlass.Float32(external)\n"
        "            value = cutlass.Float32(vec[k_lane]) + use_before_definition",
    )
    code = _source(_rewrite(source))
    assert code.index("use_before_definition =") < code.index(
        "x = cutlass.Float32(external)"
    )


def test_repeated_scalar_hoist_rejects_read_before_first_definition() -> None:
    source = _POSITIVE.replace(
        "state_value = (state.iterator + row).load()",
        "old_x = x + row_lane\n"
        "        x = cutlass.Float32(seed)\n"
        "        x = cutlass.Float32(seed)\n"
        "        state_value = (state.iterator + row).load()",
    )
    code = _source(_rewrite(source))
    row_loop = code.rindex("for row_lane in range(16)")
    assert code.index("old_x =", row_loop) < code.index("x = cutlass.Float32(seed)")


def test_repeated_cache_read_is_not_hoisted_across_cache_mutation() -> None:
    source = _POSITIVE.replace(
        "state_value = (state.iterator + row).load()",
        "cached = _fuse_cache_7[0]\n"
        "        _fuse_cache_7[0] = row_lane\n"
        "        cached = _fuse_cache_7[0]\n"
        "        state_value = (state.iterator + row).load()",
    )
    code = _source(_rewrite(source))
    row_loop = code.rindex("for row_lane in range(16)")
    assert code.count("cached = _fuse_cache_7[0]") == 2
    assert code.index("cached = _fuse_cache_7[0]") > row_loop


def test_mutable_store_staging_list_is_reinitialized_per_row() -> None:
    source = _POSITIVE.replace(
        "state_value = (state.iterator + row).load()",
        "_persistent_branch_store_values_0 = []\n"
        "        _persistent_branch_store_values_0.append(norm)\n"
        "        state_value = (state.iterator + row).load()",
    )
    code = _source(_rewrite(source))
    row_loop = code.rindex("for row_lane in range(16)")
    assert code.index("_persistent_branch_store_values_0 = []") > row_loop


def test_post_codegen_rename_alias_write_rejects_scalar_hoist() -> None:
    source = _POSITIVE.replace(
        "state_value = (state.iterator + row).load()",
        "alias = cutlass.Float32(seed)\n"
        "        use_alias = alias + row_lane\n"
        "        alias = cutlass.Float32(seed)\n"
        "        canonical = row_lane\n"
        "        state_value = (state.iterator + row).load()",
    )
    code = _source(_rewrite(source, rename_groups={"alias": "canonical"}))
    row_loop = code.rindex("for row_lane in range(16)")
    assert code.count("alias = cutlass.Float32(seed)") == 2
    assert code.index("alias = cutlass.Float32(seed)") > row_loop


def test_post_codegen_rename_alias_write_rejects_guard_unswitch() -> None:
    source = """
for row_lane in range(16):
    row = row_base + row_lane
    if cond:
        cond_alias = False
        (out.iterator + row).store(0)
    else:
        acc = cutlass.Float32(0)
        vec = cute.arch.load(key.iterator + base, vector_type)
        for k_lane in cutlass.range_constexpr(8):
            value = cutlass.Float32(vec[k_lane])
            acc = acc + value * value
        norm = cutlass.Float32(
            cute.arch.warp_reduction_sum(acc, threads_in_group=16)
        )
        (out.iterator + row).store(norm)
"""
    result = _rewrite(source, rename_groups={"cond_alias": "cond"})
    assert len(result) == 1 and isinstance(result[0], ast.For)


def test_inner_licm_rejects_two_selected_rename_aliases() -> None:
    source = _POSITIVE.replace(
        "value = cutlass.Float32(vec[k_lane])",
        "alias_1 = cutlass.Float32(seed_1)\n"
        "            use_1 = alias_1 + k_lane\n"
        "            alias_2 = cutlass.Float32(seed_2)\n"
        "            use_2 = alias_2 + k_lane\n"
        "            value = cutlass.Float32(vec[k_lane]) + use_1 + use_2",
    )
    code = _source(
        _rewrite(
            source,
            rename_groups={"alias_1": "canonical", "alias_2": "canonical"},
        )
    )
    inner_loop = code.index("for k_lane in cutlass.range_constexpr(8)")
    assert code.index("alias_1 =", inner_loop) > inner_loop
    assert code.index("alias_2 =", inner_loop) > inner_loop


def test_repeated_staging_read_is_not_hoisted_across_append() -> None:
    source = _POSITIVE.replace(
        "state_value = (state.iterator + row).load()",
        "staged = _persistent_branch_store_values_0[-1]\n"
        "        _persistent_branch_store_values_0.append(row_lane)\n"
        "        staged = _persistent_branch_store_values_0[-1]\n"
        "        state_value = (state.iterator + row).load()",
    )
    code = _source(_rewrite(source))
    row_loop = code.rindex("for row_lane in range(16)")
    assert code.count("staged = _persistent_branch_store_values_0[-1]") == 2
    assert code.index("staged = _persistent_branch_store_values_0[-1]") > row_loop


def test_operator_setitem_is_not_treated_as_pure() -> None:
    source = _POSITIVE.replace(
        "state_value = (state.iterator + row).load()",
        "operator.setitem(cache, 0, row_lane)\n"
        "        state_value = (state.iterator + row).load()",
    )
    result = _rewrite(source)
    assert len(result) == 1 and isinstance(result[0], ast.For)


def test_guard_depending_on_outer_thread_value_is_not_unswitched() -> None:
    body = ast.parse(
        "thread_flag = cute.arch.thread_idx()[0] < 16\n"
        + _POSITIVE.replace("if invalid:", "if thread_flag:")
    ).body
    loop = cast("ast.For", body[1])
    setattr(loop, HELION_LANE_LOOP_VAR_ATTR, "row_lane")
    result = hoist_lane_invariant_reductions(
        body,
        tensor_names=_TENSORS,
        proven_disjoint_tensor_pairs=_DISJOINT,
        uniform_names={"base", "batch", "row_base", "vector_type"},
    )
    assert isinstance(result[1], ast.For)


def test_guard_uniformity_tracks_prior_rename_alias_write() -> None:
    body = ast.parse(
        "cond_alias = cute.arch.thread_idx()[0]\n"
        "for row_lane in range(16):\n"
        "    row = row_base + row_lane\n"
        "    if cond:\n"
        "        (out.iterator + row).store(0)\n"
        "    else:\n"
        "        acc = cutlass.Float32(0)\n"
        "        vec = cute.arch.load(key.iterator + base, vector_type)\n"
        "        for k_lane in cutlass.range_constexpr(8):\n"
        "            value = cutlass.Float32(vec[k_lane])\n"
        "            acc = acc + value * value\n"
        "        norm = cutlass.Float32(\n"
        "            cute.arch.warp_reduction_sum(acc, threads_in_group=16)\n"
        "        )\n"
        "        (out.iterator + row).store(norm)\n"
    ).body
    loop = cast("ast.For", body[1])
    setattr(loop, HELION_LANE_LOOP_VAR_ATTR, "row_lane")
    result = hoist_lane_invariant_reductions(
        body,
        tensor_names=_TENSORS,
        proven_disjoint_tensor_pairs=_DISJOINT,
        rename_groups={"cond_alias": "cond"},
        uniform_names={"base", "cond", "row_base", "vector_type"},
    )
    assert isinstance(result[1], ast.For)


def test_inner_licm_rejects_cache_mutation_after_selected_read() -> None:
    source = _POSITIVE.replace(
        "value = cutlass.Float32(vec[k_lane])",
        "cached = _fuse_cache_7[0]\n"
        "            value = cutlass.Float32(vec[k_lane]) + cached\n"
        "            _fuse_cache_7[0] = k_lane",
    )
    code = _source(_rewrite(source))
    inner_loop = code.index("for k_lane in cutlass.range_constexpr(8)")
    assert code.index("cached = _fuse_cache_7[0]", inner_loop) > inner_loop


def test_inner_licm_rejects_cache_mutation_before_selected_read() -> None:
    source = _POSITIVE.replace(
        "value = cutlass.Float32(vec[k_lane])",
        "_fuse_cache_7[0] = k_lane\n"
        "            cached = _fuse_cache_7[0]\n"
        "            value = cutlass.Float32(vec[k_lane]) + cached",
    )
    code = _source(_rewrite(source))
    inner_loop = code.index("for k_lane in cutlass.range_constexpr(8)")
    assert code.index("cached = _fuse_cache_7[0]", inner_loop) > inner_loop


def test_reduction_hoist_rejects_external_staging_mutation() -> None:
    source = _POSITIVE.replace(
        "acc = cutlass.Float32(0)",
        "acc = cutlass.Float32(_persistent_branch_store_values_0[-1])",
    ).replace(
        "state_value = (state.iterator + row).load()",
        "_persistent_branch_store_values_0.append(row_lane)\n"
        "        state_value = (state.iterator + row).load()",
    )
    result = _rewrite(source)
    assert isinstance(result[-1], ast.If)
    lifted = result[-1]
    assert len(lifted.orelse) == 1
    assert isinstance(lifted.orelse[0], ast.For)


def test_identical_repeated_scalar_definitions_hoist_once() -> None:
    source = _POSITIVE.replace(
        "state_value = (state.iterator + row).load()",
        "factor = cutlass.Float32(seed)\n"
        "        left = factor + row\n"
        "        factor = cutlass.Float32(seed)\n"
        "        state_value = (state.iterator + row).load()",
    ).replace("norm + state_value", "norm + state_value + left + factor")
    code = _source(_rewrite(source))
    row_loop = code.rindex("for row_lane in range(16)")
    assert code.count("factor = cutlass.Float32(seed)") == 1
    assert code.index("factor = cutlass.Float32(seed)") < row_loop


def test_cutlass_pipeline_call_is_not_treated_as_pure() -> None:
    source = _POSITIVE.replace(
        "state_value = (state.iterator + row).load()",
        "pipeline = cutlass.pipeline.PipelineAsync.create(storage)\n"
        "        state_value = (state.iterator + row).load()",
    )
    result = _rewrite(source)
    assert len(result) == 1 and isinstance(result[0], ast.For)


def test_fully_hoistable_loop_keeps_loop_target_semantics() -> None:
    source = """
for row_lane in range(16):
    acc = cutlass.Float32(seed)
    norm = cutlass.Float32(
        cute.arch.warp_reduction_sum(acc, threads_in_group=16)
    )
"""
    result = _rewrite(source)
    assert len(result) == 1 and isinstance(result[0], ast.For)


_VECTOR_CACHE = """
_fuse_cache_10 = cute.make_rmem_tensor(4, cutlass.BFloat16)
_fuse_cache_11 = cute.make_rmem_tensor(4, cutlass.Float32)
for row_lane in range(16):
    head_value = (head.iterator + base).load()
    head_scaled = head_value * scale
    for element in cutlass.range_constexpr(4):
        index = base + cutlass.Int32(element)
        raw = (gate.iterator + index).load()
        _fuse_cache_10[(element - 0) // 1] = raw
        gate_value = cutlass.Float32(raw)
        bias_value = (bias.iterator + index).load()
        _fuse_cache_11[(element - 0) // 1] = bias_value
        combined = gate_value + bias_value
        decay_input = head_scaled * combined
        sigmoid = cute.math.rcp(
            1.0 + cute.math.exp2(
                -cutlass.Float32(decay_input) * 1.4426950408889634,
                fastmath=True,
            ),
            approx=True,
            ftz=True,
        )
        decay = cute.math.exp2(sigmoid * scale)
        sink_0 = decay + row_lane
    for element in cutlass.range_constexpr(4):
        index = base + cutlass.Int32(element)
        raw = _fuse_cache_10[(element - 0) // 1]
        gate_value = cutlass.Float32(raw)
        bias_value = _fuse_cache_11[(element - 0) // 1]
        combined = gate_value + bias_value
        decay_input = head_scaled * combined
        sigmoid = cute.math.rcp(
            1.0 + cute.math.exp2(
                -cutlass.Float32(decay_input) * 1.4426950408889634,
                fastmath=True,
            ),
            approx=True,
            ftz=True,
        )
        decay = cute.math.exp2(sigmoid * scale)
        sink_1 = decay + row_lane
    head_value = (head.iterator + base).load()
    head_scaled = head_value * scale
    for element in cutlass.range_constexpr(4):
        index = base + cutlass.Int32(element)
        raw = _fuse_cache_10[(element - 0) // 1]
        gate_value = cutlass.Float32(raw)
        bias_value = _fuse_cache_11[(element - 0) // 1]
        combined = gate_value + bias_value
        decay_input = head_scaled * combined
        sigmoid = cute.math.rcp(
            1.0 + cute.math.exp2(
                -cutlass.Float32(decay_input) * 1.4426950408889634,
                fastmath=True,
            ),
            approx=True,
            ftz=True,
        )
        decay = cute.math.exp2(sigmoid * scale)
        sink_2 = decay + row_lane
    (out.iterator + row_lane).store(sink_0 + sink_1 + sink_2)
"""


def test_nested_constexpr_vector_chain_is_cached_once() -> None:
    code = _source(_rewrite_body(_VECTOR_CACHE))
    row_loop = code.index("for row_lane in range(16)")

    assert code.count("cute.math.exp2(") == 2
    assert code.count("fastmath=True") == 1
    assert code.count("approx=True") == 1
    assert code.count("ftz=True") == 1
    assert code.count("_fuse_cache_11[element]") == 3
    assert "_fuse_cache_11[_lane_invariant_0] =" in code
    assert "cute.make_rmem_tensor(4, cutlass.Float32)" in code
    assert "cute.make_rmem_tensor(4, cutlass.BFloat16)" not in code
    assert code.index("for _lane_invariant_0 in cutlass.range_constexpr(4)") < row_loop
    assert code.count("(out.iterator + row_lane).store") == 1


def test_extended_ast_vector_chain_runs_after_uniform_unswitch() -> None:
    parsed = ast.parse(_VECTOR_CACHE)
    row_loop = cast("ast.For", parsed.body[-1])
    declarations = ast.unparse(ast.Module(body=parsed.body[:-1], type_ignores=[]))
    branch_body = ast.unparse(ast.Module(body=row_loop.body, type_ignores=[]))
    source = (
        f"{declarations}\n"
        "for row_lane in range(16):\n"
        "    invalid = base < 0\n"
        "    if invalid:\n"
        "        (out.iterator + row_lane).store(0)\n"
        "    else:\n"
        f"{textwrap.indent(branch_body, '        ')}\n"
    )
    converted = cast("ast.Module", convert(ast.parse(source)))
    assert isinstance(converted.body[0], ExtendedAST)

    result = _rewrite_body(source, extended=True)
    code = _source(result)

    assert code.count("cute.math.exp2(") == 2
    assert code.count("for row_lane in range(16)") == 2
    assert code.index("if invalid:") < code.index(
        "for _lane_invariant_0 in cutlass.range_constexpr(4)"
    )
    assert code.index(
        "for _lane_invariant_0 in cutlass.range_constexpr(4)"
    ) < code.rindex("for row_lane in range(16)")


def test_unswitched_vector_chain_rejects_cross_branch_cache_accesses() -> None:
    parsed = ast.parse(_VECTOR_CACHE)
    row_loop = cast("ast.For", parsed.body[-1])
    declarations = ast.unparse(ast.Module(body=parsed.body[:-1], type_ignores=[]))
    branch_body = textwrap.indent(
        ast.unparse(ast.Module(body=row_loop.body, type_ignores=[])), "        "
    )
    source = (
        f"{declarations}\n"
        "for row_lane in range(16):\n"
        "    invalid = base < 0\n"
        "    if invalid:\n"
        f"{branch_body}\n"
        "    else:\n"
        f"{branch_body}\n"
    )

    code = _source(_rewrite_body(source, extended=True))

    assert code.count("cute.math.exp2(") == 12
    assert code.count("for row_lane in range(16)") == 2
    assert "_lane_invariant_" not in code


def test_nested_constexpr_vector_chain_requires_disjoint_loads() -> None:
    code = _source(_rewrite_body(_VECTOR_CACHE, disjoint=set()))
    assert code.count("cute.math.exp2(") == 6
    assert "_lane_invariant_" not in code


def test_nested_constexpr_vector_chain_validates_cache_producer_load_root() -> None:
    source = _VECTOR_CACHE.replace(
        "_fuse_cache_11[(element - 0) // 1] = bias_value",
        "_fuse_cache_11[(element - 0) // 1] = (bias_alias.iterator + index).load()",
    )
    code = _source(_rewrite_body(source, rename_groups={"bias_alias": "bias"}))
    assert code.count("cute.math.exp2(") == 6
    assert "_lane_invariant_" not in code


def test_nested_constexpr_vector_chain_rejects_row_dependency() -> None:
    source = _VECTOR_CACHE.replace(
        "combined = gate_value + bias_value",
        "combined = gate_value + bias_value + row_lane",
    )
    code = _source(_rewrite_body(source))
    assert code.count("cute.math.exp2(") == 6
    assert "_lane_invariant_" not in code


def test_nested_constexpr_vector_chain_rejects_unknown_math_leaf() -> None:
    source = _VECTOR_CACHE.replace(
        "decay = cute.math.exp2(sigmoid * scale)",
        "decay = cute.math.unreviewed(sigmoid * scale)",
    )
    code = _source(_rewrite_body(source))
    assert code.count("cute.math.unreviewed(") == 3
    assert "_lane_invariant_" not in code


def test_nested_constexpr_vector_chain_rejects_arbitrary_bitcast_receiver() -> None:
    source = _VECTOR_CACHE.replace(
        "gate_value = cutlass.Float32(raw)",
        "gate_value = opaque(raw).bitcast(cutlass.Float32)",
    )
    code = _source(_rewrite_body(source))
    assert code.count("opaque(raw).bitcast(cutlass.Float32)") == 3
    assert "_lane_invariant_" not in code


def test_nested_constexpr_vector_chain_rejects_mismatched_bitcast_width() -> None:
    source = _VECTOR_CACHE.replace(
        "gate_value = cutlass.Float32(raw)",
        "gate_value = cutlass.Uint16(raw).bitcast(cutlass.Float32)",
    )
    code = _source(_rewrite_body(source))
    assert code.count("bitcast(cutlass.Float32)") == 3
    assert "_lane_invariant_" not in code


def test_nested_constexpr_vector_chain_rejects_wrong_cache_dtype() -> None:
    source = _VECTOR_CACHE.replace(
        "cute.make_rmem_tensor(4, cutlass.Float32)",
        "cute.make_rmem_tensor(4, cutlass.BFloat16)",
    )
    code = _source(_rewrite_body(source))
    assert code.count("cute.math.exp2(") == 6
    assert "_lane_invariant_" not in code


def test_nested_constexpr_vector_chain_models_input_cache_conversion() -> None:
    source = _VECTOR_CACHE.replace(
        "cute.make_rmem_tensor(4, cutlass.BFloat16)",
        "cute.make_rmem_tensor(4, cutlass.Float16)",
    )
    code = _source(_rewrite_body(source))
    assert code.count("cute.math.exp2(") == 6
    assert "_lane_invariant_" not in code


def test_nested_constexpr_vector_chain_rejects_nonidentity_cache_index() -> None:
    source = _VECTOR_CACHE.replace(
        "_fuse_cache_11[(element - 0) // 1] = bias_value",
        "_fuse_cache_11[element // 2] = bias_value",
    )
    code = _source(_rewrite_body(source))
    assert code.count("cute.math.exp2(") == 6
    assert "_lane_invariant_" not in code


def test_nested_constexpr_vector_chain_rejects_intervening_cache_write() -> None:
    source = _VECTOR_CACHE.replace(
        "    for element in cutlass.range_constexpr(4):\n"
        "        index = base + cutlass.Int32(element)\n"
        "        raw = _fuse_cache_10[(element - 0) // 1]",
        "    _fuse_cache_11[0] = cutlass.Float32(row_lane)\n"
        "    for element in cutlass.range_constexpr(4):\n"
        "        index = base + cutlass.Int32(element)\n"
        "        raw = _fuse_cache_10[(element - 0) // 1]",
        1,
    )
    code = _source(_rewrite_body(source))
    assert code.count("cute.math.exp2(") == 6
    assert "_lane_invariant_" not in code


def test_nested_constexpr_vector_chain_rejects_intervening_augassign() -> None:
    source = _VECTOR_CACHE.replace(
        "    for element in cutlass.range_constexpr(4):",
        "    head_scaled += row_lane\n    for element in cutlass.range_constexpr(4):",
        1,
    )
    code = _source(_rewrite_body(source))
    assert code.count("cute.math.exp2(") == 6
    assert "_lane_invariant_" not in code


def test_nested_constexpr_vector_chain_rejects_cache_handle_alias() -> None:
    source = _VECTOR_CACHE.replace(
        "for row_lane in range(16):",
        "cache_alias = _fuse_cache_11\nfor row_lane in range(16):",
    ).replace(
        "(out.iterator + row_lane).store(sink_0 + sink_1 + sink_2)",
        "alias_value = cache_alias[0]\n"
        "    (out.iterator + row_lane).store("
        "sink_0 + sink_1 + sink_2 + alias_value)",
    )
    code = _source(_rewrite_body(source))
    assert code.count("cute.math.exp2(") == 6
    assert "_lane_invariant_" not in code


def test_nested_constexpr_vector_chain_rejects_cache_handle_rebind() -> None:
    source = _VECTOR_CACHE.replace(
        "for row_lane in range(16):",
        "other_cache = cute.make_rmem_tensor(4, cutlass.Float32)\n"
        "for row_lane in range(16):\n"
        "    _fuse_cache_11 = other_cache",
    )
    code = _source(_rewrite_body(source))
    assert code.count("cute.math.exp2(") == 6
    assert "_lane_invariant_" not in code


def test_nested_constexpr_vector_chain_rejects_control_flow_before_value() -> None:
    source = _VECTOR_CACHE.replace(
        "    for element in cutlass.range_constexpr(4):\n"
        "        index = base + cutlass.Int32(element)",
        "    for element in cutlass.range_constexpr(4):\n"
        "        if element == 0:\n"
        "            continue\n"
        "        index = base + cutlass.Int32(element)",
    )
    code = _source(_rewrite_body(source))
    assert code.count("cute.math.exp2(") == 6
    assert "_lane_invariant_" not in code


def test_nested_constexpr_vector_chain_rejects_outer_control_transfer() -> None:
    source = _VECTOR_CACHE.replace(
        "    head_value = (head.iterator + base).load()",
        "    if cond:\n        continue\n"
        "    head_value = (head.iterator + base).load()",
        1,
    )
    code = _source(_rewrite_body(source))
    assert code.count("cute.math.exp2(") == 6
    assert "_lane_invariant_" not in code


def test_nested_constexpr_vector_chain_rejects_induction_variable_write() -> None:
    source = _VECTOR_CACHE.replace(
        "    for element in cutlass.range_constexpr(4):\n"
        "        index = base + cutlass.Int32(element)",
        "    for element in cutlass.range_constexpr(4):\n"
        "        element = element + 1\n"
        "        index = base + cutlass.Int32(element)",
    )
    code = _source(_rewrite_body(source))
    assert code.count("cute.math.exp2(") == 6
    assert "_lane_invariant_" not in code


def test_nested_constexpr_vector_chain_keeps_cache_with_external_consumer() -> None:
    source = (
        _VECTOR_CACHE.replace(
            "        sink_0 = decay + row_lane",
            "        sink_0 = decay + row_lane\n        other_0 = raw + row_lane",
        )
        .replace(
            "        sink_1 = decay + row_lane",
            "        sink_1 = decay + row_lane\n        other_1 = raw + row_lane",
        )
        .replace(
            "        sink_2 = decay + row_lane",
            "        sink_2 = decay + row_lane\n        other_2 = raw + row_lane",
        )
        .replace(
            "sink_0 + sink_1 + sink_2)",
            "sink_0 + sink_1 + sink_2 + other_0 + other_1 + other_2)",
        )
    )
    code = _source(_rewrite_body(source))
    assert code.count("cute.math.exp2(") == 6
    assert code.count("_fuse_cache_10[") == 3
    assert "_fuse_cache_10[(element - 0) // 1] = raw" in code
    assert "_lane_invariant_" not in code


def test_nested_constexpr_vector_chain_rejects_terminal_intermediate_collision() -> (
    None
):
    source = """
_fuse_cache_20 = cute.make_rmem_tensor(4, cutlass.Float32)
for row_lane in range(16):
    for element in cutlass.range_constexpr(4):
        raw = cutlass.Float32((bias.iterator + element).load())
        _fuse_cache_20[element] = raw
        decay = cute.math.exp2(raw)
        sink_0 = decay + row_lane
    for element in cutlass.range_constexpr(4):
        decay = _fuse_cache_20[element]
        transformed = cute.math.exp2(decay)
        other = decay + row_lane
        sink_1 = transformed + row_lane
    (out.iterator + row_lane).store(sink_0 + sink_1 + other)
"""
    code = _source(_rewrite_body(source))
    assert code.count("cute.math.exp2(") == 2
    assert "_fuse_cache_20[element] = raw" in code
    assert "_lane_invariant_" not in code


def test_nested_constexpr_vector_fresh_names_cover_enclosing_scope() -> None:
    source = _VECTOR_CACHE.replace(
        "_fuse_cache_10 =",
        "_lane_invariant_0 = cutlass.Float32(7)\n_fuse_cache_10 =",
    ).replace(
        "sink_0 + sink_1 + sink_2)",
        "sink_0 + sink_1 + sink_2 + _lane_invariant_0)",
    )
    code = _source(_rewrite_body(source))
    assert code.count("cute.math.exp2(") == 2
    assert "for _lane_invariant_1 in cutlass.range_constexpr(4)" in code
    assert "_lane_invariant_0 = cutlass.Float32(7)" in code


def test_nested_constexpr_vector_fresh_names_cover_future_renames() -> None:
    result = _rewrite_body(
        _VECTOR_CACHE,
        rename_groups={"scale": "_lane_invariant_0"},
    )
    ast_rename(ast.Module(body=result, type_ignores=[]), {"scale": "_lane_invariant_0"})
    code = _source(result)
    assert code.count("cute.math.exp2(") == 2
    assert "for _lane_invariant_1 in cutlass.range_constexpr(4)" in code
    assert "for _lane_invariant_0 in" not in code


def test_nested_constexpr_vector_chain_rejects_unknown_operand_dtype() -> None:
    source = _VECTOR_CACHE.replace(
        "decay = cute.math.exp2(sigmoid * scale)",
        "decay = cute.math.exp2(bias_value * mystery)",
    )
    code = _source(_rewrite_body(source))
    assert code.count("cute.math.exp2(") == 6
    assert "_lane_invariant_" not in code


def test_nested_constexpr_vector_chain_rejects_mutated_external_subscript() -> None:
    source = _VECTOR_CACHE.replace(
        "combined = gate_value + bias_value",
        "combined = gate_value + bias_value + cutlass.Float32(coeff[0])",
    ).replace(
        "    (out.iterator + row_lane).store",
        "    coeff[0] = row_lane\n    (out.iterator + row_lane).store",
    )
    code = _source(_rewrite_body(source))
    assert code.count("cute.math.exp2(") == 6
    assert "_lane_invariant_" not in code


def test_nested_constexpr_vector_chain_rejects_mutated_selected_alias() -> None:
    source = _VECTOR_CACHE.replace(
        "    head_value = (head.iterator + base).load()",
        "    coeff_alias = coeff\n"
        "    coeff_alias[0] = row_lane\n"
        "    head_value = (head.iterator + base).load()",
        1,
    ).replace(
        "combined = gate_value + bias_value",
        "combined = gate_value + bias_value + cutlass.Float32(coeff_alias[0])",
    )
    code = _source(_rewrite_body(source))
    assert code.count("cute.math.exp2(") == 6
    assert "_lane_invariant_" not in code


def test_nested_constexpr_vector_chain_does_not_cache_boolean_as_float() -> None:
    source = _VECTOR_CACHE.replace(
        "decay = cute.math.exp2(sigmoid * scale)",
        "decay = not cute.math.exp2(sigmoid * scale)",
    )
    code = _source(_rewrite_body(source))
    assert code.count("decay = not cute.math.exp2") == 3
    assert not any(
        "_fuse_cache_11[_lane_invariant_" in line and "not " in line
        for line in code.splitlines()
    )


def test_nested_constexpr_vector_chain_rejects_integer_true_divide_dtype() -> None:
    source = """
_fuse_cache_20 = cute.make_rmem_tensor(4, cutlass.Int32)
for row_lane in range(16):
    for element in cutlass.range_constexpr(4):
        raw = (indices.iterator + element).load()
        _fuse_cache_20[element] = raw
        transformed = (
            cutlass.Int32(cute.math.exp2(cutlass.Float32(raw)))
            / cutlass.Int32(2)
        )
        sink_0 = transformed + row_lane
    for element in cutlass.range_constexpr(4):
        raw = _fuse_cache_20[element]
        transformed = (
            cutlass.Int32(cute.math.exp2(cutlass.Float32(raw)))
            / cutlass.Int32(2)
        )
        sink_1 = transformed + row_lane
    (out.iterator + row_lane).store(sink_0 + sink_1)
"""
    code = _source(_rewrite_body(source))
    assert code.count("cute.math.exp2(") == 2
    assert "_fuse_cache_20[element] = raw" in code
    assert "_lane_invariant_" not in code


def test_nested_constexpr_vector_chain_uses_explicit_arch_load_dtype() -> None:
    source = _VECTOR_CACHE.replace(
        "bias_value = (bias.iterator + index).load()",
        "bias_value = cute.arch.load(bias.iterator + index, cutlass.Float16)",
    ).replace(
        "decay = cute.math.exp2(sigmoid * scale)",
        "decay = cute.math.exp2(bias_value)",
    )
    code = _source(_rewrite_body(source))
    assert code.count("cute.math.exp2(") == 6
    assert "_lane_invariant_" not in code


def test_nested_constexpr_vector_chain_accepts_matching_arch_load_dtype() -> None:
    source = _VECTOR_CACHE.replace(
        "bias_value = (bias.iterator + index).load()",
        "bias_value = cute.arch.load(bias.iterator + index, cutlass.Float32)",
    )
    code = _source(_rewrite_body(source))
    assert code.count("cute.math.exp2(") == 2
    assert "_fuse_cache_11[_lane_invariant_0]" in code


def test_nested_constexpr_vector_chain_rejects_unknown_arch_load_option() -> None:
    source = _VECTOR_CACHE.replace(
        "bias_value = (bias.iterator + index).load()",
        "bias_value = cute.arch.load("
        "bias.iterator + index, cutlass.Float32, volatile=True)",
    )
    code = _source(_rewrite_body(source))
    assert code.count("cute.math.exp2(") == 6
    assert "_lane_invariant_" not in code


def test_nested_constexpr_vector_chain_rejects_rename_alias_mutation() -> None:
    source = _VECTOR_CACHE.replace(
        "    head_value = (head.iterator + base).load()",
        "    alias = row_lane\n    head_value = (head.iterator + base).load()",
        1,
    )
    code = _source(_rewrite_body(source, rename_groups={"head_value": "alias"}))
    assert code.count("cute.math.exp2(") == 6
    assert "_lane_invariant_" not in code


def test_nested_constexpr_vector_chain_rejects_lane_rename_alias() -> None:
    source = _VECTOR_CACHE.replace(
        "combined = gate_value + bias_value",
        "combined = gate_value + bias_value + cutlass.Float32(other)",
    )
    code = _source(_rewrite_body(source, rename_groups={"other": "row_lane"}))
    assert code.count("cute.math.exp2(") == 6
    assert "_lane_invariant_" not in code


def test_nested_constexpr_vector_cache_requires_same_path_allocation() -> None:
    source = """
if cond:
    _fuse_cache_10 = cute.make_rmem_tensor(4, cutlass.BFloat16)
    _fuse_cache_11 = cute.make_rmem_tensor(4, cutlass.Float32)
else:
""" + "\n".join(
        f"    {line}" if line else line for line in _VECTOR_CACHE.splitlines()[3:]
    )
    code = _source(_rewrite_body(source))
    assert code.count("cute.math.exp2(") == 6
    assert "_lane_invariant_" not in code


def test_nested_constexpr_vector_chain_supports_other_extent() -> None:
    source = """
_fuse_cache_20 = cute.make_rmem_tensor(3, cutlass.Float32)
for row_lane in range(16):
    for element in cutlass.range_constexpr(3):
        raw = (gate.iterator + element).load()
        _fuse_cache_20[element] = raw
        value = cutlass.Float32(raw)
        transformed = cute.math.exp2(value)
        sink_0 = transformed + row_lane
    for element in cutlass.range_constexpr(3):
        raw = _fuse_cache_20[element]
        value = cutlass.Float32(raw)
        transformed = cute.math.exp2(value)
        sink_1 = transformed + row_lane
    (out.iterator + row_lane).store(sink_0 + sink_1)
"""
    dtypes = {**_TENSOR_DTYPES, "gate": "cutlass.Float32"}
    code = _source(_rewrite_body(source, tensor_dtypes=dtypes))
    assert code.count("cute.math.exp2(") == 1
    assert "cute.make_rmem_tensor(3, cutlass.Float32)" in code
    assert "for _lane_invariant_0 in cutlass.range_constexpr(3)" in code


def test_nested_constexpr_vector_chain_rejects_narrow_float_literal_promotion() -> None:
    source = """
_fuse_cache_20 = cute.make_rmem_tensor(3, cutlass.Float16)
for row_lane in range(16):
    for element in cutlass.range_constexpr(3):
        raw = (gate.iterator + element).load()
        _fuse_cache_20[element] = raw
        value = cutlass.Float16(raw)
        transformed = cute.math.exp2(value) + 1
        sink_0 = transformed + row_lane
    for element in cutlass.range_constexpr(3):
        raw = _fuse_cache_20[element]
        value = cutlass.Float16(raw)
        transformed = cute.math.exp2(value) + 1
        sink_1 = transformed + row_lane
    (out.iterator + row_lane).store(sink_0 + sink_1)
"""
    dtypes = {**_TENSOR_DTYPES, "gate": "cutlass.Float16"}
    code = _source(_rewrite_body(source, tensor_dtypes=dtypes))
    assert code.count("cute.math.exp2(") == 2
    assert "_fuse_cache_20[element] = raw" in code
    assert "_lane_invariant_" not in code


def test_nested_constexpr_vector_chain_rejects_large_integer_promotion() -> None:
    source = """
_fuse_cache_20 = cute.make_rmem_tensor(3, cutlass.Float32)
for row_lane in range(16):
    for element in cutlass.range_constexpr(3):
        raw = (bias.iterator + element).load()
        _fuse_cache_20[element] = raw
        transformed = cute.math.exp2(raw) + 4294967296
        sink_0 = transformed + row_lane
    for element in cutlass.range_constexpr(3):
        raw = _fuse_cache_20[element]
        transformed = cute.math.exp2(raw) + 4294967296
        sink_1 = transformed + row_lane
    (out.iterator + row_lane).store(sink_0 + sink_1)
"""
    code = _source(_rewrite_body(source))
    assert code.count("cute.math.exp2(") == 2
    assert "_fuse_cache_20[element] = raw" in code
    assert "_lane_invariant_" not in code


def test_nested_constexpr_vector_chain_rejects_mixed_ifexp_dtypes() -> None:
    source = """
_fuse_cache_20 = cute.make_rmem_tensor(3, cutlass.Float32)
for row_lane in range(16):
    for element in cutlass.range_constexpr(3):
        raw = (bias.iterator + element).load()
        _fuse_cache_20[element] = raw
        transformed = cute.math.exp2(raw) if cond else scale
        sink_0 = transformed + row_lane
    for element in cutlass.range_constexpr(3):
        raw = _fuse_cache_20[element]
        transformed = cute.math.exp2(raw) if cond else scale
        sink_1 = transformed + row_lane
    (out.iterator + row_lane).store(sink_0 + sink_1)
"""
    code = _source(_rewrite_body(source))
    assert code.count("cute.math.exp2(") == 2
    assert "_fuse_cache_20[element] = raw" in code
    assert "_lane_invariant_" not in code
