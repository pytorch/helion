"""Tests for fast-math affine sum-reduction factoring."""

from __future__ import annotations

import ast
from typing import cast

import torch

from helion._compiler.ast_extension import ExtendedAST
from helion._compiler.ast_extension import convert
from helion._compiler.ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from helion._compiler.cute.factor_affine_reductions import factor_affine_reductions
from helion._compiler.cute.factor_affine_reductions import hoist_factored_reductions
from helion._compiler.cute.factor_affine_reductions import (
    hoist_packed_factored_reductions,
)
from helion._compiler.cute.factor_affine_reductions import pack_fp32_constexpr_loops
from helion.language import _tracing_ops
from helion.language import view_ops


def _add_node(
    graph: torch.fx.Graph,
    target: object,
    args: tuple[object, ...],
    value: torch.Tensor,
) -> torch.fx.Node:
    # pyrefly: ignore [bad-argument-type]
    node = graph.call_function(target, args)
    node.meta["val"] = value
    return node


def _affine_sum_graph(
    *, dtype: torch.dtype = torch.float32, mask_value: float | None = None
) -> tuple[torch.fx.Graph, torch.fx.Node]:
    graph = torch.fx.Graph()
    base = graph.placeholder("base")
    scalar = graph.placeholder("scalar")
    direction = graph.placeholder("direction")
    weight = graph.placeholder("weight")
    base.meta["val"] = torch.empty(4, 8, dtype=dtype)
    scalar.meta["val"] = torch.empty(4, dtype=dtype)
    direction.meta["val"] = torch.empty(8, dtype=dtype)
    weight.meta["val"] = torch.empty(8, dtype=dtype)

    scalar_view = _add_node(
        graph,
        view_ops.subscript,
        (scalar, [slice(None), None]),
        torch.empty(4, 1, dtype=dtype),
    )
    direction_view = _add_node(
        graph,
        view_ops.subscript,
        (direction, [None, slice(None)]),
        torch.empty(1, 8, dtype=dtype),
    )
    affine = _add_node(
        graph,
        torch.ops.aten.mul.Tensor,
        (scalar_view, direction_view),
        torch.empty(4, 8, dtype=dtype),
    )
    update = _add_node(
        graph,
        torch.ops.aten.add.Tensor,
        (base, affine),
        torch.empty(4, 8, dtype=dtype),
    )
    weight_view = _add_node(
        graph,
        view_ops.subscript,
        (weight, [None, slice(None)]),
        torch.empty(1, 8, dtype=dtype),
    )
    product = _add_node(
        graph,
        torch.ops.aten.mul.Tensor,
        (update, weight_view),
        torch.empty(4, 8, dtype=dtype),
    )
    reduction_input = product
    if mask_value is not None:
        reduction_input = _add_node(
            graph,
            _tracing_ops._mask_to,
            (product, mask_value),
            torch.empty(4, 8, dtype=dtype),
        )
    reduction = _add_node(
        graph,
        torch.ops.aten.sum.dim_IntList,
        (reduction_input, [-1]),
        torch.empty(4, dtype=dtype),
    )
    graph.output(reduction)
    return graph, reduction


def test_factors_affine_sum_when_fast_math_is_enabled() -> None:
    graph, original_reduction = _affine_sum_graph()

    assert factor_affine_reductions(graph, fast_math=True) == 1

    output = next(node for node in graph.nodes if node.op == "output")
    factored = output.args[0]
    assert isinstance(factored, torch.fx.Node)
    assert factored.target is torch.ops.aten.add.Tensor
    assert original_reduction not in graph.nodes
    reductions = [
        node
        for node in graph.nodes
        if node.op == "call_function" and node.target is torch.ops.aten.sum.dim_IntList
    ]
    assert len(reductions) == 3
    assert all(tuple(node.meta["val"].shape) == (4,) for node in reductions)
    assert any(node.name == "_helion_factored_base_sum" for node in reductions)
    assert any(node.name == "_helion_factored_dot_sum" for node in reductions)


def test_does_not_factor_without_fast_math() -> None:
    graph, original_reduction = _affine_sum_graph()

    assert factor_affine_reductions(graph, fast_math=False) == 0

    assert original_reduction in graph.nodes


def test_requires_direct_prelowering_sum_input() -> None:
    graph, original_reduction = _affine_sum_graph(mask_value=0)

    assert factor_affine_reductions(graph, fast_math=True) == 0

    assert original_reduction in graph.nodes


def test_requires_fp32_arithmetic() -> None:
    graph, original_reduction = _affine_sum_graph(dtype=torch.float16)

    assert factor_affine_reductions(graph, fast_math=True) == 0

    assert original_reduction in graph.nodes


def test_rejects_sum_dtype_override() -> None:
    graph, original_reduction = _affine_sum_graph()
    original_reduction.kwargs = {"dtype": torch.float64}
    original_reduction.meta["val"] = torch.empty(4, dtype=torch.float64)

    assert factor_affine_reductions(graph, fast_math=True) == 0

    assert original_reduction in graph.nodes


def test_scalar_sum_does_not_crash_dimension_normalization() -> None:
    graph = torch.fx.Graph()
    source = graph.placeholder("source")
    source.meta["val"] = torch.empty((), dtype=torch.float32)
    reduction = _add_node(
        graph,
        torch.ops.aten.sum.dim_IntList,
        (source, [0]),
        torch.empty((), dtype=torch.float32),
    )
    graph.output(reduction)

    assert factor_affine_reductions(graph, fast_math=True) == 0


def test_requires_direction_invariant_over_retained_dims() -> None:
    graph, original_reduction = _affine_sum_graph()
    affine = next(
        node
        for node in graph.nodes
        if node.op == "call_function" and node.target is torch.ops.aten.add.Tensor
    ).args[1]
    assert isinstance(affine, torch.fx.Node)
    direction_view = affine.args[1]
    assert isinstance(direction_view, torch.fx.Node)
    direction_view.args = (direction_view.args[0], [slice(None), slice(None)])

    assert factor_affine_reductions(graph, fast_math=True) == 0

    assert original_reduction in graph.nodes


def _tagged_lane_loop(source: str) -> list[ast.stmt]:
    body = ast.parse(source).body
    loop = next(
        node
        for statement in body
        for node in ast.walk(statement)
        if isinstance(node, ast.For)
    )
    setattr(loop, HELION_LANE_LOOP_VAR_ATTR, "row_lane")
    return body


def test_hoists_factored_row_invariant_reduction() -> None:
    body = _tagged_lane_loop(
        """
k_cache = cute.make_rmem_tensor(8, cutlass.Float32)
q_cache = cute.make_rmem_tensor(8, cutlass.Float32)
if valid:
    for row_lane in range(16):
        row = row_base + row_lane
        dot_acc = cutlass.Float32(0)
        for k_lane in cutlass.range_constexpr(8):
            k = cutlass.Float32(k_cache[k_lane])
            q = cutlass.Float32(q_cache[k_lane])
            product = k * q
            dot_acc = dot_acc + cutlass.Float32(product)
        _helion_factored_dot_sum = cutlass.Float32(
            cute.arch.warp_reduction_sum(dot_acc, threads_in_group=16)
        )
        output = base[row_lane] + scalar[row_lane] * _helion_factored_dot_sum
"""
    )

    rewritten = hoist_factored_reductions(body, fast_math=True)
    code = ast.unparse(ast.Module(body=rewritten, type_ignores=[]))

    assert code.index("_factored_acc_0") < code.index("for row_lane in range(16)")
    assert code.count("warp_reduction_sum") == 1
    assert "k_cache[_factored_lane_0]" in code
    assert "q_cache[_factored_lane_0]" in code
    assert "scalar[row_lane] * _helion_factored_dot_sum" in code


def test_does_not_hoist_row_dependent_reduction() -> None:
    body = _tagged_lane_loop(
        """
for row_lane in range(16):
    dot_acc = cutlass.Float32(0)
    for k_lane in cutlass.range_constexpr(8):
        product = k_cache[k_lane] * row_lane
        dot_acc = dot_acc + cutlass.Float32(product)
    _helion_factored_dot_sum = cutlass.Float32(
        cute.arch.warp_reduction_sum(dot_acc, threads_in_group=16)
    )
    output = scalar[row_lane] * _helion_factored_dot_sum
"""
    )

    code_before = ast.unparse(ast.Module(body=body, type_ignores=[]))
    rewritten = hoist_factored_reductions(body, fast_math=True)
    code_after = ast.unparse(ast.Module(body=rewritten, type_ignores=[]))

    assert code_after == code_before


def test_does_not_hoist_factored_reduction_without_fast_math() -> None:
    body = _tagged_lane_loop(
        """
for row_lane in range(16):
    dot_acc = cutlass.Float32(0)
    for k_lane in cutlass.range_constexpr(8):
        product = k_cache[k_lane] * q_cache[k_lane]
        dot_acc = dot_acc + cutlass.Float32(product)
    _helion_factored_dot_sum = cutlass.Float32(
        cute.arch.warp_reduction_sum(dot_acc)
    )
    output = scalar[row_lane] * _helion_factored_dot_sum
"""
    )

    code_before = ast.unparse(ast.Module(body=body, type_ignores=[]))
    rewritten = hoist_factored_reductions(body, fast_math=False)
    code_after = ast.unparse(ast.Module(body=rewritten, type_ignores=[]))

    assert code_after == code_before


def test_does_not_hoist_unmarked_invariant_reduction() -> None:
    body = _tagged_lane_loop(
        """
for row_lane in range(16):
    dot_acc = cutlass.Float32(0)
    for k_lane in cutlass.range_constexpr(8):
        product = k_cache[k_lane] * q_cache[k_lane]
        dot_acc = dot_acc + cutlass.Float32(product)
    ordinary_dot = cutlass.Float32(cute.arch.warp_reduction_sum(dot_acc))
    output = scalar[row_lane] * ordinary_dot
"""
    )

    code_before = ast.unparse(ast.Module(body=body, type_ignores=[]))
    rewritten = hoist_factored_reductions(body, fast_math=True)
    code_after = ast.unparse(ast.Module(body=rewritten, type_ignores=[]))

    assert code_after == code_before


def test_fuses_adjacent_reductions_with_same_lane_cache() -> None:
    body = _tagged_lane_loop(
        """
cache = cute.make_rmem_tensor(8, cutlass.Float32)
for row_lane in range(16):
    first_acc = cutlass.Float32(0)
    for k_lane in cutlass.range_constexpr(8):
        value = state[row_lane, k_lane]
        cache[k_lane] = value
        common = value * 2.0
        first_term = common * key[k_lane]
        first_acc = first_acc + cutlass.Float32(first_term)
    first = cutlass.Float32(cute.arch.warp_reduction_sum(first_acc))
    second_acc = cutlass.Float32(0)
    for other_lane in cutlass.range_constexpr(8):
        value = cache[other_lane]
        common = value * 2.0
        second_term = common * query[other_lane]
        second_acc = second_acc + cutlass.Float32(second_term)
    _helion_factored_base_sum = cutlass.Float32(
        cute.arch.warp_reduction_sum(second_acc)
    )
    output = first + _helion_factored_base_sum
"""
    )

    rewritten = hoist_factored_reductions(body, fast_math=True)
    code = ast.unparse(ast.Module(body=rewritten, type_ignores=[]))
    row_loop = next(
        node
        for node in ast.walk(ast.Module(body=rewritten, type_ignores=[]))
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "row_lane"
    )

    assert code.count("cutlass.range_constexpr(8)") == 1
    assert "cache[k_lane] = value" in ast.unparse(row_loop)
    assert "value = cache[k_lane]" not in ast.unparse(row_loop)
    assert ast.unparse(row_loop).count("common = value * 2.0") == 1
    assert ast.unparse(row_loop).index("first_acc = first_acc") < ast.unparse(
        row_loop
    ).index("second_acc = second_acc")


def _assert_factored_ast_unchanged(source: str) -> None:
    body = _tagged_lane_loop(source)
    before = ast.unparse(ast.Module(body=body, type_ignores=[]))

    rewritten = hoist_factored_reductions(body, fast_math=True)

    assert ast.unparse(ast.Module(body=rewritten, type_ignores=[])) == before


def test_rejected_fusion_does_not_mutate_second_lane_name() -> None:
    _assert_factored_ast_unchanged(
        """
cache = cute.make_rmem_tensor(8, cutlass.Float32)
for row_lane in range(16):
    first_acc = cutlass.Float32(0)
    for k_lane in cutlass.range_constexpr(8):
        value = source[k_lane]
        cache[k_lane] = value
        first_acc = first_acc + cutlass.Float32(value)
    first = cutlass.Float32(cute.arch.warp_reduction_sum(first_acc))
    second_acc = cutlass.Float32(0)
    for other_lane in cutlass.range_constexpr(8):
        value = cache[other_lane + 1]
        second_acc = second_acc + cutlass.Float32(value)
    _helion_factored_base_sum = cutlass.Float32(
        cute.arch.warp_reduction_sum(second_acc)
    )
"""
    )


def test_does_not_fuse_when_second_initializer_crosses_a_read() -> None:
    _assert_factored_ast_unchanged(
        """
cache = cute.make_rmem_tensor(8, cutlass.Float32)
second_acc = cutlass.Float32(7)
for row_lane in range(16):
    first_acc = cutlass.Float32(0)
    for lane in cutlass.range_constexpr(8):
        prior = second_acc
        value = source[lane] + prior
        cache[lane] = value
        first_acc = first_acc + cutlass.Float32(value)
    first = cutlass.Float32(cute.arch.warp_reduction_sum(first_acc))
    second_acc = cutlass.Float32(0)
    for lane in cutlass.range_constexpr(8):
        value = cache[lane]
        second_acc = second_acc + cutlass.Float32(value)
    _helion_factored_base_sum = cutlass.Float32(
        cute.arch.warp_reduction_sum(second_acc)
    )
"""
    )


def test_does_not_fuse_cross_loop_scalar_dataflow() -> None:
    _assert_factored_ast_unchanged(
        """
cache = cute.make_rmem_tensor(8, cutlass.Float32)
for row_lane in range(16):
    first_acc = cutlass.Float32(0)
    for lane in cutlass.range_constexpr(8):
        shared = source[lane]
        cache[lane] = shared
        first_acc = first_acc + cutlass.Float32(shared)
    first = cutlass.Float32(cute.arch.warp_reduction_sum(first_acc))
    second_acc = cutlass.Float32(0)
    for lane in cutlass.range_constexpr(8):
        term = shared * weight[lane]
        second_acc = second_acc + cutlass.Float32(term)
    _helion_factored_base_sum = cutlass.Float32(
        cute.arch.warp_reduction_sum(second_acc)
    )
"""
    )


def test_does_not_fuse_second_loop_across_first_reduction_dependency() -> None:
    _assert_factored_ast_unchanged(
        """
cache = cute.make_rmem_tensor(8, cutlass.Float32)
for row_lane in range(16):
    first_acc = cutlass.Float32(0)
    for lane in cutlass.range_constexpr(8):
        value = source[lane]
        cache[lane] = value
        first_acc = first_acc + cutlass.Float32(value)
    first = cutlass.Float32(cute.arch.warp_reduction_sum(first_acc))
    second_acc = cutlass.Float32(0)
    for lane in cutlass.range_constexpr(8):
        term = cache[lane] * first
        second_acc = second_acc + cutlass.Float32(term)
    _helion_factored_base_sum = cutlass.Float32(
        cute.arch.warp_reduction_sum(second_acc)
    )
"""
    )


def test_does_not_fuse_second_loop_overwriting_first_reduction() -> None:
    _assert_factored_ast_unchanged(
        """
cache = cute.make_rmem_tensor(8, cutlass.Float32)
for row_lane in range(16):
    first_acc = cutlass.Float32(0)
    for lane in cutlass.range_constexpr(8):
        value = source[lane]
        cache[lane] = value
        first_acc = first_acc + cutlass.Float32(value)
    first = cutlass.Float32(cute.arch.warp_reduction_sum(first_acc))
    second_acc = cutlass.Float32(0)
    for lane in cutlass.range_constexpr(8):
        value = cache[lane]
        first = replacement[lane]
        second_acc = second_acc + cutlass.Float32(value)
    _helion_factored_base_sum = cutlass.Float32(
        cute.arch.warp_reduction_sum(second_acc)
    )
"""
    )


def test_does_not_fuse_cache_read_after_write_or_noninjective_stores() -> None:
    for store_index in ("lane", "0"):
        _assert_factored_ast_unchanged(
            f"""
cache = cute.make_rmem_tensor(8, cutlass.Float32)
for row_lane in range(16):
    first_acc = cutlass.Float32(0)
    for lane in cutlass.range_constexpr(8):
        value = cache[lane]
        cache[{store_index}] = source[lane]
        first_acc = first_acc + cutlass.Float32(value)
    first = cutlass.Float32(cute.arch.warp_reduction_sum(first_acc))
    second_acc = cutlass.Float32(0)
    for lane in cutlass.range_constexpr(8):
        value = cache[lane]
        second_acc = second_acc + cutlass.Float32(value)
    _helion_factored_base_sum = cutlass.Float32(
        cute.arch.warp_reduction_sum(second_acc)
    )
"""
        )


def test_does_not_fuse_aliased_register_fragment() -> None:
    _assert_factored_ast_unchanged(
        """
cache = cute.make_rmem_tensor(8, cutlass.Float32)
alias = cache
for row_lane in range(16):
    first_acc = cutlass.Float32(0)
    for lane in cutlass.range_constexpr(8):
        value = source[lane]
        cache[lane] = value
        first_acc = first_acc + cutlass.Float32(value)
    first = cutlass.Float32(cute.arch.warp_reduction_sum(first_acc))
    second_acc = cutlass.Float32(0)
    for lane in cutlass.range_constexpr(8):
        value = alias[lane]
        second_acc = second_acc + cutlass.Float32(value)
    _helion_factored_base_sum = cutlass.Float32(
        cute.arch.warp_reduction_sum(second_acc)
    )
"""
    )


def test_does_not_hoist_unused_effect_or_forward_dependency() -> None:
    for prefix in ("junk = observe()", "a = q\n        q = q_cache[k_lane]"):
        _assert_factored_ast_unchanged(
            f"""
k_cache = cute.make_rmem_tensor(8, cutlass.Float32)
q_cache = cute.make_rmem_tensor(8, cutlass.Float32)
for row_lane in range(16):
    dot_acc = cutlass.Float32(0)
    for k_lane in cutlass.range_constexpr(8):
        {prefix}
        k = k_cache[k_lane]
        product = k * q
        dot_acc = dot_acc + cutlass.Float32(product)
    _helion_factored_dot_sum = cutlass.Float32(
        cute.arch.warp_reduction_sum(dot_acc)
    )
    output = scalar[row_lane] * _helion_factored_dot_sum
"""
        )


def test_does_not_hoist_dynamic_warp_width_or_escaping_accumulator() -> None:
    for suffix in (
        "output = dot_acc",
        "output = scalar[row_lane] * _helion_factored_dot_sum",
    ):
        source = f"""
k_cache = cute.make_rmem_tensor(8, cutlass.Float32)
q_cache = cute.make_rmem_tensor(8, cutlass.Float32)
for row_lane in range(16):
    dot_acc = cutlass.Float32(0)
    for k_lane in cutlass.range_constexpr(8):
        product = k_cache[k_lane] * q_cache[k_lane]
        dot_acc = dot_acc + cutlass.Float32(product)
    _helion_factored_dot_sum = cutlass.Float32(
        cute.arch.warp_reduction_sum(dot_acc, threads_in_group=row_lane)
    )
    {suffix}
consume(dot_acc)
"""
        _assert_factored_ast_unchanged(source)


def test_does_not_hoist_untracked_arch_load() -> None:
    _assert_factored_ast_unchanged(
        """
for row_lane in range(16):
    dot_acc = cutlass.Float32(0)
    for k_lane in cutlass.range_constexpr(8):
        value = cute.arch.load(pointer + k_lane, cutlass.Float32)
        dot_acc = dot_acc + cutlass.Float32(value)
    _helion_factored_dot_sum = cutlass.Float32(
        cute.arch.warp_reduction_sum(dot_acc)
    )
    output = scalar[row_lane] * _helion_factored_dot_sum
"""
    )


def _packed_code(
    source: str,
    *,
    fast_math: bool = True,
    target_device_capability: tuple[int, int] | None = (10, 0),
    float_scalar_names: set[str] | frozenset[str] = frozenset(),
) -> str:
    body = ast.parse(source).body
    rewritten = pack_fp32_constexpr_loops(
        body,
        fast_math=fast_math,
        target_device_capability=target_device_capability,
        float_scalar_names=float_scalar_names,
    )
    return ast.unparse(ast.Module(body=rewritten, type_ignores=[]))


_PACKED_ROW_REDUCTION = """
left = cute.make_rmem_tensor(8, cutlass.Float32)
right = cute.make_rmem_tensor(8, cutlass.Float32)
for row_lane in cutlass.range_constexpr(16):
    row_value = row_base + row_lane
    dot_lo = cutlass.Float32(0)
    dot_hi = cutlass.Float32(0)
    for pair_lane in cutlass.range_constexpr(4):
        dead_index = pair_lane * 2
        dead_value = dead_index + offset
        left_lo = cutlass.Float32(left[pair_lane * 2])
        left_hi = cutlass.Float32(left[pair_lane * 2 + 1])
        right_lo = cutlass.Float32(right[pair_lane * 2])
        right_hi = cutlass.Float32(right[pair_lane * 2 + 1])
        dot_lo, dot_hi = cute.arch.fma_packed_f32x2(
            (cutlass.Float32(left_lo), cutlass.Float32(left_hi)),
            (cutlass.Float32(right_lo * scale), cutlass.Float32(right_hi * scale)),
            (cutlass.Float32(dot_lo), cutlass.Float32(dot_hi)),
        )
    dot_lane = cutlass.Float32(dot_lo) + cutlass.Float32(dot_hi)
    dot = cutlass.Float32(
        cute.arch.warp_reduction_sum(dot_lane, threads_in_group=16)
    )
    output = row_value + dot
"""


def _hoisted_packed_code(
    source: str = _PACKED_ROW_REDUCTION,
    *,
    fast_math: bool = True,
    marked: bool = True,
    rename_groups: dict[str, str] | None = None,
    extended: bool = False,
) -> tuple[str, list[ast.stmt]]:
    parsed = ast.parse(source)
    module = cast("ast.Module", convert(parsed)) if extended else parsed
    row_loop = next(
        node
        for statement in module.body
        for node in ast.walk(statement)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "row_lane"
    )
    if marked:
        setattr(row_loop, HELION_LANE_LOOP_VAR_ATTR, "row_lane")
    rewritten = hoist_packed_factored_reductions(
        module.body,
        fast_math=fast_math,
        rename_groups=rename_groups,
    )
    return ast.unparse(ast.Module(body=rewritten, type_ignores=[])), rewritten


def test_hoists_post_pack_row_invariant_reduction_dependency_slice() -> None:
    code, rewritten = _hoisted_packed_code(extended=True)
    row_loop = next(
        statement
        for statement in rewritten
        if isinstance(statement, ast.For)
        and isinstance(statement.target, ast.Name)
        and statement.target.id == "row_lane"
    )

    assert code.index("dot_lo = cutlass.Float32(0)") < code.index(
        "for row_lane in cutlass.range_constexpr(16)"
    )
    assert code.count("cute.arch.fma_packed_f32x2") == 1
    assert "dead_index" not in code
    assert "dead_value" not in code
    assert "output = row_value + dot" in ast.unparse(row_loop)
    assert getattr(row_loop, HELION_LANE_LOOP_VAR_ATTR) == "row_lane"
    assert all(isinstance(statement, ExtendedAST) for statement in rewritten)


def test_post_pack_reduction_hoist_accepts_profitable_two_row_loop() -> None:
    for loop in ("cutlass.range_constexpr(2)", "range(2)"):
        source = _PACKED_ROW_REDUCTION.replace("cutlass.range_constexpr(16)", loop)
        code, _rewritten = _hoisted_packed_code(source)

        assert code.index("dot_lo = cutlass.Float32(0)") < code.index(
            f"for row_lane in {loop}"
        )
        assert code.count("cute.arch.fma_packed_f32x2") == 1


def test_post_pack_reduction_hoist_requires_fast_math_mark_and_repetition() -> None:
    for source, fast_math, marked in (
        (_PACKED_ROW_REDUCTION, False, True),
        (_PACKED_ROW_REDUCTION, True, False),
        (
            _PACKED_ROW_REDUCTION.replace("range_constexpr(16)", "range_constexpr(1)"),
            True,
            True,
        ),
    ):
        original = ast.unparse(ast.parse(source))
        code, _rewritten = _hoisted_packed_code(
            source, fast_math=fast_math, marked=marked
        )
        assert code == original


def test_post_pack_reduction_hoist_rejects_row_varying_input() -> None:
    source = _PACKED_ROW_REDUCTION.replace(
        "right_lo * scale", "right_lo * scale + row_lane"
    )
    original = ast.unparse(ast.parse(source))

    code, _rewritten = _hoisted_packed_code(source)

    assert code == original


def test_post_pack_reduction_hoist_rejects_rebound_or_mutated_input() -> None:
    for source in (
        _PACKED_ROW_REDUCTION.replace(
            "row_value = row_base + row_lane",
            "row_value = row_base + row_lane\n    scale = row_lane",
        ),
        _PACKED_ROW_REDUCTION.replace(
            "output = row_value + dot",
            "left[0] = row_value\n    output = row_value + dot",
        ),
    ):
        original = ast.unparse(ast.parse(source))
        code, _rewritten = _hoisted_packed_code(source)
        assert code == original


def test_post_pack_reduction_hoist_requires_private_register_sources() -> None:
    for source in (
        _PACKED_ROW_REDUCTION.replace("left[pair_lane", "source[pair_lane"),
        _PACKED_ROW_REDUCTION.replace(
            "for row_lane in cutlass.range_constexpr(16):",
            "left_alias = left\nfor row_lane in cutlass.range_constexpr(16):",
        ).replace("left[pair_lane", "left_alias[pair_lane"),
    ):
        original = ast.unparse(ast.parse(source))
        code, _rewritten = _hoisted_packed_code(source)
        assert code == original


def test_post_pack_reduction_hoist_preserves_dead_effect_and_escaping_value() -> None:
    for source in (
        _PACKED_ROW_REDUCTION.replace(
            "dead_value = dead_index + offset", "dead_value = observe(dead_index)"
        ),
        _PACKED_ROW_REDUCTION.replace(
            "dead_value = dead_index + offset", "dead_value = operator.pow(0, -1)"
        ),
        _PACKED_ROW_REDUCTION.replace(
            "output = row_value + dot", "output = row_value + dot + dead_value"
        ),
    ):
        original = ast.unparse(ast.parse(source))
        code, _rewritten = _hoisted_packed_code(source)
        assert code == original


def test_post_pack_reduction_hoist_rejects_stale_result_and_local_rebind() -> None:
    for source in (
        _PACKED_ROW_REDUCTION.replace(
            "row_value = row_base + row_lane",
            "prior_dot = dot\n    row_value = row_base + row_lane",
        ),
        _PACKED_ROW_REDUCTION.replace(
            "row_value = row_base + row_lane",
            "del dot\n    row_value = row_base + row_lane",
        ),
        _PACKED_ROW_REDUCTION.replace(
            "left_hi = cutlass.Float32(left[pair_lane * 2 + 1])",
            "left_lo = cutlass.Float32(left[pair_lane * 2 + 1])",
        ),
    ):
        original = ast.unparse(ast.parse(source))
        code, _rewritten = _hoisted_packed_code(source)
        assert code == original


def test_post_pack_reduction_hoist_rejects_runtime_division_and_lane_alias() -> None:
    division = _PACKED_ROW_REDUCTION.replace(
        "left_lo = cutlass.Float32(left[pair_lane * 2])",
        "left_lo = cutlass.Float32(left[pair_lane * 2]) / divisor",
    )
    for source, rename_groups in (
        (division, None),
        (_PACKED_ROW_REDUCTION, {"scale": "row_lane"}),
    ):
        original = ast.unparse(ast.parse(source))
        code, _rewritten = _hoisted_packed_code(source, rename_groups=rename_groups)
        assert code == original


def test_post_pack_reduction_hoist_rejects_opaque_math_and_data_attribute() -> None:
    opaque_math = _PACKED_ROW_REDUCTION.replace(
        "right_lo * scale", "right_lo * cute.math.observe(scale)"
    )
    data_attribute = _PACKED_ROW_REDUCTION.replace(
        "row_value = row_base + row_lane",
        "holder.value = row_lane\n    row_value = row_base + row_lane",
    ).replace("right_lo * scale", "right_lo * holder.value")
    for source in (opaque_math, data_attribute):
        original = ast.unparse(ast.parse(source))
        code, _rewritten = _hoisted_packed_code(source)
        assert code == original


def test_post_pack_reduction_hoist_rejects_post_rename_name_collision() -> None:
    original = ast.unparse(ast.parse(_PACKED_ROW_REDUCTION))

    code, _rewritten = _hoisted_packed_code(rename_groups={"left_lo": "dead_index"})

    assert code == original


def test_post_pack_reduction_hoists_within_existing_control_flow() -> None:
    source = "if enabled:\n" + "\n".join(
        f"    {line}" if line else line for line in _PACKED_ROW_REDUCTION.splitlines()
    )

    code, _rewritten = _hoisted_packed_code(source)

    assert code.index("if enabled:") < code.index("dot_lo = cutlass.Float32(0)")
    assert code.index("dot_lo = cutlass.Float32(0)") < code.index(
        "for row_lane in cutlass.range_constexpr(16)"
    )


def test_post_pack_reduction_uses_earliest_safe_version_branch_position() -> None:
    declarations, row_loop = _PACKED_ROW_REDUCTION.strip().split(
        "for row_lane in cutlass.range_constexpr(16):", 1
    )
    optimized = ("for row_lane in cutlass.range_constexpr(16):" + row_loop).replace(
        "for row_lane in cutlass.range_constexpr(16):",
        "scalar_preload = load_scalar()\n"
        "cute.arch.cp_async_wait_group(2)\n"
        "initial_values = load_initial_values()\n"
        "for row_lane in cutlass.range_constexpr(16):",
    )
    source = (
        declarations
        + "if valid:\n"
        + "\n".join(f"    {line}" if line else line for line in optimized.splitlines())
        + "\nelse:\n"
        + "\n".join(
            f"    {line}" if line else line
            for line in (
                "for row_lane in cutlass.range_constexpr(16):" + row_loop
            ).splitlines()
        )
        + "\n"
    )
    original = ast.parse(source)
    original_branch = cast("ast.If", original.body[2])
    original_fallback = ast.dump(
        ast.Module(body=original_branch.orelse, type_ignores=[]),
        include_attributes=False,
    )

    code, rewritten = _hoisted_packed_code(source)
    version = cast("ast.If", rewritten[2])
    optimized_code = ast.unparse(ast.Module(body=version.body, type_ignores=[]))

    assert code.index("right = cute.make_rmem_tensor") < code.index(
        "dot_lo = cutlass.Float32(0)"
    )
    assert optimized_code.index("dot_lo = cutlass.Float32(0)") < optimized_code.index(
        "scalar_preload = load_scalar()"
    )
    assert optimized_code.index(
        "scalar_preload = load_scalar()"
    ) < optimized_code.index("cute.arch.cp_async_wait_group(2)")
    assert optimized_code.count("cute.arch.fma_packed_f32x2") == 1
    assert optimized_code.count("cute.arch.warp_reduction_sum") == 1
    assert (
        ast.dump(
            ast.Module(body=version.orelse, type_ignores=[]), include_attributes=False
        )
        == original_fallback
    )


def test_post_pack_reduction_hoist_rejects_branch_local_escape() -> None:
    declarations, row_loop = _PACKED_ROW_REDUCTION.strip().split(
        "for row_lane in cutlass.range_constexpr(16):", 1
    )
    source = (
        declarations
        + "if valid:\n"
        + "\n".join(
            f"    {line}" if line else line
            for line in (
                "for row_lane in cutlass.range_constexpr(16):" + row_loop
            ).splitlines()
        )
        + "\nelse:\n    fallback = exact_masked_path()\n"
        + "consume(dead_value)\n"
    )
    original = ast.unparse(ast.parse(source))

    code, _rewritten = _hoisted_packed_code(source)

    assert code == original


def test_post_pack_reduction_hoist_does_not_enter_exception_region() -> None:
    source = (
        "dot_lo = cutlass.Float32(9)\n"
        "try:\n"
        "    dangerous()\n"
        + "\n".join(
            f"    {line}" if line else line
            for line in _PACKED_ROW_REDUCTION.splitlines()
        )
        + "\nexcept Exception:\n    consume(dot_lo)\n"
    )
    original = ast.unparse(ast.parse(source))

    code, _rewritten = _hoisted_packed_code(source)

    assert code == original


_FP32_REDUCTION = """
cache = cute.make_rmem_tensor(8, cutlass.BFloat16)
acc = cutlass.Float32(0)
for lane in cutlass.range_constexpr(8):
    value = cutlass.Float32(cache[lane])
    product = value * cutlass.Float32(weight[lane])
    acc = acc + cutlass.Float32(product)
result = cute.arch.warp_reduction_sum(acc)
"""


def test_packs_even_fp32_reduction_on_blackwell() -> None:
    code = _packed_code(_FP32_REDUCTION)

    assert "cutlass.range_constexpr(4)" in code
    assert code.count("cute.arch.fma_packed_f32x2") == 1
    assert "cache[_factored_pair_lane_0 * 2]" in code
    assert "cache[_factored_pair_lane_0 * 2 + 1]" in code
    assert "acc = cutlass.Float32(_acc_lo_0) + cutlass.Float32(_acc_hi_0)" in code


def test_packs_fused_fp32_reductions_and_lane_local_store() -> None:
    code = _packed_code(
        """
cache = cute.make_rmem_tensor(8, cutlass.BFloat16)
first_acc = cutlass.Float32(0)
second_acc = cutlass.Float32(0)
for lane in cutlass.range_constexpr(8):
    raw = source[lane]
    cache[lane] = raw
    value = cutlass.Float32(raw) * cutlass.Float32(decay[lane])
    first_product = value * cutlass.Float32(lhs[lane])
    first_acc = first_acc + cutlass.Float32(first_product)
    second_product = value * cutlass.Float32(rhs[lane])
    second_acc = second_acc + cutlass.Float32(second_product)
first = cute.arch.warp_reduction_sum(first_acc)
second = cute.arch.warp_reduction_sum(second_acc)
"""
    )

    assert code.count("cute.arch.fma_packed_f32x2") == 2
    assert "cache[_factored_pair_lane_0 * 2]" in code
    assert "cache[_factored_pair_lane_0 * 2 + 1]" in code
    assert "first_acc = cutlass.Float32(_first_acc_lo_0)" in code
    assert "second_acc = cutlass.Float32(_second_acc_lo_0)" in code


def test_packs_fp32_update_ending_in_append() -> None:
    code = _packed_code(
        """
cache = cute.make_rmem_tensor(8, cutlass.BFloat16)
values = []
for lane in cutlass.range_constexpr(8):
    state = cutlass.Float32(cache[lane]) * cutlass.Float32(decay[lane])
    product = cutlass.Float32(query[lane]) * scale
    update = state + product
    values.append(cutlass.BFloat16(update))
consume(values)
""",
        float_scalar_names={"scale"},
    )

    assert "cutlass.range_constexpr(4)" in code
    assert "cute.arch.fma_packed_f32x2" in code
    assert code.count("values.append") == 2
    assert code.index(
        "values.append(cutlass.BFloat16(_factored_update_lo_0))"
    ) < code.index("values.append(cutlass.BFloat16(_factored_update_hi_0))")


def test_packed_fp32_loop_requires_fast_math_and_blackwell() -> None:
    original = ast.unparse(ast.parse(_FP32_REDUCTION))

    assert _packed_code(_FP32_REDUCTION, fast_math=False) == original
    assert _packed_code(_FP32_REDUCTION, target_device_capability=(9, 0)) == original
    assert _packed_code(_FP32_REDUCTION, target_device_capability=None) == original


def test_does_not_pack_odd_width_fp32_loop() -> None:
    source = _FP32_REDUCTION.replace("range_constexpr(8)", "range_constexpr(7)")

    assert _packed_code(source) == ast.unparse(ast.parse(source))


def test_does_not_pack_loop_with_unknown_effect() -> None:
    source = """
cache = cute.make_rmem_tensor(8, cutlass.BFloat16)
acc = cutlass.Float32(0)
for lane in cutlass.range_constexpr(8):
    value = cutlass.Float32(cache[lane])
    observe(value)
    acc = acc + value
result = cute.arch.warp_reduction_sum(acc)
"""

    assert _packed_code(source) == ast.unparse(ast.parse(source))


def test_does_not_pack_store_through_aliased_fragment() -> None:
    source = """
cache = cute.make_rmem_tensor(8, cutlass.BFloat16)
alias = cache
acc = cutlass.Float32(0)
for lane in cutlass.range_constexpr(8):
    value = cutlass.Float32(source[lane])
    cache[lane] = value
    acc = acc + value
result = cute.arch.warp_reduction_sum(acc)
"""

    assert _packed_code(source) == ast.unparse(ast.parse(source))


def test_does_not_pack_store_with_noninjective_lane_index() -> None:
    source = """
cache = cute.make_rmem_tensor(8, cutlass.Float32)
acc = cutlass.Float32(0)
for lane in cutlass.range_constexpr(8):
    value = cutlass.Float32(source[lane])
    cache[lane % 2] = value
    acc = acc + value
result = cute.arch.warp_reduction_sum(acc)
"""

    assert _packed_code(source) == ast.unparse(ast.parse(source))


def test_does_not_pack_overlapping_affine_store_mappings() -> None:
    source = """
cache = cute.make_rmem_tensor(9, cutlass.Float32)
first_acc = cutlass.Float32(0)
second_acc = cutlass.Float32(0)
for lane in cutlass.range_constexpr(8):
    first = cutlass.Float32(source[lane])
    second = cutlass.Float32(other[lane])
    cache[lane] = first
    cache[lane + 1] = second
    first_acc = first_acc + first
    second_acc = second_acc + second
consume(first_acc, second_acc, cache)
"""

    assert _packed_code(source) == ast.unparse(ast.parse(source))


def test_does_not_assume_loop_local_index_component_is_invariant() -> None:
    source = """
cache = cute.make_rmem_tensor(8, cutlass.Float32)
acc = cutlass.Float32(0)
for lane in cutlass.range_constexpr(8):
    offset = -lane
    value = cutlass.Float32(source[lane])
    cache[lane + offset] = value
    acc = acc + value
result = cute.arch.warp_reduction_sum(acc)
"""

    assert _packed_code(source) == ast.unparse(ast.parse(source))


def test_does_not_pack_cross_lane_fragment_dependency() -> None:
    source = """
cache = cute.make_rmem_tensor(8, cutlass.Float32)
acc = cutlass.Float32(0)
for lane in cutlass.range_constexpr(8):
    cache[lane] = cutlass.Float32(source[lane])
    value = cache[lane - 1]
    acc = acc + value
result = cute.arch.warp_reduction_sum(acc)
"""

    assert _packed_code(source) == ast.unparse(ast.parse(source))


def test_does_not_pack_loop_local_value_used_after_loop() -> None:
    source = """
cache = cute.make_rmem_tensor(8, cutlass.Float32)
acc = cutlass.Float32(0)
for lane in cutlass.range_constexpr(8):
    value = cache[lane]
    acc = acc + value
result = value + cute.arch.warp_reduction_sum(acc)
"""

    assert _packed_code(source) == ast.unparse(ast.parse(source))


def test_does_not_pack_conditionally_overwritten_loop_local() -> None:
    source = """
cache = cute.make_rmem_tensor(8, cutlass.Float32)
acc = cutlass.Float32(0)
for lane in cutlass.range_constexpr(8):
    value = cache[lane]
    acc = acc + value
if condition:
    value = cutlass.Float32(0)
consume(value, acc)
"""

    assert _packed_code(source) == ast.unparse(ast.parse(source))


def test_does_not_pack_loop_local_escaping_enclosing_branch() -> None:
    source = """
cache = cute.make_rmem_tensor(8, cutlass.Float32)
if condition:
    acc = cutlass.Float32(0)
    for lane in cutlass.range_constexpr(8):
        value = cache[lane]
        acc = acc + value
else:
    value = cutlass.Float32(0)
consume(value)
"""

    assert _packed_code(source) == ast.unparse(ast.parse(source))


def test_does_not_pack_loop_local_used_on_enclosing_loop_backedge() -> None:
    source = """
value = cutlass.Float32(7)
for outer in range(4):
    consume(value)
    cache = cute.make_rmem_tensor(8, cutlass.Float32)
    acc = cutlass.Float32(0)
    for lane in cutlass.range_constexpr(8):
        value = cache[lane]
        acc = acc + value
"""

    assert _packed_code(source) == ast.unparse(ast.parse(source))


def test_does_not_pack_mixed_fp32_float64_expression() -> None:
    source = """
left = cute.make_rmem_tensor(8, cutlass.Float32)
right = cute.make_rmem_tensor(8, cutlass.Float64)
acc = cutlass.Float32(0)
for lane in cutlass.range_constexpr(8):
    product = left[lane] * right[lane]
    acc = acc + product
consume(acc)
"""

    assert _packed_code(source) == ast.unparse(ast.parse(source))


def test_does_not_pack_fp32_expression_with_int64_operand() -> None:
    source = """
cache = cute.make_rmem_tensor(8, cutlass.Float32)
acc = cutlass.Float32(0)
for lane in cutlass.range_constexpr(8):
    wide = cutlass.Int64(source[lane])
    product = cache[lane] * wide
    acc = acc + cutlass.Float32(product)
consume(acc)
"""

    packed = _packed_code(source)

    assert "fma_packed_f32x2" not in packed
    assert "mul_packed_f32x2" not in packed
    assert packed.count("cutlass.Int64") == 2
    assert "add_packed_f32x2" in packed


def test_does_not_trust_rebound_float_scalar_fact() -> None:
    source = """
cache = cute.make_rmem_tensor(8, cutlass.Float32)
scale, other = pair
values = []
for lane in cutlass.range_constexpr(8):
    product = cache[lane] * scale
    values.append(product)
consume(values)
"""

    assert _packed_code(source, float_scalar_names={"scale"}) == ast.unparse(
        ast.parse(source)
    )


def test_does_not_pack_append_with_nested_effect() -> None:
    source = """
cache = cute.make_rmem_tensor(8, cutlass.Float32)
values = []
for lane in cutlass.range_constexpr(8):
    value = cache[lane] * scale
    values.append(convert_and_record(value))
consume(values)
"""

    assert _packed_code(source) == ast.unparse(ast.parse(source))


def test_packed_fp32_loop_is_idempotent() -> None:
    body = ast.parse(_FP32_REDUCTION).body
    once = pack_fp32_constexpr_loops(
        body,
        fast_math=True,
        target_device_capability=(10, 0),
    )
    once_code = ast.unparse(ast.Module(body=once, type_ignores=[]))

    twice = pack_fp32_constexpr_loops(
        once,
        fast_math=True,
        target_device_capability=(10, 0),
    )

    assert ast.unparse(ast.Module(body=twice, type_ignores=[])) == once_code
