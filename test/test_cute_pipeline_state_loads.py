from __future__ import annotations

import ast
import textwrap
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch
from torch._dynamo.source import LocalSource
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute.memory_ops import (
    _PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY,
)
from helion._compiler.cute.memory_ops import _persistent_vec_alignment_matrix_signature
from helion._compiler.cute.memory_ops import runtime_tensor_has_specialized_alignment
from helion._compiler.cute.pipeline_state_loads import TensorMetadata
from helion._compiler.cute.pipeline_state_loads import pipeline_state_loads
from helion._compiler.device_function import DeviceFunction
from helion._compiler.device_function import TensorArg
from helion._compiler.device_function import TensorSizeArg

if TYPE_CHECKING:
    from collections.abc import Callable

SOURCE = """
tile = cutlass.Int32(0)
batch = cutlass.Int32(0)
head = cutlass.Int32(0)
valid = batch < state_size
column_base = cutlass.Int32(cute.arch.thread_idx()[0]) * 8
independent = expensive_prepare()
for lane in range(16):
    row = tile + cutlass.Int32(cute.arch.thread_idx()[1]) * 16 + cutlass.Int32(lane)
    values = cute.arch.load(
        state.iterator
        + (
            cutlass.Int32(batch) * cutlass.Int32(state.layout.stride[0])
            + cutlass.Int32(head) * cutlass.Int32(state.layout.stride[1])
            + cutlass.Int32(row) * cutlass.Int32(state.layout.stride[2])
            + cutlass.Int32(column_base) * cutlass.Int32(state.layout.stride[3])
            if valid
            else cutlass.Int32(0)
        ),
        ir.VectorType.get([8], cutlass.Uint16.mlir_type),
    )
    updated = update(values, independent)
    store_values = []
    for element in cutlass.range_constexpr(8):
        scalar = updated[element]
        store_values.append(scalar)
    if valid:
        _cute_store_u16_vec(
            state.iterator
            + cutlass.Int32(batch) * cutlass.Int32(state.layout.stride[0])
            + cutlass.Int32(head) * cutlass.Int32(state.layout.stride[1])
            + cutlass.Int32(row) * cutlass.Int32(state.layout.stride[2])
            + cutlass.Int32(column_base) * cutlass.Int32(state.layout.stride[3]),
            store_values,
        )
    else:
        pass
"""


def _metadata(*, include_other: bool = False) -> dict[str, TensorMetadata]:
    result = {"state": TensorMetadata("cutlass.BFloat16", (264, 12, 128, 128))}
    if include_other:
        result["other"] = TensorMetadata("cutlass.BFloat16", (264, 12, 128, 128))
    return result


def _strides() -> dict[tuple[str, int], int]:
    return {
        ("state", 0): 196864,
        ("state", 1): 16384,
        ("state", 2): 128,
        ("state", 3): 1,
    }


def _packed_scalar_source(*, load_count: int = 2) -> str:
    prefix = "aux_base = cutlass.Int32(cute.arch.block_idx()[2]) * _AUX_TILE\n"
    scalar_load = """    aux_index = aux_base + row
    aux_value = (
        (
            aux.iterator
            + cutlass.Int32(batch) * cutlass.Int32(aux.layout.stride[0])
            + cutlass.Int32(aux_index) * cutlass.Int32(aux.layout.stride[1])
        ).load()
        if cutlass.Int32(aux_index) < aux_size
        else cutlass.BFloat16(0)
    )
    aux_use = cutlass.Float32(aux_value)
"""
    source = SOURCE.replace(
        "    updated = update(values, independent)",
        scalar_load * load_count + "    updated = values",
    )
    return prefix + source


def _packed_scalar_metadata(
    *, include_other: bool = False, aux_dtype: str = "cutlass.BFloat16"
) -> dict[str, TensorMetadata]:
    result = {
        **_metadata(),
        "aux": TensorMetadata(aux_dtype, (264, 4096)),
    }
    if include_other:
        result["other"] = TensorMetadata("cutlass.BFloat16", (264, 4096))
    return result


def _packed_scalar_strides(*, last_stride: int = 1) -> dict[tuple[str, int], int]:
    return {
        **_strides(),
        ("aux", 0): 4096,
        ("aux", 1): last_stride,
    }


def _packed_scalar_disjoint(*, include_other: bool = False) -> set[frozenset[str]]:
    result = {frozenset(("state", "aux"))}
    if include_other:
        result.add(frozenset(("state", "other")))
    return result


def _transform(
    source: str = SOURCE,
    *,
    metadata: dict[str, TensorMetadata] | None = None,
    stages: int = 5,
    lookahead: int = 4,
    group_rows: int = 2,
    cache_policy: str = "cg",
    store_policy: str = "default",
    thread_block_dims: tuple[int, int, int] = (16, 8, 1),
    tensor_names: frozenset[str] | None = None,
    proven_tensor_base_alignments: frozenset[str] = frozenset(),
    proven_tensor_size_values: dict[tuple[str, int], tuple[str, int]] | None = None,
    proven_disjoint_tensor_pairs: set[frozenset[str]] | None = None,
    proven_tensor_stride_values: dict[tuple[str, int], int] | None = None,
    block_grid_dims: tuple[int | None, int | None, int | None] = (None, None, None),
    constexpr_values: dict[str, int] | None = None,
    uniform_names: frozenset[str] = frozenset(("state_size",)),
    target_device_capability: tuple[int, int] | None = (10, 0),
) -> str:
    body = ast.parse(source).body
    metadata = _metadata() if metadata is None else metadata
    pipeline_state_loads(
        body,
        stages=stages,
        lookahead=lookahead,
        group_rows=group_rows,
        cache_policy=cache_policy,
        store_policy=store_policy,
        thread_block_dims=thread_block_dims,
        tensor_metadata=metadata,
        tensor_names=(frozenset(metadata) if tensor_names is None else tensor_names),
        proven_tensor_base_alignments=proven_tensor_base_alignments,
        proven_tensor_size_values=(
            {} if proven_tensor_size_values is None else proven_tensor_size_values
        ),
        proven_disjoint_tensor_pairs=(
            set()
            if proven_disjoint_tensor_pairs is None
            else proven_disjoint_tensor_pairs
        ),
        proven_tensor_stride_values=(
            _strides()
            if proven_tensor_stride_values is None
            else proven_tensor_stride_values
        ),
        block_grid_dims=block_grid_dims,
        constexpr_values={} if constexpr_values is None else constexpr_values,
        uniform_names=uniform_names,
        target_device_capability=target_device_capability,
    )
    return ast.unparse(
        ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    )


def _assert_no_match(mutator: Callable[[str], str], **kwargs: object) -> None:
    source = mutator(SOURCE)
    assert _transform(source, **kwargs) == ast.unparse(ast.parse(source))


def _replace_last(source: str, old: str, new: str) -> str:
    prefix, separator, suffix = source.rpartition(old)
    assert separator
    return prefix + new + suffix


def _add_observable_other_store(source: str) -> str:
    return source.replace(
        "    store_values = []",
        "    (other.iterator + cutlass.Int32(row) * "
        "cutlass.Int32(other.layout.stride[0]) + "
        "cutlass.Int32(column_base)).store(values[0])\n"
        "    store_values = []",
    )


def _scalar_index_mask_source() -> str:
    return SOURCE.replace(
        "tile = cutlass.Int32(0)\n"
        "batch = cutlass.Int32(0)\n"
        "head = cutlass.Int32(0)\n"
        "valid = batch < state_size\n"
        "column_base = cutlass.Int32(cute.arch.thread_idx()[0]) * 8",
        "tile = cutlass.Int32(0)\n"
        "index_offset = cutlass.Int32(cute.arch.block_idx()[0])\n"
        "index_load = (\n"
        "    indices.iterator\n"
        "    + index_offset * cutlass.Int32(indices.layout.stride[0])\n"
        ").load()\n"
        "batch = cutlass.Int32(index_load)\n"
        "head = cutlass.Int32(0)\n"
        "column_base = cutlass.Int32(cute.arch.thread_idx()[0]) * 8\n"
        "valid = (\n"
        "    batch < state_size\n"
        "    and cutlass.Int32(column_base + 0) < 128\n"
        "    and cutlass.Int32(column_base + 7) < 128\n"
        ")",
    )


def _scalar_index_metadata() -> dict[str, TensorMetadata]:
    return {
        "state": TensorMetadata("cutlass.BFloat16", (264, 12, 128, 128)),
        "indices": TensorMetadata("cutlass.Int32", (264,)),
    }


def _scalar_index_strides() -> dict[tuple[str, int], int]:
    return {**_strides(), ("indices", 0): 1}


def test_pipelines_exact_private_vector_state_region() -> None:
    result = _transform()

    assert "cutlass.Array(cutlass.Uint16, 10240" in result
    assert result.count("cute.arch.cp_async_shared_global(") == 3
    assert result.count("cute.arch.cp_async_commit_group()") == 3
    assert result.count("cute.arch.cp_async_wait_group(") == 4
    assert "cute.arch.cp_async_wait_group(2)" in result
    assert "cute.arch.cp_async_wait_group(1)" in result
    assert "cute.arch.cp_async_wait_group(0)" in result
    assert "cute.arch.load(state.iterator +" not in result
    assert "_cute_store_u16_vec(state.iterator +" in result
    assert "updated = update(values, independent)" in result
    assert "for lane in cutlass.range_constexpr(16):" in result
    assert "syncthreads" not in result


def test_l2_evict_last_rewrites_only_the_proven_state_store() -> None:
    with patch("helion._compiler.cute.pipeline_state_loads.torch.version.cuda", "13.0"):
        result = _transform(
            store_policy="l2_evict_last", target_device_capability=(10, 3)
        )

    assert result.count("_cute_store_u16x8_l2_evict_last(") == 1
    assert result.count("_async_state_0_l2_store_abi_version = 1") == 1
    assert "_cute_store_u16_vec(state.iterator +" not in result


def test_unknown_store_policy_fails_closed() -> None:
    assert _transform(store_policy="unsupported") == ast.unparse(ast.parse(SOURCE))


def test_fixed_l2_policy_fails_closed_on_unvalidated_architecture() -> None:
    assert _transform(
        store_policy="l2_evict_last", target_device_capability=(10, 0)
    ) == ast.unparse(ast.parse(SOURCE))


def test_fixed_l2_policy_fails_closed_on_unvalidated_toolchain() -> None:
    with patch("helion._compiler.cute.pipeline_state_loads.torch.version.cuda", "12.8"):
        assert _transform(
            store_policy="l2_evict_last", target_device_capability=(10, 3)
        ) == ast.unparse(ast.parse(SOURCE))


def test_only_the_exact_matched_static_row_loop_is_unrolled() -> None:
    source = (
        "for unrelated_before in range(3):\n"
        "    unrelated_value = unrelated_before\n"
        + SOURCE
        + "\nfor unrelated_after in range(5):\n"
        "    unrelated_value = unrelated_after\n"
    )
    result = _transform(source)

    assert "for unrelated_before in range(3):" in result
    assert "for lane in cutlass.range_constexpr(16):" in result
    assert "for unrelated_after in range(5):" in result
    assert result.count("for lane in cutlass.range_constexpr(16):") == 1


@pytest.mark.parametrize(
    "loop_header",
    (
        "for lane in range(row_count):",
        "for lane in cutlass.range_constexpr(16):",
        "for lane in range(1, 17):",
    ),
)
def test_noncanonical_or_nonliteral_row_loop_is_not_unrolled(
    loop_header: str,
) -> None:
    source = SOURCE.replace("for lane in range(16):", loop_header, 1)

    assert _transform(source) == ast.unparse(ast.parse(source))


def test_canonical_vector_type_is_admitted() -> None:
    assert "cute.arch.cp_async_shared_global(" in _transform()


@pytest.mark.parametrize(
    "vector_type_suffix",
    (", vector_scalability", ", scalable=vector_scalability"),
)
def test_vector_type_extra_dependency_fails_closed(
    vector_type_suffix: str,
) -> None:
    source = SOURCE.replace(
        "    values = cute.arch.load(",
        "    vector_scalability = [False]\n    values = cute.arch.load(",
        1,
    ).replace(
        "ir.VectorType.get([8], cutlass.Uint16.mlir_type)",
        f"ir.VectorType.get([8], cutlass.Uint16.mlir_type{vector_type_suffix})",
        1,
    )

    assert _transform(source) == ast.unparse(ast.parse(source))


@pytest.mark.parametrize("capability", (None, (7, 0), (7, 5)))
def test_codegen_requires_sm80_or_newer(
    capability: tuple[int, int] | None,
) -> None:
    assert _transform(target_device_capability=capability) == ast.unparse(
        ast.parse(SOURCE)
    )


def test_codegen_accepts_sm80() -> None:
    assert "cute.arch.cp_async_shared_global(" in _transform(
        target_device_capability=(8, 0)
    )


def test_shared_addresses_are_coalesced_row_major_and_private() -> None:
    result = _transform()

    assert (
        "_async_state_0_row_thread = cutlass.Int32(cute.arch.thread_idx()[1])" in result
    )
    assert (
        "_async_state_0_column = cutlass.Int32(cute.arch.thread_idx()[0]) * 8" in result
    )
    assert "(_async_state_0_row_thread * 2 + _async_state_0_row_local) * 128" in result
    assert "_async_state_0_thread" not in result

    def shared_offset(stage: int, x: int, y: int, row_local: int) -> int:
        return stage * 2048 + (y * 2 + row_local) * 128 + x * 8

    assert shared_offset(0, 1, 0, 0) - shared_offset(0, 0, 0, 0) == 8
    addresses = {
        shared_offset(stage, x, y, row_local)
        for stage in range(5)
        for y in range(8)
        for row_local in range(2)
        for x in range(16)
    }
    assert len(addresses) == 5 * 8 * 2 * 16


def test_stage_group_and_cache_knobs_change_the_schedule() -> None:
    result = _transform(stages=4, lookahead=3, group_rows=4, cache_policy="ca")

    assert "cutlass.Array(cutlass.Uint16, 16384" in result
    assert "cutlass.range_constexpr(3)" in result
    assert "cutlass.range_constexpr(4)" in result
    assert ", 16, 'ca')" in result


@pytest.mark.parametrize(
    "kwargs",
    (
        {"stages": 2},
        {"stages": 4, "lookahead": 4},
        {"group_rows": 3},
        {"cache_policy": "cs"},
        {"thread_block_dims": (8, 8, 1)},
    ),
)
def test_invalid_or_partial_schedules_are_noops(kwargs: dict[str, object]) -> None:
    assert _transform(**kwargs) == ast.unparse(ast.parse(SOURCE))


def test_missing_runtime_disjointness_proof_fails_closed() -> None:
    _assert_no_match(
        lambda source: source,
        metadata=_metadata(include_other=True),
        tensor_names=frozenset(("state", "other")),
    )


def test_cache_specialized_runtime_disjointness_proof_is_consumed() -> None:
    result = _transform(
        metadata=_metadata(include_other=True),
        tensor_names=frozenset(("state", "other")),
        proven_disjoint_tensor_pairs={frozenset(("state", "other"))},
    )

    assert "cute.arch.cp_async_shared_global(" in result


def test_runtime_alias_reclassification_cannot_reuse_disjoint_plan() -> None:
    metadata = _metadata(include_other=True)
    tensor_names = frozenset(("state", "other"))
    disjoint = _transform(
        metadata=metadata,
        tensor_names=tensor_names,
        proven_disjoint_tensor_pairs={frozenset(("state", "other"))},
    )
    overlapping = _transform(
        metadata=metadata,
        tensor_names=tensor_names,
        proven_disjoint_tensor_pairs=set(),
    )

    assert "cute.arch.cp_async_shared_global(" in disjoint
    assert "cute.arch.cp_async_shared_global(" not in overlapping


@pytest.mark.parametrize(
    "strides",
    (
        {
            ("state", 0): 196864,
            ("state", 1): 16384,
            ("state", 2): 128,
        },
        {
            ("state", 0): 196864,
            ("state", 1): 16384,
            ("state", 2): 1,
            ("state", 3): 128,
        },
    ),
)
def test_missing_or_incompatible_exact_stride_fails_closed(
    strides: dict[tuple[str, int], int],
) -> None:
    _assert_no_match(
        lambda source: source,
        proven_tensor_stride_values=strides,
    )


@pytest.mark.parametrize(
    "predicate",
    (
        "row != 1",
        "lane != 1",
        "cutlass.Int32(cute.arch.thread_idx()[0]) != 1",
        "cutlass.Int32(cute.arch.block_idx()[0]) != 1",
    ),
)
def test_nonuniform_masked_fallback_fails_closed(predicate: str) -> None:
    _assert_no_match(lambda source: source.replace("if valid", f"if {predicate}"))


def test_async_pointer_drops_statically_true_vector_column_bounds() -> None:
    predicate = (
        "batch < state_size "
        "and cutlass.Int32(column_base + 0) < 128 "
        "and cutlass.Int32(column_base + 7) < 128"
    )
    result = _transform(SOURCE.replace("if valid", f"if {predicate}"))
    module = ast.parse(result)
    async_calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "cp_async_shared_global"
    ]

    assert len(async_calls) == 3
    for call in async_calls:
        conditions = [
            node.test for node in ast.walk(call.args[1]) if isinstance(node, ast.IfExp)
        ]
        assert len(conditions) == 1
        assert ast.unparse(conditions[0]) == "0 < state_size"


def test_rebound_column_bound_alias_fails_closed() -> None:
    source = SOURCE.replace(
        "column_base = cutlass.Int32(cute.arch.thread_idx()[0]) * 8",
        "column_base = cutlass.Int32(cute.arch.thread_idx()[0]) * 8\n"
        "column_guard = column_base + 7",
        1,
    ).replace(
        "    values = cute.arch.load(",
        "    column_guard = cutlass.Int32(1000)\n    values = cute.arch.load(",
        1,
    )
    source = _add_observable_other_store(
        source.replace("if valid", "if batch < state_size and column_guard < 128")
    )

    _assert_no_match(
        lambda _: source,
        metadata=_metadata(include_other=True),
        tensor_names=frozenset(("state", "other")),
        proven_disjoint_tensor_pairs={frozenset(("state", "other"))},
    )


def test_sibling_branch_column_aliases_still_strip_bounds() -> None:
    predicate = (
        "batch < state_size "
        "and cutlass.Int32(column_base + 0) < 128 "
        "and cutlass.Int32(column_base + 7) < 128"
    )
    branch = SOURCE.replace("if valid", f"if {predicate}")
    prefix, branch_body = branch.split(
        "column_base = cutlass.Int32(cute.arch.thread_idx()[0]) * 8\n", 1
    )
    branch_body = (
        "column_base = cutlass.Int32(cute.arch.thread_idx()[0]) * 8\n" + branch_body
    )
    sibling = "column_base = cutlass.Int32(cute.arch.thread_idx()[0]) * 8\n"
    source = prefix + f"if choose_first:\n{textwrap.indent(sibling, '    ')}"
    source += f"else:\n{textwrap.indent(branch_body, '    ')}"
    result = _transform(source)
    module = ast.parse(result)
    async_calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "cp_async_shared_global"
    ]

    assert len(async_calls) == 3
    for call in async_calls:
        conditions = [
            node.test for node in ast.walk(call.args[1]) if isinstance(node, ast.IfExp)
        ]
        assert len(conditions) == 1
        assert ast.unparse(conditions[0]) == "0 < state_size"


def test_row_varying_mask_alias_fails_closed() -> None:
    def mutate(source: str) -> str:
        source = source.replace(
            "    values = cute.arch.load(",
            "    row_mask = row != 1\n    values = cute.arch.load(",
        )
        return source.replace("if valid", "if row_mask")

    _assert_no_match(mutate)


@pytest.mark.parametrize(
    "tile_expression",
    (
        "cutlass.Int32(0 - (cute.arch.lane_idx() // 16) * 15)",
        "cutlass.Int32(cute.arch.warp_idx())",
        "opaque_index()",
    ),
)
def test_row_partition_requires_positive_cta_uniform_base(
    tile_expression: str,
) -> None:
    _assert_no_match(
        lambda source: source.replace(
            "tile = cutlass.Int32(0)", f"tile = {tile_expression}", 1
        )
    )


@pytest.mark.parametrize(
    "head_expression",
    (
        "cutlass.Int32(cute.arch.lane_idx())",
        "cutlass.Int32(cute.arch.warp_idx())",
        "opaque_index()",
    ),
)
def test_nonpartition_coordinate_requires_positive_cta_uniformity(
    head_expression: str,
) -> None:
    _assert_no_match(
        lambda source: source.replace(
            "head = cutlass.Int32(0)", f"head = {head_expression}", 1
        )
    )


def test_row_partition_accepts_block_and_launch_uniform_base() -> None:
    source = SOURCE.replace(
        "tile = cutlass.Int32(0)",
        "tile = cutlass.Int32(cute.arch.block_idx()[0]) * 0 + tile_base",
        1,
    )
    result = _transform(
        source,
        uniform_names=frozenset(("state_size", "tile_base")),
    )

    assert "cute.arch.cp_async_shared_global(" in result


@pytest.mark.parametrize("exit_statement", ("return", "raise RuntimeError"))
def test_prefetch_prologue_does_not_cross_an_early_exit(
    exit_statement: str,
) -> None:
    source = SOURCE.replace(
        "independent = expensive_prepare()",
        f"if stop:\n    {exit_statement}\nindependent = expensive_prepare()",
        1,
    )
    result = _transform(
        source,
        uniform_names=frozenset(("state_size", "stop")),
    )

    assert "cute.arch.cp_async_shared_global(" in result
    assert result.index("if stop:") < result.index("independent = expensive_prepare()")
    assert result.index("independent = expensive_prepare()") < result.index(
        "_async_state_0_row_thread ="
    )


def test_pointer_dependencies_dominate_the_prefetch_prologue() -> None:
    source = SOURCE.replace(
        "cutlass.Int32(head) * cutlass.Int32(state.layout.stride[1])",
        "cutlass.Int32(state_head) * cutlass.Int32(state.layout.stride[1])",
    ).replace(
        "independent = expensive_prepare()",
        "state_head = cutlass.Int32(head)\nindependent = expensive_prepare()",
        1,
    )
    result = _transform(source)

    assert "cute.arch.cp_async_shared_global(" in result
    assert result.index("state_head = cutlass.Int32(head)") < result.index(
        "_async_state_0_row_thread ="
    )


def test_prefetch_prologue_overlaps_generated_register_preparation() -> None:
    source = ("cache = cute.make_rmem_tensor(8, cutlass.BFloat16)\n" + SOURCE).replace(
        "independent = expensive_prepare()",
        "independent = cutlass.Float32(0)\n"
        "other_values = cute.arch.load(\n"
        "    other.iterator + cutlass.Int32(0),\n"
        "    ir.VectorType.get([8], cutlass.Uint16.mlir_type),\n"
        ")\n"
        "for vector_lane in cutlass.range_constexpr(8):\n"
        "    other_value = cutlass.Uint16(\n"
        "        other_values[vector_lane]\n"
        "    ).bitcast(cutlass.BFloat16)\n"
        "    cache[vector_lane] = other_value\n"
        "    independent = independent + cutlass.Float32(other_value)\n"
        "independent = cutlass.Float32(\n"
        "    cute.arch.warp_reduction_sum(independent, threads_in_group=16)\n"
        ")\n"
        "independent = cute.math.rsqrt(independent + 1e-06)",
        1,
    )
    result = _transform(
        source,
        metadata=_metadata(include_other=True),
        tensor_names=frozenset(("state", "other")),
        proven_disjoint_tensor_pairs={frozenset(("state", "other"))},
    )

    column = result.index("column_base =")
    prologue = result.index("_async_state_0_row_thread =")
    preparation = result.index("independent = cutlass.Float32(0)")
    wait = result.index("cute.arch.cp_async_wait_group(")
    state_loop = result.index("for lane in cutlass.range_constexpr(16):")
    assert column < prologue < preparation < wait < state_loop


@pytest.mark.parametrize(
    "opaque_expression",
    (
        "cute.math.opaque_effect()",
        "cute.math.effects.store()",
        "opaque.bitcast(cutlass.BFloat16)",
    ),
)
def test_prefetch_prologue_stays_after_unknown_call_but_crosses_later_math(
    opaque_expression: str,
) -> None:
    source = SOURCE.replace(
        "independent = expensive_prepare()",
        f"opaque_value = {opaque_expression}\n"
        "independent = cute.math.rsqrt(cutlass.Float32(4))",
        1,
    )
    result = _transform(source)

    assert result.index(f"opaque_value = {opaque_expression}") < result.index(
        "_async_state_0_row_thread ="
    )
    assert result.index("_async_state_0_row_thread =") < result.index(
        "independent = cute.math.rsqrt"
    )


def test_prefetch_prologue_stays_after_latest_pointer_dependency() -> None:
    source = SOURCE.replace(
        "cutlass.Int32(head) * cutlass.Int32(state.layout.stride[1])",
        "cutlass.Int32(state_head) * cutlass.Int32(state.layout.stride[1])",
    ).replace(
        "independent = expensive_prepare()",
        "safe_before = cute.math.rsqrt(cutlass.Float32(4))\n"
        "state_head = cutlass.Int32(head)\n"
        "independent = cute.math.rsqrt(safe_before)",
        1,
    )
    result = _transform(source)

    assert result.index("safe_before =") < result.index("state_head =")
    assert result.index("state_head =") < result.index("_async_state_0_row_thread =")
    assert result.index("_async_state_0_row_thread =") < result.index(
        "independent = cute.math.rsqrt(safe_before)"
    )


def test_opposite_branch_register_fragment_does_not_authorize_hoist() -> None:
    else_source = SOURCE.replace(
        "independent = expensive_prepare()",
        "cache[0] = cutlass.BFloat16(0)\n"
        "independent = cute.math.rsqrt(cutlass.Float32(4))",
        1,
    )
    source = (
        "if create_cache:\n"
        "    cache = cute.make_rmem_tensor(8, cutlass.BFloat16)\n"
        "else:\n"
        f"{textwrap.indent(else_source, '    ')}"
    )
    result = _transform(source)

    assert result.index("cache[0] =") < result.index("_async_state_0_row_thread =")
    assert result.index("_async_state_0_row_thread =") < result.index(
        "independent = cute.math.rsqrt"
    )


def test_thread_varying_opaque_mask_name_fails_closed() -> None:
    source = _add_observable_other_store(
        SOURCE.replace(
            "valid = batch < state_size",
            "valid = (other.iterator + "
            "cutlass.Int32(cute.arch.thread_idx()[0])).load() != 0",
        )
    )
    _assert_no_match(
        lambda _: source,
        metadata=_metadata(include_other=True),
        tensor_names=frozenset(("state", "other")),
        proven_disjoint_tensor_pairs={frozenset(("state", "other"))},
    )


def test_thread_varying_mask_with_observable_output_fails_closed() -> None:
    source = _add_observable_other_store(
        SOURCE.replace(
            "if valid",
            "if cutlass.Int32(cute.arch.thread_idx()[0]) != cutlass.Int32(1)",
        )
    )

    _assert_no_match(
        lambda _: source,
        metadata=_metadata(include_other=True),
        tensor_names=frozenset(("state", "other")),
        proven_disjoint_tensor_pairs={frozenset(("state", "other"))},
    )


def test_immutable_uniform_scalar_tensor_load_mask_is_admitted() -> None:
    result = _transform(
        _scalar_index_mask_source(),
        metadata=_scalar_index_metadata(),
        tensor_names=frozenset(("state", "indices")),
        proven_disjoint_tensor_pairs={frozenset(("state", "indices"))},
        proven_tensor_stride_values=_scalar_index_strides(),
    )

    assert "cute.arch.cp_async_shared_global(" in result


@pytest.mark.parametrize(
    "mutator",
    (
        lambda source: source.replace(
            "cute.arch.block_idx()[0]", "cute.arch.thread_idx()[0]", 1
        ),
        lambda source: source.replace(
            "index_offset = cutlass.Int32(cute.arch.block_idx()[0])",
            "selector = cutlass.Int32(cute.arch.thread_idx()[0])\n"
            "index_offset = selector\n"
            "selector = cutlass.Int32(0)",
        ),
        lambda source: "indices_alias = indices\n" + source,
        lambda source: source.replace(
            "batch = cutlass.Int32(index_load)",
            "(indices.iterator + 0).store(cutlass.Int32(1))\n"
            "batch = cutlass.Int32(index_load)",
        ),
        lambda source: source.replace(
            "batch = cutlass.Int32(index_load)",
            "index_load = cutlass.Int32(0)\nbatch = cutlass.Int32(index_load)",
        ),
        lambda source: source.replace(
            "and cutlass.Int32(column_base + 7) < 128",
            "and cutlass.Int32(column_base + 7) != 7",
        ),
        lambda source: source.replace(
            "valid = (",
            "selector = cutlass.Int32(cute.arch.thread_idx()[0])\n"
            "mask_snapshot = selector\n"
            "selector = cutlass.Int32(0)\n"
            "valid = (",
        ).replace(
            "batch < state_size",
            "batch < state_size and mask_snapshot == 0",
            1,
        ),
        lambda source: source.replace(
            "and cutlass.Int32(column_base + 7) < 128",
            "and cutlass.Int32(column_base * 1073741824) < 128",
        ),
        lambda source: source.replace(
            "and cutlass.Int32(column_base + 7) < 128",
            "and cutlass.Uint32(column_base - 8) < 128",
        ),
    ),
)
def test_unsafe_scalar_tensor_load_masks_fail_closed(
    mutator: Callable[[str], str],
) -> None:
    source = mutator(_scalar_index_mask_source())
    _assert_no_match(
        lambda _: source,
        metadata=_scalar_index_metadata(),
        tensor_names=frozenset(("state", "indices")),
        proven_disjoint_tensor_pairs={frozenset(("state", "indices"))},
        proven_tensor_stride_values=_scalar_index_strides(),
    )


def test_scalar_tensor_load_requires_runtime_disjointness_from_every_argument() -> None:
    source = _scalar_index_mask_source()
    metadata = {
        **_scalar_index_metadata(),
        "other": TensorMetadata("cutlass.BFloat16", (264, 12, 128, 128)),
    }
    _assert_no_match(
        lambda _: source,
        metadata=metadata,
        tensor_names=frozenset(("state", "indices", "other")),
        proven_disjoint_tensor_pairs={frozenset(("state", "indices"))},
        proven_tensor_stride_values=_scalar_index_strides(),
    )


def test_scalar_tensor_load_requires_specialized_source_stride() -> None:
    source = _scalar_index_mask_source()
    _assert_no_match(
        lambda _: source,
        metadata=_scalar_index_metadata(),
        tensor_names=frozenset(("state", "indices")),
        proven_disjoint_tensor_pairs={frozenset(("state", "indices"))},
        proven_tensor_stride_values=_strides(),
    )


@pytest.mark.parametrize(
    "alias_lines",
    (
        "load_batch = selector",
        "snapshot = selector\nload_batch = snapshot",
    ),
)
def test_order_sensitive_rebound_pointer_alias_fails_closed(
    alias_lines: str,
) -> None:
    source = SOURCE.replace(
        "batch = cutlass.Int32(0)",
        "batch = cutlass.Int32(0)\n"
        "selector = cutlass.Int32(0)\n"
        f"{alias_lines}\n"
        "selector = cutlass.Int32(1)",
    )
    source = source.replace(
        "cutlass.Int32(batch) * cutlass.Int32(state.layout.stride[0])",
        "cutlass.Int32(load_batch) * cutlass.Int32(state.layout.stride[0])",
        1,
    )
    source = _replace_last(
        source,
        "cutlass.Int32(batch) * cutlass.Int32(state.layout.stride[0])",
        "cutlass.Int32(selector) * cutlass.Int32(state.layout.stride[0])",
    )
    metadata = {"state": TensorMetadata("cutlass.BFloat16", (2, 1, 16, 128))}
    strides = {
        ("state", 0): 128,
        ("state", 1): 2048,
        ("state", 2): 128,
        ("state", 3): 1,
    }

    _assert_no_match(
        lambda _: source,
        metadata=metadata,
        thread_block_dims=(16, 1, 1),
        proven_tensor_stride_values=strides,
    )


def test_earlier_immutable_pointer_alias_is_admitted() -> None:
    source = SOURCE.replace(
        "valid = batch < state_size",
        "load_batch = batch\nvalid = batch < state_size",
    ).replace(
        "cutlass.Int32(batch) * cutlass.Int32(state.layout.stride[0])",
        "cutlass.Int32(load_batch) * cutlass.Int32(state.layout.stride[0])",
        1,
    )

    assert "cute.arch.cp_async_shared_global(" in _transform(source)


def test_column_partition_does_not_capture_a_later_rebinding() -> None:
    source = SOURCE.replace(
        "column_base = cutlass.Int32(cute.arch.thread_idx()[0]) * 8",
        "column_base = selector * 8\n"
        "selector = cutlass.Int32(cute.arch.thread_idx()[0])",
    )

    _assert_no_match(
        lambda _: source,
        uniform_names=frozenset(("state_size", "selector")),
    )


@pytest.mark.parametrize(
    "mutator",
    (
        lambda source: "alias = state\n" + source,
        lambda source: "extra = state.iterator\n" + source,
        lambda source: _replace_last(
            source,
            "cutlass.Int32(row) * cutlass.Int32(state.layout.stride[2])",
            "cutlass.Int32(row + 1) * cutlass.Int32(state.layout.stride[2])",
        ),
        lambda source: source.replace(
            "    updated = update(values, independent)",
            "    lane = 0\n    updated = update(values, independent)",
        ),
        lambda source: source.replace(
            "    updated = update(values, independent)",
            "    tile = tile + 1\n    updated = update(values, independent)",
        ),
        lambda source: source.replace(
            "    updated = update(values, independent)",
            "    extra = (state.iterator + 0).load()\n"
            "    updated = update(values, independent)",
        ),
        lambda source: source.replace(
            "    updated = update(values, independent)",
            "    if stop:\n        break\n    updated = update(values, independent)",
        ),
    ),
)
def test_alias_address_and_control_flow_variants_fail_closed(
    mutator: Callable[[str], str],
) -> None:
    _assert_no_match(mutator)


def test_scalar_b1_style_state_access_is_a_noop() -> None:
    source = """
row = cutlass.Int32(cute.arch.thread_idx()[1])
for element in range(4):
    column = cutlass.Int32(cute.arch.thread_idx()[0]) + element * 32
    value = (state.iterator + row * state.layout.stride[0] + column).load()
    (state.iterator + row * state.layout.stride[0] + column).store(value)
"""
    assert _transform(source) == ast.unparse(ast.parse(source))


def test_preloads_exact_masked_row_scalar_loads_as_aligned_u32_pairs() -> None:
    source = _packed_scalar_source()
    result = _transform(
        source,
        metadata=_packed_scalar_metadata(),
        tensor_names=frozenset(("state", "aux")),
        proven_tensor_base_alignments=frozenset(("aux",)),
        proven_disjoint_tensor_pairs=_packed_scalar_disjoint(),
        proven_tensor_stride_values=_packed_scalar_strides(),
        constexpr_values={"_AUX_TILE": 64},
        uniform_names=frozenset(("state_size", "aux_size")),
    )

    assert (
        "_async_state_0_scalar_pairs = cute.make_rmem_tensor(8, cutlass.Uint32)"
        in result
    )
    assert "for _async_state_0_scalar_pair in cutlass.range_constexpr(8):" in result
    assert "cute.recast_ptr(aux.iterator +" in result
    assert "dtype=cutlass.Uint32).load()" in result
    assert result.count("_async_state_0_scalar_pair_bits_") == 6
    preload = result.index("for _async_state_0_scalar_pair")
    assert result.index("cute.arch.cp_async_commit_group()") < preload
    assert preload < result.index("cute.arch.cp_async_wait_group(2)")
    # A boundary pair is loaded only when both original scalar predicates hold.
    assert (
        "and"
        in result[
            result.index("for _async_state_0_scalar_pair") : result.index(
                "cute.arch.cp_async_wait_group(2)"
            )
        ]
    )
    # Both original masked expressions remain as exact partial-pair fallbacks.
    assert result.count("else (aux.iterator +") == 2
    original_values = [
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "aux_value"
    ]
    rewritten_values = [
        node.value
        for node in ast.walk(ast.parse(result))
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "aux_value"
    ]
    assert len(original_values) == len(rewritten_values) == 2
    for original, rewritten in zip(original_values, rewritten_values, strict=True):
        assert isinstance(rewritten, ast.IfExp)
        assert ast.dump(rewritten.orelse, include_attributes=False) == ast.dump(
            original, include_attributes=False
        )


def test_scalar_pair_preload_drops_mask_for_proven_full_grid_range() -> None:
    source = _packed_scalar_source().replace(
        "aux_base = cutlass.Int32(cute.arch.block_idx()[2]) * _AUX_TILE",
        "aux_base = cutlass.Int32(3072 + "
        "cutlass.Int32(cute.arch.block_idx()[0]) * 128)",
    )
    metadata = _packed_scalar_metadata()
    metadata["aux"] = TensorMetadata("cutlass.BFloat16", (264, 4608))
    result = _transform(
        source,
        metadata=metadata,
        tensor_names=frozenset(("state", "aux")),
        proven_tensor_base_alignments=frozenset(("aux",)),
        proven_tensor_size_values={("aux", 1): ("aux_size", 4608)},
        proven_disjoint_tensor_pairs=_packed_scalar_disjoint(),
        proven_tensor_stride_values=_packed_scalar_strides(),
        block_grid_dims=(12, None, 1),
        constexpr_values={"_AUX_TILE": 64},
        uniform_names=frozenset(("state_size", "aux_size")),
    )

    preload = result[
        result.index("for _async_state_0_scalar_pair") : result.index(
            "cute.arch.cp_async_wait_group(2)"
        )
    ]
    assert "dtype=cutlass.Uint32).load()" in preload
    assert "cutlass.Uint32(0)" not in preload
    assert "if " not in preload
    assert "else (aux.iterator +" not in result
    assert "aux_size" not in result


@pytest.mark.parametrize(
    "block_grid_dims,proven_tensor_size_values,aux_extent",
    (
        ((None, None, 1), {("aux", 1): ("aux_size", 4608)}, 4608),
        ((13, None, 1), {("aux", 1): ("aux_size", 4608)}, 4608),
        ((12, None, 1), {}, 4608),
        ((12, None, 1), {("other", 1): ("aux_size", 4608)}, 4608),
        ((12, None, 1), {("aux", 0): ("aux_size", 4608)}, 4608),
        ((12, None, 1), {("aux", 1): ("other_size", 4608)}, 4608),
        ((12, None, 1), {("aux", 1): ("aux_size", 4607)}, 4608),
        ((12, None, 1), {("aux", 1): ("aux_size", 4500)}, 4608),
    ),
)
def test_scalar_pair_full_range_proof_fails_closed_without_exact_facts(
    block_grid_dims: tuple[int | None, int | None, int | None],
    proven_tensor_size_values: dict[tuple[str, int], tuple[str, int]],
    aux_extent: int,
) -> None:
    source = _packed_scalar_source().replace(
        "aux_base = cutlass.Int32(cute.arch.block_idx()[2]) * _AUX_TILE",
        "aux_base = cutlass.Int32(3072 + "
        "cutlass.Int32(cute.arch.block_idx()[0]) * 128)",
    )
    metadata = _packed_scalar_metadata()
    metadata["aux"] = TensorMetadata("cutlass.BFloat16", (264, aux_extent))
    result = _transform(
        source,
        metadata=metadata,
        tensor_names=frozenset(("state", "aux")),
        proven_tensor_base_alignments=frozenset(("aux",)),
        proven_tensor_size_values=proven_tensor_size_values,
        proven_disjoint_tensor_pairs=_packed_scalar_disjoint(),
        proven_tensor_stride_values=_packed_scalar_strides(),
        block_grid_dims=block_grid_dims,
        constexpr_values={"_AUX_TILE": 64},
        uniform_names=frozenset(("state_size", "aux_size")),
    )

    assert "_async_state_0_scalar_pairs" in result
    assert "cutlass.Uint32(0)" in result
    assert "else (aux.iterator +" in result


def test_scalar_pair_full_range_proof_rejects_rebound_size_argument() -> None:
    source = "aux_size = cutlass.Int32(4608)\n" + _packed_scalar_source().replace(
        "aux_base = cutlass.Int32(cute.arch.block_idx()[2]) * _AUX_TILE",
        "aux_base = cutlass.Int32(3072 + "
        "cutlass.Int32(cute.arch.block_idx()[0]) * 128)",
    )
    result = _transform(
        source,
        metadata=_packed_scalar_metadata(),
        tensor_names=frozenset(("state", "aux")),
        proven_tensor_base_alignments=frozenset(("aux",)),
        proven_tensor_size_values={("aux", 1): ("aux_size", 4608)},
        proven_disjoint_tensor_pairs=_packed_scalar_disjoint(),
        proven_tensor_stride_values=_packed_scalar_strides(),
        block_grid_dims=(12, None, 1),
        constexpr_values={"_AUX_TILE": 64},
        uniform_names=frozenset(("state_size", "aux_size")),
    )

    assert "_async_state_0_scalar_pairs" not in result
    assert result.count("if cutlass.Int32(aux_index) < aux_size else") == 2


def test_exact_size_facts_require_matching_replayable_runtime_anchor() -> None:
    shape_env = ShapeEnv(duck_shape=False)
    shared_size = shape_env.create_unbacked_symint()
    with FakeTensorMode(shape_env=shape_env):
        source_fake = torch.empty(
            (2, shared_size),
            device="cuda",  # @ignore-device-lint
            dtype=torch.bfloat16,
        )
        anchor_fake = torch.empty(
            (3, shared_size),
            device="cuda",  # @ignore-device-lint
            dtype=torch.bfloat16,
        )

    source_arg = TensorArg("aux", source_fake, "aux")
    anchor_arg = TensorArg("anchor", anchor_fake, "anchor")
    property_arg = TensorSizeArg("anchor_size", anchor_arg, 1)
    function = object.__new__(DeviceFunction)
    function.arguments = [source_arg, anchor_arg, property_arg]
    function._tensor_properties = {
        (TensorSizeArg, shape_env.replace(shared_size._sympy_())): property_arg
    }

    source_token = object()
    anchor_token = object()
    sources: dict[int, object] = {
        id(source_fake): source_token,
        id(anchor_fake): anchor_token,
    }
    runtimes: dict[int, object] = {
        id(source_fake): torch.empty((2, 4608), dtype=torch.bfloat16),
        id(anchor_fake): torch.empty((3, 4608), dtype=torch.bfloat16),
    }
    env = SimpleNamespace(
        compiler_fact_specialization_facts={"input_tensor_metadata"},
        shape_env=shape_env,
        tensor_input_source=lambda tensor: sources.get(id(tensor)),
        runtime_value_for_tensor=lambda tensor: runtimes.get(id(tensor)),
        size_hint=lambda size: 4608,
    )

    with patch.object(CompileEnvironment, "current", return_value=env):
        assert function.proven_tensor_size_values()["aux", 1] == (
            "anchor_size",
            4608,
        )

        # A post-bind resize changes the weakly held runtime tensor but not the
        # cache-keyed fake trace hint.  It cannot strengthen a compiled bound.
        runtimes[id(source_fake)] = torch.empty((2, 4609), dtype=torch.bfloat16)
        assert ("aux", 1) not in function.proven_tensor_size_values()
        runtimes[id(source_fake)] = torch.empty((2, 4608), dtype=torch.bfloat16)

        # The launch argument comes from the deduplicated property's anchor.
        # A same-symbol runtime mismatch must not prove the source dimension.
        runtimes[id(anchor_fake)] = torch.empty((3, 4607), dtype=torch.bfloat16)
        assert ("aux", 1) not in function.proven_tensor_size_values()

        runtimes[id(anchor_fake)] = torch.empty((3, 4608), dtype=torch.bfloat16)
        sources.pop(id(source_fake))
        assert ("aux", 1) not in function.proven_tensor_size_values()

        sources[id(source_fake)] = source_token
        sources.pop(id(anchor_fake))
        assert ("aux", 1) not in function.proven_tensor_size_values()

        sources[id(anchor_fake)] = anchor_token
        runtimes[id(source_fake)] = source_fake
        assert ("aux", 1) not in function.proven_tensor_size_values()

        runtimes[id(source_fake)] = torch.empty((2, 4608), dtype=torch.bfloat16)
        runtimes[id(anchor_fake)] = anchor_fake
        assert ("aux", 1) not in function.proven_tensor_size_values()

        runtimes[id(anchor_fake)] = torch.empty((3, 4608), dtype=torch.bfloat16)
        env.compiler_fact_specialization_facts = set()
        assert function.proven_tensor_size_values() == {}


@pytest.mark.parametrize(
    "base_expression,predicate",
    (
        (
            "cutlass.Int32(-2 + cutlass.Int32(cute.arch.block_idx()[0]) * 128)",
            "cutlass.Int32(aux_index) < aux_size",
        ),
        (
            "cutlass.Int32(2147483520 + cutlass.Int32(cute.arch.block_idx()[0]) * 128)",
            "cutlass.Int32(aux_index) < aux_size",
        ),
        (
            "cutlass.Int32(3072 + cutlass.Int32(cute.arch.block_idx()[0]) * 128)",
            "cutlass.Int32(aux_index) < aux_size + 0",
        ),
        (
            "cutlass.Int32(3072 + cutlass.Int32(cute.arch.block_idx()[0]) * 128)",
            "cutlass.Int32(aux_index) <= aux_size",
        ),
    ),
)
def test_scalar_pair_full_range_proof_rejects_unsafe_integer_forms(
    base_expression: str, predicate: str
) -> None:
    source = (
        _packed_scalar_source()
        .replace(
            "cutlass.Int32(cute.arch.block_idx()[2]) * _AUX_TILE",
            base_expression,
        )
        .replace("cutlass.Int32(aux_index) < aux_size", predicate)
    )
    result = _transform(
        source,
        metadata=_packed_scalar_metadata(),
        tensor_names=frozenset(("state", "aux")),
        proven_tensor_base_alignments=frozenset(("aux",)),
        proven_tensor_size_values={("aux", 1): ("aux_size", 2**31 - 1)},
        proven_disjoint_tensor_pairs=_packed_scalar_disjoint(),
        proven_tensor_stride_values=_packed_scalar_strides(),
        block_grid_dims=(12, None, 1),
        constexpr_values={"_AUX_TILE": 64},
        uniform_names=frozenset(("state_size", "aux_size")),
    )

    assert "_async_state_0_scalar_pairs" in result
    assert "cutlass.Uint32(0)" in result
    assert "else (aux.iterator +" in result


def test_preloads_single_current_scalar_load_site() -> None:
    result = _transform(
        _packed_scalar_source(load_count=1),
        metadata=_packed_scalar_metadata(),
        tensor_names=frozenset(("state", "aux")),
        proven_tensor_base_alignments=frozenset(("aux",)),
        proven_disjoint_tensor_pairs=_packed_scalar_disjoint(),
        proven_tensor_stride_values=_packed_scalar_strides(),
        constexpr_values={"_AUX_TILE": 64},
        uniform_names=frozenset(("state_size", "aux_size")),
    )

    assert (
        "_async_state_0_scalar_pairs = cute.make_rmem_tensor(8, cutlass.Uint32)"
        in result
    )
    assert result.count("else (aux.iterator +") == 1


def test_scalar_pair_preload_requires_cache_specialized_base_alignment() -> None:
    result = _transform(
        _packed_scalar_source(),
        metadata=_packed_scalar_metadata(),
        tensor_names=frozenset(("state", "aux")),
        proven_disjoint_tensor_pairs=_packed_scalar_disjoint(),
        proven_tensor_stride_values=_packed_scalar_strides(),
        constexpr_values={"_AUX_TILE": 64},
        uniform_names=frozenset(("state_size", "aux_size")),
    )

    assert "cute.arch.cp_async_shared_global(" in result
    assert "_async_state_0_scalar_pairs" not in result


def test_runtime_alignment_fact_rejects_odd_bf16_storage_offset() -> None:
    base = torch.empty(17, dtype=torch.bfloat16)
    aligned = base[:16]
    unaligned = base[1:17]
    assert aligned.data_ptr() % 4 == 0
    assert unaligned.data_ptr() % 4 == 2
    source = LocalSource("aux", is_input=True)
    specialization = SimpleNamespace(
        sources=(source,), classifier=_persistent_vec_alignment_matrix_signature
    )
    bound_result = _persistent_vec_alignment_matrix_signature((aligned,))

    def environment(runtime: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(
            runtime_value_for_tensor=lambda _: runtime,
            tensor_input_source=lambda _: source,
            runtime_arg_values_by_name={"aux": runtime},
            runtime_input_specializations={
                _PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY: specialization
            },
            runtime_input_specialization_matches_bound=(
                lambda key, result: (
                    key == _PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY
                    and result == bound_result
                )
            ),
        )

    assert runtime_tensor_has_specialized_alignment(
        cast("CompileEnvironment", environment(aligned)), aligned, 4
    )
    assert not runtime_tensor_has_specialized_alignment(
        cast("CompileEnvironment", environment(unaligned)), unaligned, 4
    )


def test_runtime_alignment_fact_rejects_post_bind_realignment() -> None:
    base = torch.empty(17, dtype=torch.bfloat16)
    bound_value = base[1:17]
    current_value = base[:16]
    assert bound_value.data_ptr() % 4 == 2
    assert current_value.data_ptr() % 4 == 0
    source = LocalSource("aux", is_input=True)
    specialization = SimpleNamespace(
        sources=(source,), classifier=_persistent_vec_alignment_matrix_signature
    )
    bound_result = _persistent_vec_alignment_matrix_signature((bound_value,))
    env = SimpleNamespace(
        runtime_value_for_tensor=lambda _: current_value,
        tensor_input_source=lambda _: source,
        runtime_arg_values_by_name={"aux": current_value},
        runtime_input_specializations={
            _PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY: specialization
        },
        runtime_input_specialization_matches_bound=(
            lambda key, result: (
                key == _PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY
                and result == bound_result
            )
        ),
    )

    assert not runtime_tensor_has_specialized_alignment(
        cast("CompileEnvironment", env), current_value, 4
    )


def test_runtime_alignment_fact_rejects_post_bind_size_divisibility_change() -> None:
    backing = torch.empty(16, dtype=torch.bfloat16)
    bound_value = backing[:7]
    current_value = backing[:8]
    assert bound_value.data_ptr() == current_value.data_ptr()
    assert bound_value.stride() == current_value.stride()
    source = LocalSource("aux", is_input=True)
    specialization = SimpleNamespace(
        sources=(source,), classifier=_persistent_vec_alignment_matrix_signature
    )
    bound_result = _persistent_vec_alignment_matrix_signature((bound_value,))
    env = SimpleNamespace(
        runtime_value_for_tensor=lambda _: current_value,
        tensor_input_source=lambda _: source,
        runtime_arg_values_by_name={"aux": current_value},
        runtime_input_specializations={
            _PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY: specialization
        },
        runtime_input_specialization_matches_bound=(
            lambda key, result: (
                key == _PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY
                and result == bound_result
            )
        ),
    )

    assert not runtime_tensor_has_specialized_alignment(
        cast("CompileEnvironment", env), current_value, 4
    )


def test_proven_stride_rejects_post_bind_metadata_mutation() -> None:
    traced = torch.empty_strided((4, 8), (8, 1))
    current = torch.empty_strided((4, 8), (1, 4))
    source = LocalSource("value", is_input=True)
    env = SimpleNamespace(
        compiler_fact_specialization_facts=frozenset(("input_tensor_metadata",)),
        tensor_input_source=lambda _: source,
        runtime_value_for_tensor=lambda _: current,
        specialized_strides=set(),
        size_hint=int,
    )
    device_function = object.__new__(DeviceFunction)
    device_function.arguments = [TensorArg("value", traced, "value")]

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(CompileEnvironment, "current", lambda: env)
        assert device_function.proven_tensor_stride_values() == {}
        env.runtime_value_for_tensor = lambda _: traced
        assert device_function.proven_tensor_stride_values() == {
            ("value", 0): 8,
            ("value", 1): 1,
        }


def test_scalar_pair_parity_uses_assignment_time_reaching_definitions() -> None:
    source = _packed_scalar_source().replace(
        "aux_base = cutlass.Int32(cute.arch.block_idx()[2]) * _AUX_TILE",
        "aux_base = selector\nselector = cutlass.Int32(0)",
        1,
    )
    result = _transform(
        source,
        metadata=_packed_scalar_metadata(),
        tensor_names=frozenset(("state", "aux")),
        proven_tensor_base_alignments=frozenset(("aux",)),
        proven_disjoint_tensor_pairs=_packed_scalar_disjoint(),
        proven_tensor_stride_values=_packed_scalar_strides(),
        constexpr_values={"_AUX_TILE": 64},
        uniform_names=frozenset(("state_size", "aux_size", "selector")),
    )

    assert "cute.arch.cp_async_shared_global(" in result
    assert "_async_state_0_scalar_pairs" not in result


def test_scalar_pair_parity_rejects_a_rebound_constexpr_name() -> None:
    source = (
        _packed_scalar_source()
        .replace(
            "aux_base = cutlass.Int32(cute.arch.block_idx()[2]) * _AUX_TILE",
            "_AUX_TILE = cutlass.Int32(1)",
            1,
        )
        .replace("aux_base + row", "_AUX_TILE + row")
    )
    result = _transform(
        source,
        metadata=_packed_scalar_metadata(),
        tensor_names=frozenset(("state", "aux")),
        proven_tensor_base_alignments=frozenset(("aux",)),
        proven_disjoint_tensor_pairs=_packed_scalar_disjoint(),
        proven_tensor_stride_values=_packed_scalar_strides(),
        constexpr_values={"_AUX_TILE": 64},
        uniform_names=frozenset(("state_size", "aux_size", "_AUX_TILE")),
    )

    assert "cute.arch.cp_async_shared_global(" in result
    assert "_async_state_0_scalar_pairs" not in result


def test_scalar_pair_parity_does_not_treat_untyped_uniform_as_integer() -> None:
    source = _packed_scalar_source().replace(
        "cutlass.Int32(cute.arch.block_idx()[2]) * _AUX_TILE",
        "cutlass.Int32(2 * scale)",
        1,
    )
    result = _transform(
        source,
        metadata=_packed_scalar_metadata(),
        tensor_names=frozenset(("state", "aux")),
        proven_tensor_base_alignments=frozenset(("aux",)),
        proven_disjoint_tensor_pairs=_packed_scalar_disjoint(),
        proven_tensor_stride_values=_packed_scalar_strides(),
        constexpr_values={"_AUX_TILE": 64},
        uniform_names=frozenset(("state_size", "aux_size", "scale")),
    )

    assert "cute.arch.cp_async_shared_global(" in result
    assert "_async_state_0_scalar_pairs" not in result


def test_scalar_pair_address_rejects_loop_carried_alias_snapshot() -> None:
    source = (
        _packed_scalar_source(load_count=1)
        .replace(
            "aux_base = cutlass.Int32(cute.arch.block_idx()[2]) * _AUX_TILE",
            "aux_base = cutlass.Int32(cute.arch.block_idx()[2]) * _AUX_TILE\n"
            "carry = aux_base",
            1,
        )
        .replace(
            "    aux_index = aux_base + row",
            "    aux_index = carry\n    carry = aux_base + row",
            1,
        )
    )
    result = _transform(
        source,
        metadata=_packed_scalar_metadata(),
        tensor_names=frozenset(("state", "aux")),
        proven_tensor_base_alignments=frozenset(("aux",)),
        proven_disjoint_tensor_pairs=_packed_scalar_disjoint(),
        proven_tensor_stride_values=_packed_scalar_strides(),
        constexpr_values={"_AUX_TILE": 64},
        uniform_names=frozenset(("state_size", "aux_size")),
    )

    assert "cute.arch.cp_async_shared_global(" in result
    assert "_async_state_0_scalar_pairs" not in result


def test_scalar_pair_preload_does_not_cross_an_unknown_effect() -> None:
    source = _packed_scalar_source().replace(
        "    aux_index = aux_base + row",
        "    opaque_effect()\n    aux_index = aux_base + row",
        1,
    )
    result = _transform(
        source,
        metadata=_packed_scalar_metadata(),
        tensor_names=frozenset(("state", "aux")),
        proven_tensor_base_alignments=frozenset(("aux",)),
        proven_disjoint_tensor_pairs=_packed_scalar_disjoint(),
        proven_tensor_stride_values=_packed_scalar_strides(),
        constexpr_values={"_AUX_TILE": 64},
        uniform_names=frozenset(("state_size", "aux_size")),
    )

    assert "cute.arch.cp_async_shared_global(" in result
    assert "_async_state_0_scalar_pairs" not in result


def test_scalar_pair_preload_does_not_cross_a_runtime_division_trap() -> None:
    source = _packed_scalar_source().replace(
        "    aux_index = aux_base + row",
        "    may_trap = cutlass.Int32(1) // divisor\n    aux_index = aux_base + row",
        1,
    )
    result = _transform(
        source,
        metadata=_packed_scalar_metadata(),
        tensor_names=frozenset(("state", "aux")),
        proven_tensor_base_alignments=frozenset(("aux",)),
        proven_disjoint_tensor_pairs=_packed_scalar_disjoint(),
        proven_tensor_stride_values=_packed_scalar_strides(),
        constexpr_values={"_AUX_TILE": 64},
        uniform_names=frozenset(("state_size", "aux_size", "divisor")),
    )

    assert "cute.arch.cp_async_shared_global(" in result
    assert "_async_state_0_scalar_pairs" not in result


def test_scalar_pair_preload_uses_loop_local_divisor_reaching_definition() -> None:
    source = (
        _packed_scalar_source()
        .replace(
            "aux_base = cutlass.Int32(cute.arch.block_idx()[2]) * _AUX_TILE",
            "aux_base = cutlass.Int32(cute.arch.block_idx()[2]) * _AUX_TILE\n"
            "divisor_alias = cutlass.Int32(2)",
            1,
        )
        .replace(
            "    aux_index = aux_base + row",
            "    divisor_alias = divisor\n"
            "    may_trap = cutlass.Int32(1) // divisor_alias\n"
            "    aux_index = aux_base + row",
            1,
        )
    )
    result = _transform(
        source,
        metadata=_packed_scalar_metadata(),
        tensor_names=frozenset(("state", "aux")),
        proven_tensor_base_alignments=frozenset(("aux",)),
        proven_disjoint_tensor_pairs=_packed_scalar_disjoint(),
        proven_tensor_stride_values=_packed_scalar_strides(),
        constexpr_values={"_AUX_TILE": 64},
        uniform_names=frozenset(("state_size", "aux_size", "divisor")),
    )

    assert "cute.arch.cp_async_shared_global(" in result
    assert "_async_state_0_scalar_pairs" not in result


def test_scalar_pair_preload_rejects_runtime_division_in_matched_mask() -> None:
    source = _packed_scalar_source().replace(
        "if cutlass.Int32(aux_index) < aux_size",
        "if cutlass.Int32(aux_index) < aux_size and 1 // (lane - 1)",
    )
    result = _transform(
        source,
        metadata=_packed_scalar_metadata(),
        tensor_names=frozenset(("state", "aux")),
        proven_tensor_base_alignments=frozenset(("aux",)),
        proven_disjoint_tensor_pairs=_packed_scalar_disjoint(),
        proven_tensor_stride_values=_packed_scalar_strides(),
        constexpr_values={"_AUX_TILE": 64},
        uniform_names=frozenset(("state_size", "aux_size")),
    )

    assert "cute.arch.cp_async_shared_global(" in result
    assert "_async_state_0_scalar_pairs" not in result


@pytest.mark.parametrize(
    "source_mutator,metadata,strides,constexpr_values",
    (
        (
            lambda source: source.replace("_AUX_TILE\n", "_AUX_TILE + 1\n", 1),
            _packed_scalar_metadata(),
            _packed_scalar_strides(),
            {"_AUX_TILE": 64},
        ),
        (
            lambda source: source,
            _packed_scalar_metadata(),
            _packed_scalar_strides(),
            {},
        ),
        (
            lambda source: source.replace(
                "cutlass.Int32(aux_index) * cutlass.Int32(aux.layout.stride[1])",
                "cutlass.Int32(aux_index + 1) * cutlass.Int32(aux.layout.stride[1])",
            ),
            _packed_scalar_metadata(),
            _packed_scalar_strides(),
            {"_AUX_TILE": 64},
        ),
        (
            lambda source: source,
            _packed_scalar_metadata(),
            _packed_scalar_strides(last_stride=2),
            {"_AUX_TILE": 64},
        ),
        (
            lambda source: source,
            _packed_scalar_metadata(aux_dtype="cutlass.Float16"),
            _packed_scalar_strides(),
            {"_AUX_TILE": 64},
        ),
        (
            lambda source: source.replace(
                "else cutlass.BFloat16(0)", "else cutlass.BFloat16(1)", 1
            ),
            _packed_scalar_metadata(),
            _packed_scalar_strides(),
            {"_AUX_TILE": 64},
        ),
    ),
)
def test_scalar_pair_preload_rejects_shift_stride_dtype_and_mask_variants(
    source_mutator: Callable[[str], str],
    metadata: dict[str, TensorMetadata],
    strides: dict[tuple[str, int], int],
    constexpr_values: dict[str, int],
) -> None:
    result = _transform(
        source_mutator(_packed_scalar_source()),
        metadata=metadata,
        tensor_names=frozenset(("state", "aux")),
        proven_tensor_base_alignments=frozenset(("aux",)),
        proven_disjoint_tensor_pairs=_packed_scalar_disjoint(),
        proven_tensor_stride_values=strides,
        constexpr_values=constexpr_values,
        uniform_names=frozenset(("state_size", "aux_size")),
    )

    assert "cute.arch.cp_async_shared_global(" in result
    assert "_async_state_0_scalar_pairs" not in result


def test_scalar_pair_preload_requires_source_disjoint_from_every_tensor() -> None:
    result = _transform(
        _packed_scalar_source(),
        metadata=_packed_scalar_metadata(include_other=True),
        tensor_names=frozenset(("state", "aux", "other")),
        proven_tensor_base_alignments=frozenset(("aux",)),
        proven_disjoint_tensor_pairs=_packed_scalar_disjoint(include_other=True),
        proven_tensor_stride_values=_packed_scalar_strides(),
        constexpr_values={"_AUX_TILE": 64},
        uniform_names=frozenset(("state_size", "aux_size")),
    )

    assert "cute.arch.cp_async_shared_global(" in result
    assert "_async_state_0_scalar_pairs" not in result


@pytest.mark.parametrize(
    "mutator",
    (
        lambda source: "aux_alias = aux\n" + source,
        lambda source: "aux = other\n" + source,
        lambda source: source.replace(
            "    updated = values",
            "    (aux.iterator + cutlass.Int32(aux_index)).store(aux_value)\n"
            "    updated = values",
            1,
        ),
        lambda source: source + "\nescaped = aux_value\n",
        lambda source: source.replace(
            "    aux_use = cutlass.Float32(aux_value)",
            "    aux_use = consume(aux_value, aux)",
            1,
        ),
        lambda source: source.replace(
            "    aux_use = cutlass.Float32(aux_value)",
            "    aux_use = cutlass.Float32(aux_value)\n"
            "    aux_base = aux_base + cutlass.Int32(2)",
            1,
        ),
        lambda source: _replace_last(
            source,
            "    aux_index = aux_base + row",
            "    aux_index = aux_base + row + 2",
        ),
    ),
)
def test_scalar_pair_preload_rejects_alias_store_escape_and_nonidentical_sites(
    mutator: Callable[[str], str],
) -> None:
    result = _transform(
        mutator(_packed_scalar_source()),
        metadata=_packed_scalar_metadata(),
        tensor_names=frozenset(("state", "aux")),
        proven_tensor_base_alignments=frozenset(("aux",)),
        proven_disjoint_tensor_pairs=_packed_scalar_disjoint(),
        proven_tensor_stride_values=_packed_scalar_strides(),
        constexpr_values={"_AUX_TILE": 64},
        uniform_names=frozenset(("state_size", "aux_size")),
    )

    assert "cute.arch.cp_async_shared_global(" in result
    assert "_async_state_0_scalar_pairs" not in result


def test_scalar_pair_preload_rejects_odd_row_extent() -> None:
    source = (
        _packed_scalar_source()
        .replace("for lane in range(16):", "for lane in range(15):", 1)
        .replace("* 16 + cutlass.Int32(lane)", "* 15 + cutlass.Int32(lane)")
    )
    metadata = _packed_scalar_metadata()
    metadata["state"] = TensorMetadata("cutlass.BFloat16", (264, 12, 120, 128))
    result = _transform(
        source,
        metadata=metadata,
        stages=5,
        lookahead=2,
        group_rows=5,
        tensor_names=frozenset(("state", "aux")),
        proven_tensor_base_alignments=frozenset(("aux",)),
        proven_disjoint_tensor_pairs=_packed_scalar_disjoint(),
        proven_tensor_stride_values=_packed_scalar_strides(),
        constexpr_values={"_AUX_TILE": 64},
        uniform_names=frozenset(("state_size", "aux_size")),
    )

    assert "_async_state_0_scalar_pairs" not in result
