from __future__ import annotations

import ast
import textwrap

import pytest

from helion._compiler.ast_extension import ExtendedAST
from helion._compiler.ast_extension import convert
from helion._compiler.cute.pack_bf16_recurrence import pack_bf16_recurrences

SOURCE = """
key_fragment = cute.make_rmem_tensor(8, cutlass.Float32)
state_fragment = cute.make_rmem_tensor(8, cutlass.BFloat16)
query_fragment = cute.make_rmem_tensor(8, cutlass.Float32)
decay_fragment = cute.make_rmem_tensor(8, cutlass.Float32)
key_source = cute.make_rmem_tensor(4, cutlass.Float32)
query_source = cute.make_rmem_tensor(4, cutlass.Float32)
decay_source = cute.make_rmem_tensor(4, cutlass.Float32)
for expanded_lane in cutlass.range_constexpr(8):
    source_lane = (
        cutlass.Int32(cute.arch.thread_idx()[0]) * 2
        + cutlass.Int32(expanded_lane // 4)
    )
    source_element = expanded_lane % 4
    key_value = cutlass.Float32(
        cute.arch.shuffle_sync(key_source[source_element], source_lane)
    )
    query_value = cutlass.Float32(
        cute.arch.shuffle_sync(query_source[source_element], source_lane)
    )
    key_fragment[expanded_lane] = key_value * key_norm
    query_fragment[expanded_lane] = query_value * query_norm * scale
    decay_fragment[expanded_lane] = cutlass.Float32(
        cute.arch.shuffle_sync(decay_source[source_element], source_lane)
    )
kq_acc_lo = cutlass.Float32(0)
kq_acc_hi = cutlass.Float32(0)
for kq_pair in cutlass.range_constexpr(4):
    kq_lo = kq_pair * 2
    kq_hi = kq_lo + 1
    kq_acc_lo, kq_acc_hi = cute.arch.fma_packed_f32x2(
        (key_fragment[kq_lo], key_fragment[kq_hi]),
        (query_fragment[kq_lo], query_fragment[kq_hi]),
        (kq_acc_lo, kq_acc_hi),
    )
kq_lane = cutlass.Float32(kq_acc_lo) + cutlass.Float32(kq_acc_hi)
kq = cutlass.Float32(
    cute.arch.warp_reduction_sum(kq_lane, threads_in_group=16)
)
pipe_values = cute.arch.load(
    shared_pointer, ir.VectorType.get([8], cutlass.Uint16.mlir_type)
)
for row in cutlass.range_constexpr(16):
    state_values = pipe_values
    if row + 1 < 16:
        pipe_values = cute.arch.load(
            next_shared_pointer,
            ir.VectorType.get([8], cutlass.Uint16.mlir_type),
        )
    row_index = row_base + row
    prediction_acc_lo = cutlass.Float32(0)
    prediction_acc_hi = cutlass.Float32(0)
    base_acc_lo = cutlass.Float32(0)
    base_acc_hi = cutlass.Float32(0)
    for pair in cutlass.range_constexpr(4):
        lo = pair * 2
        hi = lo + 1
        decay_lo = decay_fragment[lo]
        decay_hi = decay_fragment[hi]
        state_lo = cutlass.Uint16(state_values[lo]).bitcast(cutlass.BFloat16)
        state_hi = cutlass.Uint16(state_values[hi]).bitcast(cutlass.BFloat16)
        state_fragment[lo] = state_lo
        state_fragment[hi] = state_hi
        decayed_lo, decayed_hi = cute.arch.mul_packed_f32x2(
            (cutlass.Float32(state_lo), cutlass.Float32(state_hi)),
            (decay_lo, decay_hi),
        )
        prediction_acc_lo, prediction_acc_hi = cute.arch.fma_packed_f32x2(
            (decayed_lo, decayed_hi),
            (key_fragment[lo], key_fragment[hi]),
            (prediction_acc_lo, prediction_acc_hi),
        )
        base_acc_lo, base_acc_hi = cute.arch.fma_packed_f32x2(
            (decayed_lo, decayed_hi),
            (query_fragment[lo], query_fragment[hi]),
            (base_acc_lo, base_acc_hi),
        )
    prediction_lane = cutlass.Float32(prediction_acc_lo) + cutlass.Float32(
        prediction_acc_hi
    )
    base_lane = cutlass.Float32(base_acc_lo) + cutlass.Float32(base_acc_hi)
    prediction = cutlass.Float32(
        cute.arch.warp_reduction_sum(prediction_lane, threads_in_group=16)
    )
    base = cutlass.Float32(
        cute.arch.warp_reduction_sum(base_lane, threads_in_group=16)
    )
    value = values[row]
    delta = (value - prediction) * beta
    if cute.arch.thread_idx()[0] == 0:
        (output.iterator + row_index).store(
            cutlass.BFloat16(base + delta * kq)
        )
    store_values = []
    for pair in cutlass.range_constexpr(4):
        lo = pair * 2
        hi = lo + 1
        decay_lo = decay_fragment[lo]
        decay_hi = decay_fragment[hi]
        state_lo = state_fragment[lo]
        state_hi = state_fragment[hi]
        decayed_lo, decayed_hi = cute.arch.mul_packed_f32x2(
            (cutlass.Float32(state_lo), cutlass.Float32(state_hi)),
            (decay_lo, decay_hi),
        )
        updated_lo, updated_hi = cute.arch.fma_packed_f32x2(
            (delta, delta),
            (key_fragment[lo], key_fragment[hi]),
            (decayed_lo, decayed_hi),
        )
        stored_lo = cutlass.BFloat16(updated_lo)
        stored_hi = cutlass.BFloat16(updated_hi)
        store_values.append(cutlass.BFloat16(stored_lo).bitcast(cutlass.Uint16))
        store_values.append(cutlass.BFloat16(stored_hi).bitcast(cutlass.Uint16))
    _cute_store_u16x8_l2_evict_last(state.iterator + row_index, store_values)
"""


def _separate_cache_source() -> str:
    shared = """for expanded_lane in cutlass.range_constexpr(8):
    source_lane = (
        cutlass.Int32(cute.arch.thread_idx()[0]) * 2
        + cutlass.Int32(expanded_lane // 4)
    )
    source_element = expanded_lane % 4
    key_value = cutlass.Float32(
        cute.arch.shuffle_sync(key_source[source_element], source_lane)
    )
    query_value = cutlass.Float32(
        cute.arch.shuffle_sync(query_source[source_element], source_lane)
    )
    key_fragment[expanded_lane] = key_value * key_norm
    query_fragment[expanded_lane] = query_value * query_norm * scale
    decay_fragment[expanded_lane] = cutlass.Float32(
        cute.arch.shuffle_sync(decay_source[source_element], source_lane)
    )
"""
    separate = """for expanded_lane in cutlass.range_constexpr(8):
    key_fragment[expanded_lane] = key_source[expanded_lane]
for expanded_lane in cutlass.range_constexpr(8):
    query_fragment[expanded_lane] = query_source[expanded_lane]
for expanded_lane in cutlass.range_constexpr(8):
    decay_fragment[expanded_lane] = decay_source[expanded_lane]
"""
    source = SOURCE.replace(shared, separate, 1)
    source = source.replace(
        "(key_fragment[kq_lo], key_fragment[kq_hi])",
        "(key_fragment[kq_lo] * key_norm, key_fragment[kq_hi] * key_norm)",
    ).replace(
        "(query_fragment[kq_lo], query_fragment[kq_hi])",
        "(\n"
        "            query_fragment[kq_lo] * query_norm * scale,\n"
        "            query_fragment[kq_hi] * query_norm * scale,\n"
        "        )",
    )
    return source.replace(
        "(key_fragment[lo], key_fragment[hi])",
        "(key_fragment[lo] * key_norm, key_fragment[hi] * key_norm)",
    ).replace(
        "(query_fragment[lo], query_fragment[hi])",
        "(\n"
        "                query_fragment[lo] * query_norm * scale,\n"
        "                query_fragment[hi] * query_norm * scale,\n"
        "            )",
    )


def _half_warp_source() -> str:
    source = _separate_cache_source()
    separate = """for expanded_lane in cutlass.range_constexpr(8):
    key_fragment[expanded_lane] = key_source[expanded_lane]
for expanded_lane in cutlass.range_constexpr(8):
    query_fragment[expanded_lane] = query_source[expanded_lane]
for expanded_lane in cutlass.range_constexpr(8):
    decay_fragment[expanded_lane] = decay_source[expanded_lane]
"""
    producers = """producer_base = cutlass.Int32(cute.arch.thread_idx()[0]) * 8
key_vector = cute.arch.load(
    key.iterator + (producer_base if producer_base + 7 < 128 else 0),
    ir.VectorType.get([8], cutlass.Uint16.mlir_type),
)
key_sum_lane = cutlass.Float32(0)
for expanded_lane in cutlass.range_constexpr(8):
    key_index = producer_base + expanded_lane
    key_value = (
        cutlass.Float32(
            cutlass.Uint16(key_vector[expanded_lane]).bitcast(cutlass.BFloat16)
        )
        if key_index < 128
        else cutlass.Float32(0)
    )
    key_fragment[expanded_lane] = key_value
    key_sum_lane = key_sum_lane + key_value * key_value
key_sum = cutlass.Float32(
    cute.arch.warp_reduction_sum(key_sum_lane, threads_in_group=16)
)
key_norm = cute.math.rsqrt(key_sum, fastmath=True)
query_vector = cute.arch.load(
    query.iterator + (producer_base if producer_base + 7 < 128 else 0),
    ir.VectorType.get([8], cutlass.Uint16.mlir_type),
)
query_sum_lane = cutlass.Float32(0)
for expanded_lane in cutlass.range_constexpr(8):
    query_index = producer_base + expanded_lane
    query_value = (
        cutlass.Float32(
            cutlass.Uint16(query_vector[expanded_lane]).bitcast(cutlass.BFloat16)
        )
        if query_index < 128
        else cutlass.Float32(0)
    )
    query_fragment[expanded_lane] = query_value
    query_sum_lane = query_sum_lane + query_value * query_value
query_sum = cutlass.Float32(
    cute.arch.warp_reduction_sum(query_sum_lane, threads_in_group=16)
)
query_norm = cute.math.rsqrt(query_sum, fastmath=True)
decay_vector = cute.arch.load(
    decay.iterator + (producer_base if producer_base + 7 < 128 else 0),
    ir.VectorType.get([8], cutlass.Uint16.mlir_type),
)
for expanded_lane in cutlass.range_constexpr(8):
    decay_index = producer_base + expanded_lane
    decay_fragment[expanded_lane] = (
        cutlass.Float32(
            cutlass.Uint16(decay_vector[expanded_lane]).bitcast(cutlass.BFloat16)
        )
        if decay_index < 128
        else cutlass.Float32(0)
    )
"""
    return source.replace(separate, producers, 1)


def _transform(
    source: str = SOURCE,
    *,
    enabled: bool = True,
    fast_math: bool = True,
    capability: tuple[int, int] | None = (10, 0),
    thread_block_dims: tuple[int, int, int] | None = None,
) -> tuple[str, bool]:
    body = ast.parse(source).body
    row = next(
        node
        for statement in body
        for node in ast.walk(statement)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "row"
    )
    row.__dict__["_helion_async_state_vector"] = "state_values"
    row.__dict__["_helion_async_state_vector_width"] = 8
    row.__dict__["_helion_async_state_valid_predicate"] = None
    result = pack_bf16_recurrences(
        body,
        enabled=enabled,
        fast_math=fast_math,
        target_device_capability=capability,
        uniform_names=frozenset(("key_norm", "query_norm", "scale")),
        thread_block_dims=thread_block_dims,
    )
    return (
        ast.unparse(
            ast.fix_missing_locations(ast.Module(body=result, type_ignores=[]))
        ),
        result is not body,
    )


def test_packs_complete_async_rank1_recurrence() -> None:
    result, changed = _transform()

    assert changed
    assert "_bf16x2_0_abi_version = 1" in result
    assert result.count("_cute_rank1_fma_bf16x2(") == 3
    assert "_cute_rank1_mul_bf16x2(" in result
    assert "_cute_rank1_add_bf16x2(" in result
    assert "_cute_rank1_pack_bf16x2(" in result
    assert "_cute_store_u32x4_l2_evict_last(" in result
    assert "ir.VectorType.get([4], cutlass.Uint32.mlir_type)" in result
    assert "kq_acc" not in result
    assert "fma_packed_f32x2" not in result
    assert "if cute.arch.thread_idx()[0] == 0:" in result
    assert "_bf16x2_0_query_scale = query_norm * scale" in result


def test_matching_does_not_depend_on_generated_identifiers() -> None:
    renamed = SOURCE.replace("key_fragment", "direction_a").replace(
        "query_fragment", "direction_b"
    )

    result, changed = _transform(renamed)

    assert changed
    assert "direction_a[_bf16x2_0_pair]" in result
    assert "direction_b[_bf16x2_0_pair]" in result


def test_packs_separate_factored_source_caches() -> None:
    result, changed = _transform(_separate_cache_source())

    assert changed
    assert result.count("_cute_rank1_fma_bf16x2(") == 3
    assert "cute.arch.fma_packed_f32x2" not in result
    assert "_bf16x2_0_query_scale = query_norm * scale" in result
    assert "_bf16x2_0_key_cache = cute.make_rmem_tensor(4, cutlass.Uint32)" in result


def test_shares_separate_producers_across_half_warps() -> None:
    result, changed = _transform(_half_warp_source(), thread_block_dims=(16, 8, 1))

    assert changed
    assert result.count("_bf16x2_0_half_warp_producer_abi_version = 1") == 1
    assert result.count("ir.VectorType.get([4], cutlass.Uint16.mlir_type)") == 3
    assert result.count("threads_in_group=32") == 2
    assert result.count("cute.arch.shuffle_sync(") == 3
    assert "producer_base + 3 < 128" in result


@pytest.mark.parametrize(
    ("source", "thread_block_dims"),
    (
        (_half_warp_source(), (8, 16, 1)),
        (
            _half_warp_source().replace(
                "producer_base + 7 < 128",
                "producer_base + 6 < 128",
                1,
            ),
            (16, 8, 1),
        ),
        (
            _half_warp_source().replace(
                "key_index = producer_base + expanded_lane",
                "key_index = (\n"
                "        producer_base + expanded_lane\n"
                "        + cute.arch.thread_idx()[1]\n"
                "    )",
                1,
            ),
            (16, 8, 1),
        ),
    ),
)
def test_half_warp_sharing_near_misses_use_safe_fallback(
    source: str, thread_block_dims: tuple[int, int, int]
) -> None:
    result, changed = _transform(source, thread_block_dims=thread_block_dims)

    assert changed
    assert "half_warp_producer_abi_version" not in result
    assert "ir.VectorType.get([8], cutlass.Uint16.mlir_type)" in result


@pytest.mark.parametrize(
    "source",
    (
        _half_warp_source().replace(
            "producer_base = cutlass.Int32(cute.arch.thread_idx()[0]) * 8",
            "producer_base = cutlass.Int32(cute.arch.thread_idx()[0]) * 8\n"
            "producer_base = cutlass.Int32(0)",
            1,
        ),
        (
            "row_offset = cutlass.Int32(cute.arch.thread_idx()[1]) * 128\n"
            + _half_warp_source()
        )
        .replace(
            "key.iterator + (producer_base",
            "key.iterator + row_offset + (producer_base",
        )
        .replace(
            "query.iterator + (producer_base",
            "query.iterator + row_offset + (producer_base",
        )
        .replace(
            "decay.iterator + (producer_base",
            "decay.iterator + row_offset + (producer_base",
        ),
    ),
)
def test_half_warp_sharing_rejects_hidden_thread_dependent_addresses(
    source: str,
) -> None:
    result, changed = _transform(source, thread_block_dims=(16, 8, 1))

    assert changed
    assert "half_warp_producer_abi_version" not in result
    assert "ir.VectorType.get([8], cutlass.Uint16.mlir_type)" in result


def test_half_warp_sharing_requires_exact_vector_high_bound() -> None:
    source = _half_warp_source().replace("producer_base + 7 < 128", "producer_base < 7")

    result, changed = _transform(source, thread_block_dims=(16, 8, 1))

    assert changed
    assert "half_warp_producer_abi_version" not in result
    assert "producer_base < 7" in result
    assert "ir.VectorType.get([8], cutlass.Uint16.mlir_type)" in result


def test_half_warp_sharing_rejects_thread_dependent_ancestor_alias() -> None:
    source = (
        "row_offset = cutlass.Int32(cute.arch.thread_idx()[1]) * 128\n"
        "if condition:\n"
        + textwrap.indent(
            _half_warp_source()
            .replace(
                "key.iterator + (producer_base",
                "key.iterator + row_offset + (producer_base",
            )
            .replace(
                "query.iterator + (producer_base",
                "query.iterator + row_offset + (producer_base",
            )
            .replace(
                "decay.iterator + (producer_base",
                "decay.iterator + row_offset + (producer_base",
            ),
            "    ",
        )
    )

    result, changed = _transform(source, thread_block_dims=(16, 8, 1))

    assert changed
    assert "half_warp_producer_abi_version" not in result
    assert "ir.VectorType.get([8], cutlass.Uint16.mlir_type)" in result


def test_transactional_clone_supports_extended_ast() -> None:
    module = convert(ast.parse(SOURCE))
    assert isinstance(module, ast.Module)
    row = next(
        node
        for statement in module.body
        for node in ast.walk(statement)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "row"
    )
    assert isinstance(row, ExtendedAST)
    row.__dict__["_helion_async_state_vector"] = "state_values"
    row.__dict__["_helion_async_state_vector_width"] = 8
    row.__dict__["_helion_async_state_valid_predicate"] = None

    result = pack_bf16_recurrences(
        module.body,
        enabled=True,
        fast_math=True,
        target_device_capability=(10, 0),
        uniform_names=frozenset(("key_norm", "query_norm", "scale")),
    )
    code = ast.unparse(
        ast.fix_missing_locations(ast.Module(body=result, type_ignores=[]))
    )

    assert result is not module.body
    assert "_bf16x2_0_abi_version = 1" in code


@pytest.mark.parametrize(
    ("enabled", "fast_math", "capability"),
    (
        (False, True, (10, 0)),
        (True, False, (10, 0)),
        (True, True, None),
        (True, True, (7, 5)),
    ),
)
def test_policy_gates_fail_closed(
    enabled: bool, fast_math: bool, capability: tuple[int, int] | None
) -> None:
    result, changed = _transform(
        enabled=enabled, fast_math=fast_math, capability=capability
    )

    assert not changed
    assert result == ast.unparse(ast.parse(SOURCE))


def test_incomplete_graph_fails_closed() -> None:
    source = SOURCE.replace(
        "(key_fragment[lo], key_fragment[hi]),\n            (decayed_lo, decayed_hi),",
        "(unrelated[lo], unrelated[hi]),\n            (decayed_lo, decayed_hi),",
        1,
    )

    result, changed = _transform(source)

    assert not changed
    assert result == ast.unparse(ast.parse(source))


@pytest.mark.parametrize(
    "source",
    (
        SOURCE.replace(
            "        state_fragment[hi] = state_hi\n",
            "        state_fragment[hi] = state_hi\n        observe(state_hi)\n",
            1,
        ),
        SOURCE.replace(
            "        state_fragment[hi] = state_hi\n",
            "        state_fragment[hi] = state_hi\n"
            "        observed = observe(state_hi)\n",
            1,
        ),
        SOURCE.replace(
            "        state_fragment[hi] = state_hi\n",
            "        state_fragment[hi] = state_hi\n        external[0] = state_hi\n",
            1,
        ),
        SOURCE.replace(
            "        store_values.append("
            "cutlass.BFloat16(stored_hi).bitcast(cutlass.Uint16))\n",
            "        store_values.append("
            "cutlass.BFloat16(stored_hi).bitcast(cutlass.Uint16))\n"
            "        observe(stored_hi)\n",
            1,
        ),
        SOURCE.replace(
            "        store_values.append("
            "cutlass.BFloat16(stored_hi).bitcast(cutlass.Uint16))\n",
            "        store_values.append("
            "cutlass.BFloat16(stored_hi).bitcast(cutlass.Uint16))\n"
            "        store_values[0] = cutlass.Uint16(0)\n",
            1,
        ),
    ),
)
def test_effects_inside_replaced_recurrence_loops_fail_closed(source: str) -> None:
    result, changed = _transform(source)

    assert not changed
    assert result == ast.unparse(ast.parse(source))


@pytest.mark.parametrize(
    "source",
    (
        SOURCE.replace(
            "state_fragment[lo] = state_lo", "state_fragment[lo] = state_hi", 1
        ),
        SOURCE.replace(
            "state_fragment[hi] = state_hi", "state_fragment[hi] = state_lo", 1
        ),
        SOURCE.replace(
            "        state_fragment[hi] = state_hi\n",
            "        state_fragment[hi] = state_hi\n"
            "        state_fragment[lo] = cutlass.BFloat16(0)\n",
            1,
        ),
    ),
)
def test_state_cache_writes_must_preserve_exact_lane_values(source: str) -> None:
    result, changed = _transform(source)

    assert not changed
    assert result == ast.unparse(ast.parse(source))


@pytest.mark.parametrize(
    "source",
    (
        SOURCE.replace("decay_fragment[lo]", "key_fragment[lo]").replace(
            "decay_fragment[hi]", "key_fragment[hi]"
        ),
        SOURCE.replace("state_fragment[lo] = state_lo", "decay_fragment[lo] = state_lo")
        .replace("state_fragment[hi] = state_hi", "decay_fragment[hi] = state_hi")
        .replace("state_lo = state_fragment[lo]", "state_lo = decay_fragment[lo]")
        .replace("state_hi = state_fragment[hi]", "state_hi = decay_fragment[hi]"),
    ),
)
def test_recurrence_caches_must_be_distinct(source: str) -> None:
    result, changed = _transform(source)

    assert not changed
    assert result == ast.unparse(ast.parse(source))


def test_effect_inside_replaced_qdot_loop_fails_closed() -> None:
    source = SOURCE.replace(
        "        (kq_acc_lo, kq_acc_hi),\n    )\n",
        "        (kq_acc_lo, kq_acc_hi),\n    )\n    observe(kq_acc_lo)\n",
        1,
    )

    result, changed = _transform(source)

    assert not changed
    assert result == ast.unparse(ast.parse(source))


@pytest.mark.parametrize("name", ("kq_acc_lo", "kq_lane"))
def test_replaced_qdot_temporaries_cannot_escape(name: str) -> None:
    source = SOURCE + f"\nconsumer = consume({name})\n"

    result, changed = _transform(source)

    assert not changed
    assert result == ast.unparse(ast.parse(source))


@pytest.mark.parametrize(
    "source",
    (
        SOURCE.replace("cutlass.BFloat16(base + delta * kq)", "cutlass.BFloat16(123)"),
        SOURCE.replace(
            "warp_reduction_sum(base_lane, threads_in_group=16)",
            "warp_reduction_sum(base_lane, threads_in_group=8)",
        ),
        SOURCE.replace(
            "(decayed_lo, decayed_hi),\n"
            "            (query_fragment[lo], query_fragment[hi]),",
            "(unrelated_lo, unrelated_hi),\n"
            "            (query_fragment[lo], query_fragment[hi]),",
            1,
        ),
        SOURCE.replace(
            "        prediction_acc_hi\n    )",
            "        prediction_acc_hi\n    ) + 1.0",
            1,
        ),
        SOURCE.replace(
            "kq_lane = cutlass.Float32(kq_acc_lo) + cutlass.Float32(kq_acc_hi)",
            "kq_lane = (\n"
            "    cutlass.Float32(kq_acc_lo) + cutlass.Float32(kq_acc_hi) + 1.0\n"
            ")",
            1,
        ),
    ),
)
def test_changed_output_algebra_fails_closed(source: str) -> None:
    result, changed = _transform(source)

    assert not changed
    assert result == ast.unparse(ast.parse(source))


def test_non_bf16_output_fails_closed() -> None:
    source = SOURCE.replace(
        "cutlass.BFloat16(base + delta * kq)", "base + delta * kq", 1
    )

    result, changed = _transform(source)

    assert not changed
    assert result == ast.unparse(ast.parse(source))


def test_fp16_rounded_output_fails_closed() -> None:
    source = SOURCE.replace(
        "cutlass.BFloat16(base + delta * kq)",
        "cutlass.BFloat16(cutlass.Float16(base + delta * kq))",
        1,
    )

    result, changed = _transform(source)

    assert not changed
    assert result == ast.unparse(ast.parse(source))


def test_non_bf16_state_bits_fail_closed() -> None:
    source = SOURCE.replace(
        "cutlass.Uint16(state_values[lo]).bitcast(cutlass.BFloat16)",
        "cutlass.Uint16(state_values[lo]).bitcast(cutlass.Float16)",
        1,
    ).replace(
        "cutlass.Uint16(state_values[hi]).bitcast(cutlass.BFloat16)",
        "cutlass.Uint16(state_values[hi]).bitcast(cutlass.Float16)",
        1,
    )

    result, changed = _transform(source)

    assert not changed
    assert result == ast.unparse(ast.parse(source))


@pytest.mark.parametrize(
    ("old", "new"),
    (
        (
            "state_fragment = cute.make_rmem_tensor(8, cutlass.BFloat16)",
            "state_fragment = cute.make_rmem_tensor(8, cutlass.Float16)",
        ),
        (
            "key_fragment = cute.make_rmem_tensor(8, cutlass.Float32)",
            "key_fragment = cute.make_rmem_tensor(8, cutlass.Float16)",
        ),
        (
            "query_fragment = cute.make_rmem_tensor(8, cutlass.Float32)",
            "query_fragment = cute.make_rmem_tensor(8, cutlass.Float16)",
        ),
        (
            "decay_fragment = cute.make_rmem_tensor(8, cutlass.Float32)",
            "decay_fragment = cute.make_rmem_tensor(8, cutlass.Float16)",
        ),
    ),
)
def test_cache_allocation_dtype_must_match_packed_lowering(old: str, new: str) -> None:
    source = SOURCE.replace(old, new, 1)

    result, changed = _transform(source)

    assert not changed
    assert result == ast.unparse(ast.parse(source))


@pytest.mark.parametrize(
    "cache",
    ("key_fragment", "query_fragment", "decay_fragment"),
)
def test_bf16_source_cache_allocation_is_supported(cache: str) -> None:
    source = SOURCE.replace(
        f"{cache} = cute.make_rmem_tensor(8, cutlass.Float32)",
        f"{cache} = cute.make_rmem_tensor(8, cutlass.BFloat16)",
        1,
    )

    _result, changed = _transform(source)

    assert changed


def test_fp16_cache_value_fails_closed() -> None:
    source = SOURCE.replace(
        "key_fragment[expanded_lane] = key_value * key_norm",
        "key_fragment[expanded_lane] = cutlass.Float16(key_value * key_norm)",
        1,
    )

    result, changed = _transform(source)

    assert not changed
    assert result == ast.unparse(ast.parse(source))


def test_fp16_separate_cache_expression_fails_closed() -> None:
    source = _separate_cache_source()
    for index in ("kq_lo", "kq_hi", "lo", "hi"):
        source = source.replace(
            f"key_fragment[{index}] * key_norm",
            f"cutlass.Float16(key_fragment[{index}] * key_norm)",
        )

    result, changed = _transform(source)

    assert not changed
    assert result == ast.unparse(ast.parse(source))


@pytest.mark.parametrize(
    "cache",
    ("key_fragment", "query_fragment", "decay_fragment"),
)
def test_separate_cache_allocation_dtype_must_be_bf16_or_float32(cache: str) -> None:
    source = _separate_cache_source().replace(
        f"{cache} = cute.make_rmem_tensor(8, cutlass.Float32)",
        f"{cache} = cute.make_rmem_tensor(8, cutlass.Float16)",
        1,
    )

    result, changed = _transform(source)

    assert not changed
    assert result == ast.unparse(ast.parse(source))


@pytest.mark.parametrize("lane", ("lo", "hi"))
def test_fp16_rounded_state_store_fails_closed(lane: str) -> None:
    source = SOURCE.replace(
        f"stored_{lane} = cutlass.BFloat16(updated_{lane})",
        f"stored_{lane} = cutlass.Float16(updated_{lane})",
        1,
    )

    result, changed = _transform(source)

    assert not changed
    assert result == ast.unparse(ast.parse(source))


@pytest.mark.parametrize(
    "source",
    (
        SOURCE.replace(
            "kq_acc_lo = cutlass.Float32(0)",
            "kq_acc_lo = cutlass.Float32(1)",
            1,
        ),
        SOURCE.replace(
            "prediction_acc_lo = cutlass.Float32(0)",
            "prediction_acc_lo = cutlass.Float32(1)",
            1,
        ),
        SOURCE.replace(
            "base_acc_hi = cutlass.Float32(0)",
            "base_acc_hi = cutlass.Float32(1)",
            1,
        ),
    ),
)
def test_nonzero_reduction_initializer_fails_closed(source: str) -> None:
    result, changed = _transform(source)

    assert not changed
    assert result == ast.unparse(ast.parse(source))


@pytest.mark.parametrize(
    "source",
    (
        SOURCE.replace(
            "(key_fragment[lo], key_fragment[hi]),",
            "(key_fragment[lo] + 1, key_fragment[hi]),",
            1,
        ),
        SOURCE.replace(
            "(decay_lo, decay_hi),",
            "(decay_lo + 1, decay_hi),",
            1,
        ),
        SOURCE.replace(
            "(key_fragment[kq_lo], key_fragment[kq_hi]),",
            "(key_fragment[kq_lo] + 1, key_fragment[kq_hi]),",
            1,
        ),
    ),
)
def test_noncanonical_pair_expression_fails_closed(source: str) -> None:
    result, changed = _transform(source)

    assert not changed
    assert result == ast.unparse(ast.parse(source))


def test_combined_expansion_rejects_extra_consumer_scale() -> None:
    source = SOURCE.replace(
        "(key_fragment[kq_lo], key_fragment[kq_hi])",
        "(key_fragment[kq_lo] * key_norm, key_fragment[kq_hi] * key_norm)",
    ).replace(
        "(key_fragment[lo], key_fragment[hi])",
        "(key_fragment[lo] * key_norm, key_fragment[hi] * key_norm)",
    )

    result, changed = _transform(source)

    assert not changed
    assert result == ast.unparse(ast.parse(source))


def test_noncanonical_shuffle_lane_fails_closed() -> None:
    source = SOURCE.replace(
        "shuffle_sync(key_source[source_element], source_lane)",
        "shuffle_sync(key_source[source_element], source_lane + 1)",
        1,
    )

    result, changed = _transform(source)

    assert not changed
    assert result == ast.unparse(ast.parse(source))


@pytest.mark.parametrize(
    "source",
    (
        SOURCE.replace(
            "    source_element = expanded_lane % 4\n",
            "    source_element = expanded_lane % 4\n    observe(source_element)\n",
            1,
        ),
        SOURCE.replace(
            "    source_element = expanded_lane % 4\n",
            "    source_element = expanded_lane % 4\n    escaped = source_element\n",
            1,
        )
        + "\nconsumer = consume(escaped)\n",
    ),
)
def test_replaced_cache_expansion_effects_and_escapes_fail_closed(
    source: str,
) -> None:
    result, changed = _transform(source)

    assert not changed
    assert result == ast.unparse(ast.parse(source))


def test_reassigned_nonuniform_scale_fails_closed() -> None:
    source = SOURCE.replace(
        "for expanded_lane in cutlass.range_constexpr(8):",
        "key_norm = cute.arch.thread_idx()[0]\n"
        "for expanded_lane in cutlass.range_constexpr(8):",
        1,
    )

    result, changed = _transform(source)

    assert not changed
    assert result == ast.unparse(ast.parse(source))


def test_nested_reassigned_nonuniform_scale_fails_closed() -> None:
    source = (
        "if condition:\n"
        "    query_norm = cutlass.Float32(cute.arch.thread_idx()[1])\n" + SOURCE
    )

    result, changed = _transform(source)

    assert not changed
    assert result == ast.unparse(ast.parse(source))


def test_ancestor_reassigned_nonuniform_scale_fails_closed() -> None:
    source = "if condition:\n" + textwrap.indent(
        "query_norm = cutlass.Float32(cute.arch.thread_idx()[1])\n" + SOURCE,
        "    ",
    )

    result, changed = _transform(source)

    assert not changed
    assert result == ast.unparse(ast.parse(source))


def test_opaque_scale_call_is_not_assumed_uniform() -> None:
    source = "query_norm = scale.opaque()\n" + SOURCE

    result, changed = _transform(source)

    assert not changed
    assert result == ast.unparse(ast.parse(source))


def test_state_store_guard_is_preserved() -> None:
    source = SOURCE.replace(
        "    _cute_store_u16x8_l2_evict_last(state.iterator + row_index, store_values)",
        "    if valid_state:\n"
        "        _cute_store_u16x8_l2_evict_last(\n"
        "            state.iterator + row_index, store_values\n"
        "        )",
    )

    result, changed = _transform(source)

    assert changed
    assert "if valid_state:" in result
    assert "_cute_store_u32x4_l2_evict_last(" in result


def test_generated_names_use_a_fresh_prefix() -> None:
    source = (
        "_bf16x2_0_output = sentinel\n"
        "_bf16x2_0_abi_version = sentinel_abi\n"
        + SOURCE
        + "\nconsumer = consume(_bf16x2_0_output, _bf16x2_0_abi_version)\n"
    )

    result, changed = _transform(source)

    assert changed
    assert "_bf16x2_1_prediction = cutlass.Uint32(0)" in result
    assert "_bf16x2_1_abi_version = 1" in result
    assert "consumer = consume(_bf16x2_0_output, _bf16x2_0_abi_version)" in result


@pytest.mark.parametrize(
    "source",
    (
        SOURCE + "\nconsumer = consume(kq)\n",
        SOURCE + "\nconsumer = consume(key_fragment[0])\n",
        SOURCE.replace(
            "    store_values = []",
            "    extra = value + 7\n    store_values = []",
            1,
        ).replace(
            "    _cute_store_u16x8_l2_evict_last(state.iterator + row_index, store_values)",
            "    _cute_store_u16x8_l2_evict_last(state.iterator + row_index, store_values)\n"
            "    consumer = consume(extra)",
            1,
        ),
        SOURCE.replace(
            "    store_values = []",
            "    observe(value)\n    store_values = []",
            1,
        ),
    ),
)
def test_values_that_escape_rewritten_region_fail_closed(source: str) -> None:
    result, changed = _transform(source)

    assert not changed
    assert result == ast.unparse(ast.parse(source))


@pytest.mark.parametrize(
    "suffix",
    (
        "\nconsumer = consume(pipe_values[7])\n",
        (
            "\npipe_values = cute.arch.load(\n"
            "    other_pointer, ir.VectorType.get([8], cutlass.Uint16.mlir_type)\n"
            ")\nconsumer = consume(pipe_values[7])\n"
        ),
    ),
)
def test_pipeline_vector_cannot_escape_or_have_an_unrelated_load(suffix: str) -> None:
    source = SOURCE + suffix

    result, changed = _transform(source)

    assert not changed
    assert result == ast.unparse(ast.parse(source))
