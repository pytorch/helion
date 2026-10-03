from __future__ import annotations

import ast

import numpy as np
import pytest

from test.test_cute_resident_sequence import _execute
from test.test_cute_resident_sequence import _indent
from test.test_cute_resident_sequence import _reference
from test.test_cute_resident_sequence import _rewrite
from test.test_cute_resident_sequence import _source
from test.test_cute_resident_sequence import _Tensor

from helion._compiler.cute.resident_sequence_cache import memory_key


@pytest.mark.parametrize("coordinate", ["origin", "lane", "vector"])
def test_cache_key_distinguishes_coordinates_from_real_boundary_names(
    coordinate: str,
) -> None:
    arguments = {"origin": "offset", "lane": "lane", "vector": "vec"}
    actual = arguments[coordinate]
    boundary = f"__sequence_{coordinate}"

    def key(index: str) -> str | None:
        return memory_key(
            ast.parse(
                f"(x.iterator + cutlass.Int32({index})).load()", mode="eval"
            ).body,
            [],
            **arguments,
            rename_groups={},
        )

    assert key(actual) is not None
    assert key(boundary) is not None
    assert key(actual) != key(boundary)


def test_colliding_boundary_scalar_keeps_the_original_consumer_load() -> None:
    consumer = _consumer(1).replace(
        "copied_value =",
        "load_column = consume_offset + cutlass.Int32(__sequence_lane) * 32 + cutlass.Int32(cute.arch.thread_idx()[0])\n        copied_value =",
    )
    consumer = consumer.replace(
        "row * width + column_copy).load()) if column_copy < width",
        "row * width + load_column).load()) if load_column < width",
    )
    _, lowered = _rewrite(_source() + consumer)
    values = np.arange(2 * 128, dtype=np.float32).reshape(2, 128)
    copied = _Tensor(np.full_like(values, np.nan))
    # Supply the ordinary scalar boundary to the existing CPU SIMT model.
    _execute(
        "__sequence_lane = cutlass.Int32(1)\n" + lowered,
        values,
        rows=2,
        threads=32,
        groups=2,
        initial=(3.0, -0.5, 1.0),
        copied=copied,
    )
    np.testing.assert_array_equal(copied.values, values[:, np.arange(128) % 32 + 32])
    assert np.all(copied.writes == 1)
    assert "if sequence_initialized" not in lowered


def test_coordinate_keys_stay_equal_with_colliding_names_and_renamed_sweeps() -> None:
    def key(origin: str, lane: str, vector: str) -> str | None:
        return memory_key(
            ast.parse(
                f"(x.iterator + {origin} + {lane} + {vector} + __sequence_origin + __sequence_lane + __sequence_vector).load()",
                mode="eval",
            ).body,
            [],
            origin=origin,
            lane=lane,
            vector=vector,
            rename_groups={},
        )

    assert key("offset", "lane", "vec") is not None
    assert key("offset", "lane", "vec") == key("next_offset", "next_lane", "next_vec")


def test_later_real_binding_cannot_resolve_an_unknown_reaching_definition() -> None:
    prefix = ast.parse(
        "if condition:\n    width = 3\n__sequence_unknown_0_width = 7"
    ).body
    assert (
        memory_key(
            ast.parse("(x.iterator + width).load()", mode="eval").body,
            prefix,
            origin="offset",
            lane="lane",
            vector=None,
            rename_groups={},
        )
        is None
    )


def _consumer(vector: int) -> str:
    if vector == 1:
        body = """column_copy = consume_offset + cutlass.Int32(consume_lane) * 32 + cutlass.Int32(cute.arch.thread_idx()[0])
copied_value = cutlass.Float32((x.iterator + row * width + column_copy).load()) if column_copy < width and row < rows else cutlass.Float32(0)
if column_copy < width and row < rows:
    (copied.iterator + row * width + column_copy).store(copied_value)
"""
    else:
        body = f"""base_copy = consume_offset + (cutlass.Int32(consume_lane) * 32 + cutlass.Int32(cute.arch.thread_idx()[0])) * {vector}
packet_copy = cute.arch.load(x.iterator + row * width + base_copy, ir.VectorType.get([{vector}], cutlass.Uint32.mlir_type))
for consume_vec in cutlass.range_constexpr({vector}):
    column_copy = base_copy + cutlass.Int32(consume_vec)
    copied_value = cutlass.Uint32(packet_copy[consume_vec]).bitcast(cutlass.Float32)
    (copied.iterator + row * width + column_copy).store(copied_value)
"""
    return (
        f"for consume_offset in range(cutlass.Int32(0), cutlass.Int32(width), cutlass.Int32({128 * vector})):\n    for consume_lane in range(4):\n"
        + _indent(body, 8)
        + "\n"
    )


@pytest.mark.parametrize("columns", [0, 1, 127, 128, 129, 255, 257])
@pytest.mark.parametrize("empty_producer", [False, True])
def test_last_tile_cache_preserves_empty_partial_and_multiple_sweeps(
    columns: int, empty_producer: bool
) -> None:
    source = _source()
    if empty_producer:
        source = source.replace("cutlass.Int32(width),", "cutlass.Int32(0),", 1)
    source += _consumer(1)
    _, lowered = _rewrite(source)
    assert "if sequence_initialized" in lowered
    for seed in (83, 941):
        rng = np.random.default_rng(seed)
        values = rng.normal(size=(2, columns)).astype(np.float32)
        if columns > 2:
            values[0, 0] = -0.0
            values[0, 1] = np.nan
        copied = _Tensor(np.full_like(values, np.nan))
        _, tensor = _execute(
            lowered,
            values,
            rows=2,
            threads=32,
            groups=2,
            initial=(3.0, -0.5, 1.0),
            copied=copied,
        )
        np.testing.assert_array_equal(
            copied.values.view(np.uint32), values.view(np.uint32)
        )
        assert np.all(copied.writes == 1)
        expected_reads = np.ones_like(values, dtype=np.int64)
        if columns and not empty_producer:
            expected_reads[:, : ((columns - 1) // 128) * 128] = 2
        np.testing.assert_array_equal(
            tensor.reads.reshape(values.shape), expected_reads
        )


@pytest.mark.parametrize("columns", [512, 1024])
def test_vector_cache_removes_only_the_reused_packet_load(columns: int) -> None:
    source = _source(vector=4) + _consumer(4)
    _, lowered = _rewrite(source, vector=4)
    assert "if sequence_initialized" in lowered
    # One load in the producer, one in the original cache-miss branch.
    assert lowered.count("cute.arch.load(") == 2
    rng = np.random.default_rng(429)
    values = rng.normal(size=(2, columns)).astype(np.float32)
    copied = _Tensor(np.full_like(values, np.nan))
    _, tensor = _execute(
        lowered,
        values,
        rows=2,
        threads=32,
        groups=2,
        initial=(3.0, -0.5, 1.0),
        copied=copied,
    )
    np.testing.assert_array_equal(copied.values.view(np.uint32), values.view(np.uint32))
    assert np.all(copied.writes == 1)
    expected = np.ones_like(values, dtype=np.int64)
    expected[:, :-512] = 2
    np.testing.assert_array_equal(tensor.reads.reshape(values.shape), expected)


@pytest.mark.parametrize(
    "change", ["mask", "dtype", "coordinate", "metadata", "carried_mask"]
)
def test_cache_requires_identical_values_masks_and_reaching_definitions(
    change: str,
) -> None:
    producer, consumer = _source(), _consumer(1)
    between = ""
    if change == "mask":
        consumer = consumer.replace(
            "else cutlass.Float32(0)", "else cutlass.Float32(7)"
        )
    elif change == "dtype":
        consumer = consumer.replace(
            "copied_value = cutlass.Float32(", "copied_value = cutlass.Float64("
        )
    elif change == "coordinate":
        consumer = consumer.replace(
            "row * width + column_copy", "row * width + column_copy + 1"
        )
    elif change == "metadata":
        between = "for change in range(1):\n    width = cutlass.Int32(9)\n"
    elif change == "carried_mask":
        producer = producer.replace(
            "if column < width and row < rows",
            "if column < width and row < rows and mean > 0",
        )
        consumer = consumer.replace(
            "if column_copy < width and row < rows else",
            "if column_copy < width and row < rows and mean > 0 else",
        )
    _, lowered = _rewrite(producer + between + consumer)
    assert "if sequence_initialized" not in lowered


def test_snapshot_keeps_old_carry_in_a_later_lane_dependent_recipe() -> None:
    source = _source().replace(
        "if column < width else cutlass.Float32(0)",
        "if column < width and mean > 0 else cutlass.Float32(0)",
    )
    _, lowered = _rewrite(source)
    assert "sequence_carry_input" in lowered
    values = np.linspace(-9, -1, 2 * 128, dtype=np.float32).reshape(2, 128)
    initial = (3.0, 0.75, 1.0)
    actual, _ = _execute(lowered, values, rows=2, threads=32, groups=2, initial=initial)
    expected = _reference(values, 128, initial)
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-5)


def test_proof_expansion_has_a_bound() -> None:
    prefix = ast.parse(
        "value = cutlass.Int32(1)\n"
        + "\n".join(
            f"h{i} = {('value' if i == 0 else f'h{i - 1}')} * {('value' if i == 0 else f'h{i - 1}')} + 1"
            for i in range(30)
        )
    ).body
    expression = ast.parse("(x.iterator + h29).load()", mode="eval").body
    assert (
        memory_key(
            expression,
            prefix,
            origin="offset",
            lane="lane",
            vector=None,
            rename_groups={},
        )
        is None
    )


@pytest.mark.parametrize("policy", ["streaming", "first", "last", "l2_last"])
def test_readonly_cache_hints_preserve_vector_reuse(policy: str) -> None:
    producer, consumer = _source(vector=4), _consumer(4)
    if policy == "l2_last":
        producer = producer.replace("cute.arch.load(", "_cute_load_l2_evict_last(")
    else:
        hint = (
            "cop='cs'"
            if policy == "streaming"
            else f"level1_eviction_priority='evict_{policy}'"
        )
        producer = producer.replace(
            "cutlass.Uint32.mlir_type))", f"cutlass.Uint32.mlir_type), {hint})"
        )
    _, lowered = _rewrite(producer + consumer, vector=4)
    assert "if sequence_initialized" in lowered
    values = np.random.default_rng(921).normal(size=(2, 512)).astype(np.float32)
    copied = _Tensor(np.full_like(values, np.nan))
    _, tensor = _execute(
        lowered,
        values,
        rows=2,
        threads=32,
        groups=2,
        initial=(3.0, -0.5, 1.0),
        copied=copied,
    )
    np.testing.assert_array_equal(copied.values.view(np.uint32), values.view(np.uint32))
    assert np.all(tensor.reads == 1)
    assert np.all(copied.writes == 1)
