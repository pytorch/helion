from __future__ import annotations

import ast
from dataclasses import replace
import operator
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from . import test_cute_chained_vector_expression as frontier_fixture
from . import test_cute_chained_vector_group as group_fixture
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chained_vector_expression as vector
from helion._compiler.cute.chained_execution import ChainedExecution
from helion._compiler.cute.chained_vector_group import emit_vector_group


def _raw_frontier(*, mode="admitted", **options):
    captured = {}

    def observe(cg, plan, boundaries, node, **kwargs):
        row, base, element = "frontier_row", "frontier_base", "frontier_element"
        coords = kwargs["coordinates"](row, f"({base} + {element})")
        probe = chain._Expression(cg, plan, boundaries)
        probe.coordinate_names.update((row, base, element))
        probe.value(node, coords)
        raw = {
            leaf: f"raw_image_{index}"
            for index, (leaf, _, _, _) in enumerate(probe.loaded_inputs)
        }
        assert raw
        selected = {**boundaries, **raw}
        admitted = raw
        if mode == "missing":
            admitted = {}
        elif mode == "wrong_name":
            admitted = dict.fromkeys(raw, "wrong_image")
        elif mode == "computed":
            selected = {**boundaries, node: "computed_image"}
            admitted = {node: "computed_image"}
        elif mode == "unrelated":
            admitted = {node: "unrelated_image"}
        elif mode in ("default", "default_none", "default_empty"):
            if mode != "default":
                kwargs["raw_boundaries"] = None if mode == "default_none" else {}
            original = vector.emit_vector_expression(
                cg, plan, boundaries, node, **kwargs
            )
            captured["lines"] = original
            return original
        elif mode == "scalar":
            kwargs["vector_store"] = False
        captured["old"] = vector.emit_vector_expression(
            cg, plan, selected, node, **kwargs
        )
        captured["lines"] = vector.emit_vector_expression(
            cg, plan, selected, node, raw_boundaries=admitted, **kwargs
        )
        return captured["lines"]

    with patch.object(frontier_fixture, "emit_vector_expression", observe):
        frontier_fixture._frontier_capture(**options)
    return captured


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("host_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_exact_raw_images_preserve_original_vector_publication(dtype, host_dtype):
    result = _raw_frontier(dtype=dtype, host_dtype=host_dtype)
    assert result["old"] is None
    assert result["lines"] is not None
    source = "\n".join(result["lines"])
    ast.parse(source)
    assert "raw_image_0[frontier_row, frontier_base + frontier_element]" in source
    assert "cute.domain_offset((32, 0), prepared_image)" in source
    assert "partition_D" in source and "cute.copy(frontier_copy" in source
    assert "_leaf_" not in source and "_vectorized" not in source
    assert "cute.math.exp2" in source
    assert "cutlass.Float16" in source and "cutlass.Float32" in source
    assert "sync_threads" not in source and "arrive_and_wait" not in source


@pytest.mark.parametrize("mode", ["missing", "wrong_name", "computed", "unrelated"])
def test_arbitrary_boundary_names_or_computed_values_do_not_grant_admission(mode):
    assert _raw_frontier(mode=mode)["lines"] is None


@pytest.mark.parametrize(
    "options",
    [
        {"varying": True},
        {"transposed": True},
        {"shape": (16, 24)},
        {"shape": (16, 7)},
        {"shape": (16, 512), "execution": ChainedExecution(96)},
    ],
)
def test_original_mask_coordinate_and_vector_geometry_guards_remain(options):
    assert _raw_frontier(**options)["lines"] is None


@pytest.mark.parametrize("sigmoid", [False, True])
@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize("row", [5, 17])
def test_masked_zero_and_padding_keep_original_pointwise_semantics(
    sigmoid, scalar, row
):
    result = _raw_frontier(
        mode="scalar" if scalar else "admitted",
        sigmoid=sigmoid,
        shape=(19, 16),
        execution=ChainedExecution(384),
    )
    assert result["lines"] is not None
    tree = ast.parse("\n".join(result["lines"]))
    reads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "raw_image_0"
    ]
    assert reads
    # The exact raw value (already masked by its publisher) feeds the original
    # FP32/FP16 rounding and exp/sigmoid, not a newly masked final result.
    source = "\n".join(result["lines"])
    assert "cutlass.Float16" in source and "cute.math.exp2" in source
    assert "else cutlass.BFloat16(0)" in source
    pointwise = next(
        item
        for item in ast.walk(tree)
        if isinstance(item, ast.For) and ast.unparse(item.target) == "frontier_element"
    )
    values = [float("nan")] * 8
    target = {}
    namespace = {
        "cutlass": SimpleNamespace(
            Float32=lambda value: torch.tensor(value, dtype=torch.float32).item(),
            Float16=lambda value: torch.tensor(value, dtype=torch.float16).item(),
            BFloat16=lambda value: torch.tensor(value, dtype=torch.bfloat16).item(),
            range_constexpr=range,
        ),
        "cute": SimpleNamespace(
            math=SimpleNamespace(
                exp2=lambda value, **kwargs: 2.0**value,
                rcp=lambda value, **kwargs: 1 / value,
            )
        ),
        "operator": operator,
        "raw_image_0": torch.zeros((16, 16), dtype=torch.bfloat16),
        "frontier_values": values,
        "prepared_image": target,
        "frontier_row": row,
        "frontier_base": 0,
    }
    exec(
        compile(ast.Module(body=[pointwise], type_ignores=[]), "<raw-vector>", "exec"),
        namespace,
    )
    expected = 0 if row >= 16 else 0.5 if sigmoid else 1
    if scalar:
        assert target == {(32 + row, column): expected for column in range(8)}
        assert "partition_D" not in source and "cute.copy(" not in source
    else:
        assert values == [expected] * 8


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_absent_raw_provenance_preserves_existing_source_exactly(dtype):
    original = _raw_frontier(mode="default", dtype=dtype)["lines"]
    assert original is not None
    assert original == _raw_frontier(mode="default_none", dtype=dtype)["lines"]
    assert original == _raw_frontier(mode="default_empty", dtype=dtype)["lines"]


def test_failed_original_vector_proof_keeps_scalar_fallback():
    with patch.object(vector, "prove_vector_leaf", return_value=None):
        assert _raw_frontier()["lines"] is None


def test_original_proof_rechecks_dtype_mask_indices_and_strides():
    calls = []
    original = vector.prove_vector_leaf

    def record(*args, **kwargs):
        calls.append((args, kwargs.copy()))
        return original(*args, **kwargs)

    with patch.object(vector, "prove_vector_leaf", record):
        result = _raw_frontier(host_dtype=torch.float16)
    assert result["lines"] is not None and calls
    assert any(
        kwargs["dtype"] == torch.float16
        and kwargs["shape"] == (3, 19, 16)
        and kwargs["strides"] == (304, 16, 1)
        and kwargs["mask"] is not None
        and kwargs["element"] == "frontier_element"
        for _, kwargs in calls
    )


def _raw_group(*, mode="admitted", **options):
    captured = {}

    def observe(cg, plan, boundaries, outputs, **kwargs):
        raw = {}
        probes = []
        for item in outputs:
            probe = chain._Expression(cg, plan, boundaries)
            probe.coordinate_names.update(
                ("joined_row", "joined_base", "joined_element")
            )
            probe.value(
                item.node,
                item.coordinates("joined_row", "(joined_base + joined_element)"),
            )
            probes.append(probe)
            for leaf, _, _, _ in probe.loaded_inputs:
                raw.setdefault(leaf, f"raw_image_{len(raw)}")
        assert raw
        selected = {**boundaries, **raw}
        if mode == "residual_host":
            selected = boundaries
        elif mode == "no_work":
            common = set(probes[0].memo).intersection(probes[1].memo)
            selected = {
                **selected,
                **{
                    node: f"computed_image_{index}"
                    for index, (node, _) in enumerate(common)
                    if chain._pointwise_inputs(node)
                },
            }
        lines = emit_vector_group(cg, plan, selected, outputs, **kwargs)
        separate = [
            vector.emit_vector_expression(
                cg,
                plan,
                selected,
                item.node,
                shape=kwargs["shape"],
                coordinates=item.coordinates,
                offset=item.offset,
                tag=f"ordinary_{index}",
                target=item.target,
                final_value=item.final_value,
                vector_store=item.vector_store,
                execution=kwargs["execution"],
                raw_boundaries=raw,
            )
            for index, item in enumerate(outputs)
        ]
        captured.update(lines=lines, separate=separate)
        return lines

    with patch.object(group_fixture, "emit_vector_group", observe):
        group_fixture._capture(**options)
    return captured


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("host_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("scalar", [False, True])
def test_materialized_inputs_keep_exact_coordinate_cse_and_typed_outputs(
    dtype, host_dtype, scalar
):
    result = _raw_group(dtype=dtype, host_dtype=host_dtype, scalar=scalar)
    lines = result["lines"]
    assert lines is not None
    assert all(item is not None for item in result["separate"])
    assert group_fixture._expanded_outputs(lines, "joined_element") == [
        group_fixture._expanded_outputs(item, f"ordinary_{index}_element")[0]
        for index, item in enumerate(result["separate"])
    ]
    source = "\n".join(lines)
    assert source.count("cute.math.exp2(") == 1
    assert "_leaf_" not in source and "_vectorized" not in source
    assert "raw_image_0[" in source
    assert ("partition_D" in source) is not scalar
    assert "sync_threads" not in source and "arrive_and_wait" not in source


def test_materialized_inputs_without_shared_original_work_still_reject():
    assert _raw_group(mode="no_work")["lines"] is None


def test_no_vector_with_remaining_scalar_host_loads_retains_old_rejection():
    assert _raw_group(mode="residual_host", varying=True)["lines"] is None


def test_original_varying_host_mask_is_safe_after_exact_materialization():
    # Group CSE consumes the published exact value. Single-output preservation
    # remains stricter: the original varying host mask had no vector route.
    result = _raw_group(varying=True)
    assert result["lines"] is not None
    assert result["separate"] == [None, None]


def test_materialized_group_does_not_merge_different_coordinates():
    def change(outputs, boundaries, plan):
        right = outputs[1]
        return [
            outputs[0],
            replace(
                right,
                coordinates=lambda row, column: right.coordinates(
                    row, f"({column} + 0)"
                ),
            ),
        ], boundaries

    assert _raw_group(change=change)["lines"] is None
