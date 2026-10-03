from __future__ import annotations

import ast
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch
from torch.fx import Graph

from ._cute_aux import _cpu_codegen
from .test_cute_chained_vector_group import _args
from .test_cute_chained_vector_group import _config
from .test_cute_chained_vector_group import _outputs
from .test_cute_chained_vector_group import _shared_loop
from helion import exc
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chained_tcgen_stage as stage_module
from helion._compiler.cute.chained_execution import ChainedExecution
from helion._compiler.cute.chained_pointwise_unroll import PointwiseUnroll
from helion._compiler.cute.chained_vector_expression import emit_vector_expression
from helion._compiler.cute.chained_vector_group import emit_materialized_group
from helion._compiler.cute.chained_vector_group import emit_vector_group
from helion._compiler.cute.chained_vector_native import NativeReadInputs
from helion._compiler.cute.chained_vector_native import emit_native_inputs
from helion._compiler.cute.chained_vector_ownership import plan_vector_ownership

DTYPES = {
    torch.bfloat16: "cutlass.BFloat16",
    torch.float16: "cutlass.Float16",
    torch.float32: "cutlass.Float32",
}


def _input(node, *, shape=(32, 128), dtype=torch.bfloat16, index=7) -> Any:
    # Unit-test the emitter's consumption of the late proof. The separate
    # NativeReadInput tests own graph/frame/lifetime admission; no production
    # factory or lifetime proof is bypassed in a generated kernel here.
    return SimpleNamespace(
        node=node,
        tensor="original_full_native_alias",
        full_shape=(64, 128),
        shape=shape,
        row_offset=32,
        dtype=dtype,
        index=index,
        matches=Mock(return_value=True),
    )


def _case(dtype=torch.bfloat16, *, shape=(32, 128), tile_columns=32):
    graph = Graph()
    node = graph.placeholder("original_typed_image")
    node.meta["val"] = torch.empty(shape, dtype=dtype)
    source = _input(node, shape=shape, dtype=dtype)
    boundaries = {node: source.tensor}
    coords = ("group_row", "(group_base + group_element)")
    probe: Any = SimpleNamespace(
        memo={
            (node, coords): chain._materialized_value(
                source.tensor, shape, coords, DTYPES[dtype]
            )
        },
        boundaries=boundaries,
        fragments={},
    )
    ownership = plan_vector_ownership(shape, 128, tile_columns=tile_columns)
    assert ownership is not None
    plan: Any = SimpleNamespace()
    return plan, boundaries, source, probe, ownership


def _emit(case, *, probes=None, sources=None, ownership=None, execution=None):
    plan, boundaries, source, probe, original_ownership = case
    inputs = NativeReadInputs((source,) if sources is None else sources)
    with (
        patch.object(
            CompileEnvironment,
            "current",
            return_value=SimpleNamespace(
                backend=SimpleNamespace(dtype_str=DTYPES.__getitem__)
            ),
        ),
        patch.object(
            chain, "_shape", side_effect=lambda node: tuple(node.meta["val"].shape)
        ),
    ):
        result = emit_native_inputs(
            plan,
            boundaries,
            (probe,) if probes is None else probes,
            inputs,
            original_ownership if ownership is None else ownership,
            "group",
            "group_row",
            "group_base",
            "group_element",
            execution or ChainedExecution(128, thread="prep_thread"),
        )
    assert not inputs.activated
    return result


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("tile_columns", (0, 32, 64))
def test_exact_guarded_boundary_wrapper_and_original_full_alias(dtype, tile_columns):
    case = _case(dtype, tile_columns=tile_columns)
    plan, boundaries, source, probe, ownership = case
    result = _emit(case)
    assert result is not None
    source.matches.assert_called_once_with(plan, boundaries)
    assert "partition_S(original_full_native_alias)" in result.setup[2]
    assert "get_slice(prep_thread)" in result.setup[1]
    assert not any("domain_offset" in line for line in result.setup)
    key = next(iter(probe.memo))
    assert result.replacements[key] == probe.memo[key].replace(
        "original_full_native_alias[group_row, (group_base + group_element)]",
        "group_input_7_values[group_element]",
    )
    assert result.loads == (
        (
            "cute.copy(group_input_7_copy, group_input_7_source["
            f"{ownership.copy_indices('group_step')}], group_input_7_values)"
        ),
    )
    assert all(
        "barrier" not in line and "alloc_smem" not in line for line in result.setup
    )


@pytest.mark.parametrize(
    "reason",
    (
        "fragment",
        "unused",
        "coordinate",
        "mixed_coordinate",
        "tensor",
        "stale",
        "shape",
        "dtype",
        "full_shape",
        "duplicate_index",
        "duplicate_node",
        "bound_read",
        "ownership",
        "execution",
    ),
)
def test_native_probes_fail_closed_without_activation(reason):
    case = _case()
    plan, boundaries, source, probe, ownership = case
    key = next(iter(probe.memo))
    sources = None
    execution = None
    if reason == "fragment":
        probe.fragments[source.node] = (key[1], "register_value")
    elif reason == "unused":
        probe.memo.clear()
    elif reason in ("coordinate", "mixed_coordinate"):
        if reason == "coordinate":
            probe.memo.clear()
        probe.memo[source.node, key[1][::-1]] = "ordinary_transposed_read"
    elif reason == "tensor":
        probe.boundaries = {source.node: "other_alias"}
    elif reason == "stale":
        source.matches.return_value = False
    elif reason == "shape":
        source.shape = (16, 128)
    elif reason == "dtype":
        source.dtype = torch.float32
    elif reason == "full_shape":
        source.full_shape = (32, 128)
    elif reason in ("duplicate_index", "duplicate_node"):
        other = _input(source.node, index=8 if reason == "duplicate_node" else 7)
        sources = (source, other)
    elif reason == "bound_read":
        probe.memo[key] = "already_bound_ssa_read"
    elif reason == "ownership":
        ownership = replace(ownership, trips=1)
    else:
        execution = ChainedExecution(256)
    assert (
        _emit(case, sources=sources, ownership=ownership, execution=execution) is None
    )


def test_repeated_probes_share_one_copy_and_keep_stable_source_ordinal():
    case = _case()
    probe = case[3]
    result = _emit(case, probes=(probe, probe))
    assert result is not None and len(result.loads) == 1
    assert len(result.replacements) == 1
    assert all("input_7" in line for line in (*result.setup, *result.loads))


@pytest.mark.parametrize(
    "dtype",
    ("cutlass.Float32", "cutlass.Float16", "cutlass.BFloat16", "cutlass.Boolean"),
)
@pytest.mark.parametrize(
    "shape,coords", (((), ()), ((3,), ("row",)), ((3, 8), ("row", "column")))
)
def test_materialized_value_default_identity_and_storage_only_substitution(
    dtype, shape, coords
):
    original = chain._materialized_value("image", shape, coords, dtype)
    explicit = chain._materialized_value(
        "image", shape, coords, dtype, storage_value=None
    )
    replacement = chain._materialized_value(
        "image", shape, coords, dtype, storage_value="register_value"
    )
    index = ", ".join(coords) if shape else "0"
    assert original == explicit
    assert replacement == original.replace(f"image[{index}]", "register_value")


@pytest.mark.parametrize("row,column", ((-1, 0), (3, 0), (0, -1), (0, 8)))
def test_out_of_bounds_register_poison_still_yields_typed_positive_zero(row, column):
    expression = chain._materialized_value(
        "image",
        (3, 8),
        ("row", "column"),
        "cutlass.Float32",
        storage_value="must_not_read[0]",
    )
    value = eval(
        expression,
        {"row": row, "column": column, "cutlass": SimpleNamespace(Float32=float)},
    )
    assert value == 0 and not torch.signbit(torch.tensor(value))


def _capture(dtype, mode, *, native=True, fail=None, scalar=False):
    captured = []
    original = stage_module.emit_stage

    def observe(cg, plan, boundaries, stage, *args, **kwargs):
        if not captured:
            outputs = _outputs(plan, stage, boundaries, cg, scalar=scalar)
            probe = chain._Expression(cg, plan, boundaries)
            probe.coordinate_names.update(("row", "column"))
            probe.value(outputs[0].node, outputs[0].coordinates("row", "column"))
            leaf = next(
                node
                for node, *_ in probe.loaded_inputs
                if chain._shape(node) == (128, 128)
            )
            source = _input(leaf, shape=(128, 128), dtype=dtype)
            source.full_shape = (256, 128)
            selected = {**boundaries, leaf: source.tensor}
            inputs = NativeReadInputs((source,))
            if fail == "stale":
                source.matches.return_value = False
            if fail == "callback":

                def reject(*args):
                    raise chain._UnsupportedChain("test final callback")

                outputs[0] = replace(outputs[0], final_value=reject)
            unroll = PointwiseUnroll(3) if fail == "unroll" else None
            ownership = plan_vector_ownership((128, 128), 128, tile_columns=32)
            assert ownership is not None
            common: dict[str, Any] = {
                "shape": (128, 128),
                "tag": "native_test",
                "execution": ChainedExecution(128),
                "ownership": ownership,
                "producer_unroll": unroll,
            }
            if native:
                common["native_inputs"] = inputs
            if mode == "single":
                item = outputs[0]

                def call():
                    return emit_vector_expression(
                        cg,
                        plan,
                        selected,
                        item.node,
                        coordinates=item.coordinates,
                        offset=item.offset,
                        target=item.target,
                        final_value=item.final_value,
                        vector_store=item.vector_store,
                        raw_boundaries={leaf: source.tensor}
                        if fail != "admission"
                        else None,
                        **common,
                    )
            else:
                if fail == "admission":
                    outputs = [
                        replace(item, node=leaf, coordinates=lambda r, c: (r, c))
                        for item in outputs
                    ]
                emitter = (
                    emit_materialized_group
                    if mode == "materialized"
                    else emit_vector_group
                )

                def call():
                    return emitter(cg, plan, selected, outputs, **common)

            try:
                lines = call()
            except (chain._UnsupportedChain, exc.BackendUnsupported) as error:
                lines = error
            captured.append((lines, inputs))
        return original(cg, plan, boundaries, stage, *args, **kwargs)

    with _cpu_codegen(), patch.object(stage_module, "emit_stage", observe):
        _shared_loop._bind_isolated(_args(dtype=dtype)).to_code(
            _config(vectorize=False)
        )
    assert len(captured) == 1
    return captured[0]


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("mode", ("single", "group", "materialized"))
@pytest.mark.parametrize("scalar", (False, True))
def test_real_original_expressions_native_loads_are_guarded_and_outputs_unchanged(
    dtype, mode, scalar
):
    lines, inputs = _capture(dtype, mode, scalar=scalar)
    before, unused = _capture(dtype, mode, scalar=scalar, native=False)
    assert isinstance(lines, list) and inputs.activated
    assert isinstance(before, list) and not unused.activated
    tree = ast.parse("\n".join(lines))
    step = next(node for node in tree.body if isinstance(node, ast.For))
    guard = next(node for node in step.body if isinstance(node, ast.If))
    assert ast.unparse(guard.test) == "native_test_row < 128"
    assert ast.unparse(guard.body[0]).startswith("cute.copy(native_test_input_7_copy,")
    assert "original_full_native_alias" not in ast.unparse(guard)
    assert "else cutlass." in ast.unparse(guard)
    assert not any("sync_threads" in line or "barrier" in line for line in lines)
    assert "partition_S(original_full_native_alias)" in lines[2]
    restored = [
        ""
        if "cute.copy(native_test_input_7_copy," in line
        else line.replace(
            "native_test_input_7_values[native_test_element]",
            "original_full_native_alias[native_test_row, "
            "(native_test_base + native_test_element)]",
        )
        for line in lines[4:]
    ]
    # All original expression statements, narrow casts, bound wrappers, final
    # masks, output setup/stores and ordering have identical ASTs after removing
    # precisely the new source transport and restoring its storage read. The
    # expression lowering may already have unparsed redundant index parentheses.
    assert ast.dump(ast.parse("\n".join(restored))) == ast.dump(
        ast.parse("\n".join(before))
    )


@pytest.mark.parametrize("mode", ("single", "group", "materialized"))
def test_failed_native_request_keeps_exact_ordinary_emission(mode):
    before, _ = _capture(torch.bfloat16, mode, native=False)
    after, inputs = _capture(torch.bfloat16, mode, fail="stale")
    assert isinstance(before, list) and after == before
    assert not inputs.activated


@pytest.mark.parametrize("mode", ("single", "group", "materialized"))
@pytest.mark.parametrize("failure", ("callback", "unroll"))
def test_late_expression_or_unroll_failure_does_not_activate_native_inputs(
    mode, failure
):
    lines, inputs = _capture(torch.bfloat16, mode, fail=failure)
    assert lines is None or isinstance(
        lines, (chain._UnsupportedChain, exc.BackendUnsupported)
    )
    assert not inputs.activated


@pytest.mark.parametrize("mode", ("single", "group"))
def test_native_input_never_enables_rejected_original_producer(mode):
    lines, inputs = _capture(torch.bfloat16, mode, fail="admission")
    assert lines is None and not inputs.activated
