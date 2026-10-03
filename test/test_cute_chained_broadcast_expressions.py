from __future__ import annotations

import ast
from dataclasses import replace
import inspect
from types import SimpleNamespace
from typing import Literal
from unittest.mock import patch

import numpy as np
import pytest
import torch

from ._cute_aux import _cpu_codegen
import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chained_scan_producer_emission as scan_module
from helion._compiler.cute.chained_broadcast_expressions import (
    emit_broadcast_expressions,
)
from helion._compiler.cute.chained_broadcast_expressions import (
    plan_broadcast_expressions,
)
from helion._compiler.cute.chained_vector_ownership import plan_vector_ownership
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _broadcast_loop(a, b, z, initial, rounded: hl.constexpr, typed_end: hl.constexpr):
    steps, height, width = a.shape
    result = torch.empty_like(initial)
    history = torch.empty((steps, height, width), dtype=torch.float32, device=a.device)
    for _rows in hl.tile(height, block_size=32):
        initial_rows = hl.arange(32)
        initial_columns = hl.arange(width)
        state = initial[initial_rows, initial_columns]
        for step in hl.tile(steps, block_size=1):
            rows = hl.arange(32)
            columns = hl.arange(width)
            raw = a[step.id, rows, columns].float()
            prefix = hl.cumsum(raw * 0.125, dim=0)
            endpoint = torch.sum(torch.where((rows == 17)[:, None], prefix, 0.0), dim=0)
            values = b[step.id, rows, columns].float()
            energy = torch.sum(values * values, dim=1)
            retained = torch.tanh(energy + 0.125)
            if rounded:
                retained = retained.to(torch.float16).float()
            broadcast = retained[:, None].expand(32, width)
            if typed_end:
                left = (prefix + broadcast.to(a.dtype) + raw * 0.0625).to(a.dtype)
                right = (prefix * 0.25 + broadcast.to(a.dtype)).to(a.dtype)
            else:
                left = (prefix + broadcast + raw * 0.0625).to(a.dtype)
                right = (prefix * 0.25 + broadcast).to(a.dtype)
            prepared = hl.dot(left, right.T, out_dtype=torch.float32)
            other = hl.arange(32)
            rhs = (prepared * 0.125 + z[step.id, rows, other].float()).to(a.dtype)
            state = (
                hl.dot(
                    rhs,
                    state.to(a.dtype),
                    acc=state,
                    out_dtype=torch.float32,
                )
                + endpoint[None, :]
            )
            history[step.id, rows, columns] = state
        result[initial_rows, initial_columns] = state
    return history, result


def _capture(
    dtype=torch.bfloat16,
    rounded=False,
    change=None,
    check=None,
    *,
    tile_columns=32,
    thread_order: Literal["row_major", "column_major"] = "column_major",
    typed_end=False,
):
    captured = []
    original = scan_module.emit_scan_producer

    def observe(
        cg,
        plan,
        candidate,
        boundaries,
        operands,
        *,
        execution,
        producer_unroll,
        **kwargs,
    ):
        local = dict(boundaries)
        local[candidate.scan.node] = candidate.buffer.name
        probes = []
        for operand in operands:
            coordinates = operand.geometry.operand(
                operand.role, "row", "(base + element)"
            )[1]
            expression = chain._Expression(cg, plan, local)
            expression.coordinate_names.update(("row", "base", "element"))
            expression.value(operand.node, coordinates)
            probes.append(expression)
        ownership = plan_vector_ownership(
            candidate.shape,
            execution.threads,
            tile_columns=tile_columns,
            thread_order=thread_order,
        )
        assert ownership is not None
        selected = plan_broadcast_expressions(
            probes, ownership, row="row", base="base", element="element"
        )
        assert selected is not None
        if check is not None:
            check(cg, plan, local, probes, selected, execution)
        if change is not None:
            selected, local = change(selected, local)
        emission = emit_broadcast_expressions(
            cg, plan, local, selected, tag="broadcast", execution=execution
        )
        captured.append((selected, emission))
        return original(
            cg,
            plan,
            candidate,
            boundaries,
            operands,
            execution=execution,
            producer_unroll=producer_unroll,
            **kwargs,
        )

    args = (
        torch.empty((1, 32, 64), dtype=dtype),
        torch.empty((1, 32, 64), dtype=dtype),
        torch.empty((1, 32, 32), dtype=dtype),
        torch.empty((32, 64)),
        rounded,
        typed_end,
    )
    config = helion.Config(
        num_warps=8,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_preparation_pipeline=True,
        cute_chained_warp_mma_rows=32,
        cute_chained_scan_schedule="warp",
        cute_chained_pointwise_vectorize=True,
        cute_chained_leaf_pipeline="rectangular_tma",
        cute_chained_leaf_count=2,
        cute_chained_scan_producer_retention=True,
    )
    with _cpu_codegen(), patch.object(scan_module, "emit_scan_producer", observe):
        bound = _broadcast_loop._bind_isolated(args)
        with bound.env.use_runtime_arg_values(
            dict(inspect.signature(_broadcast_loop.fn).bind(*args).arguments)
        ):
            bound.to_code(config)
    assert len(captured) == 1
    return captured[0]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rounded", [False, True])
def test_original_unrelated_typed_broadcast(dtype, rounded):
    selected, emission = _capture(dtype, rounded)
    assert emission is not None and emission.before_steps and not emission.per_step
    assert emission.reads
    assert any("tanh" in value.value_ir for value in selected.values)
    assert all("element" not in value.value_ir for value in selected.values)
    assert emission.candidate is selected


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_retained_value_keeps_original_half_snapshot(dtype):
    selected, emission = _capture(dtype, typed_end=True)
    assert emission is not None
    assert all(value.dtype == dtype for value in selected.values)
    cast = "BFloat16" if dtype == torch.bfloat16 else "Float16"
    assert all(cast in value.value_ir for value in selected.values)


@pytest.mark.parametrize("tamper", ["values", "row", "boundary"])
def test_changed_selection_or_boundary_declines(tamper):
    def change(selected, boundaries):
        if tamper == "boundary":
            node = selected.boundaries[0][0]
            return selected, {**boundaries, node: "unpublished"}
        return replace(
            selected, **({"values": ()} if tamper == "values" else {"row": "other"})
        ), boundaries

    _, emission = _capture(change=change)
    assert emission is None


@pytest.mark.parametrize("thread_order", ["row_major", "column_major"])
@pytest.mark.parametrize("tile_columns", [32, 64])
def test_actual_step_dependency_and_broadcast_coordinate_projection(
    thread_order, tile_columns
):
    selected, emission = _capture(tile_columns=tile_columns, thread_order=thread_order)
    assert emission is not None
    # Broadcast memo keys may still mention element. Actual value dependencies
    # drop that axis, rather than relying on coordinate spelling as a proof.
    assert any(
        "element" in coordinate
        for value in selected.values
        for coordinate in value.key[1]
    )
    expected = "before_steps" if tile_columns == 32 else "per_step"
    assert all(value.placement == expected for value in selected.values)
    assert bool(emission.before_steps) == (tile_columns == 32)
    assert bool(emission.per_step) == (tile_columns == 64)
    assert all(
        "cutlass.Float32(0)" in value.value_ir or "Float32" in value.value_ir
        for value in selected.values
    )


@pytest.mark.parametrize(
    "tamper",
    [
        "ownership",
        "read",
        "value",
        "args",
        "dtype",
        "lowering_inputs",
        "lowering_ranges",
        "fast_math",
        "output_args",
    ],
)
def test_same_object_mutations_are_not_selection_authority(tamper):
    def check(cg, plan, local, probes, selected, execution):
        if tamper == "ownership":
            obj, attr, changed = (
                selected.ownership,
                "trips",
                selected.ownership.trips + 1,
            )
        elif tamper == "read":
            obj, attr, changed = selected.values[0].reads[0], "tensor", "other_tensor"
        elif tamper == "value":
            obj, attr, changed = selected.values[0], "value_ir", "changed"
        elif tamper == "output_args":
            obj = next(reversed(probes[0].memo))[0]
            attr, changed = "kwargs", {**obj.kwargs, "changed_domain": True}
        elif tamper == "args":
            obj = next(
                node
                for node in selected.nodes
                if any(type(arg) is float for arg in node.args)
            )
            attr, changed = (
                "args",
                tuple(0.5 if type(arg) is float else arg for arg in obj.args),
            )
        elif tamper == "dtype":
            obj = selected.values[0].reads[0].node
            prior = obj.meta["val"]
            obj.meta["val"] = prior.to(torch.float16)
            try:
                assert not selected.matches(plan, local)
                assert (
                    emit_broadcast_expressions(
                        cg, plan, local, selected, tag="negative", execution=execution
                    )
                    is None
                )
            finally:
                obj.meta["val"] = prior
            return
        elif tamper in ("lowering_inputs", "lowering_ranges"):
            obj = next(
                node.meta["lowering"]
                for node in selected.nodes
                if node.target is torch.ops.aten.tanh.default
            )
            if tamper == "lowering_inputs":
                attr, changed = "input_names", ["different"]
            else:
                obj = obj.buffer.data
                attr, changed = "ranges", [99]
        else:
            obj = CompileEnvironment.current().settings
            attr, changed = "fast_math", not selected.fast_math
        prior = getattr(obj, attr)
        object.__setattr__(obj, attr, changed)
        try:
            assert not selected.matches(plan, local)
            assert (
                emit_broadcast_expressions(
                    cg, plan, local, selected, tag="negative", execution=execution
                )
                is None
            )
        finally:
            object.__setattr__(obj, attr, prior)
        assert selected.matches(plan, local)

    _capture(check=check)


@pytest.mark.parametrize("rounded", [False, True])
def test_exact_original_typed_guard_values_and_read_traces(rounded):
    def check(cg, plan, local, probes, selected, execution):
        emission = emit_broadcast_expressions(
            cg, plan, local, selected, tag="trace", execution=execution
        )
        assert emission is not None and not emission.per_step
        emitted_reads = emission.reads
        cutlass = SimpleNamespace(
            Float32=np.float32,
            Float16=np.float16,
            BFloat16=lambda value: np.float32(
                torch.tensor(float(value)).bfloat16().float().item()
            ),
        )
        cute = SimpleNamespace(
            math=SimpleNamespace(
                tanh=lambda value, **kwargs: np.float32(np.tanh(value))
            )
        )
        values = np.array(
            [
                -0.0,
                0.0,
                1.0e-30,
                -1.0e-30,
                -0.125,
                0.5,
                65504,
                -65504,
                np.inf,
                -np.inf,
                np.nan,
                0.125,
                0.25,
                0.75,
                1.0,
                -1.0,
            ]
            * 2,
            dtype=np.float32,
        )
        trace = []

        class Shared:
            def __getitem__(self, row):
                assert 0 <= row < 32
                trace.append(row)
                return values[row]

        def environment():
            return dict(
                cutlass=cutlass,
                cute=cute,
                **{read.tensor: Shared() for read in emitted_reads},
            )

        for row in (-1, *range(32), 32):
            for entry, (_, replacement) in zip(
                selected.values, emission.replacements, strict=True
            ):
                original = chain._Expression(cg, plan, local)
                value = original.value(entry.key[0], entry.key[1])
                expected_reads = []
                for base in (0, 32, 56):
                    for element in range(8):
                        env = environment() | {
                            "row": row,
                            "base": base,
                            "element": element,
                        }
                        trace.clear()
                        exec(
                            "\n".join(
                                statement.code for statement in original.statements
                            ),
                            env,
                        )
                        expected = eval(value, env)
                        expected_reads = list(trace)
                        # Skip only the mechanically derived owner-row header:
                        # invalid rows exercise the exact guarded original read.
                        retained = environment() | {"trace_row": row}
                        trace.clear()
                        exec("\n".join(emission.before_steps[1:]), retained)
                        actual = retained[replacement]
                        assert (
                            np.asarray(actual, dtype=np.float32).tobytes()
                            == np.asarray(expected, dtype=np.float32).tobytes()
                        )
                        assert trace == expected_reads
                        assert trace == ([row] if 0 <= row < 32 else [])

    _capture(rounded=rounded, check=check)


def test_only_dependency_reachable_shared_coordinates_are_captured():
    def check(cg, plan, local, probes, selected, execution):
        source = selected.values[0].reads[0].node
        for probe in probes:
            output, value = next(reversed(probe.memo.items()))
            probe.value(source, ("(row + 1)",))
            # Keep the original output as this single-output probe's root.
            probe.memo.pop(output)
            probe.memo[output] = value
        actual = plan_broadcast_expressions(
            probes, selected.ownership, row="row", base="base", element="element"
        )
        assert actual is not None
        assert all(
            read.coordinates == ("row",)
            for value in actual.values
            for read in value.reads
        )

    _capture(check=check)


@pytest.mark.parametrize(
    "tamper",
    [
        "element",
        "unknown_origin",
        "statement_effect",
        "partial_rows",
        "zero",
        "unaligned_columns",
        "fragment",
        "effectful_node",
        "unpublished",
        "wrong_output_coordinates",
    ],
)
def test_unsupported_proofs_decline_without_publishing(tamper):
    def check(cg, plan, local, probes, selected, execution):
        baseline_memos = [dict(probe.memo) for probe in probes]
        if tamper == "wrong_output_coordinates":
            for probe in probes:
                (output, coordinates), value = probe.memo.popitem()
                probe.memo[output, (coordinates[0], coordinates[0])] = value
            assert (
                plan_broadcast_expressions(
                    probes,
                    selected.ownership,
                    row="row",
                    base="base",
                    element="element",
                )
                is None
            )
            return
        if tamper in ("partial_rows", "zero", "unaligned_columns"):
            shape = {
                "partial_rows": (31, 64),
                "zero": (0, 64),
                "unaligned_columns": (32, 63),
            }[tamper]
            ownership = plan_vector_ownership(
                shape, 128, tile_columns=32, thread_order="column_major"
            )
            if ownership is not None:
                assert (
                    plan_broadcast_expressions(
                        probes, ownership, row="row", base="base", element="element"
                    )
                    is None
                )
        elif tamper == "statement_effect":
            original = chain._Expression.value

            def unsupported(expression, node, coords):
                result = original(expression, node, coords)
                expression.statements.append(
                    chain._Statement("effect()", None, frozenset())
                )
                return result

            with patch.object(chain._Expression, "value", unsupported):
                assert (
                    emit_broadcast_expressions(
                        cg, plan, local, selected, tag="negative", execution=execution
                    )
                    is None
                )
        else:
            source = selected.values[0].reads[0].node
            if tamper == "unpublished":
                for probe in probes:
                    probe.boundaries.clear()
                assert (
                    plan_broadcast_expressions(
                        probes,
                        selected.ownership,
                        row="row",
                        base="base",
                        element="element",
                    )
                    is None
                )
            elif tamper == "fragment":
                for probe in probes:
                    probe.fragments[source] = (("row",), "register_value")
                result = plan_broadcast_expressions(
                    probes,
                    selected.ownership,
                    row="row",
                    base="base",
                    element="element",
                )
                assert result is None
            elif tamper == "effectful_node":
                target = next(
                    node
                    for node in selected.nodes
                    if node.target is torch.ops.aten.tanh.default
                )
                prior = target.target
                target.target = torch.ops.aten.rand_like.default
                try:
                    result = plan_broadcast_expressions(
                        probes,
                        selected.ownership,
                        row="row",
                        base="base",
                        element="element",
                    )
                    assert result is None or all(
                        target not in value.ancestors for value in result.values
                    )
                finally:
                    target.target = prior
            else:
                # A varying column or runtime origin behind a projected shared
                # read cannot be hoisted simply because the FX node broadcasts.
                for probe in probes:
                    for name, definition in tuple(probe.definitions.items()):
                        if selected.boundaries[0][1] in definition:
                            definition += " + " + (
                                "element" if tamper == "element" else "runtime_origin"
                            )
                            probe.definitions[name] = definition
                            probe.definition_inputs[name] = chain._names(
                                ast.parse(definition, mode="eval")
                            )
                assert (
                    plan_broadcast_expressions(
                        probes,
                        selected.ownership,
                        row="row",
                        base="base",
                        element="element",
                    )
                    is None
                )
        assert all(
            probe.memo == original
            for probe, original in zip(probes, baseline_memos, strict=True)
        )

    _capture(check=check)
