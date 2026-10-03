from __future__ import annotations

import ast
from dataclasses import replace
import re
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_operand_retention_integration import _kda_config
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_scan_producer import _scan_sequence
from .test_cute_chained_scan_producer import _sequence_args
from .test_cute_chained_scan_producer import _sequence_config
from helion import exc
from helion._compiler.cute import chained_preparation_pipeline as pipeline_module
from helion._compiler.cute import chained_preparation_storage as storage_module
from helion._compiler.cute.chained_pointwise_unroll import BoundedProducerUnroll
from helion._compiler.cute.chained_preparation_actions import workspace_name
from helion._compiler.cute.chained_preparation_storage import bind_preparation_storage
from helion._compiler.cute.chained_preparation_storage import (
    plan_accepted_preparation_storage,
)
from helion._compiler.cute.chained_scratch_layout import ScratchLayouts
from helion._compiler.cute.chained_vector_stage import VectorStaging


def _alpha_body(lines):
    names = {}

    class Rename(ast.NodeTransformer):
        def visit_Name(self, node):
            if re.fullmatch(r"chain_value_\d+", node.id):
                node.id = names.setdefault(node.id, f"original_ssa_{len(names)}")
            return node

    return ast.dump(
        Rename().visit(ast.parse("\n".join(lines))), include_attributes=False
    )


def _capture(dtype, compact, *, kernel=_scan_sequence, args=None, config=None):
    """Exercise the internal branch, not a public search/config activation."""
    original_emit = pipeline_module.emit_preparation_pipeline
    original_prepare = pipeline_module._prepare
    original_bind = storage_module.bind_preparation_storage
    bodies, bindings = [], []

    def emit(*args, **kwargs):
        return original_emit(*args, **kwargs, compact_preparation=compact)

    def prepare(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        bodies.append(tuple(result))
        return result

    def bind(*args, **kwargs):
        result = original_bind(*args, **kwargs)
        assert result is not None
        bindings.append(result)
        return result

    with (
        patch.object(pipeline_module, "emit_preparation_pipeline", emit),
        patch.object(pipeline_module, "_prepare", prepare),
        patch.object(storage_module, "bind_preparation_storage", bind),
    ):
        source = _source(
            kernel,
            _sequence_args(dtype) if args is None else args,
            _sequence_config() if config is None else config,
        )
    assert len(bodies) == 1
    assert len(bindings) == int(compact)
    return source, bodies[0], bindings[0] if bindings else None


@pytest.fixture(scope="module", params=[torch.bfloat16, torch.float16])
def completed(request):
    ordinary = _capture(request.param, False)
    compact = _capture(request.param, True)
    return ordinary, compact


def test_internal_compact_source_emits_math_once_and_charges_physical_stride(completed):
    ordinary, compact = completed
    source, body, bound = compact
    assert bound is not None
    physical = bound.physical
    accepted = physical.accepted
    storage = bound._state.finalized
    assert storage is not None and storage.preparation is bound
    assert bound._state.consumed is True
    assert (
        physical.layout.allocated_bytes < accepted.pipeline.frame.layout.allocated_bytes
    )
    assert dict(storage.allocations)["frames"] == accepted.pipeline.slots * bound.stride
    assert f"chain_slot * {bound.stride}" in source
    assert (
        f"cutlass.Uint8, {accepted.pipeline.slots * bound.stride}, alignment=128"
        in source
    )
    assert body == accepted.lines
    expected = ordinary[1]
    for name in accepted.workspaces:
        original = accepted.pipeline.frame.layout.region(name)
        pointer = f"cute.recast_ptr(chain_frame + {original.byte_offset}, dtype=cutlass.BFloat16)"
        expected = tuple(
            line.replace(pointer, workspace_name(name)) for line in expected
        )
    # Earlier emission changes only the allocator's SSA ordinal. Preserve all
    # AST operations, casts, guards and statement order under a bijective rename.
    assert _alpha_body(body) == _alpha_body(expected)
    changed = tuple(
        line.replace("cutlass.Float32(0) +", "cutlass.Float32(1) +") for line in body
    )
    assert _alpha_body(changed) != _alpha_body(body)
    assert tuple(
        index
        for action in accepted.actions
        for index in range(action.first, action.stop)
    ) == tuple(range(len(accepted.pipeline.frame.actions)))


def test_physical_owners_and_residual_full_view_are_proved(completed):
    bound = completed[1][2]
    physical, accepted = bound.physical, bound.physical.accepted
    assert bound.matches(accepted.revision.plan, accepted.pipeline)
    for index, left in enumerate(physical.layout.regions):
        assert left.byte_offset % 128 == 0
        assert left.byte_end <= bound.stride
        for right in physical.layout.regions[index + 1 :]:
            assert not (left.overlaps_storage(right) and left.overlaps_lifetime(right))
    scan = accepted.pipeline.scan_producer
    assert scan is not None
    view = next(
        view
        for view in physical.views
        if view.original.semantic.name == scan.buffer.name
    )
    owner = physical.layout.region(view.owner)
    assert (
        view.declared_bytes
        == accepted.pipeline.frame.layout.region(scan.buffer.name).byte_size
    )
    assert (
        owner.byte_size
        == (max(scan.residual_rows) - min(scan.residual_rows) + 1) * scan.shape[1] * 4
    )
    assert (
        view.byte_offset + min(scan.residual_rows) * scan.shape[1] * 4
        == owner.byte_offset
    )


@pytest.mark.parametrize(
    "field", ["lines", "actions", "workspaces", "scratch_mode", "transport_facts"]
)
def test_sealed_body_cannot_be_changed_or_emptied(completed, field):
    bound = completed[1][2]
    accepted = bound.physical.accepted
    replacement = "xor" if field == "scratch_mode" else ()
    tampered = replace(accepted, **{field: replacement})
    assert not tampered.matches(
        accepted.revision.plan, accepted.pipeline, dict(accepted.revision.shapes)
    )
    assert (
        plan_accepted_preparation_storage(
            accepted.revision.plan, accepted.pipeline, tampered, capacity_bytes=1 << 20
        )
        is None
    )


@pytest.mark.parametrize("field", ["layout", "views", "omitted"])
def test_derived_physical_table_cannot_be_replaced(completed, field):
    bound = completed[1][2]
    physical = bound.physical
    value = {
        "layout": replace(
            physical.layout, allocated_bytes=physical.layout.allocated_bytes - 128
        ),
        "views": physical.views[:-1],
        "omitted": (*physical.omitted, physical.views[0].original.semantic.name),
    }[field]
    tampered = replace(physical, **{field: value})
    accepted = physical.accepted
    assert (
        bind_preparation_storage(accepted.revision.plan, accepted.pipeline, tampered)
        is None
    )


def test_capacity_failure_does_not_return_a_discounted_fallback(completed):
    bound = completed[1][2]
    accepted = bound.physical.accepted
    assert (
        plan_accepted_preparation_storage(
            accepted.revision.plan,
            accepted.pipeline,
            accepted,
            capacity_bytes=bound.stride - 1,
        )
        is None
    )


def test_finalized_body_cannot_be_consumed_twice_or_from_a_copied_storage(completed):
    bound = completed[1][2]
    accepted = bound.physical.accepted
    vector = VectorStaging(
        enabled=accepted.vector_state[0], group_enabled=accepted.vector_state[2]
    )
    unroll = BoundedProducerUnroll(accepted.unroll_state[0])
    storage = bound._state.finalized
    assert storage is not None
    with pytest.raises(ValueError, match="unconsumed final allocation"):
        bound.consume_body(storage, vector, unroll)
    with pytest.raises(ValueError, match="unconsumed final allocation"):
        bound.consume_body(replace(storage), vector, unroll)
    assert not vector.activated and not unroll.activated


def test_changed_scratch_mode_rejects_before_view_emission(completed):
    bound = completed[1][2]
    with pytest.raises(ValueError, match="binding changed"):
        bound.view_lines("chain_frame", ScratchLayouts("xor"))


def test_default_does_not_discover_or_build_accepted_actions():
    from helion._compiler.cute import chained_preparation_actions

    with patch.object(
        chained_preparation_actions,
        "build_accepted_preparation",
        side_effect=AssertionError("default discovered compact preparation"),
    ):
        source, _, bound = _capture(torch.bfloat16, False)
    assert bound is None and "chain_preparation_workspace_" not in source


def test_raw_transport_captures_original_half_boundary_before_finalization():
    from helion._compiler.cute import chained_loop_tmem_carry_transport
    from helion._compiler.cute import chained_loop_tmem_transport

    observed = []

    def inspect_binding(original):
        def inspect(cg, plan, boundaries, *args, **kwargs):
            raw = plan.prepared_widenings
            assert raw is not None and raw.matches(plan)
            for binding in raw.bindings:
                assert binding.transfer.widening not in boundaries
                assert boundaries[binding.transfer.source] == binding.buffer.name
            observed.append(raw)
            return original(cg, plan, boundaries, *args, **kwargs)

        return inspect

    kernel, args = _kda_fixture()
    config = _kda_config()
    config.config["cute_chained_scan_producer_retention"] = True
    with (
        patch.object(
            chained_loop_tmem_transport,
            "prepare_loop_tmem_transports",
            inspect_binding(chained_loop_tmem_transport.prepare_loop_tmem_transports),
        ),
        patch.object(
            chained_loop_tmem_carry_transport,
            "prepare_loop_tmem_carry",
            inspect_binding(chained_loop_tmem_carry_transport.prepare_loop_tmem_carry),
        ),
    ):
        source, _, bound = _capture(
            torch.bfloat16, True, kernel=kernel, args=args, config=config
        )
    assert bound is not None
    accepted = bound.physical.accepted
    assert len(observed) == 2 and observed[0] is observed[1] is accepted.raw
    assert accepted.raw is not None and len(accepted.raw.bindings) == 1
    assert accepted.crops is not None and accepted.crops.crops
    assert accepted.revision.plan.prepared_widenings is None
    raw = accepted.raw.bindings[0]
    assert raw.buffer.dtype == torch.bfloat16
    assert raw.transfer.widening.meta["val"].dtype == torch.float32
    assert raw.buffer.name in source
    assert "bfloat16_to_float32" in source
    # The old FP32 frontier is neither allocated nor captured by a deferred
    # TMEM transport expression. Suffix matching would miss this regression.
    assert not any(
        isinstance(node, ast.Name) and node.id == raw.transfer.buffer.name
        for node in ast.walk(ast.parse(source))
    )
    for view in bound.physical.views:
        if view.crop is not None:
            owner = bound.physical.layout.region(view.owner)
            assert (
                owner.byte_size
                == view.crop.source.full_shape[0]
                * view.crop.source.full_shape[1]
                * torch.bfloat16.itemsize
            )
            assert "cute.local_tile(chain_preparation_owner_" in source


def test_failed_transport_selection_does_not_consume_or_activate_compact_body():
    from helion._compiler.cute import chained_pipeline_storage

    original = pipeline_module.emit_preparation_pipeline
    selected = []

    def emit(cg, plan, pipeline, prologue, scratch, vector, unroll, *args):
        selected.append((vector, unroll))
        return original(
            cg,
            plan,
            pipeline,
            prologue,
            scratch,
            vector,
            unroll,
            *args,
            compact_preparation=True,
        )

    with (
        patch.object(pipeline_module, "emit_preparation_pipeline", emit),
        patch.object(
            chained_pipeline_storage, "select_stage_transports", return_value=None
        ),
        pytest.raises(exc.BackendUnsupported, match="transports are inconsistent"),
    ):
        _source(_scan_sequence, _sequence_args(), _sequence_config())
    assert len(selected) == 1
    vector, unroll = selected[0]
    assert not vector.activated and not vector.group_activated and not unroll.activated


@pytest.mark.parametrize(
    "mutation", ["nested_kernel_arg", "tile", "strides", "guard", "tile_shape"]
)
def test_sealed_leaf_descriptor_rejects_mutation_at_every_binding_boundary(
    completed, mutation
):
    bound = completed[1][2]
    accepted = bound.physical.accepted
    fresh = bind_preparation_storage(
        accepted.revision.plan, accepted.pipeline, bound.physical
    )
    assert fresh is not None
    storage = replace(bound._state.finalized, preparation=fresh)
    assert fresh._finalize(storage)
    leaf = accepted.pipeline.prepared_leaves[0]
    vector = VectorStaging(
        accepted.vector_state[0], group_enabled=accepted.vector_state[2]
    )
    unroll = BoundedProducerUnroll(accepted.unroll_state[0])
    if mutation == "nested_kernel_arg":
        # Mutate the existing list rather than replacing the wrapper object.
        target, key = leaf.wrapper["kernel_args"], 0
        value = target[0] + "_changed_after_seal"
    elif mutation in ("tile", "strides"):
        target, key = leaf.wrapper, mutation
        value = (*target[key][:-1], target[key][-1] + 1)
    else:
        # Test the saved primitive fields too, independent of the dataclass's
        # ordinary frozen-assignment guard.
        target, key = vars(leaf.proof), mutation
        value = "False" if mutation == "guard" else (1, 1)
    old = target[key]
    target[key] = value
    try:
        assert not accepted.matches(
            accepted.revision.plan, accepted.pipeline, dict(accepted.revision.shapes)
        )
        assert not fresh.matches(accepted.revision.plan, accepted.pipeline)
        assert (
            bind_preparation_storage(
                accepted.revision.plan, accepted.pipeline, bound.physical
            )
            is None
        )
        with pytest.raises(ValueError, match="binding changed"):
            fresh.view_lines("chain_frame", ScratchLayouts(accepted.scratch_mode))
        with pytest.raises(ValueError, match="unconsumed final allocation"):
            fresh.consume_body(storage, vector, unroll)
        assert not fresh._state.consumed
        assert not vector.activated and not unroll.activated
    finally:
        target[key] = old
    assert fresh.matches(accepted.revision.plan, accepted.pipeline)


def test_leaf_mutation_during_body_cannot_be_sealed():
    original = pipeline_module._prepare
    restore = []

    def prepare(cg, plan, pipeline, *args, **kwargs):
        lines = original(cg, plan, pipeline, *args, **kwargs)
        values = pipeline.prepared_leaves[0].wrapper["kernel_args"]
        restore.append((values, values[0]))
        values[0] += "_changed_before_seal"
        return lines

    try:
        with (
            patch.object(pipeline_module, "_prepare", prepare),
            pytest.raises(exc.BackendUnsupported, match="changed during body emission"),
        ):
            _capture(torch.bfloat16, True)
    finally:
        for values, original_name in restore:
            values[0] = original_name
    assert len(restore) == 1
