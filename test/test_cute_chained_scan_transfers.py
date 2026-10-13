from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_frontier_materialization import completed as completed
from .test_cute_chained_preparation_actions import _capture
from .test_cute_chained_scan_producer import _sequence_config
from helion import exc
from helion._compiler.cute import chained_preparation_pipeline
from helion._compiler.cute import chained_scan_producer_emission
from helion._compiler.cute import chained_tcgen_stage
from helion._compiler.cute.chained_matmul import _UnsupportedChain
from helion._compiler.cute.chained_preparation_actions import _PreparationRecorder
from helion._compiler.cute.chained_preparation_actions import _transport_facts
from helion._compiler.cute.chained_preparation_storage import bind_preparation_storage
from helion._compiler.cute.chained_scan_producer import ScanProducer
from helion._compiler.cute.chained_scan_transfers import ScanTransferAttempt
from helion._compiler.cute.chained_scan_transfers import bind_scan_transfers
from helion._compiler.cute.chained_vector_native import NativeReadInputs


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_generic_native_scan_rebinds_original_typed_reads(dtype):
    config = _sequence_config()
    config.config["cute_chained_native_vector_reads"] = True
    source, body, bound = _capture(dtype, True, config=config)
    assert bound is not None and len(bound.scan_transfers) == 1
    transfer = bound.scan_transfers[0]
    accepted = bound.physical.accepted
    receipt = transfer.receipt
    assert len(receipt.reads) == 1
    assert receipt.reads[0][0].dtype == dtype
    assert receipt.emission.ownership is not None
    assert receipt.emission.ownership.thread_order == "column_major"
    assert body == accepted.lines
    assert (
        receipt.lines
        == body[receipt.body_first : receipt.body_first + len(receipt.lines)]
    )
    assert bind_scan_transfers(bound.physical, receipt) == transfer
    assert "chain_scan_producer_" in source
    assert f"_input_{receipt.reads[0][0].index}_source" in source
    assert bound.matches(accepted.revision.plan, accepted.pipeline)


def test_actual_three_raw_reads_and_materialized_frontier_are_composed(completed):
    source, _, bound, _ = completed
    assert len(bound.scan_transfers) == 1
    transfer = bound.scan_transfers[0]
    receipt = transfer.receipt
    accepted = bound.physical.accepted
    assert len(receipt.reads) == 3
    assert len(bound.frontier_transfers) == 1
    action = next(action for action in accepted.actions if action.scan_transfer)
    assert isinstance(action.proof, ScanProducer)
    assert receipt.matches(accepted, action)
    phase = receipt.candidate.phases[receipt.candidate.first_event]
    fused = next(item for item in transfer.phases if item.first == phase)
    assert fused.stop == phase + 1
    assert {source.tensor for source, _ in receipt.reads} <= {
        region.name for region in fused.sources
    }
    assert all(
        not left.overlaps_storage(right)
        for item in transfer.phases
        for left in item.sources
        for right in item.destinations
    )
    assert "StMatrix8x8x16bOp" in source


def test_default_scalar_scan_has_no_native_transfer():
    _, _, bound = _capture(torch.bfloat16, True)
    assert bound is not None and bound.scan_transfers == ()
    assert any(action.kind == "scan" for action in bound.physical.accepted.actions)
    assert all(
        action.scan_transfer is None for action in bound.physical.accepted.actions
    )


@pytest.mark.parametrize("field", ["body_first", "lines", "published", "reads"])
def test_changed_successful_scan_receipt_rejects_binding(completed, field):
    _, _, bound, _ = completed
    receipt = bound.scan_transfers[0].receipt
    if field == "body_first":
        changed = replace(receipt, body_first=receipt.body_first + 1)
    elif field == "lines":
        changed = replace(receipt, lines=("pass",))
    elif field == "published":
        changed = replace(receipt, published=())
    else:
        changed = replace(receipt, reads=())
    assert bind_scan_transfers(bound.physical, changed) is None


@pytest.mark.parametrize("failure", ["short", "unaligned", "late", "overlap"])
def test_relocated_raw_owner_is_revalidated(completed, failure):
    _, _, bound, _ = completed
    receipt = bound.scan_transfers[0].receipt
    phase = receipt.candidate.phases[receipt.candidate.first_event]
    name = receipt.reads[0][0].tensor
    original = bound.physical.layout.region(name)
    destination = bound.physical.layout.region(receipt.candidate.stage.a.name)
    changed = {
        "short": replace(original, byte_size=original.byte_size - 128),
        "unaligned": replace(original, byte_offset=original.byte_offset + 1),
        "late": replace(original, live_from=phase + 1),
        "overlap": replace(original, byte_offset=destination.byte_offset),
    }[failure]
    layout = replace(
        bound.physical.layout,
        regions=tuple(
            changed if region is original else region
            for region in bound.physical.layout.regions
        ),
    )
    assert bind_scan_transfers(replace(bound.physical, layout=layout), receipt) is None


@pytest.mark.parametrize("target", ["a", "b", "residual", "norm"])
@pytest.mark.parametrize("failure", ["short", "early"])
def test_every_nonvector_access_and_complete_native_union_is_validated(
    completed, target, failure
):
    _, _, bound, _ = completed
    receipt = bound.scan_transfers[0].receipt
    scan = receipt.candidate
    name = {
        "a": scan.stage.a.name,
        "b": scan.stage.b.name,
        "residual": scan.buffer.name,
        "norm": next(
            name
            for action in scan.prelude
            if action.kind == "collective"
            for name in action.writes
        ),
    }[target]
    view = next(
        item for item in bound.physical.views if item.original.semantic.name == name
    )
    original = bound.physical.layout.region(view.owner)
    changed = (
        replace(original, byte_size=original.byte_size - 128)
        if failure == "short"
        else replace(original, live_until=scan.phases[scan.first_event])
    )
    layout = replace(
        bound.physical.layout,
        regions=tuple(
            changed if region is original else region
            for region in bound.physical.layout.regions
        ),
    )
    assert bind_scan_transfers(replace(bound.physical, layout=layout), receipt) is None


def test_residual_full_view_cannot_be_replaced_by_dense_row(completed):
    _, _, bound, _ = completed
    receipt = bound.scan_transfers[0].receipt
    view = next(
        item
        for item in bound.physical.views
        if item.original.semantic.name == receipt.candidate.buffer.name
    )
    owner = bound.physical.layout.region(view.owner)
    assert view.byte_offset < owner.byte_offset
    changed = replace(view, byte_offset=owner.byte_offset)
    physical = replace(
        bound.physical,
        views=tuple(changed if item is view else item for item in bound.physical.views),
    )
    assert bind_scan_transfers(physical, receipt) is None


def _attempt(receipt):
    return ScanTransferAttempt(
        receipt.plan,
        receipt.pipeline,
        receipt.inputs,
        receipt.body_first,
        _transport_facts(receipt.pipeline),
    )


def _complete(attempt, receipt, emission=None):
    attempt.complete(
        dict(receipt.published),
        dict(receipt.outputs),
        receipt.execution,
        receipt.emission if emission is None else emission,
        list(receipt.lines),
    )


def test_completion_once_only_missing_and_failed_native_receipt(completed):
    receipt = completed[2].scan_transfers[0].receipt
    attempt = _attempt(receipt)
    with pytest.raises(ValueError, match="completion receipt changed"):
        attempt.result()
    with pytest.raises(ValueError, match="completed scan transfer changed"):
        _complete(attempt, receipt, replace(receipt.emission, native_sources=()))
    assert not attempt.completed and attempt.receipt is None
    _complete(attempt, receipt)
    assert attempt.result() == receipt
    with pytest.raises(ValueError, match="completed twice"):
        _complete(attempt, receipt)
    attempt.receipt = None
    with pytest.raises(ValueError, match="completion receipt changed"):
        attempt.result()


def test_mutable_descriptor_between_begin_and_completion_rejects(completed):
    receipt = completed[2].scan_transfers[0].receipt
    attempt = _attempt(receipt)
    wrapper = receipt.pipeline.prepared_leaves[receipt.reads[0][0].index].wrapper
    original = wrapper["kernel_args"]
    wrapper["kernel_args"] = ["wrong_atom", "wrong_tensor"]
    try:
        with pytest.raises(ValueError, match="transports changed"):
            _complete(attempt, receipt)
        assert not attempt.completed and attempt.receipt is None
    finally:
        wrapper["kernel_args"] = original
    _complete(attempt, receipt)
    assert attempt.result() == receipt


def test_lost_bound_receipt_or_changed_body_cannot_be_consumed(completed):
    bound = completed[2]
    accepted = bound.physical.accepted
    assert not replace(bound, scan_transfers=()).matches(
        accepted.revision.plan, accepted.pipeline
    )
    receipt = bound.scan_transfers[0].receipt
    lines = list(accepted.lines)
    lines[receipt.body_first] = "pass"
    changed = replace(accepted, lines=tuple(lines))
    changed = replace(changed, _selection=changed._fields())
    physical = replace(bound.physical, accepted=changed)
    assert bind_scan_transfers(physical, receipt) is None
    assert (
        bind_preparation_storage(accepted.revision.plan, accepted.pipeline, physical)
        is None
    )


def test_inplace_receipt_removal_rejects_before_and_after_binding(completed):
    bound = completed[2]
    accepted = bound.physical.accepted
    action = next(action for action in accepted.actions if action.scan_transfer)
    receipt = action.scan_transfer
    assert accepted.scan_transfers == (receipt,)
    object.__setattr__(action, "scan_transfer", None)
    try:
        assert not accepted.matches(
            accepted.revision.plan, accepted.pipeline, dict(accepted.revision.shapes)
        )
        assert (
            bind_preparation_storage(
                accepted.revision.plan, accepted.pipeline, bound.physical
            )
            is None
        )
        assert not bound.matches(accepted.revision.plan, accepted.pipeline)
    finally:
        object.__setattr__(action, "scan_transfer", receipt)
    assert bound.matches(accepted.revision.plan, accepted.pipeline)


def test_inplace_receipt_removal_before_seal_rejects_completed_native_body():
    seal = _PreparationRecorder.seal
    observed = []

    def remove(recorder, *args, **kwargs):
        action = next(action for action in recorder.actions if action.scan_transfer)
        receipt = action.scan_transfer
        assert receipt is not None and receipt.emission.native_reads
        assert recorder._scan_receipts == (receipt,)
        observed.append(receipt)
        object.__setattr__(action, "scan_transfer", None)
        return seal(recorder, *args, **kwargs)

    config = _sequence_config()
    config.config["cute_chained_native_vector_reads"] = True
    with (
        patch.object(_PreparationRecorder, "seal", remove),
        pytest.raises(exc.BackendUnsupported, match="scan transfer inventory"),
    ):
        _capture(torch.bfloat16, True, config=config)
    assert len(observed) == 1


def test_failed_full_stage_cannot_publish_scan_boundaries_or_receipt():
    original_macro = chained_preparation_pipeline._emit_scan_producer_stage
    original_stage = chained_tcgen_stage.emit_stage
    observed = []

    def observe(cg, plan, pipeline, boundaries, *args, **kwargs):
        before = dict(boundaries)
        try:
            return original_macro(cg, plan, pipeline, boundaries, *args, **kwargs)
        finally:
            observed.append(
                (before, dict(boundaries), kwargs["recorder"], kwargs["native_inputs"])
            )

    def fail(*args, **kwargs):
        original_stage(*args, **kwargs)
        assert kwargs.get("operand_producer") is not None
        raise _UnsupportedChain("failure after complete native producer")

    config = _sequence_config()
    config.config["cute_chained_native_vector_reads"] = True
    with (
        patch.object(
            chained_preparation_pipeline, "_emit_scan_producer_stage", observe
        ),
        patch.object(chained_tcgen_stage, "emit_stage", fail),
        pytest.raises(exc.BackendUnsupported),
    ):
        _capture(torch.bfloat16, True, config=config)
    assert len(observed) == 1
    before, after, recorder, inputs = observed[0]
    assert before == after and not inputs.activated
    assert recorder.scan_attempt is not None
    assert not recorder.scan_attempt.completed and recorder.scan_attempt.receipt is None
    assert all(action.kind != "scan" for action in recorder.actions)


def test_native_geometry_failure_keeps_scalar_emission_and_no_receipt():
    config = _sequence_config()
    original, body, bound = _capture(torch.bfloat16, True, config=config)
    assert bound is not None and not bound.scan_transfers
    emit = chained_scan_producer_emission.emit_scan_producer

    def unsupported(*args, **kwargs):
        return emit(*args, **kwargs, native_inputs=NativeReadInputs(()))

    # Inject an unsupported transport request, without setting a public knob
    # that correctly rejects ineffective native selection at the outer caller.
    with (
        patch.object(chained_scan_producer_emission, "emit_scan_producer", unsupported),
        patch.object(
            chained_scan_producer_emission, "emit_native_inputs", return_value=None
        ) as attempt,
    ):
        changed, changed_body, changed_bound = _capture(
            torch.bfloat16, True, config=config
        )
    assert attempt.call_count == 1
    assert changed_bound is not None and changed_bound.scan_transfers == ()
    assert changed == original and changed_body == body
