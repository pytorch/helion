from __future__ import annotations

import ast
from dataclasses import replace

import pytest
import torch

from .test_cute_chained_operand_retention_integration import _kda_config
from .test_cute_chained_preparation_actions import _capture
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_vector_group import _capture as _vector_capture
from .test_cute_chained_vector_native import _capture as _native_capture
from helion._compiler.cute.chained_frontier_materialization import (
    FrontierMaterializationAttempt,
)
from helion._compiler.cute.chained_frontier_materialization import (
    FrontierTransferReceipts,
)
from helion._compiler.cute.chained_frontier_materialization import (
    bind_frontier_transfers,
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_stable_original_ordinals_preserve_every_other_emitted_operation(dtype):
    original, _, _ = _vector_capture(dtype=dtype)

    def change(outputs, boundaries, plan):
        return [
            replace(output, ordinal=index * 2) for index, output in enumerate(outputs)
        ], boundaries

    changed, _, _ = _vector_capture(dtype=dtype, change=change)
    assert original is not None and changed is not None
    assert "joined_output_1_values" in "\n".join(original)
    assert "joined_output_2_values" in "\n".join(changed)
    assert [
        line.replace("joined_output_2_", "joined_output_1_") for line in changed
    ] == original


@pytest.mark.parametrize("ordinals", [(True, 1), (-1, 1), (0, 0), (0, 1.5)])
def test_invalid_output_ordinals_do_not_activate_unroll(ordinals):
    def change(outputs, boundaries, plan):
        return [
            replace(output, ordinal=index)
            for output, index in zip(outputs, ordinals, strict=True)
        ], boundaries

    result, _, unroll = _vector_capture(change=change)
    assert result is None and not unroll.activated


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mode", ["group", "materialized"])
def test_actual_native_receipts_include_only_successfully_emitted_sources(dtype, mode):
    lines, inputs = _native_capture(dtype, mode)
    assert isinstance(lines, list) and inputs.activated
    assert inputs.emitted == inputs.inputs
    original, unused = _native_capture(dtype, mode, native=False)
    assert isinstance(original, list) and not unused.activated and unused.emitted == ()


@pytest.mark.parametrize("failure", ["callback", "unroll", "stale"])
def test_failed_native_group_never_records_selected_but_unemitted_sources(failure):
    _, inputs = _native_capture(torch.float16, "materialized", fail=failure)
    assert inputs.emitted == () and not inputs.activated


@pytest.fixture(scope="module")
def completed():
    kernel, args = _kda_fixture()
    config = _kda_config()
    config.config.update(
        cute_chained_scan_producer_retention=True,
        cute_chained_native_vector_reads=True,
        cute_chained_frontier_tile_columns=32,
        cute_chained_frontier_stmatrix=True,
    )
    source, body, bound = _capture(
        torch.bfloat16, True, kernel=kernel, args=args, config=config
    )
    assert bound is not None
    records = tuple(
        action.proof
        for action in bound.physical.accepted.actions
        if isinstance(action.proof, FrontierTransferReceipts)
    )
    assert len(records) == 1
    return source, body, bound, records[0]


def test_actual_identity_omission_preserves_group_native_inputs_and_stores(completed):
    source, body, bound, receipt = completed
    selected = receipt.materializations
    assert selected.emitted_ordinals == (0, 2)
    assert len(selected.aliases) == 1 and selected.aliases[0][0] == 1
    assert receipt.reads and receipt.stores and receipt.ownership.tile_columns == 32
    assert bound.frontier_transfers == (
        bind_frontier_transfers(bound.physical, receipt),
    )
    tag = selected.group.buffers[0].name + "_group"
    names = {
        node.id for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Name)
    }
    assert f"{tag}_output_1_values" not in names
    assert f"{tag}_output_0_values" in names and f"{tag}_output_2_values" in names
    assert "StMatrix8x8x16bOp" in source
    assert all(f"{tag}_input_{item.index}_values" in names for item, _ in receipt.reads)
    accepted = bound.physical.accepted
    assert body == accepted.lines
    _, crop = selected.aliases[0]
    full = f"chain_{crop.source.group.stages[0]}_{crop.source.role}"
    alias = tuple(crop.alias_lines(full, crop.buffer.name))
    first = body.index(alias[0])
    stop = first + len(alias)
    assert body[first : stop + 1] == (*alias, accepted.execution.sync)
    assert body[stop + 1] != accepted.execution.sync
    assert all(
        transfer.sources and transfer.destinations
        for transfer in bound.frontier_transfers
    )
    for transfer in bound.frontier_transfers:
        assert all(
            not left.overlaps_storage(right)
            for left in transfer.sources
            for right in transfer.destinations
        )


@pytest.mark.parametrize("field", ["emitted_ordinals", "aliases", "boundaries"])
def test_original_group_selection_cannot_be_emptied(completed, field):
    _, _, bound, receipt = completed
    tampered = replace(
        receipt, materializations=replace(receipt.materializations, **{field: ()})
    )
    assert not tampered.matches()
    assert bind_frontier_transfers(bound.physical, tampered) is None


@pytest.mark.parametrize(
    "change", ["short_owner", "unaligned", "late_source", "overlap"]
)
def test_physical_transfer_binding_rechecks_relocated_complete_owners(
    completed, change
):
    _, _, bound, receipt = completed
    transfer = bound.frontier_transfers[0]
    source = transfer.sources[0]
    replacement = {
        "short_owner": replace(source, byte_size=source.byte_size - 128),
        "unaligned": replace(source, byte_offset=source.byte_offset + 1),
        "late_source": replace(source, live_from=transfer.first + 1),
        "overlap": replace(source, byte_offset=transfer.destinations[0].byte_offset),
    }[change]
    layout = replace(
        bound.physical.layout,
        regions=tuple(
            replacement if region.name == source.name else region
            for region in bound.physical.layout.regions
        ),
    )
    changed = replace(bound.physical, layout=layout)
    assert bind_frontier_transfers(changed, receipt) is None


def test_bound_receipts_cannot_be_dropped_after_finalization(completed):
    _, _, bound, _ = completed
    changed = replace(bound, frontier_transfers=())
    accepted = bound.physical.accepted
    assert not changed.matches(accepted.revision.plan, accepted.pipeline)


def test_failed_receipt_validation_does_not_complete_or_allow_second_completion(
    completed,
):
    _, _, _, receipt = completed
    attempt = FrontierMaterializationAttempt(receipt.materializations)
    reads = tuple(source for source, _ in receipt.reads)
    changed = replace(receipt.ownership, threads=receipt.ownership.threads + 1)
    with pytest.raises(ValueError, match="geometry changed|transfers changed"):
        attempt.complete(changed, reads, receipt.stores)
    assert attempt.receipt is None
    attempt.complete(receipt.ownership, reads, receipt.stores)
    assert attempt.receipt == receipt
    with pytest.raises(ValueError, match="completed twice"):
        attempt.complete(receipt.ownership, reads, receipt.stores)
