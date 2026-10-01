from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from . import test_cute_chained_root_initialized as initialized_tests
from ._cute_prepared_source import assert_prepared_root_equivalent
from .test_cute_chained_accumulator import _cpu
from .test_cute_chained_pipeline import args as leaf_args
from .test_cute_chained_pipeline import leaf_config
from .test_cute_chained_pipeline import pair
from helion._compiler.cute import chained_k_issue as issues
from helion._compiler.cute import chained_leaf_pipeline as leaves
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chained_plain_root as roots
from helion._compiler.cute import chained_root_stage as actions
from helion._compiler.cute import chained_tcgen05 as legacy
from helion._compiler.cute import chained_tcgen_stage as stages


def _source(dtype=torch.bfloat16, columns=0, schedule="overlap64", **options):
    with _cpu():
        return pair._bind_isolated(leaf_args(dtype=dtype, n=128)).to_code(
            leaf_config(
                schedule=schedule, cute_chained_seed_tile_columns=columns, **options
            )
        )


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("columns", (0, 32, 64))
@pytest.mark.parametrize("schedule", ("serial64", "overlap64"))
def test_paired_stages_have_original_program_and_completion_order(
    dtype, columns, schedule
):
    initial = torch.cuda.is_initialized()
    options = {
        "cute_chained_tmem_free": "last_read",
        "cute_chained_tmem_early_release": bool(columns),
    }
    with patch.object(roots, "codegen_plain_root", return_value=False):
        before = _source(dtype, columns, schedule, **options)
    with (
        patch.object(
            legacy,
            "codegen_chained_tcgen05",
            side_effect=AssertionError("old root entry"),
        ),
        patch.object(stages, "emit_stage", wraps=stages.emit_stage) as emitted,
    ):
        after = _source(dtype, columns, schedule, **options)
    after = assert_prepared_root_equivalent(before, after)
    assert [call.args[3] for call in emitted.call_args_list] == [0, 1]
    sequence = emitted.call_args.kwargs["root_actions"].sequence
    result = sequence.initialized
    assert result is not None and result.paired is not None
    assert sequence.paired_issued and sequence.next_stage == 2
    assert sequence.seeded_inputs is not None and sequence.handles(1)
    assert sequence.seeded_inputs.result is result
    assert sequence.queued_rhs.seed is sequence.seed_completion
    assert emitted.call_args.kwargs["root_actions"].retire_each_half
    assert sequence.seed_completion.result is result
    assert "out_name" in result.paired.transfer.wrapper
    assert "out_name" not in repr(result.paired.descriptor)
    wait = after.index("cute.arch.mbarrier_wait(chain_bars + 0, 0)")
    issue = after.index(
        "cute.arch.mbarrier_arrive_and_expect_tx(chain_leaf_bar, 16384)"
    )
    assert (
        "fence_view_async_shared()\n    cute.arch.sync_threads()" in after[wait:issue]
    )
    seed_begin = after.index("chain_0_copy" if not columns else "chain_seed_panel")
    seed_fence = after.index("cute.arch.fence_view_async_tmem_store()", seed_begin)
    assert issue < seed_begin < seed_fence < after.index("chain_1_b_ptr =")
    assert "chain_0_c =" not in after
    assert "mbarrier_wait(chain_bars + 1, chain_k_half)" in after
    assert "mbarrier_wait(chain_bars + 1, 0)" not in after
    assert "mbarrier_wait(chain_leaf_bar, chain_leaf_panel)" in after
    assert "mbarrier_wait(chain_leaf_bar, 0)" in after
    assert "mbarrier_wait(chain_leaf_bar, 1)" in after
    assert after.index("chain_allocator.free(chain_tptr)") > after.index(
        "cute.copy(chain_1_copy"
    )
    assert torch.cuda.is_initialized() == initial


@pytest.mark.parametrize(
    "case",
    (
        "kind",
        "lhs_name",
        "shape",
        "strides",
        "rows",
        "columns",
        "tile",
        "kernel_args",
        "premature_out_name",
        "unexpected",
        "leaf",
        "row",
        "col",
        "atom",
        "tensor",
        "registration",
        "duplicate_registration",
        "graph_dtype",
        "graph_kwargs",
    ),
)
def test_paired_descriptor_and_graph_are_rechecked_before_raw_publication(case):
    original = actions.RootStageAction.post_issue_lines
    checked = []

    def inspect(action):
        if action.stage != 0:
            return original(action)
        result = action.sequence.initialized
        assert result is not None and result.paired is not None
        paired = result.paired
        transfer = paired.transfer
        wrapper = transfer.wrapper
        saved = deepcopy(wrapper)
        registered = list(result.codegen.cute_wrapper_plans)
        fields = {
            name: getattr(transfer, name)
            for name in ("leaf", "row", "col", "atom", "tensor")
        }
        leaf_val, leaf_kwargs = transfer.leaf.meta["val"], transfer.leaf.kwargs
        input_users = [
            (node, dict(node.users)) for node in transfer.leaf.all_input_nodes
        ]
        body = list(result.codegen.device_function.body)
        try:
            if case in ("kind", "lhs_name", "rows", "columns"):
                wrapper[case] = "changed"
            elif case in ("shape", "strides", "tile"):
                wrapper[case] = (1, 1)
            elif case == "kernel_args":
                wrapper[case][0] = "changed"
            elif case == "premature_out_name":
                wrapper["out_name"] = "not_yet_registered"
            elif case == "unexpected":
                wrapper["unexpected"] = 0
            elif case in fields:
                object.__setattr__(
                    transfer, case, result.plan.dots[0] if case == "leaf" else "changed"
                )
            elif case == "registration":
                result.codegen.cute_wrapper_plans.remove(wrapper)
            elif case == "duplicate_registration":
                result.codegen.cute_wrapper_plans.append(wrapper)
            elif case == "graph_dtype":
                transfer.leaf.meta["val"] = leaf_val.to(torch.float16)
            else:
                transfer.leaf.kwargs = {**leaf_kwargs, "changed": True}
            with pytest.raises(
                chain._UnsupportedChain, match="paired initial publication"
            ):
                original(action)
            assert action.sequence.paired_issued is False
            assert result.codegen.device_function.body == body
        finally:
            for name, value in fields.items():
                object.__setattr__(transfer, name, value)
            transfer.leaf.meta["val"] = leaf_val
            if case == "graph_kwargs":
                transfer.leaf.kwargs = leaf_kwargs
                for node, users in input_users:
                    node.users.clear()
                    node.users.update(users)
            wrapper.clear()
            wrapper.update(saved)
            result.codegen.cute_wrapper_plans[:] = registered
        checked.append(case)
        return original(action)

    with patch.object(actions.RootStageAction, "post_issue_lines", inspect):
        _source()
    assert checked == [case]


def test_paired_initial_publication_is_required_and_cannot_repeat():
    original = actions.RootStageAction.post_issue_lines
    checked = []

    def inspect(action):
        if action.stage != 0:
            return original(action)
        with pytest.raises(chain._UnsupportedChain, match="publication missing"):
            action.complete()
        assert action.sequence.next_stage == 0
        lines = original(action)
        with pytest.raises(chain._UnsupportedChain, match="publication changed"):
            original(action)
        assert action.sequence.next_stage == 0
        checked.append(lines)
        return lines

    with patch.object(actions.RootStageAction, "post_issue_lines", inspect):
        _source()
    assert len(checked) == 1
    assert checked[0][:2] == [
        "cute.arch.fence_view_async_shared()",
        "cute.arch.sync_threads()",
    ]


def test_paired_preseed_hook_preserves_disabled_original_source():
    with patch.object(roots, "codegen_plain_root", return_value=False):
        before = _source(mode="legacy")
    original = actions.RootStageAction.post_issue_lines
    returned = []

    def inspect(action):
        lines = original(action)
        returned.append(lines)
        return lines

    with patch.object(actions.RootStageAction, "post_issue_lines", inspect):
        after = _source(mode="legacy")
    after = assert_prepared_root_equivalent(before, after)
    assert returned == [[], []]


@pytest.mark.parametrize("schedule", ("serial64", "overlap64"))
def test_paired_terminal_uses_original_half_producer_and_forced_retirement(schedule):
    produced = []
    produce = leaves.produce_half
    post = actions.RootStageAction.post_issue_lines
    completed = []

    def capture_producer(*args, **kwargs):
        lines = produce(*args, **kwargs)
        produced.append((args[4], args[5], tuple(lines)))
        return lines

    def capture_completion(action):
        lines = post(action)
        if action.stage == 1:
            assert lines == []
            completed.append(action)
        return lines

    with (
        patch.object(legacy, "produce_half", side_effect=capture_producer),
        patch.object(
            issues, "emit_k_half_issues", wraps=issues.emit_k_half_issues
        ) as issued,
        patch.object(actions.RootStageAction, "post_issue_lines", capture_completion),
    ):
        _source(schedule=schedule)
    assert len(produced) == len(completed) == issued.call_count == 1
    action = completed[0]
    selected = action.sequence.seeded_inputs
    assert selected is not None and selected.result.paired is not None
    assert produced[0][:2] == (
        selected.result.paired.transfer,
        action.sequence.plan.dots[1].args[0],
    )
    assert issued.call_args.args == ("chain_1", 1, schedule, produced[0][2])
    assert set(issued.call_args.kwargs) == {"retire_each_half", "continuation"}
    assert issued.call_args.kwargs["retire_each_half"] is True
    continuation = issued.call_args.kwargs["continuation"]
    assert continuation.sequence is action.sequence
    assert continuation.actions[1] is action
    assert continuation.continuation.first.node is action.sequence.plan.dots[0]
    assert continuation.continuation.second.node is action.sequence.plan.dots[1]
    assert action.sequence.queued_rhs.inputs is selected
    assert action.sequence.seed_completion.result is selected.result


@pytest.mark.parametrize("schedule_mode", ("serial64", "overlap64"))
@pytest.mark.parametrize(
    "case",
    (
        "seed_missing",
        "rhs_not_enqueued",
        "rhs_other_seed",
        "inputs_lines",
        "inputs_arena",
        "half_overlap",
        "half_order",
        "other_half_mode",
        "wrong_phase",
    ),
)
def test_paired_terminal_keeps_original_seed_rhs_half_phase_negatives(
    case, schedule_mode
):
    # Apply the established same-attempt negative contract to the paired input
    # fixture without changing any assertion or its original geometry proof.
    with patch.object(
        initialized_tests,
        "_serial_source",
        side_effect=lambda *, schedule: _source(schedule=schedule),
    ):
        initialized_tests.test_seeded_stage_rejects_changed_phase_and_input_authority(
            case, schedule_mode
        )


@pytest.mark.parametrize(
    "case", ("not_issued", "integer_issued", "missing", "descriptor", "half_completion")
)
def test_paired_terminal_rejects_missing_or_stale_raw_publication(case):
    emit = stages.emit_stage
    checked = []

    def inspect(*args, **kwargs):
        if args[3] != 1:
            return emit(*args, **kwargs)
        sequence = kwargs["root_actions"].sequence
        result = sequence.initialized
        assert result is not None and result.paired is not None
        wrapper = result.paired.transfer.wrapper
        saved = deepcopy(wrapper)
        old_issued, old_half = sequence.paired_issued, sequence.half_producer
        state = (
            sequence.next_stage,
            list(sequence.staged),
            dict(args[2]),
            list(args[0].device_function.body),
        )
        try:
            if case == "not_issued":
                sequence.paired_issued = False
            elif case == "integer_issued":
                sequence.paired_issued = 1
            elif case == "missing":
                sequence.initialized = replace(result, paired=None)
            elif case == "descriptor":
                wrapper["tile"] = (128, 64)
            else:
                sequence.half_producer = object()
            with pytest.raises(
                chain._UnsupportedChain, match="provenance or readiness"
            ):
                emit(*args, **kwargs)
            assert (
                sequence.next_stage,
                sequence.staged,
                args[2],
                args[0].device_function.body,
            ) == state
        finally:
            sequence.initialized = result
            sequence.paired_issued = old_issued
            sequence.half_producer = old_half
            wrapper.clear()
            wrapper.update(saved)
        checked.append(case)
        return emit(*args, **kwargs)

    with patch.object(stages, "emit_stage", side_effect=inspect):
        _source()
    assert checked == [case]
