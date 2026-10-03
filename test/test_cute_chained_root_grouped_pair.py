from __future__ import annotations

from dataclasses import replace
import hashlib
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_root_weighted_pair import _code
from helion import exc
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chained_plain_root as roots
from helion._compiler.cute import chained_root_stage as actions
from helion._compiler.cute import chained_tcgen05 as tcgen
from helion._compiler.cute import chained_tcgen_stage as stages
from helion._compiler.cute import chained_vector_stage as vector


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("mask", (0, 1))
def test_exact_original_group_source_and_both_common_stages(dtype, mask):
    initialized = torch.cuda.is_initialized()
    with patch.object(roots, "codegen_plain_root", return_value=False):
        before = _code(dtype, mask, True)
    with patch.object(stages, "emit_stage", wraps=stages.emit_stage) as shared:
        after = _code(dtype, mask, True)
    assert before == after
    assert [call.args[3] for call in shared.call_args_list] == [0, 1]
    sequence = shared.call_args.kwargs["root_actions"].sequence
    assert sequence.grouped_attempted and sequence.grouped_required
    assert sequence.grouped_production is not None
    assert sequence.grouped_obligation == (
        sequence.grouped_production,
        sequence.grouped_production.facts(),
    )
    assert sequence.pair_completed == sequence.next_stage == 2
    assert sequence.pair_bridged
    assert sequence.pair_staging.group_activated
    if dtype == torch.bfloat16 and mask == 0:
        assert hashlib.sha256(after.encode()).hexdigest() == (
            "5e8698e3be993ef4f258cb4af9dc14a15f3ba497bf303ad305d738b55183d21b"
        )
    assert torch.cuda.is_initialized() == initialized


def test_original_axis_and_group_call_order():
    original_axis = chain._operand_inner_axis
    original_group = vector.emit_vector_stage_group
    original_stage = tcgen._stage
    records = []
    order: list[tuple[object, ...]] = []

    def axis(*args, **kwargs):
        value = original_axis(*args, **kwargs)
        order.append(("axis", value))
        return value

    def group(*args, **kwargs):
        order.append(("group", kwargs["tag"]))
        return original_group(*args, **kwargs)

    def single(*args, **kwargs):
        order.append(("single", args[4], args[5]))
        return original_stage(*args, **kwargs)

    for old in (True, False):
        order.clear()
        with (
            patch.object(chain, "_operand_inner_axis", side_effect=axis),
            patch.object(tcgen, "emit_vector_stage_group", side_effect=group),
            patch.object(vector, "emit_vector_stage_group", side_effect=group),
            patch.object(tcgen, "_stage", side_effect=single),
            patch.object(roots, "codegen_plain_root", return_value=False)
            if old
            else patch.object(
                roots, "codegen_plain_root", wraps=roots.codegen_plain_root
            ),
        ):
            _code(group=True)
        records.append(list(order))
    assert (
        records[0]
        == records[1]
        == [
            ("axis", 1),
            ("axis", 0),
            ("axis", 1),
            ("axis", 1),
            ("group", "chain_0_vector_group"),
            ("single", 1, "b"),
        ]
    )


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_original_reduction_axis_mask_rejection(dtype):
    for old in (True, False):
        with (
            patch.object(roots, "codegen_plain_root", return_value=False)
            if old
            else patch.object(
                roots, "codegen_plain_root", wraps=roots.codegen_plain_root
            ),
            pytest.raises(exc.BackendUnsupported, match="shared coordinate-owned"),
        ):
            _code(dtype, 2, True)


def test_helper_none_preserves_scalar_attempt_and_final_true_rejection():
    seen = []
    original = actions.RootStageAction.after_commit

    def inspect(self):
        if self.stage == 0:
            sequence = self.sequence
            assert sequence.grouped_attempted and not sequence.grouped_required
            assert sequence.grouped_production is sequence.grouped_obligation is None
            assert not sequence.pair_staging.group_activated
            assert sequence.input_producers == ("a", "b")
            seen.append(0)
        return original(self)

    with (
        patch.object(vector, "emit_vector_stage_group", return_value=None),
        patch.object(actions.RootStageAction, "after_commit", inspect),
        pytest.raises(exc.BackendUnsupported, match="shared coordinate-owned"),
    ):
        _code(group=True)
    assert seen == [0]


@pytest.mark.parametrize(
    "mutation", ("cg", "map", "same_map", "role", "shape", "offset", "geometry")
)
def test_group_actual_arguments_before_any_mutation(mutation):
    original = actions.RootStageAction.grouped_operand_lines
    seen = []

    def inspect(self, cg, boundaries, operands):
        sequence = self.sequence
        before = (
            sequence.input_producers,
            sequence.grouped_attempted,
            sequence.grouped_required,
        )
        args = [cg, boundaries, operands]
        prior = dict(boundaries)
        if mutation == "cg":
            args[0] = object()
        elif mutation == "map":
            args[1] = {**boundaries, sequence.plan.dots[0].args[0]: "unapproved"}
        elif mutation == "same_map":
            boundaries[sequence.plan.dots[0].args[0]] = "unapproved"
        else:
            changes = {
                "role": {"role": "b"},
                "shape": {"shape": (64, 32)},
                "offset": {"offset": 8},
                "geometry": {"geometry": replace(operands[0].geometry, transpose=True)},
            }
            args[2] = (replace(operands[0], **changes[mutation]), *operands[1:])
        try:
            with pytest.raises(chain._UnsupportedChain):
                original(self, *args)
            assert before == (
                sequence.input_producers,
                sequence.grouped_attempted,
                sequence.grouped_required,
            )
            assert not sequence.pair_staging.group_activated
        finally:
            boundaries.clear()
            boundaries.update(prior)
        seen.append(mutation)
        return original(self, cg, boundaries, operands)

    with patch.object(actions.RootStageAction, "grouped_operand_lines", inspect):
        _code(group=True)
    assert seen == [mutation]


@pytest.mark.parametrize(
    "hook", ("check_grouped_body", "after_commit", "after_wait", "complete")
)
@pytest.mark.parametrize(
    "mutation", ("drop", "obligation", "required", "body", "execution", "descriptor")
)
def test_successful_receipt_and_independent_obligation_rechecked(hook, mutation):
    original = getattr(actions.RootStageAction, hook)
    seen = []

    def inspect(self, *args, **kwargs):
        if self.stage == 0:
            sequence = self.sequence
            receipt = sequence.grouped_production
            assert receipt is not None
            targets = {
                "drop": (sequence, "grouped_production", None),
                "obligation": (sequence, "grouped_obligation", None),
                "required": (sequence, "grouped_required", False),
                "body": (receipt, "body", ("different",)),
                "execution": (sequence.input_execution, "thread", "changed"),
                "descriptor": (receipt.operands[0], "offset", 32),
            }
            target, field, value = targets[mutation]
            before = getattr(target, field)
            try:
                object.__setattr__(target, field, value)
                with pytest.raises(chain._UnsupportedChain):
                    original(self, *args, **kwargs)
            finally:
                object.__setattr__(target, field, before)
            seen.append(mutation)
        return original(self, *args, **kwargs)

    with patch.object(actions.RootStageAction, hook, inspect):
        _code(group=True)
    assert seen == [mutation]


def test_helper_failure_or_stale_success_cannot_activate_or_publish():
    original = vector.emit_vector_stage_group
    original_action = actions.RootStageAction.grouped_operand_lines
    seen = []

    def inspect(self, cg, boundaries, operands):
        sequence = self.sequence
        saved = dict(boundaries)

        def stale(*args, **kwargs):
            result = original(*args, **kwargs)
            assert result is not None
            boundaries[sequence.plan.dots[0].args[0]] = "unapproved"
            return result

        try:
            with (
                patch.object(vector, "emit_vector_stage_group", side_effect=stale),
                pytest.raises(chain._UnsupportedChain),
            ):
                original_action(self, cg, boundaries, operands)
            assert sequence.input_producers == ()
            assert not sequence.grouped_attempted and not sequence.grouped_required
            assert not sequence.pair_staging.group_activated
            assert sequence.grouped_production is sequence.grouped_obligation is None
            seen.append(True)
        finally:
            boundaries.clear()
            boundaries.update(saved)
        # The deliberately failed probe consumed names. End this failed attempt;
        # never replay the mathematical body in the same codegen attempt.
        raise chain._UnsupportedChain("test stale probe stopped")

    with (
        patch.object(actions.RootStageAction, "grouped_operand_lines", inspect),
        pytest.raises(exc.BackendUnsupported, match="test stale probe stopped"),
    ):
        _code(group=True)
    assert seen == [True]


def test_group_is_once_only_and_does_not_grant_copy_or_result_readiness():
    original = actions.RootStageAction.check_grouped_body
    seen = []

    def inspect(self, cg, boundaries, lines):
        original(self, cg, boundaries, lines)
        sequence = self.sequence
        assert sequence.input_producers == ("a", "b")
        assert not sequence.input_committed and not sequence.input_waited
        assert sequence.pair_completed == 0
        with pytest.raises(chain._UnsupportedChain):
            self.grouped_operand_lines(
                cg, boundaries, sequence.grouped_production.operands
            )
        with pytest.raises(chain._UnsupportedChain):
            self.operand_lines(cg, boundaries, "a")
        with pytest.raises(chain._UnsupportedChain):
            self.after_wait(cg, boundaries)
        with pytest.raises(chain._UnsupportedChain):
            self.complete()
        with pytest.raises(chain._UnsupportedChain):
            sequence.action(1).grouped_operand_lines(
                cg, boundaries, sequence.grouped_production.operands
            )
        seen.append(True)

    with patch.object(actions.RootStageAction, "check_grouped_body", inspect):
        _code(group=True)
    assert seen == [True]


def test_all_execution_and_policy_fields_reject_before_group_attempt():
    original = actions.RootStageAction.grouped_operand_lines
    seen = []

    def inspect(self, cg, boundaries, operands):
        sequence = self.sequence
        execution = sequence.input_execution
        assert execution is not None
        changes = [
            (execution, field, 256 if field == "threads" else "changed")
            for field in execution.__dataclass_fields__
        ] + [
            (sequence, "pair_staging", None),
            (sequence, "pair_staging_selection", ()),
            (sequence.pair_staging, "enabled", False),
            (sequence.pair_staging, "group_enabled", False),
            (sequence.pair_staging, "group_activated", True),
            (sequence.pointwise_unroll, "factor", 2),
        ]
        for target, field, value in changes:
            prior = getattr(target, field)
            try:
                object.__setattr__(target, field, value)
                with pytest.raises(chain._UnsupportedChain):
                    original(self, cg, boundaries, operands)
                assert not sequence.grouped_attempted
                assert sequence.input_producers == ()
                seen.append(field)
            finally:
                object.__setattr__(target, field, prior)
        return original(self, cg, boundaries, operands)

    with patch.object(actions.RootStageAction, "grouped_operand_lines", inspect):
        _code(group=True)
    assert len(seen) == 14


@pytest.mark.parametrize("result", (None, [], ["unproved body"]))
def test_incomplete_helper_does_not_publish_activation(result):
    # None is the original fallback; fabricated/empty success is not a receipt.
    message = "shared coordinate-owned" if result is None else "did not complete"
    with (
        patch.object(vector, "emit_vector_stage_group", return_value=result),
        pytest.raises(exc.BackendUnsupported, match=message),
    ):
        _code(group=True)


def test_installed_body_matches_completed_group_receipt():
    original = actions.RootStageAction.check_grouped_body
    seen = []

    def inspect(self, cg, boundaries, lines):
        with pytest.raises(chain._UnsupportedChain, match="body changed"):
            original(self, cg, boundaries, [*lines, "unapproved extra line"])
        seen.append(True)
        return original(self, cg, boundaries, lines)

    with patch.object(actions.RootStageAction, "check_grouped_body", inspect):
        _code(group=True)
    assert seen == [True]
