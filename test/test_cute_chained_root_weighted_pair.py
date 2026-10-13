from __future__ import annotations

from dataclasses import replace
import hashlib
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_vector_group_integration import _args
from .test_cute_chained_vector_group_integration import _config
from .test_cute_chained_vector_group_integration import _cpu_codegen
from .test_cute_chained_vector_group_integration import _root_distinct_leaves
from .test_cute_chained_vector_group_integration import _source
from helion import exc
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chained_plain_root as roots
from helion._compiler.cute import chained_root_stage as actions
from helion._compiler.cute import chained_tcgen05 as tcgen
from helion._compiler.cute import chained_tcgen_stage as stages


def _code(dtype=torch.bfloat16, mask=0, group=False):
    return _source("root", _args("root", dtype=dtype, mask_kind=mask), group)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("mask", (0, 1, 2))
def test_original_source_and_shared_pair(dtype, mask):
    initial = torch.cuda.is_initialized()
    with patch.object(roots, "codegen_plain_root", return_value=False):
        before = _code(dtype, mask)
    with patch.object(stages, "emit_stage", wraps=stages.emit_stage) as shared:
        after = _code(dtype, mask)
    assert after == before
    assert [call.args[3] for call in shared.call_args_list] == [0, 1]
    sequence = shared.call_args.kwargs["root_actions"].sequence
    assert sequence.pair_inputs is sequence.pair_selection
    assert sequence.pair_completed == sequence.next_stage == 2
    assert sequence.pair_bridged is True
    assert all(geometry.native_rows == 128 for geometry in sequence.geometries)
    if dtype == torch.bfloat16 and mask == 0:
        assert hashlib.sha256(after.encode()).hexdigest() == (
            "d703e43400ac176d5f209834bd761b454f12e96686b7330f2ccc5cafcd67aeff"
        )
    assert torch.cuda.is_initialized() == initial


def test_original_four_axis_evaluations_and_local_rhs_order():
    original_axis, original_stage = chain._operand_inner_axis, tcgen._stage
    order: list[tuple[object, ...]] = []

    def axis(*args, **kwargs):
        result = original_axis(*args, **kwargs)
        order.append(("axis", result))
        return result

    def producer(*args, **kwargs):
        order.append(("producer", args[4], args[5]))
        return original_stage(*args, **kwargs)

    records = []
    for old in (True, False):
        order.clear()
        with (
            patch.object(chain, "_operand_inner_axis", side_effect=axis),
            patch.object(tcgen, "_stage", side_effect=producer),
            patch.object(roots, "codegen_plain_root", return_value=False)
            if old
            else patch.object(
                roots, "codegen_plain_root", wraps=roots.codegen_plain_root
            ),
        ):
            _code()
        records.append(list(order))
    assert (
        records[0]
        == records[1]
        == [
            ("axis", 1),
            ("axis", 0),
            ("axis", 1),
            ("axis", 1),
            ("producer", 0, "a"),
            ("producer", 0, "b"),
            ("producer", 1, "b"),
        ]
    )


def test_group_source_unchanged_with_shared_pair():
    with patch.object(roots, "codegen_plain_root", return_value=False):
        before = _code(group=True)
    with patch.object(stages, "emit_stage", wraps=stages.emit_stage) as shared:
        after = _code(group=True)
    assert after == before
    assert [call.args[3] for call in shared.call_args_list] == [0, 1]


@pytest.mark.parametrize("stage", (0, 1))
@pytest.mark.parametrize(
    "hook", ("operand_lines", "after_commit", "after_wait", "complete")
)
def test_context_mutation_rejected_at_every_protocol_point(stage, hook):
    original = getattr(actions.RootStageAction, hook)
    seen = set()

    def inspect(self, *args, **kwargs):
        if self.stage == stage:
            execution = self.sequence.input_execution
            assert execution is not None
            for name in execution.__dataclass_fields__:
                old = getattr(execution, name)
                try:
                    object.__setattr__(
                        execution, name, 256 if name == "threads" else "changed"
                    )
                    with pytest.raises(chain._UnsupportedChain, match="input context"):
                        original(self, *args, **kwargs)
                finally:
                    object.__setattr__(execution, name, old)
                seen.add(name)
        return original(self, *args, **kwargs)

    with patch.object(actions.RootStageAction, hook, inspect):
        _code()
    assert len(seen) == 8


@pytest.mark.parametrize("stage", (0, 1))
@pytest.mark.parametrize("hook", ("after_commit", "after_wait"))
def test_missing_copy_completion_rejects(stage, hook):
    original = getattr(actions.RootStageAction, hook)

    def drop(self, *args, **kwargs):
        return [] if self.stage == stage else original(self, *args, **kwargs)

    with (
        patch.object(actions.RootStageAction, hook, drop),
        pytest.raises(exc.BackendUnsupported, match="input (copy wait|completion)"),
    ):
        _code()


def test_local_rhs_and_completed_c_are_separate_authorities():
    original = actions.RootStageAction.after_wait
    seen = []

    def inspect(self, *args, **kwargs):
        sequence = self.sequence
        assert sequence.pair_completed == self.stage
        assert not sequence.input_waited
        assert not sequence.pair_bridged
        result = original(self, *args, **kwargs)
        assert sequence.input_waited
        assert sequence.pair_bridged is (self.stage == 1)
        seen.append(self.stage)
        return result

    with patch.object(actions.RootStageAction, "after_wait", inspect):
        _code()
    assert seen == [0, 1]


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("reduction", (16, 64, 128))
def test_unrelated_weighted_inputs_and_native_shapes(dtype, reduction):
    args = (
        torch.zeros((256, reduction), dtype=dtype),
        torch.zeros((128, reduction), dtype=dtype),
        torch.zeros((128, 256), dtype=dtype),
    )

    def source():
        with _cpu_codegen():
            return _root_distinct_leaves._bind_isolated(args).to_code(
                _config("root", False)
            )

    with patch.object(roots, "codegen_plain_root", return_value=False):
        before = source()
    with patch.object(stages, "emit_stage", wraps=stages.emit_stage) as shared:
        after = source()
    assert after == before
    assert [call.args[3] for call in shared.call_args_list] == [0, 1]


@pytest.mark.parametrize(
    "case", ("missing", "role", "source", "operand", "revision", "axis_a", "axis_b")
)
def test_late_decline_preserves_original_prepared_loop(case):
    original = actions.capture_root_pair_inputs
    with patch.object(roots, "codegen_plain_root", return_value=False):
        before = _code()
    checked = []

    def decline(cg, plan, boundaries, axes, bridges):
        axes, bridges = dict(axes), dict(bridges)
        if case == "missing":
            bridges = {}
        elif case.startswith("axis"):
            axes[0, case[-1]] = 0
        else:
            fields = {
                "role": "b",
                "source": plan.dots[1],
                "operand": plan.dots[1].args[1],
                "revision": (),
            }
            bridges[1] = replace(bridges[1], **{case: fields[case]})
        assert original(cg, plan, boundaries, axes, bridges) is None
        checked.append(case)
        return None

    with (
        patch.object(actions, "capture_root_pair_inputs", side_effect=decline),
        patch.object(
            stages, "emit_stage", side_effect=AssertionError("unproved action")
        ),
    ):
        assert _code() == before
    assert checked == [case]


@pytest.mark.parametrize("stage", (0, 1))
def test_deep_layout_graph_bridge_and_selection_mutations(stage):
    original = stages.emit_stage
    checked = []

    def inspect(*args, **kwargs):
        action = kwargs["root_actions"]
        if action.stage != stage:
            return original(*args, **kwargs)
        sequence = action.sequence
        selected = sequence.pair_inputs
        assert selected is not None
        mutations = (
            (sequence, "pair_inputs", None),
            (sequence, "pair_selection", None),
            (sequence, "inner_axes", ((0, 0), (1, 0))),
            (sequence, "pair_completed", 8),
            (selected, "axes", ((0, 0), (1, 0))),
            (selected, "options", ()),
            (selected, "revision", ()),
            (selected.layouts[0], "shape", (64, 32)),
            (selected.layouts[0], "owner_elements", 1),
            (selected.layouts[1], "dtype", torch.float32),
            (selected.layouts[2], "inner", 3),
            (selected.layouts[2], "separate", True),
            (selected.bridge, "lines", ("different bridge",)),
            (selected.bridge, "operand", args[1].dots[1].args[1]),
            (args[1], "axes", ()),
        )
        for target, name, value in mutations:
            old = getattr(target, name)
            try:
                object.__setattr__(target, name, value)
                with pytest.raises(
                    chain._UnsupportedChain, match="provenance or readiness"
                ):
                    original(*args, **kwargs)
            finally:
                object.__setattr__(target, name, old)
            checked.append(name)
        graph_node = args[1].dots[0]
        old_value = graph_node.meta["val"]
        try:
            graph_node.meta["val"] = old_value.to(torch.float16)
            with pytest.raises(
                chain._UnsupportedChain, match="provenance or readiness"
            ):
                original(*args, **kwargs)
        finally:
            graph_node.meta["val"] = old_value
        checked.append("graph_dtype")
        return original(*args, **kwargs)

    with patch.object(stages, "emit_stage", side_effect=inspect):
        _code()
    assert len(checked) == 16


@pytest.mark.parametrize(
    "hook", ("operand_lines", "after_commit", "after_wait", "complete")
)
def test_cross_stage_action_never_advances_active_ledger(hook):
    original = getattr(actions.RootStageAction, hook)
    checked = []

    def inspect(self, *args, **kwargs):
        if self.stage == 0:
            state = (
                self.sequence.input_producers,
                self.sequence.input_committed,
                self.sequence.input_waited,
                self.sequence.next_stage,
            )
            with pytest.raises(chain._UnsupportedChain):
                original(self.sequence.action(1), *args, **kwargs)
            assert state == (
                self.sequence.input_producers,
                self.sequence.input_committed,
                self.sequence.input_waited,
                self.sequence.next_stage,
            )
            checked.append(hook)
        return original(self, *args, **kwargs)

    with patch.object(actions.RootStageAction, hook, inspect):
        _code()
    assert checked


def test_stale_accepted_pair_cannot_fall_back_before_first_stage():
    original = actions.plan_root_stage_sequence

    def corrupt(*args, **kwargs):
        selected = kwargs["pair_inputs"]
        object.__setattr__(selected.layouts[0], "owner_elements", 1)
        return original(*args, **kwargs)

    with (
        patch.object(actions, "plan_root_stage_sequence", side_effect=corrupt),
        patch.object(stages, "emit_stage", side_effect=AssertionError("stale action")),
        pytest.raises(exc.BackendUnsupported, match="accepted local root pair"),
    ):
        _code()


@pytest.mark.parametrize("stage", (0, 1))
@pytest.mark.parametrize("hook", ("operand_lines", "after_wait"))
@pytest.mark.parametrize("change", ("new_map", "same_map", "codegen"))
def test_actual_hook_arguments_rechecked_before_effects(stage, hook, change):
    original = getattr(actions.RootStageAction, hook)
    checked = []

    def inspect(self, cg, boundaries, *args, **kwargs):
        if self.stage == stage:
            sequence = self.sequence
            state = (
                sequence.input_producers,
                sequence.input_committed,
                sequence.input_waited,
                sequence.pair_completed,
                sequence.pair_bridged,
                tuple(sequence.staged),
            )
            supplied = dict(boundaries) if change == "new_map" else boundaries
            old = dict(boundaries)
            if change != "codegen":
                supplied[sequence.plan.dots[stage].args[0]] = "unapproved_boundary"
            try:
                with pytest.raises(chain._UnsupportedChain, match="context|arguments"):
                    original(
                        self,
                        object() if change == "codegen" else cg,
                        supplied,
                        *args,
                        **kwargs,
                    )
            finally:
                if change == "same_map":
                    boundaries.clear()
                    boundaries.update(old)
            assert state == (
                sequence.input_producers,
                sequence.input_committed,
                sequence.input_waited,
                sequence.pair_completed,
                sequence.pair_bridged,
                tuple(sequence.staged),
            )
            checked.append(change)
        return original(self, cg, boundaries, *args, **kwargs)

    with patch.object(actions.RootStageAction, hook, inspect):
        _code()
    assert checked
