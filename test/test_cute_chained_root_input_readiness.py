from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from ._cute_prepared_source import assert_prepared_root_equivalent
from .test_cute_chained_accumulator import _cpu
from .test_cute_chained_accumulator import _initialized_args
from .test_cute_chained_accumulator import _initialized_config
from .test_cute_chained_accumulator import _initialized_pair
from .test_cute_chained_accumulator import _late_rhs_args
from .test_cute_chained_accumulator import _late_rhs_config
from .test_cute_chained_accumulator import _late_rhs_pair
from helion import exc
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chained_plain_root as roots
from helion._compiler.cute import chained_root_stage as actions
from helion._compiler.cute import chained_tcgen_stage as stages


def _source(mode="local", dtype=torch.bfloat16, columns=0, kind="dense"):
    with _cpu():
        if mode == "local":
            return _initialized_pair._bind_isolated(
                _initialized_args(dtype=dtype, n=128, kind=kind)
            ).to_code(_initialized_config(128, cute_chained_seed_tile_columns=columns))
        return _late_rhs_pair._bind_isolated(
            _late_rhs_args(dtype=dtype, n=128, kind=kind)
        ).to_code(
            _late_rhs_config(
                value=mode == "deferred", cute_chained_seed_tile_columns=columns
            )
        )


@pytest.mark.parametrize("mode", ("local", "upfront", "deferred"))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("columns", (0, 32, 64))
def test_original_source_and_both_shared_stages(mode, dtype, columns):
    initial_cuda_state = torch.cuda.is_initialized()
    with patch.object(roots, "codegen_plain_root", return_value=False):
        before = _source(mode, dtype, columns)
    with patch.object(stages, "emit_stage", wraps=stages.emit_stage) as emitted:
        after = _source(mode, dtype, columns)
    after = assert_prepared_root_equivalent(before, after)
    assert [call.args[3] for call in emitted.call_args_list] == [0, 1]
    sequence = emitted.call_args.kwargs["root_actions"].sequence
    assert sequence.seeded_inputs.mode == mode
    assert sequence.next_stage == 2
    assert sequence.seed_completion.result is sequence.initialized
    assert "chain_1_mma.set(tcgen05.Field.ACCUMULATE, True)" in after
    assert "chain_1_mma.set(tcgen05.Field.ACCUMULATE, False)" not in after
    if mode != "deferred":
        assert sequence.queued_rhs is None
        assert sequence.input_completion == actions.RootInputCompletion(
            sequence.seeded_inputs, 1
        )
    assert torch.cuda.is_initialized() == initial_cuda_state


@pytest.mark.parametrize("mode", ("local", "upfront"))
def test_source_control_preserves_already_initialized_worker_state(mode):
    with patch("torch.cuda.is_initialized", return_value=True):
        test_original_source_and_both_shared_stages(mode, torch.bfloat16, 0)


@pytest.mark.parametrize("mode", ("local", "upfront"))
@pytest.mark.parametrize("stage", (0, 1))
@pytest.mark.parametrize("hook", ("after_commit", "after_wait", "complete"))
def test_same_object_execution_fields_rechecked(mode, stage, hook):
    original = getattr(actions.RootStageAction, hook)
    checked = []

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
                checked.append(name)
        return original(self, *args, **kwargs)

    with patch.object(actions.RootStageAction, hook, inspect):
        _source(mode)
    assert len(checked) == 8


@pytest.mark.parametrize("mode", ("local", "upfront"))
@pytest.mark.parametrize("stage", (0, 1))
@pytest.mark.parametrize("hook", ("after_commit", "after_wait"))
def test_missing_copy_protocol_never_completes(mode, stage, hook):
    original = getattr(actions.RootStageAction, hook)

    def drop(self, *args, **kwargs):
        return [] if self.stage == stage else original(self, *args, **kwargs)

    with (
        patch.object(actions.RootStageAction, hook, drop),
        pytest.raises(
            exc.BackendUnsupported,
            match="initialized input (copy wait changed|completion missing)",
        ),
    ):
        _source(mode)


@pytest.mark.parametrize("mode", ("local", "upfront"))
def test_same_object_owner_layout_and_program_mutations(mode):
    original = stages.emit_stage
    checked = []

    def inspect(*args, **kwargs):
        action = kwargs["root_actions"]
        if args[3] == 1:
            selected = action.sequence.seeded_inputs
            assert selected is not None and selected.matches(args[1])
            for target, name, value in (
                (selected, "lines", ("different program",)),
                (selected, "mode", "deferred"),
                (selected, "inner_axis", 7),
                (selected, "k_schedule", object()),
                (selected.layouts[0], "shape", (64, 128)),
                (selected.layouts[0], "owner_elements", 1),
                (selected.layouts[1], "inner", 7),
                (selected.layouts[1], "dtype", torch.float32),
                (selected.layouts[1], "separate", not selected.layouts[1].separate),
            ):
                old = getattr(target, name)
                try:
                    object.__setattr__(target, name, value)
                    assert not selected.matches(args[1])
                    with pytest.raises(
                        chain._UnsupportedChain, match="provenance or readiness"
                    ):
                        original(*args, **kwargs)
                finally:
                    object.__setattr__(target, name, old)
                checked.append(name)
        return original(*args, **kwargs)

    with patch.object(stages, "emit_stage", side_effect=inspect):
        _source(mode)
    assert len(checked) == 9


@pytest.mark.parametrize("mode", ("local", "upfront"))
def test_stage_zero_copy_completion_is_not_local_rhs_completion(mode):
    original = actions.RootStageAction.after_wait
    events = []

    def inspect(self, *args, **kwargs):
        seq = self.sequence
        if self.stage == 0:
            assert seq.input_completion is None
            assert seq.seed_completion is None
        else:
            assert seq.input_completion == actions.RootInputCompletion(
                seq.seeded_inputs, 0
            )
            assert seq.seed_completion.result is seq.initialized
            assert seq.input_waited is False
            assert seq.input_producers == ("a", "b")
        result = original(self, *args, **kwargs)
        assert seq.input_completion is not None
        assert seq.input_completion.stage == self.stage
        assert seq.input_waited is True
        events.append(self.stage)
        return result

    with patch.object(actions.RootStageAction, "after_wait", inspect):
        _source(mode)
    assert events == [0, 1]


@pytest.mark.parametrize("mode", ("local", "upfront"))
@pytest.mark.parametrize("case", ("seed", "copy", "wrong_stage", "queued", "half"))
def test_stage_one_requires_exact_seed_and_input_receipts(mode, case):
    original = stages.emit_stage
    checked = []

    def inspect(*args, **kwargs):
        if args[3] == 1:
            seq = kwargs["root_actions"].sequence
            name, value = {
                "seed": ("seed_completion", None),
                "copy": ("input_completion", None),
                "wrong_stage": (
                    "input_completion",
                    actions.RootInputCompletion(seq.seeded_inputs, 1),
                ),
                "queued": (
                    "queued_rhs",
                    actions.RootQueuedRhs(seq.seeded_inputs, seq.seed_completion),
                ),
                "half": ("half_producer", object()),
            }[case]
            old = getattr(seq, name)
            try:
                setattr(seq, name, value)
                with pytest.raises(
                    chain._UnsupportedChain, match="provenance or readiness"
                ):
                    original(*args, **kwargs)
            finally:
                setattr(seq, name, old)
            checked.append(case)
        return original(*args, **kwargs)

    with patch.object(stages, "emit_stage", side_effect=inspect):
        _source(mode)
    assert checked == [case]


@pytest.mark.parametrize("mode", ("local", "upfront"))
@pytest.mark.parametrize("stage", (0, 1))
def test_duplicate_wait_commit_and_reordered_producer_reject(mode, stage):
    original = actions.RootStageAction.after_wait
    checked = []

    def inspect(self, cg, boundaries):
        result = original(self, cg, boundaries)
        if self.stage == stage:
            for operation in (
                lambda: original(self, cg, boundaries),
                self.after_commit,
                lambda: self.operand_lines(cg, boundaries, "a"),
                lambda: self.operand_lines(cg, boundaries, "b"),
            ):
                with pytest.raises(chain._UnsupportedChain, match="initialized input"):
                    operation()
                checked.append(True)
        return result

    with patch.object(actions.RootStageAction, "after_wait", inspect):
        _source(mode)
    assert len(checked) == 4


@pytest.mark.parametrize("mode", ("local", "upfront"))
def test_failed_original_producer_does_not_publish_readiness(mode):
    from helion._compiler.cute import chained_tcgen05 as original

    method = actions.RootStageAction.operand_lines
    checked = []

    def inspect(self, cg, boundaries, role):
        if self.stage == 1 and role == "a":
            before = self.sequence.input_producers
            with (
                patch.object(
                    original,
                    "_stage",
                    side_effect=chain._UnsupportedChain("producer failed"),
                ),
                pytest.raises(chain._UnsupportedChain, match="producer failed"),
            ):
                method(self, cg, boundaries, role)
            assert self.sequence.input_producers == before == ()
            assert self.sequence.input_waited is False
            assert self.sequence.input_completion is not None
            assert self.sequence.input_completion.stage == 0
            checked.append(True)
        return method(self, cg, boundaries, role)

    with patch.object(actions.RootStageAction, "operand_lines", inspect):
        _source(mode)
    assert checked == [True]


@pytest.mark.parametrize("mode", ("local", "upfront"))
@pytest.mark.parametrize("stage", (0, 1))
@pytest.mark.parametrize("hook", ("operand_lines", "after_commit", "after_wait"))
def test_other_stage_cannot_use_current_input_context(mode, stage, hook):
    original = getattr(actions.RootStageAction, hook)
    checked = []

    def inspect(self, *args, **kwargs):
        if self.stage == stage:
            seq = self.sequence
            before = (
                seq.input_producers,
                seq.input_committed,
                seq.input_waited,
                seq.input_completion,
            )
            with pytest.raises(chain._UnsupportedChain, match="input context"):
                original(seq.action(1 - stage), *args, **kwargs)
            assert before == (
                seq.input_producers,
                seq.input_committed,
                seq.input_waited,
                seq.input_completion,
            )
            checked.append(True)
        return original(self, *args, **kwargs)

    with patch.object(actions.RootStageAction, hook, inspect):
        _source(mode)
    assert len(checked) == (2 if hook == "operand_lines" else 1)


@pytest.mark.parametrize("mode", ("local", "upfront"))
@pytest.mark.parametrize(
    "hook", ("operand_lines", "after_commit", "after_wait", "complete")
)
def test_active_input_protocol_cannot_be_dropped(mode, hook):
    original = getattr(actions.RootStageAction, hook)
    checked = []

    def inspect(self, *args, **kwargs):
        if self.stage == 1:
            seq = self.sequence
            selected = seq.seeded_inputs
            assert selected is not None
            for target, name, value in (
                (seq, "seeded_inputs", None),
                (selected, "mode", "deferred"),
            ):
                old = getattr(target, name)
                try:
                    object.__setattr__(target, name, value)
                    with pytest.raises(chain._UnsupportedChain, match="input context"):
                        original(self, *args, **kwargs)
                finally:
                    object.__setattr__(target, name, old)
                checked.append(True)
        return original(self, *args, **kwargs)

    with patch.object(actions.RootStageAction, hook, inspect):
        _source(mode)
    assert len(checked) == (4 if hook == "operand_lines" else 2)
