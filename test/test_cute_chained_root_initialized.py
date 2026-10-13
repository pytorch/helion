from __future__ import annotations

from dataclasses import replace
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
from .test_cute_chained_pipeline import args as leaf_args
from .test_cute_chained_pipeline import leaf_config
from .test_cute_chained_pipeline import pair as leaf_pair
from helion import exc
from helion._compiler.cute import chained_initialized_accumulator as seeds
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chained_plain_root as roots
from helion._compiler.cute import chained_root_stage as actions
from helion._compiler.cute import chained_tcgen05 as legacy
from helion._compiler.cute import chained_tcgen_stage as stages


def _source(*, columns=0, late=True, kind="dense", **options):
    with _cpu():
        if late:
            return _late_rhs_pair._bind_isolated(_late_rhs_args(kind=kind)).to_code(
                _late_rhs_config(cute_chained_seed_tile_columns=columns, **options)
            )
        return _initialized_pair._bind_isolated(_initialized_args(kind=kind)).to_code(
            _initialized_config(cute_chained_seed_tile_columns=columns, **options)
        )


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("columns", (0, 32, 64))
@pytest.mark.parametrize("kind", ("dense", "major"))
def test_initialized_first_stage_keeps_original_full_source(dtype, columns, kind):
    values = _initialized_args(dtype=dtype, n=128, kind=kind)
    config = _initialized_config(128, cute_chained_seed_tile_columns=columns)
    initial = torch.cuda.is_initialized()
    with _cpu(), patch.object(roots, "codegen_plain_root", return_value=False):
        before = _initialized_pair._bind_isolated(values).to_code(config)
    with (
        _cpu(),
        patch.object(
            legacy, "codegen_chained_tcgen05", side_effect=AssertionError("old root")
        ),
        patch.object(stages, "emit_stage", wraps=stages.emit_stage) as emitted,
    ):
        after = _initialized_pair._bind_isolated(values).to_code(config)
    after = assert_prepared_root_equivalent(before, after)
    assert [call.args[3] for call in emitted.call_args_list] == [0, 1]
    action = emitted.call_args.kwargs["root_actions"]
    assert action.sequence.initialized.max_columns == columns
    assert action.sequence.next_stage == 2
    assert action.sequence.handles(1)
    assert action.sequence.seeded_inputs.mode == "local"
    assert "chain_0_c =" not in after
    assert ("chain_0_values =" in after) is (columns == 0)
    assert torch.cuda.is_initialized() == initial


@pytest.mark.parametrize("columns", (0, 32))
@pytest.mark.parametrize("kind", ("dense", "offset", "stride"))
def test_seed_lowering_and_deferred_rhs_have_one_original_position(columns, kind):
    order: list[object] = []
    seed = seeds.codegen_seed
    stage = legacy._stage
    emit = stages.emit_stage

    def seed_spy(*args, **kwargs):
        order.append("seed")
        return seed(*args, **kwargs)

    def stage_spy(*args, **kwargs):
        order.append((args[4], args[5]))
        if args[4] == 1:
            assert args[1].dots[0] not in args[2]
        return stage(*args, **kwargs)

    def emit_spy(*args, **kwargs):
        order.append("shared")
        result = emit(*args, **kwargs)
        assert args[1].dots[0] not in args[2]
        assert result[-1] == "cute.arch.sync_threads()"
        return result

    with patch.object(roots, "codegen_plain_root", return_value=False):
        before = _source(columns=columns, kind=kind)
    with (
        patch.object(seeds, "codegen_seed", side_effect=seed_spy),
        patch.object(legacy, "_stage", side_effect=stage_spy),
        patch.object(stages, "emit_stage", side_effect=emit_spy),
    ):
        after = _source(columns=columns, kind=kind)
    after = assert_prepared_root_equivalent(before, after)
    assert order == ["seed", (1, "b"), "shared", (0, "a"), (0, "b"), "shared", (1, "a")]
    assert after.count("chain_1_b_ptr = chain_b_workspace") == 1
    assert after.index("fence_view_async_tmem_store()") < after.index("chain_1_b_ptr =")


@pytest.mark.parametrize("schedule", ("serial64", "overlap64"))
def test_paired_tma_stays_byte_exact_with_both_shared_stages(schedule):
    values = leaf_args()
    config = leaf_config(schedule=schedule)
    with _cpu(), patch.object(roots, "codegen_plain_root", return_value=False):
        before = leaf_pair._bind_isolated(values).to_code(config)
    with (
        _cpu(),
        patch.object(stages, "emit_stage", wraps=stages.emit_stage) as emitted,
    ):
        after = leaf_pair._bind_isolated(values).to_code(config)
    after = assert_prepared_root_equivalent(before, after)
    assert [call.args[3] for call in emitted.call_args_list] == [0, 1]
    assert "chained_paired_leaf_tma" in after


@pytest.mark.parametrize(
    "case",
    (
        "missing",
        "lines",
        "max_columns",
        "max_columns_bool",
        "axes",
        "boundaries",
        "scans",
        "schedule",
        "revision",
        "codegen",
        "plan",
        "selection",
        "graph_dtype",
        "graph_kwargs",
        "seed_node",
        "config",
        "readiness",
        "early_release",
        "next_stage",
        "stage_one",
        "geometry",
        "terminal",
    ),
)
def test_initialized_result_rejects_stale_provenance_before_stage_emission(case):
    emit = stages.emit_stage
    checked = []

    def inspect(*args, **kwargs):
        if args[3] != 0:
            return emit(*args, **kwargs)
        action = kwargs["root_actions"]
        sequence = action.sequence
        selected = sequence.initialized
        assert selected is not None
        cg, plan, boundaries = args[:3]
        node = plan.initialized_accumulator.seed
        config = cg.device_function.config.config
        old_config, old_boundaries = dict(config), dict(boundaries)
        old_meta, old_kwargs = node.meta["val"], node.kwargs
        old_readiness, old_release = sequence.input_readiness, sequence.early_release
        before = (
            sequence.next_stage,
            list(sequence.staged),
            dict(boundaries),
            list(cg.device_function.body),
        )
        changed_args, changed_kwargs = list(args), dict(kwargs)
        try:
            if case == "missing":
                sequence.initialized = None
            elif case in (
                "lines",
                "max_columns",
                "max_columns_bool",
                "axes",
                "boundaries",
                "scans",
                "schedule",
                "revision",
                "codegen",
                "plan",
                "selection",
            ):
                changes = {
                    "lines": {"lines": ()},
                    "max_columns": {"max_columns": 32},
                    "max_columns_bool": {"max_columns": False},
                    "axes": {"axes": ((0, 0), (0, 0))},
                    "boundaries": {"boundaries": ((node, "unpublished"),)},
                    "scans": {"scans": (object(),)},
                    "schedule": {"schedule": ()},
                    "revision": {"revision": ()},
                    "codegen": {"codegen": object()},
                    "plan": {"plan": replace(plan)},
                    "selection": {"_selection": ()},
                }
                sequence.initialized = replace(selected, **changes[case])
            elif case == "graph_dtype":
                node.meta["val"] = old_meta.to(torch.float16)
            elif case == "graph_kwargs":
                node.kwargs = dict(node.kwargs) | {"unknown": True}
            elif case == "seed_node":
                changed_args[1] = replace(
                    plan,
                    initialized_accumulator=replace(
                        plan.initialized_accumulator, seed=plan.dots[0]
                    ),
                )
            elif case == "config":
                config["cute_chained_seed_tile_columns"] = 32
            elif case == "readiness":
                sequence.input_readiness = ()
            elif case == "early_release":
                sequence.early_release = not old_release
                changed_kwargs["pending_allocation"] = sequence.early_release
            elif case == "next_stage":
                sequence.next_stage = 1
            elif case == "stage_one":
                changed_args[3] = 1
                changed_kwargs["root_actions"] = sequence.action(1)
            elif case == "geometry":
                changed_args[4] = replace(args[4], logical=(128, 32, 128))
            elif case == "terminal":
                changed_kwargs["terminal_fragment"] = True
            with pytest.raises(
                chain._UnsupportedChain,
                match=(
                    "contraction dependency graph changed"
                    if case in ("graph_dtype", "graph_kwargs")
                    else "provenance or readiness"
                ),
            ):
                emit(*changed_args, **changed_kwargs)
            assert sequence.staged == before[1]
            assert boundaries == before[2]
            assert cg.device_function.body == before[3]
        finally:
            sequence.initialized = selected
            sequence.next_stage = before[0]
            sequence.input_readiness = old_readiness
            sequence.early_release = old_release
            node.meta["val"], node.kwargs = old_meta, old_kwargs
            config.clear()
            config.update(old_config)
            boundaries.clear()
            boundaries.update(old_boundaries)
        checked.append(case)
        return emit(*args, **kwargs)

    with patch.object(stages, "emit_stage", side_effect=inspect):
        _source()
    assert checked == [case]


def test_failed_late_result_receipt_never_installs_partial_body():
    capture = actions.capture_initialized_result

    def corrupt(*args, **kwargs):
        return replace(capture(*args, **kwargs), lines=())

    with (
        patch.object(actions, "capture_initialized_result", side_effect=corrupt),
        patch.object(
            chain,
            "_install_chained_body",
            side_effect=AssertionError("partial install"),
        ),
        pytest.raises(
            exc.BackendUnsupported, match="accepted root inputs or result changed"
        ),
    ):
        _source()


def _serial_source(*, schedule="serial64", **options):
    return _source(
        cute_chained_k_schedule=schedule,
        cute_chained_pointwise_vectorize=True,
        cute_chained_pointwise_read_cache=True,
        cute_chained_pointwise_unroll=8,
        **options,
    )


@pytest.mark.parametrize("columns", (0, 32))
@pytest.mark.parametrize("schedule", ("full", "serial64", "overlap64"))
def test_seeded_stage_exact_issue_modes_and_last_read_tail(columns, schedule):
    options = {
        "columns": columns,
        "cute_chained_k_schedule": schedule,
        "cute_chained_pointwise_vectorize": True,
        "cute_chained_pointwise_read_cache": True,
        "cute_chained_pointwise_unroll": 8,
        "cute_chained_tmem_free": "last_read",
    }
    with patch.object(roots, "codegen_plain_root", return_value=False):
        before = _source(**options)
    with patch.object(stages, "emit_stage", wraps=stages.emit_stage) as emitted:
        after = _source(**options)
    after = assert_prepared_root_equivalent(before, after)
    assert [call.args[3] for call in emitted.call_args_list] == [0, 1]
    action = emitted.call_args.kwargs["root_actions"]
    assert action.seeded_accumulator
    assert action.tmem_input is None
    assert action.accumulator == "chain_tptr + 0"
    assert action.sequence.next_stage == 2
    assert action.sequence.seed_completion.result is action.sequence.initialized
    assert action.sequence.queued_rhs.seed is action.sequence.seed_completion
    end = "cute.copy(chain_1_copy, chain_1_source, chain_1_values)\n    cute.arch.fence_view_async_tmem_load()\n    cute.arch.sync_threads()\n    chain_allocator.free(chain_tptr)"
    assert end in after
    terminal = after[after.index("chain_1_mma =") :]
    assert "chain_1_mma.set(tcgen05.Field.ACCUMULATE, False)" not in terminal
    assert "chain_seed_copy" not in terminal
    assert "chain_1_seed" not in terminal
    if schedule != "full":
        prehalf = terminal[: terminal.index("for chain_k_half")]
        assert "cp_async_wait_group" not in prehalf
        assert "cute.arch.sync_threads()" not in prehalf
        assert "chain_1_kk = chain_k_half * 4 + chain_1_local_kk" in terminal
        if schedule == "serial64":
            assert "mbarrier_wait(chain_bars + 1, chain_k_half)" in terminal
        else:
            assert "mbarrier_wait(chain_bars + 1, chain_k_half)" not in terminal
            assert terminal.count("mbarrier_wait(chain_bars + 1, 0)") == 1
            assert terminal.count("tcgen05.commit(chain_bars + 1)") == 1


@pytest.mark.parametrize(
    "case",
    (
        "seed_missing",
        "seed_wrong",
        "rhs_not_enqueued",
        "rhs_other_seed",
        "rhs_other_input",
        "rhs_duplicate",
        "inputs_lines",
        "inputs_node",
        "inputs_axis",
        "inputs_arena",
        "half_overlap",
        "half_width",
        "half_order",
        "other_half_mode",
        "seeded_missing",
        "early_half",
        "wrong_offset",
        "wrong_phase",
        "premature_snapshot",
    ),
)
@pytest.mark.parametrize("schedule_mode", ("serial64", "overlap64"))
def test_seeded_stage_rejects_changed_phase_and_input_authority(case, schedule_mode):
    emit = stages.emit_stage
    checked = []

    def inspect(*args, **kwargs):
        if args[3] != 1:
            return emit(*args, **kwargs)
        sequence = kwargs["root_actions"].sequence
        selected = sequence.seeded_inputs
        assert selected is not None and selected.k_schedule is not None
        seed, rhs = sequence.seed_completion, sequence.queued_rhs
        half, prior_stage = sequence.half_producer, sequence.next_stage
        before = (
            list(sequence.staged),
            dict(args[2]),
            list(args[0].device_function.body),
        )
        changed_args, changed_kwargs = list(args), dict(kwargs)
        try:
            if case == "seed_missing":
                sequence.seed_completion = None
            elif case == "seed_wrong":
                sequence.seed_completion = actions.RootSeedCompletion(
                    replace(sequence.initialized)
                )
            elif case == "rhs_not_enqueued":
                sequence.queued_rhs = None
            elif case == "rhs_other_seed":
                sequence.queued_rhs = replace(rhs, seed=replace(seed))
            elif case == "rhs_other_input":
                sequence.queued_rhs = replace(rhs, inputs=replace(selected))
            elif case == "rhs_duplicate":
                with pytest.raises(chain._UnsupportedChain, match="enqueue order"):
                    sequence.enqueue_rhs(list(selected.lines))
            elif case in ("inputs_lines", "inputs_node", "inputs_axis", "inputs_arena"):
                changes = {
                    "inputs_lines": {"lines": ()},
                    "inputs_node": {"rhs": args[1].dots[0].args[1]},
                    "inputs_axis": {"inner_axis": 1 - selected.inner_axis},
                    "inputs_arena": {
                        "arena": replace(
                            selected.arena, b_bytes=selected.arena.b_bytes // 2
                        )
                    },
                }
                sequence.seeded_inputs = replace(selected, **changes[case])
            elif case in (
                "half_overlap",
                "half_width",
                "half_order",
                "other_half_mode",
            ):
                schedule = selected.k_schedule
                changes = {
                    "half_overlap": {"byte_spans": ((0, 16384), (8192, 24576))},
                    "half_width": {"half_width": 32},
                    "half_order": {"byte_spans": schedule.byte_spans[::-1]},
                    "other_half_mode": {
                        "mode": "overlap64"
                        if schedule_mode == "serial64"
                        else "serial64"
                    },
                }
                sequence.seeded_inputs = replace(
                    selected, k_schedule=replace(schedule, **changes[case])
                )
            elif case == "seeded_missing":
                sequence.seeded_inputs = None
            elif case == "early_half":
                sequence.half_producer = object()
            elif case == "wrong_offset":
                changed_kwargs["tmem_accumulator"] = "chain_tptr + 64"
            elif case == "wrong_phase":
                changed_args[5] = "1"
            elif case == "premature_snapshot":
                sequence.next_stage = 0
            if case != "rhs_duplicate":
                with pytest.raises(chain._UnsupportedChain):
                    emit(*changed_args, **changed_kwargs)
            assert (sequence.staged, args[2], args[0].device_function.body) == before
        finally:
            sequence.seeded_inputs = selected
            sequence.seed_completion = seed
            sequence.queued_rhs = rhs
            sequence.half_producer = half
            sequence.next_stage = prior_stage
        checked.append(case)
        return emit(*args, **kwargs)

    with patch.object(stages, "emit_stage", side_effect=inspect):
        _serial_source(schedule=schedule_mode)
    assert checked == [case]


@pytest.mark.parametrize("case", ("missing", "wrong_node", "lines"))
@pytest.mark.parametrize("schedule_mode", ("serial64", "overlap64"))
def test_half_producer_requires_the_original_current_action(case, schedule_mode):
    original = actions.RootStageAction.half_lines
    checked = []

    def inspect(action):
        producer = action.sequence.half_producer
        assert producer is not None
        try:
            if case == "missing":
                action.sequence.half_producer = None
            elif case == "wrong_node":
                action.sequence.half_producer = replace(
                    producer, operand=action.sequence.plan.dots[1].args[1]
                )
            else:
                action.sequence.half_producer = replace(
                    producer, lines=("wrong_phase",)
                )
            with pytest.raises(chain._UnsupportedChain, match="half producer"):
                original(action)
        finally:
            action.sequence.half_producer = producer
        checked.append(case)
        return original(action)

    with patch.object(actions.RootStageAction, "half_lines", inspect):
        _serial_source(schedule=schedule_mode)
    # Read the original producer, then recheck it against the bound continuation
    # before issuing either half. Both checks must reject the same mutation.
    assert checked == [case, case]


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("schedule", ("full", "serial64", "overlap64"))
def test_seed64_resident_terminal_stage_preserves_original_program(dtype, schedule):
    values = _late_rhs_args(dtype=dtype, n=128)
    config = _late_rhs_config(
        cute_chained_k_schedule=schedule,
        cute_chained_seed_tile_columns=64,
        cute_chained_pointwise_vectorize=True,
        cute_chained_pointwise_read_cache=True,
        cute_chained_pointwise_unroll=8,
    )
    with _cpu(), patch.object(roots, "codegen_plain_root", return_value=False):
        before = _late_rhs_pair._bind_isolated(values).to_code(config)
    with _cpu(), patch.object(stages, "emit_stage", wraps=stages.emit_stage) as emitted:
        after = _late_rhs_pair._bind_isolated(values).to_code(config)
    after = assert_prepared_root_equivalent(before, after)
    assert [call.args[3] for call in emitted.call_args_list] == [0, 1]
    assert "chain_0_values =" not in after
    assert "chain_1_values =" in after
