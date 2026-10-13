from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_pipeline import _one
from .test_cute_chained_pipeline import _startup_m64_config
from .test_cute_chained_pipeline import _startup_m64_values
from .test_cute_chained_pipeline import cpu_codegen
from helion import exc
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chained_root_stage as roots
from helion._compiler.cute import chained_tcgen05 as legacy
from helion._compiler.cute import chained_tcgen_stage as stages


def _source(dtype=torch.bfloat16, width=128, major="MM", *, vector=True, role=None):
    a, b = _startup_m64_values(dtype, width, major)
    if role == "a":
        b = torch.empty((128, width * 2), dtype=dtype)[:, ::2]
    elif role == "b":
        a = torch.empty((64, 256), dtype=dtype)[:, ::2]
    config = _startup_m64_config(
        width,
        direct=True,
        cute_chained_tmem_early_release=True,
    )
    if vector:
        config.config.update(
            cute_chained_pointwise_vectorize=True,
            cute_chained_pointwise_inplace_async=True,
            cute_chained_pointwise_unroll=2,
        )
    with cpu_codegen():
        return _one._bind_isolated((a, b)).to_code(config)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("width", (64, 128))
@pytest.mark.parametrize("major", ("KK", "KM", "MK", "MM"))
@pytest.mark.parametrize("vector", (False, True))
def test_startup_native_m64_shares_stage_with_exact_original_source(
    dtype, width, major, vector
):
    initialized = torch.cuda.is_initialized()
    with patch.object(roots, "supports_independent_root", return_value=False):
        before = _source(dtype, width, major, vector=vector)
    with (
        patch.object(
            legacy, "codegen_chained_tcgen05", side_effect=AssertionError("old root")
        ),
        patch.object(stages, "emit_stage", wraps=stages.emit_stage) as emitted,
    ):
        after = _source(dtype, width, major, vector=vector)
    assert after == before
    assert emitted.call_count == 1
    action = emitted.call_args.kwargs["root_actions"]
    assert action.sequence.startup is not None
    assert tuple(t.role for t in action.sequence.startup.transfers) == ("a", "b")
    assert action.sequence.startup_issued
    assert emitted.call_args.args[4].physical == (64, width, 128)
    assert after.count("mbarrier_wait(chain_start_bar, 0)") == 2
    assert after.index("tma_bar_ptr=chain_start_bar") < after.index("chain_0_mma =")
    assert after.index("mbarrier_wait(chain_start_bar, 0)") < after.index(
        "chain_0_b_raw_partition" if vector else "chain_start_b_index"
    )
    assert after.rindex("mbarrier_wait(chain_start_bar, 0)") < after.index(
        "chain_allocator.wait_for_alloc()"
    )
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize("role", ("a", "b"))
def test_partial_startup_selection_keeps_other_original_producer(role):
    with patch.object(roots, "supports_independent_root", return_value=False):
        before = _source(vector=False, role=role)
    with (
        patch.object(
            legacy, "codegen_chained_tcgen05", side_effect=AssertionError("old root")
        ),
        patch.object(stages, "emit_stage", wraps=stages.emit_stage) as emitted,
    ):
        after = _source(vector=False, role=role)
    assert after == before
    startup = emitted.call_args.kwargs["root_actions"].sequence.startup
    assert startup is not None
    assert tuple(t.role for t in startup.transfers) == (role,)


@pytest.mark.parametrize(
    "case",
    (
        "missing",
        "not_issued",
        "integer_issued",
        "duplicate_role",
        "operand",
        "leaf",
        "coordinates",
        "shape",
        "inner",
        "origin",
        "descriptor",
        "descriptor_arguments",
        "wrapper_registry",
        "wrapper_parameters",
        "inplace_option",
        "scan_readiness",
    ),
)
def test_startup_action_rejects_mutated_receipt_before_any_stage_lines(case):
    original = stages.emit_stage
    checked = []

    def inspect(*args, **kwargs):
        action = kwargs["root_actions"]
        sequence = action.sequence
        startup = sequence.startup
        assert startup is not None
        cg, plan, boundaries = args[:3]
        transfer = startup.transfers[-1]
        wrapper = dict(transfer.wrapper)
        old_registry = list(cg.cute_wrapper_plans)
        old_parameters = list(cg.device_function.wrapper_only_params)
        old_readiness = sequence.input_readiness
        old_config = dict(cg.device_function.config.config)
        before = (
            sequence.next_stage,
            list(sequence.staged),
            dict(boundaries),
            list(cg.device_function.body),
        )
        try:
            if case == "missing":
                sequence.startup = None
            elif case == "not_issued":
                sequence.startup_issued = False
            elif case == "integer_issued":
                sequence.startup_issued = 1
            elif case == "duplicate_role":
                sequence.startup = replace(startup, transfers=(transfer, transfer))
            elif case in ("operand", "leaf", "coordinates", "shape", "inner", "origin"):
                changes = {
                    "operand": {"operand": plan.dots[0].args[0]},
                    "leaf": {"leaf": plan.dots[0]},
                    "coordinates": {"coordinates": transfer.coordinates[::-1]},
                    "shape": {"shape": (32, 128)},
                    "inner": {"inner": 1 - transfer.inner},
                    "origin": {"row": f"({transfer.row}) + 1"},
                }
                sequence.startup = replace(
                    startup,
                    transfers=(
                        *startup.transfers[:-1],
                        replace(transfer, **changes[case]),
                    ),
                )
            elif case == "descriptor":
                transfer.wrapper["tile"] = (32, 128)
            elif case == "descriptor_arguments":
                transfer.wrapper["kernel_args"].append("unexpected_argument")
            elif case == "wrapper_registry":
                cg.cute_wrapper_plans.pop()
            elif case == "wrapper_parameters":
                cg.device_function.wrapper_only_params.reverse()
            elif case == "inplace_option":
                cg.device_function.config.config[
                    "cute_chained_pointwise_inplace_async"
                ] = False
            else:
                sequence.input_readiness = ()
            with pytest.raises(
                chain._UnsupportedChain, match="provenance or readiness"
            ):
                original(*args, **kwargs)
            assert (
                sequence.next_stage,
                sequence.staged,
                boundaries,
                cg.device_function.body,
            ) == before
        finally:
            sequence.startup = startup
            sequence.startup_issued = True
            sequence.input_readiness = old_readiness
            transfer.wrapper.clear()
            transfer.wrapper.update(wrapper)
            # The nested-list mutation needs its own restoration, not a shallow
            # copy that aliases the exact object exercised by the negative.
            if case == "descriptor_arguments":
                transfer.wrapper["kernel_args"].pop()
            cg.cute_wrapper_plans[:] = old_registry
            cg.device_function.wrapper_only_params[:] = old_parameters
            cg.device_function.config.config.clear()
            cg.device_function.config.config.update(old_config)
        checked.append(case)
        return original(*args, **kwargs)

    with patch.object(stages, "emit_stage", side_effect=inspect):
        _source()
    assert checked == [case]


@pytest.mark.parametrize("case", ("duplicate", "shape", "role", "axes"))
def test_startup_capture_rejects_a_changed_original_selection(case):
    original = roots.capture_root_startup

    def capture(cg, plan, transfers, axes):
        changed = list(transfers)
        if case == "duplicate":
            changed.append(changed[0])
        elif case == "shape":
            changed[-1] = replace(changed[-1], shape=(32, 128))
        elif case == "role":
            changed[-1] = replace(changed[-1], role="invalid")
        else:
            axes = ((1 - axes[0][0], axes[0][1]),)
        return original(cg, plan, changed, axes)

    with (
        patch.object(roots, "capture_root_startup", side_effect=capture),
        pytest.raises(exc.BackendUnsupported, match="startup input selection"),
    ):
        _source()


def test_rejected_late_startup_selection_cannot_fall_through_to_legacy_stages():
    with (
        patch.object(roots, "plan_root_stage_sequence", return_value=None),
        patch.object(stages, "emit_stage", side_effect=AssertionError("stage entered")),
        pytest.raises(exc.BackendUnsupported, match="changed before stage selection"),
    ):
        _source()
