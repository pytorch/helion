from __future__ import annotations

import ast
from dataclasses import replace
from typing import Any
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_completed_store import _source
from helion._compiler.cute import chained_completed_store as completed
from helion._compiler.cute import chained_pipeline_storage as storage
from helion._compiler.cute.chained_matmul import _UnsupportedChain
from helion._compiler.cute.chained_store_expression import UnboundStoreTarget


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("output_dtype", (torch.bfloat16, torch.float16, torch.float32))
def test_original_point_is_prepared_once_before_charge(dtype, output_dtype):
    lower = completed.lower_store_point
    finalizer = storage.finalize_pipeline_storage
    lowered = []
    charged = []

    def point(*args, **kwargs):
        result = lower(*args, **kwargs)
        lowered.append((args[2], result))
        return result

    def finalize(*args, **kwargs):
        contract = kwargs["completed_store"]
        assert any(
            store is contract.candidate.store and value is contract.prepared.point
            for store, value in lowered
        )
        result = finalizer(*args, **kwargs)
        assert result is not None and contract.matches_storage(result)
        assert contract.c_name not in {
            region.name for region in result.recurrence.layout.regions
        }
        assert contract.candidate.node not in dict(result.recurrence.bindings)
        charged.append(result)
        return result

    with (
        patch.object(completed, "lower_store_point", point),
        patch.object(storage, "finalize_pipeline_storage", finalize),
    ):
        source, actions = _source(dtype, output_dtype, enabled=True)
    (action,), (allocation,) = actions, charged
    contract = allocation.completed_store
    assert contract is not None
    assert sum(store is action.candidate.store for store, _ in lowered) == 1
    assert action._progress.prepared is not None
    assert action._progress.prepared.point == replace(
        contract.prepared.point, target=action._progress.prepared.point.target
    )
    assert (
        contract.prepared.lines is None and action._progress.prepared.lines is not None
    )
    assert action.c_name not in {
        node.id for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Name)
    }
    assert len(contract.prepared.reads) == 2  # Original gathered index and mask.
    assert action.candidate.node in allocation.omitted_results
    action.validate_consumed()


def test_disabled_branch_never_prepares_an_allocation_contract():
    with patch.object(
        completed,
        "prepare_completed_store_plan",
        side_effect=AssertionError("disabled"),
    ):
        source, actions = _source(enabled=False)
    assert not actions and "chain_store_0_register" not in source


def test_exact_quota_and_one_byte_short_precede_any_body_consumption():
    _, (reference,) = _source(enabled=True)
    quota = reference.storage.charged_bytes
    original = storage.finalize_pipeline_storage
    calls = []

    def finalize(*args, **kwargs: Any):
        contract = kwargs["completed_store"]
        assert contract.matches()
        options: dict[str, Any] = dict(kwargs)
        options["capacity_bytes"] = quota - 1
        assert original(*args, **options) is None
        assert contract.preparation is None  # This generic fixture has no compact body.
        options["capacity_bytes"] = quota
        result = original(*args, **options)
        assert result is not None and result.charged_bytes == quota
        calls.append(result)
        return result

    with patch.object(storage, "finalize_pipeline_storage", finalize):
        _, (action,) = _source(enabled=True)
    assert calls == [action.storage]
    action.validate_consumed()


@pytest.mark.parametrize(
    "change",
    (
        "point",
        "metadata",
        "request",
        "native",
        "store",
        "extra_use",
        "alias_container",
        "alias_receipt",
    ),
)
def test_changed_precharge_contract_rejects_before_finalization(change):
    original = storage.finalize_pipeline_storage
    seen = []

    def finalize(*args, **kwargs):
        contract = kwargs["completed_store"]
        assert contract.matches()
        if change == "point":
            object.__setattr__(contract.prepared.point, "bounds", ())
        elif change == "metadata":
            node, name = contract.prepared.reads[0]
            region = contract.pipeline.frame.layout.region(name)
            object.__setattr__(region, "byte_size", 1)
        elif change == "request":
            object.__setattr__(
                contract.request, "live_until", contract.request.live_until + 1
            )
        elif change == "native":
            object.__setattr__(contract.candidate.native, "selected_slots", ())
        elif change == "store":
            contract.candidate.store.kwargs = {"changed_store": True}
        elif change == "alias_container":
            object.__setattr__(
                contract.plan, "tensor_aliases", dict(contract.plan.tensor_aliases)
            )
        elif change == "alias_receipt":
            object.__setattr__(contract, "_aliases", ())
        else:
            # An original different store is now an additional use of C.
            contract.candidate.node.users[contract.plan.loop.final_stores[0]] = None
        assert not contract.matches()
        assert original(*args, **kwargs) is None
        seen.append(contract)
        return None

    with (
        patch.object(storage, "finalize_pipeline_storage", finalize),
        patch.object(
            completed,
            "prepare_completed_store",
            side_effect=AssertionError("late bind forbidden"),
        ) as bind,
        pytest.raises(Exception, match="post-transport allocation"),
    ):
        _source(enabled=True)
    bind.assert_not_called()
    assert len(seen) == 1


def test_equivalent_but_foreign_final_contract_cannot_bind():
    original = completed.prepare_completed_store
    calls = []

    def prepare(plan, pipeline, allocation, *args, **kwargs):
        contract = allocation.completed_store
        assert contract is not None
        foreign = replace(contract)
        assert foreign is not contract and foreign == contract
        assert not foreign.matches_storage(allocation)
        calls.append(contract)
        return original(plan, pipeline, allocation, *args, **kwargs)

    with patch.object(completed, "prepare_completed_store", prepare):
        _, (action,) = _source(enabled=True)
    assert calls == [action.storage.completed_store]
    action.validate_consumed()


def test_late_metadata_failure_still_cannot_publish_or_complete():
    actions = []
    with (
        patch.object(
            completed.CompletedStoreAction, "_metadata_read", return_value=False
        ),
        pytest.raises(Exception, match="unproved metadata reads"),
    ):
        _source(enabled=True, observer=actions.append)
    (action,) = actions
    assert action._progress.token is action._tokens[1]
    assert action._progress.prepared is None and not action._progress.stage
    assert action.candidate.node not in dict(action.inputs)


def test_default_target_registration_never_uses_deferred_binding():
    with patch.object(
        UnboundStoreTarget, "bind", side_effect=AssertionError("default")
    ):
        _source(enabled=False)


def test_changed_target_identity_rejects_at_original_binding():
    observed = []
    original = completed.CompletedStoreAction.prepare_point

    def inspect(action, cg, plan, boundaries):
        contract = action.storage.completed_store
        assert contract is not None
        target = contract.prepared.point.target
        assert isinstance(target, UnboundStoreTarget)
        assert target.store is action.candidate.store
        assert target.output is action.candidate.store.args[0]
        old = target.output
        object.__setattr__(target, "output", action.candidate.node)
        try:
            with pytest.raises(_UnsupportedChain, match="target binding changed"):
                target.bind(cg, plan, action.candidate.store, dict(boundaries))
        finally:
            object.__setattr__(target, "output", old)
        assert action.matches(action.plan)
        observed.append(action)
        return original(action, cg, plan, boundaries)

    with patch.object(completed.CompletedStoreAction, "prepare_point", inspect):
        _source(enabled=True)
    assert len(observed) == 1


def test_unresolved_target_cannot_reach_publication_or_consume():
    actions = []
    with (
        patch.object(UnboundStoreTarget, "bind", lambda target, *args: target),
        pytest.raises(Exception, match="target is not bound"),
    ):
        _source(enabled=True, observer=actions.append)
    (action,) = actions
    assert action._progress.token is action._tokens[1]
    assert action._progress.prepared is None and not action._progress.stage


@pytest.mark.parametrize("change", ("alias", "alias_container", "argument_order"))
def test_other_early_host_alias_registration_rejects_before_charge(change):
    original = completed.lower_store_point

    def lower(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get("defer_target"):
            if change == "alias":
                args[1].tensor_aliases["unexpected_host_input"] = "unexpected_alias"
            elif change == "alias_container":
                object.__setattr__(
                    args[1], "tensor_aliases", dict(args[1].tensor_aliases)
                )
            else:
                assert len(args[0].device_function.arguments) > 1
                args[0].device_function.arguments.reverse()
        return result

    with (
        patch.object(completed, "lower_store_point", lower),
        patch.object(
            storage,
            "finalize_pipeline_storage",
            side_effect=AssertionError("charge forbidden"),
        ) as finalize,
        pytest.raises(Exception, match="early completed store registered a host alias"),
    ):
        _source(enabled=True)
    finalize.assert_not_called()
