from __future__ import annotations

from dataclasses import fields
from dataclasses import replace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import sympy

from test.test_cute_chunk_recurrence import _code

from helion._compiler.cute import chunk_recurrence
from helion._compiler.cute.chained_matmul import _UnsupportedChain
from helion._compiler.cute.fx_matcher import _TensorRef
from helion._compiler.cute.prepared_tcgen_binding import PreparedProjectionHost

if TYPE_CHECKING:
    from collections.abc import Callable

_BINDINGS = ("prepared_projection", "prepared_continuation", "prepared_epoch")
_SEMANTIC = tuple(
    item.name
    for item in fields(chunk_recurrence.CuteChunkRecurrencePlan)
    if item.name not in _BINDINGS
)


class _Captured(BaseException):
    pass


def _with_host(
    check: Callable[[PreparedProjectionHost], None], mode: str = "prepared_epoch"
) -> None:
    planner = chunk_recurrence._plan_chunk_recurrence
    captured = []

    def select(*args, **kwargs):
        plan = planner(*args, **kwargs, **{mode: True})
        assert plan is not None and plan.prepared_projection is not None
        host = PreparedProjectionHost(plan)
        check(host)
        host.check()
        captured.append(host)
        raise _Captured

    with (
        patch.object(chunk_recurrence, "_plan_chunk_recurrence", select),
        patch("cutlass.cute.compile", side_effect=AssertionError("native forbidden")),
        pytest.raises(_Captured),
    ):
        _code(fp32_state=True, dv_partitions=2, pipeline="wide")
    assert len(captured) == 1


@pytest.mark.parametrize("name", _SEMANTIC)
def test_every_original_semantic_and_physical_field_is_checked(name):
    def check(host):
        original = getattr(host.plan, name)
        if isinstance(original, _TensorRef):
            changed = replace(original, name=original.name + "_changed")
        elif isinstance(original, sympy.Expr):
            changed = sympy.Add(original, 1)
        elif isinstance(original, int):
            changed = original + 1
        else:
            assert isinstance(original, str)
            changed = original + "_changed"
        object.__setattr__(host.plan, name, changed)
        try:
            with pytest.raises(_UnsupportedChain):
                host.check()
        finally:
            object.__setattr__(host.plan, name, original)

    _with_host(check)


@pytest.mark.parametrize("name", _BINDINGS)
@pytest.mark.parametrize("change", ("clear", "replace"))
def test_exact_prepared_binding_cannot_be_reset(name, change):
    def check(host):
        original = getattr(host.plan, name)
        assert original is not None
        changed = None if change == "clear" else replace(original)
        object.__setattr__(host.plan, name, changed)
        try:
            with pytest.raises(_UnsupportedChain, match="host binding changed"):
                host.check()
        finally:
            object.__setattr__(host.plan, name, original)

    _with_host(check)


@pytest.mark.parametrize(
    "name", ("projection", "continuation", "graph", "epoch", "segment")
)
def test_original_captured_revision_identity_cannot_be_reset(name):
    def check(host):
        plan = host.plan
        projection, continuation, epoch = (
            plan.prepared_projection,
            plan.prepared_continuation,
            plan.prepared_epoch,
        )
        assert projection is not None and continuation is not None and epoch is not None
        target, field = {
            "projection": (projection, "revision"),
            "continuation": (continuation, "_revision"),
            "graph": (continuation.graph, "_revision"),
            "epoch": (epoch, "_identity"),
            "segment": (epoch.projected, "_revision"),
        }[name]
        original = getattr(target, field)
        assert isinstance(original, tuple)
        replacement = tuple(iter(original))
        assert replacement is not original
        object.__setattr__(target, field, replacement)
        try:
            with pytest.raises(_UnsupportedChain, match="host binding changed"):
                host.check()
        finally:
            object.__setattr__(target, field, original)

    _with_host(check)


@pytest.mark.parametrize(
    "mode", ("prepared_edge", "prepared_continuation", "prepared_epoch")
)
def test_private_modes_keep_present_and_absent_identity_authority(mode):
    def check(host):
        assert (host.plan.prepared_continuation is not None) is (
            mode != "prepared_edge"
        )
        assert (host.plan.prepared_epoch is not None) is (mode == "prepared_epoch")
        assert host.host_keywords() == {"PREPARED_EDGE": True}
        context = host._context
        host.check()
        assert host._context is context
        original = host.plan
        host.plan = replace(original)
        try:
            with pytest.raises(_UnsupportedChain, match="host binding changed"):
                host.check()
        finally:
            host.plan = original

    _with_host(check, mode)


@pytest.mark.parametrize(
    "component", ("projection", "continuation", "segment", "epoch")
)
def test_original_component_checks_still_execute(component):
    def check(host):
        plan = host.plan
        assert plan.prepared_epoch is not None
        target = {
            "projection": plan.prepared_projection,
            "continuation": plan.prepared_continuation,
            "segment": plan.prepared_epoch.projected,
            "epoch": plan.prepared_epoch,
        }[component]
        assert target is not None
        with (
            patch.object(
                type(target),
                "check",
                side_effect=_UnsupportedChain("component sentinel"),
            ) as checked,
            pytest.raises(_UnsupportedChain, match="component sentinel"),
        ):
            host.check()
        checked.assert_called_once()

    _with_host(check)


@pytest.mark.parametrize("change", ("graph", "segment", "epoch-join"))
def test_original_graph_segment_and_epoch_join_changes_reject(change):
    def check(host):
        epoch = host.plan.prepared_epoch
        assert epoch is not None
        if change == "graph":
            target, name = epoch.projection.source, "args"
            original = target.args
            changed = (*original[:2], epoch.projection.state, *original[3:])
        elif change == "segment":
            target, name = epoch.projected.segments[0], "end"
            original = target.end
            changed = original + 1
        else:
            target, name = epoch, "projection"
            original = epoch.projection
            changed = replace(original)
        object.__setattr__(target, name, changed)
        try:
            with pytest.raises(_UnsupportedChain):
                host.check()
        finally:
            object.__setattr__(target, name, original)

    _with_host(check)
