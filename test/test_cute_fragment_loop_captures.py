from __future__ import annotations

import ast
import operator
from unittest.mock import patch

import pytest
import torch

from helion._compiler.cute.fragment_context import Fragment
from helion._compiler.cute.fragment_context import FragmentContext
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_fragment_loop_captures import loop_scans
from helion._testing.cute_fragment_loop_captures import loop_tensor_bound

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "escape,readonly_alias", [(False, False), (True, False), (False, True)]
)
def test_actual_loop_copies_retire_only_consumed_initials(escape, readonly_alias):
    observations = []
    original = FragmentContext.retired_loop_initials

    def inspect(context, node, values, copied, readonly):
        # Use the real context manager and inspect both sides of its boundary.
        class Scope:
            def __enter__(self):
                self.original_values = dict(values)
                self.before = context.live_buffers()
                self.manager = original(context, node, values, copied, readonly)
                self.manager.__enter__()
                observations.append(
                    (
                        self.before,
                        context.live_buffers(),
                        set(self.original_values) - set(values),
                    )
                )

            def __exit__(self, *args):
                result = self.manager.__exit__(*args)
                assert values == self.original_values
                return result

        return Scope()

    with (
        cpu_render_target(),
        patch.object(FragmentContext, "retired_loop_initials", inspect),
    ):
        bound = cpu_bind(loop_scans, (torch.ones(2, 17), 3, escape, readonly_alias))
        source = bound.to_code(bound.config_spec.default_config())
    assert len(observations) == 1
    before, after, removed = observations[0]
    if escape:
        # A dead `state` binding can retire while the escaping `initial`
        # binding keeps the same backing storage live throughout the loop.
        assert before == after
    elif readonly_alias:
        assert removed and before == after
    else:
        assert removed and len(before - after) == 1
    parsed = ast.parse(source)
    # Runtime loop and both result stores remain present. The copy preceding
    # the loop covers steps=0; the loop still handles arbitrary positive steps.
    assert any(
        isinstance(node, ast.For) and "steps" in ast.unparse(node.iter)
        for node in ast.walk(parsed)
    )
    assert ".store(" in source


def _scope_fixture(later_use=False, readonly_alias=False):
    graph = torch.fx.Graph()
    source = graph.placeholder("initial")
    call = graph.call_function(operator.neg, (source,))
    if later_use:
        graph.call_function(operator.add, (source, 1))
    value = Fragment((4,), torch.float32, lambda _: "initial[0]", True, (), "initial")
    alias = Fragment((4,), torch.float32, lambda _: "initial[0] + 7", True, (value,))
    context = object.__new__(FragmentContext)
    values = {source: value}
    context.scopes = [values]
    context.scope_uses = [{source: 2 if later_use else 1}]
    context.held = [
        Fragment((4,), torch.float32, lambda _: "carry[0]", True, (), "carry")
    ]
    return context, call, values, source, [alias] if readonly_alias else []


@pytest.mark.parametrize(
    "later_use,readonly_alias", [(False, False), (True, False), (False, True)]
)
def test_initial_scope_restores_after_zero_trip_or_lowering_failure(
    later_use, readonly_alias
):
    context, call, values, source, readonly = _scope_fixture(later_use, readonly_alias)
    initial = dict(values)
    for fail in (False, True):
        try:
            with context.retired_loop_initials(call, values, {source}, readonly):
                assert (source in values) == later_use
                assert ("initial" in context.live_buffers()) == (
                    later_use or readonly_alias
                )
                assert "carry" in context.live_buffers()
                if fail:
                    raise ValueError("lowering failure")
        except ValueError:
            assert fail
        assert values == initial and len(context.held) == 1


def test_loop_retirement_requires_its_active_parent_scope():
    context, call, values, source, readonly = _scope_fixture()
    context.scope_uses = []
    with (
        pytest.raises(AssertionError),
        context.retired_loop_initials(call, values, {source}, readonly),
    ):
        pytest.fail("an inconsistent internal scope must not be lowered")


def test_mutable_tensor_bound_stays_live_through_loop_masks():
    observations = []
    original = FragmentContext.retired_loop_initials

    def inspect(context, node, values, copied, readonly):
        class Scope:
            def __enter__(self):
                ranges = [
                    value
                    for args in node.args[1:3]
                    for value in args
                    if isinstance(value, torch.fx.Node)
                ]
                self.range_buffers = context.referenced_buffers(
                    [values[value] for value in ranges]
                )
                self.manager = original(context, node, values, copied, readonly)
                self.manager.__enter__()
                observations.append((self.range_buffers, context.live_buffers()))

            def __exit__(self, *args):
                return self.manager.__exit__(*args)

        return Scope()

    with (
        cpu_render_target(),
        patch.object(FragmentContext, "retired_loop_initials", inspect),
    ):
        bound = cpu_bind(
            loop_tensor_bound,
            (torch.ones(1, 17), torch.tensor([2, 3, 0, 0], dtype=torch.int64)),
        )
        source = bound.to_code(bound.config_spec.default_config())
    assert len(observations) == 1
    range_buffers, live_buffers = observations[0]
    assert range_buffers, "the loop bound must actually use resident storage"
    assert range_buffers <= live_buffers
    assert "fragment_tile" in source
