from __future__ import annotations

import ast
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from test.test_cute_chained_seeded_body import _code

from helion import exc
from helion._compiler.cute import chained_body_program as body
from helion._compiler.cute import prepared_state_body as state
from helion._compiler.cute.chained_tcgen05 import _load_result


class _OriginalInstructions(ast.NodeTransformer):
    """Closed inverse; all untouched source and every native argument retained."""

    def visit_ImportFrom(self, node):
        if node.module == "helion._compiler.cute" and [
            (item.name, item.asname) for item in node.names
        ] == [("prepared_tcgen_edge", None)]:
            return None
        return node

    def visit_Expr(self, node):
        call = node.value
        if not isinstance(call, ast.Call):
            return self.generic_visit(node)
        name = ast.unparse(call.func)
        if name == "prepared_tcgen_edge.execute_prepared_read":
            assert not call.keywords and len(call.args) == 9
            source, values, copy, *controls = call.args
            assert [ast.literal_eval(item) for item in controls] == [
                None,
                None,
                None,
                False,
                None,
                0,
            ]
            return ast.parse(
                f"cute.copy({ast.unparse(copy)}, {ast.unparse(source)}, {ast.unparse(values)})\n"
                "cute.arch.fence_view_async_tmem_load()"
            ).body
        if name == "prepared_tcgen_edge.execute_prepared_store":
            assert not call.keywords and len(call.args) == 5
            values, target, copy, descriptor, shape = call.args
            assert ast.literal_eval(descriptor) is False
            assert ast.literal_eval(shape) is None
            return ast.parse(
                f"cute.copy({ast.unparse(copy)}, {ast.unparse(values)}, {ast.unparse(target)})"
            ).body[0]
        if name == "prepared_tcgen_edge.execute_prepared_store_completion":
            assert not call.keywords and len(call.args) == 1
            assert ast.literal_eval(call.args[0]) is False
            return ast.parse("cute.arch.fence_view_async_tmem_store()").body[0]
        return self.generic_visit(node)


def _selected(mode, dtype, columns):
    planner = body.plan_root_action_body

    def select(*args, **kwargs):
        return planner(*args, **kwargs, prepared_continuation=True)

    with patch.object(body, "plan_root_action_body", select):
        return _code(mode, dtype, columns)


@pytest.mark.parametrize("mode", ("local", "serial64", "overlap64"))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("columns", (0, 32))
def test_original_initialized_state_runs_shared_effects(mode, dtype, columns):
    def original_result(owner, action, result, geometry, execution):
        read = (
            []
            if result.max_columns
            else _load_result("chain_0", owner.plan.shapes[0][:2], execution=execution)
        )
        return [*read, *result.lines]

    with patch.object(state, "emit_initialized_state", original_result):
        before = _selected(mode, dtype, columns)
    observed = []
    original = state.RootStateBody.accept

    def accept(current, lines):
        original(current, lines)
        effects = current.program.actions
        assert [a.kind for a in effects] == ["read", "transform", "store"] * (
            (len(effects) - 1) // 3
        ) + ["complete"]
        assert len(effects) == (
            4 if columns == 0 else 1 + 3 * current.owner.plan.shapes[0][1] // columns
        )
        assert effects[-1].binding is effects[-2].binding
        assert all(
            a.source is current.owner.plan.initialized_accumulator.first
            for a in effects
        )
        assert current.owner.pending is current.action
        assert not current.owner.consumed
        observed.append(tuple(effects))

    with patch.object(state.RootStateBody, "accept", accept):
        after = _selected(mode, dtype, columns)
    assert len(observed) == 1
    # Complete source equality after the closed inverse of the three shared
    # transport instructions; arithmetic, coordinates and ordering stay exact.
    assert "prepared_tcgen_edge.execute_prepared_read(" in after
    assert "prepared_tcgen_edge.execute_prepared_store(" in after
    assert "prepared_tcgen_edge.execute_prepared_store_completion(False)" in after
    assert ast.dump(_OriginalInstructions().visit(ast.parse(after))) == ast.dump(
        ast.parse(before)
    )


@pytest.mark.parametrize(
    "mutation", ("drop", "copy", "reverse", "line", "geometry", "returned", "accepted")
)
def test_actual_state_program_mutations_reject(mutation):
    original = state.RootStateBody.emit
    touched = []

    def emit(current, effect):
        if not touched:
            touched.append(True)
            if mutation == "geometry":
                current.geometry = replace(current.geometry, transpose=True)
            elif mutation == "line":
                object.__setattr__(
                    effect.binding.point, "value", "unemitted_state_value"
                )
            elif mutation not in ("returned", "accepted"):
                actions = current.program.actions
                object.__setattr__(
                    current.program,
                    "actions",
                    {
                        "drop": actions[1:],
                        "copy": (replace(actions[0]), *actions[1:]),
                        "reverse": tuple(reversed(actions)),
                    }[mutation],
                )
        return original(current, effect)

    original_interpret = body.emit_body_program
    original_accept = state.RootStateBody.accept

    def accept(current, lines):
        original_accept(current, lines)
        if mutation == "accepted":
            lines.append("foreign_write")

    def interpret(*args, **kwargs):
        lines = original_interpret(*args, **kwargs)
        if mutation == "returned" and kwargs.get("state_body") is not None:
            lines.append("foreign_write")
        return lines

    with (
        patch.object(state.RootStateBody, "emit", emit),
        patch.object(body, "emit_body_program", interpret),
        patch.object(state.RootStateBody, "accept", accept),
        pytest.raises(exc.BackendUnsupported),
    ):
        _selected("local", torch.bfloat16, 32)
    assert touched == [True]
