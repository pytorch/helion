"""Conditional page emission must not read recurrence carries as invariants."""

from __future__ import annotations

import ast
from collections import Counter
import inspect
import json
import textwrap

import pytest

import helion
from helion._compiler.cute.analysis.conditional_attention_program import (
    prove_conditional_attention_program,
)
from helion._compiler.cute.analysis.conditional_attention_recurrence import (
    match_conditional_attention_recurrence,
)
from helion._compiler.cute.conditional_attention_codegen import (
    ConditionalAttentionCompiler,
)
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_conditional_attention import conditional_attention_args
from helion._testing.cute_conditional_attention import (
    conditional_online_attention_prefix,
)
from helion._testing.cute_source import function_from_ast
from helion.language import memory_ops

pytestmark = pytest.mark.cpu_only


def _variant(kind):
    original = conditional_online_attention_prefix.fn
    function = ast.parse(textwrap.dedent(inspect.getsource(original))).body[0]
    function.decorator_list = []
    function.name = f"conditional_carry_{kind}"
    loops = [
        n
        for n in ast.walk(function)
        if isinstance(n, ast.For)
        and isinstance(n.target, ast.Name)
        and n.target.id == "iteration"
    ]
    assert len(loops) == 1
    loop = loops[0]
    if kind in ("visible", "scores"):
        node = next(
            n
            for n in loop.body
            if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == "visible"
        )
        node.value = ast.BinOp(
            left=node.value,
            op=ast.BitAnd(),
            right=ast.parse(
                "maximum[:, None] < 1.0" if kind == "visible" else "scores > 0.0",
                mode="eval",
            ).body,
        )
    elif kind == "panel":
        node = next(
            n
            for n in loop.body
            if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == "shift"
        )
        node.value = ast.BinOp(
            left=node.value,
            op=ast.Add(),
            right=ast.parse(
                "(torch.amax(maximum) > 1.0).to(torch.int32)", mode="eval"
            ).body,
        )
    elif kind == "query":
        for parent in ast.walk(function):
            if isinstance(parent, (ast.If, ast.For)) and loop in parent.body:
                query = next(
                    n
                    for n in parent.body
                    if isinstance(n, ast.Assign)
                    and isinstance(n.targets[0], ast.Name)
                    and n.targets[0].id == "q"
                )
                parent.body.remove(query)
                query.value = ast.parse(
                    "query_storage[query_address + (maximum[:, None] > 1.0).to(torch.int32)]",
                    mode="eval",
                ).body
                loop.body.insert(0, query)
                break
        else:
            raise AssertionError("query parent")
    else:
        assert kind in ("captured_head", "branch_head")
        for parent in ast.walk(function):
            if isinstance(parent, (ast.If, ast.For)) and loop in parent.body:
                parent.body.insert(
                    parent.body.index(loop),
                    ast.parse(
                        "panel_head = lengths[request] % kv_heads"
                        if kind == "captured_head"
                        else "if length > 0:\n"
                        "    panel_head = lengths[request] % kv_heads\n"
                        "else:\n"
                        "    panel_head = (lengths[request] + 1) % kv_heads"
                    ).body[0],
                )
                break
        else:
            raise AssertionError("loop parent")
        for node in ast.walk(loop):
            if isinstance(node, ast.Name) and node.id == "kv_head":
                node.id = "panel_head"
    fn = function_from_ast(
        ast.fix_missing_locations(function),
        original.__globals__,
        f"<conditional-carry-{kind}>",
        copy_namespace=False,
    )
    return helion.kernel(fn, backend="cute", static_shapes=True, fast_math=True)


def test_unchanged_conditional_source(tmp_path, record_property):
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(
            conditional_online_attention_prefix, conditional_attention_args()
        )
        source = bound.to_code(bound.config_spec.default_config())
        assert "_conditional_attention_page(" in source
        (tmp_path / "source.py").write_text(source)
        record_property("source", str(tmp_path / "source.py"))


@pytest.mark.parametrize("kind", ["visible", "panel", "query", "scores"])
def test_conditional_carry_dependency_refuses_specialization(kind):
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(_variant(kind), conditional_attention_args())
        ir = bound.host_function.device_ir
        with bound.env, bound.host_function:
            recurrence = match_conditional_attention_recurrence(ir, bound.env)
            assert recurrence is not None
            program = prove_conditional_attention_program(ir, bound.env)
        assert program is None, f"{kind} depends on physical attention state"


@pytest.mark.parametrize("kind", [None, "captured_head"])
def test_captured_scalar_loads_keep_their_original_region(
    kind, monkeypatch, tmp_path, record_property
):
    reads = {}
    original = ConditionalAttentionCompiler.node

    def observe(self, node, values, *args):
        if (
            node.target is memory_ops.load
            and node.graph is not self.program.recurrence.loop.graph
        ):
            reads.setdefault(self, Counter())[node] += 1
        return original(self, node, values, *args)

    monkeypatch.setattr(ConditionalAttentionCompiler, "node", observe)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(
            conditional_online_attention_prefix if kind is None else _variant(kind),
            conditional_attention_args(),
        )
        source = bound.to_code(bound.config_spec.default_config())
    assert "_conditional_attention_stage_packed(" in source
    assert reads
    (tmp_path / "source.py").write_text(source)
    (tmp_path / "loads.json").write_text(
        json.dumps(
            [
                {node.name: count for node, count in counts.items()}
                for counts in reads.values()
            ]
        )
    )
    record_property("observation", str(tmp_path))
    assert all(count == 1 for counts in reads.values() for count in counts.values()), (
        reads
    )


def test_branch_head_keeps_generic_lowering():
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(_variant("branch_head"), conditional_attention_args())
        with bound.env, bound.host_function:
            assert (
                prove_conditional_attention_program(
                    bound.host_function.device_ir, bound.env
                )
                is None
            )
