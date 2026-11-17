"""Emission admission for the bounded two-phase conditional attention gate."""

from __future__ import annotations

import ast

from .cute_codegen import stage_sources


def validate_page_lifetime(kernel: ast.FunctionDef) -> None:
    """Require one uniform nonempty guard around acquire, full loop, retire."""
    named = {
        name: [
            node
            for node in ast.walk(kernel)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == name
        ]
        for name in (
            "_conditional_attention_begin",
            "_conditional_attention_page",
            "_conditional_attention_end",
        )
    }
    assert all(len(calls) == 1 for calls in named.values())
    begin, page, end = (calls[0] for calls in named.values())
    owners = [
        node
        for node in ast.walk(kernel)
        if isinstance(node, ast.If)
        and len(node.body) == 3
        and isinstance(node.body[0], ast.Assign)
        and node.body[0].value is begin
    ]
    assert len(owners) == 1
    owner = owners[0]
    assign, loop, retire = owner.body
    assert not owner.orelse
    assert isinstance(assign, ast.Assign) and len(assign.targets) == 1
    assert isinstance(assign.targets[0], ast.Tuple)
    handles = assign.targets[0].elts
    assert len(handles) == 2 and all(isinstance(x, ast.Name) for x in handles)
    assert ast.unparse(handles[0]) != ast.unparse(handles[1])
    assert isinstance(loop, ast.For) and not loop.orelse
    assert isinstance(loop.iter, ast.Call) and ast.unparse(loop.iter.func) == "range"
    assert len(loop.iter.args) == 3 and ast.literal_eval(loop.iter.args[2]) == 1
    assert isinstance(owner.test, ast.Compare) and len(owner.test.ops) == 1
    assert isinstance(owner.test.ops[0], ast.Lt)
    assert ast.unparse(owner.test.left) == ast.unparse(loop.iter.args[0])
    assert ast.unparse(owner.test.comparators[0]) == ast.unparse(loop.iter.args[1])
    assert page in list(ast.walk(loop))
    assert not any(
        isinstance(node, (ast.Break, ast.Continue, ast.Return))
        for node in ast.walk(loop)
    )
    assert isinstance(retire, ast.Expr) and retire.value is end
    assert len(begin.args) == 3 and len(end.args) == 5 and len(page.args) == 22
    assert [ast.unparse(x) for x in page.args[-2:]] == [ast.unparse(x) for x in handles]
    assert [ast.unparse(x) for x in end.args[:2]] == [ast.unparse(x) for x in handles]
    assert ast.unparse(begin.args[0]) == ast.unparse(page.args[8])
    assert ast.unparse(begin.args[0]) == ast.unparse(end.args[2])
    assert ast.unparse(begin.args[1]) == ast.unparse(page.args[10])
    assert ast.unparse(begin.args[1]) == ast.unparse(end.args[3])
    assert ast.unparse(begin.args[2]) == ast.unparse(page.args[14])
    assert ast.unparse(begin.args[2]) == ast.unparse(end.args[4])


def validate_conditional_source(source: str) -> None:
    """Require both dedicated phases and the continuation seed in the actual module."""
    stages = stage_sources(source)
    assert len(stages) == 2
    for index, stage in enumerate(stages):
        tree = ast.parse(stage)
        kernels = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and any(
                ast.unparse(value) == "cute.kernel" for value in node.decorator_list
            )
        ]
        assert len(kernels) == 1
        validate_page_lifetime(kernels[0])
        assert (
            kernels[0].name == f"_helion_written_conditional_attention__region_{index}"
        )
        launches = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_launcher"
        ]
        assert len(launches) == 1
        launch = launches[0]
        assert ast.unparse(launch.args[0]) == kernels[0].name
        assert ast.literal_eval(launch.args[1]) == (6,)
        assert len(launch.keywords) == 1
        assert launch.keywords[0].arg == "block"
        assert ast.literal_eval(launch.keywords[0].value) == (128, 1, 1)
        pages = [
            node
            for node in ast.walk(kernels[0])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_conditional_attention_page"
        ]
        assert len(pages) == 1
        rescale = pages[0].args[11]
        if index == 0:
            assert isinstance(rescale, ast.Compare)
        else:
            assert isinstance(rescale, ast.BinOp) and isinstance(rescale.op, ast.BitOr)
            assert isinstance(rescale.left, ast.Compare)
            assert isinstance(rescale.right, ast.Compare)
            assert isinstance(rescale.right.ops[0], ast.NotEq)
        arenas = [
            node
            for node in ast.walk(kernels[0])
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "fragment_arena"
                for t in node.targets
            )
        ]
        assert not arenas
