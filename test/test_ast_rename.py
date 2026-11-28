"""Free compiler names may be coalesced without capturing Python expression locals."""

from __future__ import annotations

import ast

import pytest
from torch.fx.immutable_collections import immutable_list

from helion._compiler.ast_extension import expr_from_string
from helion._compiler.ast_read_writes import ast_rename

pytestmark = pytest.mark.cpu_only


def _run(source: str, renames: dict[str, str]) -> object:
    tree = ast.parse(source)
    assert ast_rename(tree, renames) is tree
    scope = {}
    exec(compile(ast.fix_missing_locations(tree), "<renamed>", "exec"), scope)
    return scope["result"]


def test_closed_lambda_ignores_an_outer_rename() -> None:
    source = "carry = 7\nf = lambda a, b: a * b\nresult = f(a=2, b=3)"
    assert _run(source, {"a": "carry"}) == 6
    tree = ast.parse(source)
    before = ast.dump(tree)
    ast_rename(tree, {"a": "carry", "unused": "b"})
    assert ast.dump(tree) == before


def test_free_name_is_read_at_call_time_without_capturing_a_keyword() -> None:
    source = """
x = 7
f = lambda a, /, b=2, *, c=3: x * b + c
x = 11
result = f(1, b=4, c=5)
"""
    assert _run(source, {"x": "b"}) == 49


def test_nested_defaults_see_their_defining_scope() -> None:
    source = """
x = 5
f = lambda a=x: (lambda b=a: (a, b, x))
g = f()
x = 11
result = g(), f(a=3)(b=4)
"""
    assert _run(source, {"x": "a"}) == ((5, 5, 11), (3, 4, 11))


def test_comprehension_uses_outer_first_iterable_and_inner_targets() -> None:
    source = """
x = [2, 4]
result = [(a, b, x[0]) for a in x for b in (a+i for i in range(a)) if b%2==0]
"""
    assert _run(source, {"x": "a", "i": "junk"}) == [(2, 2, 2), (4, 4, 2), (4, 6, 2)]


def test_later_generator_target_is_local_even_before_it_is_assigned() -> None:
    source = """
later = 1
result = [item for item in [2] if later for later in [3]]
"""
    with pytest.raises(UnboundLocalError):
        _run(source, {"later": "outside"})


def test_plain_kernel_rename_preserves_signature_and_ast_objects() -> None:
    fn = ast.parse(
        "def f(input):\n    temporary = input + 2\n    return temporary"
    ).body[0]
    assert isinstance(fn, ast.FunctionDef)
    value = fn.body[-1]
    assert ast_rename(fn, {"temporary": "iv", "iv": "next"}) is fn
    assert fn.body[-1] is value
    assert ast.unparse(fn) == "def f(input):\n    iv = input + 2\n    return iv"


def test_factory_defaults_run_once_in_defining_scope_and_keep_all_keywords() -> None:
    source = """
events = []
def once(x):
    events.append(x)
    return x
free = 4
f = lambda a=(record := once(6)), *rest, b=once(7), **kw: (free, a, rest, b, kw)
free = 9
result = f(a=2, b=3, _helion_scoped_0=11), f(1, 5), record, events
"""
    assert _run(source, {"free": "a"}) == (
        (9, 2, (), 3, {"_helion_scoped_0": 11}),
        (9, 1, (5,), 7, {}),
        6,
        [6, 7],
    )


def test_rename_inside_two_comprehensions_and_lambda_keeps_outer_capture() -> None:
    source = """
x = 7
callbacks = {a: [lambda a: x * a for a in (2,)] for a in (3, 5)}
x = 11
result = {key: callback(a=2) for key, (callback,) in callbacks.items()}
"""
    assert _run(source, {"x": "a"}) == {3: 22, 5: 22}


def test_walrus_inside_comprehension_binds_enclosing_lambda() -> None:
    source = """
x = 7
f = lambda foo: ([captured := i for i in (foo,)], captured + x)
result = f(2)
"""
    assert _run(source, {"captured": "other", "x": "captured"}) == ([2], 9)


def test_outlined_helper_owner_coalesces_formals_before_renaming_the_body() -> None:
    # DeviceFunction.codegen_helper_functions rewrites the argument declarations
    # independently, then passes the statement module to ast_rename.
    (fn,) = ast.parse(
        "def f(source):\n    return (lambda source: source)(3) + source"
    ).body
    assert isinstance(fn, ast.FunctionDef)
    renames = {"source": "coalesced"}
    fn.args.args[0].arg = renames["source"]
    module = ast.Module(body=fn.body, type_ignores=[])
    ast_rename(module, renames)
    assert (
        ast.unparse(fn)
        == "def f(coalesced):\n    return (lambda source: source)(3) + coalesced"
    )
    assert _run(ast.unparse(fn) + "\nresult = f(7)", {}) == 10


def test_template_placeholders_remain_independent_across_lambda_scopes() -> None:
    for text, renames, outer, expected in [
        ("a", {"a": "carry"}, {"carry": 7}, (7, 6)),
        ("x + 1", {"x": "a"}, {"a": 7}, (8, 24)),
    ]:
        operand = expr_from_string(text)
        tree = expr_from_string("({v}, (lambda a,b: {v} * b)(2,3))", v=operand)
        assert isinstance(tree, ast.Tuple) and tree.elts[0] is operand
        changed = ast.Expression(ast_rename(tree, renames))
        assert (
            eval(
                compile(ast.fix_missing_locations(changed), "<template>", "eval"), outer
            )
            == expected
        )


def test_fx_immutable_arguments_inside_and_outside_a_scoped_expression() -> None:
    tree = ast.parse("hold(lambda a: read(x))", mode="eval").body
    assert isinstance(tree, ast.Call)
    fn = tree.args[0]
    assert isinstance(fn, ast.Lambda)
    call = fn.body
    assert isinstance(call, ast.Call)
    before = immutable_list(call.args)
    call.args = before
    tree.args = immutable_list(tree.args)
    changed = ast.Expression(ast_rename(tree, {"x": "a"}))
    assert ast.unparse(before[0]) == "x"
    assert isinstance(tree.args, immutable_list)
    factory = tree.args[0]
    assert isinstance(factory, ast.Call) and isinstance(factory.func, ast.Lambda)
    rewritten = factory.func.body
    assert isinstance(rewritten, ast.Lambda) and isinstance(rewritten.body, ast.Call)
    assert isinstance(rewritten.body.args, immutable_list)
    result = eval(
        compile(ast.fix_missing_locations(changed), "<fx>", "eval"),
        {"hold": lambda f: f, "read": lambda x: x, "a": 41},
    )
    assert result(a=7) == 41


def test_deep_expression_does_not_recurse_inside_or_beside_a_callback() -> None:
    def chain(name: str) -> ast.expr:
        node: ast.expr = ast.Name(name, ast.Load())
        for _i in range(1500):
            node = ast.BinOp(node, ast.Add(), ast.Constant(1))
        return node

    def leaf(node: ast.expr) -> ast.expr:
        depth = 0
        while isinstance(node, ast.BinOp):
            node = node.left
            depth += 1
        assert depth == 1500
        return node

    module = ast.parse("f = lambda a: x")
    assignment = module.body[0]
    assert isinstance(assignment, ast.Assign)
    fn = assignment.value
    assert isinstance(fn, ast.Lambda)
    fn.body = chain("x")
    unrelated = ast.Expr(chain("base"))
    module.body.append(unrelated)
    ast_rename(module, {"x": "a"})
    assert isinstance(assignment.value, ast.Call) and isinstance(
        assignment.value.func, ast.Lambda
    )
    renamed = assignment.value.func.body
    assert isinstance(renamed, ast.Lambda)
    assert isinstance(leaf(renamed.body), ast.Call)
    base = leaf(unrelated.value)
    assert isinstance(base, ast.Name) and base.id == "base"
