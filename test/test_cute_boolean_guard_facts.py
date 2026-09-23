from __future__ import annotations

import ast
import copy
import itertools
import struct
import textwrap
from types import SimpleNamespace

import pytest

from helion._compiler.ast_read_writes import ast_rename
from helion._compiler.cute import boolean_guards
from helion._compiler.cute.boolean_guards import reassociate_boolean_guards


def _pair(source: str) -> tuple[list[ast.stmt], list[ast.stmt]]:
    before = ast.parse(textwrap.dedent(source)).body
    after = copy.deepcopy(before)
    assert reassociate_boolean_guards(after) is after
    return before, after


def _dump(body: list[ast.stmt]) -> str:
    return ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)


def _value(body: list[ast.stmt], name: str = "result") -> ast.expr:
    values = [
        node.value
        for node in ast.walk(ast.Module(body=body, type_ignores=[]))
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    assert len(values) == 1
    return values[0]


def _right_leaves(node: ast.expr) -> list[str]:
    leaves = []
    while isinstance(node, ast.BoolOp):
        assert isinstance(node.op, ast.And)
        assert len(node.values) == 2
        leaves.append(ast.unparse(node.values[0]))
        node = node.values[1]
    return [*leaves, ast.unparse(node)]


def _compile(body: list[ast.stmt]):
    return compile(
        ast.fix_missing_locations(
            ast.Module(body=copy.deepcopy(body), type_ignores=[])
        ),
        "<boolean-facts>",
        "exec",
    )


def _run(code, **values: object) -> dict[str, object]:
    namespace = {
        "cutlass": SimpleNamespace(Int32=int, Float32=float, Boolean=bool),
        **values,
    }
    exec(code, namespace)
    return namespace


def test_dominating_boolean_aliases_preserve_complete_truth_table() -> None:
    leaves = ["alias", *(f"x{index} != 0" for index in range(1, 11))]
    before, after = _pair(
        "mask = x0 != 0\nalias = mask\nresult = " + " and ".join(leaves)
    )
    assert _right_leaves(_value(after)) == leaves
    assert ast.dump(before[0]) == ast.dump(after[0])
    assert ast.dump(before[1]) == ast.dump(after[1])
    codes = [_compile(body) for body in (before, after)]
    for values in itertools.product((0, 1), repeat=11):
        inputs = {f"x{index}": value for index, value in enumerate(values)}
        expected, actual = (_run(code, **inputs)["result"] for code in codes)
        assert actual is expected
        assert type(actual) is bool


def test_no_definition_substitution_after_input_changes() -> None:
    before, after = _pair("""
        x = 1
        mask = x > 0
        alias = mask
        x = -1
        result = alias and x < 0 and True
    """)
    assert _right_leaves(_value(after)) == ["alias", "x < 0", "True"]
    for body in (before, after):
        assert _run(_compile(body))["result"] is True


@pytest.mark.parametrize(
    "x", [float("nan"), float("inf"), -float("inf"), -0.0, 0.0, -2.0, 2.0]
)
def test_named_guard_retains_inactive_conversion_and_division(x: float) -> None:
    before, after = _pair("""
        finite = x == x and negative_inf < x < inf
        result = finite and cutlass.Int32(x) != 0 and 8 // cutlass.Int32(x) > 0
    """)
    assert len(_right_leaves(_value(after))) == 3
    values = {"x": x, "inf": float("inf"), "negative_inf": -float("inf")}
    expected, actual = (
        _run(_compile(body), **values)["result"] for body in (before, after)
    )
    assert actual is expected


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    ("tail", "x", "exception"),
    [
        ("8 // x > 0", 0, ZeroDivisionError),
        ("cutlass.Int32(x) > 0", float("nan"), ValueError),
        ("cutlass.Int32(x) > 0", float("inf"), OverflowError),
    ],
)
def test_named_guard_retains_active_errors(
    enabled: bool, tail: str, x: float, exception: type[Exception]
) -> None:
    before, after = _pair(
        f"mask = enabled == True\nresult = mask and x != 2 and {tail}"
    )
    assert len(_right_leaves(_value(after))) == 3
    for body in (before, after):
        if enabled:
            with pytest.raises(exception):
                _run(_compile(body), enabled=enabled, x=x)
        else:
            assert _run(_compile(body), enabled=enabled, x=x)["result"] is False


@pytest.mark.parametrize("value", [2, 2.5, -0.0, 0.0, float("nan")])
def test_unproved_numeric_alias_keeps_value_and_conversion_order(value: float) -> None:
    before, after = _pair("""
        mask = value
        result = first > 0 and mask and last > 0
    """)
    assert _dump(before) == _dump(after)
    expected, actual = (
        _run(_compile(body), value=value, first=1, last=0)["result"]
        for body in (before, after)
    )
    assert type(actual) is type(expected)
    if isinstance(actual, float):
        assert struct.pack("d", actual) == struct.pack("d", expected)
    else:
        assert actual == expected


@pytest.mark.parametrize(
    "definition",
    [
        "mask = 1",
        "mask = cutlass.Int32(x)",
        "mask = x < 2 and 3",
        "mask = x < 2 or -0.0",
        "mask = effect() > 0",
        "mask = values[0] > 0",
        "mask = values.load() > 0",
        "mask = x < 2 if enabled else False",
        "mask: bool = x < 2",
        "mask = alias = x < 2",
    ],
)
def test_unknown_boolean_definitions_keep_flat_guard(definition: str) -> None:
    before, after = _pair(definition + "\nresult = mask and x != 0 and x < 8")
    assert _dump(before) == _dump(after)


@pytest.mark.parametrize("stop", range(4))
def test_effectful_guard_keeps_calls_and_their_evaluation_order(stop: int) -> None:
    before, after = _pair("""
        mask = enabled > 0
        result = mask and call(0) > 0 and call(1) > 0 and call(2) > 0 and call(3) > 0
    """)
    assert _dump(before) == _dump(after)
    for body in (before, after):
        trace: list[int] = []

        def call(index: int, trace: list[int] = trace) -> int:
            trace.append(index)
            return int(index != stop)

        assert _run(_compile(body), enabled=1, call=call)["result"] is False
        assert trace == list(range(stop + 1))


@pytest.mark.parametrize(
    "rebind",
    [
        "mask = 2",
        "if enabled:\n    mask = 2",
        "mask += 1",
        "for mask in range(2):\n    pass",
        "for lane in range(2):\n    mask = lane",
        "while enabled:\n    mask = 2\n    enabled = False",
        "del mask",
        "mask.value = 2",
        "mask[0] = 2",
        "mask.value[0] = 2",
    ],
)
def test_all_syntactic_writes_prevent_stale_boolean_facts(rebind: str) -> None:
    before, after = _pair(
        "mask = x > 0\n" + rebind + "\nresult = mask and x != 0 and x < 8"
    )
    assert _dump(before) == _dump(after)


@pytest.mark.parametrize("enabled", [False, True])
def test_conditional_rebinding_keeps_numeric_semantics(enabled: bool) -> None:
    before, after = _pair("""
        mask = x > 0
        if enabled:
            mask = -0.0
        result = True and mask and True
    """)
    assert _dump(before) == _dump(after)
    expected, actual = (
        _run(_compile(body), x=1, enabled=enabled)["result"] for body in (before, after)
    )
    assert type(actual) is type(expected)
    assert struct.pack("d", actual) == struct.pack("d", expected)


def test_final_variable_renaming_exposes_alias_rebinding() -> None:
    module = ast.parse("""
mask = x > 0
later_alias = -0.0
result = mask and True and True
""")
    ast_rename(module, {"later_alias": "mask"})
    before = _dump(module.body)
    reassociate_boolean_guards(module.body)
    assert _dump(module.body) == before
    result = _run(_compile(module.body), x=1)["result"]
    assert type(result) is float
    assert struct.pack("d", result) == struct.pack("d", -0.0)


@pytest.mark.parametrize("enabled", [False, True])
def test_branch_facts_are_local_and_dominating(enabled: bool) -> None:
    before, after = _pair("""
        if enabled:
            mask = x > 0
            inside = mask and x != 0 and x < 8
        result = mask and x != 0 and x < 8
    """)
    assert len(_right_leaves(_value(after, "inside"))) == 3
    assert ast.dump(_value(before)) == ast.dump(_value(after))
    expected, actual = (
        _run(_compile(body), enabled=enabled, x=1, mask=-0.0)["result"]
        for body in (before, after)
    )
    assert type(actual) is type(expected)
    assert struct.pack("d", actual) == struct.pack("d", expected)


@pytest.mark.parametrize("count", [0, 1, 3])
def test_loop_facts_do_not_escape_or_hide_forward_carried_values(count: int) -> None:
    before, after = _pair("""
        observations = []
        for lane in range(count):
            prior = mask and True and True
            mask = lane != 1
            current = mask and True and True
            observations.append((prior, current))
        result = mask and True and True
    """)
    assert len(_right_leaves(_value(after, "current"))) == 3
    for name in ("prior", "result"):
        assert ast.dump(_value(before, name)) == ast.dump(_value(after, name))
    expected, actual = (
        _run(_compile(body), count=count, mask=-0.0) for body in (before, after)
    )
    assert actual["observations"] == expected["observations"]
    assert type(actual["result"]) is type(expected["result"])
    assert struct.pack("d", actual["result"]) == struct.pack("d", expected["result"])


def test_loop_condition_uses_only_preexisting_boolean_facts() -> None:
    before, after = _pair("""
        mask = x > 0
        lane = 0
        while mask and lane < 3 and x < 8:
            lane += 1
        result = lane
    """)
    loop = after[2]
    assert isinstance(loop, ast.While)
    assert len(_right_leaves(loop.test)) == 3
    for body in (before, after):
        assert _run(_compile(body), x=1)["result"] == 3


@pytest.mark.parametrize("argument", [-0.0, 0.0, 2, 2.5, float("nan")])
def test_lambda_scope_does_not_inherit_outer_boolean_type(argument: float) -> None:
    before, after = _pair("""
        mask = x > 0
        reduction = lambda mask: mask and True and True
        result = mask and x != 0 and x < 8
        inner = reduction(argument)
    """)
    assert len(_right_leaves(_value(after))) == 3
    assert ast.dump(_value(before, "reduction")) == ast.dump(_value(after, "reduction"))
    expected, actual = (
        _run(_compile(body), x=1, argument=argument) for body in (before, after)
    )
    assert actual["result"] is expected["result"]
    assert type(actual["inner"]) is type(expected["inner"])
    assert struct.pack("d", actual["inner"]) == struct.pack("d", expected["inner"])


@pytest.mark.parametrize("share_statement", [False, True])
def test_shared_ast_cannot_leak_branch_facts_into_another_scope(
    share_statement: bool,
) -> None:
    body = ast.parse("""
if enabled:
    mask = x > 0
    local = (mask and x != 0 and x < 8,)
outside = (mask and x != 0 and x < 8,)
""").body
    branch = body[0]
    outside = body[1]
    assert isinstance(branch, ast.If) and isinstance(outside, ast.Assign)
    local = branch.body[1]
    assert isinstance(local, ast.Assign)
    if share_statement:
        body[1] = local
    else:
        outside.value = local.value
    original_outside = ast.dump(body[1])
    reassociate_boolean_guards(body)
    assert ast.dump(body[1]) == original_outside
    rewritten_branch = body[0]
    assert isinstance(rewritten_branch, ast.If)
    rewritten_statement = rewritten_branch.body[1]
    assert isinstance(rewritten_statement, ast.Assign)
    rewritten_value = rewritten_statement.value
    assert isinstance(rewritten_value, ast.Tuple)
    assert len(_right_leaves(rewritten_value.elts[0])) == 3


def test_shared_ast_cannot_leak_outer_facts_into_lambda_scope() -> None:
    body = ast.parse("""
mask = x > 0
callback = lambda mask: (mask and x != 0 and x < 8,)
result = (mask and x != 0 and x < 8,)
""").body
    callback, result = body[1:]
    assert isinstance(callback, ast.Assign) and isinstance(result, ast.Assign)
    assert isinstance(callback.value, ast.Lambda)
    result.value = callback.value.body
    original_callback = ast.dump(callback)
    reassociate_boolean_guards(body)
    assert ast.dump(body[1]) == original_callback
    rewritten_statement = body[2]
    assert isinstance(rewritten_statement, ast.Assign)
    rewritten = rewritten_statement.value
    assert isinstance(rewritten, ast.Tuple)
    assert len(_right_leaves(rewritten.elts[0])) == 3


@pytest.mark.parametrize(
    "opaque",
    [
        "def nested():\n    pass",
        "class Nested:\n    pass",
        "values = [x for x in values]",
        "alias = (mask := False)",
        "try:\n    effect()\nexcept ValueError:\n    pass",
        "with context:\n    pass",
        "global mask",
        "match x:\n    case 0:\n        pass",
        "return mask",
        "raise ValueError()",
        "while x:\n    break",
        "while x:\n    continue",
    ],
)
def test_opaque_binding_or_control_scope_declines_before_mutation(opaque: str) -> None:
    before, after = _pair("mask = x > 0\nresult = mask and True and True\n" + opaque)
    assert _dump(before) == _dump(after)


def test_oversized_analysis_declines_before_partial_rewrite(monkeypatch) -> None:
    monkeypatch.setattr(boolean_guards, "_ANALYSIS_BUDGET", 64)
    source = "mask = x > 0\nresult = mask and True and True\n"
    source += "\n".join(f"value{index} = {index}" for index in range(64))
    before, after = _pair(source)
    assert _dump(before) == _dump(after)


def test_boolean_dependency_dag_is_not_expanded() -> None:
    source = "mask0 = x > 0\n"
    source += "\n".join(
        f"mask{index} = mask{index - 1} and mask{index - 1} and mask{index - 1}"
        for index in range(1, 65)
    )
    source += "\nresult = mask64 and x != 0 and x < 8"
    before, after = _pair(source)
    assert len(_right_leaves(_value(after))) == 3
    counts = [
        sum(1 for _ in ast.walk(ast.Module(body=body, type_ignores=[])))
        for body in (before, after)
    ]
    assert counts[1] < 2 * counts[0]
    for x in (0, 1, 9):
        expected, actual = (
            _run(_compile(body), x=x)["result"] for body in (before, after)
        )
        assert actual is expected
