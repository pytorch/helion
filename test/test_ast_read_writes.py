from __future__ import annotations

import ast
from contextlib import nullcontext
import sys
import unittest

import pytest
import torch

import helion
from helion._compiler.ast_read_writes import ReadWrites
from helion._compiler.ast_read_writes import dead_assignment_elimination
from helion._compiler.ast_read_writes import dead_expression_elimination
from helion._compiler.ast_read_writes import definitely_does_not_have_side_effects
from helion._testing import cpu_render_target
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.tunable_fragments import load_host_function
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@helion.kernel(backend="triton", static_shapes=True)
def _reassigned_on_the_host(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    flag = 0
    flag = x.add_(1)  # noqa: F841
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


@helion.kernel(backend="triton", static_shapes=True)
def _reassigned_in_a_branch(x: torch.Tensor, update: bool) -> torch.Tensor:
    out = torch.empty_like(x)
    flag = x
    if update:
        flag = x.add_(1)  # noqa: F841
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


@helion.kernel(backend="triton", static_shapes=True)
def _unread_in_a_branch(x: torch.Tensor, update: bool) -> torch.Tensor:
    out = torch.empty_like(x)
    flag = 0
    if update:
        flag = 1  # noqa: F841
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


def _unread_inner_tile_begin(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        v = x[tile, :]
        for t2 in hl.tile(x.size(1)):
            b = t2.begin  # noqa: F841
        out[tile, :] = v
    return out


class TestAstReadWrites(unittest.TestCase):
    def test_empty_if_cleanup_retains_condition_effects(self) -> None:
        source = """
if check('first', flag):
    unused = 1
elif check('second', True):
    None
else:
    unused = 2
"""
        for flag, expected in ((True, ["first"]), (False, ["first", "second"])):
            module = ast.parse(source)
            dead_assignment_elimination(module.body, ["unused"])
            compile(ast.fix_missing_locations(module), "<assignment-cleanup>", "exec")
            dead_expression_elimination(module.body)
            events = []

            def check(label, result, records=events):
                records.append(label)
                return result

            exec(
                compile(ast.fix_missing_locations(module), "<cleanup>", "exec"),
                {"check": check, "flag": flag},
            )
            self.assertEqual(events, expected)

    def test_compiler_shape_helpers_are_opt_in_pure(self) -> None:
        for source in (
            "triton.cdiv(x, 4)",
            "triton.next_power_of_2(x)",
            "_cdiv(x, 4)",
            "_next_power_of_2(x)",
        ):
            expression = ast.parse(source, mode="eval").body
            self.assertFalse(definitely_does_not_have_side_effects(expression))
            self.assertTrue(
                definitely_does_not_have_side_effects(
                    expression,
                    allow_compiler_shape_helpers=True,
                )
            )

    def test_atomic_first_argument_is_a_semantic_read(self) -> None:
        rw = ReadWrites.from_list(
            ast.parse(
                """
hl.store(stored, [index], value)
hl.atomic_add(accumulated, [index], value)
"""
            ).body
        )

        self.assertNotIn("stored", rw.atomic_reads)
        self.assertIn("accumulated", rw.atomic_reads)

    def test_dead_shape_helper_assignments_are_removed_transitively(self) -> None:
        body = ast.parse(
            """
rdim = triton.next_power_of_2(block_size)
block_size = 128
"""
        ).body

        dead_assignment_elimination(body, ["rdim", "block_size"])

        self.assertEqual(body, [])

    def test_an_assignment_with_effects_outlives_its_dead_name(self) -> None:
        # ``flag`` is never read and has a pure assignment, which makes it a
        # dead name; its other assignments still run when they have effects.
        for statements, kept in (
            ("flag = 0\nflag = x.add_(1)", "flag = x.add_(1)"),
            ("flag = 0\nflag += x.add_(1)", "flag = 0\nflag += x.add_(1)"),
            ("flag = 0\nflag, other = (x.add_(1), 0)", "flag, other = (x.add_(1), 0)"),
            ("flag = 0\nflag = other = x.add_(1)", "flag = other = x.add_(1)"),
            ("flag = 0\nif c:\n    flag = x.add_(1)", "if c:\n    flag = x.add_(1)"),
            (
                "flag = 0\nfor i in r:\n    flag = x.add_(1)",
                "for i in r:\n    flag = x.add_(1)",
            ),
            (
                "flag = 0\nwith m:\n    flag = x.add_(1)",
                "with m:\n    flag = x.add_(1)",
            ),
        ):
            body = ast.parse(statements).body
            dead_assignment_elimination(
                body,
                ["flag"],
                removable=lambda stmt: definitely_does_not_have_side_effects(
                    stmt.value
                ),
            )
            self.assertEqual(ast.unparse(body), kept, statements)

    @skipUnlessBackends(["triton"])
    @helion._testing.skipIfRefEager("inspects generated code; ref mode has none")
    def test_host_code_keeps_an_in_place_update_of_an_unread_variable(self) -> None:
        # The host's dead-code elimination took ``flag`` for dead (one pure
        # assignment, no read) and deleted ``flag = x.add_(1)`` with it, in a
        # branch too (leaving the branch empty).
        for kernel, name, args in (
            (_reassigned_on_the_host, "_reassigned_on_the_host", (torch.zeros(16),)),
            (
                _reassigned_in_a_branch,
                "_reassigned_in_a_branch",
                (torch.zeros(16), True),
            ),
        ):
            bound = cpu_bind(kernel, args)
            code = bound.to_code(bound.config_spec.default_config())
            # The code itself: the source comments quote the user's lines.
            (host,) = [
                node
                for node in ast.parse(code).body
                if isinstance(node, ast.FunctionDef) and node.name == name
            ]
            self.assertIn("flag = x.add_(1)", ast.unparse(host), name)

    def test_a_block_whose_statements_go_keeps_a_pass(self) -> None:
        body = ast.parse(
            "flag = 0\nif c:\n    flag = 1\nelse:\n    flag = 2\n"
            "try:\n    x.add_(1)\nfinally:\n    flag = 3"
        ).body
        dead_assignment_elimination(body, ["flag"])
        self.assertEqual(
            ast.unparse(body),
            "if c:\n    pass\ntry:\n    x.add_(1)\nfinally:\n    pass",
        )
        body = ast.parse("while c:\n    y").body
        dead_expression_elimination(body)
        self.assertEqual(ast.unparse(body), "while c:\n    pass")

    @unittest.skipIf(sys.version_info < (3, 11), "except* is new in Python 3.11")
    def test_a_try_star_block_whose_statements_go_keeps_a_pass(self) -> None:
        body = ast.parse("try:\n    flag = 1\nexcept* ValueError:\n    x.add_(1)").body
        dead_assignment_elimination(body, ["flag"])
        self.assertEqual(
            ast.unparse(body), "try:\n    pass\nexcept* ValueError:\n    x.add_(1)"
        )

    @skipUnlessBackends(["triton"])
    @helion._testing.skipIfRefEager("inspects generated code; ref mode has none")
    def test_host_code_of_an_unread_variable_set_in_a_branch_parses(self) -> None:
        # Both assignments of ``flag`` go; the branch left empty did not parse
        # (IndentationError).  The ``pass`` gets no source comment.
        bound = cpu_bind(_unread_in_a_branch, (torch.zeros(16), True))
        code = bound.to_code(bound.config_spec.default_config())
        (host,) = [
            node
            for node in ast.parse(code).body
            if isinstance(node, ast.FunctionDef) and node.name == "_unread_in_a_branch"
        ]
        self.assertNotIn("flag", ast.unparse(host))
        self.assertIn("if update:\n        pass\n", code)

    @skipUnlessBackends(["triton", "cute"])
    @helion._testing.skipIfRefEager("inspects generated code; ref mode has none")
    def test_device_code_of_a_loop_whose_body_goes_parses(self) -> None:
        # The device's dead-code elimination deletes the inner loop's only
        # statement (``b`` is never read); the loop left empty did not parse.
        kernel = helion.kernel(_unread_inner_tile_begin, static_shapes=True)
        bound = cpu_bind(kernel, (torch.zeros(4, 8),))
        code = bound.to_code(bound.config_spec.default_config())
        loops = [
            node for node in ast.walk(ast.parse(code)) if isinstance(node, ast.For)
        ]
        self.assertTrue(loops, code)
        self.assertIn(
            [ast.Pass], [[type(stmt) for stmt in loop.body] for loop in loops]
        )


@pytest.mark.parametrize("form", ["assignment", "specialize"])
def test_generated_host_retains_empty_branch(form):
    @helion.kernel(autotune_effort="none")
    def fn(x, form: hl.constexpr, enabled: hl.constexpr):
        _unused = 0
        if form == "assignment":
            if enabled:
                _unused = 1
            else:
                _unused = 2
        else:
            if enabled:
                hl.specialize(x.numel())
        out = torch.empty_like(x)
        for tile in hl.tile(x.numel()):
            out[tile] = x[tile]
        return out

    with cpu_render_target():
        x = torch.arange(7, dtype=torch.float32)
        bound = fn.bind((x, form, True))
        config = bound.config_spec.default_config()
        ast.parse(bound.to_code(config))
        calls = []

        def observe(_kernel, _grid, source, destination, *_args, **_kwargs):
            calls.append(source)
            destination.copy_(source)

        host = load_host_function(bound, config, observe)
        actual = host(x, form, True)
        assert len(calls) == 1 and calls[0] is x
        torch.testing.assert_close(actual, x, rtol=0, atol=0)


@pytest.mark.parametrize(
    "source",
    [
        "for item in records():\n    unused = 1\n",
        "while check():\n    unused = 1\n",
        "try:\n    unused = 1\nfinally:\n    record()\n",
        "try:\n    record()\nfinally:\n    unused = 1\n",
        "try:\n    fail()\nexcept ValueError:\n    unused = 1\nfinally:\n    record()\n",
    ],
)
def test_empty_required_suites_preserve_execution(source):
    def execute(module):
        events = []

        def records():
            events.append("iter")
            return (1, 2)

        def check():
            events.append("check")
            return len(events) < 3

        def record():
            events.append("record")

        def fail():
            events.append("fail")
            raise ValueError("expected")

        exec(
            compile(ast.fix_missing_locations(module), "<cleanup>", "exec"),
            {"records": records, "check": check, "record": record, "fail": fail},
        )
        return events

    expected = execute(ast.parse(source))
    module = ast.parse(source)
    dead_assignment_elimination(module.body, ["unused"])
    # Device cleanup can stop here without the later host-expression pass.
    assert execute(module) == expected
    dead_expression_elimination(module.body)
    assert execute(module) == expected


@pytest.mark.parametrize("form", ["except", "finally"])
def test_generated_host_retains_empty_exception_suites(form):
    @helion.kernel(autotune_effort="none")
    def fn(x, form: hl.constexpr):
        if form == "except":
            try:
                out = torch.empty_like(x)
            except Exception:
                _unused = 2
        else:
            try:
                _unused = 1
            finally:
                out = torch.empty_like(x)
        for tile in hl.tile(x.numel()):
            out[tile] = x[tile]
        return out

    with cpu_render_target():
        x = torch.arange(7, dtype=torch.float32)
        bound = fn.bind((x, form))
        config = bound.config_spec.default_config()
        ast.parse(bound.to_code(config))
        calls = []

        def observe(_kernel, _grid, source, destination, *_args, **_kwargs):
            calls.append(source)
            destination.copy_(source)

        host = load_host_function(bound, config, observe)
        actual = host(x, form)
        assert len(calls) == 1 and calls[0] is x
        torch.testing.assert_close(actual, x, rtol=0, atol=0)


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("def no_op():\n    None\nno_op()\nrecord()\n", id="function"),
        pytest.param("async def no_op():\n    None\nrecord()\n", id="async_function"),
        pytest.param("class Empty:\n    None\nrecord()\n", id="class"),
        pytest.param("with context():\n    None\nrecord()\n", id="with"),
        pytest.param(
            "async def no_op():\n    async with context():\n        None\nrecord()\n",
            id="async_with",
        ),
        pytest.param(
            "async def no_op():\n    async for item in values:\n        None\nrecord()\n",
            id="async_for",
        ),
        pytest.param(
            "match 1:\n    case 1:\n        None\nrecord()\n", id="match_case"
        ),
        pytest.param(
            "try:\n    None\nexcept* ValueError:\n    None\nrecord()\n",
            id="try_star",
            marks=pytest.mark.skipif(
                sys.version_info < (3, 11), reason="except* requires Python 3.11"
            ),
        ),
    ],
)
def test_empty_nested_suites_remain_valid(source):
    def execute(module):
        events = []
        exec(
            compile(ast.fix_missing_locations(module), "<cleanup>", "exec"),
            {"record": lambda: events.append("record"), "context": nullcontext},
        )
        return events

    expected = execute(ast.parse(source))
    module = ast.parse(source)
    dead_expression_elimination(module.body)
    assert execute(module) == expected == ["record"]


if __name__ == "__main__":
    unittest.main()
