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
from helion._testing.tunable_fragments import load_host_function
import helion.language as hl

pytestmark = pytest.mark.cpu_only


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
