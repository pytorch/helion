"""Closed original root issue/state inverse for source-only migration tests."""

from __future__ import annotations

import ast
from unittest.mock import patch

from helion._compiler.cute import chained_body_program as body
from helion._compiler.cute.chained_execution import ChainedExecution
from helion._compiler.cute.chained_k_issue import emit_k_half_issues
from helion._compiler.cute.chained_tcgen_stage import emit_full_k_issue
from helion._compiler.cute.prepared_continuation import ContinuationCommit
from helion._compiler.cute.prepared_continuation import ContinuationWait


class _OriginalRoot(ast.NodeTransformer):
    """Reuse audit_source.RestoreRoot's exact issue inverse, also for full stage1."""

    def __init__(self):
        self.issues = 0
        self.commits = 0
        self.imports = 0

    def visit_ImportFrom(self, node):
        if node.module == "helion._compiler.cute.prepared_tcgen_edge":
            assert [(a.name, a.asname) for a in node.names] == [
                ("execute_prepared_continuation", None)
            ]
            self.imports += 1
            return None
        return node

    def visit_Expr(self, node):
        call = node.value
        if (
            not isinstance(call, ast.Call)
            or ast.unparse(call.func) != "execute_prepared_continuation"
        ):
            return self.generic_visit(node)
        assert len(call.args) == 7 and not call.keywords
        payload = ast.literal_eval(call.args[0])
        assert ast.literal_eval(call.args[4]) == 0
        assert ast.unparse(call.args[5]) == "chain_warp == 0"
        assert ast.literal_eval(call.args[6]) is False
        assert isinstance(call.args[2], ast.Tuple) and len(call.args[2].elts) == 1
        bar = call.args[2].elts[0]
        assert isinstance(bar, ast.BinOp) and isinstance(bar.op, ast.Add)
        assert ast.unparse(bar.left) == "chain_bars"
        stage = ast.literal_eval(bar.right)
        assert stage in (0, 1)
        prefix = f"chain_{stage}"
        if payload[0][0] == 4:
            assert len(payload) == 1
            assert isinstance(call.args[1], ast.Tuple) and len(call.args[1].elts) == 1
            assert isinstance(call.args[1].elts[0], ast.Tuple)
            port = call.args[1].elts[0].elts
            assert len(port) == 11
            assert [ast.unparse(p) for p in port[:4]] == [
                f"{prefix}_{s}" for s in ("ra", "rb", "acc", "mma")
            ]
            assert ast.dump(port[4]) == ast.dump(bar)
            assert isinstance(call.args[3], ast.Tuple)
            assert [ast.dump(p) for p in call.args[3].elts] == [ast.dump(port[5])]
            assert [ast.literal_eval(p) for p in port[8:]] == [16, (), 128]
            tmem = ast.literal_eval(port[6])
            assert type(tmem) is bool
            _, number, begin, end, initialized, commit, wait = payload[0]
            assert number == 0 and begin == 0 and commit == wait
            assert initialized is (stage == 1)
            self.issues += 1
            if end == 8:
                assert commit is True and ast.literal_eval(port[7]) == 0
                assert ast.literal_eval(port[5]) == 0
                a_slice = f"{prefix}_ra[None, None, {prefix}_kk" + (
                    ", 0]" if tmem else "]"
                )
                lines = emit_full_k_issue(
                    prefix, stage, "0", a_slice, initialized, ChainedExecution(128)
                )
                return ast.parse("\n".join(lines[1:])).body
            assert stage == 1 and end == 4 and tmem is False
            assert ast.unparse(port[7]) == "chain_k_half * 4"
            assert ast.unparse(port[5]) == "chain_k_half"
            original = ast.parse(
                "\n".join(
                    emit_k_half_issues(
                        prefix, stage, "serial64" if commit else "overlap64", []
                    )
                )
            )
            half = original.body[1]
            assert isinstance(half, ast.For)
            statements = half.body[4:]
            return statements[:-1] if commit else statements
        self.commits += 1
        assert stage == 1 and payload == ((6, 0, False), (0, 0, False))
        assert isinstance(call.args[1], ast.Tuple) and len(call.args[1].elts) == 1
        assert isinstance(call.args[1].elts[0], ast.Tuple)
        port = call.args[1].elts[0].elts
        assert [ast.unparse(p) for p in port] == [
            "chain_1_ra",
            "chain_1_rb",
            "chain_1_acc",
            "chain_1_mma",
            "chain_bars + 1",
            "0",
            "False",
            "chain_k_half * 4",
            "16",
            "()",
            "128",
        ]
        assert ast.literal_eval(call.args[3]) == (0,)
        original = ast.parse(
            "\n".join(emit_k_half_issues(prefix, stage, "overlap64", []))
        )
        return original.body[2:]


def original_root_source(source):
    # The existing state inverse imports the seeded-body fixture at module load.
    # Delay this test-only import until that fixture has finished importing.
    from .test_cute_prepared_state_body import _OriginalInstructions

    restored = _OriginalInstructions().visit(ast.parse(source))
    inverse = _OriginalRoot()
    restored = inverse.visit(restored)
    assert inverse.issues == 2
    assert inverse.imports == inverse.issues + inverse.commits
    return ast.unparse(ast.fix_missing_locations(restored))


def private_false_source(build):
    constructor = body.plan_root_action_body

    def original(*args, **kwargs):
        kwargs["prepared_continuation"] = False
        result = constructor(*args, **kwargs)
        assert result is not None and result.continuation is None
        return result

    with patch.object(body, "plan_root_action_body", original):
        return build()


def original_root_call(calls):
    selected = [call for call in calls if call.kwargs.get("root_actions") is not None]
    assert len(selected) == 1
    action = selected[0].kwargs["root_actions"].actions[1]
    final = (
        action.k_schedule is not None
        and action.k_schedule.mode == "overlap64"
        and not action.retire_each_half
    )
    assert [
        tuple(
            name
            for name in ("root_actions", "prepared_body", "state_body")
            if call.kwargs.get(name) is not None
        )
        for call in calls
    ] == [
        ("root_actions",),
        ("prepared_body",),
        ("state_body",),
        ("prepared_body",),
    ] + ([("prepared_body",)] if final else [])
    if final:
        assert calls[-1].kwargs["prepared_body"].program.actions == (
            ContinuationCommit(0),
            ContinuationWait(0),
        )
    return selected[0]
