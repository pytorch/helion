"""Compiler-owned CuTe branch selection across root phases."""

from __future__ import annotations

import ast
from types import SimpleNamespace

import pytest

from helion._compiler.ast_extension import create
from helion._compiler.cute.phase_dispatch import specialize_phase_cases
from helion._compiler.program_id import ForEachProgramID
from helion._compiler.program_id import PersistentPhase

pytestmark = pytest.mark.cpu_only


def _run_at(statements: list[ast.AST], pid: int) -> tuple[object, object, object]:
    scope = {"seen": [], "pid_shared": pid, "prefix": 0, "suffix": 0}
    tree = ast.Module(body=statements, type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), "<phase>", "exec"), scope)
    return scope["prefix"], scope["seen"], scope["suffix"]


def test_phase_omits_unreachable_leaves_but_retains_pid_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = ForEachProgramID("pid_shared")
    sizes = [1, 3, 0, 2, 1]
    ends = []
    total = 0
    for index, size in enumerate(sizes):
        owner.case_bodies.append(
            ast.parse(f"seen.append(({index}, pid_shared - {total}))").body
        )
        total += size
        ends.append(total)
        if index < len(sizes) - 1:
            # Real dispatch branches carry source locations. A phase can
            # cut across either side of one without owning its entire tree.
            owner.case_branches.append(
                create(
                    ast.If,
                    test=ast.parse(f"pid_shared < {total}", mode="eval").body,
                    body=[ast.Pass()],
                    orelse=[],
                )
            )
    before, after = ast.parse("prefix += 1\nsuffix += 1").body
    full = [before, *owner.codegen_dispatch(), after]
    # A backend can replace a dispatch node with its own conditional. Even
    # an identical-looking replacement must keep all its arms.
    replacement = ast.parse(ast.unparse(full[1])).body[0]
    assert specialize_phase_cases(owner, [replacement], 0, 1) == [replacement]
    for first, stop in ((0, 1), (1, 4), (4, 5)):
        tree = ast.Module(
            body=specialize_phase_cases(owner, full, first, stop), type_ignores=[]
        )
        if (first, stop) == (1, 4):
            # Two arms survive, so the narrower phase owns a copy of its
            # original selector. Tail peeling must still see the verified PID.
            (selector,) = [stmt for stmt in tree.body if isinstance(stmt, ast.If)]
            assert all(selector is not branch for branch in owner.case_branches)
            (loop,) = ast.parse("for pid in range(7):\n    if 1 <= pid < 6: pass").body
            assert isinstance(loop, ast.For)
            guard = loop.body[0]
            assert isinstance(guard, ast.If)
            (init,) = ast.parse("pid_shared = pid").body
            guard.body = [init, selector]
            PersistentPhase.mark(loop, "1", "6", [], init)
            proof = PersistentPhase.match(loop)
            assert proof is not None
            assert id(selector) in proof.phase.uniform_guards(proof.guard)
            # Parsing identical text must not confer the original marker.
            (foreign,) = ast.parse(ast.unparse(selector)).body
            guard.body = [init, foreign]
            assert id(foreign) not in proof.phase.uniform_guards(proof.guard)
            # A pass can also replace the test on the original If. The new
            # selector must inherit neither a proof for the altered boundary
            # nor a stale one that would revive if the old test is restored.
            original = full[1]
            assert isinstance(original, ast.If)
            with monkeypatch.context() as patch:
                patch.setattr(
                    original, "test", ast.parse("pid_shared < 5", mode="eval").body
                )
                (changed,) = [
                    s
                    for s in specialize_phase_cases(owner, full, first, stop)
                    if isinstance(s, ast.If)
                ]
            guard.body = [init, changed]
            assert id(changed) not in proof.phase.uniform_guards(proof.guard)
            changed.test = original.test
            assert id(changed) not in proof.phase.uniform_guards(proof.guard)
            # A changed boundary can send a single-root phase into the
            # opposite arm. Its full execution is the reference.
            for cutoff, lo, hi, pid in ((5, 3, 4, 4), (0, 0, 1, 0)):
                with monkeypatch.context() as patch:
                    patch.setattr(
                        original,
                        "test",
                        ast.parse(f"pid_shared < {cutoff}", mode="eval").body,
                    )
                    partial = specialize_phase_cases(owner, full, lo, hi)
                    assert _run_at(partial, pid) == _run_at(full, pid)
        # A whole phase should own only its live roots' storage and helpers.
        sites = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        assert len(sites) == stop - first
        code = compile(ast.fix_missing_locations(tree), "<phase>", "exec")
        start = ends[first - 1] if first else 0
        expected = [(r, i) for r in range(first, stop) for i in range(sizes[r])]
        for pid, pair in enumerate(expected, start):
            scope = {"seen": [], "pid_shared": pid, "prefix": 0, "suffix": 0}
            exec(code, scope)
            assert (scope["prefix"], scope["seen"], scope["suffix"]) == (1, [pair], 1)


@pytest.mark.parametrize(
    ("boundary_expr", "write"),
    # Either setup before the outer branch or a write inside its first arm
    # can invalidate each of these otherwise compiler-owned predicates.
    [
        ("boundary", "pid_shared += 1"),
        ("boundary", "boundary, other = 0, 3"),
        ("limit[0]", "alias[0] = 0"),
        ("cutlass.Int64(limit[0])", "alias[0] = 0"),
        ("cutlass.Int64(boundary)", "sdk.Int64 = lambda _: 0"),
    ],
)
@pytest.mark.parametrize("inside_outer_arm", [False, True])
def test_phase_dispatch_preserves_rebound_selector_inputs(
    boundary_expr: str, write: str, inside_outer_arm: bool
) -> None:
    # The outer selector reads only the PID. Its left child's dynamic boundary
    # and PID must still have their original values to remove either leaf.
    owner = ForEachProgramID("pid_shared")
    for i in range(4):
        owner.case_bodies.append(ast.parse(f"seen.append({i})").body)
        if i < 3:
            end = boundary_expr if i == 0 else str(i + 1)
            owner.case_branches.append(
                create(
                    ast.If,
                    test=ast.parse(f"pid_shared < {end}", mode="eval").body,
                    body=[ast.Pass()],
                    orelse=[],
                )
            )
    dispatch = owner.codegen_dispatch()
    (effect,) = ast.parse(write).body
    if inside_outer_arm:
        outer = owner.case_branches[1]
        outer.body.insert(0, effect)
    else:
        dispatch.insert(0, effect)
    scoped = specialize_phase_cases(owner, dispatch, 0, 1)
    for statements in (dispatch, scoped):
        limit = [1]
        sdk = SimpleNamespace(Int64=int)
        scope = {
            "pid_shared": 0,
            "boundary": 1,
            "cutlass": sdk,
            "sdk": sdk,
            "limit": limit,
            "alias": limit,
            "seen": [],
        }
        tree = ast.Module(body=statements, type_ignores=[])
        exec(compile(ast.fix_missing_locations(tree), "<root selector>", "exec"), scope)
        assert scope["seen"] == [1]
