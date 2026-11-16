"""Pin who waits on the attention pipelines' K and V publication barriers.

A parity wait cannot tell a phase from the one two phases later. A barrier
several warps wait on is therefore safe only when its next phase needs every
one of them (the raw ring's releases, the four score releases before the next
QK, the softmax's 128-thread barriers before the next PV). The converters
publish the next K or V as soon as the MMA reading the previous one commits,
which takes the issuing warp alone: every other warp that waited there could
see two phases complete between two polls and wait forever, which hung the
kernel (``test_cute_paged_attention_handoff_stress.py``). So exactly one warp
may wait on those barriers: the issuer of the MMA that consumes them.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from helion._testing.cute_paged_attention_handoff import LaunchWatchdog

pytestmark = pytest.mark.cpu_only
CUTE = Path(__file__).parents[1] / "helion/_compiler/cute"
WAITS = ("mbar_wait", "mbar_wait_sleep", "_wait_non_mma")
PUBLISHED = ("key_ready", "value_ready")


def _publication_waits(module: str, function: str) -> dict[str, set[str]]:
    """Barrier name -> the ``warp == N`` guards its waits run under."""
    return _waits_in((CUTE / module).read_text(), module, function)


def _waits_in(source: str, module: str, function: str) -> dict[str, set[str]]:
    """``_publication_waits`` of ``function`` in a module's ``source``.

    A role helper of the same module runs under the guards of its call. The
    waits are found by barrier name, so a helper must receive each
    publication barrier under the barrier's own name: one passed under
    another name, or as part of an expression, would be waited on unseen.
    """
    tree = ast.parse(source)
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    found: dict[str, set[str]] = {}

    def visit(node: ast.AST, guards: tuple[str, ...]) -> None:
        if isinstance(node, ast.If):
            visit(node.test, guards)
            for child in node.body:
                visit(child, (*guards, ast.unparse(node.test)))
            for child in node.orelse:
                visit(child, guards)
            return
        if (
            isinstance(node, ast.Call)
            and ast.unparse(node.func).split(".")[-1] in WAITS
        ):
            barrier = ast.unparse(node.args[0]).split(" ")[0]
            if barrier in PUBLISHED:
                owners = {guard for guard in guards if guard.startswith("warp == ")}
                assert len(owners) == 1, (module, barrier, guards)
                found.setdefault(barrier, set()).update(owners)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in functions
        ):
            helper = functions[node.func.id]
            bound = dict(
                zip([arg.arg for arg in helper.args.args], node.args, strict=False)
            )
            bound.update({keyword.arg: keyword.value for keyword in node.keywords})
            for name in PUBLISHED:
                if name in bound:
                    assert ast.unparse(bound[name]) == name, (module, helper.name)
            for argument in (*node.args, *(word.value for word in node.keywords)):
                for name in PUBLISHED:
                    if any(
                        isinstance(part, ast.Name) and part.id == name
                        for part in ast.walk(argument)
                    ):
                        assert bound.get(name) is argument, (
                            module,
                            helper.name,
                            ast.unparse(argument),
                        )
            for child in helper.body:
                visit(child, guards)
        for child in ast.iter_child_nodes(node):
            visit(child, guards)

    visit(functions[function], ())
    return found


@pytest.mark.parametrize(
    "module,function,expected",
    [
        # The per-key schedules: warp 9 issues QK, compute warp 0 issues PV.
        (
            "_paged_attention_partial_pipeline.py",
            "paged_attention_partial_body",
            {"key_ready": {"warp == 9"}, "value_ready": {"warp == 0"}},
        ),
        # The reduced schedule: compute warp 0 issues both.
        (
            "_paged_attention_pipeline.py",
            "paged_attention_body",
            {"key_ready": {"warp == 0"}, "value_ready": {"warp == 0"}},
        ),
        # The paired schedule: dedicated QK and PV issuer warps.
        (
            "_paged_attention_paired_pipeline.py",
            "paged_attention_paired_body",
            {"key_ready": {"warp == 16"}, "value_ready": {"warp == 18"}},
        ),
    ],
)
def test_only_the_issuing_warp_waits_for_published_operands(module, function, expected):
    assert _publication_waits(module, function) == expected


_HELPER_CALL = """
def _issue_pv({parameter}, step):
    mbar_wait({parameter}, step & 1)

def body(warp, value_ready, step):
    if warp == 18:
        _issue_pv({argument}, step=step)
"""


@pytest.mark.parametrize(
    "parameter,argument",
    [
        # Positional and by keyword under another name, and inside an
        # expression under its own.
        ("ready", "value_ready"),
        ("ready", "ready=value_ready"),
        ("ready", "value_ready + 1"),
        ("value_ready", "value_ready + 1"),
    ],
)
def test_a_publication_barrier_reaches_a_helper_under_its_own_name(parameter, argument):
    """Otherwise the helper's waits on it, which the audit finds by barrier
    name, would be missed, and with them a second warp that waits there."""
    own = _HELPER_CALL.format(parameter="value_ready", argument="value_ready")
    assert _waits_in(own, "synthetic", "body") == {"value_ready": {"warp == 18"}}
    renamed = _HELPER_CALL.format(parameter=parameter, argument=argument)
    with pytest.raises(AssertionError):
        _waits_in(renamed, "synthetic", "body")


def test_stress_watchdog_gives_the_first_call_its_own_bound() -> None:
    # The stress test's first call also JIT-compiles the kernel. A compile
    # slower than a launch may stall is not a hang; a first call that never
    # returns still is one, and so is every later stall.
    now = [0.0]
    watchdog = LaunchWatchdog(first_call=300.0, stall=30.0, clock=lambda: now[0])
    now[0] = 100.0
    assert watchdog.expired() is None
    now[0] = 300.5
    assert watchdog.expired() == (
        "HANG: no progress for 300.0s in the first call, which also compiles"
    )
    # The first call returned after a 100 s compile: launches are bounded
    # by the stall from here on.
    now[0] = 100.0
    watchdog.progress(0)
    now[0] = 129.0
    assert watchdog.expired() is None
    now[0] = 130.5
    assert watchdog.expired() == "HANG: no progress for 30.0s after 0 launches"
    watchdog.progress(3)
    now[0] = 160.0
    assert watchdog.expired() is None
    now[0] = 161.0
    assert watchdog.expired() == "HANG: no progress for 30.0s after 3 launches"
