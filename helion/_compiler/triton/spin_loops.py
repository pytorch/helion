"""Keep Triton's software pipeliner from moving loads across a guarded wait.

Triton pipelines a ``tl.range`` loop whose body holds no loop of its own: it
issues the next iterations' loads (as asynchronous copies) ahead of the body,
following only data dependencies.  A scheduler wait spinning in a ``while``
loop directly in the body makes the loop an outer loop, which Triton leaves
alone; one nested under an ``if`` (or inside a helper) does not, so the loads
the wait orders could be issued before it.  Such loops run unpipelined.  An
inner ``for`` loop does not protect its parent: one with a single iteration
(or an empty static range) folds away before Triton pipelines.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import TypeGuard

from ..ast_extension import create

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Mapping

    from ..device_function import DeviceFunction

    Helpers = Mapping[str, tuple[tuple[ast.stmt, ...], bool]]


def _is_tl_call(node: ast.AST, name: str) -> TypeGuard[ast.Call]:
    # cross_loop_codegen's is out of reach: importing it here (the Triton
    # backend imports this module) is circular.
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "tl"
    )


def _named(node: ast.AST, names: frozenset[str]) -> bool:
    return any(isinstance(n, ast.Name) and n.id in names for n in ast.walk(node))


# Runtime wait helpers whose spin is a top-level ``while``: called directly in
# a loop's body, the inlined spin keeps Triton off the loop.
_RUNTIME_SPINS = frozenset(
    {
        "helion_dist_utils._wait_and_consume_signal",
        "helion_dist_utils._wait_at_least",
    }
)


def _runtime_spin(node: ast.AST) -> bool:
    """A call of a runtime wait helper, bound to a name or not (remote-copy
    receive waits and remote barriers emit theirs as bare statements)."""
    value = node.value if isinstance(node, (ast.Assign, ast.Expr)) else None
    return isinstance(value, ast.Call) and ast.unparse(value.func) in _RUNTIME_SPINS


def _is_wait(node: ast.AST, waits: frozenset[str]) -> bool:
    """A spin on a wait's value, a wait bound from a call (the callee spins),
    or a runtime wait helper's call.  A wait computed by inline assembly is a
    spin's first load, a sync or fence, or a token spin whose zero result the
    task's addresses add: the pipeliner cannot issue those loads before it."""
    if isinstance(node, ast.While):
        return _named(node.test, waits)
    return _runtime_spin(node) or (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in waits
        and isinstance(node.value, ast.Call)
        and not _is_tl_call(node.value, "inline_asm_elementwise")
    )


def _helper(
    statement: ast.stmt, helpers: Helpers
) -> tuple[tuple[ast.stmt, ...], bool] | None:
    if (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
    ):
        return helpers.get(statement.value.func.id)
    return None


def _top_level(
    body: list[ast.stmt] | tuple[ast.stmt, ...], helpers: Helpers
) -> Iterator[ast.stmt]:
    """The statements that land directly in a loop's body once Triton inlines
    its helpers."""
    for statement in body:
        helper = _helper(statement, helpers)
        if helper is not None and not helper[1]:
            yield from _top_level(helper[0], helpers)
        else:
            yield statement


def _holds_wait(
    body: list[ast.stmt] | tuple[ast.stmt, ...],
    helpers: Helpers,
    waits: frozenset[str],
    seen: frozenset[str] = frozenset(),
) -> bool:
    for statement in body:
        for node in ast.walk(statement):
            if _is_wait(node, waits):
                return True
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in helpers
                and node.func.id not in seen
                and _holds_wait(
                    helpers[node.func.id][0], helpers, waits, seen | {node.func.id}
                )
            ):
                return True
    return False


def pin_num_stages_one(call: ast.Call) -> None:
    """Run the ``tl.range`` loop ``call`` builds unpipelined."""
    call.keywords = [
        *(k for k in call.keywords if k.arg != "num_stages"),
        create(ast.keyword, arg="num_stages", value=ast.Constant(1)),
    ]


def _stages(call: ast.Call, default: int) -> int | None:
    for k in call.keywords:
        if k.arg == "num_stages":
            value = k.value
            if isinstance(value, ast.Constant) and isinstance(value.value, int):
                return value.value
            return None
    return default


def pin_spin_loops(
    body: list[ast.stmt],
    helpers: Helpers,
    waits: frozenset[str],
    num_stages: int,
) -> list[ast.stmt]:
    """Pin ``num_stages=1`` on each ``tl.range`` loop of ``body`` Triton would
    pipeline across a wait it holds.  ``helpers``: outlined helper bodies and
    whether each is ``noinline``, by name; ``waits``: the names scheduler
    waits assign; ``num_stages``: the kernel's, a loop's default."""
    for statement in body:
        for node in ast.walk(statement):
            if not (isinstance(node, ast.For) and _is_tl_call(node.iter, "range")):
                continue
            call = node.iter
            stages = _stages(call, num_stages)
            if stages is not None and stages <= 1:
                continue
            if any(
                k.arg == "warp_specialize"
                and not (isinstance(k.value, ast.Constant) and not k.value.value)
                for k in call.keywords
            ):
                # Warp specialization rejects a loop in its body: its waits
                # are token spins the task's addresses depend on.
                continue
            if any(
                (isinstance(s, ast.While) and _is_wait(s, waits)) or _runtime_spin(s)
                for s in _top_level(node.body, helpers)
            ):
                continue
            if _holds_wait(node.body, helpers, waits):
                pin_num_stages_one(call)
    return body


def pin_spin_loops_of(
    body: list[ast.stmt], device_function: DeviceFunction
) -> list[ast.stmt]:
    """``pin_spin_loops`` over a kernel or helper body of ``device_function``.
    ``body`` has its variables renamed; the helper bodies registered on
    ``device_function`` (which the pass looks through) do not."""
    renames = {k: v[0] for k, v in device_function._variable_renames.items()}
    waits = frozenset(
        name
        for wait, role in device_function.sync_roles.items()
        if role.kind == "wait"
        for name in (wait, renames.get(wait, wait))
    )
    helpers = {
        name: (helper_body, noinline)
        for name, _, helper_body, noinline in device_function.triton_outlined_helpers
    }
    if not waits and not any(
        _runtime_spin(node)
        for statements in (body, *(b for b, _ in helpers.values()))
        for statement in statements
        for node in ast.walk(statement)
    ):
        return body
    return pin_spin_loops(body, helpers, waits, int(device_function.config.num_stages))
