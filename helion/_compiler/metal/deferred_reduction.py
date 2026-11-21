"""Sum a loop's cross-thread partials once, after the loop.

A contraction whose K tile spreads over a threadgroup's threads (a GEMV on
the scalar contraction) sums each tile across the threads, then adds the sum
to a loop-carried accumulator::

    for offset_k in range(0, K, BLOCK_K):
        acc_copy = acc
        part = float(0)
        for lane in range(L):
            part = part + x * w
        tile_sum = helion_red.group_sum(
            site, scratch, static_cast(float, part), tid, ...
        )
        acc = static_cast(float, acc_copy + tile_sum)

Every iteration pays the cross-thread sum (shuffles, and for a threadgroup
span two barriers) for one add.  When the loop reads the accumulator only in
that add (and in the copies that feed it), each thread can keep its partial
across the loop and the threads sum once after it::

    tile_sum_carry = float(0)
    for offset_k in range(0, K, BLOCK_K):
        acc_copy = acc
        part = float(0)
        for lane in range(L):
            part = part + x * w
        tile_sum_carry = tile_sum_carry + static_cast(float, part)
    tile_sum = helion_red.group_sum(
        site, scratch, static_cast(float, tile_sum_carry), tid, ...
    )
    acc = static_cast(float, acc + tile_sum)

Several sums may add into one accumulator in turn, as two dots into one
carry do (a packed weight's low and high codes)::

    acc_1 = static_cast(float, acc_copy + sum_lo)
    acc = static_cast(float, acc_1 + sum_hi)

The accumulator is then a chain of plain copies and these adds, in the
loop body's order, from a read of the name the loop carries to the
statement that assigns it, each link reading the name the link before it
assigned; each add becomes a copy and each sum is added to the carried name
after the loop.  A sum may reach its add through a copy or a float cast of
its result (a ``.sum()``'s ``sum_1 = static_cast(float, lane_sum)``), which
the rewrite drops.

The sum is the same up to float reassociation, the order a split K changes
anyway.  A loop whose bounds don't show that it runs adds its sums only
when it ran (``if (begin < end)``), so a loop of no iterations leaves the
accumulator as it was, a ``-0.0`` too; the condition is the threadgroup's
own, as the loop's trip count is.  The rewrite needs each sum a float one, unconditional in the loop
body (every thread reaches it each iteration, as its barriers already
require), its other arguments unassigned in the loop, and its result read
only by its add; and the chain's names read and assigned only by its links,
and, but for the carried name, nowhere outside the loop: a name off the
chain, or one read after the loop, would see each tile's sum where the
rewrite adds them all to the carried name at the end.
"""

from __future__ import annotations

import ast
import collections
import copy
import dataclasses
from typing import Iterable
from typing import Iterator

GROUP_SUM = "group_sum"
_NS = "helion_red"


def defer_group_sums(stmts: list[ast.stmt], params: Iterable[str]) -> list[ast.stmt]:
    """``stmts`` with every deferrable loop sum moved after its loop, the
    innermost loops first.  ``params`` are the kernel's parameter names,
    which a carry's name must not take, as it must not take any name of
    ``stmts``."""
    root = ast.Module(body=list(stmts), type_ignores=[])
    taken = {*params, *(n.id for n in ast.walk(root) if isinstance(n, ast.Name))}
    for block, loop in list(_loops(root.body)):
        before, after = _defer(loop, root, taken)
        at = next(i for i, s in enumerate(block) if s is loop)
        block[at : at + 1] = [*before, loop, *after]
    return root.body


def _loops(block: list[ast.stmt]) -> Iterator[tuple[list[ast.stmt], ast.For]]:
    """Each loop under ``block`` with the statement list holding it, every
    loop after the loops inside it."""
    for stmt in block:
        for field in ("body", "orelse"):
            inner = getattr(stmt, field, None)
            if isinstance(inner, list) and inner and isinstance(inner[0], ast.stmt):
                yield from _loops(inner)
        if isinstance(stmt, ast.For):
            yield block, stmt


def _uncast(node: ast.expr) -> ast.expr:
    """``node`` without ``static_cast(float, ...)`` wrappers."""
    while _is_float_cast(node):
        assert isinstance(node, ast.Call)
        node = node.args[1]
    return node


def _is_float_cast(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "static_cast"
        and len(node.args) == 2
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "float"
    )


def _group_sum(stmt: ast.stmt) -> tuple[str, ast.Call] | None:
    """(target, call) of ``target = helion_red.group_sum(...)``."""
    target = _target(stmt)
    if target is None:
        return None
    assert isinstance(stmt, ast.Assign)
    call = stmt.value
    if (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == GROUP_SUM
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == _NS
        and len(call.args) >= 3
    ):
        return target, call
    return None


def _target(stmt: ast.stmt) -> str | None:
    """The name ``stmt`` assigns, when it is ``name = value``."""
    if (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
    ):
        return stmt.targets[0].id
    return None


def _assigned(stmts: list[ast.stmt]) -> set[str]:
    return {
        n.id
        for s in stmts
        for n in ast.walk(s)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }


def _reads(node: ast.AST) -> list[str]:
    return [
        n.id
        for n in ast.walk(node)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    ]


def _alias(stmt: ast.stmt) -> tuple[str, str] | None:
    """(target, source) of a plain copy ``target = source``."""
    target = _target(stmt)
    if target is not None:
        assert isinstance(stmt, ast.Assign)
        if isinstance(stmt.value, ast.Name):
            return target, stmt.value.id
    return None


@dataclasses.dataclass
class _Sum:
    """A deferrable sum: its statement and result, its call, the statements
    that copy or cast its result on the way to its add, and the add's
    statement, target and other operand, the accumulator name it reads."""

    at: int
    result: str
    call: ast.Call
    passes: list[int]
    add: int
    target: str
    other: str


def _defer(
    loop: ast.For, root: ast.Module, taken: set[str]
) -> tuple[list[ast.stmt], list[ast.stmt]]:
    """Rewrite ``loop`` in place; return the statements to put before and
    after it.  ``root`` holds the kernel, whose reads outside the loop a
    name the rewrite changes must not have; ``taken`` the names in use,
    which each carry's name joins."""
    if not _ascending_range(loop):
        return [], []
    body = loop.body
    assigned = _assigned(body) | {
        n.id for n in ast.walk(loop.target) if isinstance(n, ast.Name)
    }
    inside = collections.Counter(n for s in body for n in _reads(s))
    everywhere = collections.Counter(_reads(root))

    def local(name: str) -> bool:
        """Whether every read of ``name`` lies in the loop body."""
        return everywhere[name] == inside[name]

    def only_use(name: str, at: int) -> int | None:
        """The one statement of the body that reads ``name``, if it comes
        after ``at``, the one that assigns it, and nothing else assigns it."""
        if any(name in _assigned([s]) for j, s in enumerate(body) if j != at):
            return None
        uses = [j for j, s in enumerate(body) if name in _reads(s)]
        if len(uses) != 1 or uses[0] <= at or not local(name):
            return None
        return uses[0]

    sums: list[_Sum] = []
    for i in range(len(body)):
        found = _group_sum(body[i])
        if found is None:
            continue
        result, call = found
        site, scratch, value, *rest = call.args
        if not _is_float_cast(value):
            # The carry is a float.
            continue
        if any(name in assigned for a in (site, scratch, *rest) for name in _reads(a)):
            continue
        # The add: target = [cast] (other + operand) or (operand + other),
        # the operand the result or a copy or float cast of it.
        operand, at, passes = result, i, []
        use = only_use(operand, at)
        while use is not None:
            add = _add(body[use], operand)
            if add is not None:
                sums.append(_Sum(i, result, call, passes, use, *add))
                break
            passed = _passed(body[use], operand)
            if passed is None:
                break
            passes.append(use)
            operand, at = passed, use
            use = only_use(operand, at)
    # The accumulators: from each name the loop carries (read before it is
    # assigned), the chain of copies and adds that reads it, then each
    # link's target in turn, until a link assigns it again.
    links: dict[str, list[tuple[int, str]]] = collections.defaultdict(list)
    for j, s in enumerate(body):
        alias = _alias(s)
        if alias is not None:
            links[alias[1]].append((j, alias[0]))
    by_add = {entry.add: entry for entry in sums}
    for entry in sums:
        links[entry.other].append((entry.add, entry.target))
    before: list[ast.stmt] = []
    after: list[ast.stmt] = []
    dropped: set[int] = set()
    for acc in sorted(_exposed(body) & assigned):
        chain = _chain(acc, links)
        if chain is None:
            continue
        mine = [by_add[j] for j, _ in chain if j in by_add]
        names = {acc, *(target for _, target in chain)}
        users = {
            j for j, s in enumerate(body) if names & ({*_reads(s)} | _assigned([s]))
        }
        if (
            not mine
            or users != {j for j, _ in chain}
            or not all(map(local, names - {acc}))
        ):
            continue
        for entry in mine:
            part = f"{entry.result}_carry"
            while part in taken:
                part += "_"
            taken.add(part)
            carry = ast.Name(part, ast.Load())
            hoisted = copy.deepcopy(entry.call)
            value = hoisted.args[2]
            hoisted.args[2] = _parse_expr("static_cast(float, p)", p=carry)
            before.append(_parse(f"{part} = float(0)"))
            body[entry.at] = _assign(
                part, _parse_expr("p + static_cast(float, v)", p=carry, v=value)
            )
            body[entry.add] = _assign(entry.target, ast.Name(entry.other, ast.Load()))
            dropped.update(entry.passes)
            after.extend(
                [
                    _assign(entry.result, hoisted),
                    _parse(
                        f"{acc} = static_cast(float, "
                        f"static_cast(float, {acc}) + {entry.result})"
                    ),
                ]
            )
    loop.body = [s for j, s in enumerate(body) if j not in dropped]
    if after and (ran := _ran(loop)) is not None:
        after = [ast.If(test=ran, body=after, orelse=[])]
    return before, after


def _ascending_range(loop: ast.For) -> bool:
    """Whether ``loop`` runs over ``range`` or ``tl.range`` with a step not a
    negative literal: the ascending loops whose trip test is ``begin < end``
    (``_ran``), as the walker emits them.  A fragment or register run loop
    has other arguments, and a descending ``hl.grid`` another test."""
    call = loop.iter
    if not isinstance(call, ast.Call) or ast.unparse(call.func) not in (
        "range",
        "tl.range",
    ):
        return False
    step = call.args[2] if len(call.args) > 2 else None
    return not (
        isinstance(step, ast.UnaryOp)
        and isinstance(step.op, ast.USub)
        and isinstance(step.operand, ast.Constant)
    )


def _ran(loop: ast.For) -> ast.expr | None:
    """``begin < end`` of a loop over ``range(begin, end, ...)`` or
    ``range(end)`` (``tl.range`` too), None when its bounds are constants it
    runs between."""
    assert isinstance(loop.iter, ast.Call)
    args = loop.iter.args
    begin, end = (ast.Constant(0), args[0]) if len(args) == 1 else args[:2]
    if (
        isinstance(begin, ast.Constant)
        and isinstance(end, ast.Constant)
        and isinstance(begin.value, int)
        and isinstance(end.value, int)
        and begin.value < end.value
    ):
        return None
    return ast.Compare(
        left=copy.deepcopy(begin), ops=[ast.Lt()], comparators=[copy.deepcopy(end)]
    )


def _passed(stmt: ast.stmt, name: str) -> str | None:
    """The target of ``target = name`` or ``target = static_cast(float,
    name)``."""
    target = _target(stmt)
    if target is None:
        return None
    assert isinstance(stmt, ast.Assign)
    value = _uncast(stmt.value)
    return target if isinstance(value, ast.Name) and value.id == name else None


def _chain(
    acc: str, links: dict[str, list[tuple[int, str]]]
) -> list[tuple[int, str]] | None:
    """The links (statement, target) from a read of ``acc`` to the link that
    assigns it, in the body's order, each the one link that reads the name
    the link before it assigned; None when they don't form one."""
    chain: list[tuple[int, str]] = []
    seen = {acc}
    name, last = acc, -1
    while True:
        nxt = links.get(name, [])
        if len(nxt) != 1:
            return None
        ((at, target),) = nxt
        if at <= last:
            return None
        chain.append((at, target))
        if target == acc:
            return chain
        if target in seen:
            return None
        seen.add(target)
        name, last = target, at


def _add(stmt: ast.stmt, result: str) -> tuple[str, str] | None:
    """(target, other) of ``target = [cast] (other + result)`` or of
    ``target = [cast] (result + other)``, ``other`` a name."""
    if not (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
    ):
        return None
    add = _uncast(stmt.value)
    if not (isinstance(add, ast.BinOp) and isinstance(add.op, ast.Add)):
        return None
    left, right = _uncast(add.left), _uncast(add.right)
    if isinstance(left, ast.Name) and left.id == result:
        other = right
    elif isinstance(right, ast.Name) and right.id == result:
        other = left
    else:
        return None
    if not isinstance(other, ast.Name):
        return None
    return stmt.targets[0].id, other.id


def _exposed(stmts: list[ast.stmt]) -> set[str]:
    """Names some statement of ``stmts`` reads before ``stmts`` assigns them."""
    seen: set[str] = set()
    exposed: set[str] = set()
    for s in stmts:
        exposed |= {*_reads(s)} - seen
        seen |= _assigned([s])
    return exposed


def _assign(name: str, value: ast.expr) -> ast.stmt:
    return ast.Assign(targets=[ast.Name(name, ast.Store())], value=value, lineno=0)


def _parse(src: str) -> ast.stmt:
    return ast.parse(src).body[0]


def _parse_expr(src: str, **subs: ast.expr) -> ast.expr:
    tree = ast.parse(src, mode="eval").body

    class Sub(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.expr:
            return copy.deepcopy(subs[node.id]) if node.id in subs else node

    return Sub().visit(tree)
