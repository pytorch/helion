"""Triton's software pipeliner (``num_stages``) for gluon kernels: loads feeding
a dot issue ahead as ``cp.async`` copies into shared-memory rings, in a ring
kernel's body partition too (``GluonBackend.finalize_kernel_def``)."""

from __future__ import annotations

import ast
import collections
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import cast

from ..ast_extension import clone_ast
from ..ast_extension import create
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..ast_read_writes import ast_rename
from ..cross_loop_codegen import _all_names
from ..cross_loop_codegen import _argument_allocations
from ..cross_loop_codegen import _effect_kind
from ..cross_loop_codegen import _is_tl_call
from ..cross_loop_codegen import _may_write
from ..cross_loop_codegen import _named_allocations
from ..cross_loop_codegen import _stored_names
from . import gluon_ops
from .gluon_layout import range_bounds
from .gluon_ring import _MODULES
from .gluon_ring import ALIAS
from .gluon_ring import _arg
from .gluon_ring import _arg_index
from .gluon_ring import _call
from .gluon_ring import _name
from .gluon_ring import _names
from .gluon_ring import _slice

if TYPE_CHECKING:
    from collections.abc import Collection

    from ..device_function import DeviceFunction


def software_pipeline(
    body: list[ast.stmt], device_function: DeviceFunction
) -> list[ast.stmt]:
    """Triton's software pipeliner (``num_stages``), in a ring kernel's body
    partition too: in an innermost ``range`` loop, the loads feeding a dot (every
    load, under the loop's own ``num_stages``) whose addresses need no value the
    loop loads or carries issue ``num_stages - 1`` steps ahead as ``cp.async``
    copies into shared-memory rings (``gluon_ops.pipe_*``).  A tcgen05 dot of two
    such loads reads their slots, its MMAs async (a step's free its slots);
    other MMAs reading memory in place (a slot, a TMA copy) complete
    synchronously (``_synchronous``).  A load of memory the loop writes stays
    in program order."""
    num_stages = device_function.config.num_stages
    allocations = _argument_allocations(device_function)
    loops = iter(range(1 << 30))
    stores = collections.Counter(
        n.id
        for s in body
        for n in ast.walk(s)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    )

    class Pipeline(ast.NodeTransformer):
        def visit_For(self, node: ast.For) -> ast.AST | list[ast.stmt]:
            self.generic_visit(node)
            hoisted = _hoisted(node, stores, next(loops))
            ahead, behind = _carried_in_tmem(node, stores)
            pipelined = _pipelined(node, num_stages, next(loops), allocations)
            return [*hoisted, *ahead, *(pipelined or [node]), *behind]

    body = Pipeline().visit(create(ast.Module, body=body, type_ignores=[])).body
    # The TMA copies MMAs read in place (``gluon_ops._descriptor_load``).
    copies = {
        name
        for s in body
        for st in ast.walk(s)
        if isinstance(st, ast.Assign)
        and _is_tl_call(st.value, frozenset(["load_tensor_descriptor"]))
        and any(k.arg == "TRANSPOSED" for k in st.value.keywords)
        for name in _stored_names([st])
    }
    _synchronous(body, copies)
    return body


def _synchronous(
    body: list[ast.stmt], names: Collection[str], released: Collection[ast.AST] = ()
) -> None:
    """MMAs reading ``names`` in place complete synchronously (as Triton's
    unpipelined MMAs do), but ``released`` (the pipeline frees their slots
    after them): nothing else orders that memory's next write (the next step's
    copy) after an asynchronous MMA's read."""
    sync = gluon_ops.tmem_dot.arg_names.index("SYNC")  # pyrefly: ignore [missing-attribute]
    for call in (c for s in body for c in ast.walk(s)):
        if (
            _is_tl_call(call, frozenset(["tmem_dot"]), _MODULES)
            and call not in released
            and (_names(call.args[0]) | _names(call.args[1])) & set(names)
        ):
            call.args[sync] = ast.Constant(True)


def _carried_in_tmem(
    loop: ast.For, stores: collections.Counter[str]
) -> tuple[list[ast.stmt], list[ast.stmt]]:
    """Triton's HoistTMEMAlloc: an accumulator ``loop`` carries in registers only
    to rescale it into its next MMA (``acc_get`` after the dot, read before its
    ``acc_put``) stays in tensor memory: the rescale reads it there; the loop's
    ends store it once and read it once (statements ahead of and behind it)."""
    ahead: list[ast.stmt] = []
    behind: list[ast.stmt] = []
    for get in [st for st in loop.body if isinstance(st, ast.Assign)]:
        value, target = get.value, get.targets[0]
        if not (
            _is_tl_call(value, frozenset(["acc_get"]), _MODULES)
            and isinstance(target, ast.Name)
            and stores[target.id] > 1  # carried: also assigned ahead of the loop
            and isinstance(tm := cast("ast.Call", value).args[0], ast.Name)
        ):
            continue
        body = loop.body
        # The carried value and its copies at the top of the body.
        chain, copies = {target.id}, []
        for st in body[: body.index(get)]:
            if (
                isinstance(st, ast.Assign)
                and isinstance(st.value, ast.Name)
                and st.value.id in chain
            ):
                chain.add(cast("ast.Name", st.targets[0]).id)
                copies.append(st)
        uses = [st for st in body if st not in copies and _names(st) & chain]
        touches = [st for st in body if tm.id in _names(st)]
        puts = [
            st
            for st in touches
            if isinstance(st, ast.Expr)
            and _is_tl_call(st.value, frozenset(["acc_put"]), _MODULES)
        ]
        if (
            len(puts) != 1
            or len(touches) != 3  # its put, MMA and read only
            or not uses
            or uses[-1] is not get
            or any(body.index(u) > body.index(puts[0]) for u in uses[:-1])
            or len(read := _names(uses[0]) & chain) != 1
            or sum(1 for st in body if target.id in _stored_names([st])) != 1
        ):
            continue
        body.remove(get)
        (name,) = read
        body.insert(
            body.index(uses[0]), statement_from_string(f"{name} = {{x}}", x=value)
        )
        ahead.append(
            statement_from_string(f"{gluon_ops.ALIAS}.acc_put({tm.id}, {target.id})")
        )
        behind.append(get)
    return ahead, behind


def _hoisted(loop: ast.For, stores: collections.Counter[str], n: int) -> list[ast.stmt]:
    """Triton's LICM of a local_alloc (Gluon runs none; LLVM hoists register math
    itself): an MMA operand ``loop`` does not vary is staged once ahead of it, with
    the pure assignments it is computed from (each name assigned once in the
    kernel, not read before)."""
    varying, read = _stored_names([loop]), set()
    for st in loop.body:
        target = st.targets[0] if isinstance(st, ast.Assign) else None
        if (
            isinstance(target, ast.Name)
            and len(cast("ast.Assign", st).targets) == 1
            and stores[target.id] == 1
            and target.id not in read
            and _pure(st)
            and not _names(cast("ast.Assign", st).value) & varying
        ):
            varying.discard(target.id)
        read |= _names(st)
    # (Without the completion an in-loop restage waits on: none once hoisted.)
    staged = [
        (
            c,
            create(
                ast.Call, func=c.func, args=c.args[: _arg_index(c, "done")], keywords=[]
            ),
        )
        for s in loop.body
        for c in ast.walk(s)
        if _is_tl_call(c, frozenset(["stage"]), _MODULES)
        and not set().union(*map(_names, c.args[: _arg_index(c, "done")])) & varying
    ]
    hoisted = _slice(loop.body, set().union(*(_names(op) for _, op in staged)))
    loop.body = [st for st in loop.body if st not in hoisted]
    for k, (c, operand) in enumerate(staged):
        name = f"_helion_staged{n}_{k}"
        hoisted.append(statement_from_string(f"{name} = {{x}}", x=operand))
        loop.body = [_replaced(s, c, _name(name)) for s in loop.body]
    return hoisted


def _pipelined(
    loop: ast.For, default: int, n: int, allocations: dict[str, int]
) -> list[ast.stmt] | None:
    call, iv = loop.iter, loop.target
    if (
        not _is_tl_call(call, frozenset(["range"]))
        or not isinstance(iv, ast.Name)
        or loop.orelse
        or any(
            isinstance(x, ast.For | ast.While) for s in loop.body for x in ast.walk(s)
        )
    ):
        return None  # Triton pipelines innermost loops only
    stages = next((k.value for k in call.keywords if k.arg == "num_stages"), None)
    if stages is not None and not isinstance(stages, ast.Constant):
        return None
    S = default if stages is None else cast("int", stages.value)
    # Triton unrolls (``loop_unroll_factor``) before it pipelines: its stages are
    # unrolled steps.
    unroll = next(
        (k.value for k in call.keywords if k.arg == "loop_unroll_factor"), None
    )
    S = (S - 1) * cast(
        "int", unroll.value if isinstance(unroll, ast.Constant) else 1
    ) + 1
    stored = _stored_names(loop.body)
    written = {
        a
        for s in loop.body
        for c in ast.walk(s)
        if isinstance(c, ast.Call) and _may_write(c)
        for a in _named_allocations(c, allocations)
    }
    pipes: list[tuple[ast.Assign, ast.Call, list[ast.stmt]]] = []
    for i, st in enumerate(loop.body):
        if S < 2 or not isinstance(st, ast.Assign):
            continue
        for load in ast.walk(st.value):
            if not _is_tl_call(load, frozenset(["load"])) or len(load.args) > 3:
                continue
            other = _load_args(load)[2]
            if (
                any(k.arg == "volatile" for k in load.keywords)
                or _named_allocations(load, allocations) & written
                or not (isinstance(other, ast.Constant) and other.value in (None, 0))
            ):
                continue  # (cp.async fills masked lanes with zeros)
            if stages is None and _plan(load) is None:
                continue  # feeds no dot (``gluon_ops.dot_input``)
            reads = _names(load)
            needs = _defining(loop.body[:i], reads)
            if needs is not None and not _inputs(reads, needs, iv.id) & stored:
                pipes.append((st, load, needs))
    if not pipes:
        return None
    G = gluon_ops.ALIAS
    first, stop, stride, index, freed, *rings = (
        f"_helion_pipe{n}_{x}" for x in ["lb", "ub", "st", "i", "e", *range(len(pipes))]
    )
    # The MMA operands read in place: their slots in ``stage``'s layout.
    operand = {
        ring: ast.Constant(_plan(load)[0])  # pyrefly: ignore [unsupported-operation]
        for ring, (st, load, _) in zip(rings, pipes, strict=True)
        if (_plan(load) or [None])[0] is not None
        and (
            st.value is load
            or (
                _is_tl_call(st.value, frozenset(["broadcast_to"]))
                and st.value.args[0] is load  # pyrefly: ignore [missing-attribute]
            )
        )
    }
    names = {
        cast("ast.Name", st.targets[0]).id: ring
        for ring, (st, _, _) in zip(rings, pipes, strict=True)
        if ring in operand
    }
    dots = _slot_dots(loop.body, names)
    # Another MMA reading a slot in place would read it after its refill.
    _synchronous(loop.body, names.keys(), dots)
    mma = any(
        _is_tl_call(c, frozenset(["tmem_dot", "tmem_dot_scaled"]), _MODULES)
        and _names(c) & names.keys()
        for s in loop.body
        for c in ast.walk(s)
    )
    asynchronous = {names[a.args[0].id] for d in dots for a in d.args[:2]}  # pyrefly: ignore [missing-attribute]
    needed = [s for s in loop.body if any(s in ss for *_, ss in pipes)]
    # A slot read once its step's MMAs or loads complete is free then (Triton's
    # S - 1 buffers); an asynchronous MMA's frees one step later.
    slots = {ring: S if ring in asynchronous else S - 1 for ring in rings}
    # The slots a step reads and fills S - 1 steps ahead: indices the loop carries
    # and wraps (Triton's), no modulo per step; by (slots, first value).
    counters: dict[tuple[int, int], str] = {}

    def slot(ring: str, ahead: int) -> str:
        key = (slots[ring], ahead % slots[ring])
        return counters.setdefault(key, f"_helion_pipe{n}_s{len(counters)}")

    def issue(at: str, step: int | None, suffix: str, alloc: bool) -> list[ast.stmt]:
        """The copies of step number ``step`` (the loop's: None; ``iv`` = ``at``)."""
        renames = {v: v + suffix for v in {iv.id, *_stored_names(needed)}}
        out = [statement_from_string(f"{iv.id}{suffix} = {at}")]
        out += [ast_rename(cast("ast.stmt", clone_ast(s)), renames) for s in needed]
        for ring, (_, load, _) in zip(rings, pipes, strict=True):
            ptr, mask = (
                ast_rename(cast("ast.expr", clone_ast(a)), renames)
                for a in _load_args(load)[:2]
            )
            if alloc:
                out.append(statement_from_string(
                    f"{ring} = {G}.pipe_alloc({{p}}, {{m}}, {slots[ring]}, {{o}}, {ring in asynchronous})", p=ptr, m=mask,
                    o=operand.get(ring, ast.Constant(None)),
                ))  # fmt: skip
            to = slot(ring, S - 1) if step is None else step % slots[ring]
            put = f"{G}.pipe_put({ring}, {{p}}, {{m}}, {iv.id}{suffix} < {stop}, {to})"
            out.append(statement_from_string(put, p=ptr, m=mask))
            cast("ast.Call", cast("ast.Expr", out[-1]).value).keywords = [
                k for k in load.keywords if k.arg in _CACHE_KEYS
            ]
        return [*out, statement_from_string(f"{G}.pipe_commit()")]

    for ring, (st, load, _) in zip(rings, pipes, strict=True):
        take = _call(
            "pipe_take",
            [_name(ring), _name(slot(ring, 0)), *_load_args(load)],  # pyrefly: ignore [bad-argument-type]
        )
        take.func = cast("ast.expr", expr_from_string(f"{G}.pipe_take"))
        take.keywords = [k for k in load.keywords if k.arg in _CACHE_KEYS]
        st.value = cast("ast.expr", _replaced(st.value, load, take))
    for d in dots:
        # Asynchronous, the slots freed on ``freed`` per step.
        for name, value in (
            ("done", _name(freed)),
            ("count", _name(index)),
            ("S", ast.Constant(S)),
            ("SYNC", ast.Constant(False)),
        ):
            d.args[_arg_index(d, name)] = value
    steps = f"({stop} - {first} + {stride} - 1) // {stride}"
    loop.body = [
        *([statement_from_string(f"{index} = ({iv.id} - {first}) // {stride}")] if dots else []),
        statement_from_string(f"{G}.pipe_wait({S - 2}, {mma})"),
        *loop.body,
        *([statement_from_string(f"{G}.pipe_release({freed}, {index}, {S})")] if dots else []),
        *issue(f"{iv.id} + {S - 1} * {stride}", None, f"_pipe{n}", False),
        *(statement_from_string(f"{c} = tl.where({c} == {k - 1}, 0, {c} + 1)") for (k, _), c in counters.items()),
    ]  # fmt: skip
    lb, ub, step = range_bounds(call.args, ast.Constant(0), ast.Constant(1))
    return [
        *(statement_from_string(f"{x} = {{v}}", v=v) for x, v in ((first, lb), (stop, ub), (stride, step))),
        *([statement_from_string(f"{freed} = {ALIAS}.ring_barriers({S}, {len(dots)})")] if dots else []),
        *(st for k in range(S - 1) for st in issue(f"{first} + {k} * {stride}", k, f"_pipe{n}_{k}", k == 0)),
        *(statement_from_string(f"{c} = {start}") for (_, start), c in counters.items()),
        loop,
        statement_from_string(f"{G}.pipe_wait(0, False)"),
        *([statement_from_string(f"{G}.pipe_release({freed}, {steps}, {S}, True)")] if dots else []),
    ]  # fmt: skip


_CACHE_KEYS = ("cache_modifier", "eviction_policy")


_Node = TypeVar("_Node", bound=ast.AST)


def _replaced(tree: _Node, old: ast.AST, new: ast.expr) -> _Node:
    class Replace(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            return new if node is old else self.generic_visit(node)

    return cast("_Node", Replace().visit(tree))


def _load_args(load: ast.Call) -> list[ast.expr]:
    """A ``tl.load``'s pointer, mask and other (None if not given), positional
    or keyword."""
    names = ["pointer", "mask", "other"]
    given = dict(zip(names, load.args, strict=False))
    given.update({k.arg: k.value for k in load.keywords if k.arg in names})  # pyrefly: ignore [no-matching-overload]
    return [given.get(name, ast.Constant(None)) for name in names]


def _plan(load: ast.AST) -> tuple[bool | None] | None:
    """A load's ``gluon_ops.dot_input`` (``plan_load`` put it on the load's code)."""
    return cast("tuple[bool | None] | None", getattr(load, "_plan", None))


def _slot_dots(body: list[ast.stmt], names: Collection[str]) -> list[ast.Call]:
    """The asynchronous tcgen05 dots (``SYNC`` false: into an accumulator
    resident through the loop) whose two operands are pipelined loads read in
    place (``names``) only such dots read."""
    calls = [n for s in body for n in ast.walk(s) if isinstance(n, ast.Call)]
    # The calls each name is an argument of.
    uses = {
        v: {
            id(c)
            for c in calls
            for a in c.args
            if isinstance(a, ast.Name) and a.id == v
        }
        for v in names
    }
    dots = [
        c
        for c in calls
        if _is_tl_call(c, frozenset(["tmem_dot"]), _MODULES)
        and not cast("ast.Constant", _arg(c, "SYNC")).value
        and all(
            _is_tl_call(a, frozenset(["stage"]), _MODULES)
            and isinstance(a.args[0], ast.Name)  # pyrefly: ignore [missing-attribute]
            and a.args[0].id in names  # pyrefly: ignore [missing-attribute]
            for a in c.args[:2]
        )
    ]
    while True:
        staged = {id(a) for c in dots for a in c.args[:2]}
        ok = {v for v in names if uses[v] <= staged}
        kept = [c for c in dots if all(a.args[0].id in ok for a in c.args[:2])]  # pyrefly: ignore [missing-attribute]
        if len(kept) == len(dots):
            return dots
        dots = kept


def _pure(statement: ast.stmt) -> bool:
    """Whether ``statement`` assigns pure calls."""
    return isinstance(statement, ast.Assign) and all(
        _effect_kind(c) == "pure"
        for c in ast.walk(statement.value)
        if isinstance(c, ast.Call)
    )


def _defining(body: list[ast.stmt], names: set[str]) -> list[ast.stmt] | None:
    """The pure assignments of ``body`` the values of ``names`` depend on, in order;
    None if one of them is not pure."""
    kept = _slice(body, names)
    return kept if all(map(_pure, kept)) else None


def _inputs(names: set[str], statements: list[ast.stmt], var: str) -> set[str]:
    """What ``names`` and their defining ``statements`` (``_defining``) read from
    outside them, but the loop variable ``var``."""
    return (names | _all_names(statements)) - _stored_names(statements) - {var}
