"""The roots of a Metal kernel with several top-level loops.

The root dispatch (``ForEachProgramID``) sends each threadgroup to one root,
and every root runs in the same threadgroup: the per-axis maximum of the
thread layouts the roots' tiles take (``MetalBackend.rewrite_lowered_root``
records each).  A phased launch can instead lay the threadgroup out anew in
each root (``ticketed_phases.relayout_roots``).

A root laid out in fewer threads than its threadgroup runs its body on the
surplus threads too.  Along an axis its stores don't depend on, they
address the same elements as its own threads, so they would repeat its
stores, and an in-place update could land twice.  Along an axis it lays its
tiles out on, the tiles' masks mostly keep them from addressing anything
(``MetalBackend.thread_in_tile_mask_expr``), but a free ``hl.arange``'s
index takes no such mask.  :func:`guard_surplus_stores` keeps them from
writing.
"""

from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING
from typing import cast

from ..ast_extension import expr_from_string
from ..program_id import ForEachProgramID
from .lane_fold import is_store
from .lane_fold import names_in
from .lane_fold import pointer_base
from .lane_fold import tl_load
from .mpp_matmul import MARKER_PREFIX

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Mapping
    from collections.abc import Sequence

    from ..device_function import DeviceFunction


def case_bodies(pid: ForEachProgramID, body: list[ast.stmt]) -> list[list[ast.stmt]]:
    """The statement list of each root's case in the ``if`` tree the root
    dispatch built (``ForEachProgramID.codegen_dispatch``).  The tree sits
    in ``body``, the device function's, or in a persistent loop there; its
    branch nodes survive codegen, the lists they hold may have been
    rebuilt."""
    found: list[list[ast.stmt]] = [[] for _ in pid.cases]

    def walk(begin: int, end: int, stmts: list[ast.stmt]) -> None:
        if end - begin == 1:
            found[begin] = stmts
            return
        middle = (begin + end) // 2
        branch = pid.case_branches[middle - 1]
        assert any(stmt is branch for stmt in stmts), "root dispatch not found"
        walk(begin, middle, branch.body)
        walk(middle, end, branch.orelse)

    # A kernel with one root has no dispatch (``GenerateAST.visit_For``).
    assert len(pid.cases) > 1, pid.cases
    top = pid.case_branches[len(pid.cases) // 2 - 1]
    module = ast.Module(body=body, type_ignores=[])
    assert any(node is top for node in ast.walk(module)), "root dispatch not found"
    walk(0, len(pid.cases), [top])
    return found


def emitted_names(
    device_function: DeviceFunction, stmts: list[ast.stmt]
) -> dict[str, str]:
    """Each name in ``stmts`` mapped to the one it takes in the kernel
    ``device_function`` emits: names that codegen merges
    (``DeviceFunction.merge_variable_names``) all take the first of their
    group.  A loop-carried value is a temporary inside its loop
    (``v_1 = acc_copy_0 + load_1``) and takes the name ``acc`` only when the
    function is emitted, after these passes run."""
    return {name: device_function.variable_aliases(name)[0] for name in names_in(stmts)}


def axis_max(layouts: Sequence[tuple[int, int, int]]) -> tuple[int, int, int]:
    """The per-axis maximum of thread layouts: the threadgroup that holds
    each of them."""
    x, y, z = (max(dims[axis] for dims in layouts) for axis in range(3))
    return x, y, z


@dataclasses.dataclass
class StoreReads:
    """What a ``tl.store`` depends on (:func:`store_reads`)."""

    node: ast.Call
    #: The axes of ``tid`` it reads.
    axes: set[int]
    #: The axes of ``tid`` its pointer reads as ``tid[a]``, outside any
    #: collective: those along which threads address different elements.
    #: Its value doesn't tell the threads' elements apart, and neither do
    #: its mask and the tests around it, which only pick the threads that
    #: write, nor a collective's result, which the threads it combines share
    #: (an argmax's, though its inputs read ``tid[a]``).  This errs toward
    #: too few axes: a mask that lets one thread write still leaves the
    #: store repeated along that axis.
    address_axes: set[int]
    #: Whether it reads an MPP or fragment matmul's marker.
    matmul: bool
    #: The tensor it stores to, the base of its pointer.
    target: str
    #: The tensors loaded by all it depends on: its pointer, value and mask,
    #: and the tests and loop ranges it runs under.
    loaded: set[str]


def store_reads(
    body: list[ast.stmt], context: list[ast.stmt], renames: Mapping[str, str]
) -> list[StoreReads]:
    """Each ``tl.store`` of ``body`` and what it depends on: what its pointer,
    value and mask read, the tests and loop ranges it runs under, and the
    definitions of the names in those, transitively.  It depends on
    ``tid[a]`` for each axis ``a`` read, and on every axis for any other
    read of ``tid``.  Outside MPP and fragment matmuls, a thread learns its
    position only from ``tid``, and a collective takes all of it
    (``helion_red``), so a value computed from a reduction's result depends
    on every axis, though every thread gets the same result.  Its address
    axes count only the ``tid[a]`` its pointer reads outside collectives
    (``StoreReads.address_axes``).  The definitions come from ``context``,
    the whole device function, preamble included, so a name bound outside
    ``body`` counts; a name nothing binds is a kernel parameter, a launch
    constant, ``tgid``, a threadgroup buffer or a helper, the same on every
    thread of the threadgroup (as for ``mpp_matmul._Replicas``).  Each store
    counts only what it reads: every root runs every strategy's setup, so a
    root can hold another root's reduction index that none of its stores
    reads.  Names are taken as the emitted kernel has them (``renames``,
    :func:`emitted_names`), so a value carried through a loop depends on
    what the loop computes it from."""
    defs: dict[str, list[ast.AST]] = {}
    stores: list[tuple[ast.Call, tuple[ast.expr, ...]]] = []

    def name(written: str) -> str:
        return renames.get(written, written)

    def collect(
        stmts: list[ast.stmt],
        control: tuple[ast.expr, ...],
        found: list[tuple[ast.Call, tuple[ast.expr, ...]]] | None,
    ) -> None:
        for stmt in stmts:
            if isinstance(stmt, (ast.If, ast.While)):
                inner = (*control, stmt.test)
                collect([*stmt.body, *stmt.orelse], inner, found)
                continue
            if isinstance(stmt, ast.For):
                inner = (*control, stmt.iter)
                if found is None:
                    for node in ast.walk(stmt.target):
                        if isinstance(node, ast.Name):
                            defs.setdefault(name(node.id), []).extend(inner)
                collect([*stmt.body, *stmt.orelse], inner, found)
                continue
            for node in ast.walk(stmt):
                if found is not None:
                    if is_store(node):
                        found.append((node, control))
                    continue
                # A name, or the base of a subscript, the statement writes.
                if isinstance(node, (ast.Subscript, ast.Attribute)) and isinstance(
                    node.ctx, ast.Store
                ):
                    base = node.value
                    while isinstance(base, (ast.Subscript, ast.Attribute)):
                        base = base.value
                    if isinstance(base, ast.Name):
                        defs.setdefault(name(base.id), []).extend((stmt, *control))
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    defs.setdefault(name(node.id), []).extend((stmt, *control))

    def nodes(expr: ast.AST, collectives: bool) -> Iterator[ast.AST]:
        """The nodes of ``expr``, without a collective's arguments unless
        ``collectives``: a call handed ``tid`` whole returns what the threads
        it combines share, whatever positions its inputs read."""
        stack = [expr]
        while stack:
            node = stack.pop()
            yield node
            if not collectives and isinstance(node, ast.Call):
                if any(isinstance(a, ast.Name) and a.id == "tid" for a in node.args):
                    continue
            stack.extend(ast.iter_child_nodes(node))

    def reads(
        exprs: Sequence[ast.AST], collectives: bool = True
    ) -> tuple[set[int], bool, bool, set[str]]:
        """What ``exprs`` read, through the definitions of the names in them,
        transitively: the axes of ``tid`` read as ``tid[a]``, whether
        ``tid`` is read whole (handed to a collective), whether a matmul's
        marker is read, and the tensors loaded.  Without ``collectives``,
        what a collective reads doesn't count."""
        needed = list(exprs)
        seen: set[str] = set()
        positions: set[int] = set()
        whole = matmul = False
        loaded: set[str] = set()
        while needed:
            expr = needed.pop()
            indexed: set[int] = set()
            for node in nodes(expr, collectives):
                if (
                    isinstance(node, ast.Subscript)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "tid"
                    and isinstance(node.slice, ast.Constant)
                ):
                    positions.add(ast.literal_eval(node.slice))
                    indexed.add(id(node.value))
                elif (load := tl_load(node)) and (base := pointer_base(load.args[0])):
                    loaded.add(base)
            for node in nodes(expr, collectives):
                if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
                    continue
                if node.id == "tid" and id(node) not in indexed:
                    whole = True
                matmul = matmul or node.id.startswith(MARKER_PREFIX)
                if name(node.id) not in seen:
                    seen.add(name(node.id))
                    needed.extend(defs.get(name(node.id), ()))
        return positions, whole, matmul, loaded

    collect(context, (), None)
    collect(body, (), stores)
    out: list[StoreReads] = []
    for node, control in stores:
        positions, whole, matmul, loaded = reads([*node.args, *control])
        address_axes, _, _, _ = reads([node.args[0]], collectives=False)
        axes = {0, 1, 2} if whole else positions
        target = pointer_base(node.args[0])
        assert target is not None, ast.unparse(node)
        out.append(StoreReads(node, axes, address_axes, matmul, target, loaded))
    return out


@dataclasses.dataclass
class GuardResult:
    """What :func:`guard_surplus_stores` couldn't make safe."""

    #: Roots whose repeated stores load a tensor that one of them writes: an
    #: in-place update that several of their threads apply.
    repeated_updates: list[int]
    #: For the other roots, each tensor their repeated stores write, mapped
    #: to the tensors those stores load.  An argument that shares a storage
    #: with one they write makes them such an update after all
    #: (``metal_jit._reject_aliased_updates``).
    repeated_reads: dict[str, set[str]]


def guard_roots(
    device_function: DeviceFunction,
    root_dims: list[tuple[int, int, int]],
    frames: list[tuple[int, int, int]],
) -> tuple[list[list[ast.stmt]], GuardResult]:
    """The case body of each root in ``device_function``'s dispatch
    (:func:`case_bodies`), with its surplus threads' stores guarded in
    ``frames`` (:func:`guard_surplus_stores`), and what that left unsafe."""
    pid = device_function.pid
    assert isinstance(pid, ForEachProgramID)
    body = cast("list[ast.stmt]", device_function.body)
    bodies = case_bodies(pid, body)
    assert len(bodies) == len(root_dims), (bodies, root_dims)
    context = cast("list[ast.stmt]", [*device_function.preamble, *body])
    guarded = guard_surplus_stores(
        bodies, root_dims, frames, context, emitted_names(device_function, context)
    )
    return bodies, guarded


def guard_surplus_stores(
    bodies: list[list[ast.stmt]],
    root_dims: list[tuple[int, int, int]],
    frames: list[tuple[int, int, int]],
    context: list[ast.stmt],
    renames: Mapping[str, str],
) -> GuardResult:
    """Keep the surplus threads of each root from writing, and return what
    this leaves unsafe.  A root runs on the threads of its frame: the
    launch's, or its own layout's when the phases lay the threadgroup out per
    root; ``context`` is the device function (:func:`store_reads`).

    A root lays out its recorded threads along each axis some store of it
    depends on, else one, whatever was recorded: a reduction elsewhere in
    the kernel can widen the axes recorded for every root.  Each
    ``tl.store`` of a root laid out in fewer threads than its frame takes
    ``tid < size`` on each axis where the frame is wider: on an axis the root
    lays tiles out on, that repeats their masks, and covers a free
    ``hl.arange``'s index, which has none.  The guard covers every store of
    the root alike, so a thread it leaves out writes nothing, and one it
    keeps reads back what it wrote.  The threads still run the whole body
    together, so threadgroup collectives stay whole, as the stores of a
    split fragment K loop's SIMD groups past the first do
    (``msl_ast_walker._emit_tl_store``).

    MPP and fragment matmuls take their threads implicitly: an MPP op's
    scope is every SIMD group of the launch, and a fragment tile takes the
    launch's 32 threads, of which only the first of the SIMD groups that
    split its K loop writes (``mpp_matmul.check_mpp_launch``,
    ``msl_ast_walker._emit_tl_store``).  So a matmul's result spreads over
    the launch's threads however few its root records, and each thread
    writes the elements it holds through the matmul's element loops, which
    no store guard reaches.  A root with a matmul keeps its
    stores as they are; its other stores count as run by every thread of its
    frame.

    A store still runs on ``size`` threads along each axis, and those threads
    write the same elements along each axis its pointer doesn't read as
    ``tid[a]`` (``StoreReads.address_axes``): the threads of a root's other
    tiles, of a reduction whose result it stores, tests or indexes with, or
    of a matmul beside it.  Writing a value again is harmless, but nothing
    orders one thread's store before another's load: once a root's repeated
    stores load a tensor one of them writes (``StoreReads.loaded``, through
    a value, a mask, a test or loop range around it, or an index, directly
    or by way of another repeated store), each thread may apply the update
    again.  Those roots are returned; for the others, what their repeated
    stores write and load is returned too, since an argument that shares a
    storage with one they write is another name for it.  Stores that read a
    matmul's marker take their threads from the matmul and are left out.
    A pointer that reads ``tid[a]`` counts as one element per thread along
    that axis.  Other stores' accesses to what a repeated store writes race
    as they would with a single writer; ordering threads' accesses across
    statements is left to the kernel."""
    result = GuardResult([], {})
    for root, (body, dims, frame) in enumerate(
        zip(bodies, root_dims, frames, strict=True)
    ):
        has_matmul = any(name.startswith(MARKER_PREFIX) for name in names_in(body))
        stores = store_reads(body, context, renames)
        if has_matmul:
            sizes = list(frame)
        else:
            read = {axis for store in stores for axis in store.axes}
            sizes = [
                recorded if axis in read else 1 for axis, recorded in enumerate(dims)
            ]
        repeated = [
            store
            for store in stores
            if not store.matmul
            and any(
                axis not in store.address_axes and size > 1
                for axis, size in enumerate(sizes)
            )
        ]
        written = {store.target for store in repeated}
        loaded = {name for store in repeated for name in store.loaded}
        if written & loaded:
            result.repeated_updates.append(root)
        elif loaded:
            for target in written:
                result.repeated_reads.setdefault(target, set()).update(loaded)
        terms = [
            f"tid[{axis}] < {size}"
            for axis, (size, wide) in enumerate(zip(sizes, frame, strict=True))
            if size < wide
        ]
        if has_matmul or not terms:
            continue
        guard = " and ".join(terms)
        for store in stores:
            mask = store.node.args[2]
            store.node.args[2] = cast(
                "ast.expr",
                expr_from_string(guard)
                if isinstance(mask, ast.Constant) and mask.value is None
                else expr_from_string(f"({guard}) and {{mask}}", mask=mask),
            )
    return result
