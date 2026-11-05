"""Reductions over a tile block that its own device loop walks in a lane loop.

A tile block wider than its threads is walked by a lane loop that its
``DeviceLoopState`` wraps around the whole loop body, on one thread or on each
of several.  Every consumer of a reduction over that block needs all of its
lanes, so the lane loop is split into passes, as CuTe's
``split_lane_loop_reductions`` does for CuTe DSL:

- each reduction emits a marker ``r = _metal_lane_reduce(x)`` at the top
  level of the lane body (:meth:`LaneSplit.emit_marker`), and the device loop
  gets the :class:`LaneSplit` as a body finalizer;
- once the body is complete, each statement is lane-variant (it reads the
  lane, transitively) or lane-invariant, a marker's result being invariant.
  A marker's stage is one more than its input's, any other statement's the
  latest stage it reads;
- each stage's markers get a pass over the lanes that recomputes the variant
  statements their inputs need and folds them.  Invariant statements run once,
  between the passes, after the markers they read, and a final pass runs every
  variant statement no pass folded, stores included;
- a fold over a block that also has threads folds each thread's lanes into a
  partial, and one collective after the pass combines the threads.  The
  collective runs outside every lane loop, at the top level of the device
  loop body, so every thread of the threadgroup reaches it in the same
  iteration.

A narrow load that several passes recompute is read once: the first pass
keeps each lane's value in a per-thread array that the later passes read, as
CuTe's register stash does for the values it keeps across a split
(``_split_lane_loop_with_register_stash``).

Only register statements move relative to each other or to stores: a tensor
stored in the body is not otherwise accessed in it, under its own name or any
other that may share its memory.  :func:`check_schedule` replays the result
against the original body.  Every other layout raises
``BackendUnsupported``.
"""

from __future__ import annotations

import ast
import collections
import dataclasses
import itertools
from typing import TYPE_CHECKING
from typing import cast

from torch._inductor import ir
from torch._inductor.codegen.simd import constant_repr

from ... import exc
from ..ast_extension import statement_from_string
from ..ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from ..ast_read_writes import ReadWrites
from ..compile_environment import CompileEnvironment
from ..device_function import TensorArg
from ..reduction_strategy import reduction_acc_dtype
from ..tile_strategy import DeviceLoopState
from ..tile_strategy import _clone_lane_loop_with_body
from ..tile_strategy import _current_rename_groups
from ..tile_strategy import _merge_carried_names
from .msl_reduction import SUPPORTED_REDUCTIONS
from .reduction import reduction_combine_expr

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Sequence

    import torch

    from ..device_function import DeviceFunction
    from ..generate_ast import GenerateAST
    from ..inductor_lowering import CodegenState

MARKER = "_metal_lane_reduce"

#: Most bytes of lane arrays one split keeps per thread.  Measured on Apple
#: GPUs: arrays of 16-bit loads up to 64 bytes were neutral or faster than
#: reloading, 128 bytes mixed and 256 bytes up to 2.3x slower; 32-bit arrays
#: were neutral to 5% slower at any size, so only narrower loads are kept.
MAX_LANE_ARRAY_BYTES = 64

#: A definition no single execution of the original body sees: a lane-variant
#: value after its pass (the last lane's), or a fold result inside its pass.
_STALE = -1


def _unsupported(message: str) -> exc.BackendUnsupported:
    return exc.BackendUnsupported("metal", message)


@dataclasses.dataclass(frozen=True)
class LaneReduction:
    result: str
    input_name: str
    reduction_type: str
    acc_dtype: torch.dtype
    identity: float | int | bool
    #: The name the lanes fold into: ``result``, or a per-thread partial that
    #: ``thread_combine`` combines over the block's threads into ``result``.
    partial: str
    thread_combine: str | None


@dataclasses.dataclass
class LanePass:
    """One loop over the lanes: body statements, then the folds of markers."""

    statements: list[int]
    folds: list[int]


@dataclasses.dataclass
class LaneBody:
    """The lane loop's statements, with their reads and writes under the
    canonical names of their rename groups (a carry's loop output and initial
    value are one name, as are views of one tensor made in the host code)."""

    statements: list[ast.AST]
    reads: list[set[str]]
    writes: list[set[str]]
    reductions: dict[int, LaneReduction]

    def reaching_definitions(self) -> list[dict[str, int]]:
        """Per statement, the last statement before it to write each name."""
        result = []
        defined: dict[str, int] = {}
        for index, writes in enumerate(self.writes):
            result.append(dict(defined))
            defined.update(dict.fromkeys(writes, index))
        return result


@dataclasses.dataclass
class LaneSplit:
    """The lane reductions of one device loop, split once its body is done."""

    fn: DeviceFunction
    loop: DeviceLoopState
    lane_var: str
    reductions: dict[str, LaneReduction] = dataclasses.field(default_factory=dict)

    def emit_marker(
        self,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        fake_input: torch.Tensor,
        fake_output: torch.Tensor,
        across_threads: Callable[[str], str] | None,
    ) -> str:
        """Emit the marker of one reduction; ``across_threads`` maps the name
        of a thread's partial to its combine over the block's threads."""
        acc_dtype = reduction_acc_dtype(reduction_type, fake_input, fake_output)
        identity = ir.Reduction.default_accumulator(reduction_type, acc_dtype)
        assert isinstance(identity, (float, int, bool))
        result = self.fn.new_var(f"lane_{reduction_type}", dce=True)
        partial, thread_combine = result, None
        if across_threads is not None:
            partial = self.fn.new_var(f"lane_{reduction_type}_part", dce=True)
            thread_combine = across_threads(partial)
        self.reductions[result] = LaneReduction(
            result,
            input_name,
            reduction_type,
            acc_dtype,
            identity,
            partial,
            thread_combine,
        )
        state.add_statement(statement_from_string(f"{result} = {MARKER}({input_name})"))
        return result

    def __call__(self) -> None:
        loop = self.loop
        parent, lane_loop = _find_lane_loop(loop, self.lane_var)
        groups = _merge_carried_names(_current_rename_groups(), loop.carried_names)
        statements: list[ast.AST] = [*lane_loop.body]
        body = _lane_body(statements, self, groups)
        schedule = _schedule(body, groups.get(self.lane_var, self.lane_var))
        check_schedule(body, schedule)
        position = next(i for i, stmt in enumerate(parent) if stmt is lane_loop)
        parent[position : position + 1] = _emit(body, schedule, lane_loop, self.fn)
        # The passes now hold the body; statements added to the loop later go
        # beside them.
        loop.inner_statements = parent


def lane_split_for(
    state: CodegenState, fn: DeviceFunction, block_id: int, reduction_type: str
) -> LaneSplit | None:
    """The split for a reduction over ``block_id`` when its device loop's only
    lane loop walks the block, on one thread or on each of several threads,
    and the reduction is a top-level statement of that lane body; None for
    any other layout."""
    codegen = state.codegen
    loop = next(
        (
            active
            for active in codegen.active_device_loops.get(block_id, [])
            if isinstance(active, DeviceLoopState)
            and block_id in active.lane_loop_blocks
        ),
        None,
    )
    if (
        loop is None
        or len(loop.lane_loops) != 1
        or codegen.statements_stack[-1] is not loop.inner_statements
        or reduction_type not in SUPPORTED_REDUCTIONS
        or _in_enclosing_device_lane_loop(codegen, loop)
    ):
        return None
    split = next((f for f in loop.body_finalizers if isinstance(f, LaneSplit)), None)
    if split is None:
        (lane_var, _) = loop.lane_loops[0]
        split = LaneSplit(fn, loop, lane_var)
        loop.body_finalizers.append(split)
    return split


def _in_enclosing_device_lane_loop(codegen: GenerateAST, loop: DeviceLoopState) -> bool:
    """Whether ``loop`` runs inside another device loop's lane loop.

    That lane loop re-runs ``loop`` whole once per lane, carry updates
    included, while the carries are initialised outside it.  A grid's lane
    loops are fine: they wrap the carry initialisations too.
    """
    return any(
        isinstance(active, DeviceLoopState) and active is not loop and active.lane_loops
        for loops in codegen.active_device_loops.values()
        for active in loops
    )


def _find_lane_loop(
    loop: DeviceLoopState, lane_var: str
) -> tuple[list[ast.AST], ast.For]:
    """The lane loop whose body is the device loop's body, and its parent list."""

    def visit(statements: list[ast.AST]) -> tuple[list[ast.AST], ast.For] | None:
        for stmt in statements:
            if not isinstance(stmt, ast.For):
                continue
            if (
                getattr(stmt, HELION_LANE_LOOP_VAR_ATTR, None) == lane_var
                and stmt.body is loop.inner_statements
            ):
                return statements, stmt
            if (found := visit(cast("list[ast.AST]", stmt.body))) is not None:
                return found
        return None

    found = visit([loop.for_node])
    if found is None:
        raise _unsupported("lane reduction: its lane loop is not around the body")
    return found


def _marker_result(stmt: ast.AST) -> str | None:
    if (
        isinstance(stmt, ast.Assign)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Name)
        and stmt.value.func.id == MARKER
    ):
        (target,) = stmt.targets
        assert isinstance(target, ast.Name)
        return target.id
    return None


def _tl_call(node: ast.AST, name: str) -> ast.Call | None:
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
    ):
        return node
    return None


def _is_store(stmt: ast.AST) -> bool:
    return isinstance(stmt, ast.Expr) and _tl_call(stmt.value, "store") is not None


def _is_name_assign(stmt: ast.AST) -> bool:
    return (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
    )


def _lane_body(
    statements: list[ast.AST], split: LaneSplit, groups: dict[str, str]
) -> LaneBody:
    def canonical(names: Iterable[str]) -> set[str]:
        return {groups.get(name, name) for name in names}

    tensors = {arg.name for arg in split.fn.arguments if isinstance(arg, TensorArg)}

    def tensors_in(pointer: ast.expr) -> set[str]:
        names = {node.id for node in ast.walk(pointer) if isinstance(node, ast.Name)}
        return canonical(names & tensors)

    reductions: dict[int, LaneReduction] = {}
    for index, stmt in enumerate(statements):
        if (result := _marker_result(stmt)) is not None:
            reductions[index] = split.reductions[result]
        elif not (_is_store(stmt) or _is_name_assign(stmt)):
            raise _unsupported(
                "lane reduction beside a statement the split cannot schedule: "
                + ast.unparse(stmt).partition("\n")[0]
            )
    stored: set[str] = set()
    loaded: set[str] = set()
    for stmt in statements:
        if _is_store(stmt):
            assert isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
            for tensor in sorted(tensors_in(stmt.value.args[0])):
                if tensor in stored:
                    raise _unsupported(
                        f"lane reduction: {tensor} stored twice in its lane loop"
                    )
                stored.add(tensor)
        for node in ast.walk(stmt):
            if (load := _tl_call(node, "load")) is not None:
                loaded |= tensors_in(load.args[0])
    if both := sorted(stored & loaded):
        raise _unsupported(
            f"lane reduction: {both[0]} is stored and loaded in its lane loop; "
            "load it in another loop"
        )
    # Kernel arguments may share memory at any call (``kernel(x, out=x)``),
    # which fake tensors don't show.
    disjoint = split.fn.proven_disjoint_tensor_pairs()
    for tensor, other in itertools.product(sorted(stored), sorted(stored | loaded)):
        if tensor != other and frozenset((tensor, other)) not in disjoint:
            raise _unsupported(
                f"lane reduction: {tensor} is stored in its lane loop and may "
                f"share memory with {other}; store into a tensor the kernel "
                "allocates"
            )
    return LaneBody(
        statements=statements,
        reads=[canonical(ReadWrites.from_ast(stmt).reads) for stmt in statements],
        writes=[
            set() if _is_store(stmt) else canonical(ReadWrites.from_ast(stmt).writes)
            for stmt in statements
        ],
        reductions=reductions,
    )


def _schedule(body: LaneBody, lane_var: str) -> list[int | LanePass]:
    """The passes and the once-per-tile statements between them, in order."""
    count = len(body.statements)
    definitions = body.reaching_definitions()
    variant = [False] * count
    variant_names = {lane_var}
    for index in range(count):
        if index not in body.reductions and body.reads[index] & variant_names:
            variant[index] = True
            variant_names |= body.writes[index]
    for index in range(count):
        for name in sorted(body.reads[index] - definitions[index].keys()):
            # The loop-entry value, later overwritten per lane: each lane
            # would continue from the previous lane's value.
            if any(
                variant[writer] and name in body.writes[writer]
                for writer in range(index, count)
            ):
                raise _unsupported(
                    f"lane reduction: per-lane value {name} is carried across "
                    "lanes; carry its reduction over the lane-looped dim instead"
                )

    stage = [0] * count
    for index in range(count):
        stage[index] = max(
            (
                stage[definitions[index][name]]
                for name in body.reads[index]
                if name in definitions[index]
            ),
            default=0,
        ) + int(index in body.reductions)

    def variant_slice(starts: list[int]) -> list[int]:
        """The variant statements ``starts`` read, transitively, and the
        variant ones among ``starts``."""
        needed = {index for index in starts if variant[index]}
        pending = [*starts]
        while pending:
            index = pending.pop()
            for name in body.reads[index]:
                writer = definitions[index].get(name)
                if writer is not None and variant[writer] and writer not in needed:
                    needed.add(writer)
                    pending.append(writer)
        return sorted(needed)

    last = max(stage[index] for index in body.reductions)
    schedule: list[int | LanePass] = []
    folded: set[int] = set()
    for current in range(last + 1):
        schedule.extend(
            index
            for index in range(count)
            if stage[index] == current
            and not variant[index]
            and index not in body.reductions
        )
        if current == last:
            break
        folds = [index for index in body.reductions if stage[index] == current + 1]
        statements = variant_slice(folds)
        folded.update(statements)
        schedule.append(LanePass(statements, folds))
    rest = variant_slice(
        [index for index in range(count) if variant[index] and index not in folded]
    )
    if rest:
        schedule.append(LanePass(rest, []))
    return schedule


def check_schedule(body: LaneBody, schedule: Sequence[int | LanePass]) -> None:
    """Raise unless every read in ``schedule`` sees the definition it saw in
    one execution of the original body and every statement is emitted.

    The scheduler relies on the lane body being single-assignment, apart from
    carry writes whose readers in the body read a copy taken before them; this
    turns a body that breaks that into a rejection, not a wrong result.
    """
    original: dict[tuple[int, str], int | None] = {}
    definitions = body.reaching_definitions()
    for index, reads in enumerate(body.reads):
        for name in reads:
            original[index, name] = definitions[index].get(name)
    defined: dict[str, int] = {}
    emitted: set[int] = set()

    def check_reads(index: int, what: str) -> None:
        for name in sorted(body.reads[index]):
            if defined.get(name) != original[index, name]:
                raise _unsupported(f"lane reduction split reorders {name} for {what}")

    def run(index: int) -> None:
        check_reads(index, ast.unparse(body.statements[index]).partition("\n")[0])
        defined.update(dict.fromkeys(body.writes[index], index))
        emitted.add(index)

    for item in schedule:
        if isinstance(item, int):
            run(item)
            continue
        for fold in item.folds:
            defined.update(dict.fromkeys(body.writes[fold], _STALE))
        for index in item.statements:
            run(index)
        for fold in item.folds:
            check_reads(fold, f"the fold of {body.reductions[fold].result}")
        for index in item.statements:
            defined.update(dict.fromkeys(body.writes[index], _STALE))
        for fold in item.folds:
            defined.update(dict.fromkeys(body.writes[fold], fold))
    for index, stmt in enumerate(body.statements):
        if index not in body.reductions and index not in emitted:
            raise _unsupported(
                "lane reduction split drops " + ast.unparse(stmt).partition("\n")[0]
            )


def _lane_arrays(
    body: LaneBody,
    schedule: Sequence[int | LanePass],
    lane_loop: ast.For,
    fn: DeviceFunction,
) -> dict[int, tuple[str, str]]:
    """The loads that more than one pass recomputes and that a lane array
    keeps instead, by statement index, each with its array's name and the
    array's declaration.

    A later pass reads the same masked load the same thread made for the same
    lane, and ``_lane_body`` has already rejected any store to a loaded
    tensor, so the kept value is the one a reload would read.
    """
    passes = [item.statements for item in schedule if isinstance(item, LanePass)]
    counts = collections.Counter(itertools.chain.from_iterable(passes))
    iterator = lane_loop.iter
    # Block sizes and thread counts are compile-time constants, and so is the
    # lane count.
    assert isinstance(iterator, ast.Call)
    (lanes,) = iterator.args
    assert isinstance(lanes, ast.Constant) and isinstance(lanes.value, int)
    tensors = {arg.name: arg for arg in fn.arguments if isinstance(arg, TensorArg)}
    backend = CompileEnvironment.current().backend
    budget = MAX_LANE_ARRAY_BYTES
    arrays: dict[int, tuple[str, str]] = {}
    for index in sorted(index for index, count in counts.items() if count > 1):
        stmt = body.statements[index]
        if not _is_name_assign(stmt):
            continue
        assert isinstance(stmt, ast.Assign)
        load = _tl_call(stmt.value, "load")
        if load is None:
            continue
        (tensor,) = {
            node.id for node in ast.walk(load.args[0]) if isinstance(node, ast.Name)
        } & tensors.keys()
        dtype = tensors[tensor].fake_value.dtype
        size = lanes.value * dtype.itemsize
        if dtype.itemsize >= 4 or size > budget:
            continue
        budget -= size
        (target,) = stmt.targets
        assert isinstance(target, ast.Name)
        arrays[index] = (
            fn.new_var(f"{target.id}_lanes", dce=True),
            f"_metal_lane_array({backend.dtype_str(dtype)!r}, {lanes.value})",
        )
    return arrays


def _emit(
    body: LaneBody,
    schedule: list[int | LanePass],
    lane_loop: ast.For,
    fn: DeviceFunction,
) -> list[ast.AST]:
    init_expr = CompileEnvironment.current().backend.reduction_acc_init_expr
    arrays = _lane_arrays(body, schedule, lane_loop, fn)
    lane = cast("ast.Name", lane_loop.target).id
    kept: set[int] = set()
    emitted: list[ast.AST] = []
    for item in schedule:
        if isinstance(item, int):
            emitted.append(body.statements[item])
            continue
        lane_body: list[ast.AST] = []
        for index in item.statements:
            stmt = body.statements[index]
            if index not in arrays:
                lane_body.append(stmt)
                continue
            array, declaration = arrays[index]
            (target,) = cast("ast.Assign", stmt).targets
            value = cast("ast.Name", target).id
            if index in kept:
                lane_body.append(
                    statement_from_string(f"{value} = _metal_lane_get({array}, {lane})")
                )
                continue
            emitted.append(statement_from_string(f"{array} = {declaration}"))
            lane_body.extend(
                (
                    stmt,
                    statement_from_string(f"_metal_lane_put({array}, {lane}, {value})"),
                )
            )
            kept.add(index)
        for fold in item.folds:
            reduction = body.reductions[fold]
            identity = init_expr(
                [], constant_repr(reduction.identity), reduction.acc_dtype
            )
            combine = reduction_combine_expr(
                reduction.reduction_type,
                reduction.partial,
                reduction.input_name,
                reduction.acc_dtype,
            )
            emitted.append(statement_from_string(f"{reduction.partial} = {identity}"))
            lane_body.append(statement_from_string(f"{reduction.partial} = {combine}"))
        emitted.append(_clone_lane_loop_with_body(lane_loop, lane_body))
        emitted.extend(
            statement_from_string(f"{reduction.result} = {reduction.thread_combine}")
            for reduction in map(body.reductions.__getitem__, item.folds)
            if reduction.thread_combine is not None
        )
    return emitted
