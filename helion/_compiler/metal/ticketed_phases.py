"""Grid-wide ``hl.barrier()`` on Metal: ticketed phases in one flat launch.

A kernel whose top-level loops are split into phases by ``hl.barrier()``
launches every phase's threadgroups in one flat grid
(``MetalBackend.barrier_allows_pid_type``).  The phases run in order inside
that launch without assuming that the grid's threadgroups are co-resident:

* **Tickets.**  A threadgroup does not take its program id from its grid
  position.  Its first thread takes the next ticket from an atomic counter
  and the threadgroup runs the task with that number.  The root dispatch
  numbers the tasks phase by phase, so tickets reach a phase only after every
  earlier phase's tasks were handed out.
* **Waits and arrivals.**  A task of any phase but the last arrives on its
  phase's counter when it is done; the last arrival raises the phase's flag.
  A task of a later phase first waits for the flag of the last earlier phase
  that has tasks (a phase without tasks raises none).  Counters and flags
  each sit on their own cache line, and the wait polls with a
  read-modify-write (``msl_phases.PHASE_PREAMBLE`` says why).

This cannot deadlock, whatever the grid size and however many threadgroups
fit on the GPU: a ticket is only ever held by a threadgroup that is already
running, every task a waiting task depends on holds a smaller ticket, and the
task with the smallest unfinished ticket waits on nothing unfinished.  A grid
barrier between co-resident persistent threadgroups, by contrast, hangs once
the grid outgrows what the GPU keeps resident.

Memory ordering: an arrival orders the threadgroup's device writes before its
counter update (threadgroup barrier, then a device-scope fence by the leader),
and a wait orders the counter read before the threadgroup's later accesses.
Within one dispatch, Metal publishes one threadgroup's writes to another only
for ``coherent(device)`` memory, so every buffer that a phase writes and a
later phase reads or writes is accessed through a coherent pointer in every
phase (``msl_phases.coherent_buffers``).
"""

from __future__ import annotations

import ast
import math
from typing import TYPE_CHECKING
from typing import cast

import torch

from ... import exc
from ...language import _decorators
from ...language.barrier import barrier
from ..ast_extension import create
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..device_function import TensorArg
from ..program_id import ForEachProgramID
from ..program_id import device_str
from .lane_fold import names_in
from .msl_phases import COUNTER_STRIDE
from .msl_phases import PHASE_ACCESS_GLOBAL
from .msl_phases import PHASE_NAMESPACE
from .msl_phases import REPEATED_READS_GLOBAL
from .msl_phases import SITE_DIMS_GLOBAL
from .msl_phases import read_params
from .msl_phases import stored_params
from .multi_root import axis_max
from .multi_root import guard_roots
from .reduction import LAUNCH_DIM_VARS
from .reduction import RED_SITE_GLOBAL_PREFIX
from .reduction import alloc_threadgroup_buffer

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..device_function import DeviceFunction
    from ..device_ir import DeviceIR
    from ..generate_ast import GenerateAST
    from ..inductor_lowering import CodegenState

#: The kernel argument holding the tickets and phase counters.
STATE_PREFIX = "_helion_phase_state"


@_decorators.codegen(barrier, "metal")
def _(state: CodegenState) -> object:
    # A marker: the phases are ordered around the root dispatch
    # (:func:`order_phases`).
    return expr_from_string("None")


def is_phased_launch(device_ir: DeviceIR, config: Config) -> bool:
    """Whether this config launches ``hl.barrier()`` phases in one flat grid."""
    return len(device_ir.phases) > 1 and config.pid_type == "flat"


def add_phase_state(device_function: DeviceFunction, num_phases: int) -> str:
    """Add the zero-filled ``int32`` phase-state buffer argument and return its
    name: a ticket counter, and for each phase but the last a counter of its
    finished tasks and the flag its last task raises, each on its own line of
    ``COUNTER_STRIDE`` words (``msl_phases.PHASE_PREAMBLE``).

    Called before the roots are generated: the launcher call is emitted
    during that, and only the finished body names the buffer
    (:func:`order_phases`), so the argument is kept through DCE.
    """
    name = device_function.new_var(STATE_PREFIX, dce=False)
    device = device_str()
    words = (2 * num_phases - 1) * COUNTER_STRIDE
    device_function.arguments.append(
        TensorArg(
            name,
            torch.empty(words, device="meta", dtype=torch.int32),
            f"torch.zeros(({words},), device={device}, dtype=torch.int32)",
        )
    )
    device_function.placeholder_args.add(name)
    return name


def _contains_return(stmts: list[ast.stmt]) -> bool:
    return any(
        isinstance(node, ast.Return) for stmt in stmts for node in ast.walk(stmt)
    )


def relaid_out(root_dims: list[tuple[int, int, int]], matmuls_own_axes: bool) -> bool:
    """Whether the roots of a phased launch each lay the threadgroup out
    their own way (:func:`relayout_roots`): their layouts differ but use the
    same number of threads, and no MPP or fragment matmul runs on the launch's
    own thread axes.  Other roots share the per-axis maximum, and the
    surplus threads of each narrower one without a matmul don't write
    (``multi_root.guard_surplus_stores``)."""
    return (
        not matmuls_own_axes
        and len(set(root_dims)) > 1
        and len({math.prod(d) for d in root_dims}) == 1
    )


def launch_dims(
    root_dims: list[tuple[int, int, int]], matmuls_own_axes: bool
) -> tuple[int, int, int]:
    """The threadgroup of a phased launch whose roots laid their threads out
    as ``root_dims`` (``matmuls_own_axes``: whether the kernel's MPP or
    fragment matmuls run on the launch's own thread axes)."""
    if relaid_out(root_dims, matmuls_own_axes):
        return (math.prod(root_dims[0]), 1, 1)
    return axis_max(root_dims)


def order_phases(
    codegen: GenerateAST,
    state_arg: str,
    root_dims: list[tuple[int, int, int]],
    matmuls_own_axes: bool,
    launch: tuple[int, int, int],
) -> None:
    """Make the flat launch's root dispatch run its phases in order: the
    shared program id becomes a ticket, each case of a later phase waits for
    the last earlier phase that has tasks, and each case of an earlier phase
    arrives when done.  ``launch`` is the threadgroup (:func:`launch_dims`),
    which ``MetalBackend.after_device_codegen`` has checked fits.
    """
    device_function = codegen.device_function
    pid = device_function.pid
    assert isinstance(pid, ForEachProgramID)
    phases = pid.case_phases
    assert len(phases) == len(pid.case_bodies) == len(pid.cases)
    ordered = sorted(set(phases))
    # Roots follow the source, and so do the phases hl.barrier() splits it into.
    assert phases == sorted(phases), phases
    last = ordered[-1]
    rank = {phase: i for i, phase in enumerate(ordered)}

    # The task count of each phase, as device expressions.
    counts: dict[int, list[str]] = {}
    for case, phase in zip(pid.cases, phases, strict=True):
        counts.setdefault(phase, []).append(f"({case.total_pids_expr(is_device=True)})")

    relaid = relaid_out(root_dims, matmuls_own_axes)
    frames = root_dims if relaid else [launch] * len(root_dims)
    bodies, guarded = guard_roots(device_function, root_dims, frames)
    # Every case body must be reached by all of its threadgroup's threads:
    # the waits and arrivals synchronize the threadgroup.
    for body in bodies:
        if _contains_return(body):
            raise exc.BackendUnsupported(
                "metal", "hl.barrier(): early return in a phase"
            )
    if guarded.repeated_updates:
        raise exc.BackendUnsupported(
            "metal",
            "hl.barrier(): an in-place update that several threads of its "
            "phase repeat: stores they all run load what those stores write "
            "(outside the threads' tiles, through a reduction's result, or "
            "beside a matmul)",
        )
    # Arguments that share a storage show only at the call: metal_jit refuses
    # those that make one of these stores an in-place update.
    repeated_reads = {
        target: tuple(sorted(loaded))
        for target, loaded in sorted(guarded.repeated_reads.items())
    }
    device_function.codegen.module_statements.append(
        statement_from_string(f"{REPEATED_READS_GLOBAL} = {repeated_reads!r}")
    )

    for body, phase in zip(bodies, phases, strict=True):
        if rank[phase] > 0:
            body[:0] = _wait_for_earlier(
                state_arg,
                len(ordered),
                [(rank[p], counts[p]) for p in ordered[: rank[phase]]],
            )
        if phase != last:
            body.append(
                statement_from_string(
                    f"{PHASE_NAMESPACE}.arrive({state_arg}, {len(ordered)}, {rank[phase]}, "
                    f"{' + '.join(counts[phase])}, tid)"
                )
            )

    _record_phase_access(device_function, bodies, [rank[p] for p in phases])
    if relaid:
        relayout_roots(device_function, bodies, root_dims)
    slot = alloc_threadgroup_buffer("uint", 1, prefix="_phase_ticket")
    ticket = cast(
        "ast.expr",
        expr_from_string(f"{PHASE_NAMESPACE}.ticket({state_arg}, {slot}, tid)"),
    )
    for stmt in device_function.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == pid.shared_pid_var
        ):
            stmt.value = ticket
            return
    raise AssertionError(f"no assignment of {pid.shared_pid_var} in the kernel body")


def _wait_for_earlier(
    state_arg: str, num_phases: int, earlier: list[tuple[int, list[str]]]
) -> list[ast.stmt]:
    """The wait opening a task: for the flag of the last earlier phase that
    has tasks (``earlier``: each earlier phase's rank and the task counts of
    its roots, device expressions).  A phase with no tasks never raises its flag (an
    empty input, an MoE expert without tokens), so it is skipped; it ran no
    task, so waiting for the phase before it orders everything it would have.
    A count known at codegen picks its branch here; a runtime count becomes
    a branch the whole threadgroup takes alike."""
    stmts: list[ast.stmt] = []
    tail = stmts
    for rank, counts in reversed(earlier):
        wait = statement_from_string(
            f"{PHASE_NAMESPACE}.wait({state_arg}, {num_phases}, {rank}, tid)"
        )
        known = [_literal_int(count) for count in counts]
        if any(n is not None and n > 0 for n in known):
            tail.append(wait)
            return stmts
        runtime = [c for c, n in zip(counts, known, strict=True) if n is None]
        if not runtime:
            continue
        branch = create(
            ast.If,
            test=expr_from_string(f"({' + '.join(runtime)}) > 0"),
            body=[wait],
            orelse=[],
        )
        tail.append(branch)
        tail = branch.orelse
    return stmts


def _literal_int(expr: str) -> int | None:
    node = ast.parse(expr, mode="eval").body
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    return None


class _LaunchSizes(ast.NodeTransformer):
    """Replace reads of the launch-size constants with one root's sizes."""

    def __init__(self, consts: dict[str, int]) -> None:
        self.consts = consts

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.consts and isinstance(node.ctx, ast.Load):
            return ast.copy_location(ast.Constant(self.consts[node.id]), node)
        return node


def relayout_roots(
    device_function: DeviceFunction,
    bodies: list[list[ast.stmt]],
    root_dims: list[tuple[int, int, int]],
) -> None:
    """Let each root of a phased launch lay the threadgroup out its own way.

    The roots run one per threadgroup, so they need not share a layout: a
    matmul root may want 32 threads along K by 8 along N where a row norm
    wants 256 along the row.  The launch is one flat axis of their common
    thread count (:func:`launch_dims`); each case body opens by redefining
    ``tid`` from the flat index in its root's layout, and the launch-size
    constants its reductions read (``_TG_SIZE_*``) become its own.  Its
    reduction sites are sized for its layout too (:data:`SITE_DIMS_GLOBAL`).
    The flat index's SIMD lanes are the hardware's, so SIMD collectives keep
    working.
    """
    # The reduction sites, registered as module globals by
    # ``reduction._group_site``.
    sites = {
        target.id.removeprefix(RED_SITE_GLOBAL_PREFIX)
        for stmt in device_function.codegen.module_statements
        if isinstance(stmt, ast.Assign)
        for target in stmt.targets
        if isinstance(target, ast.Name) and target.id.startswith(RED_SITE_GLOBAL_PREFIX)
    }
    flat = device_function.new_var("_phase_lin", dce=False)
    device_function.body.insert(0, statement_from_string(f"{flat} = tid[0]"))
    site_dims: dict[str, tuple[int, int, int]] = {}
    for body, dims in zip(bodies, root_dims, strict=True):
        sizes = _LaunchSizes(dict(zip(LAUNCH_DIM_VARS, dims, strict=True)))
        body[:] = [sizes.visit(stmt) for stmt in body]
        body.insert(
            0,
            statement_from_string(
                f"tid = {PHASE_NAMESPACE}.layout({flat}, {dims[0]}, {dims[1]})"
            ),
        )
        for name in names_in(body) & sites:
            site_dims[name] = dims
    device_function.codegen.module_statements.append(
        statement_from_string(f"{SITE_DIMS_GLOBAL} = {site_dims!r}")
    )


def _record_phase_access(
    device_function: DeviceFunction, bodies: list[list[ast.stmt]], ranks: list[int]
) -> None:
    """Record, for ``metal_jit``, the phases that read each tensor argument
    and the phases that store to it (:data:`PHASE_ACCESS_GLOBAL`)."""
    params = {
        arg.name for arg in device_function.arguments if isinstance(arg, TensorArg)
    }
    access: dict[str, tuple[set[int], set[int]]] = {}
    for body, rank in zip(bodies, ranks, strict=True):
        for name in read_params(body, params):
            access.setdefault(name, (set(), set()))[0].add(rank)
        for name in stored_params(body, params):
            access.setdefault(name, (set(), set()))[1].add(rank)
    literal = {
        name: (tuple(sorted(reads)), tuple(sorted(writes)))
        for name, (reads, writes) in sorted(access.items())
    }
    device_function.codegen.module_statements.append(
        statement_from_string(f"{PHASE_ACCESS_GLOBAL} = {literal!r}")
    )
