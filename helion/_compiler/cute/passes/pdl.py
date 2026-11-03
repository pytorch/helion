"""Programmatic dependent launch (``cute_pdl``) placement in a CuTe kernel.

``prepare_device_body`` runs ahead of the device-body passes
(``CuteBackend.finalize_device_body``): it splices the ``griddepcontrol``
wait and this kernel's own release around the rendered body (the cluster
sync of the cluster-reduce sites goes between the argument preamble and the
body) and, on the tcgen05 entry-wait path, follows every wait of the body
with an early release.  ``place_late_wait`` runs after the passes and moves
the wait of a kernel whose zero fills launch as fill kernels down to the
first touch of a filled tensor.  ``insert_spread_pad`` (``cute_pdl_spread``)
pads the renamed kernel's shared memory after the kernel-def passes
(``CuteBackend.finalize_kernel_def``).  ``pdl_applies`` is the gate the host
zero-fill launches share.  The knob choices and the pure sizing helpers live
in ``cute.pdl_release``.
"""

from __future__ import annotations

import ast
import dataclasses
import math
from typing import TYPE_CHECKING
from typing import cast

import torch

from ...ast_extension import create
from ...ast_extension import expr_from_string
from ...ast_extension import statement_from_string
from ...compile_environment import CompileEnvironment
from ...host_function import HostFunction
from ...source_location import SyntheticLocation
from ..pdl_release import CUTE_PDL_RELEASE_DEFAULT
from ..pdl_release import CUTE_PDL_SPREAD_KEY
from ..pdl_release import DEFAULT_MAX_SMEM_PER_BLOCK
from ..pdl_release import DEFAULT_RESIDENT_THREADS_PER_SM
from ..pdl_release import DEFAULT_SMEM_PER_SM
from ..pdl_release import SPREAD_KEEP_HELPER
from ..pdl_release import SPREAD_PAD_NAME
from ..pdl_release import resident_ctas_per_sm
from ..pdl_release import spread_pad_bytes

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...device_function import DeviceFunction


@dataclasses.dataclass
class LateWait:
    """A ``cute_pdl`` wait held back for ``place_late_wait``: the wait, the
    entry release statements it precedes and the kernel-argument names whose
    first mention bounds it."""

    wait: ast.stmt
    release: list[ast.stmt]
    names: set[str]


def prepare_device_body(
    fn: DeviceFunction, body: list[ast.stmt], body_start: int
) -> tuple[list[ast.stmt], LateWait | None]:
    """``body`` with the cluster sync, the ``cute_pdl`` wait and this kernel's
    release placed; ``body[:body_start]`` is the argument preamble.  Returns
    the wait to place after the body passes (``place_late_wait``), if any."""
    cluster_sync: list[ast.stmt] = []
    if fn.cute_state.lanes.simt_cluster_reduce_sites > 0:
        # One fence + cluster barrier covering every cluster-reduce
        # site's mbarrier init (emitted into the preamble above).
        cluster_sync = [
            statement_from_string("cute.arch.mbarrier_init_fence()"),
            statement_from_string("cute.arch.cluster_arrive_relaxed()"),
            statement_from_string("cute.arch.cluster_wait()"),
        ]
    dependent_launch: list[ast.stmt] = []
    dependent_release: list[ast.stmt] = []
    late: LateWait | None = None
    if pdl_applies(fn):
        # Programmatic dependent launch (``cute_pdl``): the launch and
        # this prologue overlap the previous kernel in the stream; wait
        # for it before the first global memory access.  ``cute_pdl_release``
        # places the release of this kernel's own dependents: ``entry``
        # releases right after the wait so the dependents' CTAs become
        # resident while this kernel runs (they hold SM slots while they
        # wait: a one-wave kernel's gain, a multi-wave kernel's possible
        # cost), ``exit`` releases after the body's last statement,
        # ``none`` leaves it to the implicit trigger at grid completion,
        # after which a dependent starts later than a plain stream edge
        # would have launched it, ``last_wave`` releases at entry in the
        # CTAs of the launch's last wave only (``_last_wave_release``),
        # and ``auto`` (the default) picks ``entry`` while the grid fits
        # in one wave and ``last_wave`` past it (``_implicit_pdl_release``).
        # The dependents still wait for this grid's completion and memory
        # flush at their own wait, so the placement never changes what
        # they read.  A tcgen05 kernel on the entry-wait path
        # (``cute_pdl_entry_wait``) gets only the wait here: its early
        # release follows every wait of the body, this one included
        # (``_tcgen05_early_pdl_release``, below).
        with SyntheticLocation():
            wait = statement_from_string("cute.arch.griddepcontrol_wait()")
            dependent_launch = [wait]
            # Behind the fill kernels of this kernel's own zero fills the
            # wait may move down to the first statement that touches a
            # filled tensor (``_place_late_pdl_wait``, on the lowered
            # body): the fill kernel releases its dependents only after
            # its own wait, so every earlier kernel of the stream has
            # completed when this grid starts, and the fill's writes are
            # confined to the filled tensors.  The operand loads and the
            # K loop of a split-K kernel then overlap the fill instead of
            # waiting for it.
            late_wait_names = _late_pdl_wait_names(fn)
            if late_wait_names:
                dependent_launch = []
            placement = (
                "none"
                if fn.cute_state.tcgen05.pdl_entry_wait
                else fn.config.config.get("cute_pdl_release", CUTE_PDL_RELEASE_DEFAULT)
            )
            if placement == "auto":
                placement = _implicit_pdl_release(fn, "entry")
            release = statement_from_string(
                "cute.arch.griddepcontrol_launch_dependents()"
            )
            if placement == "entry":
                dependent_launch.append(release)
            elif placement == "exit":
                dependent_release = [release]
            elif placement == "last_wave":
                dependent_launch.extend(_last_wave_release(fn, release))
            if late_wait_names:
                late = LateWait(wait, list(dependent_launch), late_wait_names)
    body = [
        *dependent_launch,
        *body[:body_start],
        *cluster_sync,
        *body[body_start:],
        *dependent_release,
    ]
    if fn.cute_state.tcgen05.pdl_entry_wait:
        body = _tcgen05_early_pdl_release(fn, body)
    return body, late


def place_late_wait(body: list[ast.stmt], late: LateWait) -> list[ast.stmt]:
    """``body`` (after the device-body passes) with the held-back wait placed."""
    return _place_late_pdl_wait(body, late.wait, late.release, late.names)


def insert_spread_pad(fn: DeviceFunction, definition: ast.FunctionDef) -> None:
    """``cute_pdl_spread``: pad the body's shared memory so that an early
    launch of this kernel cannot pack its CTAs denser than one wave on an
    idle device (``_pdl_spread_statements``).  The pad goes in after every
    body pass, sized against the body they left (a fused two-pass cache
    allocates late)."""
    if pdl_applies(fn) and fn.config.config.get(CUTE_PDL_SPREAD_KEY):
        index = _pdl_spread_index(definition.body)
        definition.body[index:index] = _pdl_spread_statements(fn, definition.body)


# Element widths of the ``cutlass`` numeric types a generated body allocates
# shared memory in (``cute.arch.alloc_smem(cutlass.<type>, count, ...)``).
_CUTLASS_TYPE_BYTES: dict[str, int] = {
    "cutlass.Int8": 1,
    "cutlass.Uint8": 1,
    "cutlass.Int16": 2,
    "cutlass.Uint16": 2,
    "cutlass.Float16": 2,
    "cutlass.BFloat16": 2,
    "cutlass.Int32": 4,
    "cutlass.Uint32": 4,
    "cutlass.Float32": 4,
    "cutlass.TFloat32": 4,
    "cutlass.Int64": 8,
    "cutlass.Uint64": 8,
    "cutlass.Float64": 8,
}


# Calls through which a body reserves shared memory the literal sum cannot
# see: the DSL's ``SmemAllocator`` (its ``allocate`` / ``allocate_tensor`` /
# ``allocate_array`` carve a struct or a layout's bytes out of the kernel's
# allocation at trace time; the fa4 flash families and the tcgen05 GEMMs
# hold their rings this way, at the per-block maximum already) and the
# dynamic shared memory base.
_OPAQUE_SMEM_CALLS = frozenset(
    {
        "SmemAllocator",
        "allocate",
        "allocate_tensor",
        "allocate_array",
        "get_dyn_smem",
    }
)


def _resident_threads_per_sm(device: torch.device) -> int:
    """The resident thread slots per SM of ``device`` (the hardware limit
    ``_implicit_pdl_release`` sizes CTAs per SM against), queried from the
    device when it is a live CUDA device and the sm_90+ default otherwise."""
    if device.type == "cuda" and torch.cuda.is_available():
        threads = getattr(
            torch.cuda.get_device_properties(device),
            "max_threads_per_multi_processor",
            0,
        )
        if isinstance(threads, int) and threads > 0:
            return threads
    return DEFAULT_RESIDENT_THREADS_PER_SM


def _smem_per_sm(device: torch.device) -> int:
    """The shared memory of one SM of ``device`` (what ``cute_pdl_spread``
    sizes its pad against), queried from the device when it is a live CUDA
    device and the sm_90+ default otherwise."""
    if device.type == "cuda" and torch.cuda.is_available():
        capacity = getattr(
            torch.cuda.get_device_properties(device),
            "shared_memory_per_multiprocessor",
            0,
        )
        if isinstance(capacity, int) and capacity > 0:
            return capacity
    return DEFAULT_SMEM_PER_SM


def _max_smem_per_block(device: torch.device) -> int:
    """The opt-in shared memory one block of ``device`` may allocate."""
    if device.type == "cuda" and torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(device)
        capacity = max(
            int(getattr(properties, "shared_memory_per_block_optin", 0) or 0),
            int(properties.shared_memory_per_block),
        )
        if capacity > 0:
            return capacity
    return DEFAULT_MAX_SMEM_PER_BLOCK


def _literal_smem_bytes(statements: Sequence[ast.AST]) -> int | None:
    """A lower bound of the shared memory ``statements`` allocate themselves:
    the sum over their ``cute.arch.alloc_smem(<cutlass type>, <literal>, ...)``
    calls (without alignment padding); ``None`` when a call's size or type is
    not a literal the sum can account for, or when the body reserves shared
    memory through a call the sum cannot see (``_OPAQUE_SMEM_CALLS``), so no
    pad is sized against storage of unknown size."""
    total = 0
    for statement in statements:
        for node in ast.walk(statement):
            if not (
                isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            ):
                continue
            if node.func.attr in _OPAQUE_SMEM_CALLS:
                return None
            if ast.unparse(node.func) != "cute.arch.alloc_smem":
                continue
            if len(node.args) < 2:
                return None
            width = _CUTLASS_TYPE_BYTES.get(ast.unparse(node.args[0]))
            count = node.args[1]
            if (
                width is None
                or not isinstance(count, ast.Constant)
                or type(count.value) is not int
            ):
                return None
            total += width * count.value
    return total


def _tcgen05_early_pdl_release(
    fn: DeviceFunction, body: list[ast.stmt]
) -> list[ast.stmt]:
    """``cute_pdl_release`` on a tcgen05 kernel launched as a programmatic
    dependent (``cute_pdl_entry_wait``).

    The lowering waits where it knows each warp's first global read (the
    TMA-load role, the epilogue role, the flat kernel's all-warp wait, or
    the entry wait ``pdl_applies`` adds when those do not cover the
    body) and releases the dependents after the MMA warp's last MMA.  That
    late release leaves the next kernel's launch to the epilogue tail: in
    a back-to-back sequence of short GEMMs the dependent's launch is
    processed in the ~0.3 us the primary still runs, and its cold launch
    state (the catalogue replays after an L2 flush) is exposed on every
    node.  ``entry`` (``auto`` on a one-wave launch) puts a release right
    after every one of the lowering's waits, so the dependents launch as
    soon as this kernel's predecessor is done and overlap this whole
    kernel: fp8 256^3 GEMM nodes (64 CTAs, 16-call graphs replayed
    after an L2 flush) drop from 1.42-1.72 us (bimodal across processes)
    to a steady 1.44 us, cuBLAS 1.49 us.  A release ahead of the waits
    (the kernel's first statement) launches the dependents' dependents
    too, a cascade of parked CTAs that measured 1.54-1.68 us on the same
    nodes, so the release follows the wait as on the generic path.  ``last_wave`` guards that release to the last
    wave's CTAs, ``exit`` and ``none`` leave the family's late release as
    the only trigger (a second trigger from a CTA is a no-op, so the late
    one stays on every placement).
    """
    placement = fn.config.config.get("cute_pdl_release", CUTE_PDL_RELEASE_DEFAULT)
    if placement == "auto":
        placement = _implicit_pdl_release(fn, "entry")
    if placement not in ("entry", "last_wave"):
        return body
    release_call = "cute.arch.griddepcontrol_launch_dependents()"
    prefix: list[ast.stmt] = []
    guard: str | None = None
    if placement == "last_wave":
        with SyntheticLocation():
            statements = _last_wave_release(fn, statement_from_string(release_call))
        if len(statements) == 2:
            prefix = statements[:1]
            guard = ast.unparse(cast("ast.If", statements[1]).test)

    def release() -> ast.stmt:
        with SyntheticLocation():
            if guard is None:
                return statement_from_string(release_call)
            return create(
                ast.If,
                test=expr_from_string(guard),
                body=[statement_from_string(release_call)],
                orelse=[create(ast.Pass)],
            )

    def is_wait(statement: ast.stmt) -> bool:
        return (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and ast.unparse(statement.value) == "cute.arch.griddepcontrol_wait()"
        )

    def visit(statements: list[ast.stmt]) -> list[ast.stmt]:
        result: list[ast.stmt] = []
        for statement in statements:
            result.append(statement)
            if is_wait(statement):
                result.append(release())
                continue
            for field in ("body", "orelse", "finalbody"):
                nested = getattr(statement, field, None)
                if isinstance(nested, list) and nested:
                    setattr(statement, field, visit(nested))
            for handler in getattr(statement, "handlers", None) or ():
                handler.body = visit(handler.body)
        return result

    if body and is_wait(body[0]):
        # The entry wait stays the body's first statement; the guard's
        # CTA index follows it, ahead of the guarded release.
        return [body[0], *prefix, release(), *visit(body[1:])]
    return [*prefix, *visit(body)]


def _late_pdl_wait_names(fn: DeviceFunction) -> set[str]:
    """The kernel-argument names through which this ``cute_pdl`` kernel
    (generic path) can reach a host zero fill that launches as a
    dependent fill kernel: the filled tensors' own arguments and every
    argument whose fake tensor shares their storage (a host view or
    alias, a tensor descriptor built on them).  The first mention of any
    of these bounds the late wait.  Empty when the wait stays at the top:
    no fill kernel (none admitted or every fill elided as dead, so the
    kernel's primary is an unknown predecessor), the tcgen05 entry-wait
    path, no filled tensor the kernel touches, or a kernel with access
    paths the names cannot account for (wrapper-only or extra
    parameters: a lowering's own tensor maps and pointers, the flash
    backward's ``_fbwd_tma_dq``, which reduce into a tensor without
    naming it)."""
    from ..host_zero_fill_launches import zero_fill_launch_tensors
    from ..host_zero_fill_launches import zero_fill_launches_apply

    env = CompileEnvironment.current()
    if fn.cute_state.tcgen05.pdl_entry_wait or not zero_fill_launches_apply(
        fn, env, fn.config
    ):
        return set()
    if (
        fn.wrapper_only_params
        or fn.codegen._extra_params
        or fn.sourceless_prologue_params
    ):
        return set()
    host = HostFunction.current()
    storages = {
        fake.untyped_storage()
        for fake in zero_fill_launch_tensors(host, env, fn.config)
    }
    if not storages:
        return set()
    arguments = [
        *fn._tensor_args.values(),
        *fn._tensor_descriptor_args.values(),
    ]
    return {
        argument.name
        for argument in arguments
        if argument.fake_value.untyped_storage() in storages
    }


def _place_late_pdl_wait(
    body: list[ast.stmt],
    wait: ast.stmt,
    release: list[ast.stmt],
    names: set[str],
) -> list[ast.stmt]:
    """Put the ``cute_pdl`` wait of a kernel whose zero fills launch as
    fill kernels before the first top-level statement of the lowered body
    that mentions a filled tensor, when a statement touching memory (a
    subscript, a load / store / copy / barrier call: a split-K kernel's
    operand pipeline) precedes it; otherwise the kernel keeps the plain
    form, the wait first and the entry release right after it."""

    def mentions(statement: ast.AST) -> bool:
        return any(
            isinstance(node, ast.Name) and node.id in names
            for node in ast.walk(statement)
        )

    def touches_memory(statement: ast.AST) -> bool:
        for node in ast.walk(statement):
            if isinstance(node, ast.Subscript):
                return True
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and any(
                    token in node.func.attr
                    for token in (
                        "load",
                        "store",
                        "copy",
                        "atomic",
                        "cp_async",
                        "tma",
                        "sync",
                        "barrier",
                        "red",
                    )
                )
            ):
                return True
        return False

    release_ids = {id(statement) for statement in release}
    rest = [statement for statement in body if id(statement) not in release_ids]
    first = next((i for i, statement in enumerate(rest) if mentions(statement)), None)
    if first is not None and any(
        isinstance(s, (ast.For, ast.While)) and touches_memory(s) for s in rest[:first]
    ):
        return [*release, *rest[:first], wait, *rest[first:]]
    return [wait, *release, *rest]


def _implicit_pdl_release(fn: DeviceFunction, default: str) -> str:
    """The release placement ``cute_pdl_release: auto`` resolves to:
    ``entry`` (``default``) while the kernel's launch is known to fit in
    one wave of CTAs, ``last_wave`` past it and for a grid the host
    sizes at runtime (``_last_wave_release`` then compares against the
    launched grid).  A persistent launch sized to the resident capacity
    (``_NUM_SM * num_sm_multiplier``, or the tcgen05 schedulers'
    ``min(work, capacity)`` clusters) is one wave by construction.

    A release right after the wait lets the dependents' CTAs become
    resident while this kernel runs.  PTX launches the dependent grid
    only once every CTA of this grid has issued the trigger or exited.
    When the whole grid was resident from the start the dependents only
    fill idle slots and the next kernel starts 0.3-0.5 us earlier
    (24-27% on the catalogue's one-wave reductions and pointwise
    kernels); when the grid runs in several waves a release from every
    CTA stretches it instead (rope fwd on [2, 16, 512, 128]: 1024 CTAs
    on 148 SMs at ``cute_min_blocks_per_mp`` 4, 2.47 us unreleased,
    2.75 released at entry) while a release from the last wave's CTAs
    alone keeps the earlier waves undisturbed and still overlaps the
    dependents' launch with the tail (2.42 us; embedding of 64 x 512
    tokens from a 65536 x 1024 table, 32768 CTAs of 128 threads: 42.7
    us unreleased, 45.9 at entry, 43.0 from the last wave).

    The CTA count is the one the launcher receives: the program-id
    strategy's static grid (``ProgramIDs.static_grid_size``: the sum
    over the roots of a multi-root kernel, the parent of an L2-grouped
    one, ``num_sm * num_sm_multiplier`` for a persistent one) times the
    fused-body or split-K grid multiplier and the cluster size.  The resident
    capacity is the SM count times the CTAs an SM holds, estimated from
    the launch block (``resident_ctas_per_sm``: the thread limit, or
    ``cute_min_blocks_per_mp`` when the config asked for an occupancy).
    A launch shape or SM count codegen cannot know takes ``default``.
    """
    capacity = _pdl_resident_capacity(fn)
    if capacity is None:
        return default
    ctas = _pdl_launch_ctas(fn)
    if ctas is not None and ctas <= capacity:
        return default
    if ctas is None and _pdl_launch_is_persistent(fn):
        return default
    return "last_wave"


def _pdl_launch_is_persistent(fn: DeviceFunction) -> bool:
    """Whether the program-id strategy launches a persistent grid (one
    sized to the device: ``_NUM_SM * num_sm_multiplier`` CTAs, or the
    tcgen05 schedulers' ``min(work, capacity)`` clusters, whose count
    ``static_grid_size`` cannot give)."""
    from ...program_id import ForEachProgramID

    pid = fn.pid
    if pid is None:
        return False
    if isinstance(pid, ForEachProgramID):
        return bool(pid.cases) and pid.cases[0]._is_persistent()
    return pid._is_persistent()


def _pdl_launch_ctas(fn: DeviceFunction) -> int | None:
    """The CTA count of this kernel's launch when it is static: the
    program-id strategy's grid (``ProgramIDs.static_grid_size``) times the
    fused-body or split-K grid multiplier and the cluster size, or the
    count a wrapper plan launches (``cute_state.launch_ctas``) where the
    launcher sizes the grid from the plan; ``None`` for a grid the host
    sizes at runtime."""
    if fn.cute_state.lanes.launch_ctas is not None:
        return fn.cute_state.lanes.launch_ctas
    if fn.pid is None:
        return None
    grid = fn.pid.static_grid_size()
    if grid is None:
        return None
    return (
        grid
        * fn.cute_state.lanes.launch_grid_multiplier
        * max(fn.cute_state.lanes.simt_cluster_n, fn.cute_state.lanes.grid_cluster_n)
    )


def _pdl_resident_capacity(fn: DeviceFunction) -> int | None:
    """How many CTAs of this launch the device holds at once, as far as
    codegen can tell: the SM count times ``resident_ctas_per_sm`` of the
    launch block (the thread limit, or ``cute_min_blocks_per_mp`` when the
    config asked for an occupancy); ``None`` when the SM count is unknown
    (CPU targets)."""
    env = CompileEnvironment.current()
    num_sm = env.config_spec.num_sm
    if num_sm <= 1:
        return None
    min_blocks = fn.config.config.get("cute_min_blocks_per_mp", 0)
    if not isinstance(min_blocks, int):
        min_blocks = 0
    if fn.cute_state.tcgen05.pdl_entry_wait:
        # A tcgen05 GEMM kernel fills the SM's shared memory with its
        # operand ring: one CTA per SM unless the config asked ptxas for
        # an occupancy (the thread limit would call a 1024-CTA flat grid
        # of 192-thread CTAs one wave).
        return num_sm * max(1, min_blocks)
    launch_block = fn.cute_state.lanes.launch_block
    per_sm = resident_ctas_per_sm(
        math.prod(launch_block) if launch_block is not None else None,
        min_blocks,
        max_threads_per_sm=_resident_threads_per_sm(env.device),
    )
    return num_sm * per_sm


def _last_wave_release(fn: DeviceFunction, release: ast.stmt) -> list[ast.stmt]:
    """The statements of ``cute_pdl_release: last_wave``: ``release`` at
    entry, guarded so that only the CTAs of the launch's last wave issue
    it.

    ``griddepcontrol.launch_dependents`` is per CTA and the dependent grid
    launches once every CTA of this grid has issued it or exited.  A CTA
    that never issues it counts at its exit, so guarding the entry
    release to the CTAs whose dispatch-order index (``block_idx`` linear
    in the launch grid, x fastest, the order the hardware dispatches
    them) lies within the resident capacity (``_pdl_resident_capacity``)
    of the grid's end launches the dependents as the last wave starts:
    their CTAs fill the slots the tail frees and the earlier waves kept
    every slot.  A grid known to fit in one wave gets the plain entry
    release, as does a launch whose capacity codegen cannot size; a grid
    the host sizes at runtime compares against ``grid_dim``'s product.
    The guard carries an explicit empty else branch: the FMA fusion pass
    pads every empty statement list with a located ``pass`` (origin
    comments included), and the branch is rendered either way.
    """
    capacity = _pdl_resident_capacity(fn)
    if capacity is None:
        return [release]
    ctas = _pdl_launch_ctas(fn)
    if ctas is not None and ctas <= capacity:
        return [release]
    if ctas is None and _pdl_launch_is_persistent(fn):
        return [release]
    cta = fn.new_var("pdl_cta")
    grid_dim = "cutlass.Int32(cute.arch.grid_dim()[{}])"
    block_idx = "cutlass.Int32(cute.arch.block_idx()[{}])"
    statements = [
        statement_from_string(
            f"{cta} = {block_idx.format(0)} + {grid_dim.format(0)} * "
            f"({block_idx.format(1)} + {grid_dim.format(1)} * {block_idx.format(2)})"
        )
    ]
    if ctas is not None:
        first = f"{ctas - capacity}"
    else:
        first = (
            f"{grid_dim.format(0)} * {grid_dim.format(1)} * {grid_dim.format(2)}"
            f" - {capacity}"
        )
    statements.append(
        create(
            ast.If,
            test=expr_from_string(f"{cta} >= {first}"),
            body=[release],
            orelse=[create(ast.Pass)],
        )
    )
    return statements


def _pdl_spread_statements(
    fn: DeviceFunction, body: Sequence[ast.AST]
) -> list[ast.stmt]:
    """The statements of ``cute_pdl_spread``: one shared allocation that
    raises the kernel's footprint to the smallest one at which more
    CTAs than ``ceil(CTAs / SMs)`` no longer fit on an SM
    (``spread_pad_bytes``), kept live by ``_helion_pdl_spread_keep``;
    nothing for a grid the host sizes at runtime, a body whose own
    allocations already enforce that density or reserve shared memory
    the sum cannot account for (a non-literal size, or the storage a
    family allocates through the DSL's ``SmemAllocator``: the fa4 flash
    backward holds the per-block maximum that way, and a pad on top of
    it would fail every launch), a density the pad cannot hold with the
    slack it keeps per CTA, or an SM count codegen cannot know.

    A programmatic dependent launches while its primary runs and the
    hardware dispatches its CTAs onto whatever SMs have room: when the
    primary's CTAs leave none beside them, every CTA of a small
    dependent lands on the few SMs the primary left idle and runs
    packed there once the primary completes (vanilla linear attention
    forward: the 128 one-warp CTAs of the chunk state kernel on the
    20 SMs the 128-CTA, 164 KB output kernel leaves idle, 7.45 us per
    call against 4.1 spread).  At one wave's density the CTAs beyond
    the idle SMs wait for the primary's to retire and take their SMs
    one each, which costs nothing on an idle device (the dispatcher
    spreads a fresh grid the same way) and lets the search's isolated
    timing read the knob as a tie.
    """
    env = CompileEnvironment.current()
    num_sm = env.config_spec.num_sm
    ctas = _pdl_launch_ctas(fn)
    own = _literal_smem_bytes(body)
    if num_sm <= 1 or ctas is None or own is None:
        return []
    pad = spread_pad_bytes(
        ctas,
        num_sm,
        own,
        smem_per_sm=_smem_per_sm(env.device),
        max_smem_per_block=_max_smem_per_block(env.device),
    )
    if pad is None:
        return []
    with SyntheticLocation():
        return [
            statement_from_string(
                f"{SPREAD_PAD_NAME} = cute.arch.alloc_smem("
                f"cutlass.Uint8, {pad}, alignment=128)"
            ),
            statement_from_string(f"{SPREAD_KEEP_HELPER}({SPREAD_PAD_NAME}.toint())"),
        ]


def _pdl_spread_index(body: Sequence[ast.AST]) -> int:
    """Where the pad of ``cute_pdl_spread`` goes: right after the
    kernel's own wait and release (the plain release, or the
    ``last_wave`` CTA index and guard), which the final selection reads
    as the two statements after the wait (``generic_release_rendered``);
    after the wait alone when the kernel does not release early."""
    statements = [ast.unparse(statement) for statement in body]
    index = statements.index("cute.arch.griddepcontrol_wait()") + 1
    if index < len(statements) and statements[index] == (
        "cute.arch.griddepcontrol_launch_dependents()"
    ):
        return index + 1
    if index + 1 < len(body):
        guard_index = body[index]
        if (
            isinstance(guard_index, ast.Assign)
            and ast.unparse(guard_index.targets[0]).startswith("pdl_cta")
            and isinstance(body[index + 1], ast.If)
        ):
            return index + 2
    return index


def pdl_applies(fn: DeviceFunction) -> bool:
    """Whether ``cute_pdl`` shapes this kernel.

    The knob launches the kernel as a programmatic dependent of the
    previous kernel in the stream and splices ``griddepcontrol_wait()``
    in ahead of the scalar preamble, the preamble, the cluster sync and
    the body, so every global read, write, atomic and bulk copy, on
    every path, follows the wait; the passes that run afterwards may
    hoist register or shared allocations above it, never a memory
    access.  The generic path also places the kernel's own release
    (``cute_pdl_release``); the tcgen05 entry-wait path below gets the
    wait alone here, its early release follows every wait of the body
    (``_tcgen05_early_pdl_release``).  It stays out of kernels
    that already take part in dependent launch, which generate exactly
    the knob-off code: native plans launched with ``use_pdl``, and
    bodies that wait on or release their dependents themselves.  A release ahead of the knob's wait
    would let the dependents start before this kernel's predecessors
    finish; the materialized-fission producer gets its release inserted
    after this function runs, so ``_stage_config`` keeps the knob away
    from it.  ``griddepcontrol`` needs sm_90, so older targets are left
    alone as well.
    """
    env = CompileEnvironment.current()
    if fn.config.config.get("cute_pdl") is not True:
        return False
    capability = env.config_spec.target_device_capability
    if capability is None or capability < (9, 0):
        return False
    if fn.cute_state.tcgen05.pdl_entry_wait:
        # A tcgen05 kernel launched with ``use_pdl`` through its wrapper
        # plan places late waits where its lowering knows the first
        # global read of each warp (the flat kernel's prefetch, the
        # persistent TMA-load and epilogue roles); when the generated
        # body reads global memory ahead of those on some path (the
        # grouped layout reads of the prologue, a scheduler-warp or
        # grouped persistent builder without role waits, a root-body
        # load ahead of the K loop) it gets this wait at the top of the
        # body instead. Its releases all follow either wait.
        from ..pdl_coverage import waits_cover_global_reads

        return not waits_cover_global_reads([*fn.preamble, *fn.body])
    if any(plan.get("use_pdl") for plan in fn.codegen.cute_wrapper_plans):
        return False
    return not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr
        in ("griddepcontrol_wait", "griddepcontrol_launch_dependents")
        for stmt in [*fn.preamble, *fn.body]
        for node in ast.walk(stmt)
    )
