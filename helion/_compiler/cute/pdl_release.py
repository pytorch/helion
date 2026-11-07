"""Where a ``cute_pdl`` kernel releases its own dependents.

``cute_pdl_release`` names the placement of ``griddepcontrol_launch_dependents``
in a kernel launched as a programmatic dependent (``cute_pdl``): ``entry``
right after the kernel's own wait, ``exit`` after the body's last statement,
``none`` leaves the implicit trigger at grid completion, ``last_wave``
releases at entry in the CTAs of the launch's last wave only (the CTAs whose
dispatch-order index is within the resident capacity of the grid's end;
every earlier CTA counts as released when it exits, so the dependents launch
as the last wave starts and never hold slots the earlier waves need; a grid
that fits in one wave releases at plain entry), and ``auto`` (the default)
picks ``entry`` while the launch grid is known to fit in one wave of
resident CTAs and ``last_wave`` past it or when the host sizes the grid at
runtime (``passes.pdl._implicit_pdl_release``).  The placement follows
the grid, so the search leaves it at ``auto``; the explicit values serve
pins, seeds and users.  The tcgen05 GEMM kernels, whose lowering waits in
its warp roles and releases after the MMA warp's last MMA, take the same
placements: ``entry`` and ``last_wave`` add the (guarded) release right
after each of those waits, ``exit`` and ``none`` keep the late release
alone (``passes.pdl._tcgen05_early_pdl_release``).  The choices live
here, a leaf module, because both the config spec (the fragment and
normalization) and the device-function codegen read them.

``cute_pdl_spread`` (a boolean, off by default) keeps the CTAs of a
programmatic dependent from packing: the dependent grid launches while its
primary still runs, and the hardware dispatches its CTAs onto whatever SMs
have room at that moment.  When the primary's CTAs leave no room beside
them (a 164 KB operand ring in a 196 KB carveout leaves no 26 KB slot) every
CTA of a small dependent lands on the few SMs the primary left idle, six or
seven to an SM, and runs there when the primary completes instead of
spreading over the SMs it vacates (vanilla linear attention forward, chunk
state kernel of 128 one-warp CTAs behind the 128-CTA output kernel: 7.45 us
per call packed, 4.1 spread).  The knob pads the kernel's shared memory
(``spread_pad_bytes``) so that at most ``ceil(CTAs / SMs)`` of its CTAs fit
on one SM, its density on an idle device, which an early launch then cannot
exceed: the CTAs beyond the idle SMs wait for the primary's to retire and
take their SMs one each.  The pad is one more static allocation of the
kernel body kept live by ``_helion_pdl_spread_keep`` (the DSL drops a dead
one), so the launch's dynamic shared memory grows by it whatever the
helpers the body calls allocate on their own.
"""

from __future__ import annotations

import ast

CUTE_PDL_RELEASE_CHOICES: tuple[str, ...] = (
    "auto",
    "entry",
    "none",
    "exit",
    "last_wave",
)
CUTE_PDL_RELEASE_DEFAULT = CUTE_PDL_RELEASE_CHOICES[0]
# The hardware limits every target with ``griddepcontrol`` (sm_90 and up)
# shares: resident CTAs per SM and, when the device cannot be queried, the
# resident threads per SM (H100, B200).
MAX_RESIDENT_CTAS_PER_SM = 32
DEFAULT_RESIDENT_THREADS_PER_SM = 2048


def resident_ctas_per_sm(
    threads_per_cta: int | None,
    min_blocks_hint: int,
    *,
    max_threads_per_sm: int = DEFAULT_RESIDENT_THREADS_PER_SM,
) -> int:
    """How many CTAs of the launch an SM holds at once, as far as codegen can
    tell.

    The thread limit (``max_threads_per_sm // threads_per_cta``, at most
    ``MAX_RESIDENT_CTAS_PER_SM``) is a hard ceiling.  ``cute_min_blocks_per_mp``,
    when set, is the occupancy the config asked ptxas for (it caps registers
    so that many CTAs fit): it is taken as the estimate, bounded by the
    thread limit, since the measured multi-wave regression (rope, 1024 CTAs
    at ``min_blocks`` 4 on 148 SMs) is one the thread limit alone would call
    one wave.  Without the hint the thread limit is the estimate; without the
    launch shape (an argument-sized axis) one CTA per SM is all that is
    known.
    """
    cap = MAX_RESIDENT_CTAS_PER_SM
    if threads_per_cta is not None and threads_per_cta > 0:
        cap = max(1, min(cap, max_threads_per_sm // threads_per_cta))
    if min_blocks_hint > 0:
        return max(1, min(min_blocks_hint, cap))
    if threads_per_cta is None or threads_per_cta <= 0:
        return 1
    return cap


CUTE_PDL_SPREAD_KEY = "cute_pdl_spread"
# Shared memory per SM and the opt-in maximum per block of the sm_90 / sm_100
# parts (228 KiB and 227 KiB), the defaults when the device cannot be
# queried, and the slice of an SM's shared memory the system reserves for
# every resident CTA.
DEFAULT_SMEM_PER_SM = 233472
DEFAULT_MAX_SMEM_PER_BLOCK = 232448
SMEM_RESERVED_PER_CTA = 1024
# Shared allocations are laid out in 128-byte units, and every resident CTA
# of a padded kernel keeps one 1 KB unit of slack beyond its footprint: the
# alignment padding of the body's allocations and the allocations the
# runtime helpers make on their own (neither counted in the footprint) must
# not cost a resident CTA at the density the pad targets.
SMEM_ALLOCATION_UNIT = 128
SPREAD_SLACK_PER_CTA = 1024
# The pad allocation's variable and the helper that keeps it live.
SPREAD_PAD_NAME = "pdl_spread_pad"
SPREAD_KEEP_HELPER = "_helion_pdl_spread_keep"


def spread_pad_bytes(
    ctas: int,
    num_sm: int,
    own_smem_bytes: int,
    *,
    smem_per_sm: int = DEFAULT_SMEM_PER_SM,
    max_smem_per_block: int = DEFAULT_MAX_SMEM_PER_BLOCK,
) -> int | None:
    """The shared memory to add to a kernel using ``own_smem_bytes`` (a lower
    bound: what its body allocates itself) so that at most
    ``ceil(ctas / num_sm)`` of its CTAs are resident per SM, or ``None``
    when its own allocation already enforces that density or no pad within
    the per-block maximum can.

    The footprint is the smallest one at which ``cap + 1`` CTAs no longer fit
    in an SM's shared memory, each CTA counting its system-reserved slice:
    ``(cap + 1) * (footprint + reserved) > smem_per_sm``, while ``cap`` of
    them still fit with ``SPREAD_SLACK_PER_CTA`` to spare each, the footprint
    rounded up to the allocation unit (past a cap of 14 on a 228 KiB SM no
    footprint leaves that slack and the pad is declined: the alignment
    padding or a helper's allocation would otherwise cost a resident CTA).
    The pad is the footprint less what the body allocates; the helpers the
    body calls may allocate more, which only raises the total (the launch
    requests the kernel's whole static allocation), never below the
    footprint.
    """
    if ctas <= 0 or num_sm <= 0 or own_smem_bytes < 0:
        return None
    cap = -(-ctas // num_sm)
    footprint = smem_per_sm // (cap + 1) - SMEM_RESERVED_PER_CTA + 1
    if footprint <= own_smem_bytes or footprint > max_smem_per_block:
        return None
    allocated = -(-footprint // SMEM_ALLOCATION_UNIT) * SMEM_ALLOCATION_UNIT
    if cap * (allocated + SMEM_RESERVED_PER_CTA + SPREAD_SLACK_PER_CTA) > smem_per_sm:
        return None
    return footprint - own_smem_bytes


_WAIT_CALL = "cute.arch.griddepcontrol_wait()"
_RELEASE_CALL = "cute.arch.griddepcontrol_launch_dependents()"


def _is_bare_call(node: ast.AST, dotted_name: str) -> bool:
    """``node`` is the expression statement ``dotted_name()`` (no arguments)."""
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    call = node.value
    if call.args or call.keywords:
        return False
    parts: list[str] = []
    func: ast.expr = call.func
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if not isinstance(func, ast.Name):
        return False
    parts.append(func.id)
    return ".".join(reversed(parts)) == dotted_name


def _is_wait(node: ast.AST) -> bool:
    return _is_bare_call(node, "cute.arch.griddepcontrol_wait")


def _is_release(node: ast.AST) -> bool:
    return _is_bare_call(node, "cute.arch.griddepcontrol_launch_dependents")


# The fill kernels a ``cute_pdl`` kernel's host zero fills launch as
# (``host_zero_fill_launches``) always wait and release at entry; the
# release check below reads the kernel proper, not them.
ZERO_FILL_KERNEL_PREFIX = "_helion_zero_fill_"


def generic_release_rendered(source: str) -> bool:
    """Whether the generated module ``source`` renders the generic early
    release of ``cute_pdl_release`` (``entry``, or ``last_wave``'s guard).

    The generic path places its release as a top-level statement of the
    kernel body right after the top-level ``griddepcontrol_wait`` (``entry``;
    ``last_wave`` puts the CTA index assignment and the guard whose only
    statement is the release there).  The families that own their dependent
    launch (tcgen05, warp-MMA, collective, block-scaled) release under their
    own role predicates or after their last MMA, and a kernel on the generic
    path with ``none`` or ``exit`` releases nowhere early.  A tcgen05 kernel
    (its wrapper plan launches with ``use_pdl``) also releases right after
    each of its waits under ``entry``, the entry wait included, but stays
    outside this preference: its back-to-back timing in the autotuner sees
    the knob directly, and ranking its structurally different finalists by
    the release once picked a CLC-persistent kernel 7% behind in the
    catalogue's regime.  The autotuner's final selection asks this of a
    finalist's rendered source (``CuteBackend.finalist_preference``).
    """
    try:
        module = ast.parse(source)
    except SyntaxError:
        return False
    for node in module.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Attribute)
            and node.targets[0].attr == "_helion_cute_wrapper_plans"
            and "'use_pdl': True" in ast.unparse(node.value)
        ):
            return False
    kernels = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and any(ast.unparse(d) == "cute.kernel" for d in node.decorator_list)
        and not node.name.startswith(ZERO_FILL_KERNEL_PREFIX)
    ]
    for kernel in kernels:
        waits = [i for i, statement in enumerate(kernel.body) if _is_wait(statement)]
        if not waits:
            continue
        index = waits[0]
        if index > 0:
            # The late wait behind this kernel's fill kernels: the entry
            # release (plain, or ``last_wave``'s CTA index and guard) leads
            # the body, the wait precedes the first touch of a filled tensor.
            leading = kernel.body[:2]
            if _is_release(kernel.body[0]):
                return True
            if (
                len(leading) == 2
                and isinstance(leading[0], ast.Assign)
                and len(leading[0].targets) == 1
                and isinstance(leading[0].targets[0], ast.Name)
                and leading[0].targets[0].id.startswith("pdl_cta")
                and isinstance(leading[1], ast.If)
                and len(leading[1].body) == 1
                and _is_release(leading[1].body[0])
            ):
                return True
        following = kernel.body[index + 1 : index + 3]
        if not following:
            continue
        if _is_release(following[0]):
            return True
        first = following[0]
        if (
            len(following) == 2
            and isinstance(first, ast.Assign)
            and len(first.targets) == 1
            and isinstance(first.targets[0], ast.Name)
            and first.targets[0].id.startswith("pdl_cta")
            and isinstance(following[1], ast.If)
            and len(following[1].body) == 1
            and _is_release(following[1].body[0])
        ):
            return True
    return False


def spread_rendered(source: str) -> bool:
    """Whether the generated module ``source`` pads a kernel's shared memory
    for ``cute_pdl_spread`` (a kernel body that keeps a pad allocation live
    through ``_helion_pdl_spread_keep``); the knob is inert, and the module
    free of the pad, when the kernel's own allocation already holds its CTAs
    to one wave's density or the grid is sized at runtime."""
    try:
        module = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(module):
        if (
            isinstance(node, ast.FunctionDef)
            and any(ast.unparse(d) == "cute.kernel" for d in node.decorator_list)
            and any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == SPREAD_KEEP_HELPER
                for statement in node.body
                for call in ast.walk(statement)
            )
        ):
            return True
    return False
