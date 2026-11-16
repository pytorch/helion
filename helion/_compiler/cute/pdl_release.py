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
pins, seeds and users. Proved paired capture overlap can explicitly request
``loader_tail``: the audited loader releases four logical pages before its
end, with an unconditional release after the loop for short/empty ranges.
Ordinary search keeps release at ``auto`` and capture overlap disabled.
Its dependency waits and capture-storage proof are unchanged.
The tcgen05 GEMM kernels, whose lowering waits in
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
one), sized against what the body allocates as ``smem_footprint`` bounds
it: its own allocations and the buffers of the shared-memory reduces it
calls (``spread_pad_bytes`` says which bound).  Other device helpers that
allocate on their own are not seen; they only raise the total, and the
slack each CTA keeps covers small ones.
"""

from __future__ import annotations

import ast

from .target_facts import DEFAULT_MAX_SMEM_PER_BLOCK

CUTE_PDL_RELEASE_CHOICES: tuple[str, ...] = (
    "auto",
    "entry",
    "none",
    "exit",
    "last_wave",
    "loader_tail",
)
CUTE_PDL_RELEASE_DEFAULT = CUTE_PDL_RELEASE_CHOICES[0]
CUTE_PDL_CAPTURE_RELEASE_CHOICES: tuple[str, ...] = (
    CUTE_PDL_RELEASE_DEFAULT,
    "entry",
    "loader_tail",
)
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
# Shared memory per SM of the sm_90 / sm_100 parts (228 KiB), the default
# when the device cannot be queried, and the slice of an SM's shared memory
# the system reserves for every resident CTA.
DEFAULT_SMEM_PER_SM = 233472
SMEM_RESERVED_PER_CTA = 1024
# Shared allocations are laid out in 128-byte units, and every resident CTA
# of a padded kernel keeps one 1 KB unit of slack beyond its footprint and
# the alignment padding ahead of the body's allocations: the pad's own
# alignment and the allocations of the device helpers ``smem_footprint``
# does not model must not cost a resident CTA at the density the pad targets.
SMEM_ALLOCATION_UNIT = 128
SPREAD_SLACK_PER_CTA = 1024
# The pad allocation's variable and the helper that keeps it live.
SPREAD_PAD_NAME = "pdl_spread_pad"
SPREAD_KEEP_HELPER = "_helion_pdl_spread_keep"


def spread_pad_bytes(
    ctas: int,
    num_sm: int,
    own_low: int,
    own_high: int,
    *,
    own_padding: int = 0,
    smem_per_sm: int = DEFAULT_SMEM_PER_SM,
    max_smem_per_block: int = DEFAULT_MAX_SMEM_PER_BLOCK,
) -> int | None:
    """The shared memory to add to a kernel whose body keeps between
    ``own_low`` and ``own_high`` bytes (``smem_footprint``'s bounds, the
    buffers of the reduce helpers it calls included; the compiler may drop
    what the lower bound leaves out) so that at most ``ceil(ctas / num_sm)``
    of its CTAs are resident per SM, or ``None`` when its own allocation
    already enforces that density or no pad within the per-block maximum can.

    The footprint is the smallest one at which ``cap + 1`` CTAs no longer fit
    in an SM's shared memory, each CTA counting its system-reserved slice:
    ``(cap + 1) * (footprint + reserved) > smem_per_sm``.  With one CTA per
    SM the pad is the footprint less ``own_low``: the padded kernel reaches
    the footprint even when the compiler drops every allocation the lower
    bound leaves out.  When those allocations are live it overshoots the
    footprint by them, which costs L1 carve-out and room for a dependent
    launch's early CTAs, though no resident CTA of its own (``own_low``
    leaves out whatever the walk cannot show the compiler keeps on every
    block, ``smem_footprint``).  With more CTAs per SM a pad that large
    would cost resident CTAs when those allocations are live, so the pad is
    the footprint less ``own_high``, and a kernel whose compiler drops
    allocations stays below the footprint, with more CTAs per SM.

    The pad is declined unless ``cap`` CTAs fit with ``SPREAD_SLACK_PER_CTA``
    to spare each, laid out with the alignment padding ahead of the body's
    allocations (``own_padding``) and rounded up to the allocation unit
    (past a cap of 14 on a 228 KiB SM no footprint leaves that slack), and
    ``own_high`` plus the pad stays within the per-block maximum with the
    padding ahead of each (the pad's own ``SMEM_ALLOCATION_UNIT`` alignment
    included).  The allocations ``own_high`` leaves out (helpers
    ``smem_footprint`` does not model) only raise the total.
    """
    if ctas <= 0 or num_sm <= 0 or not 0 <= own_low <= own_high:
        return None
    cap = -(-ctas // num_sm)
    footprint = smem_per_sm // (cap + 1) - SMEM_RESERVED_PER_CTA + 1
    own = own_low if cap == 1 else own_high
    if footprint <= own or footprint > max_smem_per_block:
        return None
    allocated = (
        -(-(footprint + own_padding) // SMEM_ALLOCATION_UNIT) * SMEM_ALLOCATION_UNIT
    )
    if cap * (allocated + SMEM_RESERVED_PER_CTA + SPREAD_SLACK_PER_CTA) > smem_per_sm:
        return None
    pad = footprint - own
    if own_high + own_padding + pad + SMEM_ALLOCATION_UNIT - 1 > max_smem_per_block:
        return None
    return pad


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
    The audited capture wrapper's conditional EARLY wait has the same entry
    release ordering; its metadata identifies the exact mode parameter.
    """
    try:
        module = ast.parse(source)
    except SyntaxError:
        return False
    capture_waits: dict[str, str] = {}
    for node in module.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Attribute)
            and node.targets[0].attr == "_helion_cute_wrapper_plans"
        ):
            if "'use_pdl': True" in ast.unparse(node.value):
                return False
            if not isinstance(node.targets[0].value, ast.Name):
                continue
            try:
                plans = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            if not isinstance(plans, list):
                continue
            for plan in plans:
                if (
                    isinstance(plan, dict)
                    and plan.get("kind") == "helion_paged_attention"
                    and plan.get("capture_pdl_contract") == "paired-output-stores-v1"
                    and isinstance(mode := plan.get("capture_pdl_mode_param"), str)
                ):
                    capture_waits[node.targets[0].value.id] = mode
    kernels = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and any(ast.unparse(d) == "cute.kernel" for d in node.decorator_list)
        and not node.name.startswith(ZERO_FILL_KERNEL_PREFIX)
    ]
    for kernel in kernels:
        waits = [
            i
            for i, statement in enumerate(kernel.body)
            if _is_wait(statement)
            or _is_capture_wait(statement, capture_waits.get(kernel.name))
        ]
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


def _is_capture_wait(statement: ast.stmt, mode: str | None) -> bool:
    """Recognize only the audited wrapper's EARLY branch, not arbitrary waits."""
    return (
        mode is not None
        and isinstance(statement, ast.If)
        and not statement.orelse
        and len(statement.body) == 1
        and _is_wait(statement.body[0])
        and isinstance(statement.test, ast.Compare)
        and isinstance(statement.test.left, ast.Name)
        and statement.test.left.id == mode
        and len(statement.test.ops) == 1
        and isinstance(statement.test.ops[0], ast.Eq)
        and len(statement.test.comparators) == 1
        and isinstance(statement.test.comparators[0], ast.Constant)
        and type(statement.test.comparators[0].value) is int
        and statement.test.comparators[0].value == 0
    )


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
