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
runtime (``DeviceFunction._implicit_pdl_release``).  The placement follows
the grid, so the search leaves it at ``auto``; the explicit values serve
pins, seeds and users.  The choices live here, a leaf module, because both
the config spec (the fragment and normalization) and the device-function
codegen read them.
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


_WAIT_CALL = "cute.arch.griddepcontrol_wait()"
_RELEASE_CALL = "cute.arch.griddepcontrol_launch_dependents()"


def generic_release_rendered(source: str) -> bool:
    """Whether the generated module ``source`` renders the generic early
    release of ``cute_pdl_release`` (``entry``, or ``last_wave``'s guard).

    The generic path places its release as a top-level statement of the
    kernel body right after the top-level ``griddepcontrol_wait`` (``entry``;
    ``last_wave`` puts the CTA index assignment and the guard whose only
    statement is the release there).  The families that own their dependent
    launch (tcgen05, warp-MMA, collective, block-scaled) release under their
    own role predicates or after their last MMA, never there, and a kernel
    on the generic path with ``none`` or ``exit`` releases nowhere early.
    The autotuner's final selection asks this of a finalist's rendered
    source (``CuteBackend.finalist_preference``).
    """
    try:
        module = ast.parse(source)
    except SyntaxError:
        return False
    kernels = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and any(ast.unparse(d) == "cute.kernel" for d in node.decorator_list)
    ]
    for kernel in kernels:
        statements = [ast.unparse(statement) for statement in kernel.body]
        if _WAIT_CALL not in statements:
            continue
        index = statements.index(_WAIT_CALL)
        following = kernel.body[index + 1 : index + 3]
        if not following:
            continue
        if statements[index + 1] == _RELEASE_CALL:
            return True
        first = following[0]
        if (
            len(following) == 2
            and isinstance(first, ast.Assign)
            and len(first.targets) == 1
            and isinstance(first.targets[0], ast.Name)
            and first.targets[0].id.startswith("pdl_cta")
            and isinstance(following[1], ast.If)
            and [ast.unparse(s) for s in following[1].body] == [_RELEASE_CALL]
        ):
            return True
    return False
