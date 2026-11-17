"""Register quotas of the paired paged-attention schedule's warp roles.

The paired body gives every warp role its own register count with
``setmaxnreg``: the compute groups raise theirs with what the conversion and
control warps give up. The instruction moves registers between the warps of
one CTA, so the roles' quotas must fit the pool that CTA launched with. A
split that asks for more never gets its increase, and the kernel hangs.

This module states the roles, the pool and the admitted splits once. The
config contract, the planner and the search surface admit only those splits,
and the device template asserts the pool on the quotas it is rendered with,
which makes this file a source dependency of the paged-attention wrapper. It
imports nothing, so the template can import it.
"""

from __future__ import annotations

# 32-bit registers of one SM's register file, shared by its resident CTAs.
SM_REGISTERS = 1 << 16
THREADS_PER_WARP = 32
# ``setmaxnreg`` takes a multiple of eight in this range (PTX).
SETMAXNREG_STEP = 8
SETMAXNREG_MIN = 24
SETMAXNREG_MAX = 256

# The paired body's warp roles in quota order, with their warps: two
# 128-thread compute groups (warps 0-7), the K conversion warps (8-11), the V
# conversion warps (12-15) and the control warps (16-19: QK issuer, loader,
# PV issuer, and warp 19, which is the value-metadata actor, a balanced
# partition's spare warp or, with one unit per CTA, the boundary prefetch).
# A persistent work partition changes what the roles walk, not their warps,
# so one pool check covers every partition.
PAIRED_ROLES: tuple[str, ...] = ("compute", "K conversion", "V conversion", "control")
PAIRED_ROLE_WARPS: tuple[int, ...] = (8, 4, 4, 4)
# The paired body's shared and tensor memory admit one CTA per SM.
PAIRED_MIN_BLOCKS_PER_MP = 1


def format_register_quotas(quotas: tuple[int, ...]) -> str:
    return "/".join(str(quota) for quota in quotas)


# ``cute_mma_dag_register_quotas``: the registers per thread of each role of
# the paired body, written ``compute/K conversion/V conversion/control``.
# The split changes no result. A config may name one of the choices below;
# every other value is an InvalidConfig. The first choice is the default: a
# config that leaves the option out, or names the default, renders the same
# kernel. Every other schedule pins the option to the default.
# The choices are splits that were measured, and the search picks among them:
# which one is fastest is an outcome of the assembler's scheduling of the
# roles' unchanged code, so no rule derives a split from the others.
# 128/72/72/48 was measured as well and is left out: it was the fastest split
# of no kernel and slower than the default for some.
# Every choice must fit the CTA's register pool
# (``paired_register_quotas_reason``): the tests check each one, and the
# template asserts it on the quotas of every render.
MMA_DAG_REGISTER_QUOTAS_KEY = "cute_mma_dag_register_quotas"
# The default split: the template's parameter default (a config that leaves
# the split out renders no argument), and its spelling as the option's value.
PAIRED_DEFAULT_REGISTER_QUOTAS: tuple[int, int, int, int] = (144, 72, 72, 48)
MMA_DAG_REGISTER_QUOTAS_DEFAULT = format_register_quotas(PAIRED_DEFAULT_REGISTER_QUOTAS)
MMA_DAG_REGISTER_QUOTAS_CHOICES: tuple[str, ...] = (
    MMA_DAG_REGISTER_QUOTAS_DEFAULT,
    "136/80/80/48",
    "136/72/72/48",
    "120/96/96/48",
)


def launch_registers(threads: int, min_blocks_per_mp: int) -> int:
    """Registers per thread a CTA of ``threads`` threads launches with when
    the SM must hold ``min_blocks_per_mp`` of them: the largest ``setmaxnreg``
    count at which they share the register file.

    640 threads at one CTA per SM launch with 96 (the ``REG`` the assembler
    records for the paired kernel), 384 threads at two per SM with 80.
    """
    share = SM_REGISTERS // (threads * min_blocks_per_mp)
    return min(SETMAXNREG_MAX, share // SETMAXNREG_STEP * SETMAXNREG_STEP)


def paired_launch_threads() -> int:
    """Threads of the paired body's CTA."""
    return sum(PAIRED_ROLE_WARPS) * THREADS_PER_WARP


def paired_register_pool() -> int:
    """Registers the paired body's CTA owns from launch to exit: what its
    threads launch with (640 threads at 96 registers, 61,440)."""
    threads = paired_launch_threads()
    return threads * launch_registers(threads, PAIRED_MIN_BLOCKS_PER_MP)


def parse_register_quotas(value: object) -> tuple[int, int, int, int] | None:
    """The four quotas of a config value in its canonical spelling, or None."""
    if type(value) is not str:
        return None
    parts = value.split("/")
    if len(parts) != len(PAIRED_ROLES) or not all(
        part.isascii() and part.isdigit() for part in parts
    ):
        return None
    compute, key, values, control = (int(part) for part in parts)
    quotas = (compute, key, values, control)
    return quotas if format_register_quotas(quotas) == value else None


def paired_register_quotas_reason(quotas: tuple[int, ...]) -> str | None:
    """Why the paired body cannot run with these per-role quotas, or None.

    The body raises the compute groups above the launch count and lowers the
    other roles to theirs, each with one ``setmaxnreg``; the raise is granted
    only from what the pool holds.
    """
    if len(quotas) != len(PAIRED_ROLES):
        return f"expected {len(PAIRED_ROLES)} quotas ({', '.join(PAIRED_ROLES)})"
    for role, quota in zip(PAIRED_ROLES, quotas, strict=True):
        if (
            type(quota) is not int
            or quota % SETMAXNREG_STEP
            or not SETMAXNREG_MIN <= quota <= SETMAXNREG_MAX
        ):
            return (
                f"the {role} quota {quota!r} is not a multiple of "
                f"{SETMAXNREG_STEP} in [{SETMAXNREG_MIN}, {SETMAXNREG_MAX}]"
            )
    launch = launch_registers(paired_launch_threads(), PAIRED_MIN_BLOCKS_PER_MP)
    if quotas[0] <= launch:
        return (
            f"the compute quota {quotas[0]} is not above the {launch} registers "
            "a thread launches with"
        )
    for role, quota in zip(PAIRED_ROLES[1:], quotas[1:], strict=True):
        if quota > launch:
            return (
                f"the {role} quota {quota} is above the {launch} registers a "
                "thread launches with"
            )
    needed = sum(
        warps * THREADS_PER_WARP * quota
        for warps, quota in zip(PAIRED_ROLE_WARPS, quotas, strict=True)
    )
    pool = paired_register_pool()
    if needed > pool:
        return (
            f"the roles need {needed} registers and the CTA's pool holds {pool} "
            f"({paired_launch_threads()} threads at {launch})"
        )
    return None
