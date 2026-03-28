from __future__ import annotations

import triton
import triton.language as tl

__all__ = ["triton_wait_signal"]


@triton.jit
def triton_wait_signal(
    addr: tl.tensor,
    expect: tl.constexpr,
    update: tl.constexpr,
    sem: tl.constexpr,
    scope: tl.constexpr,
    op: tl.constexpr,
    skip_sync: tl.constexpr,
    # pyrefly: ignore [bad-function-definition]
    sync_before: tl.constexpr = False,
) -> None:
    """
    Wait for a global memory barrier to reach the expected value.

    This function implements a spin-wait loop that continuously checks a memory location
    until it reaches the expected value, providing synchronization across CTAs.

    Args:
        addr: Memory address of the barrier to wait on (Must be a scalar)
        expect: Expected value to wait for
        update: Update the barrier with once acquired
        sem: Memory semantics for the atomic operation. Options: "acquire", "relaxed".
        scope: Scope of the atomic operation. Options: "gpu", "sys"
        op: Atomic operation type: "ld", "atomic_cas"
        skip_sync: Skip CTA sync after acquiring the barrier (default: False)
        sync_before: Add a CTA sync before the wait (default: False)
    """
    tl.static_assert(
        # pyrefly: ignore [missing-attribute]
        addr.type.is_ptr(),
        "Barrier address must be a scalar pointer. ",
    )

    tl.static_assert(
        (sem == "acquire" or sem == "relaxed") or sem == "release",
        "Invalid memory semantic. options: 'acquire', 'relaxed', 'release'. ",
    )
    tl.static_assert(
        scope == "gpu" or scope == "sys", "Invalid scope. options: 'gpu', 'sys'. "
    )
    tl.static_assert(
        op == "ld" or op == "atomic_cas",
        "Invalid op. options: 'ld', 'atomic_cas'. ",
    )

    if sync_before:
        tl.inline_asm_elementwise(
            "bar.sync 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
        )

    # Spin-wait loop:
    #   Uses atomic_add with update=0 for ld.global.{sem}.{scope}
    #   Triton generates smem broadcasting of tl.atomic_add return value in ptx,
    #   but it is optimized away by ptxas in SASS, hence no performance overhead.
    if op == "ld":
        while tl.atomic_add(addr, 0, sem=sem, scope=scope) != expect:
            pass
    elif op == "atomic_cas":
        while tl.atomic_cas(addr, expect, update, sem=sem, scope=scope) != expect:
            pass
    else:
        raise NotImplementedError(
            f"Unsupported op '{op}' for wait signal on gmem barrier. "
        )

    if not skip_sync:
        tl.inline_asm_elementwise(
            "bar.sync 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
        )
    # tl.debug_barrier() cause significant performance loss. (Perhaps breaks triton prefetching?)
