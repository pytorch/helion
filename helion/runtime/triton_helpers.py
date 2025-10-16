from __future__ import annotations

import triton
import triton.language as tl

__all__ = ["triton_send_signal", "triton_wait_multiple_signal", "triton_wait_signal"]


@triton.jit
def triton_send_signal(
    addr: tl.tensor,
    update: tl.constexpr,
    sem: tl.constexpr,
    scope: tl.constexpr,
    op: tl.constexpr,
    skip_sync: tl.constexpr,
) -> tl.tensor:
    """
    Signal global memory barrier(s).

    This function atomically sets global memory barriers to a update value,
    signaling to other CTAs waiting on the barrier(s).

    Args:
        addr: Memory address of the barrier(s) to wait on
        update: Set the barrier to
        sem: Memory semantics for the atomic operation. Options: "release", "relaxed".
        scope: Scope of the atomic operation. Options: "gpu", "sys"
        op: Atomic operation type: "atomic_xchg", "atomic_add"
        skip_sync: Skip CTA synchronization before setting the barrier. (default: False)
    Returns:
        The old value of the barrier(s) before the update.
    """
    if not skip_sync:
        tl.inline_asm_elementwise(
            "bar.sync 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
        )

    tl.static_assert(
        sem == "release" or sem == "relaxed",
        "Invalid memory semantic. options: 'release', 'relaxed'. ",
    )
    tl.static_assert(
        scope == "gpu" or scope == "sys", "Invalid scope. options: 'gpu','sys'. "
    )

    if op == "atomic_xchg":
        barrier_status = tl.atomic_xchg(addr, update, sem=sem, scope=scope)
    elif op == "atomic_add":
        barrier_status = tl.atomic_add(addr, update, sem=sem, scope=scope)
    else:
        raise NotImplementedError(
            f"Unsupported op '{op}' for send signal on gmem barrier. "
        )
    return barrier_status


@triton.jit
def triton_wait_signal(
    addr: tl.tensor,
    expect: tl.constexpr,
    update: tl.constexpr,
    sem: tl.constexpr,
    scope: tl.constexpr,
    op: tl.constexpr,
    skip_sync: tl.constexpr,
    sync_before: tl.constexpr = False,  # pyright: ignore[reportArgumentType]
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
        addr.type.is_ptr(),  # pyright: ignore[reportAttributeAccessIssue]
        "Barrier address must be a scalar. Do you want to use '_triton_wait_multiple_signal'? ",
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


@triton.jit
def triton_wait_multiple_signal(
    addr: tl.tensor,
    expect: tl.constexpr,
    update: tl.constexpr,
    sem: tl.constexpr,
    scope: tl.constexpr,
    op: tl.constexpr,
    skip_sync: tl.constexpr,
    sync_before: tl.constexpr = False,  # pyright: ignore[reportArgumentType]
) -> None:
    """
    Simultaneously wait for multiple global memory barriers to reach the
    expected value.

    Each thread in a CTA spin-waits and continuously checks its assigned memory
    location until it reaches the expected value, providing synchronization
    across CTAs.

    Args:
        addr: Memory addresses of the barriers to wait on (Maximum 32 barriers)
        expect: Expected value to wait for
        update: Update the barrier with once acquired
        sem: Memory semantics for the atomic operation. Options: "acquire", "relaxed".
        scope: Scope of the atomic operation. Options: "gpu", "sys"
        op: Atomic operation type: "ld", "atomic_cas"
        skip_sync: Skip CTA synchronization after acquiring the barrier
            (default False).
        sync_before: Add a CTA sync before the wait (default False)
    """
    tl.static_assert(
        (sem == "acquire" or sem == "relaxed") or sem == "release",
        "Invalid memory semantic. options: 'acquire', 'relaxed' 'release'. ",
    )
    tl.static_assert(
        scope == "gpu" or scope == "sys", "Invalid scope. options: 'gpu', 'sys'. "
    )
    tl.static_assert(
        op == "ld" or op == "atomic_cas",
        "Invalid op. options: 'ld', 'atomic_cas'. ",
    )

    tl.static_assert(
        addr.dtype == tl.pointer_type(tl.int32),
        "Invalid barrier value type. Only supports int32 for multi barrier signal. ",
    )

    if sync_before:
        tl.inline_asm_elementwise(
            "bar.sync 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
        )

    addr = tl.ravel(addr)

    tl.static_assert(len(addr.shape) == 1, "addr must be a 1D tensor. ")
    tl.static_assert(addr.shape[0] <= 32, "Wait on at most 32 barriers at a time. ")

    # Assume Triton always sets tid.y == tid.z == 0.
    if op == "ld":
        tl.inline_asm_elementwise(
            f"""
            {{
                .reg .u32   %tmp32_<3>;
                .reg .pred  %p<2>;

                mov.u32 %tmp32_0, %tid.x;
                setp.lt.s32 %p1, %tmp32_0, $2;

                mov.u32 $0, 0;
                // initialize tmp_0 to 0
                wait_block:
                    @%p1 ld.global.{sem}.{scope}.u32 $0, [$1];
                    setp.ne.u32 %p0, $0, $3;
                    and.pred %p0, %p0, %p1;
                    @%p0 bra wait_block;
            }}
            """,
            "=r, l, r, r",
            [addr, addr.shape[0], expect],
            dtype=addr.dtype.element_ty,
            is_pure=False,
            pack=1,
        )
    elif op == "atomic_cas":
        tl.inline_asm_elementwise(
            f"""
            {{
                .reg .u32   %tmp32_<3>;
                .reg .pred  %p<2>;

                mov.u32 %tmp32_0, %tid.x;
                setp.lt.s32 %p1, %tmp32_0, $2;

                mov.u32 $0, 0;
                // initialize tmp_0 to 0
                wait_block:
                    @%p1 atom.global.{sem}.{scope}.cas.b32 $0, [$1], $3, $4;
                    setp.ne.u32 %p0, $0, $3;
                    and.pred %p0, %p0, %p1;
                    @%p0 bra wait_block;
            }}
            """,
            "=r, l, r, r, r",
            [addr, addr.shape[0], expect, update],
            dtype=addr.dtype.element_ty,
            is_pure=False,
            pack=1,
        )
    else:
        raise NotImplementedError(
            f"Unsupported op '{op}' for wait signal on gmem barrier. "
        )

    if not skip_sync:
        tl.inline_asm_elementwise(
            "bar.sync 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
        )
