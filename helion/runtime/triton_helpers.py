from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["triton_wait_signal"]


if torch.version.hip is not None:
    TRITON_WAIT_SIGNAL_IMPL = "rocm_no_inline_asm"

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
        ROCm-safe variant of wait signal without CUDA inline asm barriers.
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
            tl.debug_barrier()

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
            tl.debug_barrier()
else:
    TRITON_WAIT_SIGNAL_IMPL = "cuda_inline_asm"

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
