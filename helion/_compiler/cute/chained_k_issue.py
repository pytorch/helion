"""The existing ordered K64 issue program, shared by root and stage executors.

Callers own the accepted KSchedule, original producer and storage lifetime
proofs. This component only renders their existing issue/completion sequence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import chained_matmul as chain

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .chained_body_program import RootActionBody


def emit_k_half_issues(
    prefix: str,
    stage: int,
    mode: str,
    producer: Sequence[str],
    *,
    retire_each_half: bool = False,
    continuation: RootActionBody | None = None,
) -> list[str]:
    # Descriptors cover the full K128 arena. Half1's physical writes cannot
    # overlap half0's MMA read set; its CTA publication is not MMA completion.
    half_lines = [
        *producer,
        "cute.arch.cp_async_commit_group()",
        "cute.arch.cp_async_wait_group(0)",
        "cute.arch.fence_view_async_shared()",
        "cute.arch.sync_threads()",
        "if chain_warp == 0:",
        f"    {prefix}_mma.set(tcgen05.Field.ACCUMULATE, True)",
        f"    for {prefix}_local_kk in cutlass.range_constexpr(4):",
        f"        {prefix}_kk = chain_k_half * 4 + {prefix}_local_kk",
        f"        cute.gemm({prefix}_mma, {prefix}_acc, {prefix}_ra[None, None, {prefix}_kk], {prefix}_rb[None, None, {prefix}_kk], {prefix}_acc)",
        f"        {prefix}_mma.set(tcgen05.Field.ACCUMULATE, True)",
    ]
    if mode == "serial64" or retire_each_half:
        half_lines.extend(
            [
                "    with cute.arch.elect_one():",
                f"        tcgen05.commit(chain_bars + {stage})",
                f"cute.arch.mbarrier_wait(chain_bars + {stage}, chain_k_half)",
                "cute.arch.sync_threads()",
            ]
        )
    if continuation is not None:
        from .prepared_continuation import emit_root_continuation

        action = continuation.pending
        if (
            action is None
            or action.stage != stage
            or action.k_schedule is None
            or action.k_schedule.mode != mode
            or action.retire_each_half != retire_each_half
            or tuple(producer) != tuple(action.half_lines())
        ):
            raise chain._UnsupportedChain("prepared half issue changed")
        half_lines = [
            *producer,
            "cute.arch.cp_async_commit_group()",
            "cute.arch.cp_async_wait_group(0)",
            "cute.arch.fence_view_async_shared()",
            "cute.arch.sync_threads()",
            *emit_root_continuation(continuation, action, prefix),
        ]
        if mode == "serial64" or retire_each_half:
            half_lines.append("cute.arch.sync_threads()")
    lines = [
        f"{prefix}_rb = {prefix}_mma.make_fragment_B({prefix}_slice.partition_B({prefix}_b))",
        "for chain_k_half in cutlass.range_constexpr(2):",
        chain._indent(half_lines),
    ]
    if mode == "overlap64" and not retire_each_half:
        if continuation is not None:
            from .prepared_continuation import emit_root_continuation

            assert continuation.pending is not None
            lines.extend(
                emit_root_continuation(
                    continuation, continuation.pending, prefix, final_commit=True
                )
            )
        else:
            lines.extend(
                [
                    "if chain_warp == 0:",
                    "    with cute.arch.elect_one():",
                    f"        tcgen05.commit(chain_bars + {stage})",
                    f"cute.arch.mbarrier_wait(chain_bars + {stage}, 0)",
                ]
            )
    return lines
