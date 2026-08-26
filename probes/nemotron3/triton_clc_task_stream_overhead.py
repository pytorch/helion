# ruff: noqa: ANN001, ANN201
"""Measure the empty-task overhead of Helion's CLC ticket executor."""

from __future__ import annotations

import argparse
import json

import torch
import triton
import triton.language as tl

from probes.common import benchmark_interleaved
from probes.common import capture
from probes.common import require_idle_visible_gpu


@triton.jit
def clc_task_stream_overhead(
    ticket_state,
    TOTAL_TASKS: tl.constexpr,
    RESIDENT_TASKS: tl.constexpr,
):
    response_addr = tl.inline_asm_elementwise(
        asm=r"""
        {
            .reg .pred leader;
            .reg .b32 response_addr, mbar_addr, thread_id;
            .shared .align 16 .b8 clc_task_stream_scratch[32768];

            mov.u32 response_addr, clc_task_stream_scratch;
            add.u32 mbar_addr, response_addr, 16;
            mov.u32 thread_id, %tid.x;
            setp.eq.u32 leader, thread_id, 0;
            @leader mbarrier.init.shared::cta.b64 [mbar_addr], 1;
            bar.sync 0;
            mov.u32 $0, response_addr;
        }
        """,
        constraints="=r",
        args=[],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )
    phase = tl.full([], 0, tl.uint32)
    active = tl.full([], True, tl.int1)
    while active:
        ticket = tl.atomic_add(ticket_state, 1, sem="relaxed", scope="gpu")
        command = ticket % TOTAL_TASKS
        if command < TOTAL_TASKS - RESIDENT_TASKS:
            response_addr = tl.inline_asm_elementwise(
                asm=r"""
                {
                    .reg .pred leader;
                    .reg .b32 response_addr, mbar_addr, thread_id;

                    mov.u32 response_addr, $1;
                    add.u32 mbar_addr, response_addr, 16;
                    mov.u32 thread_id, %tid.x;
                    setp.eq.u32 leader, thread_id, 0;
                    @leader fence.proxy.async.shared::cta;
                    @leader clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [response_addr], [mbar_addr];
                    @leader mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 _, [mbar_addr], 16;
                    bar.sync 0;
                    mov.u32 $0, response_addr;
                }
                """,
                constraints="=r,r",
                args=[response_addr],
                dtype=tl.uint32,
                is_pure=False,
                pack=1,
            )
            success = tl.inline_asm_elementwise(
                asm=r"""
                {
                    .reg .pred complete, canceled, leader;
                    .reg .b32 response_addr, mbar_addr, success, phase, thread_id;
                    .reg .b128 response_value;

                    mov.u32 response_addr, $1;
                    add.u32 mbar_addr, response_addr, 16;
                    mov.u32 phase, $2;
                    mov.u32 thread_id, %tid.x;
                    setp.eq.u32 leader, thread_id, 0;
                    mov.u32 success, 0;
                    @!leader bra CLC_TASK_STREAM_WAIT_DONE;
                CLC_TASK_STREAM_WAIT:
                    mbarrier.try_wait.parity.relaxed.cta.shared.b64 complete, [mbar_addr], phase;
                    @!complete bra CLC_TASK_STREAM_WAIT;
                CLC_TASK_STREAM_WAIT_DONE:
                    bar.sync 0;
                    ld.shared.b128 response_value, [response_addr];
                    clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 canceled, response_value;
                    selp.u32 success, 1, 0, canceled;
                    bar.sync 0;
                    mov.u32 $0, success;
                }
                """,
                constraints="=r,r,r",
                args=[response_addr, phase],
                dtype=tl.uint32,
                is_pure=False,
                pack=1,
            )
            phase = 1 - phase
            if success == 0:
                active = False
        else:
            active = False


def run(args: argparse.Namespace) -> None:
    require_idle_visible_gpu()
    ticket_state = torch.zeros(1, dtype=torch.uint64, device="cuda")

    def launch() -> None:
        ticket_state.zero_()
        clc_task_stream_overhead[(args.total_tasks,)](
            ticket_state,
            TOTAL_TASKS=args.total_tasks,
            RESIDENT_TASKS=args.resident_tasks,
            num_warps=args.num_warps,
            num_stages=1,
            launch_pdl=True,
        )

    launch()
    torch.cuda.synchronize()
    if int(ticket_state.item()) != args.total_tasks:
        raise AssertionError(
            f"processed {int(ticket_state.item())} tickets, expected {args.total_tasks}"
        )
    graph, _ = capture(launch)
    timings = benchmark_interleaved(
        {"empty_clc_task_stream": graph.replay},
        args.repeats,
        args.batch_replays,
    )
    print(
        "RESULT_JSON",
        json.dumps(
            {
                "total_tasks": args.total_tasks,
                "resident_tasks": args.resident_tasks,
                "num_warps": args.num_warps,
                "timings": timings,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-tasks", type=int, required=True)
    parser.add_argument("--resident-tasks", type=int, default=592)
    parser.add_argument("--num-warps", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--batch-replays", type=int, default=10)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
