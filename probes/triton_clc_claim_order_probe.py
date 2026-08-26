"""Record which pending CTA IDs Blackwell CLC returns to the first cohort."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from probes.common import require_idle_visible_gpu


GRID = 2048
SCRATCH_BYTES = 51 * 1024


@triton.jit
def _issue_cancel():
    return tl.inline_asm_elementwise(
        asm=r"""
        {
            .reg .pred leader;
            .reg .b32 response_addr, mbar_addr, tid;
            .shared .align 128 .b8 clc_order_scratch[52224];
            mov.u32 response_addr, clc_order_scratch;
            add.u32 mbar_addr, response_addr, 16;
            mov.u32 tid, %tid.x;
            setp.eq.u32 leader, tid, 0;
            @leader mbarrier.init.shared::cta.b64 [mbar_addr], 1;
            fence.mbarrier_init.release.cluster;
            @leader clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [response_addr], [mbar_addr];
            @leader mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 _, [mbar_addr], 16;
            mov.u32 $0, response_addr;
        }
        """,
        constraints="=r",
        args=[],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _wait_cancel(response_addr):
    return tl.inline_asm_elementwise(
        asm=r"""
        {
            .reg .pred complete, canceled;
            .reg .b32 response_addr, mbar_addr, success, canceled_x;
            .reg .b128 response;
            mov.u32 response_addr, $2;
            add.u32 mbar_addr, response_addr, 16;
        WAIT:
            mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 complete, [mbar_addr], 0;
            @!complete bra WAIT;
            ld.shared.b128 response, [response_addr];
            clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 canceled, response;
            selp.u32 success, 1, 0, canceled;
            @canceled clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 canceled_x, response;
            mov.u32 $0, success;
            mov.u32 $1, canceled_x;
        }
        """,
        constraints="=r,=r,r",
        args=[response_addr],
        dtype=(tl.uint32, tl.int32),
        is_pure=False,
        pack=1,
    )


@triton.jit
def _globaltimer():
    return tl.inline_asm_elementwise(
        asm="mov.u64 $0, %globaltimer;",
        constraints="=l",
        args=[],
        dtype=tl.int64,
        is_pure=False,
        pack=1,
    )


@triton.jit
def claim_order_kernel(
    started,
    claimed,
    issue_sequence,
    completion_sequence,
    issue_counter,
    completion_counter,
):
    pid = tl.program_id(0).to(tl.int32)
    issue_seq = tl.atomic_add(issue_counter, 1)
    response = _issue_cancel()
    success, canceled = _wait_cancel(response)
    completion_seq = tl.atomic_add(completion_counter, 1)
    tl.store(started + pid, 1)
    tl.store(claimed + pid, tl.where(success != 0, canceled, -1))
    tl.store(issue_sequence + pid, issue_seq)
    tl.store(completion_sequence + pid, completion_seq)
    begin = _globaltimer()
    now = begin
    while now - begin < 10_000_000:
        now = _globaltimer()


def main() -> None:
    require_idle_visible_gpu()
    started = torch.zeros(GRID, device="cuda", dtype=torch.int32)
    claimed = torch.full((GRID,), -2, device="cuda", dtype=torch.int32)
    issue_sequence = torch.full((GRID,), -1, device="cuda", dtype=torch.int32)
    completion_sequence = torch.full(
        (GRID,), -1, device="cuda", dtype=torch.int32
    )
    issue_counter = torch.zeros((), device="cuda", dtype=torch.int32)
    completion_counter = torch.zeros((), device="cuda", dtype=torch.int32)
    claim_order_kernel[(GRID,)](
        started,
        claimed,
        issue_sequence,
        completion_sequence,
        issue_counter,
        completion_counter,
        num_warps=1,
        num_stages=1,
        num_ctas=1,
        launch_pdl=True,
    )
    torch.cuda.synchronize()
    started_ids = torch.nonzero(started, as_tuple=False).flatten().cpu()
    successful_gpu = torch.nonzero(claimed >= 0, as_tuple=False).flatten()
    successful = successful_gpu.cpu()
    records = torch.stack(
        (
            successful,
            claimed[successful_gpu].cpu(),
            issue_sequence[successful_gpu].cpu(),
            completion_sequence[successful_gpu].cpu(),
        ),
        dim=1,
    )
    requester_order = records[records[:, 0].argsort()]
    issue_order = records[records[:, 2].argsort()]
    completion_order = records[records[:, 3].argsort()]
    canceled = requester_order[:, 1]
    canceled_by_issue = issue_order[:, 1]
    canceled_by_completion = completion_order[:, 1]
    print(
        "CLC_CLAIM_ORDER",
        {
            "grid": GRID,
            "scratch_bytes": SCRATCH_BYTES,
            "started": int(started_ids.numel()),
            "started_min": int(started_ids.min()),
            "started_max": int(started_ids.max()),
            "successful": int(successful.numel()),
            "canceled_min": int(canceled.min()),
            "canceled_max": int(canceled.max()),
            "canceled_ascending_by_requester": bool(
                torch.all(canceled[1:] >= canceled[:-1])
            ),
            "canceled_ascending_by_issue": bool(
                torch.all(canceled_by_issue[1:] >= canceled_by_issue[:-1])
            ),
            "canceled_descending_by_issue": bool(
                torch.all(canceled_by_issue[1:] <= canceled_by_issue[:-1])
            ),
            "canceled_ascending_by_completion": bool(
                torch.all(
                    canceled_by_completion[1:] >= canceled_by_completion[:-1]
                )
            ),
            "canceled_descending_by_completion": bool(
                torch.all(
                    canceled_by_completion[1:] <= canceled_by_completion[:-1]
                )
            ),
            "first_by_requester": requester_order[:16].tolist(),
            "first_by_issue": issue_order[:16].tolist(),
            "first_by_completion": completion_order[:16].tolist(),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
