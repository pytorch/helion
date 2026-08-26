"""Measure the logical-task ownership produced by bounded Blackwell CLC claims."""

from __future__ import annotations

from collections import Counter

import torch
import triton
import triton.language as tl

from probes.common import require_idle_visible_gpu


GRID = 4992
INBOX_DEPTH = 5
SCRATCH_BYTES = 26_624
TASK_RANGES = (
    ("pre_partial", 0, 32),
    ("pre_quant", 32, 64),
    ("qkv", 64, 832),
    ("attention", 832, 1856),
    ("attention_merge", 1856, 2368),
    ("attention_finalize", 2368, 2400),
    ("o_projection", 2400, 2912),
    ("post_quant", 2912, 2944),
    ("gate_up", 2944, 4480),
    ("down", 4480, 4992),
)


@triton.jit
def _initialize_cancel():
    return tl.inline_asm_elementwise(
        asm=r"""
        {
            .reg .pred leader;
            .reg .b32 response_addr, mbar_addr, tid;
            .shared .align 128 .b8 clc_distribution_scratch[26624];
            mov.u32 response_addr, clc_distribution_scratch;
            add.u32 mbar_addr, response_addr, 16;
            mov.u32 tid, %tid.x;
            setp.eq.u32 leader, tid, 0;
            @leader mbarrier.init.shared::cta.b64 [mbar_addr], 1;
            fence.mbarrier_init.release.cluster;
            bar.warp.sync 0xffffffff;
            @leader clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [response_addr], [mbar_addr];
            @leader mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 _, [mbar_addr], 16;
            bar.warp.sync 0xffffffff;
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
def _wait_cancel(response_addr, phase):
    return tl.inline_asm_elementwise(
        asm=r"""
        {
            .reg .pred complete, canceled;
            .reg .b32 response_addr, mbar_addr, phase, success, canceled_x;
            .reg .b128 response;
            mov.u32 response_addr, $2;
            mov.u32 phase, $3;
            add.u32 mbar_addr, response_addr, 16;
        WAIT:
            mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 complete, [mbar_addr], phase;
            @!complete bra WAIT;
            bar.warp.sync 0xffffffff;
            ld.shared.b128 response, [response_addr];
            clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 canceled, response;
            selp.u32 success, 1, 0, canceled;
            mov.u32 canceled_x, 0xffffffff;
            @canceled clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 canceled_x, response;
            mov.u32 $0, success;
            mov.u32 $1, canceled_x;
        }
        """,
        constraints="=r,=r,r,r",
        args=[response_addr, phase],
        dtype=(tl.uint32, tl.int32),
        is_pure=False,
        pack=1,
    )


@triton.jit
def _issue_cancel(response_addr):
    return tl.inline_asm_elementwise(
        asm=r"""
        {
            .reg .pred leader;
            .reg .b32 response_addr, mbar_addr, tid;
            mov.u32 response_addr, $1;
            add.u32 mbar_addr, response_addr, 16;
            mov.u32 tid, %tid.x;
            setp.eq.u32 leader, tid, 0;
            @leader fence.proxy.async.shared::cta;
            @leader clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [response_addr], [mbar_addr];
            @leader mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 _, [mbar_addr], 16;
            bar.warp.sync 0xffffffff;
            mov.u32 $0, response_addr;
        }
        """,
        constraints="=r,r",
        args=[response_addr],
        dtype=tl.uint32,
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
def inbox_distribution_kernel(owner):
    pid = tl.program_id(0).to(tl.int32)
    tl.store(owner + pid, pid)
    response = _initialize_cancel()

    success_1, canceled_1 = _wait_cancel(response, 0)
    if success_1 != 0:
        tl.store(owner + canceled_1, pid)
        response = _issue_cancel(response)
        success_2, canceled_2 = _wait_cancel(response, 1)
        if success_2 != 0:
            tl.store(owner + canceled_2, pid)
            response = _issue_cancel(response)
            success_3, canceled_3 = _wait_cancel(response, 0)
            if success_3 != 0:
                tl.store(owner + canceled_3, pid)
                response = _issue_cancel(response)
                success_4, canceled_4 = _wait_cancel(response, 1)
                if success_4 != 0:
                    tl.store(owner + canceled_4, pid)

    begin = _globaltimer()
    now = begin
    while now - begin < 200_000:
        now = _globaltimer()


def _task_family(task: int) -> str:
    return next(name for name, begin, end in TASK_RANGES if begin <= task < end)


def main() -> None:
    require_idle_visible_gpu()
    owner = torch.full((GRID,), -1, device="cuda", dtype=torch.int32)
    inbox_distribution_kernel[(GRID,)](
        owner,
        num_warps=1,
        num_stages=1,
        num_ctas=1,
        launch_pdl=True,
    )
    torch.cuda.synchronize()
    owner_cpu = owner.cpu()
    if bool(torch.any(owner_cpu < 0)):
        missing = torch.nonzero(owner_cpu < 0).flatten().tolist()
        raise RuntimeError(f"unowned logical tasks: {missing[:32]}")

    tasks_by_owner: dict[int, list[int]] = {}
    for task, owner_id in enumerate(owner_cpu.tolist()):
        tasks_by_owner.setdefault(owner_id, []).append(task)
    family_counts_by_owner = {
        owner_id: Counter(_task_family(task) for task in tasks)
        for owner_id, tasks in tasks_by_owner.items()
    }
    gate_owners = {
        owner_id
        for owner_id, counts in family_counts_by_owner.items()
        if counts["gate_up"]
    }
    down_owners = {
        owner_id
        for owner_id, counts in family_counts_by_owner.items()
        if counts["down"]
    }
    mixed_gate_down = gate_owners & down_owners
    owner_sizes = Counter(len(tasks) for tasks in tasks_by_owner.values())
    down_gate_counts = Counter(
        family_counts_by_owner[owner_id]["gate_up"] for owner_id in down_owners
    )
    print(
        "CLC_INBOX_DISTRIBUTION",
        {
            "grid": GRID,
            "inbox_depth": INBOX_DEPTH,
            "scratch_bytes": SCRATCH_BYTES,
            "physically_started_ctas": len(tasks_by_owner),
            "canceled_tasks": GRID - len(tasks_by_owner),
            "owner_size_histogram": dict(sorted(owner_sizes.items())),
            "gate_owners": len(gate_owners),
            "down_owners": len(down_owners),
            "gate_and_down_owners": len(mixed_gate_down),
            "down_only_owners": len(down_owners - gate_owners),
            "down_owner_gate_task_histogram": dict(sorted(down_gate_counts.items())),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
