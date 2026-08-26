# ruff: noqa: ANN001, ANN202
"""Measure a ticketed WorkerSchedule prefix with a raw CLC tail."""

from __future__ import annotations

import argparse

from cuda.bindings import driver as cuda_driver
import torch
import triton
import triton.language as tl

from probes.common import benchmark_graphs_cold_l2
from probes.common import capture
from probes.qwen3 import helion_qwen3_ffn_tile_dependency as helion_probe
from probes.qwen3.helion_qwen3_layer_baseline import FFN_CONFIGS
from probes.qwen3.helion_qwen3_layer_baseline import FP8_MAX
from probes.qwen3.helion_qwen3_layer_baseline import FP8_MIN
from probes.qwen3.helion_qwen3_layer_baseline import FP8_MIN_SCALE
from probes.qwen3.helion_qwen3_layer_baseline import block_fp8_mm
from probes.qwen3.helion_qwen3_layer_baseline import compile_config
from probes.qwen3.helion_qwen3_layer_baseline import silu_and_mul_per_block_quant


WORKERS = 1024
GATE_TASKS = 1536
ACTIVATION_TASKS = 96
DOWN_TASKS = 512
DOWN_BASE = GATE_TASKS + ACTIVATION_TASKS
COMMANDS = WORKERS + DOWN_TASKS
FULL_STRAND_COMMANDS = WORKERS
CLC_SCRATCH_BYTES = 12288
TL_WORKERS = tl.constexpr(WORKERS)
TL_GATE_TASKS = tl.constexpr(GATE_TASKS)
TL_DOWN_BASE = tl.constexpr(DOWN_BASE)
TL_COMMANDS = tl.constexpr(COMMANDS)
TL_FULL_STRAND_COMMANDS = tl.constexpr(FULL_STRAND_COMMANDS)
TL_FP8_MAX = tl.constexpr(FP8_MAX)
TL_FP8_MIN = tl.constexpr(FP8_MIN)
TL_FP8_MIN_SCALE = tl.constexpr(FP8_MIN_SCALE)

_root0 = None
_root2 = None


@triton.jit
def _ticket(cursor):
    return tl.inline_asm_elementwise(
        asm=r"""
        {
            .reg .pred leader;
            .reg .b32 lane, old_lo, old_hi;
            .reg .b64 old;
            mov.u32 lane, %laneid;
            setp.eq.u32 leader, lane, 0;
            mov.u64 old, 0;
            @leader atom.global.gpu.relaxed.add.u64 old, [$1], 1;
            mov.b64 {old_lo, old_hi}, old;
            shfl.sync.idx.b32 old_lo, old_lo, 0, 0x1f, 0xffffffff;
            shfl.sync.idx.b32 old_hi, old_hi, 0, 0x1f, 0xffffffff;
            mov.b64 $0, {old_lo, old_hi};
        }
        """,
        constraints="=l,l",
        args=[cursor],
        dtype=tl.uint64,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _load_cursor(cursor):
    return tl.inline_asm_elementwise(
        asm=r"""
        {
            .reg .pred leader;
            .reg .b32 lane, value_lo, value_hi;
            .reg .b64 value;
            mov.u32 lane, %laneid;
            setp.eq.u32 leader, lane, 0;
            mov.u64 value, 0;
            @leader ld.acquire.gpu.global.u64 value, [$1];
            mov.b64 {value_lo, value_hi}, value;
            shfl.sync.idx.b32 value_lo, value_lo, 0, 0x1f, 0xffffffff;
            shfl.sync.idx.b32 value_hi, value_hi, 0, 0x1f, 0xffffffff;
            mov.b64 $0, {value_lo, value_hi};
        }
        """,
        constraints="=l,l",
        args=[cursor],
        dtype=tl.uint64,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _init_clc():
    return tl.inline_asm_elementwise(
        asm=r"""
        {
            .reg .pred leader;
            .reg .b32 response_addr, mbar_addr, thread_id;
            .shared .align 128 .b8 ticketed_strand_clc_scratch[12288];
            mov.u32 response_addr, ticketed_strand_clc_scratch;
            add.u32 mbar_addr, response_addr, 16;
            mov.u32 thread_id, %tid.x;
            setp.eq.u32 leader, thread_id, 0;
            @leader mbarrier.init.shared::cta.b64 [mbar_addr], 1;
            @leader fence.mbarrier_init.release.cluster;
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
def _issue_cancel(response_addr):
    return tl.inline_asm_elementwise(
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
def _wait_cancel(response_addr, phase):
    return tl.inline_asm_elementwise(
        asm=r"""
        {
            .reg .pred complete, canceled;
            .reg .b32 response_addr, mbar_addr, success, phase;
            .reg .b128 response;
            mov.u32 response_addr, $1;
            mov.u32 phase, $2;
            add.u32 mbar_addr, response_addr, 16;
            mov.u32 success, 0;
        WAIT_FOR_STRAND_CANCEL:
            mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 complete, [mbar_addr], phase;
            @!complete bra WAIT_FOR_STRAND_CANCEL;
            ld.shared.b128 response, [response_addr];
            clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 canceled, response;
            selp.u32 success, 1, 0, canceled;
            mov.u32 $0, success;
        }
        """,
        constraints="=r,r,r",
        args=[response_addr, phase],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _dispatch(
    ffn_q,
    w13_q,
    ffn_scale,
    w13_scale,
    gate_up,
    activation_scale,
    activation_q,
    w2_q,
    w2_scale,
    output,
    state,
    command,
    epoch,
):
    if command < TL_WORKERS:
        worker = command.to(tl.int32)
        for virtual_pid in tl.range(worker, TL_GATE_TASKS, TL_WORKERS):
            _root0(
                ffn_q,
                w13_q,
                ffn_scale,
                w13_scale,
                gate_up,
                TL_FP8_MAX,
                TL_FP8_MIN_SCALE,
                activation_scale,
                TL_FP8_MIN,
                activation_q,
                state,
                virtual_pid,
                epoch,
            )
    else:
        virtual_pid = TL_DOWN_BASE + command - TL_WORKERS
        _root2(
            activation_scale,
            activation_q,
            w2_q,
            w2_scale,
            output,
            state,
            virtual_pid,
            epoch,
        )


@triton.jit
def ticketed_strand_kernel(
    ffn_q,
    w13_q,
    ffn_scale,
    w13_scale,
    gate_up,
    activation_scale,
    activation_q,
    w2_q,
    w2_scale,
    output,
    state,
    cursor,
):
    response = _init_clc()
    ticket = _ticket(cursor)
    generation = ticket // TL_COMMANDS
    launch_base = generation * TL_COMMANDS
    command = (ticket - launch_base).to(tl.int32)
    epoch = (generation + 1).to(tl.uint32)

    admitted = _load_cursor(cursor)
    while admitted < launch_base + TL_WORKERS:
        tl.inline_asm_elementwise(
            asm="nanosleep.u32 16; mov.u32 $0, $1;",
            constraints="=r,r",
            args=[tl.arange(0, 32)],
            dtype=tl.uint32,
            is_pure=False,
            pack=1,
        )
        admitted = _load_cursor(cursor)

    phase = tl.full([], 0, tl.uint32)
    running = tl.full([], 1, tl.uint32)
    while running != 0:
        _dispatch(
            ffn_q,
            w13_q,
            ffn_scale,
            w13_scale,
            gate_up,
            activation_scale,
            activation_q,
            w2_q,
            w2_scale,
            output,
            state,
            command,
            epoch,
        )
        response = _issue_cancel(response)
        success = _wait_cancel(response, phase)
        if success != 0:
            phase = 1 - phase
            ticket = _ticket(cursor)
            command = (ticket - launch_base).to(tl.int32)
        else:
            running = tl.full([], 0, tl.uint32)


@triton.jit
def ticketed_full_strand_kernel(
    ffn_q,
    w13_q,
    ffn_scale,
    w13_scale,
    gate_up,
    activation_scale,
    activation_q,
    w2_q,
    w2_scale,
    output,
    state,
    cursor,
):
    # Keep the same static shared-memory occupancy envelope as the CLC plan,
    # but isolate the cost of ticketed worker ownership from tail dispatch.
    _init_clc()
    ticket = _ticket(cursor)
    generation = ticket // TL_FULL_STRAND_COMMANDS
    launch_base = generation * TL_FULL_STRAND_COMMANDS
    worker = (ticket - launch_base).to(tl.int32)
    epoch = (generation + 1).to(tl.uint32)

    admitted = _load_cursor(cursor)
    while admitted < launch_base + TL_WORKERS:
        tl.inline_asm_elementwise(
            asm="nanosleep.u32 16; mov.u32 $0, $1;",
            constraints="=r,r",
            args=[tl.arange(0, 32)],
            dtype=tl.uint32,
            is_pure=False,
            pack=1,
        )
        admitted = _load_cursor(cursor)

    for virtual_pid in tl.range(worker, TL_GATE_TASKS, TL_WORKERS):
        _root0(
            ffn_q,
            w13_q,
            ffn_scale,
            w13_scale,
            gate_up,
            TL_FP8_MAX,
            TL_FP8_MIN_SCALE,
            activation_scale,
            TL_FP8_MIN,
            activation_q,
            state,
            virtual_pid,
            epoch,
        )
    if worker >= 512:
        _root2(
            activation_scale,
            activation_q,
            w2_q,
            w2_scale,
            output,
            state,
            TL_DOWN_BASE + worker - 512,
            epoch,
        )


def _kernel_resources(kernel) -> dict[str, int]:
    error, blocks_per_sm = cuda_driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        cuda_driver.CUfunction(int(kernel.function)),
        32,
        int(kernel.metadata.shared),
    )
    if error != cuda_driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(error)
    return {
        "registers": kernel.n_regs,
        "spills": kernel.n_spills,
        "shared": kernel.metadata.shared,
        "blocks_per_sm": int(blocks_per_sm),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=40)
    args = parser.parse_args()

    torch.manual_seed(0)
    ffn_q = torch.randn((1, 4096), device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    ffn_scale = torch.rand((1, 32), device="cuda")
    w13_q = torch.randn((24576, 4096), device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    w13_scale = torch.rand((192, 32), device="cuda")
    w2_q = torch.randn((4096, 12288), device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    w2_scale = torch.rand((32, 96), device="cuda")
    kernel_args = (ffn_q, ffn_scale, w13_q, w13_scale, w2_q, w2_scale, 128)

    bound = helion_probe.qwen3_ffn_tile_dependency.bind(kernel_args)
    config_args = argparse.Namespace(
        batch=1,
        hidden=4096,
        intermediate=12288,
        group=128,
        w13_stages=4,
        w13_unroll=2,
        w2_stages=4,
        w2_unroll=4,
        kernel_stages=2,
        maxnreg=None,
        worker_multiplier=8,
        cross_loop_workers=WORKERS,
    )
    static_wrapper = bound.compile_config(
        helion_probe._persistent_config(bound, config_args)
    )
    static_output, gate_up, activation_q, activation_scale = static_wrapper(
        *kernel_args
    )
    torch.cuda.synchronize()
    global _root0, _root2
    _root0 = static_wrapper.__globals__["tile_dependency_root_0_scheduled_task"]
    _root2 = static_wrapper.__globals__["tile_dependency_root_2_scheduled_task"]

    output = torch.empty_like(static_output)
    state = torch.zeros(WORKERS + 3136, device="cuda", dtype=torch.uint32)
    cursor = torch.zeros(1, device="cuda", dtype=torch.uint64)
    full_output = torch.empty_like(static_output)
    full_state = torch.zeros(WORKERS + 3136, device="cuda", dtype=torch.uint32)
    full_cursor = torch.zeros(1, device="cuda", dtype=torch.uint64)

    def launch_ticketed():
        return ticketed_strand_kernel[(COMMANDS,)](
            ffn_q,
            w13_q,
            ffn_scale,
            w13_scale,
            gate_up,
            activation_scale,
            activation_q,
            w2_q,
            w2_scale,
            output,
            state,
            cursor,
            num_warps=1,
            num_stages=2,
            num_ctas=1,
            launch_pdl=True,
        )

    compiled = launch_ticketed()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, static_output, atol=0.25, rtol=5e-2)

    def launch_full_strands():
        return ticketed_full_strand_kernel[(FULL_STRAND_COMMANDS,)](
            ffn_q,
            w13_q,
            ffn_scale,
            w13_scale,
            gate_up,
            activation_scale,
            activation_q,
            w2_q,
            w2_scale,
            full_output,
            full_state,
            full_cursor,
            num_warps=1,
            num_stages=2,
            num_ctas=1,
            launch_pdl=True,
        )

    full_compiled = launch_full_strands()
    torch.cuda.synchronize()
    torch.testing.assert_close(full_output, static_output, atol=0.25, rtol=5e-2)

    _, w13 = compile_config(
        block_fp8_mm,
        (ffn_q, ffn_scale, w13_q, w13_scale, 128),
        FFN_CONFIGS["w13"],
    )
    separate_gate = w13(ffn_q, ffn_scale, w13_q, w13_scale, 128)
    _, activation = compile_config(
        silu_and_mul_per_block_quant,
        (separate_gate, 128),
        FFN_CONFIGS["silu_quant"],
    )
    separate_q, separate_scale = activation(separate_gate, 128)
    _, w2 = compile_config(
        block_fp8_mm,
        (separate_q, separate_scale, w2_q, w2_scale, 128),
        FFN_CONFIGS["w2"],
    )

    def launch_separate():
        gate = w13(ffn_q, ffn_scale, w13_q, w13_scale, 128)
        quant, scale = activation(gate, 128)
        return w2(quant, scale, w2_q, w2_scale, 128)

    ticketed_graph, _ = capture(launch_ticketed)
    full_graph, _ = capture(launch_full_strands)
    separate_graph, _ = capture(launch_separate)
    timings = benchmark_graphs_cold_l2(
        {
            "ticketed_strands": (ticketed_graph.replay, lambda: None),
            "ticketed_full_strands": (full_graph.replay, lambda: None),
            "helion_separate": (separate_graph.replay, lambda: None),
        },
        args.repeats,
        flush_mib=256,
        order_seed=37,
    )
    torch.cuda.synchronize()
    launches, remainder = divmod(int(cursor.item()), COMMANDS)
    assert remainder == 0
    expected = launches & 0xFFFFFFFF
    assert torch.all(state[WORKERS : WORKERS + 96 * 32 : 32] == expected * 16)
    assert int(state[WORKERS + 3072].item()) == expected * 64
    assert int(state[WORKERS + 3104].item()) == expected * 32
    print(
        {
            "timings": timings,
            "resources": _kernel_resources(compiled),
            "full_resources": _kernel_resources(full_compiled),
            "launches": launches,
        }
    )


if __name__ == "__main__":
    main()
