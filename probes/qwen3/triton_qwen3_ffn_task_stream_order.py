"""Compare alternative dependency-safe CLC command streams for Qwen FFN.

The compute bodies, events, and CLC protocol come directly from Helion.  This
probe changes only the symbolic order in which ticket numbers select existing
root tasks. No worker is assigned a fixed role.
"""

from __future__ import annotations

import argparse
import ast
import copy
import linecache
import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from probes.common import benchmark_interleaved
from probes.common import capture
from probes.common import require_idle_visible_gpu
from probes.common import visible_gpu_pids
from probes.qwen3.helion_qwen3_ffn_tile_dependency import _helion_resources
from probes.qwen3.helion_qwen3_ffn_tile_dependency import _persistent_config
from probes.qwen3.helion_qwen3_ffn_tile_dependency import qwen3_ffn_tile_dependency
from probes.qwen3.helion_qwen3_layer_baseline import FFN_CONFIGS
from probes.qwen3.helion_qwen3_layer_baseline import FP8_MAX
from probes.qwen3.helion_qwen3_layer_baseline import FP8_MIN
from probes.qwen3.helion_qwen3_layer_baseline import FP8_MIN_SCALE
from probes.qwen3.helion_qwen3_layer_baseline import block_fp8_mm
from probes.qwen3.helion_qwen3_layer_baseline import compile_config
from probes.qwen3.helion_qwen3_layer_baseline import silu_and_mul_per_block_quant
from probes.qwen3.triton_qwen3_ffn_orchestrator import _allocate_inputs

if TYPE_CHECKING:
    from collections.abc import Callable


def _calls_named(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == name
        for child in ast.walk(node)
    )


def _reordered_source(
    source: str,
    *,
    order: str,
    batch: int,
    intermediate: int,
    w13_tasks_per_batch: int,
    w2_tasks_per_batch: int,
    fan_in: int,
    schedule_wave_size: int,
    executor_wave_size: int,
    executor_mode: str,
    minimal_single_warp_clc: bool,
    final_arrival_acq_rel: bool,
    launch_pdl: bool,
    omit_clc_reuse_fence: bool,
    prefetch_depth: int,
) -> str:
    """Rewrite task order and, independently, the CLC resident cohort."""
    source = source.replace("_source_module.FP8_MAX", repr(FP8_MAX))
    source = source.replace("_source_module.FP8_MIN_SCALE", repr(FP8_MIN_SCALE))
    source = source.replace("_source_module.FP8_MIN", repr(FP8_MIN))
    module = ast.parse(source)
    if prefetch_depth:
        prefetch_helper = ast.parse(
            """
@triton.jit
def tile_dependency_prefetch_l2(address):
    return tl.inline_asm_elementwise(
        asm="prefetch.global.L2 [$1]; mov.u32 $0, 0;",
        constraints="=r,l",
        args=[address],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )
"""
        ).body[0]
        root_2 = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "tile_dependency_root_2"
        )
        second_wait = next(
            index
            for index, statement in enumerate(root_2.body)
            if isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id.startswith("tile_dependency_scope_wait_1")
                for target in statement.targets
            )
        )
        frontier_groups = min(
            w13_tasks_per_batch // fan_in,
            schedule_wave_size // (batch * fan_in),
        )
        prefetch = ast.parse(
            f"""
for tile_dependency_prefetch_offset in tl.static_range(0, {prefetch_depth}):
    tile_dependency_prefetch_group = {frontier_groups} + tile_dependency_prefetch_offset
    tile_dependency_prefetch_address = w2_q + indices_6 * {intermediate} + tile_dependency_prefetch_group * _BLOCK_SIZE_7
    tile_dependency_prefetch_l2(tile_dependency_prefetch_address)
"""
        ).body
        root_2.body[second_wait:second_wait] = prefetch
        module.body.insert(module.body.index(root_2), prefetch_helper)
    if final_arrival_acq_rel:
        for node in ast.walk(module):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            if not (
                isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "atomic_add"
                and any(
                    isinstance(target, ast.Name)
                    and target.id.startswith("tile_dependency_continuation_previous")
                    for target in node.targets
                )
            ):
                continue
            sem = next(
                keyword for keyword in node.value.keywords if keyword.arg == "sem"
            )
            sem.value = ast.Constant("acq_rel")
        for node in ast.walk(module):
            if not isinstance(node, ast.If):
                continue
            node.body = [
                statement
                for statement in node.body
                if not (
                    isinstance(statement, ast.Assign)
                    and any(
                        isinstance(target, ast.Name)
                        and target.id.startswith("tile_dependency_continuation_acquire")
                        for target in statement.targets
                    )
                )
            ]
    master = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name.startswith("_helion_qwen3_ffn_tile_dependency_source")
    )
    if minimal_single_warp_clc:
        for call in (
            child
            for child in ast.walk(master)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "inline_asm_elementwise"
        ):
            asm_keyword = next(
                (keyword for keyword in call.keywords if keyword.arg == "asm"),
                None,
            )
            if not (
                asm_keyword is not None
                and isinstance(asm_keyword.value, ast.Constant)
                and isinstance(asm_keyword.value.value, str)
            ):
                continue
            asm = asm_keyword.value.value
            if "mbarrier.init.shared::cta" in asm or (
                "clusterlaunchcontrol.try_cancel" in asm
                and "mbarrier.arrive.expect_tx" in asm
            ):
                asm = asm.replace("bar.warp.sync 0xffffffff;", "")
                if omit_clc_reuse_fence:
                    asm = asm.replace(
                        "@leader fence.proxy.async.shared::cta;",
                        "",
                    )
                asm_keyword.value = ast.Constant(asm)
            elif "mbarrier.try_wait.parity" in asm:
                asm_keyword.value = ast.Constant(
                    """{
            .reg .pred complete, canceled;
            .reg .b32 response_addr, mbar_addr, success, phase;
            .reg .b128 response_value;

            mov.u32 response_addr, $1;
            add.u32 mbar_addr, response_addr, 16;
            mov.u32 phase, $2;
            mov.u32 success, 0;
        HELION_CLC_WAIT:
            mbarrier.try_wait.parity.relaxed.cta.shared.b64 complete, [mbar_addr], phase;
            @!complete bra HELION_CLC_WAIT;
            ld.shared.b128 response_value, [response_addr];
            clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 canceled, response_value;
            selp.u32 success, 1, 0, canceled;
            mov.u32 $0, success;
        }"""
                )
    loop = next(node for node in ast.walk(master) if isinstance(node, ast.While))
    command_assignment_index = next(
        index
        for index, statement in enumerate(loop.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id.startswith("tile_dependency_command")
    )
    command_name = loop.body[command_assignment_index].targets[0].id
    stream_name = f"{command_name}_stream"
    loop.body[command_assignment_index].targets[0].id = stream_name

    cancellation_indices = [
        index
        for index, statement in enumerate(loop.body)
        if isinstance(statement, ast.If)
        and (
            "clusterlaunchcontrol.try_cancel" in ast.unparse(statement)
            or any(
                isinstance(child, ast.Name)
                and child.id.startswith("tile_dependency_clc_success")
                for child in ast.walk(statement)
            )
        )
    ]
    if len(cancellation_indices) != 2:
        raise RuntimeError("expected one CLC issue and one CLC completion branch")
    command_count = batch * (w13_tasks_per_batch + w2_tasks_per_batch)
    if executor_mode in ("ticket", "two-wave"):
        canceling_source = f"{stream_name} < {command_count - executor_wave_size}"
    elif executor_mode == "strided":
        if order != "root":
            raise ValueError("strided execution is only defined for root order")
        canceling_source = f"{stream_name} + {executor_wave_size} < {command_count}"
    else:
        raise ValueError(f"unsupported executor mode {executor_mode!r}")
    for index in cancellation_indices:
        loop.body[index].test = ast.parse(canceling_source, mode="eval").body

    launcher = next(
        child
        for child in ast.walk(module)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == "_launcher"
    )
    minimum_residency = next(
        keyword
        for keyword in launcher.keywords
        if keyword.arg == "_minimum_resident_programs"
    )
    minimum_residency.value = ast.Constant(executor_wave_size)
    if launch_pdl:
        launcher.keywords.append(
            ast.keyword(arg="launch_pdl", value=ast.Constant(True))
        )

    w13_tasks = batch * w13_tasks_per_batch
    if order == "root":
        mapping_source = f"{command_name} = {stream_name}\n"
    elif order == "batch":
        mapping_source = f"""
tile_dependency_batch = {stream_name} // {w13_tasks_per_batch + w2_tasks_per_batch}
tile_dependency_batch_command = {stream_name} % {w13_tasks_per_batch + w2_tasks_per_batch}
{command_name} = tl.where(
    tile_dependency_batch_command < {w13_tasks_per_batch},
    tile_dependency_batch_command % {fan_in}
    + {fan_in} * (
        tile_dependency_batch
        + {batch} * (tile_dependency_batch_command // {fan_in})
    ),
    {w13_tasks}
    + tile_dependency_batch * {w2_tasks_per_batch}
    + tile_dependency_batch_command
    - {w13_tasks_per_batch},
)
"""
    elif order == "prefix-one-batch":
        activation_groups = w13_tasks_per_batch // fan_in
        first_groups = min(
            activation_groups,
            schedule_wave_size // (batch * fan_in),
        )
        producer_prefix = first_groups * batch * fan_in
        if not 0 < producer_prefix < w13_tasks:
            raise ValueError("resident wave does not define a proper W13 prefix")
        mapping_source = f"""
{command_name} = tl.where(
    {stream_name} < {producer_prefix},
    {stream_name},
    tl.where(
        {stream_name} < {producer_prefix + w2_tasks_per_batch},
        {w13_tasks} + {stream_name} - {producer_prefix},
        tl.where(
            {stream_name} < {w13_tasks + w2_tasks_per_batch},
            {stream_name} - {w2_tasks_per_batch},
            {stream_name},
        ),
    ),
)
"""
    elif order == "interleave":
        activation_groups = w13_tasks_per_batch // fan_in
        first_groups = min(
            activation_groups,
            schedule_wave_size // (batch * fan_in),
        )
        producer_prefix = first_groups * batch * fan_in
        producer_tail = w13_tasks - producer_prefix
        w2_tasks = batch * w2_tasks_per_batch
        if producer_tail != w2_tasks:
            raise ValueError(
                "the initial interleave probe requires equal producer and consumer tails"
            )
        mapping_source = f"""
tile_dependency_interleave = {stream_name} - {producer_prefix}
{command_name} = tl.where(
    {stream_name} < {producer_prefix},
    {stream_name},
    tl.where(
        tile_dependency_interleave % 2 == 0,
        {producer_prefix} + tile_dependency_interleave // 2,
        {w13_tasks} + tile_dependency_interleave // 2,
    ),
)
"""
    else:
        raise ValueError(f"unsupported task-stream order {order!r}")
    mapping = ast.parse(mapping_source).body
    dispatch_index = next(
        index
        for index, statement in enumerate(loop.body)
        if isinstance(statement, ast.If)
        and _calls_named(statement, "tile_dependency_root_0_scheduled_task")
    )
    loop.body[dispatch_index:dispatch_index] = mapping

    if executor_mode == "two-wave":
        if command_count > 2 * executor_wave_size:
            raise ValueError("two-wave lowering requires N <= 2 * executor workers")
        ticket_assignment = loop.body[0]
        stream_assignment = loop.body[1]
        epoch_assignment = loop.body[2]
        dispatch = next(
            statement
            for statement in loop.body
            if isinstance(statement, ast.If)
            and _calls_named(statement, "tile_dependency_root_0_scheduled_task")
        )
        completion = next(
            statement
            for statement in loop.body
            if isinstance(statement, ast.If)
            and any(
                isinstance(child, ast.Name)
                and child.id.startswith("tile_dependency_clc_success")
                for child in ast.walk(statement)
            )
        )
        success_branch = next(
            statement
            for statement in completion.body
            if isinstance(statement, ast.If)
            and any(
                isinstance(child, ast.Name)
                and child.id.startswith("tile_dependency_clc_success")
                for child in ast.walk(statement.test)
            )
        )
        success_branch.body = copy.deepcopy(
            [ticket_assignment, stream_assignment, epoch_assignment, *mapping, dispatch]
        )
        success_branch.orelse = []
        completion.body = [
            statement
            for statement in completion.body
            if not (
                isinstance(statement, ast.Assign)
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id.startswith("tile_dependency_clc_phase")
            )
        ]
        completion.orelse = []
        loop_index = master.body.index(loop)
        master.body[loop_index : loop_index + 1] = loop.body
        master.body = [
            statement
            for statement in master.body
            if not (
                isinstance(statement, ast.Assign)
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id.startswith("tile_dependency_clc_active")
            )
        ]
    elif executor_mode == "strided":
        ticket_assignment = next(
            statement
            for statement in loop.body
            if isinstance(statement, ast.Assign)
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id.startswith("tile_dependency_ticket")
        )
        epoch_assignment = next(
            statement
            for statement in loop.body
            if isinstance(statement, ast.Assign)
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id.startswith("tile_dependency_epoch")
        )
        stream_assignment = next(
            statement
            for statement in loop.body
            if isinstance(statement, ast.Assign)
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == stream_name
        )
        ticket_name = ticket_assignment.targets[0].id
        ticket_state = ast.unparse(ticket_assignment.value.args[0])
        stream_assignment.value = ast.parse(
            f"({ticket_name} % {executor_wave_size}).to(tl.int32)",
            mode="eval",
        ).body
        epoch_assignment.value = ast.parse(
            f"({ticket_name} // {executor_wave_size} + 1).to(tl.uint32)",
            mode="eval",
        ).body
        loop.body = [
            statement
            for statement in loop.body
            if statement not in (ticket_assignment, stream_assignment, epoch_assignment)
        ]

        completion = next(
            statement
            for statement in loop.body
            if isinstance(statement, ast.If)
            and any(
                isinstance(child, ast.Name)
                and child.id.startswith("tile_dependency_clc_success")
                for child in ast.walk(statement)
            )
        )
        success_branch = next(
            statement
            for statement in completion.body
            if isinstance(statement, ast.If)
            and any(
                isinstance(child, ast.Name)
                and child.id.startswith("tile_dependency_clc_success")
                for child in ast.walk(statement.test)
            )
        )
        success_branch.body = ast.parse(f"{stream_name} += {executor_wave_size}").body

        cohort_ready = "tile_dependency_cohort_ready"
        cohort_target = "tile_dependency_cohort_target"
        load_source = (
            "tl.inline_asm_elementwise("
            "asm='ld.acquire.gpu.global.u64 $0, [$1];', "
            "constraints='=l,l', "
            f"args=[{ticket_state}], dtype=tl.uint64, "
            "is_pure=False, pack=1)"
        )
        cohort_statements = ast.parse(
            f"""
{cohort_target} = ({ticket_name} // {executor_wave_size} + 1) * {executor_wave_size}
{cohort_ready} = {load_source}
while {cohort_ready} < {cohort_target}:
    {cohort_ready} = {load_source}
"""
        ).body
        loop_index = master.body.index(loop)
        master.body[loop_index:loop_index] = [
            ticket_assignment,
            stream_assignment,
            epoch_assignment,
        ]
        master.body[loop_index + 4 : loop_index + 4] = cohort_statements
    return ast.unparse(ast.fix_missing_locations(module)) + "\n"


def _load_generated(
    source: str,
    path: Path,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    path.write_text(source)
    filename = str(path)
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace: dict[str, object] = {"__name__": "_qwen3_ffn_batch_stream"}
    exec(compile(source, filename, "exec"), namespace)
    return namespace["qwen3_ffn_tile_dependency_source"]


def _compile_separate(
    inputs: tuple[torch.Tensor, ...],
    group: int,
) -> Callable[[], torch.Tensor]:
    ffn_q, ffn_scale, w13_q, w13_scale, w2_q, w2_scale = inputs
    _, w13 = compile_config(
        block_fp8_mm,
        (ffn_q, ffn_scale, w13_q, w13_scale, group),
        FFN_CONFIGS["w13"],
    )
    gate = w13(ffn_q, ffn_scale, w13_q, w13_scale, group)
    _, activation = compile_config(
        silu_and_mul_per_block_quant,
        (gate, group),
        FFN_CONFIGS["silu_quant"],
    )
    activation_q, activation_scale = activation(gate, group)
    _, w2 = compile_config(
        block_fp8_mm,
        (activation_q, activation_scale, w2_q, w2_scale, group),
        FFN_CONFIGS["w2"],
    )

    def launch() -> torch.Tensor:
        current_gate = w13(ffn_q, ffn_scale, w13_q, w13_scale, group)
        current_q, current_scale = activation(current_gate, group)
        return w2(current_q, current_scale, w2_q, w2_scale, group)

    return launch


def run(args: argparse.Namespace) -> None:
    if os.environ.get("MEGAKERNEL_CLEAR_L2") != "1":
        raise RuntimeError("set MEGAKERNEL_CLEAR_L2=1 for this comparison")
    if not args.allow_busy:
        require_idle_visible_gpu()

    inputs = _allocate_inputs(args)
    kernel_args = (*inputs, args.group)
    bound = qwen3_ffn_tile_dependency.bind(kernel_args)
    config = _persistent_config(bound, args)
    root_major = bound.compile_config(config)

    source = bound.to_triton_code(config)
    fan_in = 2 * args.group // args.w13_block_n
    w13_tasks_per_batch = 2 * args.intermediate // args.w13_block_n
    w2_tasks_per_batch = args.hidden // args.w2_block_n
    executor_workers = args.executor_workers or args.cross_loop_workers
    command_count = args.batch * (w13_tasks_per_batch + w2_tasks_per_batch)
    if not 0 < executor_workers < command_count:
        raise ValueError("executor worker count must be smaller than the command grid")
    reordered_source = _reordered_source(
        source,
        order=args.order,
        batch=args.batch,
        intermediate=args.intermediate,
        w13_tasks_per_batch=w13_tasks_per_batch,
        w2_tasks_per_batch=w2_tasks_per_batch,
        fan_in=fan_in,
        schedule_wave_size=args.cross_loop_workers,
        executor_wave_size=executor_workers,
        executor_mode=args.executor_mode,
        minimal_single_warp_clc=args.minimal_single_warp_clc,
        final_arrival_acq_rel=args.final_arrival_acq_rel,
        launch_pdl=args.launch_pdl,
        omit_clc_reuse_fence=args.omit_clc_reuse_fence,
        prefetch_depth=args.prefetch_depth,
    )
    reordered = _load_generated(reordered_source, args.lowered_output.resolve())

    root_result = root_major(*kernel_args)
    reordered_result = reordered(*kernel_args)
    torch.cuda.synchronize()
    for actual, expected in zip(reordered_result, root_result, strict=True):
        torch.testing.assert_close(actual.float(), expected.float(), atol=0, rtol=0)

    separate = _compile_separate(inputs, args.group)
    root_graph, _ = capture(lambda: root_major(*kernel_args))
    reordered_graph, reordered_graph_result = capture(lambda: reordered(*kernel_args))
    separate_graph, separate_result = capture(separate)
    for _ in range(args.correctness_replays):
        reordered_graph.replay()
    root_graph.replay()
    separate_graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        reordered_graph_result[0].float(), root_result[0].float(), atol=0, rtol=0
    )
    torch.testing.assert_close(
        root_result[0].float(), separate_result.float(), atol=0.25, rtol=5e-2
    )

    pids = visible_gpu_pids()
    if not args.allow_busy and (foreign_pids := pids - {os.getpid()}):
        raise RuntimeError(
            f"GPU gained foreign compute processes {sorted(foreign_pids)}"
        )
    timings = benchmark_interleaved(
        {
            "root_major": root_graph.replay,
            args.order: reordered_graph.replay,
            "separate": separate_graph.replay,
        },
        args.repeats,
        args.batch_replays,
    )
    if visible_gpu_pids() != pids:
        raise RuntimeError("GPU process set changed during benchmark")
    print("ROOT_MAJOR_RESOURCES", _helion_resources(root_major), flush=True)
    print("REORDERED_RESOURCES", _helion_resources(reordered), flush=True)
    print("TIMINGS", timings, flush=True)
    print("REORDERED_LOWERED", args.lowered_output.resolve(), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=12288)
    parser.add_argument("--group", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--batch-replays", type=int, default=10)
    parser.add_argument("--correctness-replays", type=int, default=10)
    parser.add_argument("--w13-stages", type=int, default=4)
    parser.add_argument("--w13-unroll", type=int, default=2)
    parser.add_argument("--w13-block-n", type=int, default=16)
    parser.add_argument("--w2-stages", type=int, default=4)
    parser.add_argument("--w2-unroll", type=int, default=4)
    parser.add_argument("--w2-block-n", type=int, default=8)
    parser.add_argument("--kernel-stages", type=int, default=2)
    parser.add_argument("--num-warps", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--maxnreg", type=int, choices=(32, 64, 128, 256))
    parser.add_argument("--worker-multiplier", type=int, default=8)
    parser.add_argument("--cross-loop-workers", type=int, default=1024)
    parser.add_argument(
        "--order",
        choices=("root", "batch", "prefix-one-batch", "interleave"),
        default="root",
    )
    parser.add_argument(
        "--executor-workers",
        type=int,
        default=0,
        help="CLC residency/cancellation cohort; 0 uses --cross-loop-workers",
    )
    parser.add_argument(
        "--executor-mode",
        choices=("ticket", "two-wave", "strided"),
        default="ticket",
    )
    parser.add_argument("--minimal-single-warp-clc", action="store_true")
    parser.add_argument("--final-arrival-acq-rel", action="store_true")
    parser.add_argument("--launch-pdl", action="store_true")
    parser.add_argument("--omit-clc-reuse-fence", action="store_true")
    parser.add_argument("--prefetch-depth", type=int, default=0)
    parser.add_argument("--evict-first", type=int, action="append", default=[])
    parser.add_argument("--evict-last", type=int, action="append", default=[])
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument(
        "--lowered-output",
        type=Path,
        default=Path("/tmp/qwen3_ffn_task_stream_reordered_lowered.py"),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
