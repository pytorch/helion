# ruff: noqa: ANN001, ANN201
# pyrefly: ignore-errors
"""Ready-task handoff probe for the exact generated Gemma 4 FFN roots.

The dense projection families retain arithmetic worker assignment.  A bounded
number of otherwise idle workers may execute activation tasks either by their
logical event key or through epoch-tagged FIFO slots. Any activation task
outside that offload budget is executed immediately by its last producer.
"""

from __future__ import annotations

import argparse
import json
import linecache
from pathlib import Path
from types import SimpleNamespace

from benchmarks.gemma4.common import Gemma4E4BShape
from benchmarks.gemma4.common import allocate_layer
from benchmarks.gemma4.common import benchmark_interleaved
from benchmarks.gemma4.common import capture
from benchmarks.gemma4.common import layer_reference
from benchmarks.gemma4.common import require_idle_visible_gpu
from benchmarks.gemma4.common import visible_gpu_pids
import benchmarks.gemma4.helion_gemma4_e4b_layer as layer
import benchmarks.gemma4.helion_gemma4_e4b_megakernel as mega
import benchmarks.gemma4.triton_gemma4_codegen_schedule_probe as codegen_probe
import torch

import helion

FFN_QUEUE_SOURCE = r"""
@triton.jit
def _ffn_queue_complete_activation(
    gate_up,
    activation,
    split_arrivals,
    split_ready,
    activation_task,
    epoch,
    ROOT_8_OFFSET: tl.constexpr,
    ACTIVATION_TASKS: tl.constexpr,
    FIRST_ACTIVATION_TASKS: tl.constexpr,
):
    tile_dependency_root_8(
        gate_up,
        activation,
        ROOT_8_OFFSET + activation_task,
    )
    tl.debug_barrier()
    split = tl.where(activation_task < FIRST_ACTIVATION_TASKS, 0, 1)
    split_size = tl.where(
        split == 0,
        FIRST_ACTIVATION_TASKS,
        ACTIVATION_TASKS - FIRST_ACTIVATION_TASKS,
    ).to(tl.int32)
    previous = tl.atomic_add(
        split_arrivals + split,
        1,
        sem="acq_rel",
        scope="gpu",
    )
    if previous % split_size == split_size - 1:
        generation = previous // split_size + 1
        tl.atomic_xchg(
            split_ready + split,
            generation,
            sem="release",
            scope="gpu",
        )


@triton.jit
def _ffn_queue_publish(
    queue_tasks,
    queue_epochs,
    enqueue_cursors,
    cursor,
    base,
    activation_task,
    epoch,
    SIZE: tl.constexpr,
):
    ticket = tl.atomic_add(
        enqueue_cursors + cursor,
        1,
        sem="relaxed",
        scope="gpu",
    )
    slot = base + ticket % SIZE
    tl.store(queue_tasks + slot, activation_task)
    tl.atomic_xchg(
        queue_epochs + slot,
        epoch,
        sem="release",
        scope="gpu",
    )


@triton.jit
def _ffn_queue_producer(
    ff_input,
    gate_up_weight,
    gate_up,
    activation,
    activation_arrivals,
    split_arrivals,
    split_ready,
    queue_tasks,
    queue_epochs,
    enqueue_cursors,
    logical_task,
    epoch,
    INTERMEDIATE: tl.constexpr,
    ROOT_7_OFFSET: tl.constexpr,
    ROOT_8_OFFSET: tl.constexpr,
    ACTIVATION_TASKS: tl.constexpr,
    FIRST_ACTIVATION_TASKS: tl.constexpr,
    OFFLOAD_FIRST: tl.constexpr,
    OFFLOAD_SECOND: tl.constexpr,
    USE_QUEUE: tl.constexpr,
):
    subtiles_per_activation: tl.constexpr = _BLOCK_SIZE_24 // _BLOCK_SIZE_21
    fan_in: tl.constexpr = 2 * subtiles_per_activation
    activation_task = logical_task // fan_in
    within_activation = logical_task % fan_in
    half_tasks: tl.constexpr = INTERMEDIATE // _BLOCK_SIZE_21
    physical_task = tl.where(
        within_activation < subtiles_per_activation,
        activation_task * subtiles_per_activation + within_activation,
        half_tasks
        + activation_task * subtiles_per_activation
        + within_activation
        - subtiles_per_activation,
    )
    tile_dependency_root_7(
        ff_input,
        gate_up_weight,
        gate_up,
        ROOT_7_OFFSET + physical_task,
    )
    tl.debug_barrier()
    previous = tl.atomic_add(
        activation_arrivals + activation_task,
        1,
        sem="acq_rel",
        scope="gpu",
    )
    if previous % fan_in == fan_in - 1:
        if activation_task < FIRST_ACTIVATION_TASKS:
            if OFFLOAD_FIRST:
                if activation_task < OFFLOAD_FIRST:
                    if USE_QUEUE:
                        _ffn_queue_publish(
                            queue_tasks,
                            queue_epochs,
                            enqueue_cursors,
                            0,
                            0,
                            activation_task,
                            epoch,
                            OFFLOAD_FIRST,
                        )
                else:
                    _ffn_queue_complete_activation(
                        gate_up,
                        activation,
                        split_arrivals,
                        split_ready,
                        activation_task,
                        epoch,
                        ROOT_8_OFFSET,
                        ACTIVATION_TASKS,
                        FIRST_ACTIVATION_TASKS,
                    )
            else:
                _ffn_queue_complete_activation(
                    gate_up,
                    activation,
                    split_arrivals,
                    split_ready,
                    activation_task,
                    epoch,
                    ROOT_8_OFFSET,
                    ACTIVATION_TASKS,
                    FIRST_ACTIVATION_TASKS,
                )
        else:
            if OFFLOAD_SECOND:
                if activation_task - FIRST_ACTIVATION_TASKS < OFFLOAD_SECOND:
                    if USE_QUEUE:
                        _ffn_queue_publish(
                            queue_tasks,
                            queue_epochs,
                            enqueue_cursors,
                            1,
                            OFFLOAD_FIRST,
                            activation_task,
                            epoch,
                            OFFLOAD_SECOND,
                        )
                else:
                    _ffn_queue_complete_activation(
                        gate_up,
                        activation,
                        split_arrivals,
                        split_ready,
                        activation_task,
                        epoch,
                        ROOT_8_OFFSET,
                        ACTIVATION_TASKS,
                        FIRST_ACTIVATION_TASKS,
                    )
            else:
                _ffn_queue_complete_activation(
                    gate_up,
                    activation,
                    split_arrivals,
                    split_ready,
                    activation_task,
                    epoch,
                    ROOT_8_OFFSET,
                    ACTIVATION_TASKS,
                    FIRST_ACTIVATION_TASKS,
                )


@triton.jit
def _ffn_queue_consume(
    gate_up,
    activation,
    activation_arrivals,
    split_arrivals,
    split_ready,
    queue_tasks,
    queue_epochs,
    slot,
    direct_task,
    epoch,
    ROOT_8_OFFSET: tl.constexpr,
    ACTIVATION_TASKS: tl.constexpr,
    FIRST_ACTIVATION_TASKS: tl.constexpr,
    POLL_DELAY: tl.constexpr,
    USE_QUEUE: tl.constexpr,
):
    if USE_QUEUE:
        ready = _probe_load_acquire(queue_epochs + slot)
        while ready != epoch:
            if POLL_DELAY:
                _probe_nanosleep(POLL_DELAY)
            ready = _probe_load_acquire(queue_epochs + slot)
        _probe_sync_warp()
        activation_task = tl.load(queue_tasks + slot)
    else:
        fan_in: tl.constexpr = 2 * _BLOCK_SIZE_24 // _BLOCK_SIZE_21
        _probe_wait_count(
            activation_arrivals + direct_task,
            epoch * fan_in,
            POLL_DELAY,
            False,
        )
        activation_task = direct_task
    _ffn_queue_complete_activation(
        gate_up,
        activation,
        split_arrivals,
        split_ready,
        activation_task,
        epoch,
        ROOT_8_OFFSET,
        ACTIVATION_TASKS,
        FIRST_ACTIVATION_TASKS,
    )


@triton.jit
def gemma4_ffn_ready_queue(
    ff_input,
    gate_up_weight,
    gate_up,
    activation,
    down_weight,
    down,
    worker_epochs,
    activation_arrivals,
    split_arrivals,
    split_ready,
    enqueue_cursors,
    queue_tasks,
    queue_epochs,
    TOTAL_WORKERS: tl.constexpr,
    H: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    ROOT_7_OFFSET: tl.constexpr,
    ROOT_8_OFFSET: tl.constexpr,
    ROOT_9_OFFSET: tl.constexpr,
    FIRST_ACTIVATION_TASKS: tl.constexpr,
    CONSUMER_WORKERS: tl.constexpr,
    OFFLOAD_FIRST: tl.constexpr,
    OFFLOAD_SECOND: tl.constexpr,
    DOWN_STAGES: tl.constexpr,
    DOWN_UNROLL: tl.constexpr,
    POLL_DELAY: tl.constexpr,
    USE_QUEUE: tl.constexpr,
):
    worker = tl.program_id(0)
    epoch = tl.load(worker_epochs + worker) + 1
    fan_in: tl.constexpr = 2 * _BLOCK_SIZE_24 // _BLOCK_SIZE_21
    activation_tasks: tl.constexpr = tl.cdiv(INTERMEDIATE, _BLOCK_SIZE_24)
    gate_tasks: tl.constexpr = tl.cdiv(2 * INTERMEDIATE, _BLOCK_SIZE_21)
    tail_tasks: tl.constexpr = gate_tasks - TOTAL_WORKERS
    consumer_base: tl.constexpr = TOTAL_WORKERS - CONSUMER_WORKERS

    _ffn_queue_producer(
        ff_input,
        gate_up_weight,
        gate_up,
        activation,
        activation_arrivals,
        split_arrivals,
        split_ready,
        queue_tasks,
        queue_epochs,
        enqueue_cursors,
        worker,
        epoch,
        INTERMEDIATE,
        ROOT_7_OFFSET,
        ROOT_8_OFFSET,
        activation_tasks,
        FIRST_ACTIVATION_TASKS,
        OFFLOAD_FIRST,
        OFFLOAD_SECOND,
        USE_QUEUE,
    )

    if worker < tail_tasks:
        _ffn_queue_producer(
            ff_input,
            gate_up_weight,
            gate_up,
            activation,
            activation_arrivals,
            split_arrivals,
            split_ready,
            queue_tasks,
            queue_epochs,
            enqueue_cursors,
            TOTAL_WORKERS + worker,
            epoch,
            INTERMEDIATE,
            ROOT_7_OFFSET,
            ROOT_8_OFFSET,
            activation_tasks,
            FIRST_ACTIVATION_TASKS,
            OFFLOAD_FIRST,
            OFFLOAD_SECOND,
            USE_QUEUE,
        )

    queue_worker = worker - tail_tasks
    if queue_worker >= 0 and queue_worker < OFFLOAD_FIRST:
        _ffn_queue_consume(
            gate_up,
            activation,
            activation_arrivals,
            split_arrivals,
            split_ready,
            queue_tasks,
            queue_epochs,
            queue_worker,
            queue_worker,
            epoch,
            ROOT_8_OFFSET,
            activation_tasks,
            FIRST_ACTIVATION_TASKS,
            POLL_DELAY,
            USE_QUEUE,
        )
    second_worker = queue_worker - OFFLOAD_FIRST
    if second_worker >= 0 and second_worker < OFFLOAD_SECOND:
        _ffn_queue_consume(
            gate_up,
            activation,
            activation_arrivals,
            split_arrivals,
            split_ready,
            queue_tasks,
            queue_epochs,
            OFFLOAD_FIRST + second_worker,
            FIRST_ACTIVATION_TASKS + second_worker,
            epoch,
            ROOT_8_OFFSET,
            activation_tasks,
            FIRST_ACTIVATION_TASKS,
            POLL_DELAY,
            USE_QUEUE,
        )

    if worker >= consumer_base:
        output_task = worker - consumer_base
        _probe_down_two_splits(
            activation,
            down_weight,
            down,
            split_ready,
            output_task,
            epoch,
            H,
            INTERMEDIATE,
            FIRST_ACTIVATION_TASKS,
            DOWN_STAGES,
            DOWN_UNROLL,
            POLL_DELAY,
            True,
            False,
        )

    tl.store(worker_epochs + worker, epoch)
"""


def codegen_args(args) -> SimpleNamespace:
    return SimpleNamespace(
        config_path=args.config_path,
        attention_block=32,
        attention_q_block=None,
        attention_range_stages=None,
        full_splits=64,
        sliding_splits=16,
        num_warps=2,
        kernel_stages=4,
        qkv_block_n=8,
        qkv_block_k=256,
        qkv_range_stages=4,
        qkv_unroll_factor=0,
        o_block_n=16,
        o_block_k=512,
        o_range_stages=3,
        o_unroll_factor=0,
        gate_block_n=32,
        gate_block_k=256,
        activation_block=256,
        down_block_n=16,
        down_block_k=512,
        ple_gate_block_n=None,
        ple_gate_block_k=256,
        ple_projection_block_n=None,
        ple_projection_block_k=None,
        match_gate_eviction=True,
        disable_gate_warp_specialize=False,
        gate_unroll_factor=2,
        gate_range_stages=2,
        force_emitted_gate_stages=None,
        down_unroll_factor=None,
        down_range_stages=None,
        match_effective_standalone_ranges=False,
        match_standalone_eviction=False,
        gate_root_noinline=False,
        inline_gate_body=False,
        ffn_scheduled_activation=False,
        ffn_stream=True,
    )


def compile_namespace(args, tensors, shape, geometry):
    splits = 16
    kernel_args = mega._megakernel_args(tensors, shape, geometry, splits)
    bound = mega.NONSHARED_MEGAKERNEL.bind(kernel_args)
    generated_args = codegen_args(args)
    config = codegen_probe._config_for_probe(bound, generated_args, geometry)
    namespace, lowered = codegen_probe._generated_root_namespace(
        bound,
        config,
        generated_args,
    )
    source = lowered + "\n" + FFN_QUEUE_SOURCE
    filename = str(Path(__file__).with_name("_generated_gemma4_ffn_ready_queue.py"))
    linecache.cache[filename] = (
        len(FFN_QUEUE_SOURCE),
        None,
        FFN_QUEUE_SOURCE.splitlines(keepends=True),
        filename,
    )
    exec(compile(FFN_QUEUE_SOURCE, filename, "exec"), namespace)
    Path(args.lowered_output).write_text(source)
    return namespace, bound, config


def root_offsets(namespace, shape, geometry) -> tuple[int, int, int]:
    def block(block_id: int) -> int:
        value = namespace[f"_BLOCK_SIZE_{block_id}"]
        return int(getattr(value, "value", value))

    q_per_kv = shape.q_heads // shape.kv_heads
    projected_width = (shape.q_heads + 2 * shape.kv_heads) * geometry.head_dim
    root_7 = (
        1
        + (projected_width + block(3) - 1) // block(3)
        + (shape.q_heads + 2 * shape.kv_heads)
        * ((geometry.head_dim + block(7) - 1) // block(7))
        + 16 * shape.kv_heads * ((q_per_kv + block(10) - 1) // block(10))
        + shape.kv_heads * ((q_per_kv + block(14) - 1) // block(14))
        + (shape.hidden + block(17) - 1) // block(17)
        + 1
    )
    root_8 = root_7 + (2 * shape.intermediate + block(21) - 1) // block(21)
    root_9 = root_8 + (shape.intermediate + block(24) - 1) // block(24)
    return root_7, root_8, root_9


def allocate_state(shape, args):
    device = "cuda"
    activation_tasks = shape.intermediate // 256
    return {
        "gate_up": torch.empty(
            (1, 2 * shape.intermediate), device=device, dtype=torch.bfloat16
        ),
        "activation": torch.empty(
            (1, shape.intermediate), device=device, dtype=torch.bfloat16
        ),
        "down": torch.empty((1, shape.hidden), device=device, dtype=torch.bfloat16),
        "worker_epochs": torch.zeros(args.workers, device=device, dtype=torch.int32),
        "activation_arrivals": torch.zeros(
            activation_tasks, device=device, dtype=torch.int32
        ),
        "split_arrivals": torch.zeros(2, device=device, dtype=torch.int32),
        "split_ready": torch.zeros(2, device=device, dtype=torch.int32),
        "enqueue_cursors": torch.zeros(2, device=device, dtype=torch.int32),
        "queue_tasks": torch.empty(activation_tasks, device=device, dtype=torch.int32),
        "queue_epochs": torch.zeros(activation_tasks, device=device, dtype=torch.int32),
    }


def launch_candidate(
    kernel,
    tensors,
    reference,
    state,
    shape,
    offsets,
    args,
    offload_first,
    offload_second,
):
    return kernel[(args.workers,)](
        reference["ff_input"],
        tensors["gate_up_weight"],
        state["gate_up"],
        state["activation"],
        tensors["down_weight"],
        state["down"],
        state["worker_epochs"],
        state["activation_arrivals"],
        state["split_arrivals"],
        state["split_ready"],
        state["enqueue_cursors"],
        state["queue_tasks"],
        state["queue_epochs"],
        args.workers,
        shape.hidden,
        shape.intermediate,
        *offsets,
        args.first_groups,
        args.consumer_workers,
        offload_first,
        offload_second,
        args.down_stages,
        args.down_unroll,
        args.poll_delay,
        args.ready_queue,
        num_warps=2,
        num_stages=4,
    )


def compile_helion_kernel(kernel, kernel_args, config_dict=None):
    bound = kernel.bind(kernel_args)
    if config_dict is None:
        config = bound.config_spec.default_config()
    else:
        config = helion.Config.from_dict(config_dict)
        bound.config_spec.normalize(config.config)
    return bound, config, bound.compile_config(config)


def compile_helion_ffn(tensors, reference, shape, configs):
    gate_args = (reference["ff_input"], tensors["gate_up_weight"])
    gate_bound, gate_config, gate_kernel = compile_helion_kernel(
        layer.bf16_mm,
        gate_args,
        configs["gate_up_mm"],
    )
    gate_up = gate_kernel(*gate_args)
    activation_args = (gate_up,)
    activation_bound, activation_config, activation_kernel = compile_helion_kernel(
        layer.geglu,
        activation_args,
    )
    activation = activation_kernel(*activation_args)
    down_args = (activation, tensors["down_weight"])
    down_bound, down_config, down_kernel = compile_helion_kernel(
        layer.bf16_mm,
        down_args,
        configs["down_mm"],
    )
    down = down_kernel(*down_args)
    return {
        "bounds": (gate_bound, activation_bound, down_bound),
        "configs": (gate_config, activation_config, down_config),
        "kernels": (gate_kernel, activation_kernel, down_kernel),
        "gate_args": gate_args,
        "down_weight": tensors["down_weight"],
        "outputs": (gate_up, activation, down),
    }


def launch_helion_ffn(compiled):
    gate_kernel, activation_kernel, down_kernel = compiled["kernels"]
    gate_up = gate_kernel(*compiled["gate_args"])
    activation = activation_kernel(gate_up)
    return down_kernel(activation, compiled["down_weight"])


def helion_resources(compiled_wrapper):
    kernels = []
    for value in compiled_wrapper.__globals__.values():
        device_caches = getattr(value, "device_caches", None)
        if not device_caches or torch.cuda.current_device() not in device_caches:
            continue
        kernels.extend(device_caches[torch.cuda.current_device()][0].values())
    if len(kernels) != 1:
        raise RuntimeError(f"expected one compiled Helion kernel, found {len(kernels)}")
    kernel = kernels[0]
    return {
        "registers": kernel.n_regs,
        "spills": kernel.n_spills,
        "shared": kernel.metadata.shared,
    }


def triton_resources(compiled):
    return {
        "registers": compiled.n_regs,
        "spills": compiled.n_spills,
        "shared": compiled.metadata.shared,
        "ptx_atomics": compiled.asm["ptx"].count("atom."),
        "ptx_acquire_loads": compiled.asm["ptx"].count("ld.acquire"),
    }


def assert_exact(name, actual, expected):
    if torch.equal(actual, expected):
        return
    difference = (actual.float() - expected.float()).abs()
    raise AssertionError(
        f"{name} changed numerics: max_abs={difference.max().item()}, "
        f"mean_abs={difference.mean().item()}"
    )


def assert_close(name, actual, expected):
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=2e-1,
        rtol=8e-2,
    )


def run(args) -> None:
    require_idle_visible_gpu()
    shape = Gemma4E4BShape(context=8192, block_size=16)
    geometry = shape.layer_geometry(0)
    tensors = allocate_layer(shape, geometry, args.seed)
    reference = layer_reference(tensors, shape, geometry)
    namespace, bound, config = compile_namespace(args, tensors, shape, geometry)
    offsets = root_offsets(namespace, shape, geometry)
    kernel = namespace["gemma4_ffn_ready_queue"]

    configs = json.loads(Path(args.config_path).read_text())
    helion_ffn = compile_helion_ffn(tensors, reference, shape, configs)
    immediate_state = allocate_state(shape, args)
    queue_state = allocate_state(shape, args)
    immediate_compiled = launch_candidate(
        kernel,
        tensors,
        reference,
        immediate_state,
        shape,
        offsets,
        args,
        0,
        0,
    )
    queue_compiled = launch_candidate(
        kernel,
        tensors,
        reference,
        queue_state,
        shape,
        offsets,
        args,
        args.offload_first,
        args.offload_second,
    )
    torch.cuda.synchronize()

    helion_gate, helion_activation, helion_down = helion_ffn["outputs"]
    assert_exact(
        "queue_gate_vs_immediate",
        queue_state["gate_up"],
        immediate_state["gate_up"],
    )
    assert_exact(
        "queue_activation_vs_immediate",
        queue_state["activation"],
        immediate_state["activation"],
    )
    assert_exact(
        "queue_down_vs_immediate",
        queue_state["down"],
        immediate_state["down"],
    )
    assert_close("immediate_gate_vs_helion", immediate_state["gate_up"], helion_gate)
    assert_close(
        "immediate_activation_vs_helion",
        immediate_state["activation"],
        helion_activation,
    )
    assert_close("immediate_down_vs_helion", immediate_state["down"], helion_down)

    helion_graph, helion_graph_out = capture(lambda: launch_helion_ffn(helion_ffn))
    immediate_graph, _ = capture(
        lambda: (
            launch_candidate(
                kernel,
                tensors,
                reference,
                immediate_state,
                shape,
                offsets,
                args,
                0,
                0,
            ),
            immediate_state["down"],
        )[1]
    )
    queue_graph, _ = capture(
        lambda: (
            launch_candidate(
                kernel,
                tensors,
                reference,
                queue_state,
                shape,
                offsets,
                args,
                args.offload_first,
                args.offload_second,
            ),
            queue_state["down"],
        )[1]
    )
    for _ in range(args.correctness_replays):
        queue_graph.replay()
    immediate_graph.replay()
    helion_graph.replay()
    torch.cuda.synchronize()
    assert_exact("queue_replay", queue_state["down"], immediate_state["down"])
    assert_close("helion_replay", immediate_state["down"], helion_graph_out)

    pids = visible_gpu_pids()
    candidate_name = (
        "ready_queue_offload" if args.ready_queue else "direct_keyed_offload"
    )
    timings = benchmark_interleaved(
        {
            "best_standalone_helion_ffn": helion_graph.replay,
            "immediate_on_ready": immediate_graph.replay,
            candidate_name: queue_graph.replay,
        },
        args.repeats,
        args.batch_replays,
    )
    if visible_gpu_pids() != pids:
        raise RuntimeError("GPU process set changed during benchmark")

    baseline = timings["best_standalone_helion_ffn"]["median_us"]
    for name in ("immediate_on_ready", candidate_name):
        value = timings[name]["median_us"]
        timings[name]["reduction_vs_helion_pct"] = 100.0 * (baseline - value) / baseline

    lowered = Path(args.lowered_output)
    names = ("gate_up", "geglu", "down")
    helion_paths = []
    for name, stage_bound, stage_config in zip(
        names,
        helion_ffn["bounds"],
        helion_ffn["configs"],
        strict=True,
    ):
        path = lowered.with_name(f"{lowered.stem}_helion_{name}.py")
        path.write_text(stage_bound.to_triton_code(stage_config))
        helion_paths.append(str(path))
    queue_ptx = lowered.with_name(f"{lowered.stem}_queue.ptx")
    immediate_ptx = lowered.with_name(f"{lowered.stem}_immediate.ptx")
    queue_ptx.write_text(queue_compiled.asm["ptx"])
    immediate_ptx.write_text(immediate_compiled.asm["ptx"])

    result = {
        "device": torch.cuda.get_device_name(),
        "helion_module": helion.__file__,
        "shape": {"hidden": shape.hidden, "intermediate": shape.intermediate},
        "schedule": {
            "workers": args.workers,
            "first_groups": args.first_groups,
            "consumer_workers": args.consumer_workers,
            "offload_first": args.offload_first,
            "offload_second": args.offload_second,
            "ready_queue": args.ready_queue,
            "idle_workers": args.workers
            - (2 * shape.intermediate // 32 - args.workers)
            - args.consumer_workers,
        },
        "timings": timings,
        "resources": {
            "helion_gate_up": helion_resources(helion_ffn["kernels"][0]),
            "helion_geglu": helion_resources(helion_ffn["kernels"][1]),
            "helion_down": helion_resources(helion_ffn["kernels"][2]),
            "immediate_on_ready": triton_resources(immediate_compiled),
            candidate_name: triton_resources(queue_compiled),
        },
        "lowered": {
            "megakernel": str(lowered),
            "helion": helion_paths,
            "ptx": [str(queue_ptx), str(immediate_ptx)],
        },
    }
    print("RESULT_JSON", json.dumps(result, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=576)
    parser.add_argument("--first-groups", type=int, default=36)
    parser.add_argument("--consumer-workers", type=int, default=160)
    parser.add_argument("--offload-first", type=int, default=36)
    parser.add_argument("--offload-second", type=int, default=4)
    parser.add_argument("--down-stages", type=int, default=3)
    parser.add_argument("--down-unroll", type=int, default=0)
    parser.add_argument("--poll-delay", type=int, default=32)
    parser.add_argument(
        "--ready-queue",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--batch-replays", type=int, default=20)
    parser.add_argument("--correctness-replays", type=int, default=20)
    parser.add_argument(
        "--config-path",
        default="benchmarks/gemma4/gemma4_e4b_b200_configs.json",
    )
    parser.add_argument(
        "--lowered-output",
        default="/tmp/gemma4_ffn_ready_queue_lowered.py",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
