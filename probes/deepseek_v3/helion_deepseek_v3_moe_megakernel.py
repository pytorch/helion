# ruff: noqa: A002, ANN001, ANN202, ANN204
# pyrefly: ignore-errors
"""Compiler-generated CLC probe for a DeepSeek-V3 decode MoE.

The source preserves the important vLLM-style boundaries: router, top-k,
routed W13, SwiGLU, routed W2, weighted reduction, shared W13, shared SwiGLU,
shared W2, and the final branch join.  It is intentionally a scheduling probe,
not a replacement implementation of vLLM's fused grouped-MoE kernels.
"""

from __future__ import annotations

import argparse
import ast
import json
import linecache
import os
from pathlib import Path

import torch

from probes.common import benchmark_interleaved
from probes.common import capture
from probes.common import require_idle_visible_gpu
from probes.common import visible_gpu_pids
from probes.gemma4.helion_gemma4_e4b_megakernel import _Bridge
from probes.gemma4.helion_gemma4_e4b_megakernel import _helion_resources
from probes.gemma4.helion_gemma4_e4b_megakernel import _inline_invocation
from probes.gemma4.helion_gemma4_e4b_megakernel import _Invocation

import helion
import helion.language as hl


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def router_projection(
    hidden: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    tokens, hidden_size = hidden.size()
    experts, weight_hidden = weight.size()
    assert hidden_size == weight_hidden
    hl.specialize(hidden_size)
    hl.specialize(experts)
    output = torch.empty(
        (tokens, experts),
        dtype=torch.float32,
        device=hidden.device,
    )
    for tile_m, tile_n in hl.tile(
        [tokens, experts],
        block_size=[1, 2],
    ):
        accumulator = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(hidden_size, block_size=512):
            accumulator = torch.addmm(
                accumulator,
                hidden[tile_m, tile_k],
                weight[tile_n, tile_k].T,
            )
        output[tile_m, tile_n] = accumulator
    return output


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    backend="triton",
)
def biased_sigmoid_topk(
    logits: torch.Tensor,
    correction_bias: torch.Tensor,
    top_k: int,
    routed_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, experts = logits.size()
    top_k = hl.specialize(top_k)
    hl.specialize(experts)
    weights = torch.empty(
        (tokens, top_k),
        dtype=torch.float32,
        device=logits.device,
    )
    ids = torch.empty(
        (tokens, top_k),
        dtype=torch.int32,
        device=logits.device,
    )
    for tile_m in hl.tile(tokens, block_size=1):
        scores = torch.sigmoid(logits[tile_m, :])
        selected, selected_ids = torch.topk(
            scores + correction_bias[None, :],
            top_k,
            dim=-1,
            largest=True,
        )
        normalized = selected / torch.sum(selected, dim=-1, keepdim=True)
        weights[tile_m, :] = normalized * routed_scale
        ids[tile_m, :] = selected_ids.to(torch.int32)
    return weights, ids


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    backend="triton",
)
def routed_gate_up(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    tokens, hidden_size = hidden.size()
    experts, twice_intermediate, weight_hidden = weight.size()
    assert tokens == 1
    assert hidden_size == weight_hidden
    top_k = topk_ids.size(1)
    hl.specialize(experts)
    hl.specialize(twice_intermediate)
    flat_weight = weight.view(experts * twice_intermediate, hidden_size)
    flat_ids = topk_ids.view(top_k)
    output = torch.empty(
        (top_k, twice_intermediate),
        dtype=hidden.dtype,
        device=hidden.device,
    )
    flat_output = output.view(top_k * twice_intermediate, 1)
    for tile_row in hl.tile(top_k * twice_intermediate, block_size=16):
        slot = tile_row.index // twice_intermediate
        expert_row = tile_row.index % twice_intermediate
        expert = flat_ids[slot]
        selected_row = expert * twice_intermediate + expert_row
        accumulator = hl.zeros([tile_row, 1], dtype=torch.float32)
        for tile_k in hl.tile(hidden_size, block_size=512):
            accumulator = torch.addmm(
                accumulator,
                flat_weight[selected_row, tile_k],
                hidden[:, tile_k].T,
            )
        flat_output[tile_row, :] = accumulator.to(output.dtype)
    return output


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def swiglu(input: torch.Tensor) -> torch.Tensor:
    rows, twice_intermediate = input.size()
    intermediate = twice_intermediate // 2
    hl.specialize(rows)
    hl.specialize(intermediate)
    output = torch.empty(
        (rows, intermediate),
        dtype=input.dtype,
        device=input.device,
    )
    for tile_m, tile_n in hl.tile(
        [rows, intermediate],
        block_size=[1, 256],
    ):
        gate = input[tile_m, tile_n].to(torch.float32)
        up = input[tile_m, tile_n + intermediate]
        output[tile_m, tile_n] = (gate * torch.sigmoid(gate)).to(up.dtype) * up
    return output


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    backend="triton",
)
def routed_down(
    activation: torch.Tensor,
    weight: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    top_k, intermediate = activation.size()
    experts, hidden_size, weight_intermediate = weight.size()
    assert intermediate == weight_intermediate
    hl.specialize(experts)
    hl.specialize(hidden_size)
    flat_ids = topk_ids.view(top_k)
    flat_weight = weight.view(experts * hidden_size, intermediate)
    output = torch.empty(
        (top_k, hidden_size),
        dtype=activation.dtype,
        device=activation.device,
    )
    for tile_slot, tile_n in hl.tile(
        [top_k, hidden_size],
        block_size=[1, 32],
    ):
        expert = flat_ids[tile_slot]
        selected_row = expert * hidden_size + tile_n.index
        accumulator = hl.zeros([tile_slot, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(intermediate, block_size=256):
            accumulator = torch.addmm(
                accumulator,
                activation[tile_slot, tile_k],
                flat_weight[selected_row, tile_k].T,
            )
        output[tile_slot, tile_n] = accumulator.to(output.dtype)
    return output


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def weighted_reduce(
    expert_output: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    top_k, hidden_size = expert_output.size()
    output = torch.empty(
        (1, hidden_size),
        dtype=expert_output.dtype,
        device=expert_output.device,
    )
    flat_weights = topk_weights.view(top_k)
    for tile_n in hl.tile(hidden_size, block_size=256):
        values = expert_output[:, tile_n].to(torch.float32)
        weights = flat_weights[:].view(top_k, 1)
        output[:, tile_n] = torch.sum(values * weights, dim=0, keepdim=True).to(
            output.dtype
        )
    return output


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def dense_projection(
    input: torch.Tensor,
    weight: torch.Tensor,
    output_block: int,
    reduction_block: int,
) -> torch.Tensor:
    tokens, reduction_size = input.size()
    output_size, weight_reduction = weight.size()
    assert reduction_size == weight_reduction
    output_block = hl.specialize(output_block)
    reduction_block = hl.specialize(reduction_block)
    hl.specialize(output_size)
    output = torch.empty(
        (tokens, output_size),
        dtype=input.dtype,
        device=input.device,
    )
    for tile_m, tile_n in hl.tile(
        [tokens, output_size],
        block_size=[1, output_block],
    ):
        accumulator = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(reduction_size, block_size=reduction_block):
            accumulator = torch.addmm(
                accumulator,
                input[tile_m, tile_k],
                weight[tile_n, tile_k].T,
            )
        output[tile_m, tile_n] = accumulator.to(output.dtype)
    return output


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def add_branches(
    routed: torch.Tensor,
    shared: torch.Tensor,
) -> torch.Tensor:
    rows, width = routed.size()
    output = torch.empty_like(routed)
    for tile_m, tile_n in hl.tile([rows, width], block_size=[1, 256]):
        output[tile_m, tile_n] = routed[tile_m, tile_n] + shared[tile_m, tile_n]
    return output


def _events() -> tuple[_Invocation | _Bridge, ...]:
    return (
        _Invocation(
            "router",
            router_projection,
            {"hidden": "hidden_states", "weight": "router_weight"},
            {"output": "router_logits"},
        ),
        _Invocation(
            "shared_w13",
            dense_projection,
            {
                "input": "hidden_states",
                "weight": "shared_w13",
                "output_block": "16",
                "reduction_block": "512",
            },
            {"output": "shared_gate_up"},
        ),
        _Invocation(
            "shared_activation",
            swiglu,
            {"input": "shared_gate_up"},
            {"output": "shared_activation"},
        ),
        _Invocation(
            "topk",
            biased_sigmoid_topk,
            {
                "logits": "router_logits",
                "correction_bias": "correction_bias",
                "top_k": "top_k",
                "routed_scale": "routed_scale",
            },
            {"weights": "topk_weights", "ids": "topk_ids"},
        ),
        _Invocation(
            "routed_w13",
            routed_gate_up,
            {
                "hidden": "hidden_states",
                "weight": "expert_w13",
                "topk_ids": "topk_ids",
            },
            {"output": "expert_gate_up"},
        ),
        _Invocation(
            "routed_activation",
            swiglu,
            {"input": "expert_gate_up"},
            {"output": "expert_activation"},
        ),
        _Invocation(
            "shared_w2",
            dense_projection,
            {
                "input": "shared_activation",
                "weight": "shared_w2",
                "output_block": "32",
                "reduction_block": "256",
            },
            {"output": "shared_output"},
        ),
        _Invocation(
            "routed_w2",
            routed_down,
            {
                "activation": "expert_activation",
                "weight": "expert_w2",
                "topk_ids": "topk_ids",
            },
            {"output": "expert_outputs"},
        ),
        _Invocation(
            "routed_reduce",
            weighted_reduce,
            {
                "expert_output": "expert_outputs",
                "topk_weights": "topk_weights",
            },
            {"output": "routed_output"},
        ),
        _Invocation(
            "join",
            add_branches,
            {"routed": "routed_output", "shared": "shared_output"},
            {"output": "output"},
        ),
    )


OUTPUT_NAMES = (
    "output",
    "router_logits",
    "topk_weights",
    "topk_ids",
    "expert_gate_up",
    "expert_activation",
    "expert_outputs",
    "routed_output",
    "shared_gate_up",
    "shared_activation",
    "shared_output",
)


def _compose_source() -> str:
    preamble: list[ast.stmt] = []
    loops: list[ast.For] = []
    for event in _events():
        if isinstance(event, _Bridge):
            preamble.extend(ast.parse(event.source).body)
            continue
        event_preamble, event_loops = _inline_invocation(event)
        preamble.extend(event_preamble)
        loops.extend(event_loops)
    arguments = (
        "hidden_states",
        "router_weight",
        "correction_bias",
        "expert_w13",
        "expert_w2",
        "shared_w13",
        "shared_w2",
        "top_k",
        "routed_scale",
    )
    function = ast.FunctionDef(
        name="deepseek_v3_moe_megakernel_source",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=name) for name in arguments],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[
            *preamble,
            *loops,
            ast.Return(
                value=ast.Tuple(
                    elts=[ast.Name(id=name, ctx=ast.Load()) for name in OUTPUT_NAMES],
                    ctx=ast.Load(),
                )
            ),
        ],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    return ast.unparse(module) + "\n"


def _build_megakernel():
    source = _compose_source()
    filename = str(Path(__file__).with_name("_generated_deepseek_v3_moe.py"))
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace = globals()
    exec(compile(source, filename, "exec"), namespace)
    return (
        helion.kernel(
            static_shapes=True,
            autotune_effort="none",
            backend="triton",
        )(namespace["deepseek_v3_moe_megakernel_source"]),
        source,
    )


MEGAKERNEL, GENERATED_SOURCE = _build_megakernel()


def _allocate(args):
    torch.manual_seed(args.seed)
    device = "cuda"
    dtype = torch.bfloat16
    hidden = args.hidden
    intermediate = args.intermediate
    experts = args.experts
    hidden_states = torch.randn((1, hidden), device=device, dtype=dtype) * 0.02
    router_weight = torch.randn((experts, hidden), device=device, dtype=dtype) * 0.01
    correction_bias = torch.randn((experts,), device=device, dtype=torch.float32) * 0.01
    expert_w13 = torch.empty(
        (experts, 2 * intermediate, hidden),
        device=device,
        dtype=dtype,
    )
    expert_w2 = torch.empty(
        (experts, hidden, intermediate),
        device=device,
        dtype=dtype,
    )
    shared_w13 = torch.empty(
        (2 * intermediate, hidden),
        device=device,
        dtype=dtype,
    )
    shared_w2 = torch.empty(
        (hidden, intermediate),
        device=device,
        dtype=dtype,
    )
    expert_scale = (
        (torch.arange(experts, device=device, dtype=torch.float32) % 17 + 1) * 0.0001
    ).to(dtype)
    expert_w13.copy_(expert_scale[:, None, None])
    expert_w2.copy_(expert_scale.flip(0)[:, None, None])
    shared_w13.fill_(0.001)
    shared_w2.fill_(0.001)
    return (
        hidden_states,
        router_weight,
        correction_bias,
        expert_w13,
        expert_w2,
        shared_w13,
        shared_w2,
        args.top_k,
        args.routed_scale,
    )


def _config(bound, *, workers: int, num_warps: int, persistent: bool):
    values = dict(bound.config_spec.default_config())
    if values.get("range_num_stages"):
        values["range_num_stages"] = [4] * len(values["range_num_stages"])
    values.update({"num_warps": num_warps, "num_stages": 1})
    if persistent:
        values.update(
            {
                "pid_type": "persistent_blocked",
                "num_sm_multiplier": 4,
                "cross_loop_num_workers": workers,
            }
        )
    config = helion.Config.from_dict(values)
    bound.config_spec.normalize(config.config)
    return config


def _compile(kernel, args, *, workers=0, num_warps=1, persistent=False):
    bound = kernel.bind(args)
    return bound.compile_config(
        _config(
            bound,
            workers=workers,
            num_warps=num_warps,
            persistent=persistent,
        )
    )


class SeparateMoE:
    def __init__(self, args) -> None:
        self.args = args
        self.router = _compile(router_projection, args[:2])
        self.shared_w13 = _compile(
            dense_projection,
            (args[0], args[5], 16, 512),
        )
        self.shared_activation = _compile(
            swiglu, (self.shared_w13(args[0], args[5], 16, 512),)
        )
        shared_activation = self.shared_activation(
            self.shared_w13(args[0], args[5], 16, 512)
        )
        self.shared_w2 = _compile(
            dense_projection,
            (shared_activation, args[6], 32, 256),
        )
        router_logits = self.router(args[0], args[1])
        self.topk = _compile(
            biased_sigmoid_topk,
            (router_logits, args[2], args[7], args[8]),
        )
        topk_weights, topk_ids = self.topk(
            router_logits,
            args[2],
            args[7],
            args[8],
        )
        self.routed_w13 = _compile(
            routed_gate_up,
            (args[0], args[3], topk_ids),
        )
        gate_up = self.routed_w13(args[0], args[3], topk_ids)
        self.routed_activation = _compile(swiglu, (gate_up,))
        activation = self.routed_activation(gate_up)
        self.routed_w2 = _compile(
            routed_down,
            (activation, args[4], topk_ids),
        )
        expert_outputs = self.routed_w2(activation, args[4], topk_ids)
        self.reduce = _compile(weighted_reduce, (expert_outputs, topk_weights))
        routed_output = self.reduce(expert_outputs, topk_weights)
        shared_output = self.shared_w2(shared_activation, args[6], 32, 256)
        self.join = _compile(add_branches, (routed_output, shared_output))
        self.shared_stream = torch.cuda.Stream()
        self.shared_start = torch.cuda.Event()
        self.shared_done = torch.cuda.Event()

    def __call__(self):
        args = self.args
        router_logits = self.router(args[0], args[1])
        self.shared_start.record(torch.cuda.current_stream())
        with torch.cuda.stream(self.shared_stream):
            self.shared_stream.wait_event(self.shared_start)
            shared_gate_up = self.shared_w13(args[0], args[5], 16, 512)
            shared_activation = self.shared_activation(shared_gate_up)
            shared_output = self.shared_w2(shared_activation, args[6], 32, 256)
            self.shared_done.record(self.shared_stream)
        topk_weights, topk_ids = self.topk(
            router_logits,
            args[2],
            args[7],
            args[8],
        )
        expert_gate_up = self.routed_w13(args[0], args[3], topk_ids)
        expert_activation = self.routed_activation(expert_gate_up)
        expert_outputs = self.routed_w2(expert_activation, args[4], topk_ids)
        routed_output = self.reduce(expert_outputs, topk_weights)
        self.shared_done.wait(torch.cuda.current_stream())
        output = self.join(routed_output, shared_output)
        return (
            output,
            router_logits,
            topk_weights,
            topk_ids,
            expert_gate_up,
            expert_activation,
            expert_outputs,
            routed_output,
            shared_gate_up,
            shared_activation,
            shared_output,
        )


def _validate(actual, expected) -> None:
    for name, left, right in zip(OUTPUT_NAMES, actual, expected, strict=True):
        if name == "topk_ids":
            torch.testing.assert_close(left, right)
        else:
            torch.testing.assert_close(
                left.float(),
                right.float(),
                atol=0.08,
                rtol=0.05,
            )


def run(args) -> None:
    if not args.allow_busy:
        require_idle_visible_gpu()
    kernel_args = _allocate(args)
    bound = MEGAKERNEL.bind(kernel_args)
    config = _config(
        bound,
        workers=args.workers,
        num_warps=args.num_warps,
        persistent=True,
    )
    lowered = bound.to_triton_code(config, output_origin_lines=True)
    args.lowered_output.parent.mkdir(parents=True, exist_ok=True)
    args.lowered_output.write_text(lowered)
    print("LOWERED_TRITON", args.lowered_output.resolve(), flush=True)
    if args.print_source:
        print(GENERATED_SOURCE)
    if args.print_lowered:
        print(lowered)
    if args.inspect_only:
        return

    megakernel = bound.compile_config(config)
    separate = SeparateMoE(kernel_args)
    actual = megakernel(*kernel_args)
    expected = separate()
    torch.cuda.synchronize()
    _validate(actual, expected)
    print("MEGAKERNEL_RESOURCES", _helion_resources(megakernel), flush=True)
    if not args.benchmark:
        return

    megakernel_graph, megakernel_output = capture(lambda: megakernel(*kernel_args))
    separate_graph, separate_output = capture(separate)
    megakernel_graph.replay()
    separate_graph.replay()
    torch.cuda.synchronize()
    _validate(megakernel_output, separate_output)
    pids = visible_gpu_pids()
    timings = benchmark_interleaved(
        {
            "helion_deepseek_v3_megakernel": megakernel_graph.replay,
            "helion_deepseek_v3_separate_overlap": separate_graph.replay,
        },
        args.repeats,
        args.batch_replays,
    )
    if visible_gpu_pids() != pids:
        raise RuntimeError("GPU process set changed during benchmark")
    print(
        "RESULT_JSON",
        json.dumps(
            {
                "cold_l2": os.environ.get("MEGAKERNEL_CLEAR_L2") == "1",
                "device": torch.cuda.get_device_name(),
                "resources": _helion_resources(megakernel),
                "timings": timings,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--routed-scale", type=float, default=2.5)
    parser.add_argument("--workers", type=int, default=592)
    parser.add_argument("--num-warps", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--batch-replays", type=int, default=10)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--print-source", action="store_true")
    parser.add_argument("--print-lowered", action="store_true")
    parser.add_argument(
        "--lowered-output",
        type=Path,
        default=Path("/tmp/deepseek_v3_moe_clc_lowered.py"),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
