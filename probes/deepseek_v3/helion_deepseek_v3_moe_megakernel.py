# ruff: noqa: ANN001, ANN201, ANN202
"""Matched-boundary Helion megakernel for the DeepSeek-V3 decode MoE.

The generated Helion source preserves the standalone operator boundaries:

  router -> grouped top-k -> routed W13 -> SwiGLU -> routed W2 -> reduce
  shared W13 -> SwiGLU -> shared W2
  routed + shared

Only the launch boundary changes: all roots are lowered into one persistent
Triton kernel so Helion's cross-loop scheduler can overlap independent work.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import inspect
import json
import linecache
from pathlib import Path
import textwrap
from typing import Protocol

import torch

from probes.common import benchmark_graphs_cold_l2
from probes.common import capture_with_reset
from probes.common import error_stats
from probes.common import require_idle_visible_gpu
from probes.deepseek_v3.deepseek_v3_moe_common import DeepseekV3MoEShape
from probes.deepseek_v3.deepseek_v3_moe_common import allocate_moe
from probes.deepseek_v3.deepseek_v3_moe_common import moe_reference
import probes.deepseek_v3.helion_deepseek_v3_moe as separate

import helion
import helion.language as hl  # noqa: F401 - used by generated source


class _KernelWithFunction(Protocol):
    fn: object


@dataclasses.dataclass(frozen=True)
class _Invocation:
    prefix: str
    kernel: _KernelWithFunction
    arguments: dict[str, str]
    outputs: dict[str, str] = dataclasses.field(default_factory=dict)


class _AssignedNames(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.names.add(node.id)


class _RenameNames(ast.NodeTransformer):
    def __init__(self, names: dict[str, str]) -> None:
        self.names = names

    def visit_Name(self, node: ast.Name) -> ast.Name:
        renamed = self.names.get(node.id)
        if renamed is None:
            return node
        return ast.copy_location(ast.Name(id=renamed, ctx=node.ctx), node)


def _kernel_function_ast(kernel: _KernelWithFunction) -> ast.FunctionDef:
    source = textwrap.dedent(inspect.getsource(kernel.fn))
    module = ast.parse(source)
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    assert len(functions) == 1
    return functions[0]


def _inline_invocation(invocation: _Invocation) -> tuple[list[ast.stmt], list[ast.For]]:
    function = _kernel_function_ast(invocation.kernel)
    parameters = [argument.arg for argument in function.args.args]
    assert set(parameters) == set(invocation.arguments)

    assigned = _AssignedNames()
    for statement in function.body:
        assigned.visit(statement)
    rename = {
        name: invocation.outputs.get(name, f"__deepseek_{invocation.prefix}_{name}")
        for name in set(parameters) | assigned.names
    }
    transformer = _RenameNames(rename)
    preamble: list[ast.stmt] = [
        ast.Assign(
            targets=[ast.Name(id=rename[parameter], ctx=ast.Store())],
            value=ast.parse(invocation.arguments[parameter], mode="eval").body,
        )
        for parameter in parameters
    ]
    loops: list[ast.For] = []
    for statement in function.body:
        if isinstance(statement, ast.Return):
            continue
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            continue
        transformed = transformer.visit(ast.fix_missing_locations(statement))
        assert isinstance(transformed, ast.stmt)
        if isinstance(transformed, ast.For):
            loops.append(transformed)
        else:
            preamble.append(transformed)
    assert loops, invocation.prefix
    return preamble, loops


def _events() -> list[_Invocation]:
    return [
        _Invocation(
            "router",
            separate.router_mm_fp32,
            {"hidden": "hidden_states", "weight": "router_weight"},
            {"output": "router_logits"},
        ),
        _Invocation(
            "topk",
            separate.grouped_topk,
            {
                "logits": "router_logits",
                "correction_bias": "correction_bias",
                "top_k": "top_k",
                "num_groups": "num_groups",
                "topk_groups": "topk_groups",
                "routed_scale": "routed_scale",
            },
            {"weights": "topk_weights", "ids": "topk_ids"},
        ),
        _Invocation(
            "expert_w13",
            separate.selected_expert_w13,
            {
                "hidden": "hidden_states",
                "weight": "expert_w13",
                "topk_ids": "topk_ids",
            },
            {"output": "expert_gate_up"},
        ),
        _Invocation(
            "expert_swiglu",
            separate.silu_and_mul,
            {"gate_up": "expert_gate_up"},
            {"output": "expert_activation"},
        ),
        _Invocation(
            "expert_w2",
            separate.selected_expert_w2,
            {
                "activation": "expert_activation",
                "weight": "expert_w2",
                "topk_ids": "topk_ids",
            },
            {"output": "expert_outputs"},
        ),
        _Invocation(
            "expert_reduce",
            separate.weighted_reduce,
            {
                "expert_outputs": "expert_outputs",
                "topk_weights": "topk_weights",
            },
            {"output": "routed_output"},
        ),
        _Invocation(
            "shared_w13",
            separate.bf16_mm,
            {"x": "hidden_states", "weight": "shared_w13"},
            {"output": "shared_gate_up"},
        ),
        _Invocation(
            "shared_swiglu",
            separate.silu_and_mul,
            {"gate_up": "shared_gate_up"},
            {"output": "shared_activation"},
        ),
        _Invocation(
            "shared_w2",
            separate.bf16_mm,
            {"x": "shared_activation", "weight": "shared_w2"},
            {"output": "shared_output"},
        ),
        _Invocation(
            "final_add",
            separate.add_outputs,
            {"routed": "routed_output", "shared": "shared_output"},
            {"output": "output"},
        ),
    ]


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
        "num_groups",
        "topk_groups",
        "routed_scale",
    )
    function = ast.FunctionDef(
        name="deepseek_v3_moe_matched_megakernel_source",
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
    filename = str(Path(__file__).with_name("_generated_deepseek_v3_moe_matched.py"))
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace = globals()
    exec(compile(source, filename, "exec"), namespace)
    function = namespace["deepseek_v3_moe_matched_megakernel_source"]
    kernel = helion.kernel(
        static_shapes=True,
        autotune_effort="none",
        backend="triton",
    )(function)
    return kernel, source


MEGAKERNEL, GENERATED_SOURCE = _build_megakernel()


def megakernel_args(tensors, shape: DeepseekV3MoEShape):
    return (
        tensors["hidden_states"],
        tensors["router_weight"],
        tensors["correction_bias"],
        tensors["expert_w13"],
        tensors["expert_w2"],
        tensors["shared_w13"],
        tensors["shared_w2"],
        shape.top_k,
        shape.num_groups,
        shape.topk_groups,
        shape.routed_scale,
    )


def _by_block_id(specs, choices, defaults):
    return [
        next(
            (choices[block_id] for block_id in spec.block_ids if block_id in choices),
            default,
        )
        for spec, default in zip(specs, defaults, strict=True)
    ]


def persistent_config(bound, args) -> helion.Config:
    """Build an initial persistent config from the standalone tile geometry."""
    values = dict(bound.config_spec.default_config())
    roots = bound.host_function.device_ir.grid_block_ids

    # Each entry is the configurable output-tile block ID for that root.  The
    # one-program grouped top-k root has no configurable block size.
    output_tiles = {
        roots[0][-1]: args.router_block,
        roots[2][-1]: args.expert_w13_block,
        roots[3][-1]: args.activation_block,
        roots[4][-1]: args.expert_w2_block,
        roots[5][-1]: args.reduce_block,
        roots[6][-1]: args.shared_w13_block,
        roots[7][-1]: args.activation_block,
        roots[8][-1]: args.shared_w2_block,
        roots[9][-1]: args.add_block,
    }
    values["block_sizes"] = [
        output_tiles.get(spec.block_id, default)
        for spec, default in zip(
            bound.config_spec.block_sizes,
            values["block_sizes"],
            strict=True,
        )
    ]

    # Reduction ranges appear in source order.  Match the tuned standalone K
    # tiles while leaving elementwise/root ranges at their normalized defaults.
    reduction_choices: dict[int, int] = {}
    staged_choices: dict[int, int] = {}
    reduction_parameters = {
        roots[0][-1] + 1: (args.router_k, args.router_stages),
        roots[2][-1] + 1: (args.expert_w13_k, args.expert_w13_stages),
        roots[4][-1] + 1: (args.expert_w2_k, args.expert_w2_stages),
        roots[6][-1] + 1: (args.shared_w13_k, args.shared_w13_stages),
        roots[8][-1] + 1: (args.shared_w2_k, args.shared_w2_stages),
    }
    for block_id, (block_size, stages) in reduction_parameters.items():
        reduction_choices[block_id] = block_size
        staged_choices[block_id] = stages
    values["block_sizes"] = [
        reduction_choices.get(spec.block_id, value)
        for spec, value in zip(
            bound.config_spec.block_sizes,
            values["block_sizes"],
            strict=True,
        )
    ]
    values["range_num_stages"] = _by_block_id(
        bound.config_spec.range_num_stages,
        staged_choices,
        values["range_num_stages"],
    )
    values.update(
        {
            "pid_type": "persistent_blocked",
            "num_sm_multiplier": args.worker_multiplier,
            "num_warps": args.num_warps,
            "num_stages": args.kernel_stages,
        }
    )
    if args.workers is not None:
        values["cross_loop_num_workers"] = args.workers
    config = helion.Config.from_dict(values)
    bound.config_spec.normalize(config.config)
    return config


def _resources(compiled) -> dict[str, int]:
    return separate._resources(compiled)


def _validate(outputs, reference):
    expected = {name: reference[name] for name in OUTPUT_NAMES}
    result = {}
    for name, actual in zip(OUTPUT_NAMES, outputs, strict=True):
        wanted = expected[name]
        if name == "topk_ids":
            torch.testing.assert_close(actual, wanted)
            result[name] = {"exact": True}
        else:
            torch.testing.assert_close(
                actual.float(), wanted.float(), atol=0.2, rtol=0.08
            )
            result[name] = error_stats(actual, wanted)
    return result


@torch.inference_mode()
def run(args):
    require_idle_visible_gpu()
    shape = DeepseekV3MoEShape(batch=1)
    tensors = allocate_moe(shape, args.seed)
    reference = moe_reference(tensors, shape)
    kernel_args = megakernel_args(tensors, shape)
    bound = MEGAKERNEL.bind(kernel_args)
    host_function = bound.host_function
    assert host_function is not None
    config = persistent_config(bound, args)

    graph = host_function.device_ir.tile_dependency_graph
    edge_summaries = []
    if graph is not None:
        for edge in graph.edges:
            edge_summaries.append(
                {
                    "producer_root": edge.producer_root,
                    "consumer_root": edge.consumer_root,
                    "tensor_names": sorted(edge.tensor_names),
                    "kinds": sorted(kind.value for kind in edge.kinds),
                    "access_dependencies": [
                        {
                            "coordinates_are_exact": dependency.region.coordinates_are_exact,
                            "is_exact_contiguous": dependency.region.is_exact_contiguous,
                            "coordinate_bounds": dependency.region.coordinate_bounds,
                            "address_interval": dependency.region.address_interval,
                        }
                        for dependency in edge.access_dependencies
                    ],
                }
            )
    inspection = {
        "root_block_ids": host_function.device_ir.grid_block_ids,
        "dependency_edges": edge_summaries,
        "execution_scopes": [
            {
                "scope_id": scope.scope_id,
                "root": scope.root,
                "kind": scope.kind,
                "local_axis_order": scope.local_axis_order,
                "segmentable": scope.segmentable,
            }
            for scope in graph.execution_scopes
        ]
        if graph is not None
        else [],
        "implicit_dependency_starts": sorted(
            host_function.device_ir.implicit_dependency_starts
        ),
        "block_specs": [
            dataclasses.asdict(spec) for spec in bound.config_spec.block_sizes
        ],
        "range_block_ids": [
            tuple(spec.block_ids) for spec in bound.config_spec.range_num_stages
        ],
        "config": dict(config),
    }
    print("INSPECTION", json.dumps(inspection, default=str, sort_keys=True), flush=True)
    if args.print_source:
        print(GENERATED_SOURCE)
        return
    lowered = bound.to_triton_code(config, output_origin_lines=True)
    if args.print_lowered:
        print(lowered)
        return
    if args.inspect_only:
        return

    compiled = bound.compile_config(config)
    outputs = compiled(*kernel_args)
    torch.cuda.synchronize()
    correctness = _validate(outputs, reference)
    resources = _resources(compiled)
    print("CORRECTNESS", json.dumps(correctness, sort_keys=True), flush=True)
    print("RESOURCES", json.dumps(resources, sort_keys=True), flush=True)
    if args.smoke:
        return

    configs = json.loads(Path(args.config_path).read_text())
    standalone = separate.build_moe(
        argparse.Namespace(tune=[]),
        tensors,
        shape,
        configs,
        Path(args.config_path),
    )

    def noop() -> None:
        pass

    megakernel_graph, megakernel_graph_outputs = capture_with_reset(
        lambda: compiled(*kernel_args), noop
    )
    separate_graph, separate_output = capture_with_reset(
        standalone["launch_overlap"], noop
    )
    megakernel_graph.replay()
    separate_graph.replay()
    torch.cuda.synchronize()
    _validate(megakernel_graph_outputs, reference)
    torch.testing.assert_close(
        separate_output.float(), reference["output"].float(), atol=0.2, rtol=0.08
    )
    timings = benchmark_graphs_cold_l2(
        {
            "helion_megakernel": (megakernel_graph.replay, noop),
            "helion_cudagraph_shared_overlap": (separate_graph.replay, noop),
        },
        repeats=args.repeats,
        flush_mib=256,
        order_seed=args.order_seed,
    )
    result = {
        "shape": dataclasses.asdict(shape),
        "config": dict(config),
        "inspection": inspection,
        "resources": resources,
        "correctness": correctness,
        "timings": timings,
    }
    Path(args.output).write_text(json.dumps(result, indent=2, default=str) + "\n")
    print("RESULT", json.dumps(result, default=str, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--order-seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=592)
    parser.add_argument("--worker-multiplier", type=int, default=8)
    parser.add_argument("--num-warps", type=int, default=1)
    parser.add_argument("--kernel-stages", type=int, default=2)
    parser.add_argument("--router-block", type=int, default=2)
    parser.add_argument("--router-k", type=int, default=512)
    parser.add_argument("--router-stages", type=int, default=4)
    parser.add_argument("--expert-w13-block", type=int, default=16)
    parser.add_argument("--expert-w13-k", type=int, default=512)
    parser.add_argument("--expert-w13-stages", type=int, default=4)
    parser.add_argument("--expert-w2-block", type=int, default=32)
    parser.add_argument("--expert-w2-k", type=int, default=512)
    parser.add_argument("--expert-w2-stages", type=int, default=2)
    parser.add_argument("--shared-w13-block", type=int, default=16)
    parser.add_argument("--shared-w13-k", type=int, default=512)
    parser.add_argument("--shared-w13-stages", type=int, default=2)
    parser.add_argument("--shared-w2-block", type=int, default=32)
    parser.add_argument("--shared-w2-k", type=int, default=256)
    parser.add_argument("--shared-w2-stages", type=int, default=3)
    parser.add_argument("--activation-block", type=int, default=256)
    parser.add_argument("--reduce-block", type=int, default=256)
    parser.add_argument("--add-block", type=int, default=256)
    parser.add_argument("--print-source", action="store_true")
    parser.add_argument("--print-lowered", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--config-path",
        default=str(Path(__file__).with_name("deepseek_v3_moe_b200_configs.json")),
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).with_name("deepseek_v3_moe_megakernel_result.json")),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
