# ruff: noqa: ANN001, ANN202
# pyrefly: ignore-errors
"""Compare a GPT-OSS MXFP4 MoE megakernel with separate Helion kernels.

The megakernel is assembled from the four production-boundary kernels in
``helion_gpt_oss_moe.py``: routing, GEMM1 plus OAI SwiGLU, GEMM2 plus bias,
and weighted finalization.  The software-decode GEMV implementations are used
because they are the fastest batch-one standalone Helion kernels.

Cold-L2 timing is selected with ``MEGAKERNEL_CLEAR_L2=1``.  The same tensors,
standalone configurations, CUDA-graph capture, and cache-flush methodology are
used for both sides of the comparison.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import importlib.util
import inspect
import json
import linecache
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time
from typing import Protocol

import torch

SOURCE_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = Path(__file__).with_name("helion_gpt_oss_moe.py")
CONFIG_PATH = Path(__file__).with_name("gpt_oss_mxfp4_decode_b200.json")


def _compiler_root_from_argv() -> Path:
    flag = "--compiler-root"
    if flag in sys.argv:
        index = sys.argv.index(flag)
        return Path(sys.argv[index + 1]).resolve()
    return Path(os.environ.get("HELION_COMPILER_ROOT", SOURCE_ROOT)).resolve()


COMPILER_ROOT = _compiler_root_from_argv()
sys.path.insert(0, str(COMPILER_ROOT))

from probes.common import benchmark_interleaved  # noqa: E402
from probes.common import capture  # noqa: E402
from probes.common import require_idle_visible_gpu  # noqa: E402
from probes.common import visible_gpu_pids  # noqa: E402

import helion  # noqa: E402
from helion import exc  # noqa: E402
import helion.language as hl  # noqa: E402
from helion.language import _decorators  # noqa: E402


def _load_source():
    spec = importlib.util.spec_from_file_location(
        "_gpt_oss_moe_source",
        SOURCE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SOURCE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SOURCE = _load_source()

# These helpers remain global references in the inlined kernel bodies.
_e8m0_byte_to_f32 = SOURCE._e8m0_byte_to_f32
_trtllm_scale_offset = SOURCE._trtllm_scale_offset


@_decorators.api(is_device_only=True)
def _semantic_only(value: torch.Tensor) -> torch.Tensor:
    """Keep an access in DeviceIR while permitting backend DCE."""
    raise exc.NotInsideKernel


@_decorators.register_fake(_semantic_only)
def _(value: torch.Tensor) -> torch.Tensor:
    return value


@_decorators.codegen(_semantic_only, "triton")
def _(state):
    value = state.ast_arg(0)
    if isinstance(value, ast.Name):
        state.device_function.dce_vars.append(value.id)
    return value


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def _mxfp4_moe_gemm1_visible_store(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_scale_bytes: torch.Tensor,
    w13_bias: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """The production W13 GEMV with an explicit logical activation layout.

    Split the physical W13 row into explicit half/parity axes. This preserves
    the standalone 16-row GEMV tile and the ordinary logical activation layout,
    while making the producer partition visible to dependency analysis.
    """
    tokens, hidden = hidden_states.size()
    experts, twice_intermediate, packed_hidden = w13.size()
    top_k = topk_ids.size(1)
    intermediate = twice_intermediate // 2
    scale_k = hidden // 32
    activation_scale_k = intermediate // 32
    assert tokens == 1
    assert packed_hidden * 2 == hidden
    hl.specialize(experts)
    hl.specialize(top_k)
    hl.specialize(intermediate)
    output = torch.empty(
        (top_k, intermediate),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    physical_output = output.view(top_k, activation_scale_k, 2, 8, 2)
    flat_weight = w13.view(torch.uint8).view(-1)
    flat_bias = w13_bias.view(-1)
    flat_scale = w13_scale_bytes.view(-1)
    block_scale_k = hl.register_block_size(1, scale_k)
    for tile_half, tile_parity, tile_activation_group, tile_slot in hl.tile(
        [2, 2, activation_scale_k, top_k],
        block_size=[1, 1, 1, 1],
    ):
        slot = tile_slot.begin
        half = tile_half.begin
        parity = tile_parity.begin
        activation_group = tile_activation_group.begin
        chunk = half * 2 + parity
        physical_row = activation_group * 64 + chunk * 16 + hl.arange(16)
        expert = topk_ids[0, slot]
        expert_row = expert * twice_intermediate + physical_row
        accumulator = hl.zeros([16], dtype=torch.float32)

        for tile_scale_k in hl.tile(scale_k, block_size=block_scale_k):
            group_mask = tile_scale_k.index < scale_k
            subgroup = expert_row[:, None] * (scale_k * 2)
            subgroup += tile_scale_k.index[None, :] * 2
            valid = group_mask[None, :]
            weight_first = hl.load_float4_e2m1fn_x16_to_float16(
                flat_weight,
                subgroup,
                extra_mask=valid,
            )
            weight_second = hl.load_float4_e2m1fn_x16_to_float16(
                flat_weight,
                subgroup + 1,
                extra_mask=valid,
            )
            activation_first = hl.load_bfloat16_x16_to_float16(
                hidden_states,
                tile_scale_k.index * 2,
                extra_mask=group_mask,
            )
            activation_second = hl.load_bfloat16_x16_to_float16(
                hidden_states,
                tile_scale_k.index * 2 + 1,
                extra_mask=group_mask,
            )
            contribution = hl.zeros([16, block_scale_k], dtype=torch.float16)
            for index in hl.static_range(16):
                contribution += weight_first[index] * activation_first[index][None, :]
                contribution += weight_second[index] * activation_second[index][None, :]
            scale_offset = expert * twice_intermediate * scale_k
            scale_offset += _trtllm_scale_offset(
                physical_row[:, None],
                tile_scale_k.index[None, :],
                twice_intermediate,
                scale_k,
            )
            scale = _e8m0_byte_to_f32(hl.load(flat_scale, [scale_offset]))
            accumulator += torch.sum(contribution.to(torch.float32) * scale, dim=-1)

        preactivation = accumulator + flat_bias[expert_row]
        pairs = preactivation.reshape(2, 8).permute(1, 0)
        up_value, gate_value = hl.split(pairs)
        gate_value = torch.clamp(gate_value, max=7.0)
        up_value = torch.clamp(up_value, min=-7.0, max=7.0)
        activated = (up_value + 1.0) * gate_value * torch.sigmoid(1.702 * gate_value)
        physical_output[slot, activation_group, half, :, parity] = activated.to(
            torch.bfloat16
        )
    return output


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def _mxfp4_moe_gemm2_visible_loads(
    activation: torch.Tensor,
    w2: torch.Tensor,
    w2_scale_bytes: torch.Tensor,
    w2_bias: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """GEMM2 variant whose packed loads carry exact DeviceIR dependencies."""
    top_k, intermediate = activation.size()
    experts, hidden, packed_intermediate = w2.size()
    scale_k = intermediate // 32
    assert packed_intermediate * 2 == intermediate
    hl.specialize(experts)
    hl.specialize(top_k)
    hl.specialize(hidden)
    output = torch.empty(
        (top_k, hidden), dtype=torch.bfloat16, device=activation.device
    )
    physical_activation = activation.view(top_k, scale_k, 2, 8, 2)
    flat_weight = w2.view(torch.uint8).view(-1)
    flat_bias = w2_bias.view(-1)
    flat_scale = w2_scale_bytes.view(-1)
    block_physical_row = hl.register_block_size(8, 128)
    block_scale_k = hl.register_block_size(1, scale_k)
    for tile_physical_row, tile_slot in hl.tile(
        [hidden, top_k], block_size=[block_physical_row, 1]
    ):
        slot = tile_slot.begin
        expert = topk_ids[0, slot]
        expert_row = expert * hidden + tile_physical_row.index
        accumulator = hl.zeros([tile_physical_row], dtype=torch.float32)
        for tile_scale_k in hl.tile(scale_k, block_size=block_scale_k):
            group_mask = tile_scale_k.index < scale_k
            subgroup = expert_row[:, None] * (scale_k * 2)
            subgroup += tile_scale_k.index[None, :] * 2
            valid = (tile_physical_row.index[:, None] < hidden) & (group_mask[None, :])
            weight_first = hl.load_float4_e2m1fn_x16_to_float16(
                flat_weight,
                subgroup,
                extra_mask=valid,
            )
            weight_second = hl.load_float4_e2m1fn_x16_to_float16(
                flat_weight,
                subgroup + 1,
                extra_mask=valid,
            )
            activation_group = slot * (scale_k * 2) + tile_scale_k.index * 2
            semantic_activation = _semantic_only(
                hl.load(
                    physical_activation,
                    [
                        slot,
                        tile_scale_k.index,
                        slice(None),
                        slice(None),
                        slice(None),
                    ],
                )
            )
            packed_first = hl.load_bfloat16_x16_to_float16(
                activation,
                activation_group,
                extra_mask=group_mask,
                semantic_dependency=semantic_activation,
            )
            packed_second = hl.load_bfloat16_x16_to_float16(
                activation,
                activation_group + 1,
                extra_mask=group_mask,
                semantic_dependency=semantic_activation,
            )
            contribution = hl.zeros(
                [block_physical_row, block_scale_k], dtype=torch.float16
            )
            for index in hl.static_range(16):
                contribution += weight_first[index] * packed_first[index][None, :]
                contribution += weight_second[index] * packed_second[index][None, :]
            scale_offset = expert * hidden * scale_k + _trtllm_scale_offset(
                tile_physical_row.index[:, None],
                tile_scale_k.index[None, :],
                hidden,
                scale_k,
            )
            scale = _e8m0_byte_to_f32(hl.load(flat_scale, [scale_offset]))
            accumulator += torch.sum(contribution.to(torch.float32) * scale, dim=-1)

        result = accumulator + flat_bias[expert_row]
        physical_row = tile_physical_row.begin + hl.arange(block_physical_row)
        lane = physical_row & 31
        logical_row = physical_row - lane + (lane & 7) * 4 + (lane >> 3)
        output[slot, logical_row] = result.to(torch.bfloat16)
    return output


class _KernelWithFunction(Protocol):
    fn: object


@dataclasses.dataclass(frozen=True)
class _Invocation:
    prefix: str
    kernel: _KernelWithFunction
    arguments: dict[str, str]
    outputs: dict[str, str]
    config_name: str
    source_block_indices: tuple[int, ...] | None = None


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
    if len(functions) != 1:
        raise RuntimeError(f"expected one function for {kernel}")
    return functions[0]


def _inline_invocation(invocation: _Invocation) -> tuple[list[ast.stmt], list[ast.For]]:
    function = _kernel_function_ast(invocation.kernel)
    parameters = [argument.arg for argument in function.args.args]
    if set(parameters) != set(invocation.arguments):
        raise RuntimeError(
            f"argument mismatch for {invocation.prefix}: "
            f"{parameters} != {sorted(invocation.arguments)}"
        )

    assigned = _AssignedNames()
    for statement in function.body:
        assigned.visit(statement)
    rename = {
        name: invocation.outputs.get(name, f"__gpt_oss_{invocation.prefix}_{name}")
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
        if not isinstance(transformed, ast.stmt):
            raise RuntimeError(f"invalid transformed statement for {invocation.prefix}")
        if isinstance(transformed, ast.For):
            loops.append(transformed)
        else:
            preamble.append(transformed)
    if not loops:
        raise RuntimeError(f"no device loops in {invocation.prefix}")
    return preamble, loops


INVOCATIONS = (
    _Invocation(
        "routing",
        SOURCE.mxfp4_top4_routing,
        {"routing_logits": "routing_logits"},
        {"weights": "topk_weights", "ids": "topk_ids"},
        "routing",
    ),
    _Invocation(
        "gemm1",
        _mxfp4_moe_gemm1_visible_store,
        {
            "hidden_states": "hidden_states",
            "w13": "w13",
            "w13_scale_bytes": "w13_scale_bytes",
            "w13_bias": "w13_bias",
            "topk_ids": "topk_ids",
        },
        {"output": "activation"},
        "gemm1_swiglu_oai",
        (1,),
    ),
    _Invocation(
        "gemm2",
        _mxfp4_moe_gemm2_visible_loads,
        {
            "activation": "activation",
            "w2": "w2",
            "w2_scale_bytes": "w2_scale_bytes",
            "w2_bias": "w2_bias",
            "topk_ids": "topk_ids",
        },
        {"output": "expert_output"},
        "gemm2",
    ),
    _Invocation(
        "finalize",
        SOURCE.mxfp4_moe_finalize,
        {
            "expert_output": "expert_output",
            "topk_weights": "topk_weights",
            "output_hidden": "output_hidden",
        },
        {"output": "output"},
        "finalize",
    ),
)

OUTPUT_NAMES = (
    "output",
    "topk_weights",
    "topk_ids",
    "activation",
    "expert_output",
)


def _compose_source() -> str:
    preamble: list[ast.stmt] = []
    loops_by_prefix: dict[str, list[ast.For]] = {}
    for invocation in INVOCATIONS:
        invocation_preamble, invocation_loops = _inline_invocation(invocation)
        preamble.extend(invocation_preamble)
        loops_by_prefix[invocation.prefix] = invocation_loops
    loops = [
        *loops_by_prefix["routing"],
        *loops_by_prefix["gemm1"],
        *loops_by_prefix["gemm2"],
    ]
    loops.extend(loops_by_prefix["finalize"])
    arguments = (
        "routing_logits",
        "hidden_states",
        "w13",
        "w13_scale_bytes",
        "w13_bias",
        "w2",
        "w2_scale_bytes",
        "w2_bias",
        "output_hidden",
    )
    function = ast.FunctionDef(
        name="gpt_oss_moe_megakernel_source",
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
    filename = str(Path(__file__).with_name("_generated_gpt_oss_moe.py"))
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
        )(namespace["gpt_oss_moe_megakernel_source"]),
        source,
    )


MEGAKERNEL, GENERATED_SOURCE = _build_megakernel()


def _kernel_args(tensors: dict[str, torch.Tensor], shape) -> tuple:
    return (
        tensors["logits"],
        tensors["hidden"],
        tensors["w13"],
        tensors["w13_scale"].view(torch.uint8),
        tensors["w13_bias"],
        tensors["w2"],
        tensors["w2_scale"].view(torch.uint8),
        tensors["w2_bias"],
        shape.output_hidden,
    )


def _family_block_ids(bound) -> list[list[int]]:
    device_ir = bound.host_function.device_ir
    dependency_graph = device_ir.tile_dependency_graph
    assert dependency_graph is not None
    configurable = tuple(spec.block_id for spec in bound.config_spec.block_sizes)
    axes_by_root = [set() for _ in device_ir.root_ids]
    for scope in dependency_graph.execution_scopes:
        axes_by_root[scope.root].update(scope.local_axis_order)
    return [
        [block_id for block_id in configurable if block_id in root_axes]
        for root_axes in axes_by_root
    ]


def _copy_standalone_block_sizes(bound, values: dict, configs: dict) -> None:
    block_size_by_id: dict[int, int] = {}
    for invocation, family in zip(INVOCATIONS, _family_block_ids(bound), strict=True):
        source_sizes = configs[invocation.config_name].get("block_sizes", [])
        if invocation.source_block_indices is not None:
            source_sizes = [
                source_sizes[index] for index in invocation.source_block_indices
            ]
        if len(source_sizes) != len(family):
            raise RuntimeError(
                f"{invocation.prefix} block mapping changed: "
                f"source={source_sizes}, family={family}"
            )
        block_size_by_id.update(zip(family, source_sizes, strict=True))
    values["block_sizes"] = [
        block_size_by_id.get(spec.block_id, value)
        for spec, value in zip(
            bound.config_spec.block_sizes,
            values["block_sizes"],
            strict=True,
        )
    ]


def _megakernel_config(bound, configs: dict, args) -> helion.Config:
    values = dict(bound.config_spec.default_config())
    _copy_standalone_block_sizes(bound, values, configs)
    override_names = {
        "routing": (),
        "gemm1": ("gemm1_block_k",),
        "gemm2": ("gemm2_block", "gemm2_block_k"),
        "finalize": ("finalize_block",),
    }
    override_by_id = {
        block_id: override
        for invocation, family in zip(
            INVOCATIONS,
            _family_block_ids(bound),
            strict=True,
        )
        for block_id, name in zip(
            family,
            override_names[invocation.prefix],
            strict=True,
        )
        if (override := getattr(args, name)) is not None
    }
    values["block_sizes"] = [
        override_by_id.get(spec.block_id, value)
        for spec, value in zip(
            bound.config_spec.block_sizes,
            values["block_sizes"],
            strict=True,
        )
    ]
    values.update(
        {
            "pid_type": "persistent_blocked",
            "num_sm_multiplier": args.worker_multiplier,
            "num_warps": args.num_warps,
            "num_stages": args.kernel_stages,
        }
    )
    if (
        args.static_dispatch
        and "cross_loop_schedule" in bound.config_spec.supported_config_keys()
    ):
        values["cross_loop_schedule"] = "static_pipeline"
    if args.range_stages is not None:
        values["range_num_stages"] = [
            args.range_stages if len(spec.block_ids) == 1 else value
            for spec, value in zip(
                bound.config_spec.range_num_stages,
                values["range_num_stages"],
                strict=True,
            )
        ]
    if args.load_eviction_policy is not None:
        policy = (
            "" if args.load_eviction_policy == "none" else args.load_eviction_policy
        )
        values["load_eviction_policies"] = [
            policy for _ in values["load_eviction_policies"]
        ]
    if args.maxnreg is not None:
        values["maxnreg"] = args.maxnreg
    config = helion.Config.from_dict(values)
    bound.config_spec.normalize(config.config)
    return config


def _compile_named(name: str, kernel, kernel_args, configs: dict):
    bound = kernel.bind(kernel_args)
    config = helion.Config.from_dict(configs[name])
    bound.config_spec.normalize(config.config)
    return bound.compile_config(config)


class SeparateMoE:
    def __init__(self, tensors: dict[str, torch.Tensor], shape, configs: dict) -> None:
        self.tensors = tensors
        self.shape = shape
        self.routing = _compile_named(
            "routing",
            SOURCE.mxfp4_top4_routing,
            (tensors["logits"],),
            configs,
        )
        weights, ids = self.routing(tensors["logits"])
        self.gemm1 = _compile_named(
            "gemm1_swiglu_oai",
            SOURCE.mxfp4_moe_gemm1_swiglu_oai_decode,
            (
                tensors["hidden"],
                tensors["w13"],
                tensors["w13_scale"].view(torch.uint8),
                tensors["w13_bias"],
                ids,
            ),
            configs,
        )
        activation = self.gemm1(
            tensors["hidden"],
            tensors["w13"],
            tensors["w13_scale"].view(torch.uint8),
            tensors["w13_bias"],
            ids,
        )
        self.gemm2 = _compile_named(
            "gemm2",
            SOURCE.mxfp4_moe_gemm2_decode,
            (
                activation,
                tensors["w2"],
                tensors["w2_scale"].view(torch.uint8),
                tensors["w2_bias"],
                ids,
            ),
            configs,
        )
        expert_output = self.gemm2(
            activation,
            tensors["w2"],
            tensors["w2_scale"].view(torch.uint8),
            tensors["w2_bias"],
            ids,
        )
        self.finalize = _compile_named(
            "finalize",
            SOURCE.mxfp4_moe_finalize,
            (expert_output, weights, shape.output_hidden),
            configs,
        )

    def __call__(self) -> tuple:
        tensors = self.tensors
        weights, ids = self.routing(tensors["logits"])
        activation = self.gemm1(
            tensors["hidden"],
            tensors["w13"],
            tensors["w13_scale"].view(torch.uint8),
            tensors["w13_bias"],
            ids,
        )
        expert_output = self.gemm2(
            activation,
            tensors["w2"],
            tensors["w2_scale"].view(torch.uint8),
            tensors["w2_bias"],
            ids,
        )
        output = self.finalize(expert_output, weights, self.shape.output_hidden)
        return output, weights, ids, activation, expert_output


def _validate(actual: tuple, expected: tuple) -> None:
    for name, left, right in zip(OUTPUT_NAMES, actual, expected, strict=True):
        if name == "topk_ids":
            torch.testing.assert_close(left, right)
        else:
            SOURCE._assert_close(name, left, right)


def _helion_resources(compiled_wrapper) -> dict[str, int | None]:
    kernels = []
    for value in compiled_wrapper.__globals__.values():
        device_caches = getattr(value, "device_caches", None)
        if not device_caches or torch.cuda.current_device() not in device_caches:
            continue
        kernels.extend(device_caches[torch.cuda.current_device()][0].values())
    if len(kernels) != 1:
        raise RuntimeError(f"expected one compiled Helion kernel, found {len(kernels)}")
    kernel = kernels[0]
    launch_shared = getattr(
        kernel,
        "_helion_launch_dynamic_shared_bytes",
        kernel.metadata.shared,
    )
    return {
        "registers": int(kernel.n_regs),
        "spills": int(kernel.n_spills),
        "shared": int(launch_shared),
        "triton_required_shared": int(kernel.metadata.shared),
        "resident_blocks_per_sm": getattr(
            kernel,
            "_helion_resident_blocks_per_sm",
            None,
        ),
    }


def _compiler_provenance() -> dict[str, str]:
    def git_output(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=COMPILER_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "root": str(COMPILER_ROOT),
        "helion_module": str(Path(helion.__file__).resolve()),
        "branch": git_output("branch", "--show-current"),
        "commit": git_output("rev-parse", "HEAD"),
    }


def run(args: argparse.Namespace) -> None:
    os.environ.setdefault("MEGAKERNEL_CLEAR_L2", "1")
    if not args.allow_busy:
        require_idle_visible_gpu()
    shape = SOURCE.GptOssMoeShape()
    tensors = SOURCE._allocate(shape)
    reference = SOURCE._reference(tensors)
    expected = (
        reference[4][:, : shape.output_hidden],
        reference[0],
        reference[1],
        reference[2],
        reference[3],
    )
    configs = json.loads(args.config_path.read_text())
    kernel_args = _kernel_args(tensors, shape)
    bound = MEGAKERNEL.bind(kernel_args)
    if args.static_dispatch:
        bound.config_spec.automatic_clc_dispatch = False
    config = _megakernel_config(bound, configs, args)

    lowering_started = time.perf_counter()
    lowered = bound.to_triton_code(config, output_origin_lines=True)
    lowering_seconds = time.perf_counter() - lowering_started
    args.lowered_output.parent.mkdir(parents=True, exist_ok=True)
    args.lowered_output.write_text(lowered)
    print("LOWERED_TRITON", args.lowered_output.resolve(), flush=True)
    print("MEGAKERNEL_CONFIG", json.dumps(dict(config), sort_keys=True), flush=True)
    print("ROOT_BLOCK_IDS", bound.host_function.device_ir.grid_block_ids, flush=True)
    if args.print_source:
        print(GENERATED_SOURCE)
    if args.print_lowered:
        print(lowered)
    if args.inspect_only:
        return

    compile_started = time.perf_counter()
    megakernel = bound.compile_config(config)
    megakernel_compile_seconds = time.perf_counter() - compile_started
    separate_started = time.perf_counter()
    separate = SeparateMoE(tensors, shape, configs)
    separate_compile_seconds = time.perf_counter() - separate_started

    actual = megakernel(*kernel_args)
    separate_output = separate()
    torch.cuda.synchronize()
    _validate(actual, expected)
    _validate(separate_output, expected)
    resources = _helion_resources(megakernel)
    print("MEGAKERNEL_RESOURCES", json.dumps(resources, sort_keys=True), flush=True)
    if not args.benchmark:
        return

    megakernel_graph, megakernel_output = capture(lambda: megakernel(*kernel_args))
    separate_graph, separate_graph_output = capture(separate)
    megakernel_graph.replay()
    separate_graph.replay()
    torch.cuda.synchronize()
    _validate(megakernel_output, expected)
    _validate(separate_graph_output, expected)
    pids = visible_gpu_pids()
    timings = benchmark_interleaved(
        {
            "helion_gpt_oss_moe_megakernel": megakernel_graph.replay,
            "helion_gpt_oss_moe_separate": separate_graph.replay,
        },
        args.repeats,
        args.batch_replays,
    )
    if visible_gpu_pids() != pids:
        raise RuntimeError("GPU process set changed during benchmark")
    result = {
        "benchmark_mode": (
            "cold_l2" if os.environ.get("MEGAKERNEL_CLEAR_L2") == "1" else "warm_l2"
        ),
        "compiler": _compiler_provenance(),
        "device": torch.cuda.get_device_name(),
        "shape": dataclasses.asdict(shape),
        "compile_seconds": {
            "lowering": lowering_seconds,
            "megakernel": megakernel_compile_seconds,
            "separate": separate_compile_seconds,
        },
        "megakernel_config": dict(config),
        "resources": resources,
        "timings": timings,
    }
    args.result_output.parent.mkdir(parents=True, exist_ok=True)
    args.result_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("RESULT_JSON", json.dumps(result, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-root", type=Path, default=COMPILER_ROOT)
    parser.add_argument("--config-path", type=Path, default=CONFIG_PATH)
    parser.add_argument("--worker-multiplier", type=int, default=8)
    parser.add_argument("--num-warps", type=int, default=1)
    parser.add_argument("--kernel-stages", type=int, default=1)
    parser.add_argument("--range-stages", type=int)
    parser.add_argument(
        "--load-eviction-policy",
        choices=("none", "first", "last"),
        default="none",
    )
    parser.add_argument("--maxnreg", type=int, default=256)
    parser.add_argument("--gemm1-block-k", type=int)
    parser.add_argument("--gemm2-block", type=int, default=16)
    parser.add_argument("--gemm2-block-k", type=int)
    parser.add_argument("--finalize-block", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--batch-replays", type=int, default=1)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--print-source", action="store_true")
    parser.add_argument("--print-lowered", action="store_true")
    parser.add_argument("--static-dispatch", action="store_true")
    parser.add_argument(
        "--lowered-output",
        type=Path,
        default=Path("/tmp/gpt_oss_moe_megakernel_lowered.py"),
    )
    parser.add_argument(
        "--result-output",
        type=Path,
        default=Path("/tmp/gpt_oss_moe_megakernel_result.json"),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
