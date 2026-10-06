"""Cache-safe metadata and late lowering for direct gathered contractions."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import sympy
import torch

from ... import exc
from ..compile_environment import CompileEnvironment
from ..device_function import ConstExprArg
from ..device_function import TensorArg
from ..device_function import TensorSizeArg
from ..device_function import _exact_thread_block_dims
from ..host_function import HostFunction
from .gathered_mma import GatherTensorFacts
from .gathered_mma import analyze_gathered_mma_region
from .gathered_mma_codegen import emit_gathered_mma_region
from .memory_ops import tensor_has_specialized_base_alignment
from .promote_output_axis import _fresh_tensors

if TYPE_CHECKING:
    from ..device_function import DeviceFunction


def gathered_tensor_facts(
    df: DeviceFunction,
) -> tuple[dict[str, GatherTensorFacts], dict[str, int]] | None:
    """Read exact dimensions guarded by the bound-kernel metadata cache key.

    Runtime weak references need not remain alive. Integer hints are used only
    for input dimensions/strides whose exact values are dispatch facts; fresh
    output metadata must be an expression of those same guarded dimensions.
    Dynamic scalar arguments and device metadata contents are never constants.
    """
    env = CompileEnvironment.current()
    if "input_tensor_metadata" not in env.compiler_fact_specialization_facts:
        return None
    tensor_args = [arg for arg in df.arguments if isinstance(arg, TensorArg)]
    guarded: dict[sympy.Expr, sympy.Integer] = {}

    def expression(value: int | torch.SymInt) -> sympy.Expr:
        return (
            env.shape_env.replace(value._sympy_())
            if isinstance(value, torch.SymInt)
            else sympy.Integer(value)
        )

    for arg in tensor_args:
        if env.tensor_input_source(arg.fake_value) is None:
            continue
        for value in (*arg.fake_value.shape, *arg.fake_value.stride()):
            key = expression(value)
            guarded[key] = sympy.Integer(env.size_hint(value))

    def exact(value: int | torch.SymInt | sympy.Expr) -> int | None:
        key = value if isinstance(value, sympy.Expr) else expression(value)
        result = env.shape_env.replace(key).xreplace(guarded)
        return int(result) if isinstance(result, sympy.Integer) else None

    fresh = _fresh_tensors(HostFunction.current())
    tensors: dict[str, GatherTensorFacts] = {}
    for arg in tensor_args:
        tensor = arg.fake_value
        if tensor.dtype not in (torch.float16, torch.bfloat16, torch.int32):
            continue
        shape = tuple(exact(size) for size in tensor.shape)
        strides = tuple(exact(stride) for stride in tensor.stride())
        if any(value is None for value in (*shape, *strides)):
            continue
        alignment = tensor.element_size()
        if env.tensor_input_source(tensor) is not None:
            if tensor_has_specialized_base_alignment(env, tensor, 16):
                alignment = 16
        elif tensor in fresh and exact(tensor.storage_offset()) == 0:
            alignment = 16
        else:
            continue
        tensors[arg.name] = GatherTensorFacts(
            env.backend.dtype_str(tensor.dtype),
            tuple(value for value in shape if value is not None),
            tuple(value for value in strides if value is not None),
            alignment,
        )
    constants = {}
    for arg in df.arguments:
        if isinstance(arg, ConstExprArg):
            parsed = ast.parse(arg.host_str(), mode="eval").body
            if isinstance(parsed, ast.Constant) and type(parsed.value) is int:
                constants[arg.name] = parsed.value
        elif isinstance(arg, TensorSizeArg):
            value = exact(arg.tensor_arg.fake_value.shape[arg.dim])
            if value is not None:
                constants[arg.name] = value
    for key, arg in df._expr_args.items():
        value = exact(key)
        if value is not None:
            constants[arg.name] = value
    return tensors, constants


def lower_gathered_matmul(
    body: list[ast.stmt],
    df: DeviceFunction,
    *,
    boundary_names: set[str],
    disjoint_pairs: set[frozenset[str]],
) -> list[ast.stmt]:
    """Lower only an explicitly selected and fully proved pipeline region."""
    env = CompileEnvironment.current()
    sites = df.cute_state.collective_mma_sites
    facts = gathered_tensor_facts(df)
    thread_dims = _exact_thread_block_dims(df.tile_strategy)
    if (
        env.config_spec.target_device_capability is None
        or env.config_spec.target_device_capability[0] != 10
        or len(sites) != 1
        or facts is None
        or thread_dims is None
        or df.codegen.cute_wrapper_plans
        or df.cute_state.matmul_plan is not None
        or df.cute_state.matmul_fx_nodes
        or df.cute_state.has_tcgen05_fragment_epilogue_plan
    ):
        raise exc.BackendUnsupported(
            "cute", "gathered MMA requires one isolated SM100 region"
        )
    tensors, constants = facts
    region = analyze_gathered_mma_region(
        body,
        sites[0],
        boundary_names=boundary_names,
        tensors=tensors,
        constants=constants,
        thread_dims=thread_dims,
        disjoint_pairs=disjoint_pairs,
    )
    bn = df.config.get("cute_gathered_mma_n", 256)
    stages = df.config.get("cute_gathered_mma_stages", 3)
    if (
        region is None
        or type(bn) is not int
        or bn not in (128, 256, 512)
        or type(stages) is not int
        or stages not in (2, 3, 4)
        or (128 + bn) * 64 * 2 * stages + 18432 > 232448
    ):
        raise exc.BackendUnsupported(
            "cute", "unproved gathered MMA operands, effects, or geometry"
        )
    emit_gathered_mma_region(
        region,
        sites[0],
        df,
        bn=bn,
        stages=stages,
        constants=constants,
        thread_dims=thread_dims,
    )
    return body
