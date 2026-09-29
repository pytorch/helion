"""Typed admission for a narrow warp schedule of a materialized byte operand.

The original materialize_operand proof establishes legal extraction and exact
interleave order. This module rechecks the resulting producer, contraction,
fresh allocation, and guarded layouts before any launch is replaced. Recipes
are replayed from the staged byte with their original integer casts/operations.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import operator
from typing import TYPE_CHECKING
from typing import cast

import torch

from ...language import memory_ops
from ...language import tile_ops
from ..device_ir import RootGraphInfo
from ..indexing_strategy import subscript_tile_info
from .epilogue_fanout import _fresh_returned_tensors
from .materialize_operand import _prove_recipe
from .materialized_fission import _exact_tile
from .materialized_fission import _Ineligible
from .memory_ops import _PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY
from .memory_ops import tensor_has_specialized_base_alignment
from .promote_output_axis import _access_tensor
from .promote_output_axis import _fresh_tensors
from .promote_output_axis import _ordinary_operation
from .row_resident import _direct_load
from .row_resident import _stage
from .scalar_recipe import ScalarRecipe
from .scalar_recipe import build_recipe
from .signed_bitfield import _ANDS
from .signed_bitfield import _GES
from .signed_bitfield import _SHIFTS
from .signed_bitfield import _SUBS
from .signed_bitfield import _binary
from .signed_bitfield import _cast_input
from .signed_bitfield import _dtype
from .signed_bitfield import match_signed_byte_field

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..host_function import HostFunction


KEY = "cute_materialized_operand_schedule"
MODES = ("off", "warp_narrow4")
REDUCTIONS = (64, 128, 256)
_TYPES = {
    torch.int8: "Int8",
    torch.int16: "Int16",
    torch.int32: "Int32",
    torch.int64: "Int64",
    torch.bool: "Boolean",
    torch.bfloat16: "BFloat16",
}


@dataclass(frozen=True)
class PackedOperandPlan:
    arguments: tuple[str, str, str]
    materialized_name: str
    rows: int
    columns: int
    reduction: int
    recipes: tuple[ScalarRecipe, ScalarRecipe]

    @property
    def a_bytes(self) -> int:
        return 32 * self.reduction * 2

    @property
    def decoded_bytes(self) -> int:
        return self.reduction * 4 * 2

    @property
    def shared_bytes(self) -> int:
        return self.a_bytes + self.decoded_bytes + self.reduction // 2 * 4


def scalar_byte_recipe(value: torch.fx.Node) -> ScalarRecipe | None:
    """Replay a proved integer field without rewriting its sign extension.

    Field analysis proves the operation ranges and a single signed byte load.
    ScalarRecipe then proves reaching definitions and the pure scalar language.
    Every conversion, branch, and typed binary operation remains explicit.
    """
    field = match_signed_byte_field(value)
    if field is None:
        return None
    statements: list[ast.stmt] = []
    names: dict[torch.fx.Node, str] = {field.load: "packed_value"}

    def typed(dtype: torch.dtype, expression: str) -> str:
        return f"cutlass.{_TYPES[dtype]}({expression})"

    def lower(node: torch.fx.Node) -> str:
        if node in names:
            return names[node]
        dtype = _dtype(node)
        if node not in field.nodes or dtype not in _TYPES:
            raise _Ineligible("unsupported byte recipe type or dependency")
        source = _cast_input(node)
        if source is not None:
            expression = typed(dtype, lower(source))
        elif node.target is torch.ops.aten.where.self:
            condition, negative, raw = cast(
                "tuple[torch.fx.Node, torch.fx.Node, torch.fx.Node]", node.args
            )
            expression = (
                f"{typed(dtype, lower(negative))} if {lower(condition)} "
                f"else {typed(dtype, lower(raw))}"
            )
        else:
            expression = None
            for targets, operation in (
                (_ANDS, "and_"),
                (_SHIFTS, "rshift"),
                (_SUBS, "sub"),
                (_GES, "ge"),
            ):
                matched = _binary(node, targets)
                if matched is not None:
                    operand, constant = matched
                    operand_dtype = _dtype(operand)
                    if operand_dtype not in _TYPES:
                        raise _Ineligible("unsupported scalar operand dtype")
                    expression = typed(
                        dtype,
                        f"operator.{operation}({lower(operand)}, "
                        f"{typed(operand_dtype, repr(constant))})",
                    )
                    break
            if expression is None:
                raise _Ineligible("unsupported scalar byte operation")
        name = f"byte_recipe_{len(statements)}"
        statements.append(ast.parse(f"{name} = {expression}").body[0])
        names[node] = name
        return name

    try:
        result = ast.parse(lower(value), mode="eval").body
    except _Ineligible:
        return None
    return build_recipe(result, statements, {"packed_value"})


def _affine_row(
    value: object, axis: int, env: CompileEnvironment
) -> tuple[int, int] | None:
    """Prove the producer's actual row address is factor * tile + residue."""
    if not isinstance(value, torch.fx.Node):
        return None
    if value.op != "call_function":
        return None
    if value.target in (operator.add, torch.ops.aten.add.Tensor):
        if len(value.args) != 2 or value.kwargs not in ({}, {"alpha": 1}):
            return None
        base, offset = value.args
        if type(offset) is not int:
            return None
        inner = _affine_row(base, axis, env)
        return None if inner is None else (inner[0], inner[1] + offset)
    if value.target in (operator.mul, torch.ops.aten.mul.Tensor):
        if len(value.args) != 2 or value.kwargs:
            return None
        base, factor = value.args
        if type(factor) is not int or factor <= 0:
            return None
        inner = _affine_row(base, axis, env)
        return None if inner is None else (inner[0] * factor, inner[1] * factor)
    if value.target is torch.ops.prims.iota.default:
        if len(value.args) != 1 or set(value.kwargs) - {
            "start",
            "step",
            "dtype",
            "device",
            "requires_grad",
        }:
            return None
        start = value.kwargs.get("start", 0)
        length = value.args[0]
        if not (
            value.kwargs.get("step", 1) == 1
            and isinstance(start, torch.fx.Node)
            and start.target is tile_ops.tile_begin
            and len(start.args) == 1
            and not start.kwargs
            and _exact_tile(start.args[0], axis, env)
        ):
            return None
        length_value = (
            length.meta.get("val") if isinstance(length, torch.fx.Node) else length
        )
        if not isinstance(length_value, (int, torch.SymInt)) or not env.known_equal(
            length_value, env.block_sizes[axis].var
        ):
            return None
        return (1, 0)
    info = subscript_tile_info(env, value)
    if info is not None and info.block_id == axis and info.block_size is None:
        if env.known_equal(info.offset, 0):
            return (1, 0)
    return None


def _aligned(
    env: CompileEnvironment, name: str, tensor: torch.Tensor, width: int, binding: bool
) -> bool:
    if not binding:
        return tensor_has_specialized_base_alignment(env, tensor, width)
    value = env.runtime_arg_values_by_name.get(name)
    source = env.tensor_input_source(tensor)
    specialization = env.runtime_input_specializations.get(
        _PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY
    )
    return (
        isinstance(value, torch.Tensor)
        and value.data_ptr() % width == 0
        and source is not None
        and specialization is not None
        and source in specialization.sources
    )


def prove_packed_operand(
    env: CompileEnvironment, host: HostFunction, *, binding: bool = False
) -> PackedOperandPlan | None:
    fission = env.cute_fission_plan
    ir = host.device_ir
    if (
        env.backend_name != "cute"
        or fission is None
        or fission.operand_interleave_factor != 2
        or fission.region_count != 2
        or fission.pointwise_region_indices != (0,)
        or len(ir.root_ids) != 2
        or len(ir.graphs) != 3
        or ir.noncanonical_task_origin_block_ids
        or env._is_distributed
        or env.settings.fast_math
        or env.config_spec.target_device_capability is None
        or env.config_spec.target_device_capability[0] < 8
    ):
        return None
    returned = _fresh_returned_tensors(host)
    stage = _stage(env, host, 1, returned, torch.bfloat16)
    if stage is None or len(stage.stores) != 1 or stage.stores[0].chain.steps:
        return None
    output = stage.stores[0]
    if returned != frozenset((output.tensor,)):
        return None
    fresh = _fresh_tensors(host)
    if (
        fresh != {stage.rhs, output.tensor}
        or stage.lhs_name not in host.params.arguments
    ):
        return None
    if host.params.arguments[stage.lhs_name] is not stage.lhs:
        return None
    m, k, n = int(stage.lhs.shape[0]), int(stage.lhs.shape[1]), int(stage.rhs.shape[1])
    if (
        m % 32
        or n % 4
        or n // 4 > 65535
        or k not in REDUCTIONS
        or max(m * k, k // 2 * n, m * n) >= 2**31
    ):
        return None
    root = ir.graphs[ir.root_ids[0]]
    axes = ir.grid_block_ids[0]
    if not isinstance(root, RootGraphInfo) or len(axes) != 2:
        return None
    stores = [node for node in root.graph.nodes if node.target is memory_ops.store]
    if len(stores) != 2:
        return None
    values: dict[int, torch.fx.Node] = {}
    for store in stores:
        if (
            len(store.args) != 4
            or store.kwargs
            or store.args[3] is not None
            or _access_tensor(store) is not stage.rhs
        ):
            return None
        indices, value = store.args[1:3]
        if (
            not isinstance(indices, (tuple, list))
            or len(indices) != 2
            or not _exact_tile(indices[1], axes[1], env)
        ):
            return None
        row = _affine_row(indices[0], axes[0], env)
        if (
            row is None
            or row not in ((2, 0), (2, 1))
            or row[1] in values
            or not isinstance(value, torch.fx.Node)
        ):
            return None
        values[row[1]] = value
    try:
        recipe_nodes = _prove_recipe(
            (values[0], values[1]),
            k=axes[0],
            n=axes[1],
            inputs={
                value
                for value in host.params.arguments.values()
                if isinstance(value, torch.Tensor)
            },
            env=env,
        )
    except _Ineligible:
        return None
    fields = [match_signed_byte_field(values[i]) for i in (0, 1)]
    if any(field is None for field in fields):
        return None
    first, second = fields
    assert first is not None and second is not None
    if first.load is not second.load:
        return None
    loaded = _direct_load(env, first.load, (axes[0], axes[1]), torch.int8)
    if loaded is None:
        return None
    b_name, b = loaded
    if (
        b_name not in host.params.arguments
        or host.params.arguments[b_name] is not b
        or tuple(b.shape) != (k // 2, n)
    ):
        return None
    if any(
        not _ordinary_operation(node)
        or node.target is memory_ops.load
        and node not in recipe_nodes
        or node.target is memory_ops.store
        and node not in stores
        for node in root.graph.nodes
    ):
        return None
    # The stage proof excludes consumer effects, and the original extraction
    # excludes opaque host operations. Fresh output/temp storages cannot alias
    # either read-only input on this or a later public call.
    if not _aligned(env, stage.lhs_name, stage.lhs, 16, binding) or not _aligned(
        env, b_name, b, 4, binding
    ):
        return None
    recipes = tuple(scalar_byte_recipe(values[i]) for i in (0, 1))
    if any(recipe is None for recipe in recipes):
        return None
    lo, hi = recipes
    assert lo is not None and hi is not None
    return PackedOperandPlan(
        (stage.lhs_name, b_name, output.name), stage.rhs_name, m, n, k, (lo, hi)
    )
