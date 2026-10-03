"""Select native block scaling only after proving the complete device region."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import math
import operator
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch

from ... import exc
from ...language import _tracing_ops
from ...language import creation_ops
from ...language import memory_ops
from ...language.matmul_ops import dot
from ...language.quantized_ops import float4_e2m1fn_x2_to_float32
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..device_function import DeviceFunction
from ..device_ir import ForLoopGraphInfo
from ..host_function import HostFunction
from ..indexing_strategy import subscript_tile_info
from ..variable_origin import NameOrigin
from .block_scaled_config import block_scaled_workspace_shape
from .block_scaled_provenance import BLOCK_SCALED_PROVENANCE
from .block_scaled_provenance import BlockScaledProvenance
from .block_scaled_provenance import IntegerRecipe
from .block_scaled_provenance import PackedOperand
from .block_scaled_provenance import operand_fingerprint
from .fold_noop_stores import _is_read_only
from .promote_output_axis import _fresh_tensors

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from torch.fx import Node

    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST


@dataclass(frozen=True)
class IntegerCode:
    expression: str
    low: int
    high: int


@dataclass(frozen=True)
class PreparedOperand:
    source: PackedOperand
    byte_addresses: tuple[str, ...]
    scale_address: str
    tensor_span: int
    scale_span: int
    row_fast: bool


@dataclass(frozen=True)
class BlockScaledMmaPlan:
    root_graph_id: int
    lhs: PreparedOperand
    rhs: PreparedOperand
    output: torch.Tensor
    alpha: float | torch.SymFloat
    round_dtypes: tuple[torch.dtype, ...]
    m: int
    n: int
    k: int
    bn: int
    bk: int
    stages: int
    cluster_m: int
    persistent: bool
    num_sm: int

    @property
    def workspace_bytes(self) -> int:
        padded_m, padded_n, padded_k = block_scaled_workspace_shape(
            self.m, self.n, self.k, self.bn, self.bk
        )
        padded_rows = padded_m + padded_n
        return padded_rows * padded_k // 2 + padded_rows * padded_k // 16


def _metadata_resolver(env: CompileEnvironment) -> Callable[[object], int | None]:
    substitutions: dict[sympy.Expr, sympy.Integer] = {}
    for tensor in env.input_sources:
        for value in (*tensor.shape, *tensor.stride()):
            if isinstance(value, torch.SymInt):
                substitutions[env.shape_env.replace(value._sympy_())] = sympy.Integer(
                    env.size_hint(value)
                )

    def resolve(value: object) -> int | None:
        if isinstance(value, torch.fx.Node):
            value = value.meta.get("val")
        if type(value) is int:
            return value
        if isinstance(value, torch.SymInt):
            value = value._sympy_()
        if isinstance(value, sympy.Expr):
            value = env.shape_env.replace(value).xreplace(substitutions)
            if isinstance(value, sympy.Integer):
                return int(value)
        return None

    return resolve


def render_integer_recipe(
    recipe: IntegerRecipe,
    *,
    coordinates: dict[int, tuple[str, int]],
    resolve: Callable[[object], int | None],
    budget: int = 4096,
) -> IntegerCode | None:
    """Preserve integer widths and decline overflow or excessive expansion."""
    remaining = budget

    def visit(node: IntegerRecipe, depth: int = 0) -> IntegerCode | None:
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 128 or node.bits not in (32, 64):
            return None
        if node.operation == "constant":
            value = resolve(node.value)
            result = (
                IntegerCode(str(value), value, value) if value is not None else None
            )
        elif node.operation == "coordinate":
            coordinate = coordinates.get(cast("int", node.value))
            if coordinate is None:
                return None
            name, extent = coordinate
            result = IntegerCode(f"cutlass.Int{node.bits}({name})", 0, extent - 1)
        else:
            args = [visit(operand, depth + 1) for operand in node.operands]
            if any(arg is None for arg in args):
                return None
            operands = [arg for arg in args if arg is not None]
            if node.operation == "cast" and len(operands) == 1:
                child = operands[0]
                result = IntegerCode(child.expression, child.low, child.high)
            elif len(operands) == 2:
                left, right = operands
                if node.operation == "add":
                    low, high, symbol = (
                        left.low + right.low,
                        left.high + right.high,
                        "+",
                    )
                elif node.operation == "sub":
                    low, high, symbol = (
                        left.low - right.high,
                        left.high - right.low,
                        "-",
                    )
                elif node.operation == "mul":
                    products = [
                        x * y
                        for x in (left.low, left.high)
                        for y in (right.low, right.high)
                    ]
                    low, high, symbol = min(products), max(products), "*"
                elif (
                    node.operation in ("floordiv", "mod")
                    and left.low >= 0
                    and right.low == right.high
                    and right.low > 0
                ):
                    if node.operation == "floordiv":
                        low, high, symbol = (
                            left.low // right.low,
                            left.high // right.low,
                            "//",
                        )
                    elif left.low // right.low == left.high // right.low:
                        low, high, symbol = (
                            left.low % right.low,
                            left.high % right.low,
                            "%",
                        )
                    else:
                        low, high, symbol = 0, right.low - 1, "%"
                else:
                    return None
                result = IntegerCode(
                    f"({left.expression} {symbol} {right.expression})", low, high
                )
            else:
                return None
            result = IntegerCode(
                f"cutlass.Int{node.bits}({result.expression})", result.low, result.high
            )
        if (
            result is None
            or result.low < -(1 << (node.bits - 1))
            or result.high >= (1 << (node.bits - 1))
        ):
            return None
        return result

    return visit(recipe)


def _prepare_operand(
    source: PackedOperand,
    row_count: int,
    group_count: int,
    group_axis: int,
    resolve: Callable[[object], int | None],
) -> PreparedOperand | None:
    coordinates = {
        source.row_axis: ("row", row_count),
        group_axis: ("group", group_count),
    }

    def address(
        tensor: torch.Tensor, indices: tuple[IntegerRecipe, ...]
    ) -> tuple[str, int] | None:
        shape = tuple(resolve(value) for value in tensor.shape)
        strides = tuple(resolve(value) for value in tensor.stride())
        if any(value is None or value <= 0 for value in shape) or any(
            value is None or value < 0 for value in strides
        ):
            return None
        sizes = [value for value in shape if value is not None]
        steps = [value for value in strides if value is not None]
        if len(indices) != len(sizes):
            return None
        terms = []
        for index, size, stride in zip(indices, sizes, steps, strict=True):
            code = render_integer_recipe(
                index, coordinates=coordinates, resolve=resolve
            )
            if code is None or code.low < 0 or code.high >= size:
                return None
            terms.append(f"cutlass.Int64({code.expression}) * {stride}")
        span = 1 + sum(
            (size - 1) * stride for size, stride in zip(sizes, steps, strict=True)
        )
        if span >= 1 << 63:
            return None
        return "(" + " + ".join(terms) + ")", span

    addresses = [address(source.tensor, indices) for indices in source.byte_indices]
    scale = address(source.scale, source.scale_indices)
    if scale is None or any(item is None for item in addresses):
        return None
    checked = [item for item in addresses if item is not None]

    def uses_row(recipe: IntegerRecipe) -> bool:
        return (
            recipe.operation == "coordinate" and recipe.value == source.row_axis
        ) or any(uses_row(child) for child in recipe.operands)

    # Coalesce the original contiguous axis without changing any address. In
    # particular this covers both K-major and transposed packed sources.
    row_fast = all(
        sum(uses_row(index) for index in indices) == 1
        and any(
            index.operation == "coordinate"
            and index.value == source.row_axis
            and resolve(stride) == 1
            for index, stride in zip(indices, source.tensor.stride(), strict=True)
        )
        for indices in source.byte_indices
    )
    return PreparedOperand(
        source,
        tuple(item[0] for item in checked),
        scale[0],
        checked[0][1],
        scale[1],
        row_fast,
    )


def has_block_scaled_provenance(graphs: Sequence[GraphInfo]) -> bool:
    return any(
        BLOCK_SCALED_PROVENANCE in node.meta
        for info in graphs
        for node in info.graph.nodes
    )


def _nonoverlapping_output(
    output: torch.Tensor, resolve: Callable[[object], int | None]
) -> bool:
    dimensions = []
    for size, stride in zip(output.shape, output.stride(), strict=True):
        extent, step = resolve(size), resolve(stride)
        if extent is None or step is None or extent <= 0 or step < 0:
            return False
        if extent > 1:
            dimensions.append((step, extent))
    span = 1
    for step, extent in sorted(dimensions):
        if step < span:
            return False
        span += (extent - 1) * step
    return span < 1 << 63


def _epilogue(
    store: Node, loop: Node, seed: Node
) -> tuple[float | torch.SymFloat, tuple[torch.dtype, ...], set[Node]] | None:
    value = store.args[2]
    seen = {store}
    if not isinstance(value, torch.fx.Node):
        return None
    if not isinstance(value.meta.get("val"), torch.Tensor):
        return None
    round_dtypes: list[torch.dtype] = []
    while (
        isinstance(value, torch.fx.Node)
        and value.target is torch.ops.prims.convert_element_type.default
    ):
        if (
            len(value.args) != 2
            or value.kwargs
            or value.args[1] not in (torch.float16, torch.bfloat16, torch.float32)
        ):
            return None
        round_dtypes.append(cast("torch.dtype", value.args[1]))
        seen.add(value)
        value = value.args[0]
    alpha: float | torch.SymFloat = 1.0
    if isinstance(value, torch.fx.Node) and value.target is torch.ops.aten.mul.Tensor:
        if (
            len(value.args) != 2
            or value.kwargs
            or value.meta["val"].dtype != torch.float32
        ):
            return None
        seen.add(value)
        left, right = value.args
        for accumulator, scalar in ((left, right), (right, left)):
            scalar_value = (
                scalar.meta.get("val") if isinstance(scalar, torch.fx.Node) else scalar
            )
            if isinstance(scalar_value, (float, torch.SymFloat)):
                if isinstance(scalar, torch.fx.Node):
                    if scalar.target is not _tracing_ops._get_symnode:
                        return None
                    seen.add(scalar)
                alpha = scalar_value
                value = accumulator
                break
        else:
            return None
    if not isinstance(value, torch.fx.Node) or value.meta["val"].dtype != torch.float32:
        return None
    if value.target is _tracing_ops._phi and value.args[0] is seed:
        seen.add(value)
        value = value.args[1]
    if (
        not isinstance(value, torch.fx.Node)
        or value.target is not operator.getitem
        or value.args != (loop, 0)
        or value.kwargs
    ):
        return None
    seen.add(value)
    return alpha, tuple(reversed(round_dtypes)), seen


def _plan_block_scaled(graphs: Sequence[GraphInfo]) -> BlockScaledMmaPlan | None:
    env = CompileEnvironment.current()
    df = DeviceFunction.current()
    host = HostFunction.current()
    device_ir = host.device_ir
    if (
        env.backend_name != "cute"
        or env.config_spec.target_device_capability is None
        or env.config_spec.target_device_capability[0] != 10
        or "input_tensor_metadata" not in env.compiler_fact_specialization_facts
        or len(device_ir.root_ids) != 1
        or len(device_ir.phases) != 1
        or df.config.pid_type != "flat"
    ):
        return None
    candidates = [
        (info, node)
        for info in graphs
        for node in info.graph.nodes
        if BLOCK_SCALED_PROVENANCE in node.meta
    ]
    if len(candidates) != 1:
        return None
    info, contraction = candidates[0]
    provenance = contraction.meta[BLOCK_SCALED_PROVENANCE]
    if (
        not isinstance(info, ForLoopGraphInfo)
        or not isinstance(provenance, BlockScaledProvenance)
        or info.graph_id != provenance.loop_graph_id
        or contraction.target is not dot
    ):
        return None
    carries = list(info.graph.find_nodes(op="placeholder"))
    exits = list(info.graph.find_nodes(op="output"))
    if (
        len(carries) != 1
        or len(exits) != 1
        or exits[0].args != ([contraction],)
        or len(contraction.args) != 4
        or contraction.args[3] != torch.float32
        or contraction.kwargs
        or not all(isinstance(value, torch.fx.Node) for value in contraction.args[:2])
        or operand_fingerprint(cast("tuple[Node, Node]", contraction.args[:2]))
        != provenance.operand_fingerprint
    ):
        return None
    carry = contraction.args[2]
    if (
        isinstance(carry, torch.fx.Node)
        and carry.target is _tracing_ops._new_var
        and len(carry.args) == 1
        and not carry.kwargs
    ):
        carry = carry.args[0]
    if carry is not carries[0]:
        return None
    if any(
        node.op != "output"
        and node.target not in (dot, float4_e2m1fn_x2_to_float32, _tracing_ops._new_var)
        and not _is_read_only(node)
        for node in info.graph.nodes
    ):
        return None
    root = graphs[device_ir.root_ids[0]]
    root_outputs = list(root.graph.find_nodes(op="output"))
    if len(root_outputs) != 1 or root_outputs[0].args != (None,):
        return None
    loops = [node for node in root.graph.nodes if node.target is _tracing_ops._for_loop]
    stores = [node for node in root.graph.nodes if node.target is memory_ops.store]
    if len(loops) != 1 or len(stores) != 1:
        return None
    loop, store = loops[0], stores[0]
    if (
        len(loop.args) != 4
        or loop.args[0] != info.graph_id
        or loop.args[1] != [0]
        or loop.kwargs
        or len(loop.args[2]) != 1
        or len(loop.args[3]) != 1
    ):
        return None
    seed = loop.args[3][0]
    if (
        not isinstance(seed, torch.fx.Node)
        or seed.target is not creation_ops.full
        or seed.meta["val"].dtype != torch.float32
    ):
        return None
    initial = seed.args[1]
    if (
        not isinstance(initial, (int, float))
        or initial != 0
        or math.copysign(1, initial) < 0
    ):
        return None
    if len(store.args) != 4 or store.args[3] is not None or store.kwargs:
        return None
    output = (
        store.args[0].meta.get("val")
        if isinstance(store.args[0], torch.fx.Node)
        else None
    )
    if (
        not isinstance(output, torch.Tensor)
        or output.ndim != 2
        or output.dtype not in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return None
    result = _epilogue(store, loop, seed)
    if result is None:
        return None
    alpha, round_dtypes, epilogue_nodes = result
    allowed = {*epilogue_nodes, loop, seed, store.args[0]}
    pending = list(allowed)
    while pending:
        node = pending.pop()
        for argument in node.all_input_nodes:
            if argument not in allowed:
                allowed.add(argument)
                pending.append(argument)
    if any(node.op != "output" and node not in allowed for node in root.graph.nodes):
        return None
    row_axes = (provenance.lhs.row_axis, provenance.rhs.row_axis)
    if tuple(device_ir.grid_block_ids[0]) != row_axes or len(store.args[1]) != 2:
        return None
    for index, axis in zip(store.args[1], row_axes, strict=True):
        tile = subscript_tile_info(env, index)
        if (
            tile is None
            or env.canonical_block_id(tile.block_id) != axis
            or not env.known_equal(tile.offset, 0)
            or tile.block_size is not None
        ):
            return None
    resolve = _metadata_resolver(env)
    if not _nonoverlapping_output(output, resolve):
        return None
    extents = [
        resolve(env.block_sizes[axis].numel)
        for axis in (*row_axes, provenance.group_axis)
    ]
    if any(value is None or value <= 0 or value >= 1 << 31 for value in extents):
        return None
    m, n, groups = [value for value in extents if value is not None]
    if resolve(loop.args[2][0]) != groups or tuple(
        resolve(size) for size in output.shape
    ) != (m, n):
        return None
    refs = (
        provenance.lhs.tensor,
        provenance.lhs.scale,
        provenance.rhs.tensor,
        provenance.rhs.scale,
        output,
    )
    fresh = _fresh_tensors(host)
    if any(
        env.tensor_input_source(tensor) is None and tensor not in fresh
        for tensor in refs
    ):
        return None
    names = [df.tensor_arg(tensor).name for tensor in refs]
    disjoint = df.proven_disjoint_tensor_pairs()
    if any(frozenset((name, names[-1])) not in disjoint for name in names[:-1]):
        return None
    lhs = _prepare_operand(provenance.lhs, m, groups, provenance.group_axis, resolve)
    rhs = _prepare_operand(provenance.rhs, n, groups, provenance.group_axis, resolve)
    if lhs is None or rhs is None:
        return None
    bn = df.config.get("cute_scaled_tile_n", 256)
    bk = df.config.get("cute_scaled_tile_k", 256)
    stages = df.config.get("cute_scaled_stages", 4)
    cluster_m = df.config.get("cute_scaled_cluster_m", 2)
    persistent = df.config.get("cute_scaled_persistent", True)
    if (
        type(bn) is not int
        or bn not in (128, 256)
        or type(bk) is not int
        or bk not in (64, 128, 256)
        or type(stages) is not int
        or stages not in (2, 3, 4, 5)
        or type(cluster_m) is not int
        or cluster_m not in (1, 2)
        or type(persistent) is not bool
    ):
        return None
    if ((m + 128 * cluster_m - 1) // (128 * cluster_m)) * (
        (n + bn - 1) // bn
    ) * cluster_m >= 1 << 31:
        return None
    padded_m, padded_n, padded_k = block_scaled_workspace_shape(
        m, n, groups * 16, bn, bk
    )
    padded_rows = padded_m + padded_n
    if (
        (m + 128 * cluster_m - 1) // (128 * cluster_m) * (128 * cluster_m) >= 1 << 31
        or (n + bn - 1) // bn * bn >= 1 << 31
        or padded_k >= 1 << 31
        or padded_rows * padded_k // 16 * 9 >= 1 << 63
        or (padded_rows * padded_k // 16 + 255) // 256 >= 1 << 31
    ):
        return None
    # Data B is divided across paired CTAs; SFB is replicated for the native
    # scale datapath. Leave space for epilogue tiles, alignment and barriers.
    stage_bytes = (128 + bn // cluster_m) * bk // 2 + (128 + bn) * bk // 16
    if stage_bytes * stages + 128 * 64 * output.element_size() + 8192 > 232448:
        return None
    return BlockScaledMmaPlan(
        root.graph_id,
        lhs,
        rhs,
        output,
        alpha,
        round_dtypes,
        m,
        n,
        groups * 16,
        bn,
        bk,
        stages,
        cluster_m,
        persistent,
        env.config_spec.num_sm or 1,
    )


def plan_block_scaled(graphs: Sequence[GraphInfo]) -> None:
    df = DeviceFunction.current()
    if df.config.get("cute_scaled_mma", False) is not True:
        return
    plan = _plan_block_scaled(graphs)
    if plan is None:
        raise exc.BackendUnsupported(
            "cute", "native block-scaled contraction proof declined"
        )
    df.cute_state.block_scaled_plan = plan


def codegen_block_scaled(cg: GenerateAST) -> bool:
    df = cg.device_function
    plan = df.cute_state.block_scaled_plan
    if (
        plan is None
        or cg.current_root_graph_info is None
        or cg.current_root_graph_info.graph_id != plan.root_graph_id
    ):
        return False
    host = HostFunction.current()
    tensors = (
        plan.lhs.source.tensor,
        plan.rhs.source.tensor,
        plan.lhs.source.scale,
        plan.rhs.source.scale,
        plan.output,
    )
    names = tuple(df.tensor_arg(tensor).name for tensor in tensors)
    workspace_name = df.new_var("block_scaled_workspace")
    workspace = torch.empty(
        (plan.workspace_bytes,), dtype=torch.uint8, device=plan.output.device
    )
    host.tensor_to_origin[workspace] = NameOrigin(workspace_name)
    workspace_arg = df.tensor_arg(workspace, prefer_name=workspace_name).name
    # The root call is emitted after this hook. Allocate into the outer host
    # statement list, giving every invocation and graph capture fresh storage.
    reference = host.tensor_to_origin[plan.output].host_str()
    cg.statements_stack[0].append(
        statement_from_string(
            f"{workspace_name} = torch.empty(({plan.workspace_bytes},), dtype=torch.uint8, device={reference}.device)"
        )
    )
    alpha = df.literal_expr(plan.alpha)
    wrapper_plan: dict[str, object] = {
        "kind": "block_scaled_mma",
        "lhs_name": names[0],
        "rhs_name": names[1],
        "lhs_scale_name": names[2],
        "rhs_scale_name": names[3],
        "out_name": names[4],
        "workspace_name": workspace_arg,
        "m": plan.m,
        "n": plan.n,
        "k": plan.k,
        "bn": plan.bn,
        "bk": plan.bk,
        "stages": plan.stages,
        "cluster_m": plan.cluster_m,
        "persistent": plan.persistent,
        "num_sm": plan.num_sm,
        "workspace_bytes": plan.workspace_bytes,
        "round_dtypes": tuple(map(str, plan.round_dtypes)),
        "lhs_byte_addresses": plan.lhs.byte_addresses,
        "rhs_byte_addresses": plan.rhs.byte_addresses,
        "lhs_scale_address": plan.lhs.scale_address,
        "rhs_scale_address": plan.rhs.scale_address,
        "lhs_span": plan.lhs.tensor_span,
        "rhs_span": plan.rhs.tensor_span,
        "lhs_scale_span": plan.lhs.scale_span,
        "rhs_scale_span": plan.rhs.scale_span,
        "lhs_row_fast": plan.lhs.row_fast,
        "rhs_row_fast": plan.rhs.row_fast,
    }
    if alpha.isidentifier():
        wrapper_plan["scale_name"] = alpha
    else:
        assert isinstance(plan.alpha, float)
        wrapper_plan["scale_value"] = plan.alpha
    cg.cute_wrapper_plans.append(wrapper_plan)
    df.placeholder_args.update((*names, workspace_arg, alpha))
    df.preamble = []
    df.body = [ast.Pass()]
    cg.cute_uses_matmul = True
    return True
