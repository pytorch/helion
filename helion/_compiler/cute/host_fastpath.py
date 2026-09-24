"""Emit a separate host-guarded memory fastpath without changing its fallback."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import NoReturn
from typing import cast

import torch

from ... import exc
from ..ast_extension import statement_from_string
from .host_fastpath_proof import Integer
from .host_fastpath_proof import Pointer
from .host_fastpath_proof import Proof
from .host_fastpath_proof import RegisterTensor
from .host_fastpath_proof import Tensor
from .host_fastpath_proof import assigned

if TYPE_CHECKING:
    from ..device_function import DeviceFunction
    from ..device_function import TensorArg

KEY = "cute_host_selected_fastpath"
Contract = tuple[tuple[int, ...], tuple[int, ...], str, int]


@dataclass(frozen=True)
class Plan:
    original: str
    fast: str
    contracts: tuple[Contract, ...]
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    alias: str
    removed: int


def _unsupported(reason: str) -> NoReturn:
    raise exc.BackendUnsupported("cute", f"host-selected fastpath: {reason}")


def _constants(df: DeviceFunction) -> dict[str, Integer]:
    values: dict[str, Integer] = {}
    proof = Proof({}, (1, 1, 1), (1, 1, 1))
    for arg in df.sorted_args():
        from ..device_function import ConstExprArg
        from ..device_function import _is_literal_constexpr

        if isinstance(arg, ConstExprArg) and _is_literal_constexpr(arg):
            value = proof.value(ast.parse(arg.host_str(), mode="eval").body, values)
            if isinstance(value, Integer):
                values[arg.name] = value
    for stmt in df.codegen.module_statements:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            value = proof.value(stmt.value, values)
            if isinstance(value, Integer):
                values[stmt.targets[0].id] = value
    for stmt in df.codegen.host_statements:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Constant)
            and type(stmt.value.value) is int
        ):
            value = proof.value(stmt.value, values)
            if isinstance(value, Integer):
                name = stmt.targets[0].id
                if name in values and values[name] != value:
                    _unsupported("conflicting generated integer constant")
                values[name] = value
    return values


def _parameters(df: DeviceFunction) -> list[TensorArg]:
    from ..device_function import ConstExprArg
    from ..device_function import TensorArg
    from ..device_function import _is_literal_constexpr

    args = [
        arg
        for arg in df.sorted_args()
        if not (isinstance(arg, ConstExprArg) and _is_literal_constexpr(arg))
    ]
    if not args or any(type(arg) is not TensorArg for arg in args):
        _unsupported("only direct tensor launch arguments are modeled")
    return cast("list[TensorArg]", args)


def variants(df: DeviceFunction, result: list[ast.stmt]) -> list[ast.stmt]:
    if not df.config.config.get(KEY, False):
        return result
    if df.codegen._extra_params or df.wrapper_only_params or df.has_rng_ops():
        _unsupported("extra/wrapper/RNG arguments are not modeled")
    if df.cute_state.simt_cluster_n != 1 or df.cute_state.cluster_shape is not None:
        _unsupported("cluster launch is not modeled")
    params = _parameters(df)
    contracts = []
    tensors = {}
    for arg in params:
        tensor = arg.fake_value
        if (
            tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16)
            or not all(type(n) is int and n > 0 for n in tensor.shape)
            or not all(type(n) is int and n >= 0 for n in tensor.stride())
        ):
            _unsupported("requires concrete positive tensor metadata")
        contract = (
            tuple(tensor.shape),
            tuple(tensor.stride()),
            str(tensor.dtype),
            tensor.element_size(),
        )
        contracts.append(contract)
        tensors[arg.name] = contract
    original_name = df.name
    serial = df.cute_state.serial_lane_plan
    if serial is not None:
        original_name += "_serial_lane"
    original = next(
        n for n in result if isinstance(n, ast.FunctionDef) and n.name == original_name
    )
    if (
        len(original.args.args) != len(contracts)
        or original.args.posonlyargs
        or original.args.kwonlyargs
        or original.args.vararg
        or original.args.kwarg
    ):
        _unsupported("device parameter order is not the direct tensor ABI")
    tensors = {
        arg.arg: contract
        for arg, contract in zip(original.args.args, contracts, strict=True)
    }
    if assigned(original) & set(tensors):
        _unsupported("reassigned device tensor argument")
    constants = _constants(df)
    assert df.pid is not None
    grid_expr = df.pid.codegen_grid()
    if not isinstance(grid_expr, ast.Tuple) or not 1 <= len(grid_expr.elts) <= 3:
        _unsupported("requires exact static grid")
    parser = Proof(tensors, (1, 1, 1), (1, 1, 1))
    axes = [parser.value(n, constants) for n in grid_expr.elts]
    if not all(isinstance(n, Integer) and n.low == n.high and n.low > 0 for n in axes):
        _unsupported("requires exact positive grid dimensions")
    dimensions = [n.low for n in axes if isinstance(n, Integer)]
    dimensions.extend([1] * (3 - len(dimensions)))
    grid = (dimensions[0], dimensions[1], dimensions[2])
    if serial is not None and serial.coarsen is not None:
        grid = (serial.coarsen.grid // 2, 1, 1)
    chained = df.cute_state.chained_matmul_plan
    if chained is not None:
        block = (chained.threads, 1, 1)
    else:
        from ..device_function import _exact_thread_block_dims

        block = _exact_thread_block_dims(df.tile_strategy)
        if block is None:
            _unsupported("requires exact thread block")
    proof = Proof(tensors, grid, block)
    clone = ast.parse(ast.unparse(original)).body[0]
    assert isinstance(clone, ast.FunctionDef)
    proof_values: dict[str, Integer | Pointer | Tensor | RegisterTensor] = dict(
        constants
    )
    clone.body = proof.block_body(clone.body, proof_values)
    if proof.removed == 0:
        _unsupported("no fully proven guarded vector copies")
    clone.name = original.name + "_host_fast"
    used = {n.id for stmt in result for n in ast.walk(stmt) if isinstance(n, ast.Name)}
    if clone.name in used:
        _unsupported("generated fast symbol collision")
    alias = df.new_var("host_fastpath_guard", dce=False)
    df.cute_state.host_fastpath_plan = Plan(
        original.name, clone.name, tuple(contracts), grid, block, alias, proof.removed
    )
    metadata: list[ast.stmt] = []
    for stmt in result:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Attribute)
            and isinstance(stmt.targets[0].value, ast.Name)
            and stmt.targets[0].value.id == original.name
        ):
            copied = ast.parse(ast.unparse(stmt)).body[0]
            assert isinstance(copied, ast.Assign)
            assert isinstance(copied.targets[0], ast.Attribute)
            copied.targets[0].value = ast.Name(id=clone.name, ctx=ast.Load())
            metadata.append(copied)
    if df.codegen.cute_uses_matmul:
        metadata.append(
            statement_from_string(
                f"{clone.name}._helion_cute_disable_bake_tensor_shapes = True"
            )
        )
    return [*result, clone, *metadata]


def guarded_call(df: DeviceFunction, call: ast.AST) -> ast.AST:
    if not df.config.config.get(KEY, False):
        return call
    plan = df.cute_state.host_fastpath_plan
    if plan is None:
        _unsupported("missing source proof")
    hosts = [arg.host_str() for arg in _parameters(df)]
    if not all(
        isinstance(ast.parse(name, mode="eval").body, ast.Name) for name in hosts
    ):
        _unsupported("requires direct current tensor locals")
    launch = next(
        n
        for n in ast.walk(call)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "_launcher"
    )
    block = next((k.value for k in launch.keywords if k.arg == "block"), None)
    if block is None:
        _unsupported("missing actual launch block")
    # Preserve an existing serial selection, including its named expression and
    # original-grid alias fallback. Its selected symbol remains the original
    # serial variant while the enclosing call chooses the separately compiled fast.
    launch.args[0] = ast.parse(
        f"{plan.alias}.select_kernel({ast.unparse(launch.args[0])}, {plan.original}, {plan.fast}, ({', '.join(hosts)},), {plan.contracts!r}, {ast.unparse(launch.args[1])}, {plan.grid!r}, {ast.unparse(block)}, {plan.block!r})",
        mode="eval",
    ).body
    return call


def entry_statements(df: DeviceFunction) -> list[ast.stmt]:
    plan = df.cute_state.host_fastpath_plan
    if plan is None:
        _unsupported("missing host plan")
    return [
        statement_from_string(
            f"from helion.runtime.cute import host_fastpath as {plan.alias}"
        ),
        statement_from_string(f"{plan.alias}.validate_trace()"),
    ]
