"""Preserve pure tensor programs as compact, portable register operations.

Static coordinate maps and backward element demand determine communication.
Unsupported effects, layouts and alias compositions retain scalar lowering.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import math
import operator
from typing import TYPE_CHECKING
from typing import Any
from typing import TypedDict
from typing import cast
from typing_extensions import NotRequired

import numpy as np
import torch
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.fx.node import map_arg

from .row_fragment import RowFragment
from .row_fragment import RowFragmentLayout

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.fx import GraphModule
    from torch.fx import Node

    from ..generate_ast import GenerateAST


class RegisterTensorValue(TypedDict):
    id: int
    shape: list[int]
    dtype: str


class RegisterTensorNode(RegisterTensorValue):
    op: str
    inputs: list[int]
    axis: NotRequired[int]
    map: NotRequired[int]
    static_map: NotRequired[int]
    rows: NotRequired[list[str]]
    owners: NotRequired[list[int]]
    live_registers: NotRequired[list[int]]


class RegisterTensorMap(TypedDict):
    rows: list[list[int]]
    owners: list[int]


class RegisterTensorPlan(TypedDict):
    version: int
    groups: int
    inputs: list[RegisterTensorValue]
    nodes: list[RegisterTensorNode]
    maps: list[RegisterTensorMap]
    outputs: list[int]


_DTYPES = {
    torch.int32: "int32",
    torch.int64: "int64",
    torch.float32: "float32",
    torch.bool: "bool",
}
_POINTWISE = {
    "aten.minimum.default",
    "aten.maximum.default",
    "aten.fmin.default",
    "aten.fmax.default",
    "aten.add.Tensor",
    "aten.sub.Tensor",
    "aten.mul.Tensor",
    "aten.neg.default",
    "aten.eq.Tensor",
    "aten.ne.Tensor",
    "aten.lt.Tensor",
    "aten.le.Tensor",
    "aten.gt.Tensor",
    "aten.ge.Tensor",
    "aten.bitwise_and.Tensor",
    "aten.bitwise_or.Tensor",
    "aten.bitwise_xor.Tensor",
}
_VIEWS = {
    "aten.view.default",
    "aten.reshape.default",
    "aten._unsafe_view.default",
    "aten.alias.default",
    "aten.clone.default",
    "aten.detach.default",
}
_CONSTANT_OPS = {
    "prims.iota.default",
    "prims.convert_element_type.default",
    "aten.arange.default",
    "aten.arange.start",
    "aten.arange.start_step",
    "aten.expand.default",
    "aten.unsqueeze.default",
    "aten.squeeze.dim",
    "aten.squeeze.default",
    "aten.full.default",
    "aten.zeros.default",
    "aten.ones.default",
    "aten.scalar_tensor.default",
    "aten.lift_fresh_copy.default",
    "aten._to_copy.default",
    "aten.view.dtype",
    "aten.gather.default",
    "aten.index_select.default",
    "aten.slice.Tensor",
    "aten.select.int",
    "aten.transpose.int",
    "aten.permute.default",
    "aten.cat.default",
    "aten.stack.default",
    *_VIEWS,
}


@dataclass(frozen=True)
class _Value:
    index: int
    shape: tuple[int, ...]
    dtype: torch.dtype


class _UnsupportedRegisterTensor(ValueError):
    """The pure graph or physical input layout lacks this capability."""


def _require(condition: object, message: str) -> None:
    if not condition:
        raise _UnsupportedRegisterTensor(message)


def _build_plan(
    module: GraphModule,
    *,
    groups: int,
    input_fragments: Sequence[RowFragment] | None = None,
) -> RegisterTensorPlan:
    with unset_fake_temporarily():
        plan = _compile_plan(module, groups=groups, input_fragments=input_fragments)
        _prune_communication(plan)
        _require_uncomposed_communication(plan)
        return plan


def _require_uncomposed_communication(plan: RegisterTensorPlan) -> None:
    """Decline aliases needing route composition before a resident producer."""
    ancestry = {item["id"]: set() for item in plan["inputs"]}
    alias_ops = {
        "aten.gather.default",
        "aten.index_select.default",
        "aten.slice.Tensor",
        "aten.cat.default",
    }
    for node in plan["nodes"]:
        alias = node["op"] in alias_ops or (
            node["op"] == "aten.where.self" and "static_map" in node
        )
        sources = (
            set().union(*(ancestry[index] for index in node["inputs"]))
            if alias
            else set()
        )
        if (
            node["op"] in ("aten.gather.default", "aten.index_select.default")
            and node["axis"] == 0
        ):
            mapping = plan["maps"][node["map"]]
            if any(
                (
                    mapping["rows"][owner][register] != lane
                    for lane, owner in enumerate(mapping["owners"])
                    for register in node["live_registers"]
                )
            ):
                sources.add(node["id"])
        _require(
            len(sources) <= 1, "Static aliases require composed communication routes"
        )
        ancestry[node["id"]] = sources


def _prune_communication(plan: RegisterTensorPlan) -> None:
    """Propagate element demand before deciding which columns communicate."""
    metadata = {item["id"]: item for item in [*plan["inputs"], *plan["nodes"]]}
    demand = {
        index: np.zeros(item["shape"], dtype=np.bool_)
        for index, item in metadata.items()
    }
    for index in plan["outputs"]:
        demand[index][:] = True
    groups = plan["groups"]
    for node in reversed(plan["nodes"]):
        live = demand[node["id"]]
        op = node["op"]
        if op == "constant":
            continue
        if op in (
            "aten.gather.default",
            "aten.index_select.default",
            "aten.slice.Tensor",
        ):
            mapping = plan["maps"][node["map"]]
            indices = np.asarray(mapping["rows"], dtype=np.int64)[mapping["owners"]]
            if node["axis"] == 1:
                rows = np.broadcast_to(np.arange(groups)[:, None], indices.shape)
                np.logical_or.at(demand[node["inputs"][0]], (rows, indices), live)
            else:
                columns = np.broadcast_to(
                    np.arange(node["shape"][1])[None, :], indices.shape
                )
                np.logical_or.at(demand[node["inputs"][0]], (indices, columns), live)
                node["live_registers"] = np.flatnonzero(live.any(axis=0)).tolist()
        elif op == "aten.where.self" and "static_map" in node:
            mapping = plan["maps"][node["static_map"]]
            routes = np.asarray(mapping["rows"], dtype=np.int64)[mapping["owners"]]
            width = node["shape"][1]
            rows = np.broadcast_to(np.arange(groups)[:, None], routes.shape)
            for branch, index in enumerate(node["inputs"]):
                np.logical_or.at(
                    demand[index],
                    (rows, routes % width),
                    live & (routes // width == branch),
                )
        elif op == "aten.cat.default":
            begin = 0
            for index in node["inputs"]:
                width = metadata[index]["shape"][1]
                demand[index] |= live[:, begin : begin + width]
                begin += width
        else:
            for index in node["inputs"]:
                demand[index] |= live


def _compile_plan(
    module: GraphModule,
    *,
    groups: int,
    input_fragments: Sequence[RowFragment] | None = None,
) -> RegisterTensorPlan:
    """Reject by effect, operation, shape, dtype and ownership, never algorithm."""
    _require(
        groups in (1, 2, 4, 8, 16, 32),
        "Unsupported register program: groups in (1, 2, 4, 8, 16, 32)",
    )
    plan: RegisterTensorPlan = {
        "version": 1,
        "groups": groups,
        "inputs": [],
        "nodes": [],
        "maps": [],
        "outputs": [],
    }
    values: dict[Node, Any] = {}
    next_value = 0
    map_ids = {}
    common_values = {}

    def metadata(node: Node) -> tuple[tuple[int, ...], torch.dtype]:
        tensor = node.meta["val"]
        _require(
            isinstance(tensor, torch.Tensor),
            "Unsupported register program: isinstance(tensor, torch.Tensor)",
        )
        _require(
            tensor.dtype in _DTYPES,
            "Unsupported register program: tensor.dtype in DTYPES",
        )
        shape = tuple(tensor.shape)
        _require(
            all(type(size) is int and size > 0 for size in shape),
            "Unsupported register program: all((type(size) is int and size > 0 for size in shape))",
        )
        _require(
            len(shape) == 2 and shape[0] == groups,
            "Unsupported register program: len(shape) == 2 and shape[0] == groups",
        )
        return (cast("tuple[int, ...]", shape), tensor.dtype)

    def add_map(matrix: list[list[int]]) -> int:
        rows: list[list[int]] = []
        owners: list[int] = []
        row_ids = {}
        for row in matrix:
            key = tuple(row)
            if key not in row_ids:
                row_ids[key] = len(rows)
                rows.append(row)
            owners.append(row_ids[key])
        key = (tuple(tuple(row) for row in rows), tuple(owners))
        item: RegisterTensorMap = {"rows": rows, "owners": owners}
        if key not in map_ids:
            map_ids[key] = len(plan["maps"])
            plan["maps"].append(item)
        return map_ids[key]

    def append(
        op: str,
        inputs: Sequence[_Value],
        shape: tuple[int, ...],
        dtype: torch.dtype,
        **attributes: object,
    ) -> _Value:
        nonlocal next_value
        signature = json.dumps(
            [op, [value.index for value in inputs], shape, _DTYPES[dtype], attributes],
            sort_keys=True,
            separators=(",", ":"),
        )
        if signature in common_values:
            return common_values[signature]
        result = _Value(next_value, shape, dtype)
        next_value += 1
        plan["nodes"].append(
            cast(
                "RegisterTensorNode",
                {
                    "id": result.index,
                    "op": op,
                    "inputs": [value.index for value in inputs],
                    "shape": list(shape),
                    "dtype": _DTYPES[dtype],
                    **attributes,
                },
            )
        )
        common_values[signature] = result
        return result

    def constant(
        tensor: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype
    ) -> _Value:
        _require(
            isinstance(tensor, torch.Tensor) and tensor.dtype == dtype,
            "Unsupported register program: isinstance(tensor, torch.Tensor) and tensor.dtype == dtype",
        )
        tensor = tensor.expand(shape).contiguous()
        rows = [tensor[group].numpy().tobytes().hex() for group in range(groups)]
        unique = list(dict.fromkeys(rows))
        return append(
            "constant",
            [],
            shape,
            dtype,
            rows=unique,
            owners=[unique.index(row) for row in rows],
        )

    def operand(value: object, shape: tuple[int, ...], dtype: torch.dtype) -> _Value:
        if isinstance(value, _Value):
            _require(
                value.shape == shape and value.dtype == dtype,
                "Unsupported register program: value.shape == shape and value.dtype == dtype",
            )
            return value
        return constant(cast("torch.Tensor", value), shape, dtype)

    for node in module.graph.nodes:
        if node.op == "placeholder":
            shape, dtype = metadata(node)
            if input_fragments is not None:
                fragment = input_fragments[len(plan["inputs"])]
                _require(
                    fragment.dtype == dtype and fragment.extent == math.prod(shape),
                    "Unsupported register program: fragment.dtype == dtype and fragment.extent == math.prod(shape)",
                )
                _require(
                    not fragment.replicated,
                    "Unsupported register program: not fragment.replicated",
                )
                _require(
                    fragment.layout.lanes == groups,
                    "Unsupported register program: fragment.layout.lanes == groups",
                )
                _require(
                    fragment.layout.vector_width == shape[1],
                    "Unsupported register program: fragment.layout.vector_width == shape[1]",
                )
                _require(
                    fragment.layout.owner_lanes is None,
                    "Unsupported register program: fragment.layout.owner_lanes is None",
                )
            values[node] = _Value(next_value, shape, dtype)
            plan["inputs"].append(
                {"id": next_value, "shape": list(shape), "dtype": _DTYPES[dtype]}
            )
            next_value += 1
            continue
        if node.op == "get_attr":
            values[node] = operator.attrgetter(node.target)(module)
            _require(
                isinstance(values[node], torch.Tensor),
                "Unsupported register program: isinstance(values[node], torch.Tensor)",
            )
            continue
        args = cast("Any", map_arg(node.args, values.__getitem__))
        kwargs = cast("dict[str, Any]", map_arg(node.kwargs, values.__getitem__))
        if node.op == "output":
            result = args[0] if isinstance(args[0], (tuple, list)) else (args[0],)
            _require(
                all(isinstance(value, _Value) for value in result),
                "Unsupported register program: all((isinstance(value, Value) for value in result))",
            )
            plan["outputs"] = [value.index for value in result]
            continue
        _require(
            node.op == "call_function"
            and isinstance(node.target, torch._ops.OpOverload),
            "Unsupported register program: node.op == 'call_function' and isinstance(node.target, torch._ops.OpOverload)",
        )
        _require(
            not node.target._schema.is_mutable,
            "Unsupported register program: not node.target._schema.is_mutable",
        )
        _require(
            torch.Tag.nondeterministic_seeded not in node.target.tags,
            "Unsupported register program: torch.Tag.nondeterministic_seeded not in node.target.tags",
        )
        _require(
            torch.Tag.nondeterministic_bitwise not in node.target.tags,
            "Unsupported register program: torch.Tag.nondeterministic_bitwise not in node.target.tags",
        )
        if not any(
            isinstance(values[source], _Value) for source in node.all_input_nodes
        ):
            _require(
                str(node.target) in _CONSTANT_OPS
                or torch.Tag.pointwise in node.target.tags,
                "Unsupported register program: str(node.target) in CONSTANT_OPS or torch.Tag.pointwise in node.target.tags",
            )
            constant_kwargs = (
                {**kwargs, "device": torch.device("cpu")}
                if "device" in kwargs
                else kwargs
            )
            values[node] = node.target(*args, **constant_kwargs)
            continue
        op = str(node.target)
        shape, dtype = metadata(node)
        if op in _VIEWS:
            source = args[0]
            _require(
                source.shape[0] == shape[0],
                "Unsupported register program: source.shape[0] == shape[0]",
            )
            _require(
                math.prod(source.shape[1:]) == math.prod(shape[1:]),
                "Unsupported register program: math.prod(source.shape[1:]) == math.prod(shape[1:])",
            )
            _require(
                source.dtype == dtype,
                "Unsupported register program: source.dtype == dtype",
            )
            values[node] = _Value(source.index, shape, dtype)
        elif op in ("aten.gather.default", "aten.index_select.default"):
            source, axis, index = args[:3]
            _require(
                isinstance(index, torch.Tensor)
                and index.dtype in (torch.int32, torch.int64),
                "Unsupported register program: isinstance(index, torch.Tensor) and index.dtype in (torch.int32, torch.int64)",
            )
            axis %= 2
            if index.ndim == 1:
                index = (
                    index[None, :].expand(shape)
                    if axis == 1
                    else index[:, None].expand(shape)
                )
            _require(
                tuple(index.shape) == shape,
                "Unsupported register program: tuple(index.shape) == shape",
            )
            _require(
                int(index.min()) >= 0 and int(index.max()) < source.shape[axis],
                "Unsupported register program: int(index.min()) >= 0 and int(index.max()) < source.shape[axis]",
            )
            _require(
                source.dtype == dtype,
                "Unsupported register program: source.dtype == dtype",
            )
            _require(
                axis == 1 or shape[1] <= source.shape[1],
                "Unsupported register program: axis == 1 or shape[1] <= source.shape[1]",
            )
            values[node] = append(
                op, [source], shape, dtype, axis=axis, map=add_map(index.tolist())
            )
        elif op == "aten.slice.Tensor":
            source, axis, begin, end = args[:4]
            step = args[4] if len(args) > 4 else 1
            _require(axis % 2 == 1, "Unsupported register program: axis % 2 == 1")
            indices = list(range(source.shape[1]))[slice(begin, end, step)]
            _require(
                len(indices) == shape[1],
                "Unsupported register program: len(indices) == shape[1]",
            )
            values[node] = append(
                op, [source], shape, dtype, axis=1, map=add_map([indices] * groups)
            )
        elif op == "aten.cat.default":
            sources = args[0]
            _require(
                all(isinstance(source, _Value) for source in sources),
                "Concatenation with constant tensors needs scalar lowering",
            )
            axis = args[1] if len(args) > 1 else 0
            _require(axis % 2 == 1, "Unsupported register program: axis % 2 == 1")
            _require(
                all(
                    source.dtype == dtype and source.shape[0] == groups
                    for source in sources
                ),
                "Unsupported register program: all((source.dtype == dtype and source.shape[0] == groups for source in sources))",
            )
            _require(
                sum(source.shape[1] for source in sources) == shape[1],
                "Unsupported register program: sum((source.shape[1] for source in sources)) == shape[1]",
            )
            values[node] = append(op, sources, shape, dtype)
        elif op == "aten.where.self":
            condition, left, right = args
            left, right = (operand(left, shape, dtype), operand(right, shape, dtype))
            if isinstance(condition, torch.Tensor):
                _require(
                    condition.dtype == torch.bool,
                    "Unsupported register program: condition.dtype == torch.bool",
                )
                mask = condition.expand(shape).tolist()
                routes = [
                    [
                        index if flag else index + shape[1]
                        for index, flag in enumerate(row)
                    ]
                    for row in mask
                ]
                values[node] = append(
                    op, [left, right], shape, dtype, static_map=add_map(routes)
                )
            else:
                _require(
                    condition.shape == shape and condition.dtype == torch.bool,
                    "Unsupported register program: condition.shape == shape and condition.dtype == torch.bool",
                )
                values[node] = append(op, [condition, left, right], shape, dtype)
        elif op in _POINTWISE:
            _require(
                not kwargs
                or (
                    op in ("aten.add.Tensor", "aten.sub.Tensor")
                    and kwargs == {"alpha": 1}
                ),
                "Unsupported register program: not kwargs or (op in ('aten.add.Tensor', 'aten.sub.Tensor') and kwargs == {'alpha': 1})",
            )
            sources = [arg for arg in args if isinstance(arg, _Value)]
            _require(
                bool(sources), "Pointwise operation needs a register tensor operand"
            )
            sources = [operand(arg, shape, sources[0].dtype) for arg in args]
            _require(
                sources[0].dtype != torch.bool
                or op
                in (
                    "aten.eq.Tensor",
                    "aten.ne.Tensor",
                    "aten.bitwise_and.Tensor",
                    "aten.bitwise_or.Tensor",
                    "aten.bitwise_xor.Tensor",
                ),
                "Boolean arithmetic or ordering needs a separate lowering",
            )
            _require(
                all(source.shape == shape for source in sources),
                "Unsupported register program: all((source.shape == shape for source in sources))",
            )
            _require(
                len({source.dtype for source in sources}) == 1,
                "Unsupported register program: len({source.dtype for source in sources}) == 1",
            )
            values[node] = append(op, sources, shape, dtype)
        else:
            raise _UnsupportedRegisterTensor(f"Unsupported structural operation: {op}")
    _require(
        plan["inputs"] and plan["outputs"],
        "Unsupported register program: plan['inputs'] and plan['outputs']",
    )
    _require(
        input_fragments is None or len(input_fragments) == len(plan["inputs"]),
        "Unsupported register program: input_fragments is None or len(input_fragments) == len(plan['inputs'])",
    )
    return plan


def _serialize_plan(plan: RegisterTensorPlan) -> str:
    return json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False)


def plan_register_tensor(
    module: GraphModule, inputs: Sequence[RowFragment], *, lanes: int
) -> RegisterTensorPlan | None:
    """Admit a complete graph before adding any generated source or names."""
    try:
        return _build_plan(module, groups=lanes, input_fragments=inputs)
    except _UnsupportedRegisterTensor:
        return None


def emit_register_tensor(
    cg: GenerateAST,
    plan: RegisterTensorPlan,
    inputs: Sequence[RowFragment],
    *,
    lane_expr: str,
) -> tuple[RowFragment, ...]:
    encoded = _serialize_plan(plan)
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    plan_name = f"_cute_register_plan_{digest}"
    # The _cute_ call name lets motion/divergence passes recognize collectives.
    import_statement = ast.ImportFrom(
        module="helion.runtime.cute.register_tensor",
        names=[ast.alias(name="_cute_execute_register_plan")],
        level=0,
    )
    if not any(
        isinstance(statement, ast.ImportFrom)
        and statement.module == import_statement.module
        and any(
            alias.name == "_cute_execute_register_plan" for alias in statement.names
        )
        for statement in cg.module_statements
    ):
        cg.module_statements.append(import_statement)
    if not any(
        isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == plan_name
            for target in statement.targets
        )
        for statement in cg.module_statements
    ):
        cg.module_statements.append(
            ast.Assign(
                targets=[ast.Name(id=plan_name, ctx=ast.Store())],
                value=ast.Constant(value=encoded),
            )
        )
    metadata = {item["id"]: item for item in [*plan["inputs"], *plan["nodes"]]}
    dtypes = {name: dtype for dtype, name in _DTYPES.items()}
    outputs = []
    for index in plan["outputs"]:
        item = metadata[index]
        name = cg.device_function.new_var("_cute_register_output")
        outputs.append(
            RowFragment(
                name,
                dtypes[item["dtype"]],
                plan["groups"] * item["shape"][1],
                RowFragmentLayout(plan["groups"], item["shape"][1], lane_expr),
            )
        )
    targets = ", ".join(output.name for output in outputs) + ","
    arguments = ", ".join(value.name for value in inputs) + ","
    cg.add_statement(
        ast.parse(
            f"{targets} = _cute_execute_register_plan({plan_name}, ({arguments}), {lane_expr})"
        ).body[0]
    )
    return tuple(outputs)
