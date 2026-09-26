"""Exact generated-host metadata relations for ordinary CuTe tensor arguments.

These facts describe the tensor object passed to this launch, not a symbolic
size hint or equality between two example inputs. No runtime schema is changed.
The launcher still supplies the scalar, and a private variant may replace it
with the same Int64 property of that launch's actual CuTe tensor.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..ast_extension import statement_from_string

if TYPE_CHECKING:
    from collections.abc import Collection
    from collections.abc import Mapping
    from collections.abc import Sequence


@dataclass(frozen=True)
class TensorLayoutRelation:
    scalar: str
    tensor: str
    property: str
    axis: int

    def assignment(self) -> ast.stmt:
        value = (
            f"cute.size({self.tensor}, mode=[{self.axis}])"
            if self.property == "size"
            else f"{self.tensor}.layout.stride[{self.axis}]"
        )
        # Runtime integer parameters and unbaked tensor metadata are Int64.
        # Keeping that type also preserves modular arithmetic at the consumer.
        return statement_from_string(f"{self.scalar} = cutlass.Int64({value})")


@dataclass(frozen=True)
class _Tensor:
    identity: int
    rank: int
    numel_identity: int


@dataclass(frozen=True)
class _Property:
    tensor: _Tensor
    property: str
    axis: int


@dataclass(frozen=True)
class _Scalar:
    pass


_Value = _Tensor | _Property | _Scalar | tuple["_Value", ...] | None


class _Decline(Exception):
    pass


def _literal_axis(node: ast.expr, rank: int) -> int | None:
    sign = 1
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        sign = -1 if isinstance(node.op, ast.USub) else 1
        node = node.operand
    if not isinstance(node, ast.Constant) or type(node.value) is not int:
        return None
    axis = sign * node.value
    return axis % rank if rank > 0 and -rank <= axis < rank else None


class _HostState:
    def __init__(
        self,
        tensor_inputs: Mapping[str, int],
        scalar_inputs: Collection[str],
        *,
        allow_flatten: bool,
    ) -> None:
        self.values: dict[str, _Value] = dict.fromkeys(scalar_inputs, _Scalar())
        self.allow_flatten = allow_flatten
        self.next_identity = 0
        for name, rank in tensor_inputs.items():
            self.values[name] = self.tensor(rank)

    def tensor(self, rank: int, *, numel_identity: int | None = None) -> _Tensor:
        result = _Tensor(
            self.next_identity,
            rank,
            self.next_identity if numel_identity is None else numel_identity,
        )
        self.next_identity += 1
        return result

    def read(self, node: ast.expr) -> _Value:
        if isinstance(node, ast.Name):
            return self.values.get(node.id)
        if isinstance(node, ast.Constant):
            return _Scalar()
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(self.read(item) for item in node.elts)
        if isinstance(node, ast.Attribute):
            tensor = self.read(node.value)
            if not isinstance(tensor, _Tensor):
                raise _Decline
            if node.attr == "shape":
                return tuple(_Property(tensor, "size", i) for i in range(tensor.rank))
            if node.attr in {"dtype", "device", "ndim"}:
                return None
            raise _Decline
        if isinstance(node, ast.Subscript):
            value = self.read(node.value)
            if (
                isinstance(value, tuple)
                and (axis := _literal_axis(node.slice, len(value))) is not None
            ):
                return value[axis]
            # An unknown subscript may read/mutate storage or invoke user code.
            raise _Decline
        if isinstance(node, ast.Call):
            if (
                self.allow_flatten
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "numel"
            ):
                tensor = self.read(node.func.value)
                if not isinstance(tensor, _Tensor) or node.args or node.keywords:
                    raise _Decline
                return _Property(tensor, "numel", 0)
            if (
                self.allow_flatten
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "view"
            ):
                tensor = self.read(node.func.value)
                if not (
                    isinstance(tensor, _Tensor)
                    and len(node.args) == 1
                    and not node.keywords
                    and isinstance(node.args[0], ast.UnaryOp)
                    and isinstance(node.args[0].op, ast.USub)
                    and isinstance(node.args[0].operand, ast.Constant)
                    and type(node.args[0].operand.value) is int
                    and node.args[0].operand.value == 1
                ):
                    raise _Decline
                # The unchanged host still executes view and raises on an
                # incompatible layout. A successful flatten preserves numel,
                # but it does not preserve the source's rank or strides.
                return self.tensor(1, numel_identity=tensor.numel_identity)
            if isinstance(node.func, ast.Attribute) and node.func.attr in {
                "size",
                "stride",
            }:
                tensor = self.read(node.func.value)
                if not isinstance(tensor, _Tensor) or node.keywords:
                    raise _Decline
                property_name = node.func.attr
                if not node.args:
                    return tuple(
                        _Property(tensor, property_name, i) for i in range(tensor.rank)
                    )
                if (
                    len(node.args) == 1
                    and (axis := _literal_axis(node.args[0], tensor.rank)) is not None
                ):
                    return _Property(tensor, property_name, axis)
                raise _Decline
            if ast.unparse(node.func) == "torch.empty_like":
                if len(node.args) != 1:
                    raise _Decline
                original = self.read(node.args[0])
                if not isinstance(original, _Tensor):
                    raise _Decline
                for keyword in node.keywords:
                    if keyword.arg not in {"dtype", "device", "requires_grad"}:
                        raise _Decline
                    self.read(keyword.value)
                # Fresh allocation gets a fresh identity. Equal shapes never
                # justify replacing an input's property by this output's.
                return self.tensor(original.rank)
            raise _Decline
        if isinstance(node, ast.BinOp):
            if not all(
                isinstance(self.read(operand), (_Scalar, _Property))
                for operand in (node.left, node.right)
            ):
                raise _Decline
            return _Scalar()
        if isinstance(node, ast.UnaryOp):
            if not isinstance(self.read(node.operand), (_Scalar, _Property)):
                raise _Decline
            return _Scalar()
        if isinstance(node, ast.Compare):
            if not all(
                isinstance(self.read(operand), (_Scalar, _Property))
                for operand in (node.left, *node.comparators)
            ):
                raise _Decline
            return _Scalar()
        raise _Decline

    def assign(self, target: ast.expr, value: _Value) -> None:
        if isinstance(target, ast.Name):
            if target.id in {"torch", "_launcher"}:
                raise _Decline
            self.values[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, tuple):
            if len(target.elts) != len(value):
                raise _Decline
            for item, element in zip(target.elts, value, strict=True):
                self.assign(item, element)
        else:
            raise _Decline


def prove_tensor_layout_relations(
    host_body: Sequence[ast.stmt],
    *,
    kernel_name: str,
    parameter_names: Sequence[str],
    tensor_parameters: Mapping[str, int],
    integer_parameters: Collection[str],
    tensor_inputs: Mapping[str, int],
    scalar_inputs: Collection[str] = (),
    allow_flatten: bool = False,
) -> tuple[TensorLayoutRelation, ...]:
    """Replay a bounded straight-line host prefix to one ordinary launch.

    Only metadata reads, aliases, literal/scalar expressions and known fresh
    allocation are admitted. Unknown calls/control flow and tensor mutations
    decline the entire proof. Parameters are matched by actual launch position.
    """
    count = 0
    launches = []
    for statement in host_body:
        for node in ast.walk(statement):
            count += 1
            if count > 4096:
                return ()
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_launcher"
            ):
                launches.append(node)
    if len(launches) != 1:
        return ()
    launch = launches[0]
    if (
        len(launch.args) != 2 + len(parameter_names)
        or not isinstance(launch.args[0], ast.Name)
        or launch.args[0].id != kernel_name
    ):
        return ()
    state = _HostState(tensor_inputs, scalar_inputs, allow_flatten=allow_flatten)
    try:
        for statement in host_body:
            if isinstance(statement, ast.Expr) and statement.value is launch:
                break
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                value = state.read(statement.value)
                state.assign(statement.targets[0], value)
            elif isinstance(statement, ast.Expr) and isinstance(
                statement.value, ast.Constant
            ):
                continue
            else:
                raise _Decline
        else:
            raise _Decline
        values = {
            name: state.read(value)
            for name, value in zip(parameter_names, launch.args[2:], strict=True)
        }
        tensors = {
            name: value
            for name, value in values.items()
            if name in tensor_parameters
            and isinstance(value, _Tensor)
            and value.rank == tensor_parameters[name]
        }
        relations = []
        for name, value in values.items():
            if name not in integer_parameters or not isinstance(value, _Property):
                continue
            if value.property == "numel" or (
                value.property == "size" and value.tensor.rank == 1 and value.axis == 0
            ):
                tensor_name = next(
                    (
                        key
                        for key, tensor in tensors.items()
                        if tensor.rank == 1
                        and tensor.numel_identity == value.tensor.numel_identity
                    ),
                    None,
                )
                if tensor_name is not None:
                    relations.append(TensorLayoutRelation(name, tensor_name, "size", 0))
                continue
            tensor_name = next(
                (key for key, tensor in tensors.items() if tensor == value.tensor),
                None,
            )
            if tensor_name is not None:
                relations.append(
                    TensorLayoutRelation(name, tensor_name, value.property, value.axis)
                )
        return tuple(relations)
    except _Decline:
        return ()
