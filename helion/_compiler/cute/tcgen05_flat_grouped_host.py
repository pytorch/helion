"""Prove host metadata and storage effects for the flat grouped path.

The original host remains executable. Merely finding an allocation somewhere
in it does not prove the output is still private at the device launch. Replay
the complete prefix, admitting only metadata, one input flatten and one fresh
output allocation. Unknown effects retain ordinary codegen.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch

from ... import language
from ..ast_extension import ExtendedAST
from ..ast_extension import LoopType
from ..type_info import LiteralType
from ..type_info import SequenceType
from ..type_info import TensorAttributeType
from ..type_info import TensorType

if TYPE_CHECKING:
    from ..host_function import HostFunction
    from ..type_info import TypeInfo


def _info(node: ast.AST) -> TypeInfo | None:
    return node._type_info if isinstance(node, ExtendedAST) else None


def _scalar(value: object) -> bool:
    return type(value) in (int, float, bool, str, type(None), torch.dtype, torch.device)


def _minus_one(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant) and type(node.value) is int and node.value == -1
    ) or (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and type(node.operand.value) is int
        and node.operand.value == 1
    )


class _HostProof:
    def __init__(self, host: HostFunction, a_flat: str, output: str) -> None:
        self.host = host
        self.a_flat = a_flat
        self.output = output
        self.tensor_names = {
            name
            for name, value in host.params.arguments.items()
            if isinstance(value, torch.Tensor)
        }
        self.protected_names = self.tensor_names | {a_flat, output}
        self.scalars = {
            name for name, value in host.params.arguments.items() if _scalar(value)
        }
        # A typed global identity is usable only if the host cannot rebind it.
        # All compound writes are covered, even ones in a discarded branch.
        self.local_names = set(host.params.arguments)
        self.local_names.update(
            node.id
            for statement in host.body
            for node in ast.walk(statement)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        )
        self.a_argument: str | None = None
        self.fresh_output = False

    def global_reference(self, node: ast.expr, value: object) -> bool:
        info = _info(node)
        if not isinstance(info, LiteralType) or info.value is not value:
            return False
        if isinstance(node, ast.Name):
            return node.id not in self.local_names
        if isinstance(node, ast.Attribute):
            receiver = _info(node.value)
            return (
                isinstance(node.value, ast.Name)
                and node.value.id not in self.local_names
                and isinstance(receiver, LiteralType)
                and any(receiver.value is module for module in (torch, language))
                and vars(receiver.value).get(node.attr) is value
            )
        return False

    def tensor_name(self, node: ast.expr) -> bool:
        return (
            isinstance(node, ast.Name)
            and node.id in self.tensor_names
            and isinstance(_info(node), TensorType)
        )

    def tensor_method(self, node: ast.Call, names: tuple[str, ...]) -> bool:
        function = node.func
        info = _info(function)
        return (
            isinstance(function, ast.Attribute)
            and function.attr in names
            and isinstance(info, TensorAttributeType)
            and info.attr() == function.attr
            and isinstance(_info(function.value), TensorType)
            and not node.keywords
        )

    def metadata(self, node: ast.expr) -> bool:
        if isinstance(node, ast.Constant):
            return _scalar(node.value)
        if isinstance(node, ast.Name):
            info = _info(node)
            return node.id in self.scalars or (
                isinstance(info, LiteralType)
                and _scalar(info.value)
                and self.global_reference(node, info.value)
            )
        if isinstance(node, (ast.Tuple, ast.List)):
            return all(self.metadata(item) for item in node.elts)
        if isinstance(node, ast.Attribute):
            if self.tensor_name(node.value):
                return node.attr in {"shape", "dtype", "device", "ndim"}
            info = _info(node)
            return (
                isinstance(info, LiteralType)
                and _scalar(info.value)
                and self.global_reference(node, info.value)
            )
        if isinstance(node, ast.Subscript):
            return (
                isinstance(_info(node.value), SequenceType)
                and self.metadata(node.value)
                and self.metadata(node.slice)
            )
        if isinstance(node, ast.BinOp):
            return self.metadata(node.left) and self.metadata(node.right)
        if isinstance(node, ast.UnaryOp):
            return self.metadata(node.operand)
        if isinstance(node, ast.Compare):
            return all(self.metadata(item) for item in (node.left, *node.comparators))
        if isinstance(node, ast.BoolOp):
            return all(self.metadata(item) for item in node.values)
        if isinstance(node, ast.IfExp):
            return all(
                self.metadata(item) for item in (node.test, node.body, node.orelse)
            )
        if isinstance(node, ast.Call):
            if self.global_reference(node.func, torch.promote_types):
                return (
                    len(node.args) == 2
                    and not node.keywords
                    and all(self.metadata(item) for item in node.args)
                )
            return (
                self.tensor_method(
                    node, ("size", "stride", "numel", "dim", "ndimension")
                )
                and isinstance(node.func, ast.Attribute)
                and self.tensor_name(node.func.value)
                and all(self.metadata(item) for item in node.args)
            )
        return False

    def metadata_target(self, node: ast.expr) -> bool:
        if isinstance(node, ast.Name) and node.id not in self.protected_names:
            self.scalars.add(node.id)
            return True
        if isinstance(node, (ast.Tuple, ast.List)):
            return all(self.metadata_target(item) for item in node.elts)
        return False

    def flatten(self, node: ast.Assign) -> bool:
        value = node.value
        if not (
            self.a_argument is None
            and isinstance(value, ast.Call)
            and self.tensor_method(value, ("view",))
            and len(value.args) == 1
            and _minus_one(value.args[0])
            and isinstance(value.func, ast.Attribute)
            and isinstance(value.func.value, ast.Name)
            and self.tensor_name(value.func.value)
        ):
            return False
        receiver = value.func.value.id
        tensor = self.host.params.arguments.get(receiver)
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim != 2
            or tensor.dtype is not torch.float32
            or (self.a_flat in self.tensor_names and self.a_flat != receiver)
        ):
            return False
        self.a_argument = receiver
        self.tensor_names.add(self.a_flat)
        return True

    def allocation(self, node: ast.Assign) -> bool:
        view = node.value
        if not (
            not self.fresh_output
            and self.output not in self.tensor_names
            and isinstance(view, ast.Call)
            and self.tensor_method(view, ("view",))
            and len(view.args) == 1
            and _minus_one(view.args[0])
            and isinstance(view.func, ast.Attribute)
            and isinstance(view.func.value, ast.Call)
        ):
            return False
        allocation = view.func.value
        if not (
            self.global_reference(allocation.func, torch.empty)
            and all(self.metadata(item) for item in allocation.args)
            and all(
                item.arg in {"dtype", "device", "requires_grad"}
                and self.metadata(item.value)
                for item in allocation.keywords
            )
        ):
            return False
        self.fresh_output = True
        self.tensor_names.add(self.output)
        return True

    def prefix(
        self, statements: list[ast.stmt], *, allow_tensor_bindings: bool = True
    ) -> bool:
        for statement in statements:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target = statement.targets[0]
                if isinstance(target, ast.Name) and target.id == self.a_flat:
                    if not allow_tensor_bindings or not self.flatten(statement):
                        return False
                elif isinstance(target, ast.Name) and target.id == self.output:
                    if not allow_tensor_bindings or not self.allocation(statement):
                        return False
                elif not self.metadata(statement.value) or not self.metadata_target(
                    target
                ):
                    return False
            elif isinstance(statement, ast.If):
                info = _info(statement.test)
                if not (
                    isinstance(info, LiteralType)
                    and type(info.value) is bool
                    and self.metadata(statement.test)
                ):
                    return False
                # Both branches must be metadata-only, even if tracing knows
                # the condition. A captured global Boolean is not a lifetime
                # proof that an executable host mutation stays unreachable.
                before = self.scalars.copy()
                if not self.prefix(statement.body, allow_tensor_bindings=False):
                    return False
                body_scalars = self.scalars
                self.scalars = before
                if not self.prefix(statement.orelse, allow_tensor_bindings=False):
                    return False
                self.scalars.intersection_update(body_scalars)
            elif not (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            ):
                return False
        return True

    def returned_output(self, node: ast.expr | None) -> bool:
        if isinstance(node, ast.Name):
            return node.id == self.output
        return (
            isinstance(node, ast.Call)
            and self.tensor_method(node, ("reshape", "view"))
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == self.output
            and all(self.metadata(item) for item in node.args)
        )


def prove_flat_grouped_host(host: HostFunction, a_flat: str, output: str) -> str | None:
    """Return the original A argument only for an effect-proved host prefix."""
    if a_flat == output or len(host.body) < 3:
        return None
    # The sole grid and final return have no executable host epilogue between
    # them. Unknown control flow/effects anywhere before the grid are rejected.
    grid, result = host.body[-2:]
    if not (
        isinstance(grid, ast.For)
        and isinstance(grid, ExtendedAST)
        and grid._loop_type is LoopType.GRID
        and grid._root_id == 0
        and not grid.orelse
        and isinstance(grid.iter, ast.Call)
        and isinstance(result, ast.Return)
    ):
        return None
    proof = _HostProof(host, a_flat, output)
    if not (
        proof.prefix(host.body[:-2])
        and proof.a_argument is not None
        and proof.fresh_output
        and proof.global_reference(grid.iter.func, language.tile)
        and all(proof.metadata(item) for item in grid.iter.args)
        and all(
            item.arg is not None and proof.metadata(item.value)
            for item in grid.iter.keywords
        )
        and proof.returned_output(result.value)
    ):
        return None
    return proof.a_argument
