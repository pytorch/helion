"""Guarded layout facts for static host views of one input allocation."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch

from ..ast_extension import ExtendedAST
from ..host_function import HostFunction
from ..type_info import CallableType
from ..type_info import TensorAttributeType
from ..type_info import TensorType
from .contiguous_copy import CopyTensorFacts
from .memory_ops import tensor_has_specialized_base_alignment

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment


_VIEWS = frozenset(
    {
        "view",
        "reshape",
        "flatten",
        "t",
        "transpose",
        "permute",
        "squeeze",
        "unsqueeze",
        "detach",
    }
)
_METADATA = frozenset(
    {"size", "stride", "dim", "ndimension", "numel", "element_size", "is_contiguous"}
)
_FACTORIES = frozenset(
    {
        torch.empty,
        torch.empty_like,
        torch.zeros,
        torch.zeros_like,
        torch.ones,
        torch.ones_like,
        torch.full,
        torch.full_like,
        torch.promote_types,
    }
)


def _pointer_preserving_views() -> dict[torch.Tensor, torch.Tensor] | None:
    """Follow known host view operations, declining unknown prefix effects.

    Fake input creation can normalize the source storage offset to zero. An
    explicit absolute ``as_strided(storage_offset=...)`` therefore cannot be
    proved from fake offsets alone. Only operations guaranteed to preserve the
    source data pointer are followed; any other host call declines the proof.
    """
    result: dict[torch.Tensor, torch.Tensor] = {}
    for statement in HostFunction.current().body:
        if isinstance(statement, ast.For):
            break
        if not isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Expr)):
            return None
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
            if isinstance(statement, ast.AnnAssign)
            else []
        )
        if any(
            isinstance(node, (ast.Attribute, ast.Subscript))
            for target in targets
            for node in ast.walk(target)
        ):
            return None
        for node in ast.walk(statement):
            if not isinstance(node, ast.Call):
                continue
            if (
                not isinstance(node, ExtendedAST)
                or not isinstance(node.func, ExtendedAST)
                or any(keyword.arg in (None, "out") for keyword in node.keywords)
            ):
                return None
            callee = node.func._type_info
            if isinstance(callee, CallableType) and any(
                callee.value is factory for factory in _FACTORIES
            ):
                continue
            if not isinstance(callee, TensorAttributeType):
                return None
            method = callee.attr()
            if method in _METADATA:
                continue
            if method not in _VIEWS or not isinstance(node._type_info, TensorType):
                return None
            value, base = node._type_info.proxy(), callee.tensor.proxy()
            if (
                value.dtype != base.dtype
                or value.untyped_storage()._cdata != base.untyped_storage()._cdata
                or type(value.storage_offset()) is not int
                or type(base.storage_offset()) is not int
                or value.storage_offset() != base.storage_offset()
            ):
                return None
            if value is not base:
                result[value] = base
    return result


def input_view_copy_facts(
    env: CompileEnvironment, tensor: torch.Tensor
) -> CopyTensorFacts | None:
    """Recover exact strides and pointer alignment through a static alias view.

    Full input metadata is part of the bound-kernel key. A proved pointer-
    preserving view with literal strides from exactly one traced input
    therefore inherits that input's guarded pointer alignment. Multiple input
    aliases are deliberately ambiguous: their relative offsets could change
    while the runtime storage-overlap classifier still reports overlap.

    Fresh allocations, symbolic view layouts, dtype reinterpretation and
    unguarded inputs do not receive a proof from this helper.
    """
    if (
        "input_tensor_metadata" not in env.compiler_fact_specialization_facts
        or not env.settings.static_shapes
        or env.tensor_input_source(tensor) is not None
    ):
        return None
    strides = tuple(tensor.stride())
    offset = tensor.storage_offset()
    if any(type(stride) is not int for stride in strides) or type(offset) is not int:
        return None
    storage = tensor.untyped_storage()._cdata
    inputs = [
        source
        for source in env.input_sources
        if source.untyped_storage()._cdata == storage
    ]
    if len(inputs) != 1:
        return None
    base = inputs[0]
    views = _pointer_preserving_views()
    if views is None:
        return None
    current = tensor
    visited: set[torch.Tensor] = set()
    while current is not base:
        if current in visited or current not in views:
            return None
        visited.add(current)
        current = views[current]
    base_offset = base.storage_offset()
    if (
        base.dtype != tensor.dtype
        or type(base_offset) is not int
        or env.tensor_input_source(base) is None
        or (offset - base_offset) * tensor.element_size() % 16
        or not tensor_has_specialized_base_alignment(env, base, 16)
    ):
        return None
    return CopyTensorFacts(env.backend.dtype_str(tensor.dtype), strides, 16)
