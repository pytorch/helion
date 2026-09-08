"""Plan body-local tensors that can remain entirely in TPU VMEM."""

from __future__ import annotations

import ast

import torch

from ... import exc
from ...language import distributed_ops
from ...language import memory_ops
from ...language.atomic_ops import ATOMIC_OPS
from ..compile_environment import CompileEnvironment
from ..device_function import DeviceFunction
from ..host_function import HostFunction

_EMPTY_FACTORIES = frozenset({"empty", "empty_like", "new_empty"})


def _top_level_empty_names(host: HostFunction) -> set[str]:
    names: set[str] = set()
    for statement in host.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and statement.value.func.attr in _EMPTY_FACTORIES
        ):
            names.add(statement.targets[0].id)
    return names


def _returned_name_dependencies(host: HostFunction) -> set[str]:
    """Conservatively find every host name that can feed the return value."""
    escaped: set[str] = set()
    dependencies: dict[str, set[str]] = {}
    module = ast.Module(body=host.body, type_ignores=[])
    for node in ast.walk(module):
        if isinstance(node, ast.Return) and node.value is not None:
            escaped.update(
                child.id
                for child in ast.walk(node.value)
                if isinstance(child, ast.Name)
            )
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            dependencies.setdefault(node.targets[0].id, set()).update(
                child.id
                for child in ast.walk(node.value)
                if isinstance(child, ast.Name)
            )

    pending = list(escaped)
    while pending:
        name = pending.pop()
        for dependency in dependencies.get(name, ()):
            if dependency not in escaped:
                escaped.add(dependency)
                pending.append(dependency)
    return escaped


def _tensor_storage(node: object) -> int | None:
    if not isinstance(node, torch.fx.Node):
        return None
    value = node.meta.get("val")
    return id(value.untyped_storage()) if isinstance(value, torch.Tensor) else None


def _accessed_storages(
    device_fn: DeviceFunction,
) -> tuple[set[int], set[int], set[int]]:
    read: set[int] = set()
    written: set[int] = set()
    remote: set[int] = set()
    for graph_info in device_fn.codegen.codegen_graphs:
        for node in graph_info.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target is memory_ops.load:
                if (storage := _tensor_storage(node.args[0])) is not None:
                    read.add(storage)
            elif node.target is memory_ops.store:
                if (storage := _tensor_storage(node.args[0])) is not None:
                    written.add(storage)
            elif node.target in ATOMIC_OPS:
                if (storage := _tensor_storage(node.args[0])) is not None:
                    read.add(storage)
                    written.add(storage)
            elif node.target is distributed_ops.make_async_remote_copy:
                if len(node.args) != 5:
                    raise exc.InternalError(
                        RuntimeError(
                            "remote copy was not normalized to its five-argument form"
                        )
                    )
                if (storage := _tensor_storage(node.args[0])) is not None:
                    read.add(storage)
                    remote.add(storage)
                if (storage := _tensor_storage(node.args[3])) is not None:
                    written.add(storage)
                    remote.add(storage)
    return read, written, remote


def plan_internal_remote_scratch() -> None:
    """Place private, body-local remote-copy buffers in VMEM scratch.

    The transformation is intentionally narrow. An allocation must be a
    top-level ``torch.empty``-family call, have a fully static shape, be both
    read and written, participate in remote DMA, and not feed the host return.
    """
    host = HostFunction.current()
    device_fn = DeviceFunction.current()
    allocation_names = _top_level_empty_names(host) - _returned_name_dependencies(host)
    if not allocation_names:
        return

    read, written, remote = _accessed_storages(device_fn)
    eligible_storages = read & written & remote
    if not eligible_storages:
        return

    input_storages = {
        id(tensor.untyped_storage())
        for tensor in CompileEnvironment.current().input_sources
    }
    tensors_by_storage: dict[int, list[torch.Tensor]] = {}
    names_by_storage: dict[int, str] = {}
    for tensor, origin in host.tensor_to_origin.items():
        try:
            name = origin.host_str()
        except RuntimeError:
            continue
        storage = id(tensor.untyped_storage())
        if (
            name not in allocation_names
            or storage in input_storages
            or storage not in eligible_storages
            or not all(isinstance(size, int) for size in tensor.shape)
        ):
            continue
        tensors_by_storage.setdefault(storage, []).append(tensor)
        names_by_storage[storage] = name

    for storage, tensors in tensors_by_storage.items():
        representative = max(tensors, key=lambda tensor: (tensor.ndim, tensor.numel()))
        scratch_name = device_fn.register_scratch(
            tuple(representative.shape),
            representative.dtype,
            name_hint=names_by_storage[storage],
        )
        device_fn.pallas_internal_scratch_storage_names[storage] = scratch_name
