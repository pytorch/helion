"""Plan body-local tensors that can remain entirely in TPU scratch memory."""

from __future__ import annotations

import ast
import math
from typing import cast

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
) -> tuple[set[int], set[int], set[int], set[int]]:
    read: set[int] = set()
    written: set[int] = set()
    remote: set[int] = set()
    nonscalar: set[int] = set()
    env = CompileEnvironment.current()

    def record_access(node: torch.fx.Node, storage: int) -> None:
        tensor_node = node.args[0]
        assert isinstance(tensor_node, torch.fx.Node)
        tensor = tensor_node.meta["val"]
        indices = node.args[1]
        if not isinstance(indices, (list, tuple)) or len(indices) != tensor.ndim:
            nonscalar.add(storage)
            return
        for index in indices:
            if isinstance(index, torch.fx.Node):
                index = index.meta.get("val")
            if isinstance(index, int):
                continue
            if isinstance(index, torch.SymInt):
                # Tile indices are SymInts too; only size-one tiles are scalar.
                block_id = env.get_block_id(index)
                if (
                    block_id is None
                    or env.block_sizes[block_id].from_config(device_fn.config) == 1
                ):
                    continue
            elif isinstance(index, torch.Tensor) and index.ndim == 0:
                continue
            nonscalar.add(storage)
            return

    for graph_info in device_fn.codegen.codegen_graphs:
        for node in graph_info.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target is memory_ops.load:
                if (storage := _tensor_storage(node.args[0])) is not None:
                    read.add(storage)
                    record_access(node, storage)
            elif node.target is memory_ops.store:
                if (storage := _tensor_storage(node.args[0])) is not None:
                    written.add(storage)
                    record_access(node, storage)
            elif node.target in ATOMIC_OPS:
                if (storage := _tensor_storage(node.args[0])) is not None:
                    read.add(storage)
                    written.add(storage)
                    nonscalar.add(storage)
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
    return read, written, remote, nonscalar


def plan_internal_scratch(*, include_local_temporaries: bool = False) -> None:
    """Place private, body-local buffers in on-chip scratch.

    An allocation must be a top-level ``torch.empty``-family call, have a fully
    static shape, be both read and written, and not feed the host return.
    Remote-copy buffers are always eligible. Other local temporaries are
    eligible only when ``include_local_temporaries`` is requested by the config.
    """
    host = HostFunction.current()
    device_fn = DeviceFunction.current()
    allocation_names = _top_level_empty_names(host) - _returned_name_dependencies(host)
    if not allocation_names:
        return

    read, written, remote, nonscalar = _accessed_storages(device_fn)
    eligible_storages = read & written
    if not include_local_temporaries:
        eligible_storages &= remote
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
        ):
            continue
        tensors_by_storage.setdefault(storage, []).append(tensor)
        names_by_storage[storage] = name

    for storage, tensors in tensors_by_storage.items():
        from .tracing_ops import _resolve_dim_size

        resolved = []
        for tensor in tensors:
            shape = tuple(
                _resolve_dim_size(size, CompileEnvironment.current(), device_fn.config)
                for size in tensor.shape
            )
            if all(size is not None for size in shape):
                concrete_shape = tuple(cast("int", size) for size in shape)
                resolved.append((tensor, concrete_shape))
        if not resolved:
            continue
        representative, shape = max(
            resolved,
            key=lambda item: (item[0].ndim, math.prod(item[1])),
        )
        scratch_name = device_fn.register_scratch(
            shape,
            representative.dtype,
            name_hint=names_by_storage[storage],
            scratch_type=(
                "smem"
                if storage not in remote
                and storage not in nonscalar
                and representative.dtype in (torch.int32, torch.float32)
                else "vmem"
            ),
        )
        device_fn.pallas_internal_scratch_storage_names[storage] = scratch_name
