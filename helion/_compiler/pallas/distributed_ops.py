"""Pallas lowering for Helion's one-sided communication primitives."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import inspect
from typing import TYPE_CHECKING
from typing import Callable

import torch

from ... import exc
from ...language import _decorators
from ...language.distributed_ops import _REMOTE_COPY_DESCRIPTOR_ID_META
from ...language.distributed_ops import make_async_remote_copy
from ...language.distributed_ops import remote_barrier
from ...language.distributed_ops import start_async_remote_copy_descriptor
from ...language.distributed_ops import wait_async_remote_copy
from ...language.distributed_ops import wait_recv_async_remote_copy
from ...language.distributed_ops import wait_send_async_remote_copy
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from . import codegen as pallas_codegen
from .plan_tiling import REMOTE_DST_INDEXING_PATTERNS
from .plan_tiling import REMOTE_SRC_INDEXING_PATTERNS

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState
    from ..tile_strategy import ForiLoopState
    from ..tile_strategy import RemoteRecvDrain


_PALLAS_SRC_MATERIALIZATION_META = "_helion_pallas_remote_copy_src_materialization"


def _automatic_collective_id(fn: Callable[..., object]) -> int:
    """Return a stable best-effort barrier namespace for a Helion kernel.

    The ID must agree across independent TorchTPU host processes, so Python's
    randomized hash and process-local allocation order are unsuitable. Source
    text distinguishes changed or separately defined kernels while the
    qualified name provides a stable fallback when source is unavailable.
    """
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        source = ""
    identity = f"{fn.__module__}\0{fn.__qualname__}\0{source}".encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:4], "big") & 0x7FFFFFFF


@dataclass
class _PallasRemoteCopyInfo:
    op_name: str
    source_materialization: tuple[ast.AST, ast.AST] | None
    deferred_recv: RemoteRecvDrain | None = None


@_decorators.codegen(remote_barrier, "pallas")
def _(state: CodegenState) -> ast.AST:
    proxy_device_ids = state.proxy_arg(0)
    ast_device_ids = state.ast_args[0]
    device_ids: list[int | ast.AST]
    if isinstance(proxy_device_ids, list):
        assert isinstance(ast_device_ids, list)
        device_ids = ast_device_ids
    elif isinstance(proxy_device_ids, torch.Tensor):
        assert isinstance(ast_device_ids, ast.AST)
        if proxy_device_ids.ndim == 0:
            device_ids = [ast_device_ids]
        elif proxy_device_ids.ndim == 1 and isinstance(proxy_device_ids.shape[0], int):
            device_ids = [
                expr_from_string(f"{{device_ids}}[{index}]", device_ids=ast_device_ids)
                for index in range(proxy_device_ids.shape[0])
            ]
        else:
            raise exc.TypeInferenceError(
                "remote_barrier expects a scalar or statically sized 1-D peer tensor"
            )
    elif isinstance(proxy_device_ids, int):
        device_ids = [proxy_device_ids]
    else:
        raise exc.TypeInferenceError(
            "remote_barrier expects a logical peer, a 1-D peer tensor, or a list"
        )

    if not device_ids:
        return expr_from_string("None")

    device_fn = state.device_function
    device_fn.requires_collective_id = True
    barrier = device_fn.new_var("remote_barrier", dce=False)
    state.codegen.add_statement(
        statement_from_string(f"{barrier} = pltpu.get_barrier_semaphore()")
    )
    peers: list[ast.AST] = []
    for device_id in device_ids:
        if isinstance(device_id, int):
            device_id = expr_from_string(repr(device_id))
        assert isinstance(device_id, ast.AST)
        peers.append(device_id)
        state.codegen.add_statement(
            statement_from_string(
                f"pl.semaphore_signal({barrier}, inc=1, "
                "device_id={device_id}, "
                "device_id_type=pl.DeviceIdType.LOGICAL)",
                device_id=device_id,
            )
        )
    state.codegen.add_statement(
        statement_from_string(f"pl.semaphore_wait({barrier}, {len(peers)})")
    )

    second_barrier_fn = device_fn.new_var("remote_barrier_phase_2", dce=False)
    placeholders = {f"device_id_{index}": peer for index, peer in enumerate(peers)}
    body = [f"def {second_barrier_fn}(second_barrier):"]
    for index in range(len(peers)):
        body.extend(
            (
                "    pl.semaphore_signal(",
                "        second_barrier, inc=1,",
                f"        device_id={{device_id_{index}}},",
                "        device_id_type=pl.DeviceIdType.LOGICAL,",
                "    )",
            )
        )
    body.append(f"    pl.semaphore_wait(second_barrier, {len(peers)})")
    state.codegen.add_statement(statement_from_string("\n".join(body), **placeholders))
    state.codegen.add_statement(
        statement_from_string(
            f"pl.run_scoped({second_barrier_fn}, pltpu.SemaphoreType.REGULAR)"
        )
    )
    return expr_from_string("None")


def _direct_ref_expr(tensor: ast.AST, ast_index: list[object]) -> ast.AST:
    if not ast_index:
        return tensor
    placeholders: dict[str, ast.AST] = {"tensor": tensor}
    parts = []
    for index, value in enumerate(ast_index):
        if isinstance(value, int):
            value = expr_from_string(repr(value))
        assert isinstance(value, ast.AST)
        key = f"index_{index}"
        placeholders[key] = value
        parts.append(f"{{{key}}}")
    return expr_from_string(f"{{tensor}}.at[{', '.join(parts)}]", **placeholders)


def _direct_value_expr(tensor: ast.AST, ast_index: list[object]) -> ast.AST:
    """Index an ordinary JAX value with the same leading indices as a Ref."""
    if not ast_index:
        return tensor
    placeholders: dict[str, ast.AST] = {"tensor": tensor}
    parts = []
    for index, value in enumerate(ast_index):
        if isinstance(value, int):
            value = expr_from_string(repr(value))
        assert isinstance(value, ast.AST)
        key = f"index_{index}"
        placeholders[key] = value
        parts.append(f"{{{key}}}")
    return expr_from_string(f"{{tensor}}[{', '.join(parts)}]", **placeholders)


def _canonical_recv_ref(dst_ref: ast.AST, indexed_dims: int) -> ast.AST | None:
    """Return a fixed same-shaped destination region for a receive drain."""
    if not isinstance(dst_ref, ast.Subscript):
        return None
    raw_parts = (
        dst_ref.slice.elts if isinstance(dst_ref.slice, ast.Tuple) else [dst_ref.slice]
    )
    if len(raw_parts) < indexed_dims:
        return None
    placeholders: dict[str, ast.AST] = {"base": dst_ref.value}
    rendered: list[str] = []
    for index, part in enumerate(raw_parts):
        key = f"part_{index}"
        if index < indexed_dims:
            if (
                isinstance(part, ast.Call)
                and isinstance(part.func, ast.Attribute)
                and part.func.attr == "ds"
                and len(part.args) >= 2
            ):
                part = expr_from_string("pl.ds(0, {size})", size=part.args[1])
            else:
                part = expr_from_string("0")
        placeholders[key] = part
        rendered.append(f"{{{key}}}")
    return expr_from_string(f"{{base}}[{', '.join(rendered)}]", **placeholders)


def _active_fori_loop(state: CodegenState) -> ForiLoopState | None:
    from ..tile_strategy import ForiLoopState

    return next(
        (
            loop
            for loops in state.codegen.active_device_loops.values()
            for loop in loops
            if isinstance(loop, ForiLoopState)
        ),
        None,
    )


def _materialized_source_value(
    state: CodegenState,
    tensor_ast: ast.AST,
    ast_index: list[object],
) -> ast.AST:
    """Select a computed source region without dynamic value slicing.

    A send-ring is commonly spelled by expanding one computed tile across a
    leading slot dimension and indexing that dimension with the current slot.
    The selected value is independent of the slot, so materialize the original
    pre-broadcast tile directly. Pallas otherwise sees a dynamic slice of an
    ordinary JAX value, which TPU lowering does not support.
    """
    if not ast_index or state.fx_node is None:
        return _direct_value_expr(tensor_ast, ast_index)
    src_node = state.fx_node.args[0]
    if (
        not isinstance(src_node, torch.fx.Node)
        or src_node.op != "call_function"
        or src_node.target is not torch.ops.aten.expand.default
        or not src_node.args
        or not isinstance(src_node.args[0], torch.fx.Node)
    ):
        return _direct_value_expr(tensor_ast, ast_index)

    expanded_input = src_node.args[0]
    expanded = src_node.meta.get("val")
    unexpanded = expanded_input.meta.get("val")
    if not isinstance(expanded, torch.Tensor) or not isinstance(
        unexpanded, torch.Tensor
    ):
        return _direct_value_expr(tensor_ast, ast_index)
    indexed_dims = len(ast_index)
    if expanded.ndim != unexpanded.ndim or indexed_dims > expanded.ndim:
        return _direct_value_expr(tensor_ast, ast_index)

    from ..compile_environment import CompileEnvironment

    env = CompileEnvironment.current()
    if not all(
        env.known_equal(unexpanded.shape[dim], 1)
        and (env.known_equal(expanded.shape[dim], 1) or expanded.stride(dim) == 0)
        for dim in range(indexed_dims)
    ):
        return _direct_value_expr(tensor_ast, ast_index)
    if not all(
        env.known_equal(unexpanded.shape[dim], expanded.shape[dim])
        for dim in range(indexed_dims, expanded.ndim)
    ):
        return _direct_value_expr(tensor_ast, ast_index)

    from ...language import view_ops

    if (
        expanded_input.op == "call_function"
        and expanded_input.target is view_ops.subscript
        and len(expanded_input.args) >= 2
        and isinstance(expanded_input.args[0], torch.fx.Node)
        and isinstance(expanded_input.args[1], (list, tuple))
    ):
        base_node = expanded_input.args[0]
        indices = expanded_input.args[1]
        if (
            len(indices) == expanded.ndim
            and all(index is None for index in indices[:indexed_dims])
            and all(
                isinstance(index, slice)
                and index.start is None
                and index.stop is None
                and index.step is None
                for index in indices[indexed_dims:]
            )
        ):
            base_ast = state.env.get(base_node)
            if isinstance(base_ast, ast.AST):
                return base_ast

    unexpanded_ast = state.env.get(expanded_input)
    if isinstance(unexpanded_ast, ast.AST):
        return _direct_value_expr(unexpanded_ast, [0] * indexed_dims)
    return _direct_value_expr(tensor_ast, ast_index)


def _physical_tile_shape(
    state: CodegenState, tensor: torch.Tensor
) -> tuple[int | torch.SymInt, ...]:
    """Resolve block-symbol dimensions to this config's physical tile sizes."""
    from ..compile_environment import CompileEnvironment

    env = CompileEnvironment.current()
    shape: list[int | torch.SymInt] = []
    for size in tensor.shape:
        block_id = env.resolve_block_id(size)
        resolved = (
            state.device_function.resolved_block_size(block_id)
            if block_id is not None
            else None
        )
        shape.append(resolved if resolved is not None else size)
    return tuple(shape)


def _tensor_argument_name(state: CodegenState, tensor: torch.Tensor) -> str | None:
    """Return a kernel argument name, or ``None`` for a computed device value."""
    from ..host_function import HostFunction

    if tensor not in HostFunction.current().tensor_to_origin:
        return None
    return state.device_function.tensor_arg(tensor).name


def _remote_ref_expr(
    state: CodegenState,
    tensor: torch.Tensor,
    tensor_ast: object,
    proxy_index: object,
    ast_index: object,
    metadata_key: str,
    *,
    materialize_device_value: bool = False,
) -> ast.AST:
    """Resolve a logical remote-copy region against the active VMEM block."""
    assert isinstance(proxy_index, (list, tuple))
    assert isinstance(ast_index, list)
    name = _tensor_argument_name(state, tensor)
    if name is None:
        if not materialize_device_value:
            raise exc.TypeInferenceError(
                "Pallas remote-copy destinations must be kernel tensor arguments"
            )
        # A computed tile is a JAX value, while TPU DMA requires a Ref. Allocate
        # one reusable VMEM slot now, but defer populating it until descriptor
        # start so a preceding wait can safely release the previous transfer.
        assert isinstance(tensor_ast, ast.AST)
        name = state.device_function.register_scratch(
            _physical_tile_shape(state, tensor),
            tensor.dtype,
            name_hint="remote_src",
        )
        scratch_ref = _direct_ref_expr(expr_from_string(name), ast_index)
        source_value = _materialized_source_value(state, tensor_ast, ast_index)
        assert state.fx_node is not None
        state.fx_node.meta[_PALLAS_SRC_MATERIALIZATION_META] = (
            scratch_ref,
            source_value,
        )
        return scratch_ref

    from ..device_function import PallasMemorySpace

    active_name = pallas_codegen.vmem_name(state, name)
    if (
        active_name == name
        and state.device_function.pallas_memory_space.get(id(tensor))
        == PallasMemorySpace.HBM
    ):
        if not proxy_index:
            return expr_from_string(name)
        assert state.fx_node is not None
        patterns = state.fx_node.meta.get(metadata_key)
        assert isinstance(patterns, list)
        parts, none_dims = pallas_codegen.index_parts(
            state,
            proxy_index,
            tensor,
            indexing_patterns=patterns,
            ast_subscripts=ast_index,
            pipeline_scalar_indices_local=False,
            tensor_indices_are_scalars=True,
            raw_hbm_ref=True,
        )
        assert not none_dims
        return expr_from_string(f"{name}.at[{', '.join(parts)}]")
    name = active_name
    if not proxy_index:
        return expr_from_string(name)

    assert state.fx_node is not None
    patterns = state.fx_node.meta.get(metadata_key)
    assert isinstance(patterns, list)
    parts, none_dims = pallas_codegen.index_parts(
        state,
        proxy_index,
        tensor,
        indexing_patterns=patterns,
        ast_subscripts=ast_index,
        # Remote-copy indices may select an arbitrary slot within a full VMEM
        # block. Only dimensions tied to the active tile become local offsets.
        pipeline_scalar_indices_local=False,
        tensor_indices_are_scalars=True,
    )
    assert not none_dims
    return expr_from_string(f"{name}.at[{', '.join(parts)}]")


def _make_remote_copy(state: CodegenState) -> ast.AST:
    src = state.proxy_arg(0)
    dst = state.proxy_arg(3)
    assert isinstance(src, torch.Tensor)
    assert isinstance(dst, torch.Tensor)

    device_fn = state.device_function
    device_fn.requires_remote_copy = True
    proxy_src_index = state.proxy_arg(1)
    ast_src_index = state.ast_args[1]
    assert isinstance(proxy_src_index, (list, tuple))
    assert isinstance(ast_src_index, list)

    assert state.fx_node is not None
    descriptor_id = state.fx_node.meta.get(_REMOTE_COPY_DESCRIPTOR_ID_META)
    if not isinstance(descriptor_id, int):
        raise exc.InternalError(RuntimeError("remote-copy descriptor has no ID"))

    computed_source = _tensor_argument_name(state, src) is None
    send_slot_shape: tuple[int, ...] = ()
    if computed_source and ast_src_index:
        try:
            indexed_shape = tuple(int(size) for size in src.shape[: len(ast_src_index)])
        except (TypeError, ValueError):
            indexed_shape = ()
        if indexed_shape and all(size > 0 for size in indexed_shape):
            send_slot_shape = indexed_shape
    send_sem_name = device_fn.register_dma_semaphore(
        name_hint="remote_send_sem",
        shape=send_slot_shape,
    )
    send_sem = (
        _direct_ref_expr(expr_from_string(send_sem_name), ast_src_index)
        if send_slot_shape
        else expr_from_string(send_sem_name)
    )
    try:
        payload_shape = tuple(int(size) for size in src.shape[len(proxy_src_index) :])
    except (TypeError, ValueError):
        payload_shape = ()
    fori_loop = _active_fori_loop(state)
    dst_name: str | None = None
    drain_key: tuple[str, tuple[int, ...]] | None = None
    deferred_recv: RemoteRecvDrain | None = None
    if fori_loop is not None and payload_shape:
        dst_name = _tensor_argument_name(state, dst)
        if dst_name is not None:
            read_names, _ = device_fn.get_tensor_read_write_names()
            if dst_name not in read_names:
                drain_key = (dst_name, payload_shape)
                deferred_recv = fori_loop._remote_recv_drains.get(drain_key)
    recv_sem = (
        deferred_recv.semaphore
        if deferred_recv is not None
        else device_fn.register_dma_semaphore(name_hint="remote_recv_sem")
    )
    src_ref = _remote_ref_expr(
        state,
        src,
        state.ast_args[0],
        proxy_src_index,
        ast_src_index,
        REMOTE_SRC_INDEXING_PATTERNS,
        materialize_device_value=True,
    )
    materialization = state.fx_node.meta.get(_PALLAS_SRC_MATERIALIZATION_META)
    if materialization is not None:
        assert isinstance(materialization, tuple)
    op_name = device_fn.new_var("remote_copy", dce=False)
    dst_ref = _remote_ref_expr(
        state,
        dst,
        state.ast_args[3],
        state.proxy_arg(4),
        state.ast_args[4],
        REMOTE_DST_INDEXING_PATTERNS,
    )
    if fori_loop is not None and drain_key is not None and deferred_recv is None:
        ast_dst_index = state.ast_args[4]
        assert isinstance(ast_dst_index, list)
        canonical_ref = _canonical_recv_ref(dst_ref, len(ast_dst_index))
        if canonical_ref is not None:
            from ..tile_strategy import RemoteRecvDrain

            deferred_recv = RemoteRecvDrain(recv_sem, canonical_ref)
            fori_loop._remote_recv_drains[drain_key] = deferred_recv
    device_id = state.ast_args[2]
    if isinstance(device_id, int):
        device_id = expr_from_string(repr(device_id))
    assert isinstance(device_id, ast.AST)

    state.codegen.add_statement(
        statement_from_string(
            f"{op_name} = pltpu.make_async_remote_copy("
            "{src_ref}, {dst_ref}, "
            "{send_sem}, "
            f"{recv_sem}, "
            "device_id={device_id}, "
            "device_id_type=pl.DeviceIdType.LOGICAL)",
            src_ref=src_ref,
            dst_ref=dst_ref,
            send_sem=send_sem,
            device_id=device_id,
        )
    )
    device_fn.remote_copy_descriptors[descriptor_id] = _PallasRemoteCopyInfo(
        op_name=op_name,
        source_materialization=materialization,
        deferred_recv=deferred_recv,
    )
    return expr_from_string(op_name)


@_decorators.codegen(make_async_remote_copy, "pallas")
def _(state: CodegenState) -> ast.AST:
    return _make_remote_copy(state)


def _paired_copy_info(state: CodegenState) -> _PallasRemoteCopyInfo:
    descriptor_id = state.proxy_arg(0)
    if not isinstance(descriptor_id, int):
        raise exc.InternalError(
            RuntimeError("remote-copy lifecycle operation has no descriptor ID")
        )
    info = state.device_function.remote_copy_descriptors.get(descriptor_id)
    if not isinstance(info, _PallasRemoteCopyInfo):
        raise exc.InternalError(
            RuntimeError(
                "remote-copy lifecycle operation could not resolve its descriptor"
            )
        )
    return info


def _emit_source_materialization(
    state: CodegenState, info: _PallasRemoteCopyInfo
) -> bool:
    materialization = info.source_materialization
    if materialization is None:
        return False
    target, value = materialization
    assert isinstance(target, ast.AST)
    assert isinstance(value, ast.AST)
    state.codegen.add_statement(
        statement_from_string(
            "{target}[...] = jnp.pad("
            "{value}, tuple((0, dst - src) for src, dst in "
            "zip({value}.shape, {target}.shape, strict=True)))",
            target=target,
            value=value,
        )
    )
    return True


def _emit_wait(state: CodegenState, method: str) -> ast.AST:
    info = _paired_copy_info(state)
    if method == "start":
        _emit_source_materialization(state, info)
        if info.deferred_recv is not None:
            fori_loop = _active_fori_loop(state)
            if fori_loop is None:
                info.deferred_recv = None
            elif state.codegen.statements_stack[-1] is fori_loop.inner_statements:
                info.deferred_recv.starts_per_iteration += 1
            else:
                counter = info.deferred_recv.dynamic_start_counter
                if counter is None:
                    counter = state.device_function.register_scratch(
                        (128,), torch.int32, name_hint="remote_recv_count"
                    )
                    info.deferred_recv.dynamic_start_counter = counter
                    fori_loop.outer_prefix.append(
                        statement_from_string(
                            f"{counter}[...] = jnp.zeros_like({counter}[...])"
                        )
                    )
                state.codegen.add_statement(
                    statement_from_string(
                        f"{counter}[...] = {counter}[...] + "
                        f"jnp.ones_like({counter}[...])"
                    )
                )
    if method == "wait_recv" and info.deferred_recv is not None:
        info.deferred_recv.waits_deferred = True
    else:
        state.codegen.add_statement(statement_from_string(f"{info.op_name}.{method}()"))
    return expr_from_string("None")


@_decorators.codegen(start_async_remote_copy_descriptor, "pallas")
def _(state: CodegenState) -> ast.AST:
    return _emit_wait(state, "start")


@_decorators.codegen(wait_async_remote_copy, "pallas")
def _(state: CodegenState) -> ast.AST:
    return _emit_wait(state, "wait")


@_decorators.codegen(wait_send_async_remote_copy, "pallas")
def _(state: CodegenState) -> ast.AST:
    return _emit_wait(state, "wait_send")


@_decorators.codegen(wait_recv_async_remote_copy, "pallas")
def _(state: CodegenState) -> ast.AST:
    return _emit_wait(state, "wait_recv")
