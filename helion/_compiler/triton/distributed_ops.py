"""Triton/NVSHMEM lowering for Helion communication primitives."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.fx.experimental.symbolic_shapes import guard_int

from ... import exc
from ...language import _decorators
from ...language.distributed_ops import _REMOTE_COPY_DESCRIPTOR_ID_META
from ...language.distributed_ops import _REMOTE_COPY_DESCRIPTOR_OPS_META
from ...language.distributed_ops import make_async_remote_copy
from ...language.distributed_ops import remote_barrier
from ...language.distributed_ops import start_async_remote_copy_descriptor
from ...language.distributed_ops import wait_async_remote_copy
from ...language.distributed_ops import wait_recv_async_remote_copy
from ...language.distributed_ops import wait_send_async_remote_copy
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string

if TYPE_CHECKING:
    from ..device_function import DeviceFunction
    from ..inductor_lowering import CodegenState


_FLAT_PROGRAM_ID = (
    "tl.program_id(2) * tl.num_programs(1) * tl.num_programs(0)"
    " + tl.program_id(1) * tl.num_programs(0) + tl.program_id(0)"
)
_NUM_PROGRAMS = "tl.num_programs(2) * tl.num_programs(1) * tl.num_programs(0)"
# Values from NVSHMEM's nvshmemx_signal_op_t and nvshmem_cmp_t.
_NVSHMEM_SIGNAL_ADD = 10


def _remote_barrier_peers(state: CodegenState) -> list[ast.AST]:
    proxy_device_ids = state.proxy_arg(0)
    ast_device_ids = state.ast_args[0]
    if isinstance(proxy_device_ids, list):
        assert isinstance(ast_device_ids, list)
        peers: list[ast.AST] = []
        for proxy_peer, ast_peer in zip(proxy_device_ids, ast_device_ids, strict=True):
            if isinstance(proxy_peer, torch.Tensor) and proxy_peer.ndim != 0:
                raise exc.TypeInferenceError(
                    "remote_barrier expects scalar peers in a Python list"
                )
            if isinstance(ast_peer, int):
                ast_peer = expr_from_string(repr(ast_peer))
            assert isinstance(ast_peer, ast.AST)
            peers.append(ast_peer)
        return peers
    if isinstance(proxy_device_ids, int):
        return [expr_from_string(repr(proxy_device_ids))]
    if not isinstance(proxy_device_ids, torch.Tensor):
        raise exc.TypeInferenceError(
            "remote_barrier expects a logical peer, a 1-D peer tensor, or a list"
        )
    if not isinstance(ast_device_ids, ast.AST):
        raise exc.InternalError(RuntimeError("remote_barrier has no device expression"))
    if proxy_device_ids.ndim == 0:
        return [ast_device_ids]
    if proxy_device_ids.ndim != 1:
        raise exc.TypeInferenceError(
            "remote_barrier expects a scalar or statically sized 1-D peer tensor"
        )
    peer_count = guard_int(proxy_device_ids.shape[0])
    physical_size = 1 << (peer_count - 1).bit_length() if peer_count else 1
    return [
        expr_from_string(
            "tl.sum(tl.where("
            f"tl.arange(0, {physical_size}) == {index}, "
            "{device_ids}, 0))",
            device_ids=ast_device_ids,
        )
        for index in range(peer_count)
    ]


def _reserve_remote_barrier_signal_slots(
    device_fn: DeviceFunction,
) -> tuple[str, int]:
    signal_arg = device_fn.triton_remote_barrier_signal_arg
    if signal_arg is None:
        signal_arg = device_fn.new_var("remote_barrier_signal")
        device_fn.wrapper_only_params.append(signal_arg)
        device_fn.triton_remote_barrier_signal_arg = signal_arg
    slot = device_fn.triton_remote_barrier_signal_slots
    device_fn.triton_remote_barrier_signal_slots += 2
    return signal_arg, slot


@dataclass
class _TritonRemoteCopyInfo:
    start_statements: list[ast.stmt]
    signal: str | None
    source_materialization: list[ast.stmt]


@dataclass
class _TritonRegion:
    pointer: str
    placeholders: dict[str, ast.AST]
    sizes: list[str]
    max_sizes: list[str]
    host_max_sizes: list[str]
    is_tiled: bool

    @property
    def numel(self) -> str:
        return _product(self.sizes)


def _product(values: list[str]) -> str:
    return " * ".join(values) or "1"


@_decorators.codegen(remote_barrier, "triton")
def _(state: CodegenState) -> ast.AST:
    from ..compile_environment import CompileEnvironment

    peers = _remote_barrier_peers(state)
    if not peers:
        return expr_from_string("None")
    device_fn = state.device_function
    device_fn.requires_nvshmem = True
    signal_arg, signal_slot = _reserve_remote_barrier_signal_slots(device_fn)
    signals = [
        device_fn.new_var("remote_barrier_signal", dce=False),
        device_fn.new_var("remote_barrier_signal", dce=False),
    ]
    for phase, signal in enumerate(signals):
        state.codegen.add_statement(
            statement_from_string(
                f"{signal} = {signal_arg} + "
                f"({signal_slot + phase}) * ({_NUM_PROGRAMS}) + "
                f"({_FLAT_PROGRAM_ID})"
            )
        )
    statements = [
        statement_from_string("nvshmem.quiet()"),
        statement_from_string("tl.debug_barrier()"),
    ]
    for signal in signals:
        statements.extend(
            statement_from_string(
                "helion_dist_utils._publish_signal("
                f"{signal}, tl.cast(1, tl.int64), {_NVSHMEM_SIGNAL_ADD}, "
                "{peer}, nvshmem.my_pe())",
                peer=peer,
            )
            for peer in peers
        )
        statements.append(
            statement_from_string(
                "helion_dist_utils._wait_and_consume_signal("
                f"{signal}, tl.cast({len(peers)}, tl.int64))"
            )
        )
    statements.append(statement_from_string("tl.debug_barrier()"))
    for statement in statements:
        state.codegen.add_statement(statement)
    CompileEnvironment.current().has_barrier = True
    return expr_from_string("None")


def _index_expr(
    state: CodegenState,
    proxy_index: object,
    index_ast: object,
    tensor: torch.Tensor,
    prefix: str,
) -> tuple[str, dict[str, ast.AST], list[str], list[str]]:
    from ..compile_environment import CompileEnvironment

    assert isinstance(proxy_index, (list, tuple))
    assert isinstance(index_ast, (list, tuple))
    assert len(proxy_index) == len(index_ast)
    device_fn = state.device_function
    env = CompileEnvironment.current()
    placeholders: dict[str, ast.AST] = {}
    terms: list[str] = []
    tile_extents: list[str] = []
    tile_max_extents: list[str] = []
    for position, (proxy, index) in enumerate(zip(proxy_index, index_ast, strict=True)):
        block_id = env.get_block_id(proxy) if isinstance(proxy, torch.SymInt) else None
        if block_id is not None:
            assert state.fx_node is not None
            block_id = env.resolve_codegen_block_id(
                block_id, state.codegen, state.fx_node.graph
            )
        if block_id is not None and state.codegen.active_device_loops.get(block_id):
            # Tile arguments trace as their block-size symbol. Recover the live
            # tile offset here, just as ordinary tensor indexing does.
            index_expr = state.codegen.offset_var(block_id)
            block_size = device_fn.block_size_var(block_id) or "1"
            tensor_size = device_fn.tensor_size(tensor, position).name
            tile_extents.append(
                f"tl.minimum(({block_size}), ({tensor_size}) - ({index_expr}))"
            )
            tile_max_extents.append(block_size)
        elif isinstance(index, int):
            index_expr = repr(index)
        else:
            assert isinstance(index, ast.AST)
            name = f"{prefix}{position}"
            placeholders[name] = index
            index_expr = f"{{{name}}}"
        stride = device_fn.tensor_stride(tensor, position).name
        terms.append(f"({index_expr}) * {stride}")
    return (
        " + ".join(terms) or "0",
        placeholders,
        tile_extents,
        tile_max_extents,
    )


def _region_ptr(
    state: CodegenState,
    tensor: torch.Tensor,
    proxy_index: object,
    index_ast: object,
    prefix: str,
) -> _TritonRegion:
    from ..device_function import TensorSizeArg

    device_fn = state.device_function
    base = device_fn.tensor_arg(tensor).name
    offset, placeholders, tile_extents, tile_max_extents = _index_expr(
        state, proxy_index, index_ast, tensor, prefix
    )
    assert isinstance(index_ast, (list, tuple))
    suffix_args = [
        device_fn.tensor_size(tensor, dim) for dim in range(len(index_ast), tensor.ndim)
    ]
    return _TritonRegion(
        pointer=f"{base} + {offset}",
        placeholders=placeholders,
        sizes=[*tile_extents, *(arg.name for arg in suffix_args)],
        max_sizes=[*tile_max_extents, *(arg.name for arg in suffix_args)],
        host_max_sizes=[
            *tile_max_extents,
            *(
                arg.host_str() if isinstance(arg, TensorSizeArg) else arg.name
                for arg in suffix_args
            ),
        ],
        is_tiled=bool(tile_extents),
    )


def _reserve_remote_copy_scratch(
    state: CodegenState,
    like: torch.Tensor,
    max_numel: str,
    host_max_numel: str,
) -> str:
    device_fn = state.device_function
    scratch = device_fn.new_var("remote_copy_scratch")
    device_fn.wrapper_only_params.append(scratch)
    device_fn.triton_remote_copy_scratch_args.append(scratch)
    try:
        scratch_like = device_fn.tensor_arg(like).host_str()
    except KeyError as error:
        raise exc.BackendUnsupported(
            "triton",
            "remote-copy scratch requires a host-provided symmetric tensor",
        ) from error
    device_fn.triton_remote_copy_scratch_specs.append((scratch_like, host_max_numel))
    return f"{scratch} + ({_FLAT_PROGRAM_ID}) * ({max_numel})"


def _computed_source_scratch(
    state: CodegenState,
    src: torch.Tensor,
    dst: torch.Tensor,
    dst_region: _TritonRegion,
) -> tuple[str, str, list[ast.stmt]]:
    """Materialize a computed Triton tile into contiguous global memory."""
    device_fn = state.device_function
    src_value = state.ast_args[0]
    assert isinstance(src_value, ast.AST)

    physical_sizes = [device_fn.literal_expr(size) for size in src.shape]
    if len(physical_sizes) != len(dst_region.sizes):
        raise exc.BackendUnsupported(
            "triton",
            "computed remote-copy sources must have the same rank as the "
            "destination region",
        )
    max_numel = _product(dst_region.max_sizes)
    host_max_numel = _product(dst_region.host_max_sizes)
    base = _reserve_remote_copy_scratch(
        state,
        dst,
        max_numel,
        host_max_numel,
    )

    offset_terms: list[str] = []
    mask_terms: list[str] = []
    for dim, (physical_size, logical_size) in enumerate(
        zip(physical_sizes, dst_region.sizes, strict=True)
    ):
        index_shape = ["1"] * len(physical_sizes)
        index_shape[dim] = physical_size
        index = f"tl.reshape(tl.arange(0, {physical_size}), [{', '.join(index_shape)}])"
        stride = _product(dst_region.sizes[dim + 1 :])
        offset_terms.append(f"({index}) * ({stride})")
        mask_terms.append(f"(({index}) < ({logical_size}))")
    offset = " + ".join(offset_terms) or "0"
    mask = " & ".join(mask_terms)
    store = (
        f"tl.store({base} + ({offset}), {{src_value}}, mask={mask})"
        if mask
        else f"tl.store({base}, {{src_value}})"
    )
    materialization = [
        statement_from_string(
            store,
            src_value=src_value,
        ),
        # NVSHMEM's block put reads the tile cooperatively after all Triton
        # lanes have populated their portion of the global scratch buffer.
        statement_from_string("tl.debug_barrier()"),
    ]
    return base, dst_region.numel, materialization


def _shares_storage(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    return lhs.untyped_storage()._cdata == rhs.untyped_storage()._cdata


def _has_receive_wait(node: torch.fx.Node) -> bool:
    descriptor_ops = node.meta.get(_REMOTE_COPY_DESCRIPTOR_OPS_META, set())
    assert isinstance(descriptor_ops, set)
    return bool(descriptor_ops & {wait_async_remote_copy, wait_recv_async_remote_copy})


def _reserve_signal_slot(
    device_fn: DeviceFunction, dst: torch.Tensor
) -> tuple[str, int]:
    signal_arg = device_fn.triton_remote_copy_signal_arg
    if signal_arg is None:
        try:
            dst_host_str = device_fn.tensor_arg(dst).host_str()
        except (KeyError, RuntimeError) as error:
            raise exc.BackendUnsupported(
                "triton",
                "remote-copy receive completion requires a host-provided "
                "symmetric destination tensor",
            ) from error
        signal_arg = device_fn.new_var("remote_copy_signal")
        device_fn.wrapper_only_params.append(signal_arg)
        device_fn.triton_remote_copy_signal_arg = signal_arg
        device_fn.triton_remote_copy_signal_dst = dst_host_str

    slot = device_fn.triton_remote_copy_signal_slots
    device_fn.triton_remote_copy_signal_slots += 1
    return signal_arg, slot


def _prepare_remote_copy(state: CodegenState) -> ast.AST:
    src = state.proxy_arg(0)
    dst = state.proxy_arg(3)
    assert isinstance(src, torch.Tensor)
    assert isinstance(dst, torch.Tensor)

    device_fn = state.device_function
    device_fn.requires_nvshmem = True
    device_fn.requires_remote_copy = True
    assert state.fx_node is not None
    signal_info = (
        _reserve_signal_slot(device_fn, dst)
        if _has_receive_wait(state.fx_node)
        else None
    )
    try:
        dst_region = _region_ptr(
            state,
            dst,
            state.proxy_arg(4),
            state.ast_args[4],
            "_remote_dst_index",
        )
    except (KeyError, RuntimeError) as error:
        raise exc.BackendUnsupported(
            "triton",
            "remote-copy destinations must be host-provided symmetric tensors",
        ) from error
    source_materialization: list[ast.stmt] = []
    src_region: _TritonRegion | None = None
    try:
        src_region = _region_ptr(
            state,
            src,
            state.proxy_arg(1),
            state.ast_args[1],
            "_remote_src_index",
        )
    except KeyError:
        if state.proxy_arg(1) not in ([], ()):
            raise exc.BackendUnsupported(
                "triton",
                "computed remote-copy sources must be copied in full",
            ) from None
        src_ptr, src_numel, source_materialization = _computed_source_scratch(
            state, src, dst, dst_region
        )
        src_placeholders = {}
        src_is_tiled = False
    else:
        src_ptr = src_region.pointer
        src_placeholders = src_region.placeholders
        src_numel = src_region.numel
        src_is_tiled = src_region.is_tiled
    numel = (
        f"tl.minimum(({src_numel}), ({dst_region.numel}))"
        if src_is_tiled or dst_region.is_tiled
        else src_numel
    )
    if signal_info is not None and src_region is not None and _shares_storage(src, dst):
        # Peers may write the destination while this rank is still reading the
        # same allocation. Snapshot every rank's source before any put begins.
        scratch_src = _reserve_remote_copy_scratch(
            state,
            dst,
            _product(src_region.max_sizes),
            _product(src_region.host_max_sizes),
        )
        source_materialization.extend(
            [
                statement_from_string(
                    f"nvshmem.get({scratch_src}, {src_ptr}, {numel}, nvshmem.my_pe())",
                    **src_placeholders,
                ),
                statement_from_string("tl.debug_barrier()"),
                statement_from_string("nvshmem.sync_all()"),
            ]
        )
        src_ptr = scratch_src
        src_placeholders = {}
        from ..compile_environment import CompileEnvironment

        CompileEnvironment.current().has_barrier = True
    device_id = state.ast_args[2]
    if isinstance(device_id, int):
        device_id = expr_from_string(repr(device_id))
    assert isinstance(device_id, ast.AST)

    descriptor_id = state.fx_node.meta.get(_REMOTE_COPY_DESCRIPTOR_ID_META)
    if not isinstance(descriptor_id, int):
        raise exc.InternalError(RuntimeError("remote-copy descriptor has no ID"))
    signal_name = None
    if signal_info is not None:
        signal_arg, signal_slot = signal_info
        signal_name = device_fn.new_var("remote_signal", dce=False)
        state.codegen.add_statement(
            statement_from_string(
                f"{signal_name} = {signal_arg} + "
                f"({signal_slot}) * ({_NUM_PROGRAMS}) + ({_FLAT_PROGRAM_ID})"
            )
        )
        start_statements = [
            statement_from_string(
                "helion_dist_utils._nvshmem_put_signal_nbi_block("
                f"{dst_region.pointer}, {src_ptr}, {numel}, {signal_name}, "
                f"tl.cast(1, tl.uint64), {_NVSHMEM_SIGNAL_ADD}, {{device_id}}, "
                "nvshmem.my_pe())",
                device_id=device_id,
                **src_placeholders,
                **dst_region.placeholders,
            )
        ]
    else:
        start_statements = [
            statement_from_string(
                "helion_dist_utils._nvshmem_put_nbi_block("
                f"{dst_region.pointer}, {src_ptr}, {numel}, {{device_id}}, "
                "nvshmem.my_pe())",
                device_id=device_id,
                **src_placeholders,
                **dst_region.placeholders,
            )
        ]
    device_fn.remote_copy_descriptors[descriptor_id] = _TritonRemoteCopyInfo(
        start_statements=start_statements,
        signal=signal_name,
        source_materialization=source_materialization,
    )
    return expr_from_string("None")


@_decorators.codegen(make_async_remote_copy, "triton")
def _(state: CodegenState) -> ast.AST:
    return _prepare_remote_copy(state)


def _paired_copy_info(state: CodegenState) -> _TritonRemoteCopyInfo:
    descriptor_id = state.proxy_arg(0)
    if not isinstance(descriptor_id, int):
        raise exc.InternalError(
            RuntimeError("remote-copy lifecycle operation has no descriptor ID")
        )
    info = state.device_function.remote_copy_descriptors.get(descriptor_id)
    if not isinstance(info, _TritonRemoteCopyInfo):
        raise exc.InternalError(
            RuntimeError(
                "remote-copy lifecycle operation could not resolve its descriptor"
            )
        )
    return info


def _emit_statements(
    state: CodegenState,
    statements: list[ast.stmt],
) -> ast.AST:
    for statement in statements:
        state.codegen.add_statement(statement)
    return expr_from_string("None")


@_decorators.codegen(start_async_remote_copy_descriptor, "triton")
def _(state: CodegenState) -> ast.AST:
    info = _paired_copy_info(state)
    return _emit_statements(
        state, [*info.source_materialization, *info.start_statements]
    )


def _receive_wait_statements(state: CodegenState) -> list[ast.stmt]:
    info = _paired_copy_info(state)
    signal = info.signal
    if not isinstance(signal, str):
        raise exc.InternalError(
            RuntimeError("remote-copy receive wait could not resolve its signal")
        )
    return [
        statement_from_string(
            "helion_dist_utils._wait_and_consume_signal("
            f"{signal}, tl.cast(1, tl.int64))"
        )
    ]


@_decorators.codegen(wait_async_remote_copy, "triton")
def _(state: CodegenState) -> ast.AST:
    return _emit_statements(
        state,
        [statement_from_string("nvshmem.quiet()"), *_receive_wait_statements(state)],
    )


@_decorators.codegen(wait_send_async_remote_copy, "triton")
def _(state: CodegenState) -> ast.AST:
    return _emit_statements(state, [statement_from_string("nvshmem.quiet()")])


@_decorators.codegen(wait_recv_async_remote_copy, "triton")
def _(state: CodegenState) -> ast.AST:
    return _emit_statements(state, _receive_wait_statements(state))
