"""Triton/NVSHMEM lowering for Helion communication primitives."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

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
_NVSHMEM_CMP_NE = 1


@dataclass
class _TritonRemoteCopyInfo:
    start_statement: ast.stmt
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

    device_fn = state.device_function
    device_fn.requires_nvshmem = True
    CompileEnvironment.current().has_barrier = True
    state.codegen.add_statement(statement_from_string("nvshmem.barrier_all()"))
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
    scratch = device_fn.new_var("remote_copy_scratch")
    device_fn.wrapper_only_params.append(scratch)
    device_fn.triton_remote_copy_scratch_args.append(scratch)
    try:
        scratch_like = device_fn.tensor_arg(dst).host_str()
    except KeyError as error:
        raise exc.BackendUnsupported(
            "triton",
            "remote copies from computed values require a host-provided "
            "symmetric destination tensor",
        ) from error
    device_fn.triton_remote_copy_scratch_specs.append((scratch_like, host_max_numel))

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
        mask_terms.append(f"({index}) < ({logical_size})")
    offset = " + ".join(offset_terms) or "0"
    mask = " & ".join(mask_terms)
    base = f"{scratch} + ({_FLAT_PROGRAM_ID}) * ({max_numel})"
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
        start_statement = statement_from_string(
            "nvshmem.putmem_signal_block("
            f"{dst_region.pointer}, {src_ptr}, "
            f"tl.cast(({numel}) * {src.element_size()}, tl.int64), "
            f"{signal_name}, tl.cast(1, tl.uint64), "
            f"{_NVSHMEM_SIGNAL_ADD}, {{device_id}})",
            device_id=device_id,
            **src_placeholders,
            **dst_region.placeholders,
        )
    else:
        start_statement = statement_from_string(
            f"nvshmem.put({dst_region.pointer}, {src_ptr}, {numel}, {{device_id}})",
            device_id=device_id,
            **src_placeholders,
            **dst_region.placeholders,
        )
    device_fn.remote_copy_descriptors[descriptor_id] = _TritonRemoteCopyInfo(
        start_statement=start_statement,
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
    return _emit_statements(state, [*info.source_materialization, info.start_statement])


def _paired_signal(state: CodegenState) -> str:
    signal = _paired_copy_info(state).signal
    if not isinstance(signal, str):
        raise exc.InternalError(
            RuntimeError("remote-copy receive wait could not resolve its signal")
        )
    return signal


def _receive_wait_statements(state: CodegenState) -> list[ast.stmt]:
    signal = _paired_signal(state)
    return [
        statement_from_string(
            f"nvshmem.signal_wait_until({signal}, {_NVSHMEM_CMP_NE}, 0)"
        ),
        statement_from_string(
            f"tl.atomic_add({signal}, -1, sem='relaxed', scope='sys')"
        ),
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
