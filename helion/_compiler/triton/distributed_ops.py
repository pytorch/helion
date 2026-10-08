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
from ..ast_extension import create
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..cross_loop_codegen import peer_state
from ..host_function import HostFunction
from ..indexing_strategy import SubscriptIndexing
from ..indexing_strategy import _in_warp_specialized_loop
from ..tile_dependency import TILE_ACCESS_META
from ..tile_strategy import DeviceLoopState

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..cross_loop_codegen import PeerState
    from ..cross_loop_codegen import Scatter
    from ..device_function import DeviceFunction
    from ..helper_function import CodegenInterface
    from ..inductor_lowering import CodegenState
    from ..tile_dependency import TileAccess

    StoreCodegen = Callable[
        [
            CodegenState,
            torch.Tensor,
            list[object],
            ast.AST,
            ast.AST | None,
            ast.AST | None,
        ],
        ast.AST,
    ]
    LoadCodegen = Callable[
        [
            CodegenState,
            torch.Tensor,
            list[object],
            ast.AST | None,
            ast.AST | None,
            ast.AST | None,
        ],
        ast.AST,
    ]


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
        *device_fn.async_store_drain(),
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
    statements.extend(device_fn.async_load_fence())
    for statement in statements:
        state.codegen.add_statement(statement)
    state.device_function.has_barrier = True
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
        state.device_function.has_barrier = True
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
    # Every thread reads the source, so other threads' TMA stores to it must be
    # complete first. A CTA barrier only, so has_barrier stays unset.
    sync: list[ast.stmt] = []
    if drain := state.device_function.async_store_drain():
        sync = [*drain, statement_from_string("tl.debug_barrier()")]
    return _emit_statements(
        state, [*sync, *info.source_materialization, *info.start_statements]
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
        ),
        *state.device_function.async_load_fence(),
    ]


@_decorators.codegen(wait_async_remote_copy, "triton")
def _(state: CodegenState) -> ast.AST:
    return _emit_statements(
        state,
        [statement_from_string("nvshmem.quiet()"), *_receive_wait_statements(state)],
    )


@_decorators.codegen(wait_send_async_remote_copy, "triton")
def _(state: CodegenState) -> ast.AST:
    # Order the completed source reads before later TMA writes to the source.
    return _emit_statements(
        state,
        [
            statement_from_string("nvshmem.quiet()"),
            *state.device_function.async_load_fence(),
        ],
    )


@_decorators.codegen(wait_recv_async_remote_copy, "triton")
def _(state: CodegenState) -> ast.AST:
    return _emit_statements(state, _receive_wait_statements(state))


@dataclass(frozen=True)
class ScatterPoll:
    """How a poll of an inband scatter reads a row no task of ``source`` wrote."""

    allocation_id: int
    source: str
    key: str
    # The tagged word, every lane of it, in the peer's own buffer.
    word: str


@dataclass(frozen=True)
class InbandPoll:
    """A mailbox read whose wait is deferred to the first use of any poll."""

    node: torch.fx.Node
    address: str
    mask: str | None
    # Polls of one address shape share an asm; None polls alone.
    group: str | None
    pack: int
    # Threads that may wait together on a CTA barrier; None in WS loops.
    threads: int | None
    word: str
    result: ast.stmt
    scatter: ScatterPoll | None = None

    def load(self, epoch: str) -> ast.stmt:
        # A plain volatile load vectorizes; masked lanes read as current.
        mask = "" if self.mask is None else f", {self.mask}, {epoch} << 32"
        return statement_from_string(
            f"{self.word} = tl.load({self.address}{mask}, volatile=True)"
        )


def _inband_access(state: CodegenState) -> tuple[TileAccess, tuple[int, ...]] | None:
    """The access an inband push or poll emits, and the node's access ids."""
    graph = HostFunction.current().device_ir.tile_dependency_graph
    assert state.fx_node is not None
    ids = state.fx_node.meta.get(TILE_ACCESS_META, ())
    if graph is None or not ids or not graph.is_inband(graph.accesses[ids[0]]):
        return None
    return graph.accesses[ids[0]], ids


def _bits(dtype: torch.dtype) -> str:
    return f"tl.uint{dtype.itemsize * 8}"


def _inline_asm(
    outputs: str, asm: str, constraints: str, args: list[str], dtype: str, pack: int
) -> ast.stmt:
    return statement_from_string(
        f"{outputs} = tl.inline_asm_elementwise(asm={{asm}}, "
        f"constraints={constraints!r}, args=[{', '.join(args)}], "
        f"dtype={dtype}, is_pure=False, pack={pack})",
        asm=create(ast.Constant, value=asm),
    )


def _lift(
    state: CodegenState, template: str, x: ast.AST | str, prefix: str = "inband_lane"
) -> str:
    """Name ``template`` with ``{x}`` filled in, as a dead-code-eliminable var."""
    if isinstance(x, str):
        x = expr_from_string(x)
    return state.codegen.lift(
        expr_from_string(template.replace("{value}", "{x}"), x=x),
        dce=True,
        prefix=prefix,
    ).id


def _interleaved(
    state: CodegenState, tensor: str, dims: list[str], count: int
) -> list[str]:
    """Split ``tensor``'s last axis into ``count`` parts; part i holds the
    elements at i mod ``count``."""
    if count == 1:
        return [tensor]
    levels = count.bit_length() - 1
    shape = ", ".join([*dims[:-1], f"{dims[-1]} // {count}", *["2"] * levels])
    parts = [_lift(state, f"tl.reshape({{x}}, [{shape}])", tensor)]
    for _ in range(levels):
        halves: list[str] = []
        for part in parts:
            lo = state.device_function.new_var("inband_lane")
            hi = state.device_function.new_var("inband_lane")
            state.add_statement(statement_from_string(f"{lo}, {hi} = tl.split({part})"))
            halves += [lo, hi]
        parts = halves
    # Each split peels the lowest index bit, so parts come out bit-reversed.
    return [parts[int(f"{i:0{levels}b}"[::-1], 2)] for i in range(count)]


def _group_start(
    state: CodegenState, term: str, output_size: list[int | torch.SymInt], count: int
) -> str | None:
    """A per-dim index or mask ``term`` at every ``count``-th last-axis element.

    Splitting only the last axis' vector skips the layout conversions Triton
    emits when it splits a full tile. None when ``term``'s shape is unknown.
    """
    tile_strategy = state.tile_strategy
    dims = tile_strategy.shape_dims(output_size)
    last = tile_strategy.expand_str(output_size, len(output_size) - 1)
    if len(dims) != len(output_size) or len(dims) < 2:
        return None
    if term.endswith("None]"):
        return term
    if not term.endswith(last):
        return None
    return f"({_interleaved(state, term[: -len(last)], dims[-1:], count)[0]}){last}"


def _start_offset(
    state: CodegenState,
    fake_tensor: torch.Tensor,
    indexing: SubscriptIndexing,
    output_size: list[int | torch.SymInt],
    count: int,
) -> str | None:
    """The offset of every ``count``-th last-axis element, or None."""
    terms = []
    for dim, (term, block) in enumerate(
        zip(indexing.dim_index_exprs, indexing.block_dims, strict=True)
    ):
        if CompileEnvironment.current().known_equal(fake_tensor.size(dim), 1):
            continue
        start = _group_start(state, term, output_size, count) if block else term
        if start is None:
            return None
        stride = state.device_function.tensor_stride(fake_tensor, dim).name
        terms.append(f"{start} * {stride}")
    return " + ".join(terms) or None


def _start_mask(
    state: CodegenState,
    indexing: SubscriptIndexing,
    extra_mask: ast.AST | None,
    output_size: list[int | torch.SymInt],
    count: int,
) -> str | None:
    """The mask of every ``count``-th last-axis element, or None."""
    assert state.fx_node is not None
    terms, pending = [], [indexing.mask_expr]
    while pending:
        node = pending.pop()
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
            pending += [node.left, node.right]
            continue
        term = ast.unparse(node)
        if extra_mask is not None and term == ast.unparse(extra_mask):
            # The extra mask broadcasts unchanged if it is constant on the last axis.
            mask_node = state.fx_node.args[3]
            assert isinstance(mask_node, torch.fx.Node)
            fake = mask_node.meta["val"]
            env = CompileEnvironment.current()
            if fake.ndim and not env.known_equal(fake.size(-1), 1):
                return None
        else:
            term = _group_start(state, term, output_size, count)
            if term is None:
                return None
        terms.append(f"({term})")
    return " & ".join(terms)


def inband_store_codegen(
    state: CodegenState, codegen_store: StoreCodegen
) -> StoreCodegen:
    """Make an inband store also push its tile, epoch tagged, to every rank."""
    inband = _inband_access(state)
    if inband is None:
        return codegen_store
    access, ids = inband

    def push(
        state: CodegenState,
        fake_tensor: torch.Tensor,
        subscript: list[object],
        value: ast.AST,
        extra_mask: ast.AST | None,
        cache_modifier: ast.AST | None,
    ) -> ast.AST:
        device_function = state.device_function
        if not isinstance(value, ast.Constant):
            value = state.codegen.lift(value, dce=True, prefix="inband_value")
        # The local store registers its tensor, which the peer state may need.
        local_store = codegen_store(
            state, fake_tensor, subscript, value, extra_mask, cache_modifier
        )
        peer = peer_state(device_function)
        assert peer is not None
        backend = CompileEnvironment.current().backend
        indexing = SubscriptIndexing.create(state, fake_tensor, subscript, extra_mask)
        output_size = SubscriptIndexing.compute_shape(fake_tensor, subscript, state)
        word_value = value
        if not indexing.block_shaped_offset and output_size:
            word_value = expr_from_string(
                backend.reshape_expr("{value}", "[]"), value=value
            )
        offset = indexing.index_expr
        if indexing.needs_broadcast():
            offset = expr_from_string(
                backend.broadcast_to_expr(
                    "{offset}", state.tile_strategy.shape_str(output_size)
                ),
                offset=offset,
            )
        offset = state.codegen.lift(offset, dce=True, prefix="inband_offset")
        dtype = fake_tensor.dtype
        _, _, lanes, pair = peer.mailboxes[access.allocation_id]
        bits = (
            f"tl.cast(tl.cast({{value}}, {backend.dtype_str(dtype)}), "
            f"{_bits(dtype)}, bitcast=True)"
        )
        epoch = f"({peer.epoch} << 32)"
        mask = indexing.mask_expr if indexing.has_mask() else None
        if lanes == 1 and not pair:
            words = [
                state.codegen.lift(
                    expr_from_string(
                        f"tl.cast({bits}, tl.uint64) | {epoch}", value=word_value
                    ),
                    dce=True,
                    prefix="inband_word",
                ).id
            ]
            address = offset.id
            masks = []
            if mask is not None:
                masks = [_lift(state, "tl.cast({x}, tl.int32)", mask, "inband_mask")]
        else:
            # Pack row-adjacent elements into each word, then pair adjacent words.
            assert indexing.block_shaped_offset
            dims = state.tile_strategy.shape_dims(output_size)
            shape = f"[{', '.join(dims)}]"
            group = lanes * (2 if pair else 1)
            full = _lift(state, backend.broadcast_to_expr(bits, shape), value)
            payload = " | ".join(
                f"(tl.cast({part}, tl.uint64) << {8 * dtype.itemsize * i})"
                for i, part in enumerate(_interleaved(state, full, dims, lanes))
            )
            word = _lift(state, f"{{x}} | {epoch}", expr_from_string(payload))
            row = [*dims[:-1], f"{dims[-1]} // {lanes}"]
            words = _interleaved(state, word, row, 2) if pair else [word]
            stored = f"[{', '.join([*row[:-1], f'{dims[-1]} // {group}'])}]"
            first = _start_offset(state, fake_tensor, indexing, output_size, group)
            if first is None:
                first = _interleaved(state, offset.id, dims, group)[0]
            else:
                first = backend.broadcast_to_expr(f"({first})", stored)
            address = _lift(state, f"{{x}} // {lanes}", expr_from_string(first))
            masks = []
            start_mask = None
            if mask is not None:
                start_mask = _start_mask(
                    state, indexing, extra_mask, output_size, group
                )
            if start_mask is not None:
                cast = f"tl.cast({start_mask}, tl.int32)"
                cast = backend.broadcast_to_expr(cast, stored)
                lifted = state.codegen.lift(
                    expr_from_string(cast), dce=True, prefix="inband_mask"
                )
                masks = [lifted.id]
            elif mask is not None:
                cast = backend.broadcast_to_expr("tl.cast({x}, tl.int32)", shape)
                masks = _interleaved(state, _lift(state, cast, mask), dims, group)[:1]
        store = (
            "st.relaxed.sys.global.v2.u64 [$1], {$2, $3};"
            if pair
            else "st.relaxed.sys.global.u64 [$1], $2;"
        )
        if masks:
            store = (
                f"{{ .reg .pred p; setp.ne.b32 p, ${2 + len(words)}, 0; @p {store} }}"
            )
        slot = peer.mailbox(access.allocation_id, peer.rank)
        # Each word is single-copy atomic and carries its epoch: no fence needed.
        # One asm per peer keeps that peer's stores adjacent, which is faster.
        for base in peer.bases:
            state.add_statement(
                _inline_asm(
                    device_function.new_var("inband_push", dce=False),
                    f"{store} mov.u32 $0, 0;",
                    "=r,l" + ",l" * len(words) + ",r" * len(masks),
                    [f"{base} + ({slot}) + {address}", *words, *masks],
                    "tl.int32",
                    1,
                )
            )
        if access.allocation_id in peer.scatters:
            group = lanes * (2 if pair else 1)
            _record_scatter(state, access, indexing, output_size, group, words, masks)
        device_function.inband_access_ids.update(ids)
        return local_store

    return push


def _record_scatter(
    state: CodegenState,
    access: TileAccess,
    indexing: SubscriptIndexing,
    output_size: list[int | torch.SymInt],
    group: int,
    words: list[str],
    masks: list[str],
) -> None:
    """Record the row blocks (keys) each lane of this task wrote, tagged.

    The root's last scatter store then pushes the task's done word to every
    rank. It reduces the records first, so all threads have pushed by then.
    """
    device_function = state.device_function
    peer = device_function.peer_state
    assert peer is not None and state.fx_node is not None
    env = CompileEnvironment.current()
    scatter = peer.scatters[access.allocation_id]
    root = peer.scatter_roots[scatter.root]
    terms = [
        _group_start(state, term, output_size, group) if block and group > 1 else term
        for term, block in zip(
            indexing.dim_index_exprs, indexing.block_dims, strict=True
        )
    ]
    rows = terms[scatter.position]
    if any(term is None for term in terms) or rows is None:
        raise exc.CrossLoopSchedulingError("because an inband scatter tile is ragged")
    # Each lane's first element of its block keys the block; masks are per row.
    expand = rows[rows.rindex(")") + 1 :] if scatter.lanes > 1 else ""
    lane = f"(tl.arange(0, {scatter.lanes})){expand}" if scatter.lanes > 1 else "0"
    row_dim = None
    if scatter.lanes > 1:
        entries = (
            [entry.strip() for entry in expand[1:-1].split(",")] if expand else [":"]
        )
        row_dim = entries.index(":")
    extra = state.fx_node.args[3]
    if isinstance(extra, torch.fx.Node):
        fake = extra.meta["val"]
        if any(
            dim + len(output_size) - fake.ndim != row_dim
            and not env.known_equal(size, 1)
            for dim, size in enumerate(fake.shape)
        ):
            raise exc.CrossLoopSchedulingError(
                "because an inband scatter's mask must depend on its row alone"
            )
    key = f"tl.cast({rows}, tl.int64) * {scatter.key_strides[scatter.position]}"
    first = []
    for dim, (term, extent, stride) in enumerate(
        zip(terms, scatter.extents, scatter.key_strides, strict=True)
    ):
        if dim == scatter.position:
            continue
        if extent > 1:
            first.append(f"((({term}) % {extent}) == 0)")
        if extent < scatter.shape[dim]:
            key += f" + tl.cast(({term}) // {extent}, tl.int64) * {stride}"
    bit = f"tl.cast({masks[0]} != 0, tl.uint64)" if masks else "1"
    task = root.task(
        {
            block_id: f"{state.codegen.offset_var(block_id)} // "
            f"{env.size_hint(env.block_sizes[block_id].from_config_assert(state.config))}"
            for block_id in root.axis_order
        }
    )
    parity = f"tl.cast({peer.epoch} & 1, tl.int64)"
    address = (
        f"{peer.state} + {scatter.records} + {parity} * "
        f"{root.tasks * scatter.lanes} + ({task}) * {scatter.lanes} + {lane}"
    )
    # A masked row's key may be negative; 31 bits keep it out of the tag.
    word = (
        f"({peer.epoch} << 32) | ((tl.cast({key}, tl.uint64) & 0x7FFFFFFF) << 1)"
        f" | {bit}"
    )
    predicate = " & ".join(first)
    # The anchor operand keeps the record in the pushing WS partition.
    store = "st.relaxed.sys.global.u64 [$1], $2;"
    args = [address, word, words[0]]
    if predicate:
        store = f"setp.ne.b32 p, $4, 0; @p {store}"
        args.append(f"tl.cast({predicate}, tl.int32)")
    record = device_function.new_var("inband_record", dce=False)
    state.add_statement(
        _inline_asm(
            record,
            f"{{ .reg .pred p; .reg .b64 a; mov.b64 a, $3; {store} mov.u32 $0, 0; }}",
            "=r,l,l,l" + (",r" if predicate else ""),
            args,
            "tl.int32",
            1,
        )
    )
    if access.access_id != root.last:
        return
    done = device_function.new_var("inband_done", dce=False)
    predicate = " & ".join([*first, f"({lane} == 0)"])
    stores = " ".join(
        f"@p st.relaxed.sys.global.u64 [${i + 1}], ${len(peer.bases) + 1};"
        for i in range(len(peer.bases))
    )
    world = len(peer.bases)
    state.add_statement(
        _inline_asm(
            done,
            f"{{ .reg .pred p; setp.ne.b32 p, ${world + 2}, 0; {stores} "
            "mov.u32 $0, 0; }",
            "=r" + ",l" * (world + 1) + ",r",
            [
                *(
                    f"{base} + {root.done} + {parity} * {world * root.tasks} + "
                    f"{peer.rank} * {root.tasks} + ({task})"
                    for base in peer.bases
                ),
                f"({peer.epoch} << 32) | tl.cast(tl.sum({record}), tl.uint64)",
                f"tl.cast({predicate}, tl.int32)",
            ],
            "tl.int32",
            1,
        )
    )
    device_function.inband_scatter_done[scatter.root] = done


def _poll_pack(
    state: CodegenState, output_size: list[int | torch.SymInt], lanes: int
) -> int:
    """Words per asm lane: more than a thread's words would fault."""
    env = CompileEnvironment.current()
    numel = 1
    for size in output_size:
        block_id = None if isinstance(size, int) else env.resolve_block_id(size)
        if block_id is not None:
            # A global block size is a SymInt; read its hint, as int() would pin it.
            size = env.size_hint(
                env.block_sizes[block_id].from_config_assert(state.config)
            )
        if not isinstance(size, int):
            return 1
        numel *= size
    numel //= lanes
    threads = 32 * state.config.num_warps
    return 4 if numel >= 4 * threads else 2 if numel >= 2 * threads else 1


def _joined(parts: list[str]) -> str:
    """Invert ``_interleaved``: interleave the parts along a new last axis."""
    while len(parts) > 1:
        half = len(parts) // 2
        parts = [f"tl.join({parts[i]}, {parts[i + half]})" for i in range(half)]
    return parts[0]


def _word_poll(
    state: CodegenState,
    fake_tensor: torch.Tensor,
    indexing: SubscriptIndexing,
    output_size: list[int | torch.SymInt],
    extra_mask: ast.AST | None,
    access: TileAccess,
    lanes: int,
) -> tuple[str, str | None, str] | None:
    """Word offsets, mask and shape of a block poll whose last axis is aligned
    runs of the mailbox's ``lanes``-element words, else None."""
    env = CompileEnvironment.current()
    graph = HostFunction.current().device_ir.tile_dependency_graph
    assert graph is not None

    def block_size(block_id: int) -> int:
        return env.size_hint(env.block_sizes[block_id].from_config_assert(state.config))

    tile_strategy = state.tile_strategy
    dims = tile_strategy.shape_dims(output_size)
    if (
        not indexing.block_shaped_offset
        or indexing.needs_broadcast()
        or len(dims) != len(output_size)
        or graph.inband_row(access, block_size) % lanes
    ):
        return None
    last = tile_strategy.expand_str(output_size, len(output_size) - 1)
    *rest, (term, block) = zip(
        indexing.dim_index_exprs, indexing.block_dims, strict=True
    )
    # Only the last dim may vary along the last axis.
    if not block or not term.endswith(last):
        return None
    if any(block and not term.endswith("None]") for term, block in rest):
        return None
    mask = None
    if indexing.has_mask():
        mask = _start_mask(state, indexing, extra_mask, output_size, lanes)
        if mask is None:
            return None
    vector = term[: len(term) - len(last)]
    rest.append((f"({_interleaved(state, vector, dims[-1:], lanes)[0]}){last}", True))
    terms = [
        f"{term} * {state.device_function.tensor_stride(fake_tensor, dim).name}"
        for dim, (term, _) in enumerate(rest)
        if dim == len(rest) - 1 or not env.known_equal(fake_tensor.size(dim), 1)
    ]
    shape = ", ".join([*dims[:-1], f"{dims[-1]} // {lanes}"])
    return f"({' + '.join(terms)}) // {lanes}", mask, f"[{shape}]"


def inband_load_codegen(state: CodegenState, codegen_load: LoadCodegen) -> LoadCodegen:
    """Make an inband peer load poll this rank's own mailbox instead."""
    inband = _inband_access(state)
    if inband is None:
        return codegen_load
    access, ids = inband

    def poll(
        state: CodegenState,
        fake_tensor: torch.Tensor,
        subscript: list[object],
        extra_mask: ast.AST | None,
        eviction_policy: ast.AST | None,
        cache_modifier: ast.AST | None,
    ) -> ast.AST:
        device_function = state.device_function
        peer = peer_state(device_function)
        assert peer is not None and state.fx_node is not None
        env = CompileEnvironment.current()
        backend = env.backend
        indexing = SubscriptIndexing.create(state, fake_tensor, subscript, extra_mask)
        output_size = SubscriptIndexing.compute_shape(fake_tensor, subscript, state)
        shape = state.tile_strategy.shape_str(output_size)
        offset = indexing.index_expr
        if indexing.has_mask() and not indexing.block_shaped_offset:
            offset = expr_from_string(
                f"{{offset}} + {backend.zeros_expr(shape, env.index_type())}",
                offset=offset,
            )
        slot = peer.mailbox(access.allocation_id, access.owner_rank)
        _, _, lanes, _ = peer.mailboxes[access.allocation_id]
        dtype = fake_tensor.dtype
        scatter = peer.scatters.get(access.allocation_id)
        words = None
        if lanes > 1 and scatter is None:
            words = _word_poll(
                state, fake_tensor, indexing, output_size, extra_mask, access, lanes
            )
        shift = ""
        element = ""
        if words is not None:
            offset = expr_from_string(words[0])
        elif lanes > 1 or scatter is not None:
            element = state.codegen.lift(offset, dce=True, prefix="inband_offset").id
            offset = expr_from_string(element)
        if lanes > 1 and words is None:
            # Each lane polls the word holding its element and shifts it out.
            shift = (
                f" >> (tl.cast({element} % {lanes}, tl.uint64) * {8 * dtype.itemsize})"
            )
            offset = expr_from_string(f"{element} // {lanes}")
        address = state.codegen.lift(
            expr_from_string(f"{peer.state} + ({slot}) + {{offset}}", offset=offset),
            dce=True,
            prefix="inband_address",
        )
        mask = None
        if indexing.has_mask():
            mask_expr = indexing.mask_expr
            if words is not None:
                assert words[1] is not None
                mask_expr = expr_from_string(words[1])
            mask = state.codegen.lift(mask_expr, dce=True, prefix="inband_mask").id
        word = device_function.new_var("inband_word", dce=False)
        result = expr_from_string(
            f"tl.cast(tl.cast({word}{shift}, {_bits(dtype)}), "
            f"{backend.dtype_str(dtype)}, bitcast=True)"
        )
        if words is not None:
            # A thread unpacks its whole words, keeping each row's lanes in-thread.
            parts = [
                f"tl.cast(tl.cast({word}{f' >> {8 * dtype.itemsize * i}' * (i > 0)}, "
                f"{_bits(dtype)}), {backend.dtype_str(dtype)}, bitcast=True)"
                for i in range(lanes)
            ]
            result = expr_from_string(f"tl.reshape({_joined(parts)}, {shape})")
        # Triton's pipeliner would make the first read an early, weak cp.async.
        for loops in state.codegen.active_device_loops.values():
            for loop in loops:
                call = loop.for_node.iter if isinstance(loop, DeviceLoopState) else None
                if isinstance(call, ast.Call) and ast.unparse(call.func) == "tl.range":
                    call.keywords = [
                        *(kw for kw in call.keywords if kw.arg != "num_stages"),
                        create(ast.keyword, arg="num_stages", value=ast.Constant(1)),
                    ]
        group, pack, threads = None, 1, None
        if indexing.needs_broadcast():
            result = expr_from_string(
                backend.broadcast_to_expr("{value}", shape), value=result
            )
        elif indexing.block_shaped_offset or mask is not None:
            group = shape if words is None else words[2]
            pack = _poll_pack(state, output_size, 1 if words is None else lanes)
            if not _in_warp_specialized_loop(state):
                threads = 32 * state.config.num_warps
        else:
            group = "scalar"
        fallback = None
        if scatter is not None:
            other = "" if mask is None else f", mask={mask}, other=0"
            tensor = device_function.tensor_arg(fake_tensor).name
            bits = (
                f"tl.load({tensor}.to(tl.pointer_type(tl.uint32)) + {element} // {lanes}"
                f"{other}, volatile=True)"
                if lanes > 1
                else f"tl.cast(tl.load({tensor} + {element}{other}, volatile=True), "
                f"{_bits(dtype)}, bitcast=True)"
            )
            fallback = ScatterPoll(
                allocation_id=access.allocation_id,
                source=f"({access.owner_rank})",
                key=_scatter_key(scatter, element),
                word=f"({peer.epoch} << 32) | tl.cast({bits}, tl.uint64)",
            )
        value = device_function.new_var("inband_value", dce=True)
        device_function.inband_polls.append(
            InbandPoll(
                node=state.fx_node,
                address=address.id,
                mask=mask,
                group=group,
                pack=pack,
                threads=threads,
                word=word,
                result=statement_from_string(f"{value} = {{result}}", result=result),
                scatter=fallback,
            )
        )
        device_function.inband_access_ids.update(ids)
        return create(ast.Name, id=value, ctx=ast.Load())

    return poll


# Spins before a scatter read checks whether every source task is done.
_SCATTER_SPINS = 256


def _scatter_key(scatter: Scatter, element: str) -> str:
    """The key (aligned row block) of a row-major element offset."""
    terms = []
    stride = 1
    for dim in reversed(range(len(scatter.shape))):
        size, extent = scatter.shape[dim], scatter.extents[dim]
        if extent < size:
            term = f"({element} // {stride})" if stride > 1 else element
            term = f"({term} % {size})" if dim > 0 else term
            term = f"({term} // {extent})" if extent > 1 else term
            terms.append(f"{term} * {scatter.key_strides[dim]}")
        stride *= size
    return " + ".join(terms) or "0"


def _poll_asm(polls: int, pack: int) -> tuple[str, str]:
    """Reload only the stale words of a first load until all are current.

    Operands: words, first-load words, addresses, then the tag.
    """
    count = polls * pack
    check = [
        f"mov.b64 {{lo, hi}}, ${word}; setp.ne.b32 p, hi, ${3 * count + word % pack}; "
        f"@p ld.volatile.global.b64 ${word}, [${2 * count + word}]; "
        "@p mov.u32 stale, 1;"
        for word in range(count)
    ]
    asm = " ".join(
        [
            "{ .reg .pred p; .reg .b32 lo, hi, stale;",
            *(f"mov.b64 ${word}, ${count + word};" for word in range(count)),
            "SPIN${:uid}: mov.u32 stale, 0;",
            *check,
            "setp.ne.u32 p, stale, 0; @p bra SPIN${:uid}; }",
        ]
    )
    return asm, ",".join(["=l"] * count + ["l"] * 2 * count + ["r"] * pack)


def _stale(peer_epoch: str, polls: list[InbandPoll]) -> str:
    """Whether any thread of the CTA holds a stale word of a block poll."""
    tag = f"tl.cast({peer_epoch}, tl.uint32)"
    flags = " + ".join(
        "tl.sum(("
        + " | ".join(
            f"((({poll.word} >> 32).to(tl.uint32)) != {tag})" for poll in group
        )
        + ").to(tl.int32))"
        for group in _by_group(polls).values()
    )
    asm = (
        "{ .reg .pred p, q; setp.ne.s32 p, $1, 0; "
        f"bar.red.or.pred q, 0, {polls[0].threads}, p; selp.s32 $0, 1, 0, q; }}"
    )
    return (
        f"tl.inline_asm_elementwise(asm={asm!r}, constraints='=r,r', args=[{flags}], "
        "dtype=tl.int32, is_pure=False, pack=1)"
    )


def _scatter_fallback(
    cg: CodegenInterface, peer: PeerState, scattered: list[InbandPoll]
) -> list[ast.stmt]:
    """After a long spin, copy rows no task wrote from the peer's own buffer.

    A source's records and skip marks cover every key it wrote once ``ready``.
    The copy lands in this rank's mailbox, so the loop's words keep one layout;
    a consumer on the source flags the next launch to hold for every rank.
    """
    if not scattered:
        return []
    device_function = cg.device_function
    spins = device_function.new_var("inband_spins", dce=False)
    cg.add_statement(statement_from_string(f"{spins} = 0"))
    ready: dict[tuple[int, str], str] = {}
    cover: list[ast.stmt] = []
    synthesize: list[ast.stmt] = []
    for poll in scattered:
        assert poll.scatter is not None
        scatter = peer.scatters[poll.scatter.allocation_id]
        source = poll.scatter.source
        if (poll.scatter.allocation_id, source) not in ready:
            flag = device_function.new_var("inband_ready", dce=False)
            ready[poll.scatter.allocation_id, source] = flag
            cg.add_statement(statement_from_string(f"{flag} = 0"))
            root = peer.scatter_roots[scatter.root]
            cover.append(
                statement_from_string(
                    f"{flag} = helion_dist_utils._scatter_cover({peer.state}, "
                    f"{peer.ptrs}, {source}, {peer.epoch}, {scatter.records}, "
                    f"{root.done}, {scatter.cover}, {peer.votes}, {root.tasks}, "
                    f"{scatter.lanes}, {scatter.keys}, {len(peer.bases)}, "
                    f"{poll.threads})"
                )
            )
        key = poll.scatter.key
        mask = "" if poll.mask is None else f", mask={poll.mask}, other=0"
        uncovered = device_function.new_var("inband_uncovered", dce=False)
        synthesize.extend(
            [
                statement_from_string(
                    f"{uncovered} = ({ready[poll.scatter.allocation_id, source]} != 0)"
                    f" & (tl.load({peer.state} + {scatter.cover} + {source} * "
                    f"{scatter.keys} + {key}{mask}, volatile=True) != {peer.epoch})"
                    + ("" if poll.mask is None else f" & {poll.mask}")
                ),
                statement_from_string(
                    f"tl.store({poll.address}, {poll.scatter.word}, mask={uncovered})"
                ),
                statement_from_string(
                    f"tl.store({peer.state} + {peer.flag} + ({key}) * 0, "
                    f"{peer.epoch} + 1, mask={uncovered} & ({source} == {peer.rank}))"
                ),
            ]
        )
    check = statement_from_string(f"if {spins} % {_SCATTER_SPINS} == 0: pass")
    fallback = statement_from_string(f"if {spins} >= {_SCATTER_SPINS}: pass")
    assert isinstance(check, ast.If) and isinstance(fallback, ast.If)
    check.body = cover
    fallback.body = synthesize
    return [statement_from_string(f"{spins} += 1"), check, fallback]


def _by_group(polls: list[InbandPoll]) -> dict[str, list[InbandPoll]]:
    groups: dict[str, list[InbandPoll]] = {}
    for poll in polls:
        groups.setdefault(poll.group or poll.word, []).append(poll)
    return groups


def flush_inband_polls(cg: CodegenInterface, node: torch.fx.Node) -> None:
    """Wait on the pending polls of ``node``'s graph once ``node`` reads one.

    Outside WS loops, block polls first reload together until a CTA vote finds
    them current. The per-thread spin then covers replicas the vote may skip.
    """
    device_function = cg.device_function
    polls = [
        poll for poll in device_function.inband_polls if poll.node.graph is node.graph
    ]
    if not polls or (
        node.op != "output"
        and not any(poll.node in node.all_input_nodes for poll in polls)
    ):
        return
    device_function.inband_polls = [
        poll for poll in device_function.inband_polls if poll not in polls
    ]
    peer = device_function.peer_state
    assert peer is not None
    scattered = [poll for poll in polls if poll.scatter is not None]
    if any(poll.threads is None for poll in scattered):
        raise exc.CrossLoopSchedulingError(
            "because an inband scatter read must be a block outside WS loops"
        )
    # Issue every group's first loads before any spin so their latencies overlap.
    for poll in polls:
        cg.add_statement(poll.load(peer.epoch))
    voting = [poll for poll in polls if poll.threads is not None]
    if voting:
        # A per-thread spin reloads one asm's words at a time; this reloads all.
        stale = device_function.new_var("inband_stale", dce=False)
        cg.add_statement(
            statement_from_string(f"{stale} = {_stale(peer.epoch, voting)}")
        )
        loop = statement_from_string(f"while {stale} != 0: pass")
        assert isinstance(loop, ast.While)
        loop.body = [
            *(poll.load(peer.epoch) for poll in voting),
            *_scatter_fallback(cg, peer, scattered),
            statement_from_string(f"{stale} = {_stale(peer.epoch, voting)}"),
        ]
        cg.add_statement(loop)
    for group in _by_group(polls).values():
        words = [poll.word for poll in group]
        asm, constraints = _poll_asm(len(words), group[0].pack)
        cg.add_statement(
            _inline_asm(
                ", ".join(words),
                asm,
                constraints,
                [
                    *words,
                    *(poll.address for poll in group),
                    f"tl.cast({peer.epoch}, tl.uint32)",
                ],
                "tl.uint64" if len(group) == 1 else f"({'tl.uint64, ' * len(group)})",
                group[0].pack,
            )
        )
        for poll in group:
            cg.add_statement(poll.result)
