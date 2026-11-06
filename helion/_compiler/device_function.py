from __future__ import annotations

import ast
from collections import defaultdict
import contextlib
import dataclasses
import enum
import itertools
import math
import threading
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from typing import Protocol
from typing import TypeVar
from typing import cast

import sympy
import torch
from torch._dynamo.source import TensorProperty
from torch._dynamo.source import TensorPropertySource
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.graph import _Namespace

from .. import exc
from .._compat import get_tensor_descriptor_fn_name
from .._compat import is_hip
from .._utils import indexing_uses_tensor_descriptor
from .ast_extension import ExtendedAST
from .ast_extension import clone_ast
from .ast_extension import create
from .ast_extension import create_arg
from .ast_extension import create_arguments
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .ast_read_writes import ReadWrites
from .ast_read_writes import ast_rename
from .ast_read_writes import dead_assignment_elimination
from .ast_read_writes import dead_expression_elimination
from .ast_read_writes import dead_lane_loop_elimination
from .backend_registry import all_reserved_launch_param_names
from .compile_environment import CompileEnvironment
from .host_function import HostFunction
from .host_function import NoCurrentFunction
from .lane_loop_context import LaneLoopContext
from .output_header import reserved_names
from .runtime_tensor_guards import tensor_pairs_with_proven_disjoint_storage
from .source_location import SyntheticLocation
from .variable_origin import BlockSizeOrigin
from .variable_origin import GridOrigin
from .variable_origin import Origin
from .variable_origin import TensorSizeOrigin
from .variable_origin import TensorStrideOrigin
from .variable_origin import TileBeginOrigin
from .variable_origin import TileExtentOrigin

if TYPE_CHECKING:
    from ..runtime.config import Config
    from .cross_loop_codegen import PeerState
    from .cute.bounded_cache_codegen import BoundedCacheRequest
    from .cute.device_state import CuteDeviceFunctionState
    from .device_ir import HelperFunctionGraphInfo
    from .generate_ast import GenerateAST
    from .indexing_strategy import IndexingStrategy
    from .program_id import ProgramIDs
    from .tile_dispatch import TileStrategyDispatch
    from .triton.distributed_ops import InbandPoll
    from helion._compiler.pallas.dma import DmaResources
    from helion._compiler.pallas.ordered_carry import CarryBoundaryTile
    from helion._compiler.pallas.ordered_carry import CarryScratchKey
    from helion._compiler.pallas.plan_tiling import DimensionTiling
    from helion._compiler.pallas.plan_tiling import GridScalarIndex

    _P = TypeVar("_P", bound="TensorPropertyArg")

    class _TLS(Protocol):
        functions: list[DeviceFunction]


tls: _TLS = cast("_TLS", threading.local())


def _exact_thread_block_dims(
    tile_strategy: TileStrategyDispatch,
) -> tuple[int, int, int] | None:
    """Return a proven-static launch shape, or ``None`` when inference fails.

    ``None`` as well when a launch axis is sized by a kernel argument
    (``TileStrategyDispatch.symbolic_thread_axes``): the static shape holds a
    one for it, and a pass proving bounds from that one (the lane-tile bounds
    simplifier erased the leader guard ``cute.arch.thread_idx()[axis] == 0``
    of an atomic beside the axis, which then ran once per thread of it) would
    prove the wrong thing.
    """

    try:
        if tile_strategy.symbolic_thread_axes():
            return None
        thread_dims = tile_strategy.thread_block_dims()
        if len(thread_dims) != 3 or any(
            not isinstance(dim, (int, sympy.Integer)) for dim in thread_dims
        ):
            return None
        result = int(thread_dims[0]), int(thread_dims[1]), int(thread_dims[2])
        return result if all(dim > 0 for dim in result) else None
    except (TypeError, ValueError):
        # A sympy extent that is not an integer (``int()`` of a symbolic
        # thread count); every other failure of the strategies surfaces.
        return None


@dataclasses.dataclass
class DeviceBodyContext:
    """What ``Backend.finalize_device_body`` / ``finalize_kernel_def`` see.

    The expressions ``DeviceFunction.codegen_function_def`` computed before
    handing the rendered kernel body to the backend, passed through as they
    are; the backend derives its frozen facts from them once per render
    (the CuTe ``PassContext``).  ``args`` is the kernel signature at the
    pre-rename hook; the hook returns the parameters it appends.
    ``constexpr_values`` carries the pre-rename stage's result (the static
    integer value of every constexpr name) to the post-rename stage, and
    ``pass_context`` the backend's frozen facts (opaque to core).
    ``body_start`` is the index of the first statement of the traced device
    body; the statements before it are the ``scalar_arg_preamble`` statements,
    then the device function's own preamble (e.g. persistent program-id setup).
    """

    device_function: DeviceFunction
    args: tuple[ast.arg, ...]
    param_args: list[Argument]
    sorted_arguments: list[Argument]
    constexpr_to_inline: list[ConstExprArg]
    wrapper_only_params: list[str]
    bounded_cache_request: BoundedCacheRequest | None
    constexpr_values: dict[str, int] = dataclasses.field(default_factory=dict)
    pass_context: object | None = None
    body_start: int = 0


class VarInfo(NamedTuple):
    """Information about a variable derived from a sympy expression."""

    name: str
    fx_node: torch.fx.Node


def find_block_size_symbols(
    expr: sympy.Expr,
) -> tuple[dict[sympy.Symbol, int], set[sympy.Symbol]]:
    """
    Find block size symbols in a sympy expression.

    Returns:
        tuple of (block_size_mapping, non_block_size_symbols) where:
        - block_size_mapping: dict mapping block size symbols to their block_id
        - non_block_size_symbols: set of symbols that are NOT block sizes
    """
    if not isinstance(expr, sympy.Expr):
        return {}, set()

    hf = HostFunction.current()
    block_sizes = {}
    non_block_size_symbols = set()

    for symbol in expr.free_symbols:
        origin_info = hf.expr_to_origin.get(symbol)
        if origin_info is None or not isinstance(origin_info.origin, BlockSizeOrigin):
            # pyrefly: ignore [bad-argument-type]
            non_block_size_symbols.add(symbol)
        else:
            # pyrefly: ignore [unsupported-operation]
            block_sizes[symbol] = origin_info.origin.block_id

    # pyrefly: ignore[bad-return]
    return block_sizes, non_block_size_symbols


def substitute_configured_block_sizes(expr: sympy.Expr, config: Config) -> sympy.Expr:
    """Substitute only block-size symbols fixed by this config, without hints.

    Runtime size symbols remain symbolic. This also works before the device
    function's strategies exist, when Metal budgets their thread counts.
    """
    env = CompileEnvironment.current()
    block_symbols, _ = find_block_size_symbols(expr)
    values = {
        symbol: sympy.Integer(size)
        for symbol, block_id in block_symbols.items()
        if isinstance(size := env.block_sizes[block_id].from_config(config), int)
    }
    return expr.xreplace(values)


def contains_only_block_size_symbols(expr: sympy.Expr) -> bool:
    """Check if expression contains only block size symbols (no other variables)."""
    _, non_block = find_block_size_symbols(expr)
    return len(non_block) == 0


@dataclasses.dataclass
class Argument:
    name: str  # in the device function

    def host_str(self) -> str:
        raise NotImplementedError

    def arg_def_node(self) -> ast.arg:
        return create_arg(self.name)

    def sort_key(self) -> tuple[object, ...]:
        return (_sort_order[type(self)],)


@dataclasses.dataclass
class TensorArg(Argument):
    fake_value: torch.Tensor
    _host_str: str | None

    def host_str(self) -> str:
        if self._host_str is None:
            raise RuntimeError("TensorArg has no host representation")
        return self._host_str


@dataclasses.dataclass
class TensorDescriptorArg(TensorArg):
    # Permutation applied to make stride==1 dimension last
    permutation: list[int] | None = None

    def host_str(self) -> str:
        if self._host_str is None:
            raise RuntimeError(
                "TensorDescriptorArg is device-only and has no host representation"
            )
        return self._host_str

    @property
    def inverse_permutation(self) -> list[int]:
        """Get the inverse permutation to undo the applied permutation."""
        if (permutation := self.permutation) is None:
            raise RuntimeError("TensorDescriptorArg.permutation is None")
        inverse_perm = [0] * len(permutation)
        for i, p in enumerate(permutation):
            inverse_perm[p] = i
        return inverse_perm


@dataclasses.dataclass
class TensorPropertyArg(Argument):
    tensor_arg: TensorArg
    dim: int

    def sort_key(self) -> tuple[object, ...]:
        return (_sort_order[type(self)], self.tensor_arg.name, self.dim)


class TensorSizeArg(TensorPropertyArg):
    def host_str(self) -> str:
        return f"{self.tensor_arg.host_str()}.size({self.dim})"


class TensorStrideArg(TensorPropertyArg):
    def host_str(self) -> str:
        return f"{self.tensor_arg.host_str()}.stride({self.dim})"


@dataclasses.dataclass
class NumericArgument(Argument):
    _host_str: str

    def host_str(self) -> str:
        return self._host_str


class ConstExprArg(NumericArgument):
    def arg_def_node(self) -> ast.arg:
        return create_arg(
            self.name, CompileEnvironment.current().backend.constexpr_type
        )


@dataclasses.dataclass
class SymbolArgument(NumericArgument):
    pass


class StaticShape(Argument):
    def __init__(self, val: int) -> None:
        super().__init__(repr(val))


_sort_order: dict[type[Argument], int] = {
    TensorDescriptorArg: 0,
    TensorArg: 0,
    TensorSizeArg: 1,
    TensorStrideArg: 2,
    SymbolArgument: 3,
    ConstExprArg: 4,
}


@dataclasses.dataclass
class ScratchArg:
    """A scratch memory buffer allocated in device memory (e.g., VMEM on TPU).

    scratch_type can be "vmem" (default) or "dma_semaphore" for Pallas
    scratch.
    """

    name: str
    shape: tuple[int | torch.SymInt, ...]
    dtype: torch.dtype | None  # None for semaphores
    scratch_type: str = "vmem"
    shape_sources: tuple[tuple[torch.Tensor, int] | None, ...] | None = None


def _is_literal_constexpr(arg: ConstExprArg) -> bool:
    """Check if a constexpr arg has a known literal value that can be inlined at module level."""
    host_str = arg.host_str()
    if host_str == arg.name:
        return False
    try:
        ast.literal_eval(host_str)
        return True
    except (ValueError, SyntaxError):
        return False


class PallasMemorySpace(enum.Enum):
    """TPU memory space for Pallas tensors."""

    HBM = "hbm"  # Pipeline body tensors (DMA)
    SMEM = "smem"  # Scalar-only access
    VMEM = "vmem"  # Vector/slice access (default)


def _selects_tma_access(config: Config) -> bool:
    """Whether the config may lower some global access through TMA (async proxy).

    Decided before codegen, since a handoff can be emitted before the access.
    Over-reports: fact slots do not always match codegen's memory-op slots.
    """
    env = CompileEnvironment.current()
    capability = env.config_spec.target_device_capability
    if (
        env.backend_name != "triton"
        or env.device.type != "cuda"
        or is_hip()
        or capability is None
        or capability < (9, 0)
    ):
        return False
    return indexing_uses_tensor_descriptor(config.atomic_indexing) or (
        indexing_uses_tensor_descriptor(config.indexing)
        and any(2 <= fact.ndim <= 5 for fact in env.config_spec.memory_op_facts)
    )


class DeviceFunction:
    def __init__(
        self,
        name: str,
        config: Config,
        codegen: GenerateAST,
    ) -> None:
        super().__init__()
        self.name = name
        self.config = config
        self.codegen = codegen
        self.has_barrier = CompileEnvironment.current().has_barrier
        self.uses_async_proxy_global = _selects_tma_access(config)
        self._async_store_drained = False
        self._tma_store_warp_specialized = False
        self.arguments: list[Argument] = []
        self.preamble: list[ast.AST] = []
        self.body: list[ast.AST] = []
        self._tensor_args: dict[torch.Tensor, TensorArg] = {}
        self._tensor_descriptor_args: dict[
            tuple[torch.Tensor, str], TensorDescriptorArg
        ] = {}
        self._expr_args: dict[sympy.Expr, NumericArgument] = {}
        self._constexpr_exprs: set[sympy.Expr] = set()
        self._constexpr_args: dict[str, ConstExprArg] = {}
        self._constexpr_host_defs: set[str] = set()
        self._scratch_args: list[ScratchArg] = []
        self.wrapper_only_params: list[str] = []
        # Optional parameter annotations of wrapper-only params (e.g. TMA
        # descriptors passed as ``cutlass.GridConstant[...]``).
        self.wrapper_only_param_annotations: dict[str, str] = {}
        self._tensor_properties: dict[tuple[object, ...], TensorPropertyArg] = {}
        self._unique_counter: dict[str, itertools.count[int]] = defaultdict(
            itertools.count
        )
        self.pid: ProgramIDs | None = None
        self.namespace: _Namespace = _Namespace()
        self.namespace._used_names.update(reserved_names())
        self.namespace._used_names.update(
            CompileEnvironment.current().backend.reserved_device_names()
        )

        self.namespace._used_names.update(all_reserved_launch_param_names())
        self.namespace._used_names.update(
            x.removeprefix("_triton_config_")
            for x in config
            if x.startswith("_triton_config_")
        )
        self._variable_renames: dict[str, list[str]] = {}
        self.dce_vars: list[str] = []
        # Arg names referenced only by fusion placeholder strings
        # (<STORE_OUTPUT_*>, <LOAD_INPUT_*>), not by the AST body.
        # DCE would incorrectly strip them without this exemption.
        self.placeholder_args: set[str] = set()
        # Sourceless prologue params (e.g. ones_like) that are fully inlined
        # by the prologue hook.  These should be DCE'd away and also removed
        # from the host function signature (populated by _codegen_prologue_fusion).
        self.sourceless_prologue_params: set[str] = set()
        self.block_size_var_cache: dict[tuple[int, ...], str] = {}
        self.expr_to_var_info: dict[sympy.Expr, VarInfo] = {}
        self.deferred_rdim_defs: list[tuple[str, sympy.Expr]] = []
        # The backend's per-device-function state (``Backend.new_device_state``);
        # ``cute_state`` is the CuTe backend's typed view of it.
        self.backend_state: object | None = (
            CompileEnvironment.current().backend.new_device_state(config)
        )
        # The lane loop of each block a per-thread tile strategy or a
        # vectorizable reduction strategy owns, keyed by block id
        # (``lane_loop_context.py``); the strategy that opens the loop
        # registers it (``register_lane``) and fills it in.
        self.lane_contexts: dict[int, LaneLoopContext] = {}

        from .helper_function import HelperFunctionManager

        self.helper_manager = HelperFunctionManager()

        from .tile_dispatch import TileStrategyDispatch

        self.tile_strategy: TileStrategyDispatch = TileStrategyDispatch(self, config)

        # Store indexing config to lazily create strategies per load/store
        self._indexing_config = config.indexing
        self.indexing_strategies: list[IndexingStrategy] = []

        # Atomic indexing config (separate from load/store indexing)
        self._atomic_indexing_config = config.atomic_indexing
        self.atomic_indexing_strategies: list[IndexingStrategy] = []
        self.atomic_op_index = 0

        self.rng_seed_count = 0
        self.device_load_index = 0
        self.device_load_cache_modifier_index = 0
        self.device_store_index = 0
        self.device_store_cache_modifier_index = 0
        # Single counter for both loads and stores for indexing assignment
        self.device_memory_op_index = 0
        self.epilogue_subtile_store_indices: dict[str, int] = {}
        self.epilogue_subtile_atomic_indices: dict[str, int] = {}
        self.rng_seed_buffer_param_name = None
        self.requires_nvshmem = False
        self.requires_collective_id = False
        self.requires_remote_copy = False
        # Descriptor lifecycle operations may live in nested control-flow
        # graphs. Backends resolve them through this compiler-owned table.
        self.remote_copy_descriptors: dict[int, object] = {}
        # Triton/NVSHMEM receive completion is hidden from the frontend. The
        # lowering appends one compiler-managed signal-pad pointer to the kernel
        # signature and assigns one slot per receive-wait descriptor.
        self.triton_remote_copy_signal_arg: str | None = None
        self.triton_remote_copy_signal_dst: str | None = None
        self.triton_remote_copy_signal_slots = 0
        # Triton peer barriers use a separate compiler-owned symmetric signal
        # pad because they do not necessarily have a remote-copy destination.
        self.triton_remote_barrier_signal_arg: str | None = None
        self.triton_remote_barrier_signal_slots = 0
        # NVSHMEM takes pointer sources. Computed Triton tiles are materialized
        # into compiler-owned global scratch before the transfer starts.
        self.triton_remote_copy_scratch_args: list[str] = []
        self.triton_remote_copy_scratch_specs: list[tuple[str, str]] = []
        # Compiler-owned state that must persist across launches (for example,
        # epoch-scaled tile-dependency counters). The Triton launcher allocates it
        # once per kernel/device/stream and appends it to the kernel arguments.
        # A symmetric spec also appends the per-rank base pointer table.
        self.triton_persistent_state_args: list[str] = []
        self.triton_persistent_state_specs: list[tuple[str, str, str, bool]] = []
        # Cross-rank transport state (cross_loop_codegen.peer_state), the polls
        # awaiting their first use, the inband accesses emitted so far, and the
        # done-word asm of each root with inband scatter stores.
        self.peer_state: PeerState | None = None
        self.inband_polls: list[InbandPoll] = []
        self.inband_access_ids: set[int] = set()
        self.inband_scatter_done: dict[int, str] = {}
        # Cross-grid polling is safe in isolation only when the required worker
        # cohort can reside together. The launcher validates exact compiled
        # occupancy, but does not reserve capacity against concurrent streams.
        self.triton_minimum_resident_programs: str | None = None
        # Opaque tile bodies can be outlined behind Triton helper boundaries.
        # Most helpers are inlined; selected scheduling regions remain
        # ``noinline`` to bound register lifetimes in the fused kernel.
        self.triton_outlined_helpers: list[
            tuple[str, tuple[str, ...], tuple[ast.stmt, ...], bool]
        ] = []
        self.triton_outlined_helper_constexprs: dict[str, int] = {}
        # Per-backend codegen state for this function, keyed by backend name
        # (covered by helion.backend_api; see docs/api/backend_api.md).
        self.backend_codegen_state: dict[str, Any] = {}
        # Pallas: id(fake_tensor) → [DimensionTiling], recorded during `plan_tiling`
        self.pallas_tensor_dim_tilings: dict[int, list[DimensionTiling]] = {}
        # Pallas: tensor dimensions selected by scalar metadata indexed by one
        # outer grid axis. The launcher scalar-prefetches the metadata and uses
        # it directly in the selected tensor's BlockSpec index map.
        self.pallas_grid_scalar_indices: dict[int, dict[int, GridScalarIndex]] = {}
        # Track Pallas remote-copy operands by tensor and storage identity. The
        # storage key keeps views of the same allocation consistent across
        # nested control-flow graphs.
        self.pallas_remote_copy_tensor_ids: set[int] = set()
        self.pallas_remote_copy_storage_ids: set[int] = set()
        # Read-only inputs selected for explicit local HBM-to-VMEM DMA at
        # their load sites, rather than launcher-managed VMEM BlockSpecs.
        self.pallas_direct_hbm_load_tensor_ids: set[int] = set()
        self.pallas_hoisted_direct_dma_copy_names: set[str] = set()
        # Body-local ``torch.empty`` tensors that are read and written but do
        # not escape the kernel return can live entirely in compiler-managed
        # VMEM scratch instead of becoming hidden HBM in/out arguments.
        self.pallas_internal_scratch_storage_names: dict[int, str] = {}
        # Pallas: id(fake_tensor) → memory space, determined during
        # tracing (HBM for pipeline) and codegen (SMEM for scalar access).
        # NOTE: Currently each tensor can only have one memory space.
        # If a tensor needs both SMEM (scalar access) and VMEM (slice
        # access), it will need tensor duplication — passing the same
        # data as two separate args in different memory spaces. This
        # dict would then need to support multiple entries per tensor
        # or the tensor would get distinct arg IDs per memory space.
        self.pallas_memory_space: dict[int, PallasMemorySpace] = {}
        # Root-grid memory operations routed through explicit DMA resources.
        # Inner-loop schedulers keep iteration-specific bindings on loop state.
        self.pallas_grid_dma_bindings: dict[torch.fx.Node, DmaResources] = {}
        # Pallas: VMEM bytes of the largest emit_pipeline's own block buffers.
        # emit_pipeline allocates them inside the kernel, so the launcher's
        # VMEM estimate cannot see them unless codegen passes them along.
        self.pallas_pipeline_vmem_bytes: int = 0
        # Pallas: id(fake_tensor) → {dim: {block_id: extra_pad}} for dims
        # using pl.ds() that may need host-side padding.  A dim can be tiled
        # by several loops (e.g. a reduction loop then an elementwise loop),
        # each with its own block size, so every block_id is kept.
        self.pallas_pad_info: dict[int, dict[int, dict[int, int]]] = {}
        # Pallas ordered carry: jagged row block_id -> CarryBoundaryTile.  Filled by
        # the inner-loop codegen when the tile is a legal map axis; read by
        # the store codegen to stitch the boundary across neighbouring groups.
        self.carry_tiles: dict[int, CarryBoundaryTile] = {}
        # Pallas: jagged tile block_id -> proven runtime-window alignment.
        self.aligned_tiles: dict[int, int] = {}
        # CarryScratchKey(row block_id, output name) -> carry scratch var name.
        # One scratch per output buffer (a tile may feed several stores),
        # allocated at the store.
        self.carry_scratch: dict[CarryScratchKey, str] = {}

    def mark_pallas_remote_copy_operand(self, tensor: torch.Tensor) -> None:
        self.pallas_remote_copy_tensor_ids.add(id(tensor))
        self.pallas_remote_copy_storage_ids.add(id(tensor.untyped_storage()))

    def is_pallas_remote_copy_operand(self, tensor: torch.Tensor) -> bool:
        return (
            id(tensor) in self.pallas_remote_copy_tensor_ids
            or id(tensor.untyped_storage()) in self.pallas_remote_copy_storage_ids
        )

    def is_pallas_direct_hbm_load(self, tensor: torch.Tensor) -> bool:
        return id(tensor) in self.pallas_direct_hbm_load_tensor_ids

    def pallas_internal_scratch_name(self, tensor: torch.Tensor) -> str | None:
        return self.pallas_internal_scratch_storage_names.get(
            id(tensor.untyped_storage())
        )

    def pallas_tensor_ref_name(self, tensor: torch.Tensor) -> str:
        scratch = self.pallas_internal_scratch_name(tensor)
        return scratch if scratch is not None else self.tensor_arg(tensor).name

    def allocate_store_index(self) -> int:
        """Bump store counters and return the indexing strategy slot."""
        self.device_store_index += 1
        idx = self.device_memory_op_index
        self.device_memory_op_index += 1
        return idx

    def get_indexing_strategy(self, index: int) -> IndexingStrategy:
        from .indexing_strategy import IndexingStrategy
        from .indexing_strategy import PointerIndexingStrategy

        # Expand strategies list if needed
        while len(self.indexing_strategies) <= index:
            idx = len(self.indexing_strategies)

            if isinstance(self._indexing_config, str):
                # Single string: all loads/stores use the same strategy
                if not self.indexing_strategies:
                    strategy = IndexingStrategy.select(self._indexing_config)
                else:
                    strategy = self.indexing_strategies[0]
            elif isinstance(self._indexing_config, list) and self._indexing_config:
                # List: one strategy per load/store
                assert idx < len(self._indexing_config), (
                    f"Load/Store operation {idx} exceeds indexing config length "
                    f"{len(self._indexing_config)}. Please specify indexing for all loads and stores."
                )
                strategy = IndexingStrategy.select(self._indexing_config[idx])
            else:
                # Empty/default: use pointer
                strategy = PointerIndexingStrategy()

            self.indexing_strategies.append(strategy)

        return self.indexing_strategies[index]

    def get_atomic_indexing_strategy(self, index: int) -> IndexingStrategy:
        from .indexing_strategy import IndexingStrategy
        from .indexing_strategy import PointerIndexingStrategy

        while len(self.atomic_indexing_strategies) <= index:
            idx = len(self.atomic_indexing_strategies)

            if isinstance(self._atomic_indexing_config, str):
                if not self.atomic_indexing_strategies:
                    strategy = IndexingStrategy.select(self._atomic_indexing_config)
                else:
                    strategy = self.atomic_indexing_strategies[0]
            elif (
                isinstance(self._atomic_indexing_config, list)
                and self._atomic_indexing_config
            ):
                assert idx < len(self._atomic_indexing_config), (
                    f"Atomic operation {idx} exceeds atomic_indexing config length "
                    f"{len(self._atomic_indexing_config)}. Please specify atomic_indexing for all atomic ops."
                )
                strategy = IndexingStrategy.select(self._atomic_indexing_config[idx])
            else:
                strategy = PointerIndexingStrategy()

            self.atomic_indexing_strategies.append(strategy)

        return self.atomic_indexing_strategies[index]

    def has_rng_ops(self) -> bool:
        """Check if this kernel uses any RNG operations."""
        return self.rng_seed_count > 0 and self.rng_seed_buffer_param_name is not None

    def reserve_rng_seed(self, seed_index: int) -> None:
        """Ensure the RNG seed buffer is available up to a specific index."""
        assert seed_index >= 0
        self.rng_seed_count = max(self.rng_seed_count, seed_index + 1)
        if self.rng_seed_buffer_param_name is None:
            # pyrefly: ignore [bad-assignment]
            self.rng_seed_buffer_param_name = self.new_var("rng_seed_buffer")

    def block_size_var(self, block_id: int) -> str | None:
        key = (block_id,)

        # Block size var could be used outside of a hl.tile loop, and at that point
        # no tile strategy has populated the cache yet, so we must lazily create
        # the constexpr argument here and lift it as device function argument;
        # later strategies will reuse the cached name or intentionally replace it
        # (e.g. flattened loops, reductions).
        if key not in self.block_size_var_cache:
            env = CompileEnvironment.current()
            block_value = env.block_sizes[block_id].from_config(self.config)

            if block_value is None:
                return None

            var_name = self.new_var(f"_BLOCK_SIZE_{block_id}")
            self.block_size_var_cache[key] = var_name
            self.constexpr_arg_with_host_def(var_name, block_value)

        return self.block_size_var_cache[key]

    def resolved_block_size(self, block_id: int) -> int | torch.SymInt | None:
        """Resolve a block_id to its concrete size for the current config."""
        env = CompileEnvironment.current()
        return env.block_sizes[block_id].from_config(self.config)

    def proven_sublane_alignment(self, block_id: int) -> int | None:
        """The sublane alignment that block_id's offsets are proven to satisfy.

        A bf16 window rounded down to 16 rows and stepped by a block of 32 is
        proven: every offset is 16, 48, 80, ... A block of 24 over the same
        window is not, since 40 and 64 are not multiples of 16. Neither is any
        dim without a recorded window, whose begin is an arbitrary runtime row.

        ``None`` means promise nothing.
        """
        sublane = self.aligned_tiles.get(block_id)
        if sublane is None:
            return None
        block = self.resolved_block_size(block_id)
        if not isinstance(block, int) or block % sublane != 0:
            return None
        return sublane

    def evaluate_constexpr_condition(self, test: object) -> bool | None:
        """Resolve a control-flow test to a concrete bool for the current config.

        Block sizes are symbolic during the (single) frontend pass but become
        concrete integers per-config at codegen time.  A branch condition that
        depends only on block sizes (and other constants) is therefore a
        compile-time constant here, even though it could not be folded earlier.
        Substituting the config's block sizes lets us decide the branch and emit
        only the live side, which is what makes shape-divergent branches legal.

        Returns the concrete bool, or ``None`` if the test is not a compile-time
        constant for this config (e.g. it depends on a runtime value).
        """
        if isinstance(test, (bool, int)):
            return bool(test)
        if not isinstance(test, torch.SymBool):
            return None
        # NB: a boolean expression (And/Or/Relational) is a sympy.Basic but not a
        # sympy.Expr, so we classify its free symbols directly rather than via
        # find_block_size_symbols (which only handles sympy.Expr).
        expr = test._sympy_()
        if not isinstance(expr, sympy.Basic):
            return None
        env = CompileEnvironment.current()
        hf = HostFunction.current()
        subs: dict[sympy.Basic, sympy.Basic] = {}
        for symbol in expr.free_symbols:
            origin_info = hf.expr_to_origin.get(symbol)
            if origin_info is None or not isinstance(
                origin_info.origin, BlockSizeOrigin
            ):
                return None
            value = env.block_sizes[origin_info.origin.block_id].from_config(
                self.config
            )
            if not isinstance(value, int):
                return None
            subs[symbol] = sympy.Integer(value)
        resolved = expr.xreplace(subs)
        if resolved.free_symbols:
            return None
        return bool(resolved)

    def constexpr_taken_graph_id(self, if_node: torch.fx.Node) -> int | None:
        """The graph id of the side an ``_if`` node takes when its test is a
        constant for this config (``evaluate_constexpr_condition``), or
        ``None`` when the test is decided at runtime."""
        test, if_graph_id, else_graph_id, *_ = if_node.args
        taken = self.evaluate_constexpr_condition(
            test.meta["val"] if isinstance(test, torch.fx.Node) else test
        )
        if taken is None:
            return None
        return cast("int", if_graph_id if taken else else_graph_id)

    def try_map_block_symbols_to_vars(self, expr: sympy.Expr) -> sympy.Expr | None:
        """Try to map all block size symbols in expression to their variable names.

        Returns:
            - The expression with symbols replaced if ALL symbols are block sizes and have variables
            - None if the expression contains non-block symbols or unmapped block symbols
        """
        block_mapping, non_block_symbols = find_block_size_symbols(expr)

        # Can't map if there are non-block symbols
        if non_block_symbols:
            return None

        # No symbols to map - return as-is
        if not block_mapping:
            return expr

        # Try to map all block symbols to their variables
        var_map = {}
        for symbol, block_id in block_mapping.items():
            block_var = self.block_size_var(block_id)
            if not block_var:
                # Can't map this block symbol - fail
                return None
            var_map[symbol] = sympy.Symbol(block_var, integer=True)

        # Successfully mapped all symbols
        # pyrefly: ignore [bad-return]
        return expr.xreplace(var_map)

    def merge_variable_names(self, a: str, b: str) -> None:
        name_group = [
            *self._variable_renames.get(a, [a]),
            *self._variable_renames.get(b, [b]),
        ]
        for n in name_group:
            self._variable_renames[n] = name_group

    def variable_aliases(self, name: str) -> tuple[str, ...]:
        return tuple(self._variable_renames.get(name, [name]))

    @property
    def cute_state(self) -> CuteDeviceFunctionState:
        return cast("CuteDeviceFunctionState", self.backend_state)

    def register_lane(
        self, block_id: int, *, flat_multi: bool = False
    ) -> LaneLoopContext:
        """Open the lane-loop context of ``block_id`` for the strategy owning it."""
        context = LaneLoopContext(block_id, flat_multi=flat_multi)
        self.lane_contexts[block_id] = context
        return context

    def set_pid(self, pid: ProgramIDs) -> None:
        if self.pid is not None:
            raise exc.InvalidAPIUsage(
                "Multiple top-level grid loops are not supported with this config. "
                "Try using pid_type='persistent' or combining the loops into a single "
                "hl.tile/hl.grid call."
            )
        self.pid = pid

    def sympy_expr(self, expr: sympy.Expr) -> str:
        env = CompileEnvironment.current()
        with contextlib.suppress(Exception):
            expr = env.shape_env.simplify(expr)
        expr = env.specialize_expr(expr)
        host = HostFunction.current()
        resolved = host.resolve_tunable_expr(expr, self.config)
        if isinstance(resolved, (int, float, bool)):
            return env.backend.scalar_literal_expr(resolved)
        expr = resolved
        if not expr.free_symbols:
            return env.backend.sympy_printer_expr(expr)
        if expr in self.expr_to_var_info:
            return self.expr_to_var_info[expr].name
        expr_to_origin = HostFunction.current().expr_to_origin
        if expr in expr_to_origin:
            return self._lift_sympy_arg(expr)
        if env.backend.lowers_integer_powers:
            from .integer_power import prepare_integer_powers

            # Expressions already evaluated on the host or in a device local
            # need no new arithmetic. Prove integer powers only for the residual
            # expression that will actually be evaluated in this device scope.
            expr = prepare_integer_powers(expr, self)
        replacements = {}
        for sym in sorted(expr.free_symbols, key=lambda x: x.name):
            assert isinstance(sym, sympy.Symbol)
            selected = (
                host.resolve_tunable_expr(sym, self.config)
                if sym not in expr_to_origin
                else sym
            )
            if isinstance(selected, (int, float, bool)):
                # A selected constant cannot depend on an inner-loop alias:
                # lift_symnode may hoist this expression outside that loop.
                replacements[sym] = sympy.Symbol(
                    f"({env.backend.scalar_literal_expr(selected)})"
                )
            elif sym in self.expr_to_var_info:
                replacements[sym] = sympy.Symbol(
                    self.expr_to_var_info[sym].name, integer=True
                )
            elif sym in expr_to_origin:
                replacements[sym] = sympy.Symbol(
                    self._lift_sympy_arg(sym), integer=True
                )
            else:
                raise AssertionError(f"no origin found for {sym.name}")
        # pyrefly: ignore [bad-argument-type]
        return env.backend.sympy_printer_expr(expr.xreplace(replacements))

    def _lift_sympy_arg(self, expr: sympy.Expr) -> str:
        env = CompileEnvironment.current()
        origin = HostFunction.current().expr_to_origin[expr]
        if isinstance(origin.origin, TensorSizeOrigin):
            assert origin.fake_value is not None
            arg = self.tensor_size(
                origin.fake_value,
                origin.origin.key,
            )
            return arg.name
        if (
            isinstance(origin.origin, TensorStrideOrigin)
            and origin.fake_value is not None
        ):
            return self.tensor_stride(origin.fake_value, origin.origin.key).name
        if isinstance(origin.origin, BlockSizeOrigin):
            result = self.block_size_var(env.canonical_block_id(origin.origin.block_id))
            assert result is not None
            return result
        if isinstance(origin.origin, GridOrigin):
            # Only the tile's begin is the loop offset.  The other tile edges
            # render as compound expressions (``tile.end`` clamps to the loop
            # end, ``tile.count`` divides, ``tile.id`` shifts), so returning the
            # offset for them would silently substitute the begin.  This path is
            # reached whenever such a symbol survives into a sympy expression
            # rather than its own op -- e.g. ``min(n, tile.end)``, which
            # ``_builtin_min`` folds into ``sympy.Min`` at trace time.  The
            # result becomes a sympy Symbol name, so parenthesize it to keep
            # precedence intact inside the enclosing expression.
            resolved = env.resolve_codegen_block_id(
                origin.origin.block_id, self.codegen
            )
            if type(origin.origin) is GridOrigin:
                return self.codegen.offset_var(resolved)
            if type(origin.origin) is TileBeginOrigin:
                return self.codegen.tile_begin_var(resolved)
            if type(origin.origin) is TileExtentOrigin:
                return self.tile_extent_expr(resolved)
            # Render through the origin so each derived edge keeps its own
            # formula, but on the resolved live loop: host_str() reads
            # offset_var and active_device_loops by block_id, and an aliased
            # symbol's own block may have no active loop.
            derived = dataclasses.replace(origin.origin, block_id=resolved)
            return f"({derived.host_str()})"
        return self.expr_arg(expr, origin.origin).name

    def tile_extent_expr(self, block_id: int) -> str:
        """The elements the current tile of ``block_id`` holds, as device code.

        The block size when the loop needs no mask (the block divides the
        dim), otherwise ``min(begin + block, end) - begin``: the last tile of
        a dim the block does not divide, or a block wider than the dim, holds
        fewer elements than the block, and the masked ones must not count
        in a mean.  Spelled as a parenthesized compound expression in that
        case, as the other derived tile edges are.

        A masked loop that carries no end variable has no per-dim extent to
        divide by: a flattened loop's dims share one flat bound, and some
        strategies keep a data-dependent end as a tensor.  No shipped
        strategy hosts a reduction in such a loop (a loop with a reduction
        is never flattened); should one, the mean is declined rather than
        divided by the block.
        """
        env = CompileEnvironment.current()
        block_size = self.block_size_var(env.canonical_block_id(block_id))
        if block_size is None:
            block_size = "1"
        if self.codegen.mask_var(block_id) is None:
            return block_size
        end = (
            self.codegen.active_device_loops[block_id][-1]
            .block_id_to_info[block_id]
            .end_var_name
        )
        if end is None:
            raise exc.BackendUnsupported(
                env.backend.name,
                f"a mean over tile dim {block_id} in a masked loop without an end "
                "variable: the tile's extent is unknown there",
            )
        begin = self.codegen.tile_begin_var(block_id)
        clamped = env.backend.minimum_expr(f"{begin} + {block_size}", end)
        return f"(({clamped}) - {begin})"

    def user_sympy_expr(self, expr: sympy.Expr) -> str:
        """A sympy expression that flows into user computations."""
        expr_to_origin = HostFunction.current().expr_to_origin
        replacements = {}
        for sym in sorted(expr.free_symbols, key=lambda s: s.name):
            assert isinstance(sym, sympy.Symbol)
            origin_info = expr_to_origin.get(sym)
            if origin_info is None:
                continue
            origin = origin_info.origin
            if isinstance(origin, BlockSizeOrigin):
                replacements[sym] = self.tile_strategy.user_size(origin.block_id)
        if replacements:
            # pyrefly: ignore [bad-assignment]
            expr = expr.xreplace(replacements)
        return self.sympy_expr(expr)

    def literal_expr(self, expr: object) -> str:
        if isinstance(expr, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            # SymBool's `_sympy_()` is a boolean, not an Expr that sympy_expr wants.
            # pyrefly: ignore [bad-argument-type]
            return self.sympy_expr(expr._sympy_())
        if isinstance(expr, sympy.Expr):
            return self.sympy_expr(expr)
        if isinstance(expr, float) and not math.isfinite(expr):
            return f"float('{expr}')"
        return repr(expr)

    def unique_name(self, prefix: str, dce: bool = False) -> str:
        return self.new_var(f"{prefix}_{next(self._unique_counter[prefix])}", dce=dce)

    def new_var(self, name: str, *, dce: bool = False) -> str:
        name = self.namespace.create_name(name, None)
        if dce:
            self.dce_vars.append(name)
        return name

    def _thread_asm(self, prefix: str, asm: str) -> ast.stmt:
        var = self.new_var(prefix, dce=False)
        return statement_from_string(
            f"{var} = tl.inline_asm_elementwise("
            f"asm='{asm} mov.u32 $0, $1;', "
            "constraints='=r,r', args=[tl.arange(0, 32)], "
            "dtype=tl.uint32, is_pure=False, pack=1)"
        )

    def async_store_drain(self) -> list[ast.stmt]:
        """Complete this thread's TMA global writes before a cross-thread sync."""
        if not self.uses_async_proxy_global:
            return []
        self._async_store_drained = True
        self._check_async_store_drain()
        # wait_group.read (emitted per TMA store) only frees the smem source.
        return [
            self._thread_asm(
                "async_store_drain",
                "cp.async.bulk.wait_group 0; fence.proxy.async.global;",
            )
        ]

    def async_load_fence(self) -> list[ast.stmt]:
        """Order a completed acquire before later TMA global reads or writes."""
        if not self.uses_async_proxy_global:
            return []
        return [self._thread_asm("async_load_fence", "fence.proxy.async.global;")]

    def cta_barrier(self, barrier: str = "tl.debug_barrier()") -> list[ast.stmt]:
        """A barrier that also orders TMA global accesses across threads."""
        return [
            *self.async_store_drain(),
            statement_from_string(barrier),
            *self.async_load_fence(),
        ]

    def note_tma_store(self, *, warp_specialized: bool) -> None:
        """Record a TMA global store or reduction for async_store_drain."""
        self._tma_store_warp_specialized |= warp_specialized
        self._check_async_store_drain()

    def _check_async_store_drain(self) -> None:
        # Bulk groups are per issuing thread; a warp-specialized worker's stores
        # are invisible to the default warps that drain and publish.
        if self._async_store_drained and self._tma_store_warp_specialized:
            raise exc.InvalidConfig(
                "TMA stores in a range_warp_specialize loop cannot be drained "
                "before a cross-thread sync"
            )

    def tensor_arg(
        self, fake_value: torch.Tensor, prefer_name: str | None = None
    ) -> TensorArg:
        if fake_value not in self._tensor_args:
            origin = HostFunction.current().tensor_to_origin[fake_value]
            arg = TensorArg(
                self.new_var(prefer_name or origin.suggest_var_name()),
                fake_value,
                origin.host_str(),
            )
            self.arguments.append(arg)
            self._tensor_args[fake_value] = arg
        return self._tensor_args[fake_value]

    def tensor_descriptor_arg(
        self, fake_value: torch.Tensor, block_size: list[int | torch.SymInt]
    ) -> TensorDescriptorArg:
        host_function = HostFunction.current()
        block_size_expr = ", ".join(map(self.literal_expr, block_size))
        key = (fake_value, block_size_expr)
        if key not in self._tensor_descriptor_args:
            origin = host_function.tensor_to_origin[fake_value]
            desc_name = self.new_var(origin.suggest_var_name() + "_desc")
            env = CompileEnvironment.current()

            # Find which dimension has stride==1
            layout_signature = env.tensor_descriptor_layout_signature(fake_value)
            assert layout_signature is not None
            stride_one_dim = layout_signature[0]
            assert stride_one_dim is not None

            # Determine if we need permutation (stride==1 dimension is not last)
            permutation = None
            if stride_one_dim != fake_value.ndim - 1:
                # Create permutation to move stride==1 dimension to last position
                permutation = [*range(fake_value.ndim)]
                permutation.pop(stride_one_dim)
                permutation.append(stride_one_dim)

            # Apply permutation if needed
            if permutation is not None:
                block_size = [block_size[i] for i in permutation]
                # Update block_size_expr for the permuted order
                block_size_expr = ", ".join(map(self.literal_expr, block_size))

            descriptor_dims = (
                permutation if permutation is not None else [*range(fake_value.ndim)]
            )
            assert descriptor_dims[-1] == stride_one_dim
            if self.config.host_tensor_descriptors:
                tensor_host = origin.host_str()
                tensor_base = env.backend.tensor_descriptor_host_base(
                    fake_value, tensor_host
                )
                sizes = [
                    f"{tensor_host}.size({dimension})" for dimension in descriptor_dims
                ]
                strides = [
                    f"{tensor_host}.stride({dimension})"
                    for dimension in descriptor_dims
                ]
                # The layout proof above establishes this exactly. Keeping it
                # literal avoids specializing a dynamic stride solely to build
                # a launch-time descriptor.
                strides[-1] = "1"
                host_block_size = ", ".join(map(host_function.literal_expr, block_size))
                arg = TensorDescriptorArg(
                    desc_name,
                    fake_value,
                    f"_helion_tensor_descriptor({tensor_base}, [{', '.join(sizes)}], "
                    f"[{', '.join(strides)}], [{host_block_size}])",
                    permutation,
                )
                self.arguments.append(arg)
            else:
                # Create the regular tensor arg and size/stride args used by
                # tl.make_tensor_descriptor in each device program.
                tensor_arg = self.tensor_arg(fake_value)
                size_args = [
                    self.tensor_size(fake_value, i) for i in range(fake_value.ndim)
                ]
                stride_args = [
                    self.tensor_stride(fake_value, i) for i in range(fake_value.ndim)
                ]
                if permutation is not None:
                    size_args = [size_args[i] for i in permutation]
                    stride_args = [stride_args[i] for i in permutation]
                # The descriptor permutation above makes the last descriptor
                # dimension the proven stride-one dimension. Triton checks this
                # predicate at JIT time, so emit it as a literal even when other
                # dynamic strides are runtime scalars.
                stride_args[-1] = StaticShape(1)
                sizes = ", ".join(arg.name for arg in size_args)
                strides = ", ".join(arg.name for arg in stride_args)
                tensor_descriptor_fn_name = get_tensor_descriptor_fn_name()
                self.preamble.append(
                    statement_from_string(
                        f"{desc_name} = {tensor_descriptor_fn_name}({tensor_arg.name}, [{sizes}], [{strides}], [{block_size_expr}])"
                    )
                )
                arg = TensorDescriptorArg(
                    desc_name,
                    fake_value,
                    None,  # No host_str since this is device-only
                    permutation,
                )
            self._tensor_descriptor_args[key] = arg
        return self._tensor_descriptor_args[key]

    def expr_arg(self, sym: sympy.Expr, origin: Origin) -> NumericArgument:
        if sym not in self._expr_args:
            # A tunable-only expression is constant for each compiled config.
            arg_type = (
                ConstExprArg
                if sym in self._constexpr_exprs
                or CompileEnvironment.current().is_tunable_expr(sym)
                else SymbolArgument
            )
            arg = arg_type(
                name=self.new_var(origin.suggest_var_name()),
                _host_str=origin.host_str(),
            )
            self.arguments.append(arg)
            self._expr_args[sym] = arg
        return self._expr_args[sym]

    def promote_expr_arg_to_constexpr(self, expr: object) -> None:
        """Bake a runtime scalar into the CuTe launcher specialization key.

        This is intended for fail-closed structural lowerings whose generated
        schedule depends on an exact scalar value.  The CuTe launcher keys and
        recompiles ``cutlass.Constexpr`` parameters by value, so a later call
        with a different scalar cannot reuse value-specialized device code.
        """

        if isinstance(expr, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            sym = expr._sympy_()
            if not isinstance(sym, sympy.Expr):
                return
        elif isinstance(expr, sympy.Expr):
            sym = expr
        else:
            return
        self._constexpr_exprs.add(sym)
        previous = self._expr_args.get(sym)
        if previous is None or isinstance(previous, ConstExprArg):
            return
        replacement = ConstExprArg(previous.name, previous.host_str())
        self.arguments[self.arguments.index(previous)] = replacement
        self._expr_args[sym] = replacement

    def constexpr_arg(self, name: str, value: object | None = None) -> bool:
        """Create a constexpr argument, returns True if created, False if already exists."""
        if name in self._constexpr_args:
            return False
        host_str = name if value is None else self._format_constexpr_value(value)
        self._constexpr_args[name] = rv = ConstExprArg(name, host_str)
        self.arguments.append(rv)
        return True

    def resolved_int_constexpr(self, name: str) -> int | None:
        """Return a registered constexpr's value as an int, if it is a static literal."""
        arg = self._constexpr_args.get(name)
        if arg is None:
            return None
        host_str = arg.host_str()
        return int(host_str) if host_str.isdigit() else None

    def constexpr_arg_with_host_def(self, name: str, value: object) -> None:
        """Create a constexpr argument and add its host-side definition if needed."""
        created = self.constexpr_arg(name, value)
        host_expr = self._constexpr_args[name].host_str()
        if created or name not in self._constexpr_host_defs:
            self.codegen.host_statements.append(
                statement_from_string(f"{name} = {host_expr}")
            )
        self._constexpr_host_defs.add(name)

    def host_constexpr_def(self, name: str, host_expr: str) -> str:
        """Define a host-only constant once (launch-grid inputs, not kernel args).

        Returns ``name``; the definition is appended to the host statements
        the first time it is requested for this function.
        """
        if name not in self._constexpr_host_defs:
            self._constexpr_host_defs.add(name)
            self.codegen.host_statements.append(
                statement_from_string(f"{name} = {host_expr}")
            )
        return name

    def _format_constexpr_value(self, value: object) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return repr(value)

        # Extract sympy expression from torch symbolic types
        if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            value = value._sympy_()

        # Handle sympy expressions
        if isinstance(value, sympy.Expr):
            return HostFunction.current().sympy_expr(value, self.config)

        return HostFunction.current().literal_expr(value, self.config)

    def _tensor_property(
        self,
        prop_cls: type[_P],
        fake_value: torch.Tensor,
        dim: int,
        prefix: str,
    ) -> _P:
        key: tuple[object, ...] = (prop_cls, fake_value, dim)
        if prop_cls is TensorSizeArg:
            size = fake_value.size(dim)
            assert isinstance(size, torch.SymInt)
            key = (
                prop_cls,
                CompileEnvironment.current().shape_env.replace(size._sympy_()),
            )
        if key not in self._tensor_properties:
            arg = self.tensor_arg(fake_value)
            prop = prop_cls(f"{arg.name}_{prefix}_{dim}", arg, dim)
            self.arguments.append(prop)
            self._tensor_properties[key] = prop
        return cast("_P", self._tensor_properties[key])

    def tensor_size(self, fake_value: torch.Tensor, dim: int) -> Argument:
        if isinstance(v := fake_value.size(dim), int) or isinstance(
            v._sympy_(), sympy.Integer
        ):
            return StaticShape(int(v))
        return self._tensor_property(TensorSizeArg, fake_value, dim, "size")

    def tensor_stride(self, fake_value: torch.Tensor, dim: int) -> Argument:
        known = CompileEnvironment.current().known_stride(fake_value, dim)
        if known is not None:
            return StaticShape(known)
        return self._tensor_property(TensorStrideArg, fake_value, dim, "stride")

    def sorted_args(self) -> list[Argument]:
        self.arguments.sort(key=lambda arg: arg.sort_key())
        return [
            arg
            for arg in self.arguments
            if not (
                isinstance(arg, TensorArg)
                and self.pallas_internal_scratch_name(arg.fake_value) is not None
            )
        ]

    def codegen_function_def(
        self, *, bounded_cache_request: BoundedCacheRequest | None = None
    ) -> list[ast.stmt]:
        prefix = []
        if self._tensor_descriptor_args:
            prefix.append(
                statement_from_string("helion.runtime.set_triton_allocator()")
            )

        backend = CompileEnvironment.current().backend
        sorted_arguments = self.sorted_args()

        # Separate constexpr args: inline those with known literal values at
        # module level, keep dynamic ones as function parameters
        constexpr_to_inline = [
            arg
            for arg in sorted_arguments
            if isinstance(arg, ConstExprArg) and _is_literal_constexpr(arg)
        ]
        inlined_names = {arg.name for arg in constexpr_to_inline}
        param_args = [
            arg
            for arg in sorted_arguments
            if not isinstance(arg, ConstExprArg) or arg.name not in inlined_names
        ]

        args = [arg.arg_def_node() for arg in param_args]
        # Ordering invariant:
        # [param_args, extra_params, rng_seed, scratch_args, wrapper_only_params].
        # codegen_function_call must match this order — it builds positional args
        # from param_args, extends with extra_params, then build_launcher_args
        # appends rng_seed_buffer.
        args.extend(create_arg(name) for name in self.codegen._extra_params)
        if self.has_rng_ops():
            # Add the seed buffer as a pointer parameter to kernel signature
            assert self.rng_seed_buffer_param_name is not None
            args.append(create_arg(self.rng_seed_buffer_param_name))

        # Add scratch memory parameters (for emit_pipeline on Pallas/TPU)
        for scratch_arg in self._scratch_args:
            args.append(create_arg(scratch_arg.name))
        # Remote-copy arguments are supplied by the Triton launcher rather than
        # the generated host wrapper. Keep their signature order aligned with
        # the launcher's canonical signal-then-scratch append order, independent
        # of which descriptor kind the kernel encounters first.
        remote_copy_params = set(self.triton_remote_copy_scratch_args)
        remote_copy_params.update(self.triton_persistent_state_args)
        if self.triton_remote_copy_signal_arg is not None:
            remote_copy_params.add(self.triton_remote_copy_signal_arg)
        if self.triton_remote_barrier_signal_arg is not None:
            remote_copy_params.add(self.triton_remote_barrier_signal_arg)
        wrapper_only_params = [
            name for name in self.wrapper_only_params if name not in remote_copy_params
        ]
        if self.triton_remote_copy_signal_arg is not None:
            wrapper_only_params.append(self.triton_remote_copy_signal_arg)
        if self.triton_remote_barrier_signal_arg is not None:
            wrapper_only_params.append(self.triton_remote_barrier_signal_arg)
        wrapper_only_params.extend(self.triton_remote_copy_scratch_args)
        wrapper_only_params.extend(self.triton_persistent_state_args)
        args.extend(
            create_arg(name, self.wrapper_only_param_annotations.get(name))
            for name in wrapper_only_params
        )

        # Generate inlined constexpr assignments at module level
        # (e.g., _BLOCK_SIZE_0 = tl.constexpr(256))
        # Use SyntheticLocation to suppress source origin comments on these statements
        with SyntheticLocation():
            for arg in constexpr_to_inline:
                self.codegen.module_statements.append(
                    statement_from_string(
                        backend.inline_constexpr(arg.name, arg.host_str())
                    )
                )

        # Generate preamble to dereference scalar refs (e.g., Pallas 0-dim tensors)
        scalar_preamble: list[ast.AST] = []
        for arg in param_args:
            scalar_preamble.extend(backend.scalar_arg_preamble(arg))

        function_decorator = backend.function_decorator_for_args(param_args)
        decorators = (
            [expr_from_string("requires_nvshmem")] if self.requires_nvshmem else []
        )
        if function_decorator:
            decorators.append(expr_from_string(function_decorator))
        env = CompileEnvironment.current()
        if (
            self.triton_persistent_state_specs
            and env.pdl_exit + env.pdl_entry
            and "wait" not in env.pdl_entry
        ):
            raise exc.PdlStateWithoutWait
        kernel_body: list[ast.stmt] = cast(
            "list[ast.stmt]",
            [
                *scalar_preamble,
                *self.preamble,
                *self.body,
            ],
        )
        body_context = DeviceBodyContext(
            device_function=self,
            args=tuple(args),
            param_args=param_args,
            sorted_arguments=sorted_arguments,
            constexpr_to_inline=constexpr_to_inline,
            wrapper_only_params=wrapper_only_params,
            bounded_cache_request=bounded_cache_request,
            body_start=len(scalar_preamble) + len(self.preamble),
        )
        kernel_body, extra_args = backend.finalize_device_body(
            kernel_body, body_context
        )
        args.extend(extra_args)
        definition = ast_rename(
            create(
                ast.FunctionDef,
                name=self.name,
                args=create_arguments(args),
                body=kernel_body,
                decorator_list=decorators,
                type_params=[],
            ),
            {k: v[0] for k, v in self._variable_renames.items()},
        )
        trailing = backend.finalize_kernel_def(definition, prefix, body_context)
        # Explicit entry/exit markers surround the final body: backend schedule
        # rewrites must neither move the wait past memory nor release earlier
        # than an explicit exit trigger.
        definition.body = [
            *[
                statement_from_string(backend.pdl_instruction(op))
                for op in env.pdl_entry
            ],
            *definition.body,
            *[
                statement_from_string(backend.pdl_instruction(op))
                for op in env.pdl_exit
            ],
        ]
        return [*prefix, definition, *trailing]

    def proven_disjoint_tensor_pairs(self) -> set[frozenset[str]]:
        """Return tensor argument pairs with a cache-safe non-aliasing proof.

        Different user/global tensor arguments are not a non-aliasing proof:
        callers may pass the same tensor, or overlapping views, under multiple
        names. Positively identified wrapper allocations cannot overlap an
        external input or another allocation. Synthetic host-join storage
        instead represents every branch's possible storage, all of which must
        prove disjointness. Two external inputs are disjoint only when their
        current runtime storage spans do not overlap and that predicate is part
        of the bound kernel cache key. The predicate compares whole storages,
        so it also covers host-side views (``x.view(...)``) through the input whose
        storage they share.  Origin type alone is insufficient: host-local
        ``torch.empty``-family outputs commonly carry ``NameOrigin`` too.
        """
        return tensor_pairs_with_proven_disjoint_storage(
            CompileEnvironment.current(),
            {
                arg.name: arg.fake_value
                for arg in self.arguments
                if isinstance(arg, TensorArg)
            },
        )

    def proven_tensor_stride_values(self) -> dict[tuple[str, int], int]:
        """Return tensor strides whose exact value is cache-safe.

        ``static_shapes`` keys the bound kernel on every input tensor's exact
        sizes and strides (see ``_tensor_key``), so under that setting every
        input stride is already a cache-safe fact; otherwise a stride is
        cache-safe only when the ``input_tensor_metadata`` compiler fact or an
        explicit ``hl.specialize`` guard covers it.
        """
        env = CompileEnvironment.current()
        input_metadata_specialized = (
            env.settings.static_shapes
            or "input_tensor_metadata" in env.compiler_fact_specialization_facts
        )
        result: dict[tuple[str, int], int] = {}
        for arg in self.arguments:
            if not isinstance(arg, TensorArg):
                continue
            source = env.tensor_input_source(arg.fake_value)
            runtime = env.runtime_value_for_tensor(arg.fake_value)
            if not isinstance(runtime, torch.Tensor) or isinstance(runtime, FakeTensor):
                if source is not None or not input_metadata_specialized:
                    continue
                # A tensor the host function allocates (``torch.empty_like``)
                # has no runtime value at compile time; its layout is what
                # the traced allocation computed from the specialized input
                # metadata, so a literal traced stride is the stride of every
                # call this bound kernel serves.
                for dim, fake_stride in enumerate(arg.fake_value.stride()):
                    if type(fake_stride) is int:
                        result[arg.name, dim] = fake_stride
                continue
            for dim, fake_stride in enumerate(arg.fake_value.stride()):
                cache_safe = source is not None and (
                    input_metadata_specialized
                    or TensorPropertySource(source, TensorProperty.STRIDE, dim)
                    in env.specialized_strides
                )
                if not cache_safe:
                    continue
                try:
                    traced_stride = env.size_hint(fake_stride)
                except (RuntimeError, TypeError, ValueError):
                    continue
                runtime_stride = runtime.stride(dim)
                if runtime_stride == traced_stride:
                    result[arg.name, dim] = traced_stride
        return result

    def proven_tensor_size_values(
        self,
    ) -> dict[tuple[str, int], tuple[str, int]]:
        """Return cache-safe exact runtime sizes and their generated argument names."""
        env = CompileEnvironment.current()
        if "input_tensor_metadata" not in env.compiler_fact_specialization_facts:
            return {}
        result: dict[tuple[str, int], tuple[str, int]] = {}
        for tensor_arg in self.arguments:
            if not isinstance(tensor_arg, TensorArg) or isinstance(
                tensor_arg, TensorDescriptorArg
            ):
                continue
            fake_tensor = tensor_arg.fake_value
            if env.tensor_input_source(fake_tensor) is None:
                continue
            runtime_tensor = env.runtime_value_for_tensor(fake_tensor)
            if not isinstance(runtime_tensor, torch.Tensor) or isinstance(
                runtime_tensor, FakeTensor
            ):
                continue
            for dim, size in enumerate(fake_tensor.shape):
                if isinstance(size, int) or isinstance(size._sympy_(), sympy.Integer):
                    continue
                expression = env.shape_env.replace(size._sympy_())
                property_arg = self._tensor_properties.get((TensorSizeArg, expression))
                if (
                    not isinstance(property_arg, TensorSizeArg)
                    or env.tensor_input_source(property_arg.tensor_arg.fake_value)
                    is None
                ):
                    continue
                property_runtime = env.runtime_value_for_tensor(
                    property_arg.tensor_arg.fake_value
                )
                if not isinstance(property_runtime, torch.Tensor) or isinstance(
                    property_runtime, FakeTensor
                ):
                    continue
                property_fake_size = property_arg.tensor_arg.fake_value.size(
                    property_arg.dim
                )
                try:
                    runtime_size = int(runtime_tensor.size(dim))
                    source_hint = int(env.size_hint(size))
                    property_size = int(property_runtime.size(property_arg.dim))
                    property_hint = int(env.size_hint(property_fake_size))
                except (RuntimeError, TypeError, ValueError):
                    continue
                if not (runtime_size == source_hint == property_size == property_hint):
                    continue
                result[tensor_arg.name, dim] = (
                    property_arg.name,
                    property_hint,
                )
        return result

    def codegen_function_call(self) -> ast.AST:
        env = CompileEnvironment.current()
        backend = env.backend

        args: list[str] = []
        tensor_host_args: list[str] = []
        arg_objects: list[Argument] = []
        for arg in self.sorted_args():
            # Skip constexpr args that are inlined at module level
            if isinstance(arg, ConstExprArg) and _is_literal_constexpr(arg):
                continue
            if isinstance(arg, ConstExprArg) and arg.name in self._constexpr_host_defs:
                host_arg = arg.name
            else:
                host_arg = arg.host_str()
            if isinstance(arg, TensorArg):
                tensor_host_args.append(host_arg)
            host_arg = backend.transform_host_arg(arg, host_arg, tensor_host_args)
            args.append(host_arg)
            arg_objects.append(arg)

        pid = self.pid
        assert pid is not None

        call_grid_expr = backend.transform_launch_grid(self, pid.codegen_grid())
        # Extra params are positional and must come before any keyword args that
        # build_launcher_args appends (e.g. num_warps=, num_stages=).
        args.extend(self.codegen._extra_params)
        call_args = backend.build_launcher_args(
            args,
            tensor_host_args=tensor_host_args,
            has_rng_ops=self.has_rng_ops(),
            config=self.config,
            has_barrier=self.has_barrier,
            sorted_args=arg_objects,
        )
        # The launcher's return values for output-only tensors, if any.
        output_only_names = backend.output_only_names()
        launcher_call = (
            f"_launcher({self.name}, {{call_grid_expr}}, {', '.join(call_args)})"
        )
        if output_only_names:
            if len(output_only_names) == 1:
                assign_target = output_only_names[0]
            else:
                assign_target = ", ".join(output_only_names)
            call_statement = statement_from_string(
                f"{assign_target} = {launcher_call}",
                call_grid_expr=call_grid_expr,
            )
        else:
            call_statement = statement_from_string(
                launcher_call,
                call_grid_expr=call_grid_expr,
            )
        assert isinstance(call_statement, ExtendedAST)
        # Mark the kernel call so we can find it in codegen_precompile_def
        call_statement._is_kernel_call = True
        return call_statement

    def dead_code_elimination(self) -> None:
        """
        Remove variables that are not used in the function body.
        """

        rename_groups = {
            name: aliases[0] for name, aliases in self._variable_renames.items()
        }
        rw = ReadWrites.from_list([*self.preamble, *self.body])
        while True:
            # An unchanged body reads and writes exactly what ``rw`` recorded,
            # so the fixed point is reached without recounting it.
            changed = dead_assignment_elimination(self.body, self.dce_vars, 1, rw)
            changed |= dead_assignment_elimination(self.preamble, self.dce_vars, 1, rw)
            changed |= dead_lane_loop_elimination(self.body, rename_groups)
            changed |= dead_lane_loop_elimination(self.preamble, rename_groups)
            if not changed:
                break
            rw = ReadWrites.from_list([*self.preamble, *self.body])

        # Drop unused args, but keep placeholder_args (fusion-injected tensor
        # pointers referenced only by placeholder strings, not the AST body).
        # sourceless_prologue_params are intentionally NOT exempted — they are
        # fully inlined by the prologue hook and should be removed by DCE.
        args_to_remove = {
            arg.name
            for arg in self.arguments
            # pyrefly: ignore [unbound-name]
            if arg.name not in rw.reads and arg.name not in self.placeholder_args
        }
        if args_to_remove:
            self.arguments = [
                arg for arg in self.arguments if arg.name not in args_to_remove
            ]
            for cache in cast(
                "list[dict[object, Argument]]",
                [
                    self._tensor_args,
                    self._tensor_descriptor_args,
                    self._expr_args,
                    self._tensor_properties,
                ],
            ):
                for k, v in [*cache.items()]:
                    if v.name in args_to_remove:
                        del cache[k]

    def register_helper_function(
        self, helper_graph_info: HelperFunctionGraphInfo
    ) -> None:
        """Register a helper function to be generated at global scope."""
        name = self.namespace.create_name(helper_graph_info.name, None)
        self.helper_manager.register_helper_function(helper_graph_info, name)

    def register_triton_outlined_helper(
        self,
        name_hint: str,
        body: list[ast.stmt],
        *,
        extra_argument_names: tuple[str, ...] = (),
        noinline: bool = False,
    ) -> tuple[str, tuple[str, ...]]:
        """Outline an opaque body without changing its computation AST."""
        if CompileEnvironment.current().backend_name != "triton":
            raise AssertionError("outlined Triton helpers require the Triton backend")
        cloned_body = cast("list[ast.stmt]", clone_ast(body))
        if tuple(
            ast.dump(statement, include_attributes=False) for statement in cloned_body
        ) != tuple(ast.dump(statement, include_attributes=False) for statement in body):
            raise AssertionError("outlining changed an opaque tile body")
        # Case extraction can carry common, compiler-generated index definitions
        # into more than one branch. Standalone kernels eliminate those too; do
        # the same before choosing helper parameters so dead definitions cannot
        # accidentally turn a module constexpr into a host-wrapper local.
        dead_assignment_elimination(cast("list[ast.AST]", cloned_body), self.dce_vars)
        dead_expression_elimination(cast("list[ast.AST]", cloned_body))
        reads = set(ReadWrites.from_list(cloned_body).reads)
        local_names = {
            node.id
            for statement in cloned_body
            for node in ast.walk(statement)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        helper_names = {name for name, _, _, _ in self.triton_outlined_helpers}
        environment = CompileEnvironment.current()
        for name in reads:
            if not name.startswith("_BLOCK_SIZE_"):
                continue
            suffix = name.removeprefix("_BLOCK_SIZE_")
            if not suffix.isdigit():
                continue
            block_id = int(suffix)
            self.triton_outlined_helper_constexprs[name] = int(
                environment.block_sizes[block_id].from_config_assert(self.config)
            )
        argument_names = [
            argument.name
            for argument in self.arguments
            if argument.name in reads
            and not (
                isinstance(argument, ConstExprArg) and _is_literal_constexpr(argument)
            )
        ]
        argument_names.extend(
            name
            for name in self.wrapper_only_params
            if name in reads and name not in argument_names
        )
        # Outlined roots may read values synthesized by compiler preamble code
        # rather than kernel arguments.  Tensor descriptors are the common
        # example: the preamble constructs ``*_desc`` from the underlying
        # pointer/shape/stride arguments, and the opaque root calls methods on
        # that descriptor.  Treat every such live preamble definition as part
        # of the generated helper ABI instead of leaving a free name that only
        # exists in the parent kernel scope.
        preamble_writes = ReadWrites.from_list(self.preamble).writes
        argument_names.extend(
            name
            for name in preamble_writes
            if name in reads and name not in local_names and name not in argument_names
        )
        argument_names.extend(
            name
            for name in extra_argument_names
            if name in reads and name not in argument_names
        )
        # Ready actions can create an outlined child before their enclosing
        # scheduled helper exists. Capture those compiler-owned parent locals
        # by namespace; ordinary source names must still enter through the
        # kernel, preamble, or explicit generated ABI above.
        argument_names.extend(
            name
            for name in sorted(reads)
            if name.startswith("tile_dependency_")
            and name not in local_names
            and name not in helper_names
            and name not in argument_names
        )
        helper_name = self.namespace.create_name(name_hint, None)
        self.triton_outlined_helpers.append(
            (helper_name, tuple(argument_names), tuple(cloned_body), noinline)
        )
        return helper_name, tuple(argument_names)

    def codegen_helper_functions(self) -> list[ast.stmt]:
        """Generate helper function definitions at global scope."""
        existing_module_assignments = {
            target.id
            for statement in self.codegen.module_statements
            if isinstance(statement, ast.Assign)
            for target in statement.targets
            if isinstance(target, ast.Name)
        }
        helpers = [
            statement_from_string(f"{name} = tl.constexpr({value})")
            for name, value in self.triton_outlined_helper_constexprs.items()
            if name not in existing_module_assignments
        ]
        helpers.extend(self.helper_manager.codegen_helper_functions())
        renames = {name: values[0] for name, values in self._variable_renames.items()}
        argument_by_name = {argument.name: argument for argument in self.arguments}
        for name, argument_names, body, noinline in self.triton_outlined_helpers:
            arguments: list[ast.arg] = []
            for argument_name in argument_names:
                argument = argument_by_name.get(argument_name)
                if argument is None:
                    argument_node = create_arg(argument_name)
                else:
                    argument_node = argument.arg_def_node()
                argument_node.arg = renames.get(argument_node.arg, argument_node.arg)
                arguments.append(argument_node)
            helper_module = ast.Module(
                body=cast("list[ast.stmt]", clone_ast(list(body))),
                type_ignores=[],
            )
            ast_rename(helper_module, renames)
            helpers.append(
                create(
                    ast.FunctionDef,
                    name=name,
                    args=create_arguments(arguments),
                    body=helper_module.body,
                    decorator_list=[
                        expr_from_string(
                            "triton.jit(noinline=True)" if noinline else "triton.jit"
                        )
                    ],
                    type_params=[],
                )
            )
        return helpers

    def flush_deferred_rdim_defs(self, codegen: GenerateAST) -> None:
        """Add all deferred RDIM definitions to host statements."""
        backend = CompileEnvironment.current().backend
        for var_name, expr in self.deferred_rdim_defs:
            expr_str = HostFunction.current().sympy_expr(expr)
            stmt = statement_from_string(
                f"{var_name} = {backend.dynamic_rdim_size_expr(expr_str)}"
            )
            codegen.host_statements.append(stmt)
        self.deferred_rdim_defs.clear()

    def register_scratch(
        self,
        shape: tuple[int | torch.SymInt, ...],
        dtype: torch.dtype | None,
        name_hint: str = "scratch",
        scratch_type: str = "vmem",
        shape_sources: tuple[tuple[torch.Tensor, int] | None, ...] | None = None,
    ) -> str:
        """Register a scratch memory buffer and return its variable name."""
        if CompileEnvironment.current().backend_name != "pallas":
            raise NotImplementedError(
                "register_scratch is only supported by the Pallas backend"
            )
        name = self.new_var(name_hint)
        if shape_sources is not None and len(shape_sources) != len(shape):
            raise ValueError("scratch shape sources must match the scratch rank")
        self._scratch_args.append(
            ScratchArg(name, shape, dtype, scratch_type, shape_sources)
        )
        return name

    def scratch_read_slice(self, name: str) -> str | None:
        """Return the index expression for reading logical data from a padded scratch.

        Returns None if no padding was applied.
        """
        return None

    def register_dma_semaphore(
        self, name_hint: str = "sem", shape: tuple[int, ...] = ()
    ) -> str:
        """Register a DMA semaphore scratch buffer and return its variable name."""
        return self.register_scratch(
            shape, None, name_hint=name_hint, scratch_type="dma_semaphore"
        )

    def get_tensor_read_write_names(self) -> tuple[set[str], set[str]]:
        """Returns AST names of read and written tensors"""
        from helion.language import distributed_ops
        from helion.language import memory_ops
        from helion.language import tile_index
        from helion.language.atomic_ops import ATOMIC_OPS

        read_names: set[str] = set()
        write_names: set[str] = set()
        for graph in self.codegen.codegen_graphs:
            for node in graph.graph.nodes:
                if node.op != "call_function":
                    continue

                def _get_tensor_name(node: torch.fx.Node) -> str | None:
                    tensor_arg = node.args[0]
                    assert isinstance(tensor_arg, torch.fx.Node)
                    # tile.index loads operate on a synthesized FakeTensor
                    # that is not registered in ``tensor_to_origin``; they
                    # are materialized inline by the load codegen rather
                    # than referencing a kernel-arg tensor.
                    if (
                        tensor_arg.op == "call_function"
                        and tensor_arg.target == tile_index
                    ):
                        return None
                    tensor_val = tensor_arg.meta.get("val")
                    assert isinstance(tensor_val, torch.Tensor)
                    try:
                        return self.tensor_arg(tensor_val).name
                    except KeyError:
                        # Device-created tensors are compiler-managed scratch,
                        # not host arguments that need BlockSpec read/write
                        # classification.
                        return None

                if node.target is memory_ops.load:
                    name = _get_tensor_name(node)
                    if name is not None:
                        read_names.add(name)
                elif node.target is memory_ops.store:
                    name = _get_tensor_name(node)
                    if name is not None:
                        write_names.add(name)
                elif node.target in ATOMIC_OPS:
                    name = _get_tensor_name(node)
                    if name is not None:
                        read_names.add(name)
                        write_names.add(name)
                elif node.target is distributed_ops.make_async_remote_copy:
                    if len(node.args) != 5:
                        raise exc.InternalError(
                            RuntimeError(
                                "remote copy was not normalized to "
                                "its five-argument form"
                            )
                        )
                    src_name = _get_tensor_name(node)
                    if src_name is not None:
                        read_names.add(src_name)
                    dst_arg = node.args[3]
                    if isinstance(dst_arg, torch.fx.Node):
                        dst_value = dst_arg.meta.get("val")
                        if isinstance(dst_value, torch.Tensor):
                            with contextlib.suppress(KeyError):
                                write_names.add(self.tensor_arg(dst_value).name)
        return read_names, write_names

    def __enter__(self) -> None:
        try:
            tls.functions.append(self)
        except AttributeError:
            tls.functions = [self]

    def __exit__(self, *args: object) -> None:
        tls.functions.pop()

    @staticmethod
    def current() -> DeviceFunction:
        try:
            return tls.functions[-1]
        except (AttributeError, IndexError):
            raise NoCurrentFunction from None
