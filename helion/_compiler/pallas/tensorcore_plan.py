"""TensorCore plans for Pallas memory operations."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .memory_access import MEMORY_ACCESS_META
from .memory_access import MemoryAccess
from .memory_access import MemoryAccessKind
from .memory_access import indirect_access
from .memory_access import memory_access_mask
from .memory_access import memory_access_value
from .memory_access import tensor_index_positions
from .plan_tiling import ArbitraryIndexPattern
from .plan_tiling import TileBeginWithOffsetPattern
from .plan_tiling import TilePattern

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from ...runtime.config import Config
    from ..device_ir import DeviceIR
    from ..device_ir import GraphInfo
    from .gather import GatherPlan
    from .gather import ScatterPlan


TENSORCORE_PLAN_META = "pallas_tensorcore_plan"
DMA_ACCESS_CAPABLE_META = "pallas_dma_access_capable"


@dataclass(frozen=True)
class DmaAccessSpec:
    """Config-independent structural eligibility for one indirect access."""

    access: MemoryAccess
    index_access: MemoryAccess
    index_block_id: int
    selected_starts: tuple[int, ...]
    selected_extents: tuple[int, ...]

    @property
    def node(self) -> torch.fx.Node:
        return self.access.node

    @property
    def index_node(self) -> torch.fx.Node:
        return self.index_access.node


@dataclass(frozen=True)
class DmaAccessCandidate:
    """One load, optionally paired with its exact state writeback."""

    graph_id: int
    load: DmaAccessSpec
    store: DmaAccessSpec | None
    metadata_tensor_ids: frozenset[int]


@dataclass(frozen=True)
class TensorCorePlan:
    """TensorCore implementation of one memory operation."""

    access: MemoryAccess
    indirect_positions: tuple[int, ...]  # Positions in access.subscript.


@dataclass(frozen=True)
class OneHotGatherPlan(TensorCorePlan):
    """Resident-table one-hot/MXU fallback for an indirect load."""

    plan: GatherPlan


@dataclass(frozen=True)
class OneHotScatterPlan(TensorCorePlan):
    """One-hot fallback for an indirect store."""

    plan: ScatterPlan


@dataclass(frozen=True)
class DmaAccessPlan(TensorCorePlan):
    """Config-resolved local-DMA implementation of an indirect access."""

    spec: DmaAccessSpec
    group_count: int
    transfer_shape: tuple[int, ...]


@dataclass(frozen=True)
class DmaGatherPlan(DmaAccessPlan):
    """Local-DMA implementation of an indirect load."""


@dataclass(frozen=True)
class DmaScatterPlan(DmaAccessPlan):
    """Local-DMA writeback paired with an exact indirect load."""


def _metadata_dma_block_id(index_access: MemoryAccess) -> int | None:
    """Recognize metadata indexing accepted by the indirect DMA scheduler."""
    patterns = index_access.patterns
    group_patterns = [
        pattern for pattern in patterns if isinstance(pattern, TilePattern)
    ]
    if len(group_patterns) != 1 or any(
        not isinstance(
            pattern,
            (ArbitraryIndexPattern, TileBeginWithOffsetPattern, TilePattern),
        )
        for pattern in patterns
    ):
        return None
    group_block_id = group_patterns[0].block_id
    if any(
        isinstance(pattern, TileBeginWithOffsetPattern)
        and pattern.block_id == group_block_id
        for pattern in patterns
    ):
        return None
    return group_block_id


def build_dma_access_spec(
    access: MemoryAccess,
) -> DmaAccessSpec | None:
    """Build config-independent TensorCore DMA eligibility for one access."""
    from ..backend import PallasBackend
    from ..compile_environment import CompileEnvironment

    if access.kind not in (MemoryAccessKind.LOAD, MemoryAccessKind.STORE):
        return None
    if memory_access_mask(access) is not None:
        return None
    indirect = indirect_access(access)
    # The first implementation copies one contiguous member row per address.
    # Keeping the indirect axis leading makes every member an HBM Ref slice;
    # rank >= 3 and a floating payload match the state/cache workloads that
    # justify the per-member DMA setup cost.
    if (
        indirect is None
        or indirect.position != 0
        or access.tensor.ndim < 3
        or not access.tensor.dtype.is_floating_point
        or not access.tensor.is_contiguous()
    ):
        return None

    from .plan_tiling import build_pallas_memory_access

    index_node = indirect.index_node
    index_access = index_node.meta.get(MEMORY_ACCESS_META)
    if (
        not isinstance(index_access, MemoryAccess)
        or index_access.node is not index_node
    ):
        index_access = build_pallas_memory_access(index_node)
    index = memory_access_value(index_access)
    if (
        index_access.kind is not MemoryAccessKind.LOAD
        or index is None
        or index.ndim != 1
        or index.dtype != torch.int32
        or memory_access_mask(index_access) is not None
    ):
        return None

    env = CompileEnvironment.current()
    if not env.settings.static_shapes:
        return None
    block_id = env.resolve_block_id(index.shape[0])
    if block_id is None or _metadata_dma_block_id(index_access) != block_id:
        return None
    backend = env.backend
    assert isinstance(backend, PallasBackend)
    if len(access.subscript) > access.tensor.ndim:
        return None

    selected_starts: list[int] = []
    selected_extents: list[int] = []
    for tensor_dim in range(1, access.tensor.ndim):
        item = (
            access.subscript[tensor_dim]
            if tensor_dim < len(access.subscript)
            else slice(None)
        )
        value = item.meta.get("val") if isinstance(item, torch.fx.Node) else item
        if not isinstance(value, slice) or value.step not in (None, 1):
            return None
        dim_size = env.try_concretize_symint(access.tensor.shape[tensor_dim])
        start = 0 if value.start is None else value.start
        stop = dim_size if value.stop is None else value.stop
        if (
            not isinstance(dim_size, int)
            or not isinstance(start, int)
            or not isinstance(stop, int)
            or not 0 <= start <= stop <= dim_size
        ):
            return None
        alignment = backend._get_pallas_required_alignment(
            access.tensor.ndim - tensor_dim - 1,
            access.tensor.ndim,
            access.tensor.dtype.itemsize * 8,
        )
        if start % alignment != 0:
            return None
        selected_starts.append(start)
        selected_extents.append(stop - start)
    if any(extent <= 0 for extent in selected_extents):
        return None

    # A grouped member is one contiguous HBM region. Reject rectangular
    # subviews whose row-major strides would introduce gaps between rows.
    contiguous_stride = 1
    for dim, extent in reversed(tuple(enumerate(selected_extents, start=1))):
        if extent > 1 and access.tensor.stride(dim) != contiguous_stride:
            return None
        contiguous_stride *= extent

    value = memory_access_value(access)
    if (
        value is None
        or value.ndim != access.tensor.ndim
        or env.resolve_block_id(value.shape[0]) != block_id
        or any(
            env.try_concretize_symint(value.shape[dim]) != extent
            for dim, extent in enumerate(selected_extents, start=1)
        )
    ):
        return None
    last_dim = env.try_concretize_symint(value.shape[-1])
    if not isinstance(last_dim, int) or last_dim <= 0 or last_dim % 128 != 0:
        return None
    return DmaAccessSpec(
        access,
        index_access,
        block_id,
        tuple(selected_starts),
        tuple(selected_extents),
    )


def _exact_dma_layout(
    tensor: torch.Tensor,
) -> tuple[int, tuple[int, ...], tuple[int, ...]] | None:
    if not tensor.is_contiguous():
        return None
    try:
        return (
            int(tensor.storage_offset()),
            tuple(int(size) for size in tensor.shape),
            tuple(int(stride) for stride in tensor.stride()),
        )
    except (TypeError, ValueError):
        return None


def _load_consumed_before_store(
    load: torch.fx.Node,
    store: torch.fx.Node,
) -> bool:
    """Whether shared scratch may be overwritten at ``store`` without aliasing."""
    positions = {node: position for position, node in enumerate(load.graph.nodes)}
    store_position = positions[store]
    seen: set[torch.fx.Node] = set()
    stack = [*load.users]
    while stack:
        node = stack.pop()
        if node in seen or node is store:
            continue
        seen.add(node)
        if positions[node] > store_position:
            return False
        stack.extend(node.users)
    return True


def build_dma_access_candidates(
    graphs: Sequence[GraphInfo],
    excluded_storage_ids: frozenset[int] = frozenset(),
) -> tuple[DmaAccessCandidate, ...]:
    """Find read-only gathers and exact read/write state pairs."""
    from ...language import distributed_ops
    from ...language import memory_ops
    from ...language.atomic_ops import ATOMIC_OPS
    from .plan_tiling import build_pallas_memory_access

    memory_targets = ATOMIC_OPS | {memory_ops.load, memory_ops.store}
    excluded = set(excluded_storage_ids)
    accesses_by_storage: dict[int, list[tuple[GraphInfo, MemoryAccess]]] = {}
    for owner in graphs:
        for node in owner.graph.nodes:
            if (
                node.op == "call_function"
                and node.target is distributed_ops.make_async_remote_copy
            ):
                for position in (0, 3):
                    tensor_node = node.args[position]
                    tensor = (
                        tensor_node.meta.get("val")
                        if isinstance(tensor_node, torch.fx.Node)
                        else None
                    )
                    if isinstance(tensor, torch.Tensor):
                        excluded.add(id(tensor.untyped_storage()))
            if node.op != "call_function" or node.target not in memory_targets:
                continue
            access = node.meta.get(MEMORY_ACCESS_META)
            if not isinstance(access, MemoryAccess) or access.node is not node:
                access = build_pallas_memory_access(node)
            accesses_by_storage.setdefault(
                id(access.tensor.untyped_storage()), []
            ).append((owner, access))

    candidates: list[DmaAccessCandidate] = []
    for storage_id, accesses in accesses_by_storage.items():
        # A tensor and all of its aliases must observe one access strategy. A
        # direct, atomic, or second same-direction access beside an indirect
        # DMA could otherwise observe a different memory generation.
        layout = _exact_dma_layout(accesses[0][1].tensor)
        graph_id = accesses[0][0].graph_id
        tensor_node = accesses[0][1].tensor_node
        if (
            storage_id in excluded
            or layout is None
            or any(
                owner.graph_id != graph_id
                or access.tensor_node is not tensor_node
                or _exact_dma_layout(access.tensor) != layout
                for owner, access in accesses
            )
        ):
            continue

        specs: list[DmaAccessSpec] = []
        metadata_ids: set[int] = set()
        for _owner, access in accesses:
            current = build_dma_access_spec(access)
            if current is None:
                break
            metadata = current.index_access.tensor
            metadata_layout = _exact_dma_layout(metadata)
            metadata_accesses = accesses_by_storage.get(
                id(metadata.untyped_storage()), ()
            )
            try:
                metadata_bytes = int(metadata.numel()) * metadata.dtype.itemsize
            except (TypeError, ValueError):
                break
            if (
                metadata_layout is None
                or id(metadata.untyped_storage()) in excluded
                or metadata_bytes > (16 << 20)
                or not metadata_accesses
                or any(
                    metadata_owner.graph_id != graph_id
                    or metadata_access.kind is not MemoryAccessKind.LOAD
                    or metadata_access.tensor_node
                    is not current.index_access.tensor_node
                    or _exact_dma_layout(metadata_access.tensor) != metadata_layout
                    for metadata_owner, metadata_access in metadata_accesses
                )
            ):
                break
            metadata_ids.update(
                id(metadata_access.tensor)
                for _metadata_owner, metadata_access in metadata_accesses
            )
            specs.append(current)
        else:
            directions = [spec.access.kind for spec in specs]
            if directions == [MemoryAccessKind.LOAD]:
                candidates.append(
                    DmaAccessCandidate(
                        graph_id,
                        specs[0],
                        None,
                        frozenset(metadata_ids),
                    )
                )
            elif directions == [MemoryAccessKind.LOAD, MemoryAccessKind.STORE]:
                load, store = specs
                # Reusing one VMEM stage is valid only for an exact in-place
                # state update. Equivalent-looking but distinct index loads are
                # deliberately rejected because their values may differ.
                if (
                    store.index_node is load.index_node
                    and store.access.tensor_node is load.access.tensor_node
                    and store.index_block_id == load.index_block_id
                    and store.selected_starts == load.selected_starts
                    and store.selected_extents == load.selected_extents
                    and _load_consumed_before_store(load.node, store.node)
                ):
                    candidates.append(
                        DmaAccessCandidate(
                            graph_id,
                            load,
                            store,
                            frozenset(metadata_ids),
                        )
                    )
    return tuple(candidates)


def dma_access_admission(
    candidates: Sequence[DmaAccessCandidate],
    owner_block_extents: Mapping[int, Mapping[int, int]],
    block_size_ranges: Mapping[int, tuple[int, int]],
) -> tuple[set[torch.fx.Node], dict[int, tuple[int, ...]]]:
    """Admit candidates sharing at least one jointly legal block size."""

    grouped: dict[int, list[tuple[DmaAccessCandidate, set[int]]]] = {}
    for candidate in candidates:
        spec = candidate.load
        extent = owner_block_extents.get(candidate.graph_id, {}).get(
            spec.index_block_id
        )
        bounds = block_size_ranges.get(spec.index_block_id)
        if extent is None or bounds is None:
            continue
        table_rows = spec.access.tensor.shape[0]
        if not isinstance(table_rows, int):
            continue
        block_size, maximum = bounds
        legal: set[int] = set()
        while block_size <= maximum:
            if extent % block_size == 0 and block_size <= table_rows:
                legal.add(block_size)
            block_size *= 2
        if legal:
            grouped.setdefault(spec.index_block_id, []).append((candidate, legal))

    result: set[torch.fx.Node] = set()
    legal_sizes: dict[int, tuple[int, ...]] = {}
    for block_id, entries in grouped.items():
        common = set.intersection(*(sizes for _candidate, sizes in entries))
        if not common:
            continue
        legal_sizes[block_id] = tuple(sorted(common))
        for candidate, _sizes in entries:
            result.add(candidate.load.node)
            if candidate.store is not None:
                result.add(candidate.store.node)
    return result, legal_sizes


def dma_autotuner_floor(
    legal_sizes: tuple[int, ...], min_size: int, autotuner_min: int
) -> int:
    """Choose a legal first generated value nearest the existing floor."""
    floor = max(min_size, autotuner_min)
    return next((size for size in legal_sizes if size >= floor), legal_sizes[-1])


def _dma_owner_block_extents(device_ir: DeviceIR) -> dict[int, dict[int, int]]:
    """Return static scheduler extents used by indirect DMA admission."""
    from ...language import _tracing_ops
    from ..compile_environment import CompileEnvironment
    from ..device_ir import ForLoopGraphInfo
    from ..device_ir import control_flow_parent_entries
    from ..device_ir import device_loop_bounds

    env = CompileEnvironment.current()
    result: dict[int, dict[int, int]] = {}
    for root_id, block_ids in zip(
        device_ir.root_ids, device_ir.grid_block_ids, strict=True
    ):
        extents: dict[int, int] = {}
        for block_id in block_ids:
            size = env.block_sizes[block_id].size
            if not isinstance(size, (int, torch.SymInt)):
                break
            extent = env.try_concretize_symint(size)
            if not isinstance(extent, int) or extent <= 0:
                break
            extents[block_id] = extent
        if len(extents) == len(block_ids):
            result[root_id] = extents

    parent_entries = control_flow_parent_entries(device_ir.graphs)
    parents = {graph_id: entry[0] for graph_id, entry in parent_entries.items()}
    root_graphs = {
        device_ir.graphs[root_id].graph
        for root_id in device_ir.root_ids
        if 0 <= root_id < len(device_ir.graphs)
    }
    for graph_info in device_ir.graphs:
        if (
            not isinstance(graph_info, ForLoopGraphInfo)
            or len(graph_info.block_ids) != 1
        ):
            continue
        parent = parents.get(graph_info.graph_id)
        block_id = graph_info.block_ids[0]
        bounds = device_loop_bounds(graph_info, parents, block_id)
        if parent is None or parent.graph not in root_graphs or bounds is None:
            continue
        raw_begin, raw_end = bounds
        if not isinstance(raw_begin, (int, torch.SymInt)) or not isinstance(
            raw_end, (int, torch.SymInt)
        ):
            continue
        begin = env.try_concretize_symint(raw_begin)
        end = env.try_concretize_symint(raw_end)
        if not isinstance(begin, int) or not isinstance(end, int) or end <= begin:
            continue
        if parent.target is _tracing_ops._for_loop_step:
            steps = parent.args[4]
            step = steps[0] if isinstance(steps, (list, tuple)) and steps else None
            step = step.meta.get("val") if isinstance(step, torch.fx.Node) else step
            if step not in (0, 1):
                continue
        result[graph_info.graph_id] = {block_id: end - begin}
    return result


def indirect_access_modes(device_ir: DeviceIR) -> tuple[str, ...]:
    """Return kernel-wide indirect access modes worth exposing to autotuning."""
    from ... import exc
    from ...language import memory_ops
    from ..compile_environment import CompileEnvironment
    from ..compile_environment import FixedBlockSizeSource
    from ..device_ir import ForLoopGraphInfo
    from .gather import one_hot_access_may_fit_vmem
    from .gather import one_hot_access_supported
    from .plan_tiling import build_pallas_memory_access
    from .view_ops import indirect_loads_requiring_resident_refs

    indirect_accesses: dict[torch.fx.Node, MemoryAccess] = {}
    for graph_info in device_ir.graphs:
        for node in graph_info.graph.nodes:
            if node.op != "call_function" or node.target not in (
                memory_ops.load,
                memory_ops.store,
            ):
                continue
            access = build_pallas_memory_access(node)
            if tensor_index_positions(access):
                indirect_accesses[node] = access
    indirect_nodes = set(indirect_accesses)
    if not indirect_nodes:
        return ()

    required_dma_loads = (
        indirect_loads_requiring_resident_refs(device_ir.graphs) & indirect_nodes
    )
    env = CompileEnvironment.current()
    config_spec = env.config_spec
    block_size_ranges = {
        spec.block_id: (spec.min_size, spec.max_size)
        for spec in config_spec.block_sizes
    }
    for info in env.block_sizes:
        if isinstance(info.block_size_source, FixedBlockSizeSource):
            value = env.try_concretize_symint(info.block_size_source.value)
            if isinstance(value, int):
                block_size_ranges[info.block_id] = (value, value)
    block_size_minimums = {
        spec.block_id: spec.min_size for spec in config_spec.block_sizes
    }
    dma_access_nodes, legal_dma_sizes = (
        (set(), {})
        if env.settings.pallas_interpret
        else dma_access_admission(
            build_dma_access_candidates(device_ir.graphs),
            _dma_owner_block_extents(device_ir),
            block_size_ranges,
        )
    )
    # Make the generated fragment's first value jointly legal. Larger invalid
    # powers remain ordinary skippable configs, while DMA-only kernels retain a
    # compilable default even when grid heuristics raised ``autotuner_min``.
    for block_id, legal in legal_dma_sizes.items():
        try:
            spec = config_spec.block_sizes.block_id_lookup(block_id)
        except KeyError:
            continue
        spec.autotuner_min = dma_autotuner_floor(
            legal, spec.min_size, spec.autotuner_min
        )
    # Specs contain graph-local FX nodes. Per-config graph copies must rebuild
    # them, so only copy this capability bit through node metadata.
    for node in dma_access_nodes:
        node.meta[DMA_ACCESS_CAPABLE_META] = True
    config_spec.pallas_indirect_dma_requires_fori = any(
        isinstance(graph_info, ForLoopGraphInfo)
        and any(node in dma_access_nodes for node in graph_info.graph.nodes)
        for graph_info in device_ir.graphs
    )
    one_hot_structural_nodes = {
        node
        for node, access in indirect_accesses.items()
        if one_hot_access_supported(access)
    }
    one_hot_capable_nodes = {
        node
        for node in one_hot_structural_nodes
        if (
            indirect_accesses[node].kind is not MemoryAccessKind.LOAD
            or one_hot_access_may_fit_vmem(
                node,
                indirect_accesses[node].tensor,
                indirect_accesses[node].subscript,
                block_size_minimums,
            )
        )
    }
    # DMA-capable accesses use DMA in that mode; every other indirect access
    # still falls back to one-hot and must have at least one potentially legal
    # tiling. Conversely, the one-hot mode must support the whole kernel.
    dma_legal = (
        bool(dma_access_nodes)
        and required_dma_loads <= dma_access_nodes
        and indirect_nodes - dma_access_nodes <= one_hot_capable_nodes
    )
    one_hot_legal = not required_dma_loads and indirect_nodes <= one_hot_capable_nodes
    if dma_legal and one_hot_legal:
        return ("one_hot", "dma")
    if dma_legal:
        # DMA-only defaults must not pick a heuristic value above the jointly
        # legal domain. Powers below this maximum remain divisors as well.
        for block_id, legal in legal_dma_sizes.items():
            with contextlib.suppress(KeyError):
                config_spec.block_sizes.block_id_lookup(block_id).update_max(legal[-1])
        return ("dma",)
    if one_hot_legal:
        return ("one_hot",)

    unsupported_without_dma = (
        indirect_nodes - one_hot_structural_nodes - dma_access_nodes
    )
    if unsupported_without_dma:
        raise exc.BackendUnsupported(
            "pallas",
            "an indirect access is unsupported by both one-hot and DMA lowering; "
            "use one unmasked tensor index and a supported load/store dtype and shape",
        )
    oversized_without_dma = (
        one_hot_structural_nodes - one_hot_capable_nodes - dma_access_nodes
    )
    if oversized_without_dma:
        raise exc.BackendUnsupported(
            "pallas",
            "an indirect access has a minimum resident one-hot block above the "
            "VMEM threshold and is not eligible for indirect DMA; reduce its "
            "untiled dimensions or dtype size, or use a DMA-compatible access",
        )
    raise exc.BackendUnsupported(
        "pallas",
        "an indirect resident access requires DMA but is not eligible for a DMA "
        "schedule",
    )


def _concrete_size(size: int | torch.SymInt, config: Config) -> int | None:
    from ..compile_environment import CompileEnvironment

    env = CompileEnvironment.current()
    concrete = env.try_concretize_symint(size)
    if isinstance(concrete, int):
        return concrete
    block_id = env.get_block_id(size)
    if block_id is None:
        return None
    block_size = env.block_sizes[block_id].from_config(config)
    return block_size if isinstance(block_size, int) else None


def build_dma_access_plan(
    access: MemoryAccess,
    positions: list[int],
    config: Config,
) -> DmaAccessPlan | None:
    """Resolve a structural DMA access for one concrete config."""
    spec = build_dma_access_spec(access)
    if spec is None or positions != [0]:
        return None
    # Raw HBM refs have no BlockSpec to apply data-member TilePatterns.
    if any(isinstance(pattern, TilePattern) for pattern in access.patterns):
        return None
    index = memory_access_value(spec.index_access)
    value = memory_access_value(access)
    if index is None or value is None:
        return None
    group_count = _concrete_size(index.shape[0], config)
    table_rows = _concrete_size(access.tensor.shape[0], config)
    transfer_shape = tuple(_concrete_size(size, config) for size in value.shape)
    if (
        group_count is None
        or table_rows is None
        or group_count <= 0
        or group_count > table_rows
        or None in transfer_shape
        or not transfer_shape
        or transfer_shape[0] != group_count
    ):
        return None
    concrete_shape = tuple(size for size in transfer_shape if size is not None)
    if (
        concrete_shape[1:] != spec.selected_extents
        or any(size <= 0 for size in concrete_shape)
        or concrete_shape[-1] % 128 != 0
    ):
        return None
    plan_type = (
        DmaGatherPlan if access.kind is MemoryAccessKind.LOAD else DmaScatterPlan
    )
    return plan_type(
        access,
        tuple(positions),
        spec,
        group_count,
        concrete_shape,
    )


def _one_hot_gather(
    access: MemoryAccess, positions: list[int], config: Config
) -> OneHotGatherPlan:
    from .gather import build_gather_plan

    plan = build_gather_plan(
        access.tensor,
        list(access.subscript),
        positions,
        list(access.patterns),
        config,
        len(access.node.args) > 2 and access.node.args[2] is not None,
    )
    return OneHotGatherPlan(access, tuple(positions), plan)


def _one_hot_scatter(
    access: MemoryAccess, positions: list[int], _config: Config
) -> OneHotScatterPlan:
    from .gather import build_scatter_plan

    plan = build_scatter_plan(
        access.tensor,
        list(access.subscript),
        positions,
        len(access.node.args) > 3 and access.node.args[3] is not None,
    )
    return OneHotScatterPlan(access, tuple(positions), plan)


def select_tensorcore_plan(
    access: MemoryAccess, config: Config
) -> TensorCorePlan | None:
    """Choose one concrete TensorCore implementation for a memory access."""
    positions = list(tensor_index_positions(access))
    if not positions:
        return None
    if config.get(
        "pallas_indirect_access_mode", "one_hot"
    ) == "dma" and access.node.meta.get(DMA_ACCESS_CAPABLE_META, False):
        plan = build_dma_access_plan(access, positions, config)
        if plan is None:
            from ...exc import InvalidConfig

            raise InvalidConfig(
                "pallas_indirect_access_mode='dma' is not legal for this "
                "block-size configuration"
            )
        return plan
    if access.kind is MemoryAccessKind.LOAD:
        return _one_hot_gather(access, positions, config)
    if access.kind is MemoryAccessKind.STORE:
        return _one_hot_scatter(access, positions, config)
    op = access.node.target
    op_name = getattr(op, "__name__", str(op))
    raise NotImplementedError(
        f"Pallas: tensor-indexed memory op is not supported for op={op_name}."
    )


def build_tensorcore_plans(graphs: list[GraphInfo], config: Config) -> None:
    """Select TensorCore plans after shared memory analysis."""
    from .memory_access import MEMORY_ACCESS_META
    from .memory_access import MemoryAccess

    for graph_info in graphs:
        for node in graph_info.graph.nodes:
            access = node.meta.get(MEMORY_ACCESS_META)
            if isinstance(access, MemoryAccess):
                node.meta[TENSORCORE_PLAN_META] = select_tensorcore_plan(access, config)
