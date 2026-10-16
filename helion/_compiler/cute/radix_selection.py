"""Prove a read-only unordered selection and its scalar slot consumers."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import operator
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch._subclasses import FakeTensor
from torch.fx import Node

from ... import exc
from ...language import _tracing_ops
from ...language import memory_ops
from ..compile_environment import CompileEnvironment
from ..compile_environment import RuntimeInputSpecialization
from ..device_ir import RootGraphInfo
from ..host_function import HostFunction
from ..variable_origin import GridOrigin
from .fragment_storage import aligned_shared_bytes
from .memory_ops import runtime_tensors_are_proven_disjoint
from .ordered_selection import unordered_topk_args
from .radix_selection_config import ALGORITHM_KEY
from .radix_selection_config import HISTOGRAM_KEY
from .register_loads import host_load_is_readonly
from .row_fragment import row_fragment_arange
from .tcgen05_config import CuteTcgen05Config

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...autotuner.config_spec import ConfigSpec
    from ..device_ir import GraphInfo
    from ..tile_dispatch import TileStrategyDispatch

FLOAT_DTYPES = frozenset((torch.float16, torch.bfloat16, torch.float32))
SCALAR_DTYPES = FLOAT_DTYPES | frozenset(
    (
        torch.bool,
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float64,
    )
)
INT32_MAX = (1 << 31) - 1
_PACKET_ALIGNMENT_KEY = "cute_radix_packet_alignment_32_v1"


def _packet_alignment_residue(values: Sequence[object]) -> int | None:
    if (
        len(values) != 1
        or not isinstance(values[0], torch.Tensor)
        or isinstance(values[0], FakeTensor)
        or values[0].device.type == "meta"
    ):
        return None
    return int(values[0].data_ptr()) % 32


def register_radix_packet_alignment(
    env: CompileEnvironment, graph: RadixSelectionGraph
) -> None:
    """An operation-local 32-byte fact; ordinary vector caches stay unchanged."""
    owner = env.tensor_alignment_owner(graph.x)
    if owner is None:
        return
    source, _offset_bytes = owner
    env.register_runtime_input_specialization(
        _PACKET_ALIGNMENT_KEY,
        RuntimeInputSpecialization(
            sources=(source,),
            classifier_identity=("radix_packet_base_alignment_32_v1", repr(source)),
            classifier=_packet_alignment_residue,
            reusable_tensor_properties=frozenset(("data_ptr",)),
        ),
    )


def radix_packet_base_is_aligned(env: CompileEnvironment, tensor: torch.Tensor) -> bool:
    owner = env.tensor_alignment_owner(tensor)
    if owner is not None:
        source, offset_bytes = owner
        specialization = env.runtime_input_specializations.get(_PACKET_ALIGNMENT_KEY)
        residue = env.bound_runtime_input_specialization_results.get(
            _PACKET_ALIGNMENT_KEY
        )
        return (
            specialization is not None
            and specialization.sources == (source,)
            and type(residue) is int
            and (residue + offset_bytes) % 32 == 0
        )
    if env.tensor_storage_is_compiler_allocated(tensor):
        offset = tensor.storage_offset()
        return type(offset) is int and offset * tensor.element_size() % 32 == 0
    return False


def tensor_value(node: object) -> torch.Tensor | None:
    if not isinstance(node, Node):
        return None
    value = node.meta.get("val")
    return value if isinstance(value, torch.Tensor) else None


def strip_alias(node: object) -> object:
    while isinstance(node, Node) and node.target in (
        _tracing_ops._new_var,
        torch.ops.aten.alias.default,
    ):
        if len(node.args) != 1 or node.kwargs:
            return node
        source = node.args[0]
        value, original = tensor_value(node), tensor_value(source)
        if (
            value is None
            or original is None
            or value.dtype != original.dtype
            or value.shape != original.shape
        ):
            return node
        node = source
    return node


def literal_scalar(node: object, dtype: torch.dtype | None = None) -> object:
    node = strip_alias(node)
    if type(node) in (int, float, bool):
        return node
    value = tensor_value(node)
    if (
        not isinstance(node, Node)
        or node.target is not torch.ops.aten.scalar_tensor.default
        or len(node.args) != 1
        or type(node.args[0]) not in (int, float, bool)
        or value is None
        or value.ndim != 0
        or dtype is not None
        and value.dtype != dtype
    ):
        return None
    return node.args[0]


def _int32_literal(node: object, expected: int) -> bool:
    value = literal_scalar(node, torch.int32)
    return type(value) is int and value == expected


def _call(node: object, target: object, count: int) -> Node | None:
    node = strip_alias(node)
    return (
        node
        if isinstance(node, Node)
        and node.op == "call_function"
        and node.target is target
        and len(node.args) == count
        and not node.kwargs
        else None
    )


def _host_tensor(node: object) -> torch.Tensor | None:
    if (
        not isinstance(node, Node)
        or node.target is not _tracing_ops._host_tensor
        or len(node.args) != 1
        or type(node.args[0]) is not str
        or node.kwargs
    ):
        return None
    return tensor_value(node)


def _static_layout(tensor: torch.Tensor) -> bool:
    return (
        tensor.dtype in SCALAR_DTYPES
        and all(type(size) is int and size > 0 for size in tensor.shape)
        and all(type(stride) is int and stride >= 0 for stride in tensor.stride())
        and sum(
            (size - 1) * stride
            for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
        )
        <= ((1 << 63) - 1) // tensor.element_size()
    )


def _row_block(node: object, rows: int) -> int | None:
    if (
        not isinstance(node, Node)
        or node.target is not _tracing_ops._get_symnode
        or len(node.args) != 1
        or node.kwargs
    ):
        return None
    value = node.meta.get("val")
    if not isinstance(value, torch.SymInt):
        return None
    origin = HostFunction.current().expr_to_origin.get(value._sympy_())
    if origin is None or type(origin.origin) is not GridOrigin:
        return None
    bid = origin.origin.block_id
    env = CompileEnvironment.current()
    block = env.block_sizes[bid]
    return (
        bid
        if not block.reduction and env.specialize_expr(block.numel) == rows
        else None
    )


def _closure(node: Node) -> frozenset[Node]:
    pending = [node]
    seen: set[Node] = set()
    while pending:
        value = pending.pop()
        if value in seen:
            continue
        seen.add(value)
        pending.extend(value.all_input_nodes)
    return frozenset(seen)


@dataclass(frozen=True)
class RadixSelectionGraph:
    root_graph: torch.fx.Graph
    root_graph_id: int
    selection: Node
    source: Node
    row: Node
    row_block_id: int
    columns: Node
    x_node: Node
    x: torch.Tensor
    source_load: Node
    length: Node
    rows: int
    cols: int
    logical_cols: int
    k: int
    largest: bool
    source_nodes: frozenset[Node]
    slot_domains: dict[Node, int]
    loads: tuple[Node, ...]
    stores: tuple[Node, ...]
    value_getters: frozenset[Node]
    index_getters: frozenset[Node]


@dataclass(frozen=True)
class RadixSelectionPlan:
    graph: RadixSelectionGraph
    threads: int
    cluster: int
    candidate_capacity: int
    histogram_mode: str = "direct"
    candidate_key_cache: bool = False
    scalar_candidates: bool = False
    valid_domain: bool = False
    ticket_stores: bool = False
    packet_bits: int = 128
    scan_unroll: int = 1
    cached_payload_inverse: bool = False

    @property
    def cache_keys_enabled(self) -> bool:
        return self.candidate_key_cache and self.candidate_capacity > 0

    @property
    def cached_key_dtype(self) -> torch.dtype:
        # Cache the full oriented key, including its invalid class and all
        # equality bits; digits alone cannot reconstruct the final threshold.
        return torch.uint16 if self.primary_width == 16 else torch.uint32

    @property
    def packet(self) -> int:
        return self.packet_bits // (8 * self.graph.x.element_size())

    @property
    def primary_width(self) -> int:
        return self.graph.x.element_size() * 8

    @property
    def valid_cols(self) -> int:
        # A declared iota may cover only part of the physical input row.
        return min(self.graph.cols, self.graph.logical_cols)

    @property
    def cta_cols(self) -> int:
        return (
            (self.graph.logical_cols + self.cluster * self.packet - 1)
            // (self.cluster * self.packet)
            * self.packet
        )

    @property
    def column_dtype(self) -> torch.dtype:
        # Include padded packets and ceiling additions before choosing Int32.
        step = self.threads * self.packet
        scan = max(self.cta_cols, self.candidate_capacity)
        last_slot = ((scan + step - 1) // step) * step - 1
        maximum = max(
            (self.cluster - 1) * self.cta_cols + last_slot,
            scan + step - 1,
        )
        return torch.int32 if maximum <= INT32_MAX else torch.int64

    @property
    def allocations(self) -> tuple[tuple[str, torch.dtype, int], ...]:
        allocations = (
            ("histogram", torch.int32, 256),
            ("warp_totals", torch.int32, 8),
            ("selected_indices", torch.int32, self.graph.k),
            ("primary_prefix", torch.int32, 1),
            ("primary_mask", torch.int32, 1),
            ("coarse_prefix", torch.int32, 1),
            ("coarse_mask", torch.int32, 1),
            ("remaining", torch.int32, 1),
            (
                "length_slot",
                cast("torch.Tensor", tensor_value(self.graph.length)).dtype,
                1,
            ),
            ("candidate_counter", torch.int32, 1),
            ("counters", torch.int32, 2),
            ("metadata", torch.int32, 4),
        )
        if self.ticket_stores:
            allocations = tuple(
                allocation
                for allocation in allocations
                if allocation[0] not in ("selected_indices", "metadata")
            )
        if self.candidate_capacity:
            allocations += (
                ("candidate_indices", torch.int32, self.candidate_capacity),
            )
        if self.cache_keys_enabled:
            allocations += (
                ("candidate_keys", self.cached_key_dtype, self.candidate_capacity),
            )
        return allocations

    @property
    def shared_bytes(self) -> int:
        return sum(
            aligned_shared_bytes(count, dtype)
            for _name, dtype, count in self.allocations
        )


def _rank_load(rank: object) -> Node | None:
    """Prove the canonical F32 integer-rank operation, independent of names."""
    rank = _call(rank, torch.ops.aten.where.self, 3)
    if rank is None or not _int32_literal(rank.args[1], INT32_MAX):
        return None
    gt = _call(rank.args[0], torch.ops.aten.gt.Scalar, 2)
    sub = _call(rank.args[2], torch.ops.aten.sub.Tensor, 2)
    if (
        gt is None
        or sub is None
        or type(gt.args[1]) is not int
        or gt.args[1] != 0x7F800000
    ):
        return None
    magnitude = _call(gt.args[0], torch.ops.aten.bitwise_and.Scalar, 2)
    sign = _call(sub.args[1], torch.ops.aten.__rshift__.Scalar, 2)
    xor = _call(sub.args[0], torch.ops.aten.bitwise_xor.Tensor, 2)
    if (
        magnitude is None
        or sign is None
        or xor is None
        or type(magnitude.args[1]) is not int
        or magnitude.args[1] != INT32_MAX
        or type(sign.args[1]) is not int
        or sign.args[1] != 31
        or strip_alias(magnitude.args[0]) is not strip_alias(sign.args[0])
        or {strip_alias(arg) for arg in xor.args} != {magnitude, sign}
    ):
        return None
    bits = _call(magnitude.args[0], torch.ops.aten.view.dtype, 2)
    if bits is None or bits.args[1] is not torch.int32:
        return None
    value = strip_alias(bits.args[0])
    conversion = _call(value, torch.ops.prims.convert_element_type.default, 2)
    if conversion is not None:
        if conversion.args[1] is not torch.float32:
            return None
        value = strip_alias(conversion.args[0])
    load = _call(value, memory_ops.load, 4)
    fake = tensor_value(value)
    if load is None or fake is None or fake.dtype not in FLOAT_DTYPES:
        return None
    if conversion is None and fake.dtype != torch.float32:
        return None
    width = fake.shape
    for node in _closure(rank) - _closure(load):
        fake_node = tensor_value(node)
        if fake_node is not None and (
            fake_node.dtype not in (torch.int32, torch.bool)
            and node is not conversion
            or fake_node.ndim != 0
            and fake_node.shape != width
        ):
            return None
    return load


def _mask_columns(mask: object, columns: Node, cols: int) -> Node | None:
    mask = _call(mask, torch.ops.aten.bitwise_and.Tensor, 2)
    if mask is None:
        return None
    for physical, dynamic in (mask.args, tuple(reversed(mask.args))):
        physical = _call(physical, torch.ops.aten.lt.Scalar, 2)
        dynamic = _call(dynamic, torch.ops.aten.lt.Tensor, 2)
        if (
            physical is not None
            and dynamic is not None
            and strip_alias(physical.args[0]) is columns
            and type(physical.args[1]) is int
            and physical.args[1] == cols
            and strip_alias(dynamic.args[0]) is columns
        ):
            length = strip_alias(dynamic.args[1])
            return length if isinstance(length, Node) else None
    return None


def _suffix_domains(
    graph: RadixSelectionGraph,
) -> dict[Node, int] | None:
    """Prove identity selected-slot maps and exact declared padding domains."""
    memo: dict[Node, int] = {graph.length: 0}
    visiting: set[Node] = set()
    getters = graph.value_getters | graph.index_getters

    def infer(node: object) -> int | None:
        if not isinstance(node, Node):
            return 0 if type(node) in (int, float, bool) else None
        if node in memo:
            return memo[node]
        if node in visiting or node in graph.source_nodes:
            return None
        value = tensor_value(node)
        if value is None or value.dtype not in SCALAR_DTYPES or value.ndim > 1:
            return None
        if any(type(size) is not int or size <= 0 for size in value.shape):
            return None
        visiting.add(node)
        result = calculate(node, value)
        visiting.remove(node)
        expected = value.size(0) if value.ndim else 0
        if result is None or result != expected:
            return None
        memo[node] = result
        return result

    def calculate(node: Node, value: torch.Tensor) -> int | None:
        if node in getters:
            return graph.k
        if literal_scalar(node) is not None:
            return 0
        if (progression := row_fragment_arange(node)) is not None:
            return progression[2] if progression[:2] == (0, 1) else None
        if node.target is memory_ops.load:
            if len(node.args) != 4 or node.args[3] is not None or node.kwargs:
                return None
            target = _host_tensor(node.args[0])
            indices = node.args[1]
            if (
                target is None
                or not _static_layout(target)
                or target.ndim not in (1, 2)
                or target.size(0) != graph.rows
                or not isinstance(indices, (list, tuple))
                or len(indices) != target.ndim
                or indices[0] is not graph.row
                or value.dtype != target.dtype
            ):
                return None
            if target.ndim == 1:
                domain = 0
            else:
                index = indices[1]
                index_tensor = tensor_value(index)
                domain = infer(index)
                if index_tensor is None or index_tensor.dtype not in (
                    torch.int32,
                    torch.int64,
                ):
                    return None
            mask = node.args[2]
            if mask is not None:
                masked = tensor_value(mask)
                mask_domain = infer(mask)
                if (
                    masked is None
                    or masked.dtype != torch.bool
                    or mask_domain not in (0, 1, domain)
                ):
                    return None
            return domain
        if node.target is torch.ops.aten.constant_pad_nd.default:
            if len(node.args) != 3 or node.kwargs:
                return None
            source, padding, other = node.args
            domain = infer(source)
            source_value = tensor_value(source)
            if (
                domain is None
                or domain == 0
                or not isinstance(padding, (tuple, list))
                or len(padding) != 2
                or type(padding[0]) is not int
                or padding[0] != 0
                or type(padding[1]) is not int
                or padding[1] < 0
                or type(other) not in (int, float, bool)
                or source_value is None
                or source_value.dtype != value.dtype
            ):
                return None
            return domain + padding[1]
        if node.target in (
            _tracing_ops._new_var,
            torch.ops.aten.alias.default,
            torch.ops.aten.clone.default,
            torch.ops.aten.detach.default,
            torch.ops.aten.view.default,
            torch.ops.aten.reshape.default,
            torch.ops.aten._unsafe_view.default,
        ):
            views = (
                torch.ops.aten.view.default,
                torch.ops.aten.reshape.default,
                torch.ops.aten._unsafe_view.default,
            )
            if len(node.args) != (2 if node.target in views else 1) or node.kwargs:
                return None
            domain = infer(node.args[0])
            original = tensor_value(node.args[0])
            if original is None or original.dtype != value.dtype:
                return None
            return domain
        if node.target is torch.ops.aten.view.dtype:
            if len(node.args) != 2 or node.kwargs:
                return None
            original = tensor_value(node.args[0])
            if (
                original is None
                or original.element_size() != value.element_size()
                or original.shape != value.shape
                or node.args[1] != value.dtype
            ):
                return None
            return infer(node.args[0])
        if not (
            isinstance(node.target, torch._ops.OpOverload)
            and torch.Tag.pointwise in node.target.tags
            and not node.target._schema.is_mutable
            and torch.Tag.nondeterministic_seeded not in node.target.tags
        ):
            return None
        inputs = [infer(source) for source in node.all_input_nodes]
        if not inputs or any(domain is None for domain in inputs):
            return None
        domains = cast("list[int]", inputs)
        domain = max(domains)
        return domain if all(size in (0, 1, domain) for size in domains) else None

    for node in graph.root_graph.nodes:
        if (
            node in graph.source_nodes
            or node is graph.selection
            or node.op == "output"
            or node.target is _tracing_ops._host_tensor
        ):
            continue
        if node.target is memory_ops.store:
            if len(node.args) != 4 or node.kwargs:
                return None
            target = _host_tensor(node.args[0])
            indices = node.args[1]
            column = (
                indices[1]
                if isinstance(indices, (tuple, list)) and len(indices) == 2
                else None
            )
            progression = (
                row_fragment_arange(column) if isinstance(column, Node) else None
            )
            if (
                target is None
                or target.ndim != 2
                or not _static_layout(target)
                or target.size(0) != graph.rows
                or target.size(1) > graph.k
                or target.stride(1) <= 0
                or target.stride(0) < (target.size(1) - 1) * target.stride(1) + 1
                or not isinstance(indices, (tuple, list))
                or len(indices) != 2
                or indices[0] is not graph.row
                or not isinstance(indices[1], Node)
                or progression is None
                or progression[:2] != (0, 1)
                or infer(indices[1]) is None
                or infer(node.args[2]) is None
            ):
                return None
            value_domain = infer(node.args[2])
            index_domain = infer(indices[1])
            if value_domain not in (0, 1, index_domain):
                return None
            mask = node.args[3]
            if mask is not None:
                mask_tensor = tensor_value(mask)
                if (
                    mask_tensor is None
                    or mask_tensor.dtype != torch.bool
                    or infer(mask) not in (0, 1, index_domain)
                ):
                    return None
        elif infer(node) is None:
            return None
    return memo


def match_radix_selection_root(
    graphs: Sequence[GraphInfo],
    *,
    allow_unbound: bool = False,
    check_storage: bool = True,
) -> RadixSelectionGraph | None:
    """Prove one complete scalar row and an operation-defined source recipe."""
    if len(graphs) != 1 or not isinstance(graphs[0], RootGraphInfo):
        return None
    root = graphs[0]
    nodes = tuple(root.graph.nodes)
    if not nodes or nodes[-1].op != "output" or nodes[-1].args != (None,):
        return None
    selections = [node for node in nodes if node.target is torch.ops.aten.topk.default]
    stores = tuple(node for node in nodes if node.target is memory_ops.store)
    if len(selections) != 1 or not stores:
        return None
    selection = selections[0]
    args = unordered_topk_args(selection)
    if args is None:
        return None
    source, k, dim, largest = args
    value = tensor_value(source)
    keys = _call(source, torch.ops.aten.where.self, 3)
    if (
        value is None
        or value.ndim != 1
        or value.dtype != torch.int32
        or dim not in (-1, 0)
        or keys is None
        or not _int32_literal(keys.args[2], -(1 << 31))
    ):
        return None
    load = _rank_load(keys.args[1])
    if load is None or load.args[3] is not None:
        return None
    x = _host_tensor(load.args[0])
    indices = load.args[1]
    if (
        x is None
        or x.ndim != 2
        or x.dtype not in FLOAT_DTYPES
        or not _static_layout(x)
        or x.stride(1) != 1
        or not isinstance(indices, (tuple, list))
        or len(indices) != 2
        or not isinstance(indices[0], Node)
        or not isinstance(indices[1], Node)
        or strip_alias(load.args[2]) is not strip_alias(keys.args[0])
    ):
        return None
    row, columns = cast("Node", indices[0]), cast("Node", indices[1])
    progression = row_fragment_arange(columns)
    if progression is None or progression[:2] != (0, 1):
        return None
    logical_cols = progression[2]
    if not 0 < k <= logical_cols <= INT32_MAX or value.shape != (logical_cols,):
        return None
    loaded = tensor_value(load)
    mask_value = tensor_value(keys.args[0])
    if (
        loaded is None
        or loaded.dtype != x.dtype
        or loaded.shape != (logical_cols,)
        or mask_value is None
        or mask_value.dtype != torch.bool
        or mask_value.shape != (logical_cols,)
    ):
        return None
    rows, cols = x.shape
    if cols > INT32_MAX:
        # The source's scalar comparison bound must fit its declared iota type.
        return None
    row_bid = _row_block(row, rows)
    host = HostFunction.current()
    ir = host.device_ir
    if (
        row_bid is None
        or len(ir.root_ids) != 1
        or ir.grid_block_ids != [[row_bid]]
        or row_bid in ir.noncanonical_task_origin_block_ids
    ):
        return None
    length = _mask_columns(keys.args[0], columns, cols)
    length_load = _call(length, memory_ops.load, 4)
    if length_load is None or length_load.args[2:] != (None, None):
        return None
    lengths = _host_tensor(length_load.args[0])
    length_value = tensor_value(length_load)
    if (
        lengths is None
        or lengths.ndim != 1
        or lengths.shape != (rows,)
        or not _static_layout(lengths)
        or lengths.dtype not in (torch.int32, torch.int64)
        or length_load.args[1] not in ([row], (row,))
        or length_value is None
        or length_value.ndim != 0
        or length_value.dtype != lengths.dtype
    ):
        return None
    values: set[Node] = set()
    indices_out: set[Node] = set()
    for user in selection.users:
        if (
            user.graph is not selection.graph
            or user.target is not operator.getitem
            or len(user.args) != 2
            or user.args[0] is not selection
            or type(user.args[1]) is not int
            or user.args[1] not in (0, 1)
            or user.kwargs
        ):
            return None
        getter = tensor_value(user)
        if (
            getter is None
            or getter.shape != (k,)
            or getter.dtype != (torch.int32 if user.args[1] == 0 else torch.int64)
        ):
            return None
        (values if user.args[1] == 0 else indices_out).add(user)
    if not values and not indices_out:
        return None
    source_nodes = _closure(source)
    loads = tuple(node for node in nodes if node.target is memory_ops.load)
    graph = RadixSelectionGraph(
        root.graph,
        root.graph_id,
        selection,
        source,
        row,
        row_bid,
        columns,
        cast("Node", load.args[0]),
        x,
        load,
        length_load,
        rows,
        cols,
        logical_cols,
        k,
        largest,
        source_nodes,
        {},
        loads,
        stores,
        frozenset(values),
        frozenset(indices_out),
    )
    domains = _suffix_domains(graph)
    if domains is None:
        return None
    graph = replace(graph, slot_domains=domains)
    if not check_storage:
        return graph
    env = CompileEnvironment.current()
    if any(
        not host_load_is_readonly(node, env, graphs, allow_unbound=allow_unbound)
        for node in loads
    ):
        return None
    targets = tuple(dict.fromkeys(cast("Node", node.args[0]) for node in stores))
    for position, left_node in enumerate(targets):
        left = cast("torch.Tensor", _host_tensor(left_node))
        for right_node in targets[:position]:
            right = cast("torch.Tensor", _host_tensor(right_node))
            if left.untyped_storage() == right.untyped_storage():
                return None
            if (
                left.untyped_storage() not in env._symbolically_exact_layout_storages
                and right.untyped_storage()
                not in env._symbolically_exact_layout_storages
                and not runtime_tensors_are_proven_disjoint(
                    env, left, right, allow_unbound=allow_unbound
                )
            ):
                return None
    return graph


def radix_schedule(
    graph: RadixSelectionGraph, config: dict[str, object]
) -> RadixSelectionPlan:
    return RadixSelectionPlan(
        graph,
        cast("int", config.get("cute_radix_threads", 256)),
        cast("int", config.get("cute_radix_cluster_ctas", 1)),
        cast("int", config.get("cute_radix_candidate_capacity", 0)),
        cast("str", config.get(HISTOGRAM_KEY, "direct")),
        cast("bool", config.get("cute_radix_candidate_key_cache", False)),
        cast("bool", config.get("cute_radix_scalar_candidates", False)),
        cast("bool", config.get("cute_radix_valid_domain", False)),
        cast("bool", config.get("cute_radix_ticket_stores", False)),
        cast("int", config.get("cute_radix_packet_bits", 128)),
        cast("int", config.get("cute_radix_scan_unroll", 1)),
        cast("bool", config.get("cute_radix_cached_payload_inverse", False)),
    )


def validate_radix_schedule(spec: ConfigSpec, config: dict[str, object]) -> None:
    graph = spec.cute_radix_selection_graph
    if graph is None:
        raise exc.InvalidConfig("radix requires a proved unordered selection root")
    plan = radix_schedule(graph, config)
    if plan.cached_payload_inverse:
        if not (plan.cache_keys_enabled and plan.ticket_stores and plan.valid_domain):
            raise exc.InvalidConfig(
                "cached payload inverse needs paired keys and a known-valid ticket plan"
            )
        from .radix_selection_codegen import has_exact_cached_payload_consumer

        if not has_exact_cached_payload_consumer(graph):
            raise exc.InvalidConfig(
                "cached payload inverse needs an exact original source read"
            )
    if plan.valid_domain and not graph.largest:
        raise exc.InvalidConfig("radix valid-domain trimming requires largest=True")
    if config.get("pid_type", "flat") != "flat":
        raise exc.InvalidConfig("radix requires a flat scalar row grid")
    if graph.rows * plan.cluster > INT32_MAX:
        raise exc.InvalidConfig("radix clustered grid exceeds Int32 cardinality")
    if plan.cluster > 1 and (
        spec.target_device_capability is None or spec.target_device_capability < (9, 0)
    ):
        raise exc.InvalidConfig("radix clusters require an SM90 or newer target")
    if config.get("cute_cluster_n", 1) != 1:
        raise exc.InvalidConfig("radix cannot share an ordinary SIMT cluster axis")
    if config.get("cute_reduction_schedule", "scalar") != "scalar" or any(
        config.get(key, False)
        for key in (
            "cute_collective_mma",
            "cute_register_chain",
            "cute_fragment_topk_network",
            "cute_fragment_register_loads",
            "cute_fragment_producer_cache",
            "cute_fragment_warp_results",
            "cute_fragment_register_snapshots",
        )
    ):
        raise exc.InvalidConfig("radix requires exclusive root ownership")
    if spec.device is None:
        raise exc.InvalidConfig("radix requires a known compile device")
    capacity = CuteTcgen05Config.per_cta_smem_capacity_bytes(spec.device)
    if not capacity or plan.shared_bytes > capacity:
        raise exc.InvalidConfig(
            f"radix needs {plan.shared_bytes} shared bytes; capacity is {capacity}"
        )


def plan_radix_selection_root(
    graphs: Sequence[GraphInfo], tile_strategy: TileStrategyDispatch
) -> RadixSelectionPlan | None:
    config = tile_strategy.strategies[0].fn.config
    if config.get(ALGORITHM_KEY, "legacy") != "radix":
        return None
    env = CompileEnvironment.current()
    validate_radix_schedule(env.config_spec, config.config)
    graph = match_radix_selection_root(graphs)
    if graph is None:
        raise exc.InvalidConfig(
            "radix selection failed its current graph/storage proof"
        )
    fn = tile_strategy.strategies[0].fn
    if fn.resolved_block_size(graph.row_block_id) != 1:
        raise exc.InvalidConfig("radix requires one scalar logical row per cluster")
    return radix_schedule(graph, config.config)
