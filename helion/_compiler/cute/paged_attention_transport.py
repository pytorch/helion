"""Physical layout proof for the paged BF16 attention pipeline.

The semantic matcher is independent of this initial SM100 transport. This
planner admits native byte TMA tiles and one packed scale word per key; other
valid attention programs retain their ordinary compiler route.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import sympy
import torch
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.functions import PythonMod

from . import target_facts
from .analysis.paged_attention_pair import paired_attention_item_axis
from .cute_mma_dag import MMA_DAG_ATTENTION_PASS_WARPS
from .cute_mma_dag import MMA_DAG_L2_HINTS
from .cute_mma_dag import MMA_DAG_PASS_WARPS_ATTENTION_PAIRED
from .cute_mma_dag import MMA_DAG_PASS_WARPS_ATTENTION_SCOPED
from .cute_mma_dag import MMA_DAG_PASS_WARPS_KEY
from .cute_mma_dag import MMA_DAG_PROBABILITY_SCHEDULE_CHOICES
from .cute_mma_dag import MMA_DAG_PROBABILITY_SCHEDULE_KEY
from .cute_mma_dag import MMA_DAG_PROBABILITY_SCHEDULE_SERIAL
from .cute_mma_dag import MMA_DAG_QUERY_TMA_KEY
from .cute_mma_dag import MMA_DAG_SLEEP_WAITS_KEY
from .cute_mma_dag import MMA_DAG_V_LAG_CHOICES
from .cute_mma_dag import MMA_DAG_V_LAG_KEY
from .cute_mma_dag import MMA_DAG_VALUE_METADATA_READ_OVERLAP_KEY
from .cute_mma_dag import MMA_DAG_VALUE_METADATA_SKIP_KEY
from .memory_ops import cute_tensor_base_is_aligned
from .native_s2f6_contract import NATIVE_S2F6_TARGETS

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment
    from .analysis.paged_attention_match import BlockScaledOperand
    from .analysis.paged_attention_match import PagedAttentionPlan
    from .analysis.separable_index_map import SeparableIndexMap
    from .cute_mma_dag import Access
    from .cute_mma_dag import MmaDagPlan


@dataclasses.dataclass(frozen=True)
class PagedTmaOperand:
    """Descriptors use these named view pointers, including their offsets.

    Raw descriptor coordinates are [dimension, key, head, subpage, page].
    Scale coordinates are [key, head, subpage, page] in packed INT32 words.
    Shapes and strides below are descriptor elements, not bytes.
    """

    raw: Access
    scale: Access
    raw_shape: tuple[int, ...]
    raw_strides: tuple[int, ...]
    scale_shape: tuple[int, ...]
    scale_strides: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class PagedAttentionDirectIO:
    """Original tensor strides proved from explicit folded storage indices."""

    query: tuple[int, ...]
    output: tuple[int, ...]
    lse: tuple[int, ...] | None
    partial_statistics: tuple[tuple[int, ...], tuple[int, ...]] | None = None


@dataclasses.dataclass(frozen=True)
class PagedQueryTma:
    """BF16 descriptor axes: dimension, row minor, row major, head, request."""

    shape: tuple[int, ...]
    strides: tuple[int, ...]
    box: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class PagedAttentionTransport:
    semantic: PagedAttentionPlan
    key: PagedTmaOperand
    value: PagedTmaOperand
    items: int
    query_rows: int
    heads: int
    subpages: int
    key_width: int
    item: sympy.Expr
    direct_io: PagedAttentionDirectIO | None = None


@dataclasses.dataclass(frozen=True)
class PagedAttentionScheduleFacts:
    """Serializable search facts for the initial explicit attention schedule.

    These are configuration obligations, not FX nodes or tensor provenance.
    A selected render repeats the transport proof against its bound program.
    """

    query_axis: int
    query_rows: int
    key_axis: int
    dimension_axis: int
    query_block: int
    query_columns: int = 64
    denominator_per_key: bool = False
    reference_scores: bool = False
    paired_items: bool = False
    partial_statistics: bool = False
    query_tma_available: bool = False
    value_metadata_skip_available: bool = False
    key_width: int = 128
    dimension: int = 128
    compute_warps: int = 4
    raw_stages: int = 2
    tmem_columns: int = 256
    smem_bytes: int = 106496
    # Original policy slots of raw K, K scale, raw V and V scale.  None means
    # the source load has no autotuned eviction-policy slot.
    tma_load_slots: tuple[int | None, ...] = ()

    def min_blocks_per_mp_choices(
        self, *, scoped_registers: bool = False, paired_items: bool = False
    ) -> tuple[int, ...]:
        if paired_items:
            return (1,)
        if scoped_registers:
            # At 384 threads and two CTAs, the launch budget is 80 registers
            # per thread. Only this budget makes 128/64/32 an inc/dec/dec split.
            # The one-CTA budget is 168, so an inc to 128 would be invalid.
            return (2,)
        return (1, 2) if self.denominator_per_key else (1,)


def paged_attention_launch_warps(
    *, per_key: bool, scoped_registers: bool = False, paired_items: bool = False
) -> int:
    """Physical role topology shared by planning and emitted-launch checks."""
    if paired_items:
        assert per_key and not scoped_registers
        return 20
    if scoped_registers:
        assert per_key
        return 12
    return 10 if per_key else 13


@dataclasses.dataclass(frozen=True)
class PagedAttentionSchedule:
    """The bound layout and physical choices of one attention render."""

    transport: PagedAttentionTransport
    query_columns: int = 64
    raw_stages: int = 2
    launch_warps: int = 13
    tmem_columns: int = 256
    fast_math: bool = False
    min_blocks_per_mp: int = 1
    scoped_registers: bool = False
    paired_items: bool = False
    v_lag: int = 0
    sleep_waits: bool = False
    probability_schedule: str = MMA_DAG_PROBABILITY_SCHEDULE_SERIAL
    query_tma: PagedQueryTma | None = None
    value_metadata_skip: bool = False
    value_metadata_read_overlap: bool = False
    tma_load_hints: tuple[str, ...] = ("", "", "", "")


def plan_paged_attention_schedule(
    env: CompileEnvironment,
    semantic: PagedAttentionPlan,
    selected: MmaDagPlan,
    config: Config,
) -> PagedAttentionSchedule | None:
    """Accept only choices implemented by the explicit attention schedule."""
    pass_warps = config.get(MMA_DAG_PASS_WARPS_KEY)
    if pass_warps not in MMA_DAG_ATTENTION_PASS_WARPS:
        return None
    scoped_registers = pass_warps == MMA_DAG_PASS_WARPS_ATTENTION_SCOPED
    paired_items = pass_warps == MMA_DAG_PASS_WARPS_ATTENTION_PAIRED
    if semantic.partial_statistics is not None and not paired_items:
        return None
    if scoped_registers and not semantic.denominator_per_key:
        return None
    transport = plan_paged_attention_transport(env, semantic)
    if transport is None or selected.program is not semantic.program:
        return None
    facts = _paged_attention_schedule_facts(env, semantic, transport)
    if facts is None:
        return None
    if paired_items and not facts.paired_items:
        return None
    allowed_blocks = facts.min_blocks_per_mp_choices(
        scoped_registers=scoped_registers, paired_items=paired_items
    )
    min_blocks = config.get("cute_min_blocks_per_mp", allowed_blocks[0])
    v_lag = config.get(MMA_DAG_V_LAG_KEY, 0)
    if type(v_lag) is not int or v_lag not in (
        MMA_DAG_V_LAG_CHOICES if paired_items else (0,)
    ):
        return None
    sleep_waits = config.get(MMA_DAG_SLEEP_WAITS_KEY, False)
    if type(sleep_waits) is not bool or (sleep_waits and not paired_items):
        return None
    probability_schedule = config.get(
        MMA_DAG_PROBABILITY_SCHEDULE_KEY, MMA_DAG_PROBABILITY_SCHEDULE_SERIAL
    )
    if type(probability_schedule) is not str or probability_schedule not in (
        MMA_DAG_PROBABILITY_SCHEDULE_CHOICES
        if paired_items and semantic.score_reference is not None
        else (MMA_DAG_PROBABILITY_SCHEDULE_SERIAL,)
    ):
        return None
    use_query_tma = config.get(MMA_DAG_QUERY_TMA_KEY, False)
    if type(use_query_tma) is not bool or (
        use_query_tma and (not paired_items or not facts.query_tma_available)
    ):
        return None
    value_metadata_skip = config.get(MMA_DAG_VALUE_METADATA_SKIP_KEY, False)
    if type(value_metadata_skip) is not bool or (
        value_metadata_skip
        and (not paired_items or not facts.value_metadata_skip_available)
    ):
        return None
    value_metadata_read_overlap = config.get(
        MMA_DAG_VALUE_METADATA_READ_OVERLAP_KEY, False
    )
    if type(value_metadata_read_overlap) is not bool or (
        value_metadata_read_overlap and not value_metadata_skip
    ):
        return None
    query_axis, dimension_axis = semantic.query.tile_dims
    key_axis = semantic.key.raw.tile_dims[0]
    if (
        selected.compute != "tcgen05"
        or selected.warps != facts.compute_warps
        or selected.stages != facts.raw_stages
        or selected.cluster != 1
        or selected.persistent != 0
        or selected.epilogue != "row"
        or selected.loads != "tma"
        or selected.aux_rings
        or selected.tma_stores
        or selected.mma_warp
        or selected.exchange_mode != "auto"
        or selected.derived_mode != "exchange"
        or type(min_blocks) is not int
        or min_blocks not in allowed_blocks
        or (config.get("cute_pdl") and config.get("cute_pdl_spread"))
        or selected.block(query_axis) != facts.query_block
        or selected.block(dimension_axis) != 128
        or selected.block(key_axis) != 128
    ):
        return None
    return PagedAttentionSchedule(
        transport,
        query_columns=facts.query_columns,
        launch_warps=paged_attention_launch_warps(
            per_key=semantic.denominator_per_key,
            scoped_registers=scoped_registers,
            paired_items=paired_items,
        ),
        tmem_columns=512 if paired_items else 256,
        fast_math=env.settings.fast_math,
        min_blocks_per_mp=min_blocks,
        scoped_registers=scoped_registers,
        paired_items=paired_items,
        v_lag=v_lag,
        sleep_waits=sleep_waits,
        probability_schedule=probability_schedule,
        query_tma=paged_query_tma(transport, facts.query_columns)
        if use_query_tma
        else None,
        value_metadata_skip=value_metadata_skip,
        value_metadata_read_overlap=value_metadata_read_overlap,
        tma_load_hints=paged_attention_load_hints(facts, config),
    )


def paged_attention_schedule_facts(
    env: CompileEnvironment, semantic: PagedAttentionPlan
) -> PagedAttentionScheduleFacts | None:
    """Config-free eligibility for exposing the explicit attention schedule."""
    transport = plan_paged_attention_transport(env, semantic)
    if transport is None:
        return None
    return _paged_attention_schedule_facts(env, semantic, transport)


def _paged_attention_schedule_facts(
    env: CompileEnvironment,
    semantic: PagedAttentionPlan,
    transport: PagedAttentionTransport,
) -> PagedAttentionScheduleFacts | None:
    """Reuse the transport proved during this planning call, never across renders."""
    if transport.key_width != 128 or transport.subpages != 1:
        return None
    per_key = semantic.denominator_per_key
    if per_key and transport.query_rows > 40:
        return None
    if not per_key and transport.query_rows & (transport.query_rows - 1):
        # The first selected-plan route uses the ordinary MMA root-tile
        # planner, whose full-row block sizes are powers of two.
        return None
    capacity = target_facts.per_cta_smem_capacity_bytes(env.device)
    # Include alignment and barriers in the conservative shared budget:
    # 104 KiB for the reduced form, 92 KiB for the widest (40-column)
    # per-key panel. These single-head CTAs own 256 of SM100's 512 TMEM
    # columns; only the single-head per-key schedule exposes a two-block
    # launch policy. Paired CTAs need all 512 columns and the separate
    # 180 KiB shared-memory capacity check below.
    smem_bytes = 94208 if per_key else 106496
    if capacity and capacity < smem_bytes:
        return None
    if env.pdl_entry or env.pdl_exit:
        # Explicit source placement needs its own native-body proof. Keep the
        # existing compiler route rather than dropping or moving the markers.
        return None
    # Length arithmetic finishes in its scalar tensor dtype before the row
    # coordinate promotes it. The native mask supports either index width,
    # with one common width for key, grouped row and query coordinates.
    coords = semantic.coordinates
    constants = (coords.query_len, coords.group)
    if semantic.mask.window_left is not None:
        constants += (semantic.mask.window_left,)
    if any(not -(2**31) <= value < 2**31 for value in constants):
        # CuTe promotes out-of-range Python integers to Int64; the source's
        # scalar-tensor operations instead retain their proved Int32 dtype.
        # Wider literal conversion needs an explicit typed recipe first.
        return None
    if any(
        dtype != torch.int32
        for dtype in (
            semantic.page_table.dtype,
            coords.lengths.dtype,
            coords.length_base_dtype,
        )
    ):
        return None
    if (
        coords.key_dtype not in (torch.int32, torch.int64)
        or coords.row_dtype != coords.key_dtype
        or coords.query_dtype != coords.key_dtype
    ):
        return None
    query_columns = (
        (8 if transport.query_rows <= 8 else 32 if transport.query_rows <= 32 else 40)
        if per_key
        else 64
    )
    # Two heads double the resident raw/Q/P/state panels. The 180 KiB
    # conservative aligned budget includes the optional query barrier and
    # 48 bytes of value metadata flags/barriers. The largest (40-query)
    # panel uses less than 180,000 bytes including both optional allocations.
    paired_items = (not capacity or capacity >= 184320) and (
        paired_attention_item_axis(env, transport) is not None
    )
    if semantic.partial_statistics is not None and not paired_items:
        return None
    return PagedAttentionScheduleFacts(
        query_axis=semantic.query.tile_dims[0].index,
        query_rows=transport.query_rows,
        key_axis=semantic.key.raw.tile_dims[0].index,
        dimension_axis=semantic.query.tile_dims[1].index,
        query_block=1 << (transport.query_rows - 1).bit_length(),
        query_columns=query_columns,
        denominator_per_key=per_key,
        reference_scores=semantic.score_reference is not None,
        paired_items=paired_items,
        partial_statistics=semantic.partial_statistics is not None,
        query_tma_available=paired_items
        and paged_query_tma(transport, query_columns) is not None,
        value_metadata_skip_available=paired_items
        and paged_value_metadata_skip_supported(transport),
        smem_bytes=smem_bytes,
        tma_load_slots=tuple(
            access.eviction_slot for access in paged_attention_tma_accesses(semantic)
        ),
    )


def paged_value_metadata_skip_supported(transport: PagedAttentionTransport) -> bool:
    """Prove the zero-scale shortcut and bounded synchronous metadata offsets.

    The existing operand proof supplies a 16-byte-aligned view and a complete
    128-word tile with unit key stride. Only the new synchronous load requires
    its reachable byte span to fit signed Int64; do not restrict ordinary TMA.
    With the written final BF16 +0, scale byte zero and postscale <= -14 map
    every INT8 value to positive zero, including the -128 rounding boundary.
    Requiring the source addition deliberately preserves the decoded V bits;
    eligibility does not depend on how a later accumulator treats zero signs.
    """
    value = transport.semantic.value
    return (
        value.positive_zero
        and -22 <= value.post_scale_log2 <= -14
        and transport.key_width == 128
        and transport.subpages == 1
        and 3
        + 4
        * sum(
            (extent - 1) * stride
            for extent, stride in zip(
                transport.value.scale_shape,
                transport.value.scale_strides,
                strict=True,
            )
        )
        <= 2**63 - 1
    )


def paged_query_tma(
    transport: PagedAttentionTransport, columns: int
) -> PagedQueryTma | None:
    """Prove four rectangular TMA boxes cover the paired SW128 Q panels.

    Transport already proves BF16 storage, unit dimension stride and a
    16-byte-aligned view base. The descriptor checks below repeat those
    stride requirements for the packed view and also validate the direct-I/O
    strides. Padded rows must lie outside the descriptor extent so hardware
    zero-fill agrees with the scalar loader's mask.
    """
    semantic = transport.semantic
    heads, items, rows = transport.heads, transport.items, transport.query_rows
    if columns not in (8, 32, 40) or not 0 < rows <= columns:
        return None
    if transport.direct_io is None:
        item_stride, row_stride, dim_stride = semantic.query.strides
        shape = (128, rows, 1, heads, items // heads)
        strides = (
            dim_stride,
            row_stride,
            rows * row_stride,
            item_stride,
            heads * item_stride,
        )
        box = (64, columns, 1, 1, 1)
    else:
        group, query_len = semantic.coordinates.group, semantic.coordinates.query_len
        if columns % group:
            return None
        batch_stride, query_stride, head_stride, dim_stride = transport.direct_io.query
        shape = (128, group, query_len, heads, items // heads)
        strides = (
            dim_stride,
            head_stride,
            query_stride,
            group * head_stride,
            batch_stride,
        )
        box = (64, group, columns // group, 1, 1)
    if strides[0] != 1 or any(
        stride <= 0 or stride * 2 % 16 or stride * 2 >= 2**40 for stride in strides[1:]
    ):
        return None
    if any(size <= 0 or size >= 2**32 for size in shape) or any(
        not 0 < size <= 256 for size in box
    ):
        return None
    return PagedQueryTma(shape, strides, box)


def paged_attention_tma_accesses(
    semantic: PagedAttentionPlan,
) -> tuple[Access, Access, Access, Access]:
    """Descriptor and cache-policy order: raw K, scale K, raw V, scale V."""
    return (
        semantic.key.raw,
        semantic.key.scale,
        semantic.value.raw,
        semantic.value.scale,
    )


def paged_attention_load_hints(
    facts: PagedAttentionScheduleFacts, config: Config
) -> tuple[str, ...]:
    """Selected TMA hints in raw-K, scale-K, raw-V, scale-V order."""
    policies = config.get("load_eviction_policies", [])
    assert isinstance(policies, list)
    return tuple(
        MMA_DAG_L2_HINTS.get(policies[slot], "")
        if slot is not None and slot < len(policies)
        else ""
        for slot in facts.tma_load_slots
    )


def _operand(
    env: CompileEnvironment, operand: BlockScaledOperand, width: int
) -> PagedTmaOperand | None:
    raw, scale = operand.raw, operand.scale
    if (
        operand.group_size != 32
        or raw.fake.ndim != 5
        or scale.fake.ndim != 5
        # The head axis is unconstrained here; the semantic item mapping
        # and descriptor proof below establish its ownership and stride.
        or raw.fake.shape[2] != width
        or raw.fake.shape[4] != 128
        or tuple(scale.fake.shape) != (*raw.fake.shape[:-1], 4)
        or raw.strides[-1] != 1
        or scale.strides[-1] != 1
        or scale.strides[2] != 4
    ):
        return None
    # The hardware checks all outer byte strides, including singleton modes.
    # Shape checks alone cannot prove pointer alignment: use bound provenance.
    if any(stride <= 0 or stride % 16 for stride in raw.strides[:-1]):
        return None
    if any(scale.strides[i] <= 0 or scale.strides[i] % 16 for i in (0, 1, 3)):
        return None
    if not all(
        cute_tensor_base_is_aligned(env, access.fake, 16) for access in (raw, scale)
    ):
        return None
    order = (4, 2, 3, 1, 0)
    scale_order = (2, 3, 1, 0)
    raw_shape = tuple(int(raw.fake.shape[i]) for i in order)
    raw_strides = tuple(raw.strides[i] for i in order)
    scale_shape = tuple(int(scale.fake.shape[i]) for i in scale_order)
    scale_strides = tuple(scale.strides[i] // 4 for i in scale_order)
    if any(size <= 0 or size >= 2**32 for size in (*raw_shape, *scale_shape)):
        return None
    # Tensor-map byte strides are 40-bit. Scale strides above are word counts.
    if any(stride >= 2**40 for stride in raw_strides[1:]) or any(
        stride * 4 >= 2**40 for stride in scale_strides[1:]
    ):
        return None
    return PagedTmaOperand(
        raw, scale, raw_shape, raw_strides, scale_shape, scale_strides
    )


def _direct_layout(
    index_map: SeparableIndexMap,
    item: sympy.Symbol,
    heads: int,
    query_len: int,
    group: int,
    columns: int | None,
) -> tuple[int, ...] | None:
    """Prove [request, query, grouped head, column] without affine-row fiction."""
    if len(index_map.axes) != (2 if columns is not None else 1):
        return None
    row = index_map.axes[0]
    if row.axis.extent != query_len * group or row.divisor != group:
        return None
    if columns is not None:
        column = index_map.axes[1]
        if (
            column.axis.extent != columns
            or column.divisor != 1
            or column.major_stride != 1
        ):
            return None
    # Group-one rows simplify away the minor term. Recovering their head
    # stride from the base is outside this folded-row contract.
    if row.minor_stride < (columns or 1):
        return None
    if row.major_stride < heads * group * row.minor_stride:
        return None
    request = FloorDiv(item, heads)
    head = PythonMod(item, heads)
    batch_stride = sympy.expand(index_map.base).coeff(request)
    if (
        not isinstance(batch_stride, sympy.Integer)
        or batch_stride < query_len * row.major_stride
        or sympy.expand(
            sympy.Add(
                index_map.base,
                sympy.Mul(-1, request, batch_stride),
                sympy.Mul(-1, head, group, row.minor_stride),
            )
        )
        != 0
    ):
        return None
    result = (int(batch_stride), row.major_stride, row.minor_stride)
    return (*result, 1) if columns is not None else result


def _direct_io(
    plan: PagedAttentionPlan, heads: int
) -> tuple[int, int, sympy.Symbol, PagedAttentionDirectIO] | None:
    from ..host_function import HostFunction
    from ..variable_origin import TileBeginOrigin
    from ..variable_origin import TileIdOrigin

    query_map = plan.query.index_map
    output_map = plan.output.index_map
    if (
        query_map is None
        or output_map is None
        or any(access.index_map is None for access in plan.row_outputs)
    ):
        return None
    # Only the per-key denominator body implements the io_layouts parameter.
    if not plan.denominator_per_key or len(plan.program.lead_axes) != 1:
        return None
    expected_accesses = {
        plan.query.node,
        plan.output.node,
        *(access.node for access in plan.row_outputs),
    }
    if {
        access.node for access in plan.program.accesses if access.index_map is not None
    } != expected_accesses:
        return None
    lead = plan.program.lead_axes[0]
    symbols = query_map.base.free_symbols
    if len(symbols) != 1:
        return None
    item = next(iter(symbols))
    if not isinstance(item, sympy.Symbol):
        return None
    info = HostFunction.current().expr_to_origin.get(item)
    if (
        info is None
        or not isinstance(info.origin, (TileBeginOrigin, TileIdOrigin))
        or info.origin.block_id != lead.block_id
        or lead.extent % heads
        or not 0 < lead.extent <= 2**31
    ):
        return None
    query_len, group = plan.coordinates.query_len, plan.coordinates.group
    query_layout = _direct_layout(query_map, item, heads, query_len, group, 128)
    output_layout = _direct_layout(output_map, item, heads, query_len, group, 128)
    if query_layout is None or output_layout is None:
        return None
    row_layouts = []
    for access in plan.row_outputs:
        assert access.index_map is not None
        layout = _direct_layout(access.index_map, item, heads, query_len, group, None)
        if layout is None:
            return None
        row_layouts.append(layout)
    lse_layout = row_layouts[0] if plan.lse is not None else None
    statistic_layouts = (
        (row_layouts[0], row_layouts[1])
        if plan.partial_statistics is not None
        else None
    )
    return (
        lead.extent,
        query_len * group,
        item,
        PagedAttentionDirectIO(
            query_layout, output_layout, lse_layout, statistic_layouts
        ),
    )


def _reflected_subscript(
    plan: PagedAttentionPlan, expression: object, subpages: int, *, remainder: bool
) -> bool:
    reflected = plan.coordinates.reflected
    assert reflected is not None
    if not isinstance(expression, sympy.Symbol):
        return False
    node = plan.program.index_values.get(expression)
    if node is None or len(node.args) != 2:
        return False
    target = (
        torch.ops.aten.remainder.Scalar if remainder else torch.ops.aten.div.Tensor_mode
    )
    return (
        node.target is target
        and node.args[0] is reflected.value
        and type(node.args[1]) is int
        and node.args[1] == subpages
        and node.kwargs == ({} if remainder else {"rounding_mode": "floor"})
        and node.meta["val"].dtype == reflected.dtype
    )


def plan_paged_attention_transport(
    env: CompileEnvironment, plan: PagedAttentionPlan
) -> PagedAttentionTransport | None:
    """Prove the initial rank-5 page/subpage transport without changing math."""
    # Existing panels implement the FP32 exp2/cast/postscale source contract.
    # Native scale0/subnormal and canonical-NaN semantics need their own emitter.
    if any(operand.decode_kind != "float_exp2" for operand in (plan.key, plan.value)):
        return None
    capability = env.config_spec.target_device_capability
    if capability not in NATIVE_S2F6_TARGETS or plan.io_dtype != torch.bfloat16:
        return None
    width = plan.coordinates.width
    if width not in (64, 128) or plan.page_dimension != 0:
        return None
    key = _operand(env, plan.key, width)
    value = _operand(env, plan.value, width)
    if key is None or value is None or key.raw_shape != value.raw_shape:
        return None
    subpages, heads = key.raw_shape[3], key.raw_shape[2]
    query = plan.query
    direct_io = None
    if query.index_map is not None:
        direct = _direct_io(plan, heads)
        if direct is None:
            return None
        items, rows, item, direct_io = direct
    else:
        # Raw statistics currently require the separately proved folded row
        # layouts. The packed normalized route remains unchanged.
        if plan.partial_statistics is not None:
            return None
        if query.fake.ndim != 3 or query.fake.shape[-1] != 128:
            return None
        items, rows, _ = (int(size) for size in query.fake.shape)
        if query.strides[-1] != 1 or any(
            stride <= 0 or stride % 8 for stride in query.strides[:-1]
        ):
            return None
        item = query.subscripts[0].expr
    if not 0 < rows <= 64 or items % heads:
        return None
    if not cute_tensor_base_is_aligned(env, query.fake, 16):
        return None
    iteration = plan.loop.symbol
    if not isinstance(item, sympy.Symbol) or iteration is None:
        return None
    expected_head = PythonMod(item, heads)
    expected_subpage = PythonMod(iteration, subpages)
    reflected = plan.coordinates.reflected is not None
    # The reflected runtime maps token directly to a page and uses subpage zero.
    if reflected and (subpages != 1 or not plan.denominator_per_key):
        return None
    for operand in (key, value):
        subscripts = operand.raw.subscripts
        if len(subscripts) != 5:
            return None
        subpage_matches = (
            _reflected_subscript(plan, subscripts[1].expr, subpages, remainder=True)
            if reflected
            else subscripts[1].expr == expected_subpage
        )
        if (
            subscripts[0].expr != plan.page_symbol
            or not subpage_matches
            or subscripts[2].kind != "axis"
            or subscripts[3].expr != expected_head
            or subscripts[4].kind != "axis"
            or any(s.offset is not None or s.window is not None for s in subscripts)
        ):
            return None
    page = plan.page_table
    request = FloorDiv(item, heads)
    if page.fake.ndim != 2:
        return None
    page_matches = (
        _reflected_subscript(plan, page.subscripts[1].expr, subpages, remainder=False)
        if reflected
        else page.subscripts[1].expr == FloorDiv(iteration, subpages)
    )
    if (
        page.subscripts[0].expr != request
        or not page_matches
        or plan.coordinates.lengths.subscripts[0].expr != request
    ):
        return None
    if direct_io is None and (
        tuple(plan.output.fake.shape) != tuple(query.fake.shape)
        or plan.output.subscripts[0].expr != item
        or any(
            tuple(access.fake.shape) != (items, rows)
            or access.subscripts[0].expr != item
            for access in plan.row_outputs
        )
    ):
        return None
    return PagedAttentionTransport(
        plan, key, value, items, rows, heads, subpages, width, item, direct_io
    )
