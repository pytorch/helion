"""A four-warp carrier for a complete gathered producer and ordinary merge."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import torch

from .....runtime.config import Config
from ....autotuner_heuristics.registry import AutotunerHeuristic
from ....inductor_lowering import ReductionLowering
from ... import gathered_panel_layout as panels
from ... import target_facts
from ...analysis.match_common import unit_fixed_block
from ...cooperative_grid_init import grid_init_carrier
from ...gathered_launch import GATHERED_SEED_THREADS
from ..config_keys import GRID_INIT_KEY

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ....compile_environment import CompileEnvironment
    from ....device_ir import DeviceIR
    from ....device_ir import GraphInfo


def gathered_phase_carrier(
    env: CompileEnvironment, ir: DeviceIR, region: Sequence[GraphInfo]
) -> Config | None:
    if len(ir.root_ids) != 2 or ir.host_function is None:
        return None
    with ir.host_function:
        return _gathered_phase_carrier(env, ir, region)


def _gathered_phase_carrier(
    env: CompileEnvironment, ir: DeviceIR, region: Sequence[GraphInfo]
) -> Config | None:
    """Put the page reduction on x and coalesced value columns on y.

    A signed page table drives a serial loop in the producer; the sibling owns
    a 2-D merge over identical output columns. Only that shape has a measured
    witness. Other complete fragment roots keep their previous carriers.
    """
    # device_ir imports the public Config, populated after registry initialization.
    from ....device_ir import ForLoopGraphInfo
    from ...analysis.gathered_bf16_mma import prove_gathered_bf16_mma

    if len(ir.grid_block_ids) != 2:
        return None
    folds = [
        (call, fold)
        for graph in region
        for call in graph.graph.nodes
        if (fold := prove_gathered_bf16_mma(call, ir.graphs, env)) is not None
    ]
    if len(folds) != 1:
        return None
    call, fold = folds[0]
    producer, merge = ir.grid_block_ids
    if len(producer) != 2 or len(merge) != 2:
        return None
    p, v = producer
    m, d = merge
    size = env.block_sizes
    value_width, merge_width, merge_rows = size[v].size, size[d].size, size[m].size
    if not (
        unit_fixed_block(env, p)
        and type(value_width) is int
        and isinstance(merge_width, (int, torch.SymInt))
        and isinstance(merge_rows, (int, torch.SymInt))
        and env.known_equal(value_width, merge_width)
        and env.known_equal(merge_rows, fold.rows)
    ):
        return None
    # Bind the page lanes to an actual column reduction fed by this fold.
    # Another region may have a reduction of the same static width.
    page_axes: set[int] = set()
    pending = list(call.users)
    seen = set(pending)
    while pending:
        node = pending.pop()
        for user in node.users:
            if user not in seen:
                seen.add(user)
                pending.append(user)
        if node.target not in (
            torch.ops.aten.amax.default,
            torch.ops.aten.sum.dim_IntList,
        ):
            continue
        if len(node.args) < 2 or node.args[1] not in ([1], [-1]):
            continue
        source = (
            node.all_input_nodes[0].meta.get("val") if node.all_input_nodes else None
        )
        lowering = node.meta.get("lowering")
        if (
            isinstance(source, torch.Tensor)
            and source.ndim == 2
            and env.known_equal(source.size(0), fold.rows)
            and env.known_equal(source.size(1), fold.columns)
            and isinstance(lowering, ReductionLowering)
            and size[lowering.block_index].reduction
        ):
            page_axes.add(lowering.block_index)
    if len(page_axes) != 1:
        return None
    (page_axis,) = page_axes
    # Powers of two are the lane choices. X services one complete page when
    # possible, Y fills the remaining warps of the common 128-thread CTA.
    page_threads = 1 << (min(GATHERED_SEED_THREADS, fold.columns).bit_length() - 1)
    value_threads = GATHERED_SEED_THREADS // page_threads
    if value_width < value_threads:
        return None
    body_index = call.args[0]
    if not isinstance(body_index, int):
        return None

    body = ir.graphs[body_index]
    if not isinstance(body, ForLoopGraphInfo):
        return None
    (reduction,) = body.block_ids

    def down(value: int, cap: int) -> int:
        return 1 << (min(value, cap).bit_length() - 1)

    widths = {
        reduction: down(fold.depth, 32),
        v: down(value_width, 64),
        m: down(fold.rows, 8),
        d: down(value_width, 32),
    }
    lanes = {page_axis: page_threads, v: value_threads, d: value_threads}
    spec = env.config_spec
    if not set(widths) <= set(spec.block_sizes.valid_block_ids()) or not set(
        lanes
    ) <= set(spec.num_threads.valid_block_ids()):
        return None
    return Config(
        block_sizes=[widths.get(item.block_id, 1) for item in spec.block_sizes],
        num_threads=[lanes.get(item.block_id, 1) for item in spec.num_threads],
        pid_type="persistent_blocked",
        cross_loop_pipeline="barrier",
    )


def streaming_phase_carriers(
    env: CompileEnvironment, ir: DeviceIR, base: Config, region: Sequence[GraphInfo]
) -> list[Config]:
    """When an entire page transition is warp-native, give the merge every lane.

    The producer cooperates across the whole CTA, so its logical page and
    output axes need no physical thread coordinates. Assigning all CTA lanes
    to consecutive merge columns avoids executing each split loop once per
    unused producer x-lane. At two page widths, split-first order balances
    independent CTAs and QK reuse; a one-page-width value-first panel keeps
    consecutive PIDs on live splits. Retain the wide panel for long contexts.
    """
    # The registry loads before the public Config export; FX analysis imports
    # fragment memory code that depends on that export.
    from ...analysis.gathered_softmax import prove_gathered_softmax
    from ...gathered_softmax import can_pipeline_page

    if ir.host_function is None or len(ir.grid_block_ids) != 2:
        return []
    with ir.host_function:
        pages = [
            p
            for g in region
            if (p := prove_gathered_softmax(g.graph, ir.graphs, env)) is not None
        ]
    if len(pages) != 1:
        return []
    page = pages[0]
    _, value_axis = ir.grid_block_ids[0]
    row_axis, column_axis = ir.grid_block_ids[1]
    extent = env.block_sizes[value_axis].size
    if (
        page.value_bid != value_axis
        or type(extent) is not int
        or extent < GATHERED_SEED_THREADS
        or extent % GATHERED_SEED_THREADS
    ):
        return []
    spec = env.config_spec
    lanes = [
        GATHERED_SEED_THREADS if item.block_id == column_axis else 1
        for item in spec.num_threads
    ]
    choices = []
    for width in dict.fromkeys(
        min(extent, size)
        for size in (2 * page.qk.columns, page.qk.columns, 4 * page.qk.columns)
    ):
        if width % 32 or extent % width:
            continue
        sizes = {value_axis: width, row_axis: 1, column_axis: GATHERED_SEED_THREADS}
        # The first grid axis is split, the second is value. Only this
        # narrow panel benefits from assigning neighboring physical workers
        # consecutive value panels of the same live split. Keep every other
        # root's ordinary order.
        order: dict[str, object] = {}
        if width == page.qk.columns:
            orders = []
            for item in spec.loop_orders:
                axes = list(range(len(item.block_ids)))
                if item.block_ids == ir.grid_block_ids[0]:
                    axes.reverse()
                orders.append(axes)
            if orders:
                order["loop_orders"] = orders
        choices.append(
            Config.from_dict(
                base.config
                | order
                | {
                    "block_sizes": [
                        sizes.get(item.block_id, existing)
                        for item, existing in zip(
                            spec.block_sizes, base.block_sizes, strict=True
                        )
                    ],
                    "num_threads": lanes,
                    # Both independent barrier phases can occupy all workers,
                    # including a short merge after a much larger producer.
                    "pid_type": (
                        "persistent_interleaved"
                        if "persistent_interleaved" in spec.allowed_pid_types
                        else base.pid_type
                    ),
                }
            )
        )
    # Preserve existing reproducible seeds and add opt-in pipeline
    # carriers for compatible geometry, row strides and current pointers.
    # Lowering independently requires all three cache-backed alignment proofs.
    for config in choices.copy():
        width = next(
            size
            for item, size in zip(spec.block_sizes, config.block_sizes, strict=True)
            if item.block_id == value_axis
        )
        if can_pipeline_page(
            env, page, width, GATHERED_SEED_THREADS, require_bound_pointer=False
        ):
            pipeline = config.config | {"cute_gathered_page_pipeline": True}
            choices.extend(
                (
                    Config.from_dict(pipeline),
                    Config.from_dict(pipeline | {"cute_gathered_page_stages": 4}),
                )
            )
            # A long contraction can start eight distinct cp.async groups
            # immediately. Do not plant this high-storage carrier on a short
            # contraction or a device where even its page buffers do not fit.
            # Codegen subsequently budgets the complete phase and launcher.
            maximum = panels.RING_STAGES[-1]
            capacity = target_facts.per_cta_smem_capacity_bytes(env.device)
            if (
                page.qk.depth >= maximum * panels.DEPTH_PANEL
                and capacity
                and panels.working_set_bytes(
                    page.qk.rows, page.qk.columns, width, maximum
                )
                <= capacity
            ):
                deepest = pipeline | {"cute_gathered_page_stages": maximum}
                choices.append(Config.from_dict(deepest))
                # A short merge should not pay a second device launch just
                # to zero its grid counter. Keep the host-zeroed schedule as
                # a comparison and offer only one additional long-K carrier.
                if grid_init_carrier(spec, deepest):
                    choices.append(Config.from_dict(deepest | {GRID_INIT_KEY: True}))
    return choices


class CuteGatheredBf16MmaSeed(AutotunerHeuristic):
    name = "cute_gathered_bf16_mma"
    backend = "cute"
    priority = 104

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return bool(env.config_spec.cute_gathered_phase_carriers)

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        carriers = cls.get_seed_configs(env, device_ir)
        return carriers[0] if carriers else None

    @classmethod
    def get_seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> list[Config]:
        return [
            Config.from_dict(deepcopy(c) | {"cute_fragment_half_mma": True})
            for c in env.config_spec.cute_gathered_phase_carriers
        ]
