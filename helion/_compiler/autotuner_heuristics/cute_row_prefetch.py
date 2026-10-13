from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import torch

from ...autotuner.config_spec import _CUTE_LANE_UNROLL_CHOICES
from ...exc import InvalidConfig
from ...language.memory_ops import load
from ...language.memory_ops import store
from .cute import CutePersistentSubwarpRowsHeuristic

if TYPE_CHECKING:
    from typing import Any

    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR


def _read_write_packet_bytes(
    env: CompileEnvironment,
    device_ir: DeviceIR,
    row: int,
    extent: int,
    width: int,
) -> int | None:
    """Budget the input dtype of actual row/reduction read-write fragments."""
    row_symbol = env.block_sizes[row].var
    sizes = []
    for graph_info in device_ir.graphs:
        written = {
            node.args[0]
            for node in graph_info.graph.nodes
            if node.target is store and isinstance(node.args[0], torch.fx.Node)
        }
        for node in graph_info.graph.nodes:
            if node.target is not load or node.args[0] not in written:
                continue
            tensor = node.args[0].meta.get("val")
            value = node.meta.get("val")
            if not isinstance(value, torch.Tensor) or value.ndim < 2:
                continue
            if type(value.shape[-1]) is not int or value.shape[-1] != extent:
                continue
            if not any(
                isinstance(size, torch.SymInt) and size._sympy_() == row_symbol
                for size in value.shape[:-1]
            ):
                continue
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32)
                or tensor.stride(-1) != 1
            ):
                return None
            packet_bytes = width * tensor.element_size()
            # Existing persistent FP32 lowering admits two physical 16B
            # subpackets; other stores retain their original single-packet cap.
            if packet_bytes > (32 if tensor.dtype == torch.float32 else 16):
                return None
            sizes.append(packet_bytes)
    return max(sizes) if sizes else None


def append_row_prefetch_seeds(
    env: CompileEnvironment, device_ir: DeviceIR, seeds: list[Config]
) -> list[Config]:
    """Append packet-register schedules without changing the existing prefix.

    The original persistent-row matcher supplies the typed axes and one
    reduction fragment per lane. A two-warp CTA shares coefficients across
    rows; its prefetch group holds at most 128 input bytes per lane.
    This is a search resource budget, not a register-allocation guarantee.
    Half/full-warp one-row-per-group siblings expose independent CTAs.
    Original address, alignment, mask and alias proofs still govern lowering.
    """
    family = CutePersistentSubwarpRowsHeuristic
    plan = family._plan(env, device_ir)
    spec = env.config_spec
    if (
        plan is None
        or not spec.cute_proven_bounds_enabled
        or "input_tensor_metadata" not in env.compiler_fact_specialization_facts
    ):
        return seeds
    default = spec.default_config().config
    if default.get("pid_type", "flat") != "flat":
        return seeds
    row, reduction, packet_threads = plan
    extent = int(env.block_sizes[reduction].numel)
    packet_width = extent // packet_threads
    packet_bytes = _read_write_packet_bytes(env, device_ir, row, extent, packet_width)
    if packet_bytes is None:
        return seeds
    row_spec = cast("Any", spec.block_sizes[0])
    warp = 32
    row_threads = 2 * warp // packet_threads
    choices = [
        count
        for count in _CUTE_LANE_UNROLL_CHOICES
        if count * packet_bytes <= 128
        and row_spec.min_size <= row_threads * count <= row_spec.max_size
    ]
    schedules = []
    if choices:
        prefetch = max(choices)
        schedules.append((packet_threads, row_threads, prefetch))
    # A full warp can use an equal or smaller physical packet. Four row
    # groups in four warps keep the small-grid sibling independent of batch
    # sizes or model names; it remains a candidate for every eligible graph.
    schedules.extend(((packet_threads, row_threads, 1), (warp, 4, 1)))
    appended = list(seeds)
    for feature_threads, rows, unroll in schedules:
        row_size = rows * unroll
        if not row_spec.min_size <= row_size <= row_spec.max_size:
            continue
        if extent % feature_threads or extent // feature_threads > packet_width:
            continue
        proposed = (
            default
            | family._seed_config(
                env,
                row,
                reduction,
                feature_threads,
                row_block_size=row_size,
                row_threads=rows,
            ).config
            | {"cute_proven_bounds": True, "cute_lane_unroll": unroll}
        )
        if any(
            proposed.get(name) != value
            for name, value in env.settings.autotune_config_overrides.items()
        ):
            continue
        try:
            seed = spec.normalized_config(proposed)
        except InvalidConfig:
            continue
        if seed not in appended:
            appended.append(seed)
    if len(appended) == len(seeds):
        return seeds
    name = "cute_resident_row_prefetch_v2"
    if name not in spec.autotuner_heuristics:
        spec.autotuner_heuristics.append(name)
    return appended
