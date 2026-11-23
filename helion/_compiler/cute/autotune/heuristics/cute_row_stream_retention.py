"""Row-reduction seeds that keep one input stream in L2.

A row reduction over two or more per-row inputs (a residual add feeding a
LayerNorm) reads each of them once per launch.  With an L2 evict-last policy
on ONE of those streams, the launches that follow find it in L2; with the
policy on both they displace each other.  On B200, an 8192 x 1024 bf16
residual-add LayerNorm against Inductor's Triton kernel in the same
processes, under a timer that replays its inputs after an L2 flush: the
two-warp 32-byte-packet row seed read 0.92x as it is and 1.09x with the
policy on its second stream, the four-warp 16-byte one 0.93x and 1.05x; a
search over the same kernel read 8.99 us with one stream retained, 10.11
with both and 10.82 with neither.  The gain belongs to that timer's replay
(hinted lines survive its flush): with private inputs per call the same
twins read level with their carriers.  No row seed carried a policy, so a
search met the retained family only when its random draw held one on a
usable geometry.

``row_stream_retention_seed_configs`` plants the twins of the first row
seeds that load packets with their own threads: each stream in turn takes
``l1_l2_last`` while the others take the read-only evict-first packet load
where the target has it.  Only a kernel none of whose seeds names a load
policy gets them: where another family already chose policies for the
kernel's loads, its seeds are the measured ones.  Seeds only, appended
after every other seed (``postprocess_seed_configs``), so the existing seed
prefixes stay as they were; the policies are cache hints and change no
value.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import torch

from .....exc import InvalidConfig
from .....runtime.config import Config
from ....autotuner_heuristics.registry import AutotunerHeuristic
from ...cutedsl_compat import readonly_packet_load_supported
from .cute import CuteReductionTileHeuristic
from .cute import _device_l2_bytes

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .....autotuner.config_spec import ConfigSpec
    from .....autotuner.config_spec import MemoryOpFact
    from ....compile_environment import CompileEnvironment
    from ....device_ir import DeviceIR

# The policy of the retained stream: Triton's ``evict_last`` (L1 and L2).
RETAINED_POLICY = "l1_l2_last"
# The policy of the other streams where the target loads read-only packets.
STREAMED_POLICY = "nc_l2_first"
# Carrier seeds twinned, and streams retained in turn on each.
MAX_CARRIERS = 2
MAX_RETAINED_STREAMS = 2
# A stream below this many bytes costs a launch no bandwidth worth a seed.
MIN_STREAM_BYTES = 1 << 20


class CuteRowStreamRetentionHeuristic(AutotunerHeuristic):
    """``row_stream_retention_seed_configs``, appended after every other
    seed (``postprocess_seed_configs``); a seed family of that
    postprocessor, not an entry of the ranked registry."""

    name = "cute_row_stream_retention"
    backend = "cute"


def row_streams(
    spec: ConfigSpec, facts: Sequence[MemoryOpFact]
) -> dict[str, list[int]]:
    """The input streams among ``facts`` (the loads of one config, in its
    slot order): tensor name -> its ``load_eviction_policies`` slots, in
    lowering order.

    A stream's every load feeds a reduction and addresses every grid axis
    with no axis broadcast (a zero stride), so each CTA reads its own rows
    of the tensor and a launch reads each element once, and it walks its
    row element after element (``_walks_row``), so a row seed loads it in
    packets: the retained family was measured on
    packet streams, and a component of a wider tensor (a stride-2 lane)
    loads element by element on every row seed.  (A rolled reduction loads
    its inputs again in each pass; all of a tensor's slots belong to its
    stream.)  A tensor some load reads any other way (a broadcast weight, a
    gathered table, a strided component) is no stream.
    """
    grid = {item.block_id for item in spec.block_sizes}
    streams: dict[str, list[int]] = {}
    others: set[str] = set()
    for fact in facts:
        if fact.kind != "load" or fact.tensor_name is None:
            continue
        if (
            fact.eviction_index is not None
            and fact.reductions_fed
            and grid <= set(fact.subscript_block_ids)
            and 0 not in fact.subscript_strides
            and _walks_row(grid, fact)
        ):
            streams.setdefault(fact.tensor_name, []).append(fact.eviction_index)
        else:
            others.add(fact.tensor_name)
    return {name: slots for name, slots in streams.items() if name not in others}


def _walks_row(grid: set[int], fact: MemoryOpFact) -> bool:
    """Whether the load ``fact`` reads consecutive elements along its row:
    at the last subscript position that is no grid axis (the lane a row
    seed vectorizes) the tensor's layout stride is 1 and so is the scale of
    the index (``x[tile, :]``, not ``x[tile, :, 0]`` of a ``[rows, n, 2]``
    tensor or ``x[tile, 2 * i]``)."""
    lanes = [
        position
        for position, block_id in enumerate(fact.subscript_block_ids)
        if block_id not in grid
    ]
    if not lanes:
        return False
    lane = lanes[-1]
    return (
        lane < len(fact.subscript_strides)
        and fact.subscript_strides[lane] == 1
        and lane < len(fact.subscript_index_scales)
        and fact.subscript_index_scales[lane] == 1
    )


def _stream_bytes(facts: Sequence[MemoryOpFact]) -> dict[str, int]:
    return {
        fact.tensor_name: fact.accessed_numel * fact.dtype.itemsize
        for fact in facts
        if fact.kind == "load" and fact.tensor_name is not None and fact.dtype
    }


def _names_a_policy(seed: Config) -> bool:
    policies = seed.config.get("load_eviction_policies")
    if isinstance(policies, str):
        return bool(policies)
    return isinstance(policies, list) and any(policies)


def _is_carrier(seed: Config) -> bool:
    """A row seed that loads packets with threads of its own: vector lanes
    and a thread count."""
    widths = seed.config.get("cute_vector_widths")
    threads = seed.config.get("num_threads")
    return (
        isinstance(widths, list)
        and any(isinstance(width, int) and width >= 2 for width in widths)
        and isinstance(threads, list)
        and any(isinstance(count, int) and count > 0 for count in threads)
    )


def _keeps_row_in_registers(carrier: Config) -> bool:
    """Whether the normalized row seed ``carrier`` holds its row in registers
    (a persistent reduction: no rolled trip over the row)."""
    loops = carrier.config.get("reduction_loops")
    return isinstance(loops, list) and None in loops


def row_stream_retention_seed_configs(
    env: CompileEnvironment, device_ir: DeviceIR, configs: Sequence[Config]
) -> list[Config]:
    """Twins of the first ``MAX_CARRIERS`` row seeds of ``configs`` that load
    packets (``_is_carrier``), each with one input stream kept in L2
    (``RETAINED_POLICY`` on every load slot of it) and the others streamed
    (``STREAMED_POLICY``, or left alone where the target has no read-only
    packet load): one twin per stream for the first ``MAX_RETAINED_STREAMS``
    streams, so either argument order of a two-input kernel gets the twin
    that retains each input.

    No seeds unless the kernel is the row reduction
    ``CuteReductionTileHeuristic`` seeds, its config has load policies, no
    seed of ``configs`` names one, and at least two streams (``row_streams``)
    remain; a stream is retained only when it is at least
    ``MIN_STREAM_BYTES`` and fits in L2 (no policy keeps a larger one
    resident).  A seed that does not normalize (the search repairs or skips
    it) is no carrier, and neither is a row held in registers in a kernel
    whose memory accesses may need an order across threads (a workspace row
    written before the reduction and read after it), which the code
    generator refuses (``ConfigSpec.cute_register_lane_rows``): the twins
    then go to the rolled rows behind it.  The twins name ``cute_wide_packet_hints``, so a
    retained 32-byte packet renders its hint wherever the twin is rendered;
    the search sets the knob on every config it decodes.  Whether a carrier
    renders is not known here beyond that: the twin of a seed the code
    generator refuses for another reason is refused with it.
    """
    spec = env.config_spec
    if (
        not spec.supports_config_key("load_eviction_policies")
        or spec.load_eviction_policies.length <= 0
        or RETAINED_POLICY not in spec.search_surface.eviction_policies()
        or not CuteReductionTileHeuristic.is_eligible(env, device_ir)
        or any(_names_a_policy(seed) for seed in configs)
    ):
        return []
    length = spec.load_eviction_policies.length
    l2_bytes = _device_l2_bytes(env)
    streamed = (
        STREAMED_POLICY
        if readonly_packet_load_supported(
            spec.target_device_capability, torch.version.cuda
        )
        else ""
    )
    seeds: list[Config] = []
    carriers = 0
    for carrier in configs:
        if carriers == MAX_CARRIERS:
            break
        if not _is_carrier(carrier):
            continue
        try:
            normalized = spec.normalized_config(carrier)
        except InvalidConfig:
            continue
        facts = device_ir.config_memory_op_facts(normalized)
        streams = row_streams(spec, facts)
        if len(streams) < 2 or any(
            not 0 <= slot < length for slots in streams.values() for slot in slots
        ):
            continue
        sizes = _stream_bytes(facts)
        retained = [
            name
            for name in list(streams)[:MAX_RETAINED_STREAMS]
            if MIN_STREAM_BYTES <= sizes.get(name, 0) <= l2_bytes
        ]
        if not retained:
            continue
        if _keeps_row_in_registers(normalized) and not spec.cute_register_lane_rows:
            continue
        carriers += 1
        for name in retained:
            policies = [""] * length
            for stream, slots in streams.items():
                for slot in slots:
                    policies[slot] = RETAINED_POLICY if stream == name else streamed
            seeds.append(
                Config.from_dict(
                    deepcopy(carrier.config)
                    | {
                        "load_eviction_policies": policies,
                        "cute_wide_packet_hints": True,
                    }
                )
            )
    return seeds
