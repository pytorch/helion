from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import sympy
import torch

from ...autotuner.config_fragment import EnumFragment
from ...autotuner.config_generation import ConfigGeneration
from ...exc import InvalidConfig
from ...language.memory_ops import load as language_load
from ...runtime.config import Config
from .common import is_canonical_row_reduction
from .cute import _seq_config_list
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ...autotuner.config_spec import ReductionLoopSpec
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR


class CuteRolledClusterPacketHeuristic(AutotunerHeuristic):
    """Seed one-trip register packets across a low-grid row cluster.

    Append at most sixteen cluster profiles and one full-row profile after the
    existing seed prefix. Each CTA
    owns a power-of-two slice covering its portion of a static row; the compiler
    masks any padding. Count every original load site and one additional
    consumer-prefetch reserve at one 32-bit slot per
    retained element. The 128-slot cap is a conservative seed estimate; actual
    cache, mask, alias, collective, and launch legality remain compiler checks.

    Neutral PDL and replicated/independent L2-retention profiles use existing
    public knobs. These are search candidates, never promoted defaults.
    """

    name = "cute_rolled_cluster_packet"
    backend = "cute"
    _CACHE_SLOTS = 128
    CACHE_SPECIALIZATION_FACTS = frozenset({"device_num_sm", "input_tensor_metadata"})

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        spec = env.config_spec
        if (
            env.backend_name != "cute"
            or not is_canonical_row_reduction(env)
            or device_ir.has_atomic_ops()
            or spec.cute_indexed_reduction_block_ids
            or spec.target_device_capability is None
            or spec.target_device_capability < (9, 0)
            or spec.max_reduction_threads is None
            or spec.max_reduction_threads < 128
            or not spec.reduction_loops[0].allow_full_extent
        ):
            return False
        reduction = spec.reduction_loops[0]
        row = spec.block_sizes[0].block_id
        # Fixed outer tiles can leave a jagged tile as the sole tunable block.
        # A jagged domain has no scalar row extent for these cluster seeds.
        if env.is_jagged_tile(row):
            return False
        extent = env.specialize_expr(env.block_sizes[reduction.block_id].numel)
        row_extent = env.specialize_expr(env.block_sizes[row].numel)
        if (
            not isinstance(extent, (int, sympy.Integer))
            or not isinstance(row_extent, (int, sympy.Integer))
            or int(extent) <= 32
            or int(row_extent) <= 0
        ):
            return False
        # Runtime registers this heuristic, so keep the metadata import lazy.
        from ...runtime import get_num_sm

        num_sm = get_num_sm(env.device)
        return num_sm > 0 and int(row_extent) <= num_sm

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        seeds = cls.get_seed_configs(env, device_ir)
        return seeds[0] if seeds else None

    @staticmethod
    def _load_metadata(
        device_ir: DeviceIR, block_id: int, size: int
    ) -> tuple[torch.dtype, int, tuple[int, ...]] | None:
        graph_ids = {
            info.original_graph_id
            for info in device_ir.rolled_reductions
            if info.rolled_block_ids == [block_id]
        }
        if not graph_ids:
            return None
        dtypes: set[torch.dtype] = set()
        alignments: list[int] = []
        load_sites = 0
        matrix_source = False
        for graph_id in sorted(graph_ids):
            for node in device_ir.graphs[graph_id].graph.nodes:
                if node.target is not language_load:
                    continue
                if not node.args or not isinstance(node.args[0], torch.fx.Node):
                    return None
                tensor = node.args[0].meta.get("val")
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tensor.ndim not in (1, 2)
                    or type(tensor.shape[-1]) is not int
                    or tensor.shape[-1] != size
                    or tensor.dtype
                    not in (torch.float16, torch.bfloat16, torch.float32)
                    or type(tensor.stride(-1)) is not int
                    or tensor.stride(-1) != 1
                    or type(tensor.storage_offset()) is not int
                    or any(type(stride) is not int for stride in tensor.stride()[:-1])
                    or (tensor.ndim == 2 and tensor.stride(0) < size)
                ):
                    return None
                indices = node.args[1]
                if (
                    not isinstance(indices, (list, tuple))
                    or len(indices) != tensor.ndim
                    or not isinstance(indices[-1], slice)
                    or indices[-1] != slice(None)
                ):
                    return None
                alignments.extend(tensor.stride()[:-1])
                alignments.append(cast("int", tensor.storage_offset()))
                dtypes.add(tensor.dtype)
                matrix_source |= tensor.ndim == 2
                load_sites += 1
        if not matrix_source or len(dtypes) != 1:
            return None
        return next(iter(dtypes)), load_sites, tuple(alignments)

    @classmethod
    def get_seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> list[Config]:
        seeds = cls._cluster_seed_configs(env, device_ir)
        for seed in cls.full_row_seed_configs(env, device_ir):
            if seed not in seeds:
                seeds.append(seed)
        return seeds

    @classmethod
    def _cluster_seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> list[Config]:
        if not cls.is_eligible(env, device_ir):
            return []
        spec = env.config_spec
        max_threads = spec.max_reduction_threads
        assert max_threads is not None
        reduction = cast("ReductionLoopSpec", spec.reduction_loops[0])
        size = int(env.specialize_expr(env.block_sizes[reduction.block_id].numel))
        metadata = cls._load_metadata(device_ir, reduction.block_id, size)
        if metadata is None:
            return []
        dtype, load_sites, alignments = metadata
        vector = 4 if dtype == torch.float32 else 8
        if any(offset % vector for offset in alignments):
            return []
        row = spec.block_sizes[0].block_id
        sequences = {
            "num_threads": spec.num_threads,
            "cute_vector_widths": spec.cute_vector_widths,
            "cute_lane_layouts": spec.cute_lane_layouts,
            "cute_reduction_reloads": spec.cute_reduction_reloads,
        }
        if any(
            reduction.block_id not in seq.valid_block_ids()
            for seq in sequences.values()
        ):
            return []
        if row not in spec.num_threads.valid_block_ids():
            return []
        row_domain = spec.block_sizes[0]._fragment(spec)
        if not row_domain.low <= 1 <= row_domain.high:
            return []
        fields = spec._flat_fields()
        cluster_domain = fields.get("cute_cluster_n")
        if not isinstance(cluster_domain, EnumFragment):
            return []
        policy_domain = spec.load_eviction_policies.inner.search_values() or ()
        policy_indices = {
            fact.eviction_index
            for fact in spec.memory_op_facts
            if fact.kind == "load"
            and fact.eviction_index is not None
            and fact.dtype == dtype
        }
        loop_values = {
            reduction._flat_config(spec, lambda fragment, value=value: value)
            for value in reduction._flat_fragment(spec).search_values() or ()
        }
        seeds = []
        for cluster in (2, 4, 8, 16):
            if cluster not in (cluster_domain.search_values() or ()):
                continue
            span = (size + cluster - 1) // cluster
            chunk = 1 << (span - 1).bit_length()
            if (
                chunk not in loop_values
                or cluster * chunk > 2**31 - 1
                or (
                    spec.max_reduction_loop is not None
                    and chunk > spec.max_reduction_loop
                )
            ):
                continue
            for threads in (128, 256):
                if (
                    threads > max_threads
                    or chunk % (threads * vector)
                    or (load_sites + 1) * (chunk // threads) > cls._CACHE_SLOTS
                ):
                    continue
                values: dict[str, Any] = {
                    "block_sizes": [1],
                    "reduction_loops": [chunk],
                    "num_threads": _seq_config_list(
                        spec.num_threads, {row: 1, reduction.block_id: threads}
                    ),
                    "cute_vector_widths": _seq_config_list(
                        spec.cute_vector_widths, {reduction.block_id: vector}
                    ),
                    "cute_lane_layouts": _seq_config_list(
                        spec.cute_lane_layouts, {reduction.block_id: "strided"}
                    ),
                    "cute_reduction_reloads": _seq_config_list(
                        spec.cute_reduction_reloads,
                        {reduction.block_id: "register_prefetch"},
                    ),
                    "cute_cluster_n": cluster,
                }
                if any(
                    value not in (item._fragment(spec).search_values() or ())
                    for name, sequence in sequences.items()
                    for item, value in zip(sequence, values[name], strict=True)
                ):
                    continue
                # Two bounded policy profiles per geometry: ordinary raw packets
                # with PDL, and independent replicated accumulation with retained
                # L2 loads. First-source retention is a general load-order policy.
                for independent, pdl, min_blocks in (
                    (False, True, 2),
                    (True, False, 1),
                ):
                    overrides = {
                        "cute_independent_reduction": independent,
                        "cute_replicated_reduction": independent,
                        "cute_pdl": pdl,
                        "cute_min_blocks_per_mp": min_blocks,
                    }
                    if any(
                        not isinstance(field := fields.get(key), EnumFragment)
                        or value not in field.choices
                        for key, value in overrides.items()
                    ):
                        continue
                    policies = spec.load_eviction_policies.default()
                    if independent:
                        if (
                            "l2_last" not in policy_domain
                            or not policy_indices
                            or any(
                                not 0 <= index < len(policies)
                                for index in policy_indices
                            )
                        ):
                            continue
                        for index in policy_indices:
                            policies[index] = "l2_last"
                        if "first" in policy_domain:
                            policies[min(policy_indices)] = "first"
                    requested = (
                        values | overrides | {"load_eviction_policies": policies}
                    )
                    try:
                        generation = ConfigGeneration(spec)
                        flat, effective = generation.strict_config_pair(
                            Config.from_dict(requested)
                        )
                        if generation.strict_unflatten(flat) != effective or any(
                            effective.config.get(key) != value
                            for key, value in requested.items()
                        ):
                            continue
                    except InvalidConfig:
                        continue
                    if effective not in seeds:
                        seeds.append(effective)
        return seeds

    @classmethod
    def full_row_seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> list[Config]:
        """Cover a single-CTA full row without displacing any cluster seed.

        A low row grid can favor more threads and smaller per-thread packets
        than its clustered alternatives. Keep the same canonical static load,
        layout, and retained-register estimates; compiler memory-effect proofs
        still decide whether caching and consumer prefetch are legal.
        """
        if not cls.is_eligible(env, device_ir):
            return []
        spec = env.config_spec
        if spec.max_reduction_threads is None or spec.max_reduction_threads < 512:
            return []
        reduction = cast("ReductionLoopSpec", spec.reduction_loops[0])
        size = int(env.specialize_expr(env.block_sizes[reduction.block_id].numel))
        metadata = cls._load_metadata(device_ir, reduction.block_id, size)
        if metadata is None:
            return []
        dtype, load_sites, alignments = metadata
        threads, vector = 512, 4
        chunk = 1 << (size - 1).bit_length()
        if (
            chunk > 2**31 - 1
            or chunk % (threads * vector)
            or (load_sites + 1) * (chunk // threads) > cls._CACHE_SLOTS
            or any(offset % vector for offset in alignments)
            or (spec.max_reduction_loop is not None and chunk > spec.max_reduction_loop)
        ):
            return []
        loop_values = {
            reduction._flat_config(spec, lambda fragment, value=value: value)
            for value in reduction._flat_fragment(spec).search_values() or ()
        }
        if chunk not in loop_values:
            return []
        row = spec.block_sizes[0].block_id
        sequences = {
            "num_threads": spec.num_threads,
            "cute_vector_widths": spec.cute_vector_widths,
            "cute_lane_layouts": spec.cute_lane_layouts,
            "cute_reduction_reloads": spec.cute_reduction_reloads,
        }
        if (
            any(
                reduction.block_id not in sequence.valid_block_ids()
                for sequence in sequences.values()
            )
            or row not in spec.num_threads.valid_block_ids()
        ):
            return []
        row_domain = spec.block_sizes[0]._fragment(spec)
        if not row_domain.low <= 1 <= row_domain.high:
            return []
        values: dict[str, Any] = {
            "block_sizes": [1],
            "reduction_loops": [chunk],
            "num_threads": _seq_config_list(
                spec.num_threads, {row: 1, reduction.block_id: threads}
            ),
            "cute_vector_widths": _seq_config_list(
                spec.cute_vector_widths, {reduction.block_id: vector}
            ),
            "cute_lane_layouts": _seq_config_list(
                spec.cute_lane_layouts, {reduction.block_id: "strided"}
            ),
            "cute_reduction_reloads": _seq_config_list(
                spec.cute_reduction_reloads,
                {reduction.block_id: "register_prefetch"},
            ),
        }
        if any(
            value not in (item._fragment(spec).search_values() or ())
            for name, sequence in sequences.items()
            for item, value in zip(sequence, values[name], strict=True)
        ):
            return []
        fields = spec._flat_fields()
        overrides = {
            "cute_cluster_n": 1,
            "cute_independent_reduction": True,
            "cute_replicated_reduction": True,
            "cute_pdl": True,
            "cute_min_blocks_per_mp": 0,
            "cute_vector_packet_unroll": False,
            "cute_vloop_sink": False,
            "cute_lane_unroll": 1,
        }
        if any(
            not isinstance(field := fields.get(key), EnumFragment)
            or value not in field.choices
            for key, value in overrides.items()
        ):
            return []
        policies = spec.load_eviction_policies.default()
        policy_indices = {
            fact.eviction_index
            for fact in spec.memory_op_facts
            if fact.kind == "load"
            and fact.eviction_index is not None
            and fact.dtype == dtype
        }
        if (
            "l2_last" not in (spec.load_eviction_policies.inner.search_values() or ())
            or not policy_indices
            or any(not 0 <= index < len(policies) for index in policy_indices)
        ):
            return []
        for index in policy_indices:
            policies[index] = "l2_last"
        requested = values | overrides | {"load_eviction_policies": policies}
        try:
            generation = ConfigGeneration(spec)
            flat, effective = generation.strict_config_pair(Config.from_dict(requested))
            if generation.strict_unflatten(flat) != effective or any(
                effective.config.get(key) != value for key, value in requested.items()
            ):
                return []
        except InvalidConfig:
            return []
        return [effective]
