from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import sympy
import torch

from ...autotuner.config_fragment import EnumFragment
from ...autotuner.config_generation import ConfigGeneration
from ...exc import InvalidConfig
from ...language import _tracing_ops
from ...language.memory_ops import load as language_load
from ...language.memory_ops import store as language_store
from ...language.view_ops import subscript
from ...runtime.config import Config
from .common import is_canonical_row_reduction
from .cute import _seq_config_list
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ...autotuner.config_spec import ReductionLoopSpec
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR


class CuteFullChunkRowHeuristic(AutotunerHeuristic):
    """Bounded full-chunk packet schedules; never promote a default.

    The 512/1024-thread choices and 128-slot retained-cache estimate are seed
    policy, not occupancy guarantees. Count every original load site, including
    loads not cached, and reserve another full fragment for consumer prefetch.
    Low-thread profiles omit that reserve only when a closed two-pass graph
    proves that the consumer load already has its own original-load charge.
    Charge each value one 32-bit register slot even for FP16/BF16 inputs.
    Requested packet widths are not alignment proofs: existing vector lowering
    retains its scalar fallback for offset or otherwise unaligned inputs.
    Combined policy/replication profiles require PDL. A default-geometry
    replication carrier keeps search structure independent of profitable
    geometry, dtype, and layout; the lowering can safely leave it unchanged.
    Append at most one independent-accumulation carrier with the same reference
    geometry. After that unchanged prefix, append at most eight 128/256-thread
    independent packet profiles under the same retained-cache budget. These
    thread counts and the preference for PDL are seed policy, not occupancy or
    storage-alignment guarantees (at most seventeen seeds total).
    """

    name = "cute_full_chunk_row"
    backend = "cute"

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        spec = env.config_spec
        return (
            env.backend_name == "cute"
            and is_canonical_row_reduction(env)
            and not spec.cute_indexed_reduction_block_ids
            and spec.reduction_loops[0].allow_full_extent
        )

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        seeds = cls.get_seed_configs(env, device_ir)
        return seeds[0] if seeds else None

    @classmethod
    def get_seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> list[Config]:
        if not cls.is_eligible(env, device_ir):
            return []
        seeds = cls._packet_seeds(env, device_ir)
        for key in ("cute_replicated_reduction", "cute_independent_reduction"):
            field = env.config_spec._flat_fields().get(key)
            if (
                not isinstance(field, EnumFragment)
                or True not in field.choices
                or any(seed.get(key, False) for seed in seeds)
            ):
                continue
            # Enum search subsets participate in structural_fingerprint. A
            # geometry-dependent mode witness would otherwise reject
            # autotune_multi for the same row kernel at neighboring extents.
            # No packet/cache/launch overrides belong on this carrier.
            carrier = Config.from_dict(
                env.config_spec.autotune_reference_config().config | {key: True}
            )
            try:
                generation = ConfigGeneration(env.config_spec)
                flat, effective = generation.strict_config_pair(carrier)
                if generation.strict_unflatten(flat) != effective:
                    continue
            except InvalidConfig:
                continue
            seeds.append(carrier)
        seeds.extend(
            seed for seed in cls._low_thread_seeds(env, device_ir) if seed not in seeds
        )
        return seeds

    @classmethod
    def _low_thread_seeds(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> list[Config]:
        seeds = cls._packet_seeds(env, device_ir, independent=True)
        if cls._consumer_prefetch_is_counted(device_ir):
            # Preserve the complete old prefix. A consumer load already in
            # the original load count must not also cost a reserve fragment.
            seeds.extend(
                seed
                for seed in cls._packet_seeds(
                    env, device_ir, independent=True, prefetch_already_counted=True
                )
                if seed not in seeds
            )
        return seeds

    @staticmethod
    def _consumer_prefetch_is_counted(device_ir: DeviceIR) -> bool:
        """Prove the reserve duplicates a charged load in a closed two-pass row.

        Keep the old estimate for additional graphs, statistics, stores, or
        opaque operations. This concerns retained raw load fragments only;
        scalar temporaries and independent accumulators are not full-row
        caches. The emitted prefetch pass still checks every actual private
        allocation against its unchanged 128-slot cap.
        """
        # Lowering imports the runtime that registers these heuristics.
        from ..device_ir import ReductionLoopGraphInfo
        from ..inductor_lowering import PointwiseLowering
        from ..inductor_lowering import ReductionLowering

        loops = [
            info
            for info in device_ir.graphs
            if isinstance(info, ReductionLoopGraphInfo)
        ]
        if (
            len(loops) != 2
            or len(device_ir.graphs) != 3
            or len(device_ir.rolled_reductions) != 1
        ):
            return False
        rolled = device_ir.rolled_reductions[0]
        producer, consumer = loops
        if (
            len(rolled.rolled_block_ids) != 1
            or producer.block_ids != rolled.rolled_block_ids
            or consumer.block_ids != rolled.rolled_block_ids
        ):
            return False
        original = device_ir.graphs[rolled.original_graph_id].graph
        reductions = [
            node
            for node in original.nodes
            if isinstance(node.meta.get("lowering"), ReductionLowering)
        ]
        if len(reductions) != 1:
            return False
        reduction_name = reductions[0].name
        if (
            sum(node.name == reduction_name for node in producer.graph.nodes) != 1
            or any(node.name == reduction_name for node in consumer.graph.nodes)
            or sum(node.target is language_store for node in original.nodes) != 1
        ):
            return False
        allowed = {
            _tracing_ops._host_tensor,
            _tracing_ops._get_symnode,
            language_load,
            language_store,
            subscript,
        }
        if any(
            node.op != "output"
            and node.target not in allowed
            and not isinstance(
                node.meta.get("lowering"), (PointwiseLowering, ReductionLowering)
            )
            for node in original.nodes
        ):
            return False
        charged = {
            node.name: node for node in original.nodes if node.target is language_load
        }
        names: list[set[str]] = []
        for loop in loops:
            present: set[str] = set()
            for node in loop.graph.nodes:
                if node.target is not language_load:
                    continue
                old = charged.get(node.name)
                if (
                    old is None
                    or not node.args
                    or not old.args
                    or not isinstance(node.args[0], torch.fx.Node)
                    or not isinstance(old.args[0], torch.fx.Node)
                    or node.args[0].meta.get("val") is None
                    or node.args[0].meta.get("val") is not old.args[0].meta.get("val")
                    or torch.fx.map_arg((node.args, node.kwargs), lambda n: n.name)
                    != torch.fx.map_arg((old.args, old.kwargs), lambda n: n.name)
                ):
                    return False
                present.add(node.name)
            names.append(present)
        # Retain the charge for every original load, including loads that
        # cannot be cached. Only the additional reserve is redundant when
        # an already-charged load exists exclusively in the consumer pass.
        return set.union(*names) == set(charged) and bool(names[1] - names[0])

    @classmethod
    def _packet_seeds(
        cls,
        env: CompileEnvironment,
        device_ir: DeviceIR,
        *,
        independent: bool = False,
        prefetch_already_counted: bool = False,
    ) -> list[Config]:
        spec = env.config_spec
        reduction = cast("ReductionLoopSpec", spec.reduction_loops[0])
        extent = env.block_sizes[reduction.block_id].numel
        if not isinstance(extent, sympy.Integer):
            return []
        size = int(extent)
        chunk = reduction._full_extent_chunk()
        if (
            chunk is None
            or size != reduction.size_hint
            or (spec.max_reduction_loop is not None and chunk > spec.max_reduction_loop)
        ):
            return []
        graph_ids = {
            info.original_graph_id
            for info in device_ir.rolled_reductions
            if info.rolled_block_ids == [reduction.block_id]
        }
        if not graph_ids:
            return []
        dtypes: set[torch.dtype] = set()
        row_strides: list[int] = []
        load_sites = 0
        matrix_source = False
        for graph_id in sorted(graph_ids):
            for node in device_ir.graphs[graph_id].graph.nodes:
                if node.target is not language_load:
                    continue
                if not node.args or not isinstance(node.args[0], torch.fx.Node):
                    return []
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
                    or any(type(stride) is not int for stride in tensor.stride()[:-1])
                ):
                    return []
                # Only a full source slice proves the intended packet address
                # family. Tensor strides alone do not rule out gathers or an
                # arithmetic permutation of the reduction coordinate.
                indices = node.args[1]
                if (
                    not isinstance(indices, (list, tuple))
                    or len(indices) != tensor.ndim
                    or not isinstance(indices[-1], slice)
                    or indices[-1] != slice(None)
                ):
                    return []
                row_strides.extend(tensor.stride()[:-1])
                dtypes.add(tensor.dtype)
                matrix_source |= tensor.ndim == 2
                load_sites += 1
        if not matrix_source or len(dtypes) != 1:
            return []
        itemsize = next(iter(dtypes)).itemsize
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
        block_domain = spec.block_sizes[0]._fragment(spec)
        if not block_domain.low <= 1 <= block_domain.high:
            return []
        fields = spec._flat_fields()
        replicated = fields.get("cute_replicated_reduction")
        pdl = fields.get("cute_pdl")
        independent_domain = fields.get("cute_independent_reduction")
        if independent and (
            not isinstance(independent_domain, EnumFragment)
            or True not in independent_domain.choices
            or not isinstance(replicated, EnumFragment)
            or not isinstance(pdl, EnumFragment)
            or not pdl.search_values()
        ):
            return []
        # Replication is deliberately seed-activated: test legal choices here,
        # then the completed compiler seed list enables its search choices.
        modifiers = (
            isinstance(replicated, EnumFragment)
            and True in replicated.choices
            and isinstance(pdl, EnumFragment)
            and True in (pdl.search_values() or ())
        )
        policy_domain = spec.load_eviction_policies.inner.search_values() or ()
        policy_indices = {
            fact.eviction_index
            for fact in spec.memory_op_facts
            if fact.kind == "load"
            and fact.eviction_index is not None
            and fact.dtype in dtypes
        }
        seeds = []
        for threads in (128, 256) if independent else (512, 1024):
            if (
                threads > (spec.max_reduction_threads or 1024)
                or chunk % threads
                or (load_sites + (not prefetch_already_counted)) * (chunk // threads)
                > 128
            ):
                continue
            vector_domain = (
                spec.cute_vector_widths.block_id_lookup(reduction.block_id)
                ._fragment(spec)
                .search_values()
                or ()
            )
            vector = next(
                (
                    value
                    for value in (8, 4, 2)
                    if value <= 16 // itemsize
                    and value in vector_domain
                    and chunk % (threads * value) == 0
                    and all(stride % value == 0 for stride in row_strides)
                ),
                None,
            )
            if vector is None:
                continue
            profiles: list[tuple[str, str | None, bool]] = [
                ("register", None, False),
                ("register_prefetch", None, False),
            ]
            if modifiers and policy_indices:
                profiles.extend(
                    ("register_prefetch", policy, True)
                    for policy in ("l2_last", "l1_l2_last")
                    if policy in policy_domain
                )
            if independent:
                assert isinstance(replicated, EnumFragment)
                profiles = [
                    ("register_prefetch", policy, replicate)
                    for replicate in (False, True)
                    if replicate in replicated.choices
                    for policy in ("l2_last", "l1_l2_last")
                    if policy in policy_domain and policy_indices
                ]
            for reload, policy, replicate in profiles:
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
                        spec.cute_reduction_reloads, {reduction.block_id: reload}
                    ),
                }
                if any(
                    value not in (item._fragment(spec).search_values() or ())
                    for name, sequence in sequences.items()
                    for item, value in zip(sequence, values[name], strict=True)
                ):
                    continue
                if policy is not None:
                    policies = spec.load_eviction_policies.default()
                    if any(not 0 <= index < len(policies) for index in policy_indices):
                        continue
                    for index in policy_indices:
                        policies[index] = policy
                    values.update(
                        cute_replicated_reduction=replicate,
                        cute_pdl=True,
                        load_eviction_policies=policies,
                    )
                if independent:
                    assert isinstance(pdl, EnumFragment)
                    values.update(
                        cute_independent_reduction=True,
                        cute_pdl=True in (pdl.search_values() or ()),
                    )
                seed = Config.from_dict(values)
                if independent:
                    # Use current legal domains, including any caller-imposed
                    # narrowing, and never seed a silently repaired geometry.
                    try:
                        generation = ConfigGeneration(spec)
                        flat, effective = generation.strict_config_pair(seed)
                        if generation.strict_unflatten(flat) != effective or any(
                            effective.config[key] != value
                            for key, value in values.items()
                        ):
                            continue
                    except InvalidConfig:
                        continue
                if seed not in seeds:
                    seeds.append(seed)
        return seeds
