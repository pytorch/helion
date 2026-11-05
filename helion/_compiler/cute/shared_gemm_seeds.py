"""Canonical full-kernel seeds for an already-proved shared GEMM epilogue.

This is not an admission or single-output allocation proof. The complete original
shared DAG is re-proved locally; every candidate prices all its output arenas
and passes the existing strict protocol/normalization and flat-transfer gates.
The hook runs after legacy coverage registration, changing no random fragment,
source expression, default config, classifier or persistent proof certificate.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING

import torch

from ...autotuner.config_generation import ConfigGeneration
from ...exc import InvalidConfig
from ...language import memory_ops
from ...runtime.config import Config
from . import cute_epilogue as epi
from .epilogue_fanout import PairedFanoutPlan
from .epilogue_fanout import _chain_key
from .shared_epilogue_dag import SharedDagFanoutPlan
from .shared_epilogue_dag import dag_keys
from .shared_epilogue_dag import prove_epilogue_fanout
from .shared_epilogue_dag import prove_shared_dag_fanout
from .shared_epilogue_dag import schedule_supported
from .tcgen05_schedule_facts import native_tma_schedule_supported

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .shared_epilogue_dag import FanoutPlan


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SharedMmaGeometry:
    """Original anchor geometry, deliberately not a one-output schedule proof."""

    shape: tuple[int, int, int]
    dtype: torch.dtype
    rhs_k_major: bool
    lhs_k_major: bool
    block_ids: tuple[int, int, int]


@dataclass(frozen=True)
class SharedGemmSeedFacts:
    # Geometry comes from the original MMA, not a representative output. The
    # separate complete shared proof and all-store model own output allocation.
    geometry: SharedMmaGeometry
    whole: SharedDagFanoutPlan
    protocol: FanoutPlan


def _strict_protocol_key(plan: FanoutPlan) -> tuple[object, ...]:
    # Auxiliary expression carriers are deliberately eq=False and are rebuilt
    # by every walker. Reuse the unchanged final renderer's strict FX/DAG keys,
    # not carrier object identity or the independent semantic-value memo.
    base = (
        type(plan),
        plan.stores,
        plan.output_dtype,
        plan.block_ids,
        plan.shape,
        plan.pre_wait_aux_safe,
    )
    if isinstance(plan, SharedDagFanoutPlan):
        keys = []
        for chain in plan.chains:
            if len(chain.steps) != 1 or not isinstance(
                chain.steps[0].expr, epi._PointwiseDagExpr
            ):
                raise ValueError("incomplete shared source proof")
            keys.append(dag_keys(chain.steps[0].expr))
        return (*base, tuple(keys))
    assert isinstance(plan, PairedFanoutPlan)
    return (
        *base,
        plan.prefix_steps,
        plan.rounded_suffix,
        plan.unrounded_prefix,
        tuple(_chain_key(chain) for chain in plan.chains),
    )


def prove_shared_gemm_seed_facts(
    env: CompileEnvironment, device_ir: DeviceIR
) -> SharedGemmSeedFacts | None:
    from ...language import _tracing_ops
    from .aux_tensor import _tcgen05_aux_detector_mma_nodes
    from .cute_mma import analyze_cute_mma_node
    from .promote_output_axis import _ordinary_operation

    state = env.config_spec._cute_tcgen05_config
    host = device_ir.host_function
    axes = state.matmul_block_ids
    shape = state.matmul_compile_time_static_extents
    stores = tuple(
        node
        for info in device_ir.graphs
        for node in info.graph.nodes
        if node.target is memory_ops.store
    )
    if (
        not state.search_enabled
        or host is None
        or not 2 <= len(stores) <= 4
        or axes is None
        or shape is None
        or env.config_spec.target_device_capability not in ((10, 0), (10, 3))
        or not state.matmul_operands_tma_provable
        or state.materialized_matmul_block_ids
        or env.cute_fission_plan is not None
        or len(device_ir.root_ids) != 1
        or device_ir.has_atomic_ops()
        or env.config_spec.reduction_block_ids - {axes[2]}
    ):
        return None
    m, n, k = shape
    if (
        m is None
        or n is None
        or k is None
        or any(type(value) is not int or value <= 0 for value in (m, n, k))
    ):
        return None
    shape = (m, n, k)
    if any(
        not (
            _ordinary_operation(node)
            or node.target in epi._ZERO_ARG_TARGETS
            or _tracing_ops.is_for_loop_target(node.target)
        )
        for info in device_ir.graphs
        for node in info.graph.nodes
    ):
        return None
    anchors = _tcgen05_aux_detector_mma_nodes(device_ir.graphs)
    if len(anchors) != 1:
        return None
    (anchor,) = anchors
    with host:
        candidate = analyze_cute_mma_node(anchor, device_ir=device_ir)
        if candidate is None or candidate.requires_accumulator_seed:
            return None
        operands = candidate.operands
        if (
            candidate.requires_fragment_epilogue
            or operands.has_leading_passthrough
            or operands.output_block_ids != axes[:2]
            or (operands.m_block_id, operands.n_block_id, operands.k_block_id) != axes
            or operands.lhs.source_fake.ndim != 2
            or operands.rhs.source_fake.ndim != 2
            or operands.lhs.source_fake.dtype not in (torch.float16, torch.bfloat16)
            or operands.rhs.source_fake.dtype != operands.lhs.source_fake.dtype
            or operands.lhs.matrix_major not in ("row", "col")
            or operands.rhs.matrix_major not in ("row", "col")
            or operands.lhs.source_to_logical_order is not None
            or operands.rhs.source_to_logical_order is not None
            or operands.rhs.rhs_is_grouped
        ):
            return None
        # Registration is before the managed snapshot publication. Reuse only
        # the existing complete proof's registered-live lookup, never install
        # unbound facts or relax any final rendering reader.
        whole = prove_shared_dag_fanout(
            env, host, device_ir.graphs, stores, {anchor}, allow_unbound=True
        )
        protocol = prove_epilogue_fanout(
            env, host, device_ir.graphs, stores, {anchor}, allow_unbound=True
        )
    if (
        whole is None
        or protocol is None
        or whole.block_ids != axes[:2]
        or whole.shape != shape[:2]
        or not whole.pre_wait_aux_safe
    ):
        return None
    try:
        matched = any(
            _strict_protocol_key(protocol) == _strict_protocol_key(plan)
            for plan in state.epilogue_fanout_plans
        )
    except ValueError:
        return None
    if not matched:
        return None
    allocations = state.rowvec_aux_facts
    budget = state.ab_stages_three_search_constraints
    if (
        allocations is None
        or budget is None
        or allocations.output_count != len(whole.stores)
        or allocations.output_itemsize != whole.output_dtype.itemsize
        or budget.dtype_bytes != operands.lhs.source_fake.dtype.itemsize
        or budget.output_stores != len(whole.stores)
        or budget.output_dtype_bytes != whole.output_dtype.itemsize
    ):
        return None
    return SharedGemmSeedFacts(
        SharedMmaGeometry(
            shape,
            operands.lhs.source_fake.dtype,
            rhs_k_major=operands.rhs.matrix_major == "col",
            lhs_k_major=operands.lhs.matrix_major == "row",
            block_ids=axes,
        ),
        whole,
        protocol,
    )


def shared_gemm_seed_configs(
    env: CompileEnvironment, device_ir: DeviceIR
) -> list[Config]:
    if env.settings.disable_autotuner_heuristics:
        return []
    facts = prove_shared_gemm_seed_facts(env, device_ir)
    if facts is None:
        return []
    spec = env.config_spec
    state = spec._cute_tcgen05_config
    try:
        generation = ConfigGeneration(spec)
    except InvalidConfig:
        return []
    identities = generation._flat_coordinate_identities()

    def values(key: str, sequence_index: int | None = None) -> tuple[object, ...]:
        indices = [
            i
            for i, (name, index) in enumerate(identities)
            if name == key and (sequence_index is None or index == sequence_index)
        ]
        if len(indices) != 1:
            return ()
        fragment = generation.flat_spec[indices[0]]
        # Enum choices preserve strictly valid seeded values even when a prior
        # preference keeps them out of random sampling. Flat transfer below
        # still must retain every field without repairing the configuration.
        from ...autotuner.config_fragment import EnumFragment

        if isinstance(fragment, EnumFragment):
            return tuple(fragment.choices)
        result = fragment.search_values(limit=64)
        return () if result is None else tuple(result)

    axes = facts.geometry.block_ids
    indices = tuple(spec.block_sizes.block_id_to_index(axis) for axis in axes)
    # Public defaults deliberately omit some inactive fields (e.g. the
    # min-blocks coverage knob). Build a complete seed-owned default through
    # the existing flat schema before deriving a schedule, so its serialized
    # round trip does not silently add fields.
    try:
        _, base = generation.canonicalize_flat(
            generation.flatten(spec.default_config())
        )
    except InvalidConfig:
        return []
    # Persistent families are an existing strict protocol, even when a
    # preference leaves only flat in the random fragment. Do not change that
    # fragment: explicit normalization and complete flat transfer below must
    # preserve this requested family or the optional seed is discarded.
    pids = ("persistent_interleaved", "persistent_blocked")
    if 2 not in values("tcgen05_acc_stages"):
        return []
    result: list[Config] = []
    for cm in (1, 2):
        if cm not in values("tcgen05_cluster_m"):
            continue
        for bm in (128, 64, 256):
            if bm not in values("block_sizes", indices[0]):
                continue
            for bn in (256, 128, 64, 32, 16):
                if bn not in values("block_sizes", indices[1]):
                    continue
                for bk in (128, 64, 32):
                    if bk not in values("block_sizes", indices[2]) or not (
                        native_tma_schedule_supported(
                            facts.geometry, (bm, bn, bk), cluster_m=cm, cluster_n=1
                        )
                    ):
                        continue
                    for c_stages in (2, 4):
                        if c_stages not in values("tcgen05_c_stages"):
                            continue
                        admitted: list[Config] = []
                        for ab in values("tcgen05_ab_stages"):
                            if (
                                type(ab) is not int
                                or not 1 <= ab <= facts.geometry.shape[2] // bk
                            ):
                                continue
                            config = Config.from_dict(deepcopy(base.config))
                            blocks = config.block_sizes.copy()
                            for index, value in zip(indices, (bm, bn, bk), strict=True):
                                blocks[index] = value
                            config.config.update(
                                block_sizes=blocks,
                                pid_type=pids[0],
                                tcgen05_persistence_model="static_persistent",
                                tcgen05_cta_group="auto",
                                tcgen05_cluster_m=cm,
                                tcgen05_cluster_n=1,
                                tcgen05_ab_stages=ab,
                                tcgen05_acc_stages=2,
                                tcgen05_c_stages=c_stages,
                                tcgen05_epilogue_fanout="shared",
                            )
                            if not schedule_supported(facts.protocol, config.config):
                                continue
                            if not state.default_layout_smem_fits(
                                bm=bm,
                                bn=bn,
                                bk=bk,
                                cluster_m=cm,
                                ab_stages=ab,
                                c_stages=c_stages,
                                stage_rows=state._store_lowering_stages_rows(
                                    config.config
                                ),
                                epi_warps=state._config_epi_warps(config.config),
                                acc_stages=2,
                            ):
                                continue
                            try:
                                normalized = spec.normalized_config(config)
                                if normalized != config:
                                    continue
                                flat = generation.flatten(config)
                                _, transferred = generation.canonicalize_flat(flat)
                                if transferred != config:
                                    continue
                            except (InvalidConfig, ValueError, KeyError):
                                continue
                            admitted.append(config)
                        if admitted:
                            admitted.sort(key=lambda c: c.config["tcgen05_ab_stages"])
                            # A shallow/deep pair per proved geometry/C depth;
                            # deterministic, no random draws or extra arenas.
                            for config in (admitted[0], admitted[-1]):
                                if config not in result:
                                    result.append(config)
    return result


def register_shared_gemm_seed_encoding(
    env: CompileEnvironment, device_ir: DeviceIR
) -> None:
    """Declare primitive seed-only encodings before coverage records its layout.

    This is the same complete seeder proof, not an independent shape/count
    approximation. No seeds, random choices or proof certificates are published.
    A failed optional probe cannot later commit a newly encoded seed family.
    """
    state = env.config_spec._cute_tcgen05_config
    state._shared_gemm_seed_encoding_available = False
    if (
        env.config_spec.compiler_seed_configs
        or env.settings.disable_autotuner_heuristics
        or not state.epilogue_fanout_search_enabled
    ):
        return
    try:
        state._shared_gemm_seed_encoding_available = (
            prove_shared_gemm_seed_facts(env, device_ir) is not None
        )
    except Exception:
        log.debug("optional shared-GEMM seed encoding proof failed", exc_info=True)


def append_shared_gemm_seed_configs(
    env: CompileEnvironment, device_ir: DeviceIR
) -> None:
    """Isolate optional discovery, never suppress explicit selected-config errors."""
    spec = env.config_spec
    if (
        spec.compiler_seed_configs
        or not spec._cute_tcgen05_config._shared_gemm_seed_encoding_available
    ):
        return
    # The registry contributes directly to the autotune cache fingerprint.
    # Stage the entire optional population before committing: a generator or
    # dedup failure must not leave one accepted prefix in that identity. Keep
    # registration metadata by reference; discovery owns no correctness facts.
    registry = spec.compiler_seed_configs
    original_seeds = registry.copy()
    original_default = spec.compiler_default_config
    original_overflow = spec.compiler_seed_overflow_requested
    original_retry = spec.compiler_seed_timeout_retry_repetitions
    heuristics = spec.autotuner_heuristics
    original_heuristics = heuristics.copy()
    original_facts = env.compiler_fact_specialization_facts
    try:
        staged = original_seeds.copy()
        for config in shared_gemm_seed_configs(env, device_ir):
            if config not in staged:
                staged.append(config)
    except Exception:
        # Only this additive-discovery call is optional. Normalization and
        # rendering of an explicitly selected config remain outside this guard.
        registry[:] = original_seeds
        spec.compiler_seed_configs = registry
        spec.compiler_default_config = original_default
        spec.compiler_seed_overflow_requested = original_overflow
        spec.compiler_seed_timeout_retry_repetitions = original_retry
        heuristics[:] = original_heuristics
        spec.autotuner_heuristics = heuristics
        env.compiler_fact_specialization_facts = original_facts
        log.debug("optional shared-GEMM seed discovery failed", exc_info=True)
        return
    registry[:] = staged
