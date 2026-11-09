"""Whole-kernel generators and the regeneration policy of the CuTe backend.

``KERNEL_GENERATORS`` is the ordered tuple ``CuteBackend.kernel_generators``
returns: ``_generate_ast`` runs the first generator that applies to the config
and whose transform policy honors the call; a generator that applies but does
not honor the call declines as its family always did (``InvalidConfig``,
``BackendUnsupported``, or stripping / rewriting its knobs so the later
generators and the ordinary pass continue with that config).  First match wins,
so the order is part of the semantics.

``REGENERATION`` is the ``RetryPolicy`` for the two CuTe decisions made before
the tile body exists and revisited once it does, each by generating the kernel
again (see ``CuteRegeneration``).
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import logging
from typing import TYPE_CHECKING
from typing import cast

from ... import exc
from ...runtime.config import Config
from ..compile_environment import CompileEnvironment
from ..emitter_protocols import KernelGenerator
from ..emitter_protocols import KernelTransforms
from ..emitter_protocols import RetryPolicy
from ..emitter_protocols import TransformPolicy
from . import target_facts
from .fragment_mma_dot import FragmentMmaNotApplied
from .fragment_mma_dot import fragment_mma_off_config
from .register_tile_admission import RegisterTileUnsupported
from .ring_mma_dot import RingMmaNotApplied
from .ring_mma_dot import ring_mma_off_config
from .sink_vector_loops import VloopSinkNotApplied
from .sink_vector_loops import vloop_sink_off_config
from .state import cute_env_state
from .tcgen05_flat_grouped_config import CONFIG_KEYS as GROUPED_CONFIG_KEYS
from .tcgen05_flat_grouped_config import K_KEY as GROUPED_RNA_K_KEY
from .tcgen05_flat_grouped_config import K_STAGES
from .tcgen05_flat_grouped_config import STAGES_KEY as GROUPED_RNA_STAGES_KEY
from .tcgen05_flat_grouped_config import WARPS_KEY as GROUPED_RNA_WARPS_KEY
from .unroll_lane_loads import LaneUnrollNotApplied
from .unroll_lane_loads import lane_unroll_off_config

if TYPE_CHECKING:
    import ast
    from collections.abc import Callable
    from collections.abc import Iterator

    from ..host_function import HostFunction

log: logging.Logger = logging.getLogger(__name__)

_ALL_TRANSFORMS = frozenset(
    (
        "store_transform",
        "load_transform",
        "extra_params",
        "codegen_graphs",
        "memory_counters",
        "host_prefix",
        "bounded_cache_request",
    )
)
_ALL_BUT_BOUNDED_CACHE = _ALL_TRANSFORMS - {"bounded_cache_request"}
_MEMORY_TRANSFORMS = frozenset(("store_transform", "load_transform", "extra_params"))


def _refusing(error: Callable[[], Exception]) -> Callable[[Config], Config]:
    def decline(config: Config) -> Config:
        raise error()

    return decline


def _without_keys(keys: frozenset[str]) -> Callable[[Config], Config]:
    def decline(config: Config) -> Config:
        return Config.from_dict(
            {key: value for key, value in config.config.items() if key not in keys}
        )

    return decline


def _scalar_reduction_sequence(config: Config) -> Config:
    # Unsupported external transforms keep the ordinary lowering.
    return type(config).from_dict(
        {**config.config, "cute_reduction_sequence": "scalar"}
    )


def _complete_kernel_only(message: str) -> TransformPolicy:
    return TransformPolicy(
        excluded=_ALL_TRANSFORMS,
        decline=_refusing(lambda: exc.InvalidConfig(message)),
    )


def _unsupported_transforms(excluded: frozenset[str], detail: str) -> TransformPolicy:
    return TransformPolicy(
        excluded=excluded,
        decline=_refusing(lambda: exc.BackendUnsupported("cute", detail)),
    )


def _plain_generator(
    module: str, name: str
) -> Callable[[HostFunction, Config, bool, KernelTransforms], ast.Module]:
    """``generate_*(func, config, emit_repro_caller)`` of
    ``helion._compiler.cute.<module>``, imported when it runs."""

    def generate(
        func: HostFunction,
        config: Config,
        emit_repro_caller: bool,
        transforms: KernelTransforms,
    ) -> ast.Module:
        entry = getattr(
            importlib.import_module(f"helion._compiler.cute.{module}"), name
        )
        return entry(func, config, emit_repro_caller)

    return generate


def _generate_grouped_warp_tf32(
    func: HostFunction,
    config: Config,
    emit_repro_caller: bool,
    transforms: KernelTransforms,
) -> ast.Module:
    from .tcgen05_flat_grouped_codegen import generate_grouped_warp_tf32_candidate

    return generate_grouped_warp_tf32_candidate(
        func,
        config,
        emit_repro_caller,
        shared_capacity=target_facts.per_cta_smem_capacity_bytes(
            CompileEnvironment.current().device
        ),
    )


def _generate_flat_grouped_native(
    func: HostFunction,
    config: Config,
    emit_repro_caller: bool,
    transforms: KernelTransforms,
) -> ast.Module:
    from .tcgen05_flat_grouped_codegen import generate_flat_grouped_native_candidate

    block_k = config[GROUPED_RNA_K_KEY]
    converter_warps = config.get(GROUPED_RNA_WARPS_KEY, 0)
    assert isinstance(block_k, int)
    assert isinstance(converter_warps, int)
    return generate_flat_grouped_native_candidate(
        func,
        config,
        emit_repro_caller,
        converter_warps=converter_warps,
        block_k=block_k,
        ab_stages=cast("int", config.get(GROUPED_RNA_STAGES_KEY, K_STAGES[block_k])),
        shared_capacity=target_facts.per_cta_smem_capacity_bytes(
            CompileEnvironment.current().device
        ),
    )


def _generate_materialized_fission(
    func: HostFunction,
    config: Config,
    emit_repro_caller: bool,
    transforms: KernelTransforms,
) -> ast.Module:
    from .materialized_fission_codegen import generate_materialized_fission

    plan = cute_env_state(CompileEnvironment.current()).fission_plan
    assert plan is not None
    return generate_materialized_fission(
        func,
        config,
        emit_repro_caller,
        plan,
        store_transform=transforms.store_transform,
        load_transform=transforms.load_transform,
        extra_params=transforms.extra_params,
    )


def _generate_ordered_phases(
    func: HostFunction,
    config: Config,
    emit_repro_caller: bool,
    transforms: KernelTransforms,
) -> ast.Module:
    from .ordered_phases import generate_ordered_phases

    return generate_ordered_phases(
        func,
        config,
        emit_repro_caller,
        store_transform=transforms.store_transform,
        load_transform=transforms.load_transform,
        extra_params=transforms.extra_params,
    )


KERNEL_GENERATORS: tuple[KernelGenerator, ...] = (
    KernelGenerator(
        "grouped_prefix_adaptive_n",
        applies=lambda func, config, transforms: bool(
            config.get("tcgen05_grouped_prefix_adaptive_n", False)
        ),
        generate=_plain_generator(
            "grouped_prefix_adaptive_n_codegen",
            "generate_grouped_prefix_adaptive_n_candidate",
        ),
        transform_policy=_complete_kernel_only(
            "grouped prefix-adaptive N requires complete ordinary codegen"
        ),
    ),
    KernelGenerator(
        "packed_operand",
        applies=lambda func, config, transforms: (
            config.get("cute_materialized_operand_schedule", "off") != "off"
        ),
        generate=_plain_generator("packed_operand_codegen", "generate_packed_operand"),
        transform_policy=TransformPolicy(
            excluded=_ALL_TRANSFORMS,
            config_conflict=lambda config: (
                config.get("cute_materialized_schedule", "off") != "off"
            ),
            decline=_refusing(
                lambda: exc.InvalidConfig(
                    "packed-operand schedule requires complete ordinary codegen"
                )
            ),
        ),
    ),
    KernelGenerator(
        "row_resident",
        applies=lambda func, config, transforms: (
            config.get("cute_materialized_schedule", "off") != "off"
        ),
        generate=_plain_generator("row_resident_codegen", "generate_row_resident"),
        transform_policy=_complete_kernel_only(
            "row-resident schedule does not support external transforms or partial codegen"
        ),
    ),
    KernelGenerator(
        "grouped_warp_tf32",
        applies=lambda func, config, transforms: (
            config.get("cute_mma_f32_conversion") == "warp_raw"
        ),
        generate=_generate_grouped_warp_tf32,
        transform_policy=TransformPolicy(
            excluded=_ALL_TRANSFORMS, decline=_without_keys(GROUPED_CONFIG_KEYS)
        ),
    ),
    KernelGenerator(
        "flat_grouped_native",
        applies=lambda func, config, transforms: bool(
            config.get("cute_grouped_rna_warps", 0)
            or config.get("cute_mma_f32_conversion") == "tma_rn"
        ),
        generate=_generate_flat_grouped_native,
        transform_policy=TransformPolicy(
            excluded=_ALL_TRANSFORMS, decline=_without_keys(GROUPED_CONFIG_KEYS)
        ),
    ),
    KernelGenerator(
        "bounded_cache",
        applies=lambda func, config, transforms: (
            config.get("cute_reduction_sequence") in {"bounded", "bounded_layout"}
        ),
        generate=_plain_generator("bounded_cache_codegen", "generate_bounded_cache"),
        transform_policy=TransformPolicy(
            excluded=_ALL_BUT_BOUNDED_CACHE, decline=_scalar_reduction_sequence
        ),
    ),
    KernelGenerator(
        "split_k_cluster",
        applies=lambda func, config, transforms: (
            config.get("cute_split_k_schedule") == "cluster8_k4"
        ),
        generate=_plain_generator(
            "split_k_cluster_codegen", "generate_split_k_cluster"
        ),
        transform_policy=_unsupported_transforms(
            _ALL_BUT_BOUNDED_CACHE,
            "cluster split-K excludes external lowering transforms",
        ),
    ),
    KernelGenerator(
        "split_k_workspace",
        applies=lambda func, config, transforms: bool(
            config.get("cute_split_k_workspace", False)
            and transforms.codegen_graphs is None
        ),
        generate=_plain_generator(
            "split_k_workspace_codegen", "generate_split_k_workspace"
        ),
        transform_policy=_unsupported_transforms(
            _MEMORY_TRANSFORMS,
            "split-K workspace does not support external memory transforms",
        ),
    ),
    KernelGenerator(
        "materialized_fission",
        applies=lambda func, config, transforms: (
            cute_env_state(CompileEnvironment.current()).fission_plan is not None
            and len(func.device_ir.root_ids) > 1
        ),
        generate=_generate_materialized_fission,
    ),
    KernelGenerator(
        "ordered_phases",
        applies=lambda func, config, transforms: (
            len(func.device_ir.phases) > 1 and config.pid_type == "flat"
        ),
        generate=_generate_ordered_phases,
    ),
)


@dataclasses.dataclass(frozen=True)
class CuteRegeneration(RetryPolicy):
    """Generate the kernel again when a before-the-body decision is revisited.

    - A persistent reduction chooses its register-tile lane nesting from the
      device IR; when the generated body then holds a statement the two-pass
      register-tile schedule cannot place (``register_tile_admission.py``,
      ``RegisterTileUnsupported``), the kernel is generated again without the
      register tile.  The register tile was the only reason normalization kept
      a reduction thread count below the extent persistent, so the retry
      re-normalizes ``config`` with register tiles withheld: the looped
      reduction when the remaining threads cannot cover the extent, otherwise
      the same config with the rolled lane nesting the strategy used before
      register tiles existed.  Caller-built codegen graphs were rolled for
      ``config`` as given, so that retry keeps the config and only the lane
      nesting changes.
    - ``cute_vloop_sink`` shapes the thread layout during layout planning;
      when no vector loop is sunk after all (``VloopSinkNotApplied``), the
      kernel is generated again with the knob off (``vloop_sink_off_config``)
      so the knob alone never changes the code.  Layout planning annotated a
      caller's graphs in place, so a caller that supplied them rebuilds them
      for the knob-off run itself (the signal propagates).
    - ``cute_lane_unroll`` without sinking asks the last kernel-body pass to
      unroll the grid lane loops loads-first; when no loop takes it
      (``LaneUnrollNotApplied``) the kernel is generated again with the
      unroll off (``lane_unroll_off_config``), for the same reason.  The
      unroll is a late AST pass that touches no graph, so the knob-off run
      reuses the caller's graphs as they are.
    - ``cute_dot_fragment_mma`` / ``cute_dot_ring_mma`` rewrite a GEMV dot
      onto ``mma.sync`` fragments (through a cp.async ring for the latter) in
      the same late position; when no dot takes the rewrite
      (``FragmentMmaNotApplied`` / ``RingMmaNotApplied``) the kernel is
      generated again with that knob off (``fragment_mma_off_config`` /
      ``ring_mma_off_config``), reusing the caller's graphs.

    The retries compose: the register tile is admitted afresh on the knob-off
    pass, and the sink pass runs on the rolled body of a rejected register
    tile.  Each fires at most once per ``generate_ast`` call (its repair
    switches off what raised it), so a call generates the kernel at most
    three times.
    """

    signals: tuple[type[BaseException], ...] = (
        RegisterTileUnsupported,
        VloopSinkNotApplied,
        FragmentMmaNotApplied,
        RingMmaNotApplied,
        LaneUnrollNotApplied,
    )

    @contextlib.contextmanager
    def scope(self) -> Iterator[None]:
        env_state = cute_env_state(CompileEnvironment.current())
        register_tiles_disabled = env_state.register_tile_disabled
        try:
            yield
        finally:
            env_state.register_tile_disabled = register_tiles_disabled

    def repair(
        self, signal: BaseException, config: Config, *, caller_graphs: bool
    ) -> Config | None:
        env = CompileEnvironment.current()
        env_state = cute_env_state(env)
        if isinstance(signal, RegisterTileUnsupported):
            if env_state.register_tile_disabled:
                return None
            log.debug("regenerating with the rolled reduction lane: %s", signal)
            if not caller_graphs:
                retried = env.config_spec.normalized_config(
                    config, _cute_register_tiles=False
                )
                if retried.config != config.config:
                    log.debug(
                        "the retry loops the reduction: reduction_loops=%s",
                        retried.reduction_loops,
                    )
                    config = retried
            env_state.register_tile_disabled = True
            return config
        if isinstance(signal, VloopSinkNotApplied):
            if caller_graphs:
                return None
            return vloop_sink_off_config(config)
        if isinstance(signal, FragmentMmaNotApplied):
            # The fragment rewrite is a late AST pass like the unroll: the
            # knob-off run reuses the caller's graphs.
            return fragment_mma_off_config(config)
        if isinstance(signal, RingMmaNotApplied):
            # The ring rewrite is a late AST pass like the fragment one.
            return ring_mma_off_config(config)
        assert isinstance(signal, LaneUnrollNotApplied)
        return lane_unroll_off_config(config)


REGENERATION = CuteRegeneration()
