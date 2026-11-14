"""tcgen05 GEMM config state of the CuTe backend: facts, surface, repair, seeds, SMEM.

``CuteTcgen05Config`` is the per-``ConfigSpec`` state record the autotuner
hooks read (``helion.autotuner.config_spec``): the bind-time proof fields
(``__init__``) and two ``config_spec`` views.  Every operation is a module
function over the state object in one of the role modules, called as
``<module>.<fn>(state, ...)``; the per-CTA SMEM capacity and budgets are the
device probes of ``..target_facts``:

* ``facts``    - ``register_*`` of the analyses' proofs, block / tile / launch readers
* ``smem``     - SMEM pricing: stage limits and byte models (through ``pipeline_smem.SmemArena``)
* ``surface``  - fragments, flat fields, search toggles
* ``repair``   - ``fix_search_config`` and the strict validators
* ``seeds``    - the families' seed configs
* ``knobs``    - key tables and seed-widening fragment helpers (a leaf)

Module-level names of the former single module are re-exported from here.

Role: knob table and planner facts (section 6 of ``../README.md``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from ....autotuner.config_spec import BlockSizeSpec
    from ....autotuner.config_spec import ConfigSpec
    from ....runtime.config import PidTypeLiteral
    from ..dense_gemm import DenseGemmFacts
    from ..epilogue_fanout import PairedFanoutPlan
    from ..grouped_prefix_adaptive_n import GroupedPrefixAdaptiveNProfile
    from ..materialized_pdl import MaterializedOperandPdl
    from ..one_cta_pipeline import Tcgen05OneCtaPipelineFacts
    from ..pipeline_smem import Tcgen05PipelineSmemFacts
    from ..tcgen05_constants import Tcgen05RowvecAuxFacts
    from ..tcgen05_schedule_facts import Tcgen05FullTileScheduleFacts


from .knobs import _TWO_CTA_PERSISTENT_PID_VETO
from .knobs import CUTE_TCGEN05_CODEGEN_PROOF_CONFIG_KEYS
from .knobs import CUTE_TCGEN05_DIAGNOSTIC_CONFIG_KEYS
from .knobs import CUTE_TCGEN05_GROUPED_CONFIG_KEYS
from .knobs import CUTE_TCGEN05_STRATEGY_CONFIG_KEYS
from .knobs import CUTE_TCGEN05_TUNABLE_KEYS
from .knobs import TCGEN05_C_STAGES_CHOICES
from .knobs import TCGEN05_C_STAGES_SEARCH_CHOICES
from .knobs import TCGEN05_GROUPED_DYNAMIC_AB4_STAGE
from .knobs import TCGEN05_GROUPED_DYNAMIC_RESERVED_SMEM_BYTES
from .knobs import TCGEN05_GROUPED_DYNAMIC_STAGE_TUPLES
from .knobs import TCGEN05_GROUPED_WORKLIST_NM_FOOTPRINT_MISMATCH
from .knobs import TCGEN05_GROUPED_WORKLIST_NM_SHAPE_MISMATCH
from .knobs import TCGEN05_L2_SWIZZLE_SEARCH_SIZES
from .knobs import TCGEN05_NUM_EPI_WARPS_SEARCH_CHOICES
from .knobs import Tcgen05AbStagesThreeSearchConstraints
from .knobs import Tcgen05ClusterM2SearchConstraints
from .knobs import Tcgen05GroupedWorklistSmemFacts
from .knobs import _compiler_seed_values
from .knobs import _CompilerSeedIntegerFragment
from .knobs import _enum_fragment_with_seed_values
from .knobs import _integer_fragment_with_seed_values
from .knobs import _SeedValue
from .knobs import _tcgen05_mma_impl_allowed
from .knobs import _tile_in_fragments
from .knobs import parse_tcgen05_grouped_static_problem_signature


class CuteTcgen05Config:
    """CuTe-owned tcgen05 ConfigSpec state and normalization hooks."""

    def __init__(self, config_spec: ConfigSpec) -> None:
        self.config_spec = config_spec
        self.search_enabled: bool = False
        self.matmul_block_ids: tuple[int, int, int] | None = None
        # Separate device launches may own independent native contractions.
        # They share pipeline knobs, but never a single matrix-axis projection.
        self.materialized_matmul_block_ids: tuple[tuple[int, int, int], ...] = ()
        # The launch (root) index each triple above belongs to.  Materialized
        # fission plans place one GEMM per region, so the indices are simply
        # ``0, 1, ...``; barrier phases may interleave plain SIMT launches
        # between the GEMM phases, and the per-region pipeline knobs index by
        # position in ``materialized_matmul_block_ids``, not by launch.
        self._materialized_matmul_region_indices: tuple[int, ...] = ()
        self.materialized_matmul_shapes: tuple[tuple[int, int, int], ...] = ()
        self.materialized_pipeline_facts: tuple[
            Tcgen05PipelineSmemFacts | None, ...
        ] = ()
        self.materialized_pair_budget_facts: tuple[
            Tcgen05PipelineSmemFacts | None, ...
        ] = ()
        # An ordered ``hl.barrier()`` chain of GEMM launches: its whole-kernel
        # ``pid_type`` is ``flat`` (the ordered launches), and
        # ``tcgen05_region_pid_types`` may run GEMM launches as static
        # persistent two-CTA pipelines (``_stage_config`` projects each
        # launch's pid type, persistence model and cluster shape).
        self.region_pid_type_search_enabled: bool = False
        # DeviceIR may later replace missing MatmulFact extents with runtime
        # hints. Keep the preflight plan's compile-time provenance for CuTe.
        self.matmul_compile_time_static_extents: (
            tuple[int | None, int | None, int | None] | None
        ) = None
        self.matmul_input_dtype: torch.dtype | None = None
        self.matmul_has_leading_passthrough: bool = False
        # Product of the leading passthrough (batch) extents: the work-tile
        # multiplier of every (m, n) tile grid.
        self.matmul_leading_work_multiplier: int = 1
        self.matmul_explicit_epi_tile_compatible: bool | None = None
        self.aux_kernel_detected: bool = False
        self.exact_shape_aux_kernel_detected: bool = False
        # Any exact-shape operand of the output's shape, whatever its dtype
        # (``host_function_has_tcgen05_exact_shape_aux_operand``).
        self.exact_shape_aux_operand_detected: bool = False
        # The device program's only side effect is one tile store
        # (``split_k_exchange.host_function_has_plain_store_body``): the
        # precondition of ``tcgen05_split_k``, whose last-arriver check
        # guards that store alone. Set at bind; a spec that never went
        # through bind keeps the knob off.
        self.split_k_plain_body: bool = False
        # Every analyzed store chain loads an auxiliary operand
        # (``aux_tensor.host_function_tcgen05_aux_loads_on_every_store``): the
        # proof that ``pre_acc_wait`` renders on the plain (non-fanout)
        # epilogue, where the store lowering refuses the placement at an
        # aux-free store.
        self.aux_loads_on_every_store: bool = False
        # Row-vector aux rows of the analyzed stores (``aux_tensor``): the SMEM
        # model of the ``pre_acc_wait`` row stages the seeds request.
        self.rowvec_aux_facts: Tcgen05RowvecAuxFacts | None = None
        # True when the kernel feeds a matmul an operand sourced from a load
        # whose dtype is not a tcgen05-native MMA dtype (e.g. an int16 tensor
        # cast to bf16, ``w.to(bfloat16)``). Such an operand cannot be TMA-staged
        # for the SMEM tcgen05 MMA, so the dot lowers through the non-tcgen05
        # fallback and the FFI/flat-role direct-entry seed must stay ineligible.
        self.matmul_has_non_tcgen05_operand: bool = False
        # False when a matmul A/B operand or the output store's destination
        # fails the TensorMap alignment proof the tcgen05 TMA pipeline needs
        # (e.g. a tensor whose base pointer or outer byte stride is not a
        # 16-byte multiple, or an N-major output). Codegen then keeps the
        # scalar SMEM producers or the SIMT store body, so the TMA-only
        # flat-role / FFI direct-entry seed must stay ineligible. Decided in
        # the compiler-facts phase by
        # ``CuteTcgen05ClusterM2FfiHeuristic.register_facts``.
        self.matmul_operands_tma_provable: bool = True
        # False when an exact-shape rank-2 auxiliary operand of the analyzed
        # store chains fails the aux TMA proof (a 16-byte base and outer
        # strides, a contiguous last axis: ``cute_mma``
        # ``_tcgen05_tma_aux_tensor_is_legal``). The staged aux modes
        # (``tma`` / ``epi_tma``) bulk-load those operands through a
        # row-major TensorMap and codegen refuses them otherwise, so the
        # search surface, the seeds and the normalizer mirror the proof.
        # Decided beside ``matmul_operands_tma_provable``.
        self.exact_aux_tma_provable: bool = True
        # Every MMA store's destination is exactly the GEMM's (M, N)
        # (``host_function_matmul_output_matches_extents``): the persistent
        # one-CTA N-edge tiles were measured on exact-shape outputs only.
        self.matmul_output_matches_extents: bool = True
        self.cluster_m_search_choices: tuple[int, ...] | None = None
        self.cluster_m2_search_constraints: Tcgen05ClusterM2SearchConstraints | None = (
            None
        )
        self.ab_stages_three_search_constraints: (
            Tcgen05AbStagesThreeSearchConstraints | None
        ) = None
        self.grouped_worklist_smem_facts: Tcgen05GroupedWorklistSmemFacts | None = None
        self.grouped_device_worklist_supported: bool = False
        self.grouped_source128_search_enabled: bool = False
        self.grouped_ab_k_fusion_supported: bool = False
        self.grouped_d_shift_supported: bool = False
        self.grouped_balanced_workers_supported: bool = False
        self.grouped_prefix_adaptive_n_supported: bool = False
        self.grouped_prefix_adaptive_n_search_enabled: bool = False
        self.grouped_prefix_adaptive_n_profile: GroupedPrefixAdaptiveNProfile | None = (
            None
        )
        self.grouped_prefix_adaptive_n_profiles: dict[
            int, GroupedPrefixAdaptiveNProfile
        ] = {}
        self.grouped_device_mn_swizzle_supported: bool = False
        self.grouped_full_coverage_supported: bool = False
        self.grouped_full_coverage_eligible: bool = False
        self.grouped_row_union_supported: bool = False
        self.grouped_row_union_cluster4_supported: bool = False
        self.grouped_row_union_cluster4_eligible: bool = False
        self.grouped_row_union_paired_clc_supported: bool = False
        self.grouped_row_union_paired_clc_eligible: bool = False
        self.grouped_row_union_eligible: bool = False
        self.grouped_row_union_multi_resident_supported: bool = False
        self.epilogue_fanout_plans: tuple[PairedFanoutPlan, ...] = ()
        self.epilogue_fanout_search_enabled: bool = False
        self.deep_direct_entry_validation_enabled: bool = False
        self.pipeline_smem_facts: Tcgen05PipelineSmemFacts | None = None
        self.dense_gemm_facts: DenseGemmFacts | None = None
        self.one_cta_pipeline_facts: Tcgen05OneCtaPipelineFacts | None = None
        # (K block id, owned max_size revision, exact owning object) of the
        # native K128 search cap that a proved dense pipeline may widen.
        self._native_k_search_cap: tuple[int, int, BlockSizeSpec] | None = None
        # The K block widened from that cap; seeds keep its former K128 geometry.
        self._native_k_widened_block: BlockSizeSpec | None = None
        self.full_tile_schedule_facts: Tcgen05FullTileScheduleFacts | None = None
        self.full_tile_restored_pid_types: tuple[PidTypeLiteral, ...] = ()
        self.full_tile_added_pairing: bool = False
        self.n_edge_cluster_schedule_facts: Tcgen05FullTileScheduleFacts | None = None
        self.n_edge_cluster_restored_pid_types: tuple[PidTypeLiteral, ...] = ()
        # Root identities from the typed operand-materialization dependency proof.
        self.materialized_operand_pdl_roots: MaterializedOperandPdl | None = None
        self.num_epi_warps_search_choices: tuple[int, ...] | None = None
        self.num_epi_warps_validation_choices: tuple[int, ...] | None = None
        # SM count of the bound device (0 when unknown); small-grid seeds use
        # it to decide whether the standard tiling can fill the machine.
        self.device_sm_count: int = 0
        # (key, widths) of ``_dense_aligned_n_clipped_widths``.
        self._dense_aligned_n_clipped_cache: tuple[object, frozenset[int]] | None = None
        # The register-MMA GEMM family (``cute_matmul_family="warp_mma"``):
        # admitted by the search plan (static latency-bound problem, one
        # matmul, supported layouts); its 16x8 tiles widen the M / N block
        # floors, which ``_normalize_warp_mma_family`` clamps back to the
        # recorded tcgen05 floors for tcgen05-family configs.
        self.warp_mma_admitted: bool = False
        # The grouped row-union kernel (``static_shapes=False``) has no
        # compile-time extents: the family is admitted on the first binding's
        # (m, n, k) hints, which codegen specializes under shape guards like
        # the row-union carrier does (``register_grouped_row_union_warp_mma``).
        self.warp_mma_grouped_extents: tuple[int, int, int] | None = None
        # False when an operand is an in-register integer cast only the
        # register-MMA family lowers (``_MmaOperandInfo.cast_dtype``): the
        # search offers ``cute_matmul_family="warp_mma"`` alone and a
        # tcgen05-family request fails closed with the reason.
        self.tcgen05_family_admitted: bool = True
        # True when the kernel is a materialized-fission bundle (several
        # launches with their own dependent-launch protocol): ``cute_pdl``
        # is not searched for it.
        self.fission_bundle: bool = False
        self.tcgen05_min_search_m: int = 64
        self.tcgen05_min_search_n: int = 8
        # A 16-bit output-edge GEMM (ragged M or N, K whole) admitted to the
        # persistent and CtaGroup.TWO schedules; persistent cluster_m=1 runs
        # whole tiles, or N-edge tiles on the partial-output TMA store
        # (``one_cta_n_edge_partial_output``), so its other partial-tile
        # samples are flattened.
        self.output_edge_persistent_admitted: bool = False
        # Block id of the leading passthrough (batch) axis, when any.
        self.matmul_leading_block_id: int | None = None

    @property
    def allowed_pid_types(self) -> tuple[PidTypeLiteral, ...]:
        return self.config_spec.allowed_pid_types

    @allowed_pid_types.setter
    def allowed_pid_types(self, value: tuple[PidTypeLiteral, ...]) -> None:
        self.config_spec.allowed_pid_types = value

    @property
    def materialized_matmul_region_indices(self) -> tuple[int, ...]:
        """Launch index of each ``materialized_matmul_block_ids`` triple."""
        if self._materialized_matmul_region_indices:
            return self._materialized_matmul_region_indices
        return tuple(range(len(self.materialized_matmul_block_ids)))

    @materialized_matmul_region_indices.setter
    def materialized_matmul_region_indices(self, value: tuple[int, ...]) -> None:
        self._materialized_matmul_region_indices = tuple(value)


__all__ = [
    "CUTE_TCGEN05_CODEGEN_PROOF_CONFIG_KEYS",
    "CUTE_TCGEN05_DIAGNOSTIC_CONFIG_KEYS",
    "CUTE_TCGEN05_GROUPED_CONFIG_KEYS",
    "CUTE_TCGEN05_STRATEGY_CONFIG_KEYS",
    "CUTE_TCGEN05_TUNABLE_KEYS",
    "TCGEN05_C_STAGES_CHOICES",
    "TCGEN05_C_STAGES_SEARCH_CHOICES",
    "TCGEN05_GROUPED_DYNAMIC_AB4_STAGE",
    "TCGEN05_GROUPED_DYNAMIC_RESERVED_SMEM_BYTES",
    "TCGEN05_GROUPED_DYNAMIC_STAGE_TUPLES",
    "TCGEN05_GROUPED_WORKLIST_NM_FOOTPRINT_MISMATCH",
    "TCGEN05_GROUPED_WORKLIST_NM_SHAPE_MISMATCH",
    "TCGEN05_L2_SWIZZLE_SEARCH_SIZES",
    "TCGEN05_NUM_EPI_WARPS_SEARCH_CHOICES",
    "_TWO_CTA_PERSISTENT_PID_VETO",
    "CuteTcgen05Config",
    "Tcgen05AbStagesThreeSearchConstraints",
    "Tcgen05ClusterM2SearchConstraints",
    "Tcgen05GroupedWorklistSmemFacts",
    "_CompilerSeedIntegerFragment",
    "_SeedValue",
    "_compiler_seed_values",
    "_enum_fragment_with_seed_values",
    "_integer_fragment_with_seed_values",
    "_tcgen05_mma_impl_allowed",
    "_tile_in_fragments",
    "parse_tcgen05_grouped_static_problem_signature",
]
