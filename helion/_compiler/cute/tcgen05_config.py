from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from typing import cast

import torch

from ...autotuner.config_fragment import BooleanFragment
from ...autotuner.config_fragment import ConfigSpecFragment
from ...autotuner.config_fragment import EnumFragment
from ...autotuner.config_fragment import IntegerFragment
from ...exc import InvalidConfig
from ...runtime.config import Config
from .strategies import ROLE_LOCAL_MONOLITHIC_DEFAULT_WARP_SPEC
from .strategies import TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY
from .strategies import TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY
from .strategies import TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY
from .strategies import TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY
from .strategies import TCGEN05_LAYOUT_OVERRIDES_KEYS
from .strategies import TCGEN05_LAYOUT_OVERRIDES_SWIZZLE_A_KEY
from .strategies import TCGEN05_LAYOUT_OVERRIDES_SWIZZLE_B_KEY
from .strategies import TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY
from .strategies import TCGEN05_LEGAL_L2_SWIZZLE_SIZES
from .strategies import TCGEN05_LEGAL_SMEM_SWIZZLE_BYTES
from .strategies import TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY
from .strategies import TCGEN05_STRATEGY_CONFIG_KEY
from .strategies import TCGEN05_STRATEGY_CONFIG_KEYS
from .strategies import TCGEN05_WARP_SPEC_AB_LOAD_WARPS_KEY
from .strategies import TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY
from .strategies import TCGEN05_WARP_SPEC_DEFAULTS_BY_KEY
from .strategies import TCGEN05_WARP_SPEC_EPI_LOAD_WARPS_KEY
from .strategies import TCGEN05_WARP_SPEC_MMA_WARPS_KEY
from .strategies import TCGEN05_WARP_SPEC_REGISTER_DECREASE_KEY
from .strategies import TCGEN05_WARP_SPEC_REGISTER_INCREASE_KEY
from .strategies import TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY
from .strategies import TCGEN05_WARP_SPEC_STORE_WARPS_KEY
from .strategies import Tcgen05LayoutStrategy
from .strategies import Tcgen05PersistenceModel
from .strategies import Tcgen05Strategy
from .strategies import derive_persistence_model_from_pid_type
from .strategies import layout_overrides_from_config
from .strategies import validate_tcgen05_strategy_invariants
from .strategies import warp_spec_from_config
from .tcgen05_constants import TCGEN05_AB_CONSUMER_PHASE_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_AB_CONSUMER_PHASE_MODE_NORMAL
from .tcgen05_constants import TCGEN05_AB_CONSUMER_PHASE_MODES
from .tcgen05_constants import TCGEN05_AB_CONSUMER_WAIT_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_AB_CONSUMER_WAIT_MODE_NORMAL
from .tcgen05_constants import TCGEN05_AB_CONSUMER_WAIT_MODES
from .tcgen05_constants import TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODE_NORMAL
from .tcgen05_constants import TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODES
from .tcgen05_constants import TCGEN05_AB_PRODUCER_ACQUIRE_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_AB_PRODUCER_ACQUIRE_MODE_NORMAL
from .tcgen05_constants import TCGEN05_AB_PRODUCER_ACQUIRE_MODES
from .tcgen05_constants import TCGEN05_AB_PRODUCER_ADVANCE_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_AB_PRODUCER_ADVANCE_MODE_NORMAL
from .tcgen05_constants import TCGEN05_AB_PRODUCER_ADVANCE_MODES
from .tcgen05_constants import TCGEN05_AB_STAGES_THREE_MIN_DEVICE_SMEM_OPTIN
from .tcgen05_constants import TCGEN05_AB_STAGES_THREE_RESERVED_SMEM_BYTES
from .tcgen05_constants import TCGEN05_ACC_PRODUCER_ADVANCE_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_ACC_PRODUCER_ADVANCE_MODE_NORMAL
from .tcgen05_constants import TCGEN05_ACC_PRODUCER_ADVANCE_MODES
from .tcgen05_constants import TCGEN05_ACC_PRODUCER_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_ACC_PRODUCER_MODE_NORMAL
from .tcgen05_constants import TCGEN05_ACC_PRODUCER_MODES
from .tcgen05_constants import TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY
from .tcgen05_constants import TCGEN05_ACC_WAIT_PLACEMENTS
from .tcgen05_constants import TCGEN05_AUX_LOAD_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_AUX_LOAD_MODE_SIMT
from .tcgen05_constants import TCGEN05_AUX_LOAD_MODE_TMA
from .tcgen05_constants import TCGEN05_AUX_LOAD_MODES
from .tcgen05_constants import TCGEN05_AUX_STAGE_COUNT_CHOICES
from .tcgen05_constants import TCGEN05_AUX_STAGES_CONFIG_KEY
from .tcgen05_constants import TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY
from .tcgen05_constants import TCGEN05_C_ACQUIRE_PLACEMENTS
from .tcgen05_constants import TCGEN05_C_STORE_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_C_STORE_MODE_NORMAL
from .tcgen05_constants import TCGEN05_C_STORE_MODES
from .tcgen05_constants import TCGEN05_CLUSTER_M2_ONE_CTA_ROLE_LOCAL_CONFIG_KEY
from .tcgen05_constants import TCGEN05_CONSUMER_REGS_CHOICES
from .tcgen05_constants import TCGEN05_CONSUMER_REGS_CONFIG_KEY
from .tcgen05_constants import TCGEN05_CUBIN_LINEINFO_CONFIG_KEY
from .tcgen05_constants import TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY
from .tcgen05_constants import TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY
from .tcgen05_constants import TCGEN05_EPILOGUE_LAYOUTS
from .tcgen05_constants import TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY
from .tcgen05_constants import TCGEN05_LARGE_BN_PROOF_BLOCK_SIZES
from .tcgen05_constants import TCGEN05_LARGE_BN_PROOF_CLUSTER_M
from .tcgen05_constants import TCGEN05_LARGE_BN_PROOF_CONFIG_KEY
from .tcgen05_constants import TCGEN05_LARGE_BN_PROOF_PID_TYPE
from .tcgen05_constants import TCGEN05_LARGE_BN_PROOF_STAGE_CONFIGS
from .tcgen05_constants import TCGEN05_ONE_CTA_MAX_BLOCK_M
from .tcgen05_constants import TCGEN05_RESIDUAL_FULL_TILE_DEEP_C_STAGES
from .tcgen05_constants import TCGEN05_SCHED_CONSUMER_WAIT_MODE_CONFIG_KEY
from .tcgen05_constants import TCGEN05_SCHED_CONSUMER_WAIT_MODE_NORMAL
from .tcgen05_constants import TCGEN05_SCHED_CONSUMER_WAIT_MODES
from .tcgen05_constants import TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY
from .tcgen05_constants import TCGEN05_SCHED_STAGE_COUNTS
from .tcgen05_constants import TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY
from .tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_M
from .tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_N
from .tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K
from .tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_ACC_STAGES
from .tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_K_RANGE_FLATTEN
from .tcgen05_constants import (
    TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_K_RANGE_MULTI_BUFFER,
)
from .tcgen05_constants import (
    TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_K_RANGE_WARP_SPECIALIZE,
)
from .tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_L2_GROUPING
from .tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_ACC_STAGES
from .tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_K
from .tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N
from .tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_L2_GROUPING
from .tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_SCHEDULER_L2_SWIZZLE_SIZE
from .tcgen05_constants import TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_M
from .tcgen05_constants import TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_N
from .tcgen05_constants import TCGEN05_TWO_CTA_MAX_K_TILES
from .tcgen05_constants import TCGEN05_TWO_CTA_SEED_PID_TYPE
from .tcgen05_constants import tcgen05_ab_smem_bytes_per_cta
from .tcgen05_constants import tcgen05_c_smem_bytes_per_cta
from .tcgen05_constants import tcgen05_default_epilogue_tile_size
from .tcgen05_constants import tcgen05_direct_entry_stage_tuple_allowed
from .tcgen05_constants import tcgen05_two_cta_edge_k_tail_seed_overrides

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ...autotuner.block_id_sequence import BlockIdSequence
    from ...autotuner.config_fragment import BlockSizeFragment
    from ...autotuner.config_spec import ConfigSpec
    from ...runtime.config import PidTypeLiteral


class Tcgen05ClusterM2SearchConstraints(NamedTuple):
    """Search-only envelope where ``tcgen05_cluster_m=2`` is validated."""

    static_k: int
    max_k_tiles: int
    allow_edge_k_tail_family: bool = False
    # When True, a sampled bm<=128 cluster_m=2 candidate is projected onto the
    # fp8 small-grid 2-CTA tile (bm=128/bn=128, per-CTA 64xbn) instead of the
    # bm=256 full tile. Gated to fp8 by the caller, mirroring
    # ``_tcgen05_use_2cta_instrs`` (``bm == 128 and is_fp8``). See the
    # ``TCGEN05_TWO_CTA_FP8_SMALL_GRID_*`` constants.
    allow_fp8_small_grid: bool = False


class Tcgen05AbStagesThreeSearchConstraints(NamedTuple):
    """Search-only envelope where ``tcgen05_ab_stages=3`` is admitted.

    The 3-stage AB pipeline is only safe to search when its larger SMEM
    allocation fits after reserving space for CuTe runtime/barrier scratch.
    """

    dtype_bytes: int
    per_cta_smem_budget_bytes: int


CUTE_TCGEN05_TUNABLE_KEYS: tuple[str, ...] = (
    "tcgen05_cluster_m",
    "tcgen05_cluster_n",
    "tcgen05_ab_stages",
    "tcgen05_acc_stages",
    "tcgen05_c_stages",
    TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY,
    TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY,
    TCGEN05_C_STORE_MODE_CONFIG_KEY,
    "tcgen05_num_epi_warps",
    TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY,
)
CUTE_TCGEN05_DIAGNOSTIC_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        TCGEN05_AB_CONSUMER_PHASE_MODE_CONFIG_KEY,
        TCGEN05_AB_CONSUMER_WAIT_MODE_CONFIG_KEY,
        TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODE_CONFIG_KEY,
        TCGEN05_AB_PRODUCER_ACQUIRE_MODE_CONFIG_KEY,
        TCGEN05_AB_PRODUCER_ADVANCE_MODE_CONFIG_KEY,
        TCGEN05_ACC_PRODUCER_ADVANCE_MODE_CONFIG_KEY,
        TCGEN05_ACC_PRODUCER_MODE_CONFIG_KEY,
        TCGEN05_AUX_LOAD_MODE_CONFIG_KEY,
        TCGEN05_AUX_STAGES_CONFIG_KEY,
        TCGEN05_CLUSTER_M2_ONE_CTA_ROLE_LOCAL_CONFIG_KEY,
        TCGEN05_CONSUMER_REGS_CONFIG_KEY,
        TCGEN05_CUBIN_LINEINFO_CONFIG_KEY,
        TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY,
        TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY,
        TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY,
        TCGEN05_LARGE_BN_PROOF_CONFIG_KEY,
        TCGEN05_SCHED_CONSUMER_WAIT_MODE_CONFIG_KEY,
        TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY,
        TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY,
    }
)
CUTE_TCGEN05_STRATEGY_CONFIG_KEYS: frozenset[str] = frozenset(
    TCGEN05_STRATEGY_CONFIG_KEYS
)


class CuteTcgen05Config:
    """CuTe-owned tcgen05 ConfigSpec state and normalization hooks."""

    def __init__(self, config_spec: ConfigSpec) -> None:
        self.config_spec = config_spec
        self.search_enabled: bool = False
        self.aux_kernel_detected: bool = False
        self.exact_shape_aux_kernel_detected: bool = False
        # True when the kernel feeds a matmul an operand sourced from a load
        # whose dtype is not a tcgen05-native MMA dtype (e.g. an int16 tensor
        # cast to bf16, ``w.to(bfloat16)``). Such an operand cannot be TMA-staged
        # for the SMEM tcgen05 MMA, so the dot lowers through the non-tcgen05
        # fallback and the FFI/flat-role direct-entry seed must stay ineligible.
        self.matmul_has_non_tcgen05_operand: bool = False
        self.cluster_m_search_choices: tuple[int, ...] | None = None
        self.cluster_m2_search_constraints: Tcgen05ClusterM2SearchConstraints | None = (
            None
        )
        self.ab_stages_three_search_constraints: (
            Tcgen05AbStagesThreeSearchConstraints | None
        ) = None
        self.num_epi_warps_search_choices: tuple[int, ...] | None = None
        self.num_epi_warps_validation_choices: tuple[int, ...] | None = None

    @property
    def allowed_pid_types(self) -> tuple[PidTypeLiteral, ...]:
        return self.config_spec.allowed_pid_types

    @allowed_pid_types.setter
    def allowed_pid_types(self, value: tuple[PidTypeLiteral, ...]) -> None:
        self.config_spec.allowed_pid_types = value

    @staticmethod
    def _validate_optional_fragment_value(
        name: str, fragment: ConfigSpecFragment, value: object
    ) -> object:
        if isinstance(fragment, BooleanFragment):
            if type(value) is not bool:
                raise InvalidConfig(f"{name} must be a boolean, got {value!r}")
            return value
        if isinstance(fragment, EnumFragment):
            if value not in fragment.choices:
                raise InvalidConfig(
                    f"{name} must be one of {fragment.choices!r}, got {value!r}"
                )
            return value
        if isinstance(fragment, IntegerFragment):
            if type(value) is not int:
                raise InvalidConfig(f"{name} must be an integer, got {value!r}")
            if value < fragment.low or value > fragment.high:
                raise InvalidConfig(
                    f"{name} must be in [{fragment.low}, {fragment.high}], got {value!r}"
                )
            return value
        raise InvalidConfig(f"Unsupported optional fragment type for {name}")

    def restrict_cluster_m_search(self, choices: tuple[int, ...]) -> None:
        assert choices, "tcgen05_cluster_m search must allow at least one value"
        self.cluster_m_search_choices = choices
        if 2 not in choices:
            self.cluster_m2_search_constraints = None

    def allow_cluster_m2_search(
        self,
        *,
        static_k: int,
        max_k_tiles: int = TCGEN05_TWO_CTA_MAX_K_TILES,
        allow_edge_k_tail_family: bool = False,
        allow_fp8_small_grid: bool = False,
    ) -> None:
        assert static_k > 0, "static_k is required for cluster_m=2 K-cap checks"
        assert max_k_tiles > 0, "cluster_m=2 max K tiles must be positive"
        self.cluster_m2_search_constraints = Tcgen05ClusterM2SearchConstraints(
            static_k=static_k,
            max_k_tiles=max_k_tiles,
            allow_edge_k_tail_family=allow_edge_k_tail_family,
            allow_fp8_small_grid=allow_fp8_small_grid,
        )
        self.restrict_cluster_m_search((1, 2))

    @staticmethod
    def cluster_m2_bk_is_valid(
        bk: int, constraints: Tcgen05ClusterM2SearchConstraints
    ) -> bool:
        if bk <= 0:
            return False
        if constraints.allow_edge_k_tail_family:
            return (
                bk
                in (
                    TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K,
                    TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_K,
                )
                and constraints.static_k > bk
                and constraints.static_k % bk != 0
                and (constraints.static_k + bk - 1) // bk <= constraints.max_k_tiles
            )
        if constraints.static_k % bk == 0:
            return constraints.static_k // bk <= constraints.max_k_tiles
        return False

    def full_tile_direct_entry_seed_bk(self) -> int | None:
        """Largest valid full-tile direct-entry K tile for the live shape.

        Mirrors the full-tile branch of the heuristic bk selection: the highest
        power-of-two ``bk`` within the K block-size fragment that divides
        ``static_k`` within the ``max_k_tiles`` cap.
        """
        constraints = self.cluster_m2_search_constraints
        if (
            constraints is None
            or constraints.allow_edge_k_tail_family
            or len(self.config_spec.block_sizes) != 3
        ):
            return None
        bk_fragment = cast(
            "BlockSizeFragment",
            self.config_spec.block_sizes[2]._fragment(self.config_spec),
        )
        bk = bk_fragment.high
        while bk >= bk_fragment.low:
            if self.cluster_m2_bk_is_valid(bk, constraints):
                return bk
            bk //= 2
        return None

    def full_tile_direct_entry_seed_eligible(self) -> bool:
        """Structural eligibility for the generalized TVM-FFI direct-entry seed.

        The direct-entry codegen + runtime validator build their A/B/D TMA
        descriptors from the runtime tensor shapes, so the fast launch path is
        shape-general; the constraints are purely structural: a full-tile (NOT
        edge+K-tail) CtaGroup.TWO 16-bit (bf16/fp16) GEMM, the 256x256 CTA tile
        reachable, a direct-entry-valid ``bk``, and the ``(ab=3, c=2)`` stage
        tuple admitted
        and SMEM-fitting. Both ``optional_fragments`` (to add the FFI search
        surface) and ``CuteTcgen05ClusterM2FfiHeuristic`` (to gate the seed) use
        this so they cannot disagree.
        """
        # A non-tcgen05-native matmul operand (e.g. an int16 tensor cast to
        # bf16) forces the dot through the non-tcgen05 fallback, where the
        # flat-role / FFI seed config is rejected. The matmul facts only record
        # the post-cast dtype (bf16), so gate on the operand-source detector.
        if self.matmul_has_non_tcgen05_operand:
            return False
        constraints = self.cluster_m2_search_constraints
        if constraints is None or constraints.allow_edge_k_tail_family:
            return False
        if TCGEN05_TWO_CTA_SEED_PID_TYPE not in self.allowed_pid_types:
            return False
        if len(self.config_spec.block_sizes) != 3:
            return False
        facts = self.config_spec.matmul_facts
        if len(facts) != 1:
            return False
        fact = facts[0]
        # The direct-entry TMA descriptors, SMEM layout, and epilogue tile are
        # dtype-general for any 16-bit operand (the byte math keys on
        # ``dtype_bytes``, and bf16/fp16 are both 2 bytes), so admit bf16 and
        # fp16 with matching operand dtypes. fp32 stays excluded (no tcgen05
        # fp32 SMEM-staged MMA path).
        if fact.lhs_dtype is not fact.rhs_dtype:
            return False
        if fact.lhs_dtype not in (torch.bfloat16, torch.float16):
            return False
        bm_fragment = cast(
            "BlockSizeFragment",
            self.config_spec.block_sizes[0]._fragment(self.config_spec),
        )
        bn_fragment = cast(
            "BlockSizeFragment",
            self.config_spec.block_sizes[1]._fragment(self.config_spec),
        )
        if not (bm_fragment.low <= TCGEN05_TWO_CTA_BLOCK_M <= bm_fragment.high):
            return False
        if not (bn_fragment.low <= TCGEN05_TWO_CTA_BLOCK_N <= bn_fragment.high):
            return False
        bk = self.full_tile_direct_entry_seed_bk()
        if bk is None:
            return False
        if not tcgen05_direct_entry_stage_tuple_allowed(
            bk=bk, ab_stage_count=3, c_stage_count=2
        ):
            return False
        return self.ab_stages_three_fits(
            bm=TCGEN05_TWO_CTA_BLOCK_M,
            bn=TCGEN05_TWO_CTA_BLOCK_N,
            bk=bk,
            cluster_m=2,
        )

    def full_tile_direct_entry_seed_config(self) -> Config | None:
        """Generalized TVM-FFI direct-entry seed config for the live shape.

        Single source of truth for the FFI ``explicit_epi_tile`` + flat-role +
        ``tvm_ffi_launch`` config: emitted into the autotuner population by
        ``CuteTcgen05ClusterM2FfiHeuristic`` AND used by
        ``_fix_target1_tvm_ffi_search_config`` to project FFI-requesting search
        candidates onto the validated CtaGroup.TWO envelope (this is what the
        per-shape ``_target{N}`` seeds used to do, generalized to any eligible
        shape). Returns ``None`` for ineligible shapes.
        """
        if not self.full_tile_direct_entry_seed_eligible():
            return None
        bk = self.full_tile_direct_entry_seed_bk()
        if bk is None:
            return None
        seed: dict[str, Any] = {
            "block_sizes": [TCGEN05_TWO_CTA_BLOCK_M, TCGEN05_TWO_CTA_BLOCK_N, bk],
            "l2_groupings": [2],
            "num_warps": 8,
            "num_stages": 4,
            "pid_type": TCGEN05_TWO_CTA_SEED_PID_TYPE,
            "tcgen05_cluster_m": 2,
            "tcgen05_cluster_n": 1,
            "tcgen05_ab_stages": 3,
            "tcgen05_acc_stages": 2,
            "tcgen05_c_stages": 2,
            TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY: 1,
            "tcgen05_num_epi_warps": 4,
            TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY: (
                Tcgen05PersistenceModel.STATIC_PERSISTENT.value
            ),
            TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY: (
                Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value
            ),
            # (epi_tile_m, epi_tile_n, d_store_box_n) = (128, 32, 32) is the only
            # explicit-epilogue subtile the D-descriptor codegen accepts, so it
            # is fixed for every eligible shape.
            TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY: 128,
            TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY: 32,
            TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY: 32,
            TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY: True,
            TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY: True,
        }
        if self.config_spec.indexing.length in (3, 4):
            seed["indexing"] = ["tensor_descriptor"] * self.config_spec.indexing.length
        return Config(**seed)

    def _c_input_seed_config(self) -> Config | None:
        if not self.aux_kernel_detected:
            return None
        constraints = self.cluster_m2_search_constraints
        if constraints is None:
            return None
        if TCGEN05_TWO_CTA_SEED_PID_TYPE not in self.allowed_pid_types:
            return None
        if len(self.config_spec.block_sizes) != 3:
            return None

        bm_fragment = cast(
            "BlockSizeFragment",
            self.config_spec.block_sizes[0]._fragment(self.config_spec),
        )
        bn_fragment = cast(
            "BlockSizeFragment",
            self.config_spec.block_sizes[1]._fragment(self.config_spec),
        )
        bk_fragment = cast(
            "BlockSizeFragment",
            self.config_spec.block_sizes[2]._fragment(self.config_spec),
        )
        edge_k_tail_family = constraints.allow_edge_k_tail_family
        m_tile_reachable = (
            bm_fragment.low <= TCGEN05_TWO_CTA_BLOCK_M <= bm_fragment.high
            or (edge_k_tail_family and bm_fragment.low <= TCGEN05_TWO_CTA_BLOCK_M)
        )
        n_tile_reachable = (
            bn_fragment.low <= TCGEN05_TWO_CTA_BLOCK_N <= bn_fragment.high
            or (edge_k_tail_family and bn_fragment.low <= TCGEN05_TWO_CTA_BLOCK_N)
        )
        if not (m_tile_reachable and n_tile_reachable):
            return None

        if edge_k_tail_family:
            bk = TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K
            if not (
                bk_fragment.low <= bk <= bk_fragment.high
                and self.cluster_m2_bk_is_valid(bk, constraints)
            ):
                return None
        else:
            bk = bk_fragment.high
            while bk >= bk_fragment.low:
                if self.cluster_m2_bk_is_valid(bk, constraints):
                    break
                bk //= 2
            else:
                return None

        seed_config: dict[str, Any] = {
            "block_sizes": [
                TCGEN05_TWO_CTA_BLOCK_M,
                TCGEN05_TWO_CTA_BLOCK_N,
                bk,
            ],
            "pid_type": TCGEN05_TWO_CTA_SEED_PID_TYPE,
            "tcgen05_cluster_m": 2,
            "tcgen05_num_epi_warps": 4,
            "tcgen05_ab_stages": 2,
            TCGEN05_STRATEGY_CONFIG_KEY: (
                Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value
            ),
            TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY: (
                Tcgen05PersistenceModel.STATIC_PERSISTENT.value
            ),
            TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY: 1,
            TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY: 1,
        }
        if edge_k_tail_family:
            seed_config.update(tcgen05_two_cta_edge_k_tail_seed_overrides())
            seed_config[TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY] = (
                TCGEN05_TWO_CTA_EDGE_K_TAIL_SCHEDULER_L2_SWIZZLE_SIZE
            )
            seed_config["indexing"] = [
                "tensor_descriptor"
            ] * self.config_spec.indexing.length
        else:
            seed_config["l2_groupings"] = [1]
            if self.config_spec.indexing.length == 3:
                seed_config["indexing"] = [
                    "tensor_descriptor",
                    "tensor_descriptor",
                    "tensor_descriptor",
                ]
        return Config(**seed_config)

    def _aux_tma_edge_search_enabled(self) -> bool:
        # The TMA aux producer's original admission: the validated Target8-style
        # double-edge + K-tail family with ``cluster_m=2``. The CLC-persistent
        # variants and edge-perf knobs remain pinned to this slice.
        constraints = self.cluster_m2_search_constraints
        return (
            self.exact_shape_aux_kernel_detected
            and constraints is not None
            and constraints.allow_edge_k_tail_family
        )

    def _aux_tma_full_tile_search_enabled(self) -> bool:
        # Cycle 46: also admit aux-TMA on full-tile cluster_m=2 problems
        # (T14/T20/T25/T28 residual_add family). The codegen-side gate at
        # ``cute_mma.py`` ``tcgen05_static_output_tiles`` already accepts
        # full-tile shapes; only the search-space gate was excluding them.
        # Edge-perf knobs (``_set_clc_aux_tma_edge_perf_knobs``) and
        # CLC-persistent variants stay pinned to ``_aux_tma_edge_search_enabled``
        # so this widening does not perturb the 5000³ T12 family.
        constraints = self.cluster_m2_search_constraints
        return (
            self.exact_shape_aux_kernel_detected
            and constraints is not None
            and not constraints.allow_edge_k_tail_family
        )

    def _aux_tma_search_enabled(self) -> bool:
        # The TMA aux producer is admitted on either the edge+K-tail family or
        # the full-tile cluster_m=2 family. Exact-shape aux tensors use the
        # aux-TMA producer on both full and partial-output tiles; non-staged
        # aux operands remain on the direct guarded load path.
        return (
            self._aux_tma_edge_search_enabled()
            or self._aux_tma_full_tile_search_enabled()
        )

    def _aux_tma_seed_config(self, c_input_seed: Config) -> Config | None:
        if not self._aux_tma_search_enabled():
            return None
        seed_config: dict[str, Any] = dict(c_input_seed.config)
        seed_config[TCGEN05_AUX_LOAD_MODE_CONFIG_KEY] = TCGEN05_AUX_LOAD_MODE_TMA
        return Config(**seed_config)

    def _clc_persistence_seed_config(self, base_seed: Config) -> Config | None:
        if not self._clc_persistence_search_enabled():
            return None
        seed_config: dict[str, Any] = dict(base_seed.config)
        seed_config[TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY] = (
            Tcgen05PersistenceModel.CLC_PERSISTENT.value
        )
        return Config(**seed_config)

    def _set_clc_aux_tma_edge_perf_knobs(self, config: dict[str, object]) -> None:
        config["tcgen05_acc_stages"] = (
            TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_ACC_STAGES
        )
        config["l2_groupings"] = [TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_L2_GROUPING]
        range_knobs = self._clc_aux_tma_edge_range_knobs()
        if range_knobs is not None:
            (
                config["range_flattens"],
                config["range_multi_buffers"],
                config["range_warp_specializes"],
            ) = range_knobs

    def _clc_aux_tma_wide_n_seed_config(self, clc_aux_tma_seed: Config) -> Config:
        seed_config: dict[str, Any] = dict(clc_aux_tma_seed.config)
        self._set_clc_aux_tma_edge_perf_knobs(seed_config)
        return Config(**seed_config)

    def _clc_aux_tma_edge_range_knobs(
        self,
    ) -> tuple[list[bool | None], list[bool | None], list[bool | None]] | None:
        k_range_index = self._clc_aux_tma_matmul_k_range_index()
        if k_range_index is None:
            return None
        range_flattens: list[bool | None] = [
            None for _ in self.config_spec.range_flattens
        ]
        range_multi_buffers: list[bool | None] = [
            None for _ in self.config_spec.range_multi_buffers
        ]
        range_warp_specializes: list[bool | None] = [
            None for _ in self.config_spec.range_warp_specialize
        ]
        range_flattens[k_range_index] = (
            TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_K_RANGE_FLATTEN
        )
        range_multi_buffers[k_range_index] = (
            TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_K_RANGE_MULTI_BUFFER
        )
        range_warp_specializes[k_range_index] = (
            TCGEN05_TWO_CTA_EDGE_K_TAIL_CLC_AUX_TMA_K_RANGE_WARP_SPECIALIZE
        )
        return range_flattens, range_multi_buffers, range_warp_specializes

    def _clc_aux_tma_matmul_k_range_index(self) -> int | None:
        k_range_indices: set[int] = set()
        range_flattens_ids = self.config_spec.range_flattens.valid_block_ids()
        range_multi_buffers_ids = self.config_spec.range_multi_buffers.valid_block_ids()
        range_warp_specialize_ids = (
            self.config_spec.range_warp_specialize.valid_block_ids()
        )
        for fact in self.config_spec.matmul_facts:
            k_block_id = fact.k_block_id
            if k_block_id is None:
                continue
            in_range_maps = (
                k_block_id in range_flattens_ids,
                k_block_id in range_multi_buffers_ids,
                k_block_id in range_warp_specialize_ids,
            )
            if not any(in_range_maps):
                continue
            if not all(in_range_maps):
                return None
            range_index = self.config_spec.range_flattens.block_id_to_index(k_block_id)
            if range_index != self.config_spec.range_multi_buffers.block_id_to_index(
                k_block_id
            ):
                return None
            if range_index != self.config_spec.range_warp_specialize.block_id_to_index(
                k_block_id
            ):
                return None
            k_range_indices.add(range_index)
        if len(k_range_indices) != 1:
            return None
        return next(iter(k_range_indices))

    def _clc_aux_tma_narrow_n_seed_config(
        self, clc_aux_tma_seed: Config
    ) -> Config | None:
        if not self._has_any_matmul_fact_n_edge_for_block_n(
            TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N
        ):
            return None
        bn_fragment = cast(
            "BlockSizeFragment",
            self.config_spec.block_sizes[1]._fragment(self.config_spec),
        )
        if not (
            bn_fragment.low
            <= TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N
            <= bn_fragment.high
        ):
            return None
        constraints = self.cluster_m2_search_constraints
        if constraints is None or not self.cluster_m2_bk_is_valid(
            TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_K,
            constraints,
        ):
            return None
        seed_config: dict[str, Any] = dict(clc_aux_tma_seed.config)
        seed_config["block_sizes"] = [
            TCGEN05_TWO_CTA_BLOCK_M,
            TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N,
            TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_K,
        ]
        seed_config["tcgen05_acc_stages"] = (
            TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_ACC_STAGES
        )
        seed_config["l2_groupings"] = [TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_L2_GROUPING]
        return Config(**seed_config)

    def autotune_seed_configs(self) -> list[Config]:
        seeds: list[Config] = []
        c_input_seed = self._c_input_seed_config()
        if c_input_seed is not None:
            seeds.append(c_input_seed)
            clc_c_input_seed = self._clc_persistence_seed_config(c_input_seed)
            if clc_c_input_seed is not None:
                seeds.append(clc_c_input_seed)
            aux_tma_seed = self._aux_tma_seed_config(c_input_seed)
            if aux_tma_seed is not None:
                seeds.append(aux_tma_seed)
                clc_aux_tma_seed = self._clc_persistence_seed_config(aux_tma_seed)
                if clc_aux_tma_seed is not None:
                    clc_aux_tma_seed = self._clc_aux_tma_wide_n_seed_config(
                        clc_aux_tma_seed
                    )
                    seeds.append(clc_aux_tma_seed)
                    clc_aux_tma_narrow_n_seed = self._clc_aux_tma_narrow_n_seed_config(
                        clc_aux_tma_seed
                    )
                    if clc_aux_tma_narrow_n_seed is not None:
                        seeds.append(clc_aux_tma_narrow_n_seed)
        return seeds

    def _fix_cluster_m2_search_config(self, config: dict[str, object]) -> None:
        if not (self.search_enabled and config.get("tcgen05_cluster_m") == 2):
            return
        constraints = self.cluster_m2_search_constraints
        if constraints is None:
            config["tcgen05_cluster_m"] = 1
            return
        if TCGEN05_TWO_CTA_SEED_PID_TYPE not in self.allowed_pid_types:
            config["tcgen05_cluster_m"] = 1
            return
        block_sizes = config.get("block_sizes")
        if not isinstance(block_sizes, list) or len(block_sizes) < 3:
            config["tcgen05_cluster_m"] = 1
            return
        edge_k_tail_family = constraints.allow_edge_k_tail_family
        is_narrow_clc_aux_tma = self._is_clc_aux_tma_narrow_n_request(config)
        if edge_k_tail_family:
            block_sizes[2] = (
                TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_K
                if is_narrow_clc_aux_tma
                else TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K
            )
        bk = block_sizes[2]
        if not isinstance(bk, int) or isinstance(bk, bool):
            config["tcgen05_cluster_m"] = 1
            return
        if not self.cluster_m2_bk_is_valid(bk, constraints):
            config["tcgen05_cluster_m"] = 1
            return
        config["pid_type"] = TCGEN05_TWO_CTA_SEED_PID_TYPE
        # The tcgen05 CtaGroup.TWO MMA path does not emit the per-block-id
        # indices/masks that a fused epilogue subtile needs, so a sampled
        # ``epilogue_subtile`` on a cluster_m=2 candidate raises
        # ``BackendUnsupported`` at codegen. Drop it here (rather than letting
        # the candidate fail to compile and waste autotune budget) -- every
        # cluster_m=2 search candidate that survives to this point is committed
        # to the 2-CTA path. The edge-family prefixes below also pop it for
        # their sub-paths; doing it once here covers the full-tile and
        # small-grid paths too.
        config.pop("epilogue_subtile", None)
        # fp8 small-grid family: a sampled bm<=128 routes to the fp8-validated
        # per-CTA 64xbn 2-CTA tile (bm=128/bn=128) instead of the bm=256 full
        # tile, which underfills the device on small/wave-limited fp8 GEMMs. The
        # bm=256 full tile is still reachable from a sampled bm>128. The codegen
        # + runtime own this tile via ``_tcgen05_use_2cta_instrs``
        # (``bm == 128 and is_fp8``). Edge+K-tail candidates keep the bm=256
        # double-edge family unconditionally.
        if (
            constraints.allow_fp8_small_grid
            and not edge_k_tail_family
            and isinstance(block_sizes[0], int)
            and not isinstance(block_sizes[0], bool)
            and block_sizes[0] <= TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_M
        ):
            block_sizes[0] = TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_M
            block_sizes[1] = TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_N
            return
        block_sizes[0] = TCGEN05_TWO_CTA_BLOCK_M
        # Only the fully validated narrow-N CLC+aux-TMA seed may keep
        # block_n=128; other candidates use the canonical block_n=256.
        if is_narrow_clc_aux_tma:
            block_sizes[1] = TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N
        else:
            block_sizes[1] = TCGEN05_TWO_CTA_BLOCK_N
        if edge_k_tail_family:
            # This family is pinned to measured production stage/pipeline
            # values after search projection.
            # Placement keys remain available for non-edge diagnostic/search
            # paths, but edge+K-tail candidates do not explore partially
            # mutated placement variants.
            config.update(tcgen05_two_cta_edge_k_tail_seed_overrides())
            if self.aux_kernel_detected and self._has_any_matmul_fact_edge_tile(config):
                self._set_aux_edge_cluster_m2_prefix(config)
            if self._is_clc_aux_tma_config(config):
                self._set_clc_aux_tma_edge_perf_knobs(config)
            if is_narrow_clc_aux_tma:
                config["tcgen05_acc_stages"] = (
                    TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_ACC_STAGES
                )
                config["l2_groupings"] = [
                    TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_L2_GROUPING
                ]

    def allow_ab_stages_three_search(
        self,
        *,
        dtype_bytes: int,
        device: torch.device,
    ) -> None:
        assert dtype_bytes > 0, "dtype_bytes must be positive"
        if len(self.config_spec.block_sizes) != 3:
            self.ab_stages_three_search_constraints = None
            return
        budget_bytes = self.per_cta_ab_smem_budget_bytes(device)
        if budget_bytes <= 0:
            self.ab_stages_three_search_constraints = None
            return
        self.ab_stages_three_search_constraints = Tcgen05AbStagesThreeSearchConstraints(
            dtype_bytes=dtype_bytes,
            per_cta_smem_budget_bytes=budget_bytes,
        )

    @staticmethod
    def per_cta_ab_smem_budget_bytes(device: torch.device) -> int:
        if device.type != "cuda" or not torch.cuda.is_available():
            return 0
        props = torch.cuda.get_device_properties(device)
        optin_shared = int(getattr(props, "shared_memory_per_block_optin", 0) or 0)
        device_cap = max(props.shared_memory_per_block, optin_shared)
        if device_cap < TCGEN05_AB_STAGES_THREE_MIN_DEVICE_SMEM_OPTIN:
            return 0
        # Keep a fixed headroom reservation: CuTe's raw opt-in limit does not
        # include every barrier/runtime byte the 3-stage AB pipeline needs.
        return device_cap - TCGEN05_AB_STAGES_THREE_RESERVED_SMEM_BYTES

    def ab_stages_three_fits(
        self,
        *,
        bm: int,
        bn: int,
        bk: int,
        cluster_m: int,
        ab_stages: int = 3,
    ) -> bool:
        constraints = self.ab_stages_three_search_constraints
        if constraints is None:
            return False
        if cluster_m not in (1, 2):
            return False
        if bm <= 0 or bn <= 0 or bk <= 0:
            return False
        bytes_per_cta = tcgen05_ab_smem_bytes_per_cta(
            bm=bm,
            bn=bn,
            bk=bk,
            dtype_bytes=constraints.dtype_bytes,
            ab_stages=ab_stages,
            cluster_m=cluster_m,
        )
        return bytes_per_cta <= constraints.per_cta_smem_budget_bytes

    def c_stages_fits(
        self,
        *,
        bm: int,
        bn: int,
        bk: int,
        cluster_m: int,
        ab_stages: int,
        c_stages: int,
        has_source_c: bool,
    ) -> bool:
        # Workstream A Stage 2 (cycle 90): budget-aware admission for the deeper
        # C-store ring. Reuse the ``tcgen05_ab_stages=3`` SMEM-budget envelope
        # (same dtype_bytes + per-CTA budget after the non-AB reservation) and
        # require AB + C to fit together. This is the gate that keeps a deeper
        # C ring (``tcgen05_c_stages=4``) out of the ab=3 regime, where AB+C
        # overshoots the 232 KB B200 cap and ptxas raises a raw
        # ``too much shared`` error during tuning. The C bytes use the REAL
        # ``Tcgen05LayoutStrategy.DEFAULT`` epilogue subtile, not the (128, 32)
        # EXPLICIT_EPI_TILE direct-entry tile, so the byte count matches the
        # role-local codegen: a 256x256 16-bit tile is ``(128, 64)`` WITH a
        # source-C (residual family) but ``(128, 32)`` WITHOUT one (plain
        # matmul) -- ``compute_epilogue_tile_size`` shrinks N when no C tile
        # competes for SMEM. ``has_source_c`` threads that distinction through.
        constraints = self.ab_stages_three_search_constraints
        if constraints is None:
            return False
        if cluster_m not in (1, 2):
            return False
        if bm <= 0 or bn <= 0 or bk <= 0:
            return False
        if ab_stages <= 0 or c_stages <= 0:
            return False
        ab_bytes = tcgen05_ab_smem_bytes_per_cta(
            bm=bm,
            bn=bn,
            bk=bk,
            dtype_bytes=constraints.dtype_bytes,
            ab_stages=ab_stages,
            cluster_m=cluster_m,
        )
        # The epilogue processes the full per-CTA output tile (bm, bn); unlike
        # the AB operands it is NOT split across the cluster, so the C-ring
        # bytes do not depend on cluster_m. ``elem_width`` is the operand /
        # output element width in bits (the validated families are uniform
        # 16-bit). ``elem_width_c`` is None for no-source-C (plain) kernels so
        # the helper picks the smaller no-source-C epilogue tile.
        elem_width = constraints.dtype_bytes * 8
        epi_tile_m, epi_tile_n = tcgen05_default_epilogue_tile_size(
            bm,
            bn,
            elem_width_d=elem_width,
            elem_width_c=elem_width if has_source_c else None,
        )
        c_bytes = tcgen05_c_smem_bytes_per_cta(
            epi_tile_m=epi_tile_m,
            epi_tile_n=epi_tile_n,
            dtype_bytes=constraints.dtype_bytes,
            c_stages=c_stages,
        )
        return ab_bytes + c_bytes <= constraints.per_cta_smem_budget_bytes

    def _fix_c_stages_search_config(self, config: dict[str, object]) -> None:
        # Workstream A Stage 2 (cycle 90): true admission gate for the deeper C
        # ring. ``tcgen05_c_stages`` is an ``EnumFragment((2, 4))`` knob, so the
        # autotuner can SAMPLE c=4 independently of any projection — a directly
        # sampled 256x256 cluster_m=2 ab=3 + c=4 reaches ptxas and fails with a
        # raw ``too much shared`` error (verified cycle-90). Mirror
        # ``_fix_ab_stages_three_search_config``: when a config carries c=4 from
        # ANY source and ``c_stages_fits`` is False, demote it to 2.
        #
        # Scope: judge ONLY the canonical full-tile 256x256 DEFAULT-layout path,
        # which is exactly where the AB+C arithmetic is calibrated (the validated
        # CtaGroup.TWO cosize shapes) and where the role-local C ring lives. The
        # narrow-N / bm=128 edge family (``_fix_aux_edge_search_config`` sets its
        # own validated c=4 at a different cosize that the analytic model would
        # mis-judge) and the EXPLICIT_EPI_TILE direct-entry seeds (separate
        # (128, 32) tile + own admission) keep their c=4 untouched.
        if not self.search_enabled:
            return
        if config.get("tcgen05_c_stages") != TCGEN05_RESIDUAL_FULL_TILE_DEEP_C_STAGES:
            return
        if not self._is_default_layout_full_tile_config(config):
            return
        # Fail CLOSED: the ``(2, 4)`` c-stages fragment is offered on every
        # device, but with no SMEM budget recorded (non-B200 / CPU host) we
        # cannot prove c=4 fits — demote rather than leave the ptxas-overflow
        # window open. ``c_stages_fits`` itself returns False when constraints
        # are absent, so a single ``not c_stages_fits`` check covers both the
        # over-budget and the no-budget arms.
        block_sizes = cast("list[int]", config["block_sizes"])
        cluster_m = cast("int", config.get("tcgen05_cluster_m", 1))
        ab_stages = cast("int", config.get("tcgen05_ab_stages", 2))
        if not self.c_stages_fits(
            bm=block_sizes[0],
            bn=block_sizes[1],
            bk=block_sizes[2],
            cluster_m=cluster_m,
            ab_stages=ab_stages,
            c_stages=TCGEN05_RESIDUAL_FULL_TILE_DEEP_C_STAGES,
            has_source_c=self.aux_kernel_detected,
        ):
            config["tcgen05_c_stages"] = 2

    @staticmethod
    def _is_default_layout_full_tile_config(config: dict[str, object]) -> bool:
        # The canonical 256x256 DEFAULT-layout role-local tile, where the C-ring
        # AB+C SMEM model is calibrated. EXPLICIT_EPI_TILE configs use a separate
        # tile/admission and are excluded; an absent layout key defaults to
        # DEFAULT.
        layout = config.get(
            TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY,
            Tcgen05LayoutStrategy.DEFAULT.value,
        )
        if layout != Tcgen05LayoutStrategy.DEFAULT.value:
            return False
        block_sizes = config.get("block_sizes")
        if not isinstance(block_sizes, list) or len(block_sizes) < 2:
            return False
        return (
            block_sizes[0] == TCGEN05_TWO_CTA_BLOCK_M
            and block_sizes[1] == TCGEN05_TWO_CTA_BLOCK_N
        )

    @staticmethod
    def _get_dtype_ab_stages_hard_cap(dtype_bytes: int) -> int:
        """Get hardware-validated maximum ab_stages for a dtype.

        The maximum practical ab_stages depends on dtype size because
        smaller dtypes fit more pipeline stages in the same SMEM budget:
        - FP8 (1 byte): 12 stages - validated on B200, fits 2x bf16
        - FP16/BF16 (2 bytes): 6 stages - TVM-FFI seed validated
        - FP32 (4 bytes): 3 stages - baseline for larger dtypes

        Args:
            dtype_bytes: Size of data type in bytes

        Returns:
            Maximum ab_stages for this dtype, or 0 if invalid
        """
        if dtype_bytes <= 0:
            return 0
        if dtype_bytes == 1:  # FP8
            return 12
        if dtype_bytes == 2:  # FP16/BF16
            return 6
        # FP32 or larger
        return 3

    def max_ab_stages_that_fit(
        self,
        *,
        bm: int,
        bn: int,
        bk: int,
        cluster_m: int,
        hard_cap: int | None = None,
    ) -> int:
        """Compute maximum ab_stages that fits in per-CTA SMEM budget.

        Mirrors CUTLASS's ``_compute_stages``: fill SMEM with as many AB
        pipeline stages as fit the hardware budget. Uses direct calculation
        since SMEM usage scales linearly with ab_stages.

        For FP8 (1-byte operands), this enables ~2x deeper staging than
        BF16 (2-byte), which is critical for hiding K-loop TMA latency
        in compute-bound kernels.

        Args:
            bm: Block size in M dimension
            bn: Block size in N dimension
            bk: Block size in K dimension
            cluster_m: CTA cluster size (1 or 2)
            hard_cap: Optional maximum stages override. If None, uses
                dtype-specific default (12 for FP8, 6 for FP16, 3 for FP32)

        Returns:
            Maximum valid ab_stages in [1, hard_cap], or 0 if constraints
            are unknown or configuration is invalid (e.g., ab_stages=1
            doesn't fit budget).

        Example:
            >>> # FP8 256x256x64 cluster_m=2
            >>> config.max_ab_stages_that_fit(bm=256, bn=256, bk=64, cluster_m=2)
            8  # FP8 fits 8 stages

            >>> # BF16 same tile (2x larger per stage)
            >>> config.max_ab_stages_that_fit(bm=256, bn=256, bk=64, cluster_m=2)
            4  # BF16 fits only 4 stages
        """
        constraints = self.ab_stages_three_search_constraints
        if constraints is None or bm <= 0 or bn <= 0 or bk <= 0:
            return 0
        if cluster_m not in (1, 2):
            return 0

        # Calculate SMEM cost for ab_stages=1 (baseline)
        bytes_per_stage = tcgen05_ab_smem_bytes_per_cta(
            bm=bm,
            bn=bn,
            bk=bk,
            dtype_bytes=constraints.dtype_bytes,
            ab_stages=1,
            cluster_m=cluster_m,
        )

        # Edge cases: invalid calculation or even ab_stages=1 doesn't fit
        if bytes_per_stage <= 0:
            return 0
        if bytes_per_stage > constraints.per_cta_smem_budget_bytes:
            return 0

        # Direct calculation: SMEM usage scales linearly with ab_stages
        # Solve: N * bytes_per_stage <= budget
        max_from_budget = constraints.per_cta_smem_budget_bytes // bytes_per_stage

        # Apply hard cap (dtype-specific default if not provided)
        if hard_cap is None:
            hard_cap = self._get_dtype_ab_stages_hard_cap(constraints.dtype_bytes)

        # Return clamped value: at least 1, at most hard_cap or budget limit
        return max(1, min(max_from_budget, hard_cap))

    def _fix_ab_stages_three_search_config(self, config: dict[str, object]) -> None:
        if self.ab_stages_three_search_constraints is None:
            return
        if not self.search_enabled:
            return
        if config.get("tcgen05_ab_stages") != 3:
            return
        block_sizes = cast("list[int]", config["block_sizes"])
        cluster_m = cast("int", config.get("tcgen05_cluster_m", 1))
        if not self.ab_stages_three_fits(
            bm=block_sizes[0],
            bn=block_sizes[1],
            bk=block_sizes[2],
            cluster_m=cluster_m,
        ):
            config["tcgen05_ab_stages"] = 2

    def _fix_ab_stages_search_config(self, config: dict[str, object]) -> None:
        # Budget-aware ab=3 admission for the lifted ``for_search`` cap (see
        # ``optional_fragments`` and cute_plan.md §4.5 for the empirical narrative).
        # Mirror ``_fix_c_stages_search_config`` (fail-CLOSED, cast-based): on the
        # canonical 256x256 DEFAULT-layout role-local path, demote a directly-sampled
        # ab=3 to 2 when it does not fit the per-CTA SMEM budget. The invariant — the
        # new dimension over the bare-AB ``_fix_ab_stages_three_search_config`` gate —
        # is REAL source-C presence, keyed on the PRECISE
        # ``exact_shape_aux_kernel_detected`` (rank-2 exact-shape residual_add), NOT
        # the broad ``aux_kernel_detected`` (also True for a rowvec broadcast bias
        # that has no source-C ring): a real source-C materializes the larger
        # (128, 64) C ring, so AB(ab=3) + C overflows the cap even at c=2 and MUST
        # demote, while the plain / rowvec-bias family (no source-C ring) keeps the
        # bare-AB calibration so its ab=3 winner stays searchable. The exact-shape
        # residual cluster_m=2 full-tile candidates are already forced to ab=2 by
        # ``_fix_aux_tma_full_tile_search_config`` (runs first); this gate's source-C
        # branch is the fail-closed backstop for any exact-shape residual ab=3 that
        # projection does not claim (e.g. cluster_m=1). The EXPLICIT_EPI_TILE
        # direct-entry (TVM-FFI) seeds use a separate (128, 32) tile + own admission
        # and are out of the DEFAULT-layout scope, so their seeded ab=3/ab=6 is
        # untouched.
        if not self.search_enabled:
            return
        if config.get("tcgen05_ab_stages") != 3:
            return
        if not self._is_default_layout_full_tile_config(config):
            return
        block_sizes = cast("list[int]", config["block_sizes"])
        cluster_m = cast("int", config.get("tcgen05_cluster_m", 1))
        if self.exact_shape_aux_kernel_detected:
            # Real source-C present: require AB(ab=3) + the (128, 64) C ring to fit
            # together. ``c_stages_fits`` fails CLOSED when no SMEM budget is
            # recorded (non-B200 / CPU host), so the over-budget and no-budget
            # arms are both covered by a single ``not c_stages_fits`` check.
            c_stages = cast("int", config.get("tcgen05_c_stages", 2))
            fits = self.c_stages_fits(
                bm=block_sizes[0],
                bn=block_sizes[1],
                bk=block_sizes[2],
                cluster_m=cluster_m,
                ab_stages=3,
                c_stages=c_stages,
                has_source_c=True,
            )
        else:
            # Plain / rowvec-bias store (no source-C ring): the bare-AB gate is the
            # calibrated admission — the small no-source-C epilogue D ring rides the
            # non-AB reservation. ``ab_stages_three_fits`` returns False with no
            # budget recorded, so this also fails CLOSED.
            fits = self.ab_stages_three_fits(
                bm=block_sizes[0],
                bn=block_sizes[1],
                bk=block_sizes[2],
                cluster_m=cluster_m,
            )
        if not fits:
            config["tcgen05_ab_stages"] = 2

    def _fix_with_scheduler_search_config(self, config: dict[str, object]) -> None:
        if not (self.search_enabled and self.aux_kernel_detected):
            return
        strategy = config.get(TCGEN05_STRATEGY_CONFIG_KEY)
        scheduler_warps = config.get(TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY)
        c_input_warps = config.get(TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY)
        ab_stages = config.get("tcgen05_ab_stages")
        if strategy in (
            Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value,
            Tcgen05Strategy.PURE_MATMUL_ROLE_LIFECYCLE.value,
        ):
            if scheduler_warps != 0:
                config[TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY] = 0
            if c_input_warps != 0:
                config[TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY] = 0
        elif strategy == Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value:
            if scheduler_warps != 1:
                config[TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY] = 1
        if ab_stages == 3 and config.get(TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY) == 1:
            config[TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY] = 0

    def _fix_aux_tma_search_config(self, config: dict[str, object]) -> None:
        if config.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY) != TCGEN05_AUX_LOAD_MODE_TMA:
            return
        if not self._aux_tma_search_enabled():
            config[TCGEN05_AUX_LOAD_MODE_CONFIG_KEY] = TCGEN05_AUX_LOAD_MODE_SIMT
            return
        # Aux-TMA is kept when the projected cluster_m=2 candidate matches either
        # the validated edge+K-tail shape or the cycle-46 full-tile shape, and
        # the strategy is the ROLE_LOCAL_WITH_SCHEDULER + c-input warp combo that
        # the aux-TMA producer warp requires.
        if not self._is_with_scheduler_c_input_config(config):
            config[TCGEN05_AUX_LOAD_MODE_CONFIG_KEY] = TCGEN05_AUX_LOAD_MODE_SIMT
            return
        if not (
            self._is_validated_cluster_m2_edge_search_candidate(config)
            or self._is_validated_cluster_m2_full_tile_search_candidate(config)
        ):
            config[TCGEN05_AUX_LOAD_MODE_CONFIG_KEY] = TCGEN05_AUX_LOAD_MODE_SIMT

    def _fix_aux_tma_full_tile_search_config(self, config: dict[str, object]) -> None:
        # Cycle 88 (Workstream B): on the residual full-tile cluster_m=2
        # family (T20/T14/T25/T28 — exact-shape-gated by
        # ``_aux_tma_full_tile_search_enabled``), project every cluster_m=2
        # candidate onto the validated aux-TMA producer regime
        # (role_local_with_scheduler + scheduler/c-input warp + ab=2 +
        # aux_load_mode=tma). Clean force-config measurements show aux-TMA
        # strictly beats the same-config SIMT control on this family (T20
        # 962.6 vs 844.0 TF = +14%; T25 730.8 vs 324.7 TF = +125%) and also
        # beats the autotuner's default monolithic-ab=3 SIMT pick (T20 ~915
        # TF). Without this projection the population search either keeps the
        # monolithic-ab=3 SIMT region (T20 reliably) or randomly lands on a
        # catastrophic SIMT cluster_m=2 pick (T25 variance: 756 aux-TMA vs 286
        # SIMT), so the +2.4-14 pp aux-TMA gain is not banked deterministically.
        # This is the aux-TMA analogue of the FFI direct-entry search
        # projection (``_fix_target1_tvm_ffi_search_config``). Tightly bounded:
        # cluster_m=1 candidates are left untouched (search still explores
        # them), the edge+K-tail T12 family keeps its own
        # ``_aux_tma_edge_search_enabled`` CLC path, and the gate fires only
        # when ``exact_shape_aux_kernel_detected`` is True (residual subset
        # only — no non-residual target is perturbed).
        if not self.search_enabled:
            return
        if not self._aux_tma_full_tile_search_enabled():
            return
        if config.get("tcgen05_cluster_m") != 2:
            return
        if not self._is_validated_cluster_m2_full_tile_search_candidate(config):
            return
        # ab=2 is the aux-TMA producer's validated stage depth for this family
        # (the aux SMEM ring forces ab<=2 under the 232 KB B200 cap; the
        # cycle-86/88 measurements ran at ab=2). ``_c_input_seed_config``
        # already emits exactly this regime as a seed, so the shape envelope
        # validated above is sufficient — no extra SMEM-fit check is needed.
        config[TCGEN05_STRATEGY_CONFIG_KEY] = (
            Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value
        )
        config[TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY] = 1
        config[TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY] = 1
        config["tcgen05_ab_stages"] = 2
        config[TCGEN05_AUX_LOAD_MODE_CONFIG_KEY] = TCGEN05_AUX_LOAD_MODE_TMA
        # Workstream A Stage 2 (cycle 90): give this family the deeper C-store
        # ring (foundation for the Stage-4 store-warp split — the store warp
        # drains tile N's TMA-D from the ring while the 4 epi warps run tile
        # N+1's T2R; a 2-stage ring leaves no slack). At ab=2 the c=4 ring fits
        # (128 KB AB + 64 KB C = 192 KB < 232 KB cap; the DEFAULT epi tile is
        # (128, 64), so each C stage is 16 KB), confirmed correct on T20
        # (cycle-90 force-config: c=2 942.2 TF vs c=4 936.1 TF — perf-neutral
        # standalone, exactly the cycle-89 NCU prediction since the TMA-D store
        # is already c_pipeline-overlapped). Gated by ``c_stages_fits`` so a
        # future ab=3 candidate in this family cannot be lifted into overflow.
        block_sizes = cast("list[int]", config["block_sizes"])
        if self.c_stages_fits(
            bm=block_sizes[0],
            bn=block_sizes[1],
            bk=block_sizes[2],
            cluster_m=2,
            ab_stages=2,
            c_stages=TCGEN05_RESIDUAL_FULL_TILE_DEEP_C_STAGES,
            has_source_c=True,
        ):
            config["tcgen05_c_stages"] = TCGEN05_RESIDUAL_FULL_TILE_DEEP_C_STAGES

    @staticmethod
    def _is_with_scheduler_c_input_config(config: dict[str, object]) -> bool:
        return (
            config.get(TCGEN05_STRATEGY_CONFIG_KEY)
            == Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value
            and config.get(TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY) == 1
            and config.get(TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY) == 1
        )

    def _clc_persistence_search_enabled(self) -> bool:
        """CLC search is the sm100+ slice of the aux-TMA edge+K-tail gate.

        Cycle 46 widened ``_aux_tma_search_enabled`` to also admit the full-tile
        cluster_m=2 family, but the CLC-persistent perf knobs and validated
        candidate shape are still scoped to ``_aux_tma_edge_search_enabled``.
        """
        if not self._aux_tma_edge_search_enabled():
            return False
        capability = self.config_spec.target_device_capability
        if capability is None:
            return False
        return capability[0] >= 10 and "flat" in self.allowed_pid_types

    def _is_clc_aux_tma_request(self, config: dict[str, object]) -> bool:
        return (
            config.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY) == TCGEN05_AUX_LOAD_MODE_TMA
            and config.get(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY)
            == Tcgen05PersistenceModel.CLC_PERSISTENT.value
            and self._clc_persistence_search_enabled()
        )

    def _is_clc_aux_tma_config(self, config: dict[str, object]) -> bool:
        return self._is_clc_aux_tma_request(
            config
        ) and self._is_validated_clc_persistence_search_candidate(config)

    def _is_clc_aux_tma_narrow_n_request(self, config: dict[str, object]) -> bool:
        block_sizes = config.get("block_sizes")
        if not (
            isinstance(block_sizes, list)
            and len(block_sizes) >= 3
            and block_sizes[1] == TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N
            and self._is_clc_aux_tma_request(config)
        ):
            return False
        projected_config = dict(config)
        projected_block_sizes = list(block_sizes)
        projected_config["block_sizes"] = projected_block_sizes
        projected_config["pid_type"] = TCGEN05_TWO_CTA_SEED_PID_TYPE
        projected_block_sizes[0] = TCGEN05_TWO_CTA_BLOCK_M
        projected_block_sizes[2] = TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_K
        if (
            self.aux_kernel_detected
            and self._has_any_matmul_fact_edge_tile(projected_config)
            and projected_config.get(TCGEN05_STRATEGY_CONFIG_KEY)
            == Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value
        ):
            projected_config[TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY] = 1
            projected_config[TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY] = 1
        return self._is_validated_clc_persistence_search_candidate(projected_config)

    def implicit_default_keys_to_preserve(self, config: dict[str, object]) -> set[str]:
        if not self._is_clc_aux_tma_config(config):
            return set()
        preserve_keys = {"l2_groupings"}
        if self._clc_aux_tma_matmul_k_range_index() is not None:
            preserve_keys.update(
                {
                    "range_flattens",
                    "range_multi_buffers",
                    "range_warp_specializes",
                }
            )
        return preserve_keys

    def _validate_target1_ab_stage_envelope(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        ab_stages = config.get("tcgen05_ab_stages")
        if type(ab_stages) is not int or ab_stages <= 3:
            return
        # ab>3 is only valid on the TVM-FFI direct-entry path, and only for the
        # (bk, ab, c) stage tuples the direct-entry codegen accepts (bk=64
        # admits (ab=6, c=4)). Everything else clamps (or rejects) to ab=3.
        block_sizes = config.get("block_sizes")
        bk = (
            block_sizes[2]
            if isinstance(block_sizes, list) and len(block_sizes) >= 3
            else None
        )
        c_stages = config.get("tcgen05_c_stages")
        if (
            config.get(TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY) is True
            and isinstance(bk, int)
            and not isinstance(bk, bool)
            and type(c_stages) is int
            and tcgen05_direct_entry_stage_tuple_allowed(
                bk=bk, ab_stage_count=ab_stages, c_stage_count=c_stages
            )
        ):
            return
        # FP8 (1-byte) operands fit a deeper AB pipeline than the bf16-tuned
        # cap of 3; admit ab_stages > 3 for fp8 as long as the AB SMEM fits the
        # per-CTA budget. This lets Helion emit the same deeply-pipelined
        # CtaGroup.TWO kernel CUTLASS uses for fp8 compute-bound GEMMs.
        constraints = self.ab_stages_three_search_constraints
        if constraints is not None and constraints.dtype_bytes == 1:  # FP8
            block_sizes = cast("list[int]", config.get("block_sizes"))
            cluster_m = cast("int", config.get("tcgen05_cluster_m", 1))
            if isinstance(block_sizes, list) and len(block_sizes) >= 3:
                fit_max = self.max_ab_stages_that_fit(
                    bm=block_sizes[0],
                    bn=block_sizes[1],
                    bk=block_sizes[2],
                    cluster_m=cluster_m,
                )
                if fit_max > 0 and ab_stages <= fit_max:
                    return
                if fix_invalid and fit_max > 0:
                    config["tcgen05_ab_stages"] = fit_max
                    return
        if fix_invalid:
            config["tcgen05_ab_stages"] = 3
            return
        raise InvalidConfig(
            "tcgen05_ab_stages > 3 is only supported by the validated "
            "Target1 TVM-FFI seed (or fp8 within the SMEM budget)"
        )

    def _is_validated_clc_persistence_search_candidate(
        self, config: dict[str, object]
    ) -> bool:
        if not self._clc_persistence_search_enabled():
            return False
        if not self._is_validated_cluster_m2_edge_search_candidate(config):
            return False
        if config.get("pid_type") != TCGEN05_TWO_CTA_SEED_PID_TYPE:
            return False
        if config.get("tcgen05_cluster_n", 1) != 1:
            return False
        if not self._is_with_scheduler_c_input_config(config):
            return False
        block_sizes = config.get("block_sizes")
        if not isinstance(block_sizes, list) or len(block_sizes) < 3:
            return False
        if block_sizes[0] != TCGEN05_TWO_CTA_BLOCK_M:
            return False
        if block_sizes[1] not in (
            TCGEN05_TWO_CTA_BLOCK_N,
            TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N,
        ):
            return False
        is_narrow_n = block_sizes[1] == TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N
        if is_narrow_n:
            if not self._has_any_matmul_fact_n_edge_for_block_n(
                TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N
            ):
                return False
            if (
                config.get(TCGEN05_AUX_LOAD_MODE_CONFIG_KEY)
                != TCGEN05_AUX_LOAD_MODE_TMA
            ):
                return False
            if block_sizes[2] not in (
                TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K,
                TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_K,
            ):
                return False
        elif block_sizes[2] != TCGEN05_TWO_CTA_EDGE_K_TAIL_BLOCK_K:
            return False
        if self.config_spec.supports_config_key("indexing"):
            indexing = config.get("indexing")
            if (
                not isinstance(indexing, list)
                or indexing != ["tensor_descriptor"] * self.config_spec.indexing.length
            ):
                return False
        return True

    def _fix_clc_persistence_search_config(self, config: dict[str, object]) -> None:
        if (
            config.get(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY)
            != Tcgen05PersistenceModel.CLC_PERSISTENT.value
        ):
            return
        if self._is_validated_clc_persistence_search_candidate(config):
            return
        config[TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY] = (
            self.persistence_model_default_from_config(config).value
        )

    def _validate_sched_stage_count_config(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        self._validate_int_enum_config(
            config,
            TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY,
            TCGEN05_SCHED_STAGE_COUNTS,
            fix_invalid=fix_invalid,
        )
        block_sizes = config.get("block_sizes")
        is_full_role_local_two_cta_shape = (
            isinstance(block_sizes, list)
            and len(block_sizes) >= 3
            and block_sizes[0] == TCGEN05_TWO_CTA_BLOCK_M
            and config.get("pid_type") == TCGEN05_TWO_CTA_SEED_PID_TYPE
            and config.get("tcgen05_cluster_m", 1) == 2
            and config.get("tcgen05_cluster_n", 1) == 1
            and config.get(TCGEN05_CLUSTER_M2_ONE_CTA_ROLE_LOCAL_CONFIG_KEY) is not True
        )
        if (
            TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY in config
            and config.get(TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY) != 1
            and (
                config.get(TCGEN05_STRATEGY_CONFIG_KEY)
                != Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value
                or config.get(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY)
                != Tcgen05PersistenceModel.CLC_PERSISTENT.value
                or config.get(TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY, 0) == 0
                or not is_full_role_local_two_cta_shape
            )
        ):
            if fix_invalid:
                config.pop(TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY, None)
            else:
                raise InvalidConfig(
                    f"{TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY}=2 is only supported "
                    "with "
                    f"{TCGEN05_STRATEGY_CONFIG_KEY}="
                    f"{Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value!r} and "
                    f"{TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY}="
                    f"{Tcgen05PersistenceModel.CLC_PERSISTENT.value!r} and "
                    "the omitted shared-loop full role-local CtaGroup.TWO "
                    "shape: pid_type='persistent_interleaved', "
                    "tcgen05_cluster_m=2, tcgen05_cluster_n=1, block_m=256, "
                    f"and {TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY} > 0"
                )

    def prepare_override_normalization(
        self,
        config: dict[str, object],
        overrides: Mapping[str, object],
    ) -> None:
        if "pid_type" not in overrides:
            return
        if TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY in overrides:
            return
        persistence_value = config.get(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY)
        if persistence_value is None:
            return
        try:
            persistence_model = Tcgen05PersistenceModel(persistence_value)
        except ValueError:
            return
        pid_type = overrides["pid_type"]
        if pid_type not in self.allowed_pid_types:
            return
        derived = derive_persistence_model_from_pid_type(pid_type)
        compatible = persistence_model is derived or (
            persistence_model is Tcgen05PersistenceModel.CLC_PERSISTENT
            and derived is Tcgen05PersistenceModel.STATIC_PERSISTENT
        )
        if not compatible:
            config.pop(TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY, None)

    def persistence_model_default_from_config(
        self,
        config: dict[str, object],
    ) -> Tcgen05PersistenceModel:
        """Derive default persistence from pid_type."""
        pid_type = config.get("pid_type", self.allowed_pid_types[0])
        if pid_type not in self.allowed_pid_types:
            pid_type = self.allowed_pid_types[0]
        return derive_persistence_model_from_pid_type(pid_type)

    def flatten_missing_field_default(
        self,
        key: str,
        config: dict[str, object],
    ) -> tuple[bool, object]:
        if key == TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY:
            # The autotuner search surface for this key is the collapsed
            # ``EnumFragment((True,))``; autotuner-generated configs always
            # set it via ``default_flat()`` mutation. ``flatten`` only hits
            # this branch on user-supplied configs that omit the key, where
            # absence means "no FFI promotion requested" — matches the
            # validation-view default and the special case at
            # ``normalize_pre_pid_type``.
            return True, False
        if key != TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY:
            return False, None
        projected_config = {
            config_key: [*value] if isinstance(value, list) else value
            for config_key, value in config.items()
        }
        self.fix_search_config(projected_config)
        return True, self.persistence_model_default_from_config(projected_config).value

    def _matmul_fact_has_edge_tile(
        self, config: dict[str, object], *, fact_index: int
    ) -> bool:
        block_sizes = config.get("block_sizes")
        if not isinstance(block_sizes, list):
            return False
        fact = self.config_spec.matmul_facts[fact_index]
        for static_size, block_id in (
            (fact.static_m, fact.m_block_id),
            (fact.static_n, fact.n_block_id),
            (fact.static_k, fact.k_block_id),
        ):
            if static_size is None or block_id is None:
                continue
            try:
                block_idx = self.config_spec.block_sizes.block_id_to_index(block_id)
            except KeyError:
                continue
            if block_idx >= len(block_sizes):
                continue
            block_size = block_sizes[block_idx]
            if (
                not isinstance(block_size, int)
                or isinstance(block_size, bool)
                or block_size <= 0
            ):
                continue
            if static_size % block_size != 0:
                return True
        return False

    def _has_any_matmul_fact_edge_tile(self, config: dict[str, object]) -> bool:
        return any(
            self._matmul_fact_has_edge_tile(config, fact_index=i)
            for i in range(len(self.config_spec.matmul_facts))
        )

    def _has_any_matmul_fact_n_edge_for_block_n(self, block_n: int) -> bool:
        for fact in self.config_spec.matmul_facts:
            if fact.static_n is None or fact.n_block_id is None:
                continue
            try:
                # Presence check: skip facts whose N block id is not registered
                # in this config spec.
                self.config_spec.block_sizes.block_id_to_index(fact.n_block_id)
            except KeyError:
                continue
            if fact.static_n % block_n != 0:
                return True
        return False

    def _fix_aux_edge_search_config(self, config: dict[str, object]) -> None:
        if not (self.search_enabled and self.aux_kernel_detected):
            return
        if not self._has_any_matmul_fact_edge_tile(config):
            return

        if self._is_validated_cluster_m2_edge_search_candidate(config):
            self._set_aux_edge_cluster_m2_prefix(config)
            return

        self._set_aux_edge_monolithic_prefix(config)
        config["tcgen05_acc_stages"] = 2
        config["tcgen05_c_stages"] = 4
        config[TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY] = 1
        if isinstance(config.get("l2_groupings"), list):
            config["l2_groupings"] = [1]
        if isinstance(config.get("indexing"), list):
            indexing = cast("list[object]", config["indexing"])
            for i in range(len(indexing)):
                indexing[i] = "pointer"
        block_sizes = config.get("block_sizes")
        if not isinstance(block_sizes, list):
            return
        for fact in self.config_spec.matmul_facts:
            if fact.static_m is not None and fact.m_block_id is not None:
                try:
                    m_idx = self.config_spec.block_sizes.block_id_to_index(
                        fact.m_block_id
                    )
                except KeyError:
                    m_idx = -1
                if 0 <= m_idx < len(block_sizes):
                    bm = block_sizes[m_idx]
                    if (
                        isinstance(bm, int)
                        and not isinstance(bm, bool)
                        and bm > 128
                        and fact.static_m % bm != 0
                    ):
                        block_sizes[m_idx] = 128
            # K-only bk=128 tails keep the default A SMEM atom; output-edge
            # bk=128 fallback is handled in cute_mma by forcing A INTER.
            # Both cases have runtime coverage in test_cute_lowerings.

    @staticmethod
    def _set_aux_edge_monolithic_prefix(config: dict[str, object]) -> None:
        config[TCGEN05_STRATEGY_CONFIG_KEY] = (
            Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value
        )
        config[TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY] = 0
        config[TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY] = 0
        if config.get("tcgen05_ab_stages") == 1:
            config["tcgen05_ab_stages"] = 2
        config.pop("epilogue_subtile", None)

    @staticmethod
    def _set_aux_edge_cluster_m2_prefix(config: dict[str, object]) -> None:
        if (
            config.get(TCGEN05_STRATEGY_CONFIG_KEY)
            == Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value
        ):
            config[TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY] = 1
            config[TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY] = 1
            config[TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY] = (
                TCGEN05_TWO_CTA_EDGE_K_TAIL_SCHEDULER_L2_SWIZZLE_SIZE
            )
        else:
            CuteTcgen05Config._set_aux_edge_monolithic_prefix(config)
            return
        if config.get("tcgen05_ab_stages") == 1:
            config["tcgen05_ab_stages"] = 2
        config.pop("epilogue_subtile", None)

    def _is_validated_cluster_m2_edge_search_candidate(
        self, config: dict[str, object]
    ) -> bool:
        if config.get("tcgen05_cluster_m") != 2:
            return False
        constraints = self.cluster_m2_search_constraints
        if constraints is None:
            return False
        return constraints.allow_edge_k_tail_family

    def _is_validated_cluster_m2_full_tile_search_candidate(
        self, config: dict[str, object]
    ) -> bool:
        # Cycle 46: full-tile cluster_m=2 candidate gate for aux-TMA admission.
        # After ``_fix_cluster_m2_search_config`` projects a full-tile sample to
        # the canonical 256x256x bk shape with ``persistent_interleaved`` pid,
        # this returns True so aux-TMA stays during the search-time fixup.
        # bk validity is already enforced by ``_fix_cluster_m2_search_config``;
        # we re-check it here so a stale/unprojected config cannot slip through.
        if config.get("tcgen05_cluster_m") != 2:
            return False
        constraints = self.cluster_m2_search_constraints
        if constraints is None or constraints.allow_edge_k_tail_family:
            return False
        if config.get("pid_type") != TCGEN05_TWO_CTA_SEED_PID_TYPE:
            return False
        if config.get("tcgen05_cluster_n", 1) != 1:
            return False
        block_sizes = config.get("block_sizes")
        if not isinstance(block_sizes, list) or len(block_sizes) < 3:
            return False
        if block_sizes[0] != TCGEN05_TWO_CTA_BLOCK_M:
            return False
        if block_sizes[1] != TCGEN05_TWO_CTA_BLOCK_N:
            return False
        bk = block_sizes[2]
        if not isinstance(bk, int) or isinstance(bk, bool):
            return False
        return self.cluster_m2_bk_is_valid(bk, constraints)

    def _fix_cluster_m1_persistent_search_config(
        self, config: dict[str, object]
    ) -> None:
        if not (
            self.search_enabled
            and config.get("tcgen05_cluster_m", 1) == 1
            and config.get("pid_type")
            in {"persistent_blocked", "persistent_interleaved"}
        ):
            return
        block_sizes = config.get("block_sizes")
        if not isinstance(block_sizes, list) or not block_sizes:
            return
        constraints = self.cluster_m2_search_constraints
        if constraints is not None and constraints.allow_edge_k_tail_family:
            # persistent_interleaved stays in the flat enum so cluster_m=2
            # edge-family samples can encode; cluster_m=1 samples from the
            # same surface must use the validated flat edge fallback.
            config["pid_type"] = "flat"
            return
        bm = block_sizes[0]
        if isinstance(bm, int) and not isinstance(bm, bool):
            block_sizes[0] = min(bm, TCGEN05_ONE_CTA_MAX_BLOCK_M)

    def restrict_num_epi_warps_search(self, choices: tuple[int, ...]) -> None:
        assert choices, "tcgen05_num_epi_warps search must allow at least one value"
        self.num_epi_warps_search_choices = choices

    def restrict_num_epi_warps_validation(self, choices: tuple[int, ...]) -> None:
        assert choices, "tcgen05_num_epi_warps validation must allow at least one value"
        self.num_epi_warps_validation_choices = choices

    def narrow_autotune_to_validated_configs(
        self,
        *,
        allow_persistent_pid_types: bool = False,
        allow_cluster_m2_search: bool = False,
        cluster_m2_static_k: int | None = None,
        allow_cluster_m2_edge_k_tail_family: bool = False,
        allow_cluster_m2_fp8_small_grid: bool = False,
        ab_stages_three_dtype_bytes: int | None = None,
        ab_stages_three_device: torch.device | None = None,
    ) -> None:
        # Keep the default tcgen05 surface to combinations with runtime
        # coverage. Some unvalidated combinations fail loudly at CuTe
        # construction/launch, while diagnostic pipeline modes can compile and
        # intentionally produce wrong output.
        if allow_cluster_m2_edge_k_tail_family:
            assert allow_cluster_m2_search, (
                "cluster_m=2 edge/K-tail admission requires cluster_m=2 search"
            )
        if allow_cluster_m2_fp8_small_grid:
            assert allow_cluster_m2_search, (
                "cluster_m=2 fp8 small-grid admission requires cluster_m=2 search"
            )
        cluster_m2_static_k_int: int | None = None
        if allow_cluster_m2_search:
            assert allow_persistent_pid_types or allow_cluster_m2_edge_k_tail_family, (
                "cluster_m=2 search requires persistent pid types or the "
                "validated output-edge + K-tail admission"
            )
            if cluster_m2_static_k is None:
                raise AssertionError("cluster_m=2 search requires a static K extent")
            cluster_m2_static_k_int = cluster_m2_static_k
        if allow_cluster_m2_edge_k_tail_family and (
            TCGEN05_TWO_CTA_SEED_PID_TYPE not in self.allowed_pid_types
        ):
            self.allowed_pid_types = (
                *self.allowed_pid_types,
                cast("PidTypeLiteral", TCGEN05_TWO_CTA_SEED_PID_TYPE),
            )
        if not allow_persistent_pid_types:
            self.config_spec.disallow_pid_type(
                "persistent_blocked",
                reason="tcgen05 two-CTA launch-grid contract does not allow "
                "persistent pid types here",
            )
            if not allow_cluster_m2_edge_k_tail_family:
                self.config_spec.disallow_pid_type(
                    "persistent_interleaved",
                    reason="tcgen05 two-CTA launch-grid contract does not allow "
                    "persistent pid types here",
                )
        if allow_cluster_m2_search:
            assert cluster_m2_static_k_int is not None
            self.allow_cluster_m2_search(
                static_k=cluster_m2_static_k_int,
                allow_edge_k_tail_family=allow_cluster_m2_edge_k_tail_family,
                allow_fp8_small_grid=allow_cluster_m2_fp8_small_grid,
            )
        else:
            self.restrict_cluster_m_search((1,))
        self.restrict_num_epi_warps_search((4,))
        self.restrict_num_epi_warps_validation((4,))
        if ab_stages_three_dtype_bytes is not None:
            assert ab_stages_three_device is not None, (
                "ab_stages_three_dtype_bytes requires ab_stages_three_device "
                "so the SMEM-budget gate consults the operand's device, not "
                "the host's current CUDA device"
            )
            self.allow_ab_stages_three_search(
                dtype_bytes=ab_stages_three_dtype_bytes,
                device=ab_stages_three_device,
            )

    def optional_fragments(
        self, *, for_search: bool = False
    ) -> dict[str, ConfigSpecFragment]:
        if for_search and self.cluster_m_search_choices is not None:
            cluster_m_choices = self.cluster_m_search_choices
        else:
            cluster_m_choices = (1, 2)
        cluster_n_choices: tuple[int, ...] = (1,) if for_search else (1, 2)
        if for_search and self.num_epi_warps_search_choices is not None:
            num_epi_warps_fragment: ConfigSpecFragment = EnumFragment(
                self.num_epi_warps_search_choices
            )
        elif not for_search and self.num_epi_warps_validation_choices is not None:
            num_epi_warps_fragment = EnumFragment(self.num_epi_warps_validation_choices)
        else:
            num_epi_warps_fragment = IntegerFragment(1, 4, 4)
        if not for_search:
            # Validation admits what the direct-entry codegen supports: bk=64
            # accepts the deep (ab=6, c=4) tuple (see
            # ``TCGEN05_DIRECT_ENTRY_STAGE_TUPLES_BY_BK``), so the validation
            # surface lifts the AB cap to 6 for FFI-eligible shapes. The actual
            # (bk, ab, c) tuple is gated by ``_validate_target1_ab_stage_envelope``;
            # SEARCH stays capped at 3 (budget-aware) since the generalized seed
            # runs at ab=3 and deeper pipelines are not worth searching.
            ab_stages_max = 6 if self.full_tile_direct_entry_seed_eligible() else 3
            # FP8 (1-byte) operands fit a deeper AB pipeline than the bf16-tuned
            # cap; admit a frozen deep-staged fp8 config on the validation
            # surface too (``_validate_target1_ab_stage_envelope`` clamps it to
            # the actual per-CTA SMEM budget for the chosen block sizes).
            constraints = self.ab_stages_three_search_constraints
            if constraints is not None and constraints.dtype_bytes == 1:  # FP8
                ab_stages_max = self._get_dtype_ab_stages_hard_cap(
                    constraints.dtype_bytes
                )
        elif self.ab_stages_three_search_constraints is not None:
            # Cycle 97: make ab=3 BUDGET-AWARE-SEARCHABLE. Where the device/dtype
            # admits ab=3 at all (the SMEM-budget constraints were recorded by
            # ``allow_ab_stages_three_search`` at bind time — B200-class optin cap,
            # bf16/fp16), lift the ``for_search`` cap to 3 so the autotuner can
            # SAMPLE ab=3 directly instead of reaching it only through the per-shape
            # FFI / gelu seeds. ``_fix_ab_stages_search_config`` then demotes any
            # sampled ab=3 that does not fit (the residual/source-C ring overflows;
            # cluster_m=1 256x256 overflows bare-AB) before codegen, so admission is
            # free but an overflowing kernel is never generated.
            ab_stages_max = 3
            # FP8 (1-byte) operands fit a deeper AB pipeline; widen the
            # validation range so an explicit deep-staged fp8 config is
            # accepted (``_validate_target1_ab_stage_envelope`` clamps it to
            # the actual per-CTA SMEM budget for the chosen block sizes).
            constraints = self.ab_stages_three_search_constraints
            if constraints is not None and constraints.dtype_bytes == 1:  # FP8
                ab_stages_max = self._get_dtype_ab_stages_hard_cap(
                    constraints.dtype_bytes
                )
        else:
            ab_stages_max = 2
        if for_search:
            l2_swizzle_choices = tuple(
                v for v in TCGEN05_LEGAL_L2_SWIZZLE_SIZES if v <= 8
            )
        else:
            l2_swizzle_choices = TCGEN05_LEGAL_L2_SWIZZLE_SIZES
        fragments: dict[str, ConfigSpecFragment] = {
            "tcgen05_cluster_m": EnumFragment(cluster_m_choices),
            "tcgen05_cluster_n": EnumFragment(cluster_n_choices),
            "tcgen05_ab_stages": IntegerFragment(1, ab_stages_max, 2),
            "tcgen05_acc_stages": IntegerFragment(1, 2, 2),
            "tcgen05_c_stages": EnumFragment((2, 4)),
            "tcgen05_num_epi_warps": num_epi_warps_fragment,
            TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY: EnumFragment(l2_swizzle_choices),
        }
        if self.full_tile_direct_entry_seed_eligible():
            # Search collapses the FFI-launch knob to a single True
            # choice — the runtime now always enables FFI, so the
            # autotuner only needs to explore the True arm to keep the
            # seeded direct-entry config family in scope. Validation
            # keeps a Boolean surface so an absent user-config key still
            # means "no FFI promotion requested" (default False).
            tvm_ffi_launch_fragment: ConfigSpecFragment = (
                EnumFragment((True,)) if for_search else BooleanFragment()
            )
            fragments.update(
                {
                    TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY: BooleanFragment(),
                    TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY: tvm_ffi_launch_fragment,
                    TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY: EnumFragment((None, 128)),
                    TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY: EnumFragment((None, 32)),
                    TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY: EnumFragment(
                        (None, 32)
                    ),
                }
            )
        return fragments

    @staticmethod
    def _target1_tvm_ffi_promotion_requested(
        config: dict[str, object], *, seed_enabled: bool
    ) -> bool:
        return (
            config.get(TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY) is True
            or config.get(TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY) is True
            or (seed_enabled and config.get("tcgen05_cluster_m") == 2)
            or config.get(TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY)
            == Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value
            or config.get(TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY) is not None
            or config.get(TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY) is not None
            or config.get(TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY) is not None
        )

    @staticmethod
    def _clear_target1_tvm_ffi_promotion_surface(config: dict[str, object]) -> None:
        for key in list(config):
            if key.startswith("tcgen05_") or key == "epilogue_subtile":
                config.pop(key, None)

    def _fix_target1_tvm_ffi_search_config(self, config: dict[str, object]) -> None:
        # The generalized direct-entry seed projects FFI-requesting search
        # candidates onto the validated CtaGroup.TWO envelope for ANY eligible
        # shape (returns None for ineligible shapes, in which case the
        # promotion surface is stripped back to the DEFAULT layout below).
        seed = self.full_tile_direct_entry_seed_config()
        if not self._target1_tvm_ffi_promotion_requested(
            config, seed_enabled=seed is not None
        ):
            return
        if seed is None:
            config[TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY] = False
            config[TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY] = False
            if (
                config.get(TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY)
                == Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value
            ):
                config[TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY] = (
                    Tcgen05LayoutStrategy.DEFAULT.value
                )
            for key in (
                TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY,
                TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY,
                TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY,
            ):
                config[key] = None
            return
        self._clear_target1_tvm_ffi_promotion_surface(config)
        config.update(seed.config)

    def aux_load_mode_autotune_fragments(self) -> dict[str, ConfigSpecFragment]:
        if not self._aux_tma_search_enabled():
            return {}
        return {
            TCGEN05_AUX_LOAD_MODE_CONFIG_KEY: EnumFragment(
                (TCGEN05_AUX_LOAD_MODE_SIMT, TCGEN05_AUX_LOAD_MODE_TMA)
            )
        }

    def aux_stages_autotune_fragments(self) -> dict[str, ConfigSpecFragment]:
        """Per-config aux-pipeline stage-count knob.

        Admitted only under ``_aux_tma_edge_search_enabled``, which pins the
        surface to the validated edge+K-tail family with ``cluster_m=2``
        and the c-input warp + aux-TMA combination. Configs outside that
        gate never see the knob; codegen at the default of 2 is unchanged.

        Cycle 46 intentionally keeps this scoped to the edge+K-tail gate
        even though ``_aux_tma_search_enabled`` was widened to admit the
        full-tile cluster_m=2 family. The stage-count choices were tuned
        on the T8/CLC edge rows; exposing them to T14/T20/T25/T28 would
        let autotune sample stage counts on shapes they were not measured
        on.
        """
        if not self._aux_tma_edge_search_enabled():
            return {}
        return {
            TCGEN05_AUX_STAGES_CONFIG_KEY: EnumFragment(TCGEN05_AUX_STAGE_COUNT_CHOICES)
        }

    def consumer_regs_autotune_fragments(self) -> dict[str, ConfigSpecFragment]:
        """Per-config consumer-warp ``setmaxregister_increase`` ceiling knob.

        Admission mirrors ``aux_stages_autotune_fragments``: the
        ``_aux_tma_edge_search_enabled`` gate pins the search to the
        validated wide-N CLC + aux-TMA seed family with the c-input warp +
        aux-TMA combination. Configs outside that gate never see the
        knob. The default value (256) is included in
        ``TCGEN05_CONSUMER_REGS_CHOICES`` so default-with-knob emits the
        same code as default-without-knob.

        Cycle 46 intentionally keeps this scoped to the edge+K-tail gate
        even though ``_aux_tma_search_enabled`` was widened (see
        ``aux_stages_autotune_fragments``).
        """
        if not self._aux_tma_edge_search_enabled():
            return {}
        return {
            TCGEN05_CONSUMER_REGS_CONFIG_KEY: EnumFragment(
                TCGEN05_CONSUMER_REGS_CHOICES
            )
        }

    def persistence_model_autotune_fragments(self) -> dict[str, ConfigSpecFragment]:
        if not self._clc_persistence_search_enabled():
            return {}
        default_model = derive_persistence_model_from_pid_type(
            self.allowed_pid_types[0]
        ).value
        choices = tuple(
            dict.fromkeys(
                (
                    default_model,
                    Tcgen05PersistenceModel.NON_PERSISTENT.value,
                    Tcgen05PersistenceModel.STATIC_PERSISTENT.value,
                    Tcgen05PersistenceModel.CLC_PERSISTENT.value,
                )
            )
        )
        return {TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY: EnumFragment(choices)}

    def strategy_autotune_fragments(self) -> dict[str, ConfigSpecFragment]:
        # Aux kernels are the only current trigger for scheduler/c_input warp
        # search. The surface is derived from aux_kernel_detected so repeated
        # detection or repeated fragment construction stays idempotent.
        direct_entry_seed_eligible = self.full_tile_direct_entry_seed_eligible()
        if self.aux_kernel_detected:
            strategy_choices: tuple[str, ...] = (
                Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value,
                Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value,
            )
            scheduler_warps_choices: tuple[int, ...] = (0, 1)
            c_input_warps_choices: tuple[int, ...] = (0, 1)
        else:
            strategy_choices = (Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value,)
            scheduler_warps_choices = (0,)
            c_input_warps_choices = (0,)
        # The store-warp slot stays narrowed to ``0`` in the autotune surface —
        # only an explicit ``helion.Config(tcgen05_warp_spec_store_warps=1)``
        # activates it. Cycle 93 (Workstream A Stage 4) landed the productive
        # decouple body (the C-store edge + tail split, +1.1 % on T20), but the
        # autotune surface stays at ``0`` until Stage 5 wires store=1 into the
        # residual family's production config + runs the full regression sweep,
        # so no passing target can pick it before it is characterized per-family.
        store_warps_choices: tuple[int, ...] = (0,)
        if direct_entry_seed_eligible:
            layout_choices = (
                Tcgen05LayoutStrategy.DEFAULT.value,
                Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value,
            )
        else:
            layout_choices = (Tcgen05LayoutStrategy.DEFAULT.value,)
        return {
            TCGEN05_STRATEGY_CONFIG_KEY: EnumFragment(strategy_choices),
            TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY: EnumFragment(layout_choices),
            TCGEN05_WARP_SPEC_MMA_WARPS_KEY: EnumFragment((1,)),
            TCGEN05_WARP_SPEC_AB_LOAD_WARPS_KEY: EnumFragment((1,)),
            TCGEN05_WARP_SPEC_EPI_LOAD_WARPS_KEY: EnumFragment((0,)),
            TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY: EnumFragment(
                scheduler_warps_choices
            ),
            TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY: EnumFragment(c_input_warps_choices),
            TCGEN05_WARP_SPEC_STORE_WARPS_KEY: EnumFragment(store_warps_choices),
            TCGEN05_WARP_SPEC_REGISTER_DECREASE_KEY: EnumFragment(
                (ROLE_LOCAL_MONOLITHIC_DEFAULT_WARP_SPEC.register_split[0],)
            ),
            TCGEN05_WARP_SPEC_REGISTER_INCREASE_KEY: EnumFragment(
                (ROLE_LOCAL_MONOLITHIC_DEFAULT_WARP_SPEC.register_split[1],)
            ),
        }

    def strategy_validation_fragments(self) -> dict[str, ConfigSpecFragment]:
        fragments = self.strategy_autotune_fragments()
        fragments[TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY] = EnumFragment(
            (
                Tcgen05PersistenceModel.NON_PERSISTENT.value,
                Tcgen05PersistenceModel.STATIC_PERSISTENT.value,
                Tcgen05PersistenceModel.CLC_PERSISTENT.value,
            )
        )
        fragments[TCGEN05_STRATEGY_CONFIG_KEY] = EnumFragment(
            (
                Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value,
                Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value,
                Tcgen05Strategy.PURE_MATMUL_ROLE_LIFECYCLE.value,
            )
        )
        fragments[TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY] = EnumFragment(
            (
                Tcgen05LayoutStrategy.DEFAULT.value,
                Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value,
            )
        )
        fragments[TCGEN05_WARP_SPEC_SCHEDULER_WARPS_KEY] = EnumFragment((0, 1))
        fragments[TCGEN05_WARP_SPEC_C_INPUT_WARPS_KEY] = EnumFragment((0, 1))
        # Cycle 91 (Workstream A Stage 3): the user-config validation surface
        # accepts ``{0, 1}`` so an explicit ``store_warps=1`` round-trips; the
        # per-strategy accept set in ``_STRATEGY_SUPPORTED_STORE_WARPS`` still
        # pins it to ``{0}`` outside ROLE_LOCAL_WITH_SCHEDULER.
        fragments[TCGEN05_WARP_SPEC_STORE_WARPS_KEY] = EnumFragment((0, 1))
        return fragments

    @staticmethod
    def strategy_field_default(key: str, *, pid_type: object = None) -> object:
        if key == TCGEN05_STRATEGY_CONFIG_KEY:
            return Tcgen05Strategy.ROLE_LOCAL_MONOLITHIC.value
        if key == TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY:
            return derive_persistence_model_from_pid_type(pid_type).value
        if key == TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY:
            return Tcgen05LayoutStrategy.DEFAULT.value
        if key in TCGEN05_WARP_SPEC_DEFAULTS_BY_KEY:
            return TCGEN05_WARP_SPEC_DEFAULTS_BY_KEY[key]
        raise KeyError(f"Unknown tcgen05 strategy field: {key!r}")

    def validate_strategy_invariants(
        self,
        config: dict[str, object],
        *,
        fix_invalid: bool,
    ) -> None:
        strategy = Tcgen05Strategy(config[TCGEN05_STRATEGY_CONFIG_KEY])
        persistence_model = Tcgen05PersistenceModel(
            config[TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY]
        )
        layout_strategy = Tcgen05LayoutStrategy(
            config[TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY]
        )
        warp_spec = warp_spec_from_config(config)
        layout_overrides = layout_overrides_from_config(config)

        cluster_m_raw = config.get("tcgen05_cluster_m", 1)
        cluster_m = int(cluster_m_raw) if isinstance(cluster_m_raw, int) else 1
        cluster_n_raw = config.get("tcgen05_cluster_n", 1)
        cluster_n = int(cluster_n_raw) if isinstance(cluster_n_raw, int) else 1
        capability = self.config_spec.target_device_capability
        arch_major = capability[0] if capability is not None else None
        errors = validate_tcgen05_strategy_invariants(
            strategy=strategy,
            persistence_model=persistence_model,
            layout_strategy=layout_strategy,
            warp_spec=warp_spec,
            layout_overrides=layout_overrides,
            pid_type=config.get("pid_type"),
            cluster_m=cluster_m,
            cluster_n=cluster_n,
            arch_major=arch_major,
        )
        if not errors:
            return
        if fix_invalid:
            pid_type = config.get("pid_type")
            for key in TCGEN05_STRATEGY_CONFIG_KEYS:
                if key in TCGEN05_LAYOUT_OVERRIDES_KEYS:
                    config[key] = None
                else:
                    config[key] = self.strategy_field_default(key, pid_type=pid_type)
            return
        message = "; ".join(errors)
        raise InvalidConfig(f"tcgen05 strategy invariants violated: {message}")

    def _clamp_l2_swizzle_size_to_shape(self, config: dict[str, object]) -> None:
        # CuTe layout construction assumes the L2 swizzle does not exceed the
        # number of N tile-clusters; clamp before layout objects are built.
        swizzle_value = config.get(TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY)
        if swizzle_value is None or swizzle_value == 1:
            return
        if len(self.config_spec.block_sizes) < 2:
            return
        block_sizes = config.get("block_sizes")
        if not isinstance(block_sizes, list) or len(block_sizes) < 2:
            return
        bn = block_sizes[1]
        if not isinstance(bn, int) or isinstance(bn, bool) or bn <= 0:
            return
        n_hint = self.config_spec.block_sizes[1].size_hint
        if n_hint <= 0:
            return
        cluster_n_raw = config.get("tcgen05_cluster_n", 1)
        cluster_n = (
            cluster_n_raw
            if isinstance(cluster_n_raw, int) and not isinstance(cluster_n_raw, bool)
            else 1
        )
        cluster_n = max(cluster_n, 1)
        ncluster_n = max(((n_hint + bn - 1) // bn) // cluster_n, 1)
        if not isinstance(swizzle_value, int) or isinstance(swizzle_value, bool):
            return
        if swizzle_value <= ncluster_n:
            return
        clamped = max(
            (v for v in TCGEN05_LEGAL_L2_SWIZZLE_SIZES if v <= ncluster_n),
            default=1,
        )
        config[TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY] = clamped

    def normalize_pre_pid_type(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        optional_fragments = self.optional_fragments()
        optional_search_fragments = self.optional_fragments(for_search=True)
        if self.search_enabled:
            for key, fragment in optional_fragments.items():
                if key in config:
                    config[key] = self._validate_optional_fragment_value(
                        key, fragment, config[key]
                    )
                elif key in optional_search_fragments:
                    if key == TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY:
                        # An omitted user-config means "no FFI promotion
                        # requested" — fill from the validation surface
                        # so the non-seed envelope and promotion gates
                        # that key off ``config.get(...) is True`` stay
                        # consistent.
                        config[key] = optional_fragments[key].default()
                    else:
                        config[key] = optional_search_fragments[key].default()
            self._clamp_l2_swizzle_size_to_shape(config)
            self._validate_target1_ab_stage_envelope(config, fix_invalid=fix_invalid)
        else:
            for key in optional_fragments:
                if key not in config:
                    continue
                if fix_invalid:
                    config.pop(key, None)
                else:
                    raise InvalidConfig(
                        f"{key} is only supported for tcgen05-enabled CuTe matmul kernels"
                    )

        strategy_validation_fragments = self.strategy_validation_fragments()
        if not self.search_enabled:
            for key in (
                *strategy_validation_fragments.keys(),
                *TCGEN05_LAYOUT_OVERRIDES_KEYS,
            ):
                if key not in config:
                    continue
                if fix_invalid:
                    config.pop(key, None)
                else:
                    raise InvalidConfig(
                        f"{key} is only supported for tcgen05-enabled CuTe matmul kernels"
                    )

        self._validate_enum_config(
            config,
            TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY,
            TCGEN05_C_ACQUIRE_PLACEMENTS,
            fix_invalid=fix_invalid,
        )
        self._validate_enum_config(
            config,
            TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY,
            TCGEN05_ACC_WAIT_PLACEMENTS,
            fix_invalid=fix_invalid,
        )
        self._validate_enum_config(
            config,
            TCGEN05_AUX_LOAD_MODE_CONFIG_KEY,
            TCGEN05_AUX_LOAD_MODES,
            fix_invalid=fix_invalid,
        )
        self._validate_int_enum_config(
            config,
            TCGEN05_AUX_STAGES_CONFIG_KEY,
            TCGEN05_AUX_STAGE_COUNT_CHOICES,
            fix_invalid=fix_invalid,
        )
        self._validate_int_enum_config(
            config,
            TCGEN05_CONSUMER_REGS_CONFIG_KEY,
            TCGEN05_CONSUMER_REGS_CHOICES,
            fix_invalid=fix_invalid,
        )
        self._validate_bool_config(
            config,
            TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY,
            fix_invalid=fix_invalid,
        )

        for key, modes, normal_mode in (
            (
                TCGEN05_C_STORE_MODE_CONFIG_KEY,
                TCGEN05_C_STORE_MODES,
                TCGEN05_C_STORE_MODE_NORMAL,
            ),
            (
                TCGEN05_ACC_PRODUCER_MODE_CONFIG_KEY,
                TCGEN05_ACC_PRODUCER_MODES,
                TCGEN05_ACC_PRODUCER_MODE_NORMAL,
            ),
            (
                TCGEN05_ACC_PRODUCER_ADVANCE_MODE_CONFIG_KEY,
                TCGEN05_ACC_PRODUCER_ADVANCE_MODES,
                TCGEN05_ACC_PRODUCER_ADVANCE_MODE_NORMAL,
            ),
            (
                TCGEN05_AB_PRODUCER_ACQUIRE_MODE_CONFIG_KEY,
                TCGEN05_AB_PRODUCER_ACQUIRE_MODES,
                TCGEN05_AB_PRODUCER_ACQUIRE_MODE_NORMAL,
            ),
            (
                TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODE_CONFIG_KEY,
                TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODES,
                TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODE_NORMAL,
            ),
            (
                TCGEN05_AB_PRODUCER_ADVANCE_MODE_CONFIG_KEY,
                TCGEN05_AB_PRODUCER_ADVANCE_MODES,
                TCGEN05_AB_PRODUCER_ADVANCE_MODE_NORMAL,
            ),
            (
                TCGEN05_AB_CONSUMER_WAIT_MODE_CONFIG_KEY,
                TCGEN05_AB_CONSUMER_WAIT_MODES,
                TCGEN05_AB_CONSUMER_WAIT_MODE_NORMAL,
            ),
            (
                TCGEN05_AB_CONSUMER_PHASE_MODE_CONFIG_KEY,
                TCGEN05_AB_CONSUMER_PHASE_MODES,
                TCGEN05_AB_CONSUMER_PHASE_MODE_NORMAL,
            ),
        ):
            self._validate_diagnostic_mode(
                config, key, modes, normal_mode, fix_invalid=fix_invalid
            )

        self._validate_bool_config(
            config, TCGEN05_CUBIN_LINEINFO_CONFIG_KEY, fix_invalid=fix_invalid
        )
        self._validate_bool_config(
            config, TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY, fix_invalid=fix_invalid
        )
        self._validate_bool_config(
            config, TCGEN05_LARGE_BN_PROOF_CONFIG_KEY, fix_invalid=fix_invalid
        )
        self._validate_bool_config(
            config,
            TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY,
            fix_invalid=fix_invalid,
        )
        if config.get(TCGEN05_LARGE_BN_PROOF_CONFIG_KEY) is True:
            proof_envelope_matches = (
                tuple(cast("list[int]", config.get("block_sizes", [])))
                == TCGEN05_LARGE_BN_PROOF_BLOCK_SIZES
                and config.get("tcgen05_cluster_m", 1)
                == TCGEN05_LARGE_BN_PROOF_CLUSTER_M
                and config.get("pid_type", "flat") == TCGEN05_LARGE_BN_PROOF_PID_TYPE
                and all(
                    config.get(key) == expected
                    for key, expected in TCGEN05_LARGE_BN_PROOF_STAGE_CONFIGS
                )
            )
            if not proof_envelope_matches:
                if fix_invalid:
                    config.pop(TCGEN05_LARGE_BN_PROOF_CONFIG_KEY, None)
                else:
                    raise InvalidConfig(
                        f"{TCGEN05_LARGE_BN_PROOF_CONFIG_KEY}=True requires "
                        f"block_sizes={list(TCGEN05_LARGE_BN_PROOF_BLOCK_SIZES)}, "
                        f"tcgen05_cluster_m={TCGEN05_LARGE_BN_PROOF_CLUSTER_M}, "
                        f"pid_type={TCGEN05_LARGE_BN_PROOF_PID_TYPE!r}, "
                        "tcgen05_ab_stages=2, tcgen05_acc_stages=1, "
                        "and tcgen05_c_stages=2"
                    )
        self._validate_bool_config(
            config,
            TCGEN05_CLUSTER_M2_ONE_CTA_ROLE_LOCAL_CONFIG_KEY,
            fix_invalid=fix_invalid,
        )
        self._validate_enum_config(
            config,
            TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY,
            TCGEN05_EPILOGUE_LAYOUTS,
            fix_invalid=fix_invalid,
        )
        self._validate_enum_config(
            config,
            TCGEN05_SCHED_CONSUMER_WAIT_MODE_CONFIG_KEY,
            TCGEN05_SCHED_CONSUMER_WAIT_MODES,
            fix_invalid=fix_invalid,
        )
        if (
            TCGEN05_SCHED_CONSUMER_WAIT_MODE_CONFIG_KEY in config
            and config.get(TCGEN05_SCHED_CONSUMER_WAIT_MODE_CONFIG_KEY)
            != TCGEN05_SCHED_CONSUMER_WAIT_MODE_NORMAL
            and config.get(TCGEN05_STRATEGY_CONFIG_KEY)
            != Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value
        ):
            if fix_invalid:
                config.pop(TCGEN05_SCHED_CONSUMER_WAIT_MODE_CONFIG_KEY, None)
            else:
                raise InvalidConfig(
                    f"{TCGEN05_SCHED_CONSUMER_WAIT_MODE_CONFIG_KEY} is only "
                    "supported with "
                    f"{TCGEN05_STRATEGY_CONFIG_KEY}="
                    f"{Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value!r}"
                )
        self._validate_sched_stage_count_config(config, fix_invalid=fix_invalid)

    def _validate_enum_config(
        self,
        config: dict[str, object],
        key: str,
        choices: tuple[str, ...],
        *,
        fix_invalid: bool,
    ) -> None:
        if key not in config:
            return
        if not self.search_enabled:
            if fix_invalid:
                config.pop(key, None)
                return
            raise InvalidConfig(
                f"{key} is only supported for tcgen05-enabled CuTe matmul kernels"
            )
        if config[key] not in choices:
            if fix_invalid:
                config.pop(key, None)
                return
            raise InvalidConfig(
                f"{key} must be one of {choices!r}, got {config[key]!r}"
            )

    def _validate_int_enum_config(
        self,
        config: dict[str, object],
        key: str,
        choices: tuple[int, ...],
        *,
        fix_invalid: bool,
    ) -> None:
        if key not in config:
            return
        if not self.search_enabled:
            if fix_invalid:
                config.pop(key, None)
                return
            raise InvalidConfig(
                f"{key} is only supported for tcgen05-enabled CuTe matmul kernels"
            )
        value = config[key]
        # ``bool`` is an ``int`` subclass, but it is not a valid stage count.
        if type(value) is not int or value not in choices:
            if fix_invalid:
                config.pop(key, None)
                return
            raise InvalidConfig(f"{key} must be one of {choices!r}, got {value!r}")

    def _validate_bool_config(
        self,
        config: dict[str, object],
        key: str,
        *,
        fix_invalid: bool,
    ) -> None:
        if key not in config:
            return
        if not self.search_enabled:
            if fix_invalid:
                config.pop(key, None)
                return
            raise InvalidConfig(
                f"{key} is only supported for tcgen05-enabled CuTe matmul kernels"
            )
        if not isinstance(config[key], bool):
            if fix_invalid:
                config.pop(key, None)
                return
            raise InvalidConfig(f"{key} must be a boolean")

    def _validate_diagnostic_mode(
        self,
        config: dict[str, object],
        key: str,
        modes: tuple[str, ...],
        normal_mode: str,
        *,
        fix_invalid: bool,
    ) -> None:
        if key not in config:
            return
        if not self.search_enabled:
            if fix_invalid:
                config.pop(key, None)
                return
            raise InvalidConfig(
                f"{key} is only supported for tcgen05-enabled CuTe matmul kernels"
            )
        if config[key] not in modes:
            if fix_invalid:
                config.pop(key, None)
                return
            raise InvalidConfig(f"{key} must be one of {modes!r}, got {config[key]!r}")
        if (
            config[key] != normal_mode
            and config.get(TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY) is not True
        ):
            if fix_invalid:
                config.pop(key, None)
                return
            raise InvalidConfig(
                f"{key}={config[key]!r} changes output correctness; set "
                f"{TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY}=True "
                "only for diagnostic invalid-output runs"
            )

    def fix_search_config(self, config: dict[str, object]) -> None:
        self._fix_aux_edge_search_config(config)
        self._fix_cluster_m2_search_config(config)
        self._fix_cluster_m1_persistent_search_config(config)
        self._fix_ab_stages_three_search_config(config)
        self._fix_target1_tvm_ffi_search_config(config)
        self._fix_aux_tma_full_tile_search_config(config)
        self._fix_with_scheduler_search_config(config)
        self._fix_aux_tma_search_config(config)
        # Cycle 97: budget-aware ab-stages admission for the lifted for_search
        # cap. Runs after the family projections (FFI / gelu / aux-TMA full-tile)
        # have set their validated stage tuple, so a directly-SAMPLED ab=3 that no
        # projection claimed — and that does not fit (residual/source-C ring
        # overflow, or cluster_m=1 256x256 bare-AB overflow) — is demoted to 2 on
        # the DEFAULT-layout full-tile path before codegen. The plain silu/gelu
        # ab=3 winner (no source-C, cluster_m=2) is preserved.
        self._fix_ab_stages_search_config(config)
        # Final c-stages admission: demote a directly-sampled (or any unclaimed)
        # deeper C ring on the canonical 256x256 DEFAULT-layout path that does
        # not fit AB+C under the B200 cap. Runs after the family fixups so their
        # validated c=4 (residual ab=2 projection above; edge/seed families,
        # which this gate's scope excludes) is set first.
        self._fix_c_stages_search_config(config)
        # CLC admission depends on the projected cluster/pid/strategy tuple
        # above, including the scheduler/c-input warp fix-ups.
        self._fix_clc_persistence_search_config(config)
        self._validate_sched_stage_count_config(config, fix_invalid=True)

    def normalize_strategy(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        if not self.search_enabled:
            return
        pid_type_for_default = config.get("pid_type")
        strategy_validation_fragments = self.strategy_validation_fragments()
        for key, fragment in strategy_validation_fragments.items():
            if key in config:
                config[key] = self._validate_optional_fragment_value(
                    key, fragment, config[key]
                )
            else:
                config[key] = self.strategy_field_default(
                    key, pid_type=pid_type_for_default
                )
        swizzle_keys = {
            TCGEN05_LAYOUT_OVERRIDES_SWIZZLE_A_KEY,
            TCGEN05_LAYOUT_OVERRIDES_SWIZZLE_B_KEY,
        }
        for key in TCGEN05_LAYOUT_OVERRIDES_KEYS:
            if key in config:
                value = config[key]
                if value is None:
                    continue
                if key in swizzle_keys:
                    if (
                        type(value) is not int
                        or value not in TCGEN05_LEGAL_SMEM_SWIZZLE_BYTES
                    ):
                        if fix_invalid:
                            config[key] = None
                        else:
                            raise InvalidConfig(
                                f"{key} must be one of "
                                f"{TCGEN05_LEGAL_SMEM_SWIZZLE_BYTES!r} "
                                f"or None, got {value!r}"
                            )
                elif not (type(value) is int and value > 0):
                    if fix_invalid:
                        config[key] = None
                    else:
                        raise InvalidConfig(
                            f"{key} must be a positive integer or None, got {value!r}"
                        )
            else:
                config[key] = None
        self.validate_strategy_invariants(config, fix_invalid=fix_invalid)
        if fix_invalid:
            # Strategy validation can reset scheduler/c-input fields for
            # inconsistent user configs. Revalidate aux-TMA after that reset so
            # TMA aux loads do not outlive their producer warp. CLC does not
            # need a matching second pass because the reset path never produces
            # a CLC persistence model.
            self._fix_aux_tma_search_config(config)

    def flat_fields(
        self,
    ) -> dict[str, BlockIdSequence[Any] | ConfigSpecFragment]:
        fields: dict[str, BlockIdSequence[Any] | ConfigSpecFragment] = {
            "l2_groupings": self.config_spec.l2_groupings,
        }
        fields.update(self.optional_fragments(for_search=True))
        fields.update(self.strategy_autotune_fragments())
        fields.update(self.aux_load_mode_autotune_fragments())
        fields.update(self.aux_stages_autotune_fragments())
        fields.update(self.consumer_regs_autotune_fragments())
        fields.update(self.persistence_model_autotune_fragments())
        if self.config_spec.supports_config_key("pid_type"):
            fields["pid_type"] = EnumFragment(self.allowed_pid_types)
        if (
            self.config_spec.supports_config_key("indexing")
            and self.config_spec.indexing.length > 0
        ):
            fields["indexing"] = self.config_spec.indexing
        return fields
