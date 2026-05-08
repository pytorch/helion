from __future__ import annotations

import functools
import hashlib
import itertools
import logging
import math
import operator
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import cast

import torch
from torch._inductor.runtime.runtime_utils import next_power_of_2
import torch.distributed as dist

from .._compat import _regs_per_block
from .._compat import num_compute_units
from .._compat import supports_amd_cdna_tunables
from .._compat import supports_maxnreg
from .._compat import supports_tensor_descriptor
from .._compat import warps_to_threads
from .._compiler.cute.tcgen05_constants import TCGEN05_AB_CONSUMER_PHASE_MODE_CONFIG_KEY
from .._compiler.cute.tcgen05_constants import TCGEN05_AB_CONSUMER_PHASE_MODE_NORMAL
from .._compiler.cute.tcgen05_constants import TCGEN05_AB_CONSUMER_PHASE_MODES
from .._compiler.cute.tcgen05_constants import TCGEN05_AB_CONSUMER_WAIT_MODE_CONFIG_KEY
from .._compiler.cute.tcgen05_constants import TCGEN05_AB_CONSUMER_WAIT_MODE_NORMAL
from .._compiler.cute.tcgen05_constants import TCGEN05_AB_CONSUMER_WAIT_MODES
from .._compiler.cute.tcgen05_constants import (
    TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODE_CONFIG_KEY,
)
from .._compiler.cute.tcgen05_constants import (
    TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODE_NORMAL,
)
from .._compiler.cute.tcgen05_constants import TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODES
from .._compiler.cute.tcgen05_constants import (
    TCGEN05_AB_PRODUCER_ACQUIRE_MODE_CONFIG_KEY,
)
from .._compiler.cute.tcgen05_constants import TCGEN05_AB_PRODUCER_ACQUIRE_MODE_NORMAL
from .._compiler.cute.tcgen05_constants import TCGEN05_AB_PRODUCER_ACQUIRE_MODES
from .._compiler.cute.tcgen05_constants import (
    TCGEN05_AB_PRODUCER_ADVANCE_MODE_CONFIG_KEY,
)
from .._compiler.cute.tcgen05_constants import TCGEN05_AB_PRODUCER_ADVANCE_MODE_NORMAL
from .._compiler.cute.tcgen05_constants import TCGEN05_AB_PRODUCER_ADVANCE_MODES
from .._compiler.cute.tcgen05_constants import (
    TCGEN05_ACC_PRODUCER_ADVANCE_MODE_CONFIG_KEY,
)
from .._compiler.cute.tcgen05_constants import TCGEN05_ACC_PRODUCER_ADVANCE_MODE_NORMAL
from .._compiler.cute.tcgen05_constants import TCGEN05_ACC_PRODUCER_ADVANCE_MODES
from .._compiler.cute.tcgen05_constants import TCGEN05_ACC_PRODUCER_MODE_CONFIG_KEY
from .._compiler.cute.tcgen05_constants import TCGEN05_ACC_PRODUCER_MODE_NORMAL
from .._compiler.cute.tcgen05_constants import TCGEN05_ACC_PRODUCER_MODES
from .._compiler.cute.tcgen05_constants import TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY
from .._compiler.cute.tcgen05_constants import TCGEN05_ACC_WAIT_PLACEMENTS
from .._compiler.cute.tcgen05_constants import TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY
from .._compiler.cute.tcgen05_constants import TCGEN05_C_ACQUIRE_PLACEMENTS
from .._compiler.cute.tcgen05_constants import TCGEN05_C_STORE_MODE_CONFIG_KEY
from .._compiler.cute.tcgen05_constants import TCGEN05_C_STORE_MODE_NORMAL
from .._compiler.cute.tcgen05_constants import TCGEN05_C_STORE_MODES
from .._compiler.cute.tcgen05_constants import (
    TCGEN05_CLUSTER_M2_ONE_CTA_ROLE_LOCAL_CONFIG_KEY,
)
from .._compiler.cute.tcgen05_constants import TCGEN05_CUBIN_LINEINFO_CONFIG_KEY
from .._compiler.cute.tcgen05_constants import (
    TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY,
)
from .._compiler.cute.tcgen05_constants import TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY
from .._compiler.cute.tcgen05_constants import TCGEN05_EPILOGUE_LAYOUTS
from .._compiler.cute.tcgen05_constants import TCGEN05_LARGE_BN_PROOF_BLOCK_SIZES
from .._compiler.cute.tcgen05_constants import TCGEN05_LARGE_BN_PROOF_CLUSTER_M
from .._compiler.cute.tcgen05_constants import TCGEN05_LARGE_BN_PROOF_CONFIG_KEY
from .._compiler.cute.tcgen05_constants import TCGEN05_LARGE_BN_PROOF_PID_TYPE
from .._compiler.cute.tcgen05_constants import TCGEN05_LARGE_BN_PROOF_STAGE_CONFIGS
from .._compiler.cute.tcgen05_constants import TCGEN05_ONE_CTA_MAX_BLOCK_M
from .._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_M
from .._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_N
from .._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_MAX_K_TILES
from .._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_SEED_L2_GROUPING
from .._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_SEED_PID_TYPE
from ..exc import InvalidConfig
from .block_id_sequence import BlockIdSequence
from .block_id_sequence import _BlockIdItem
from .block_id_sequence import _PowerOfTwoBlockIdItem
from .config_fragment import BlockSizeFragment
from .config_fragment import BooleanFragment
from .config_fragment import ConfigSpecFragment
from .config_fragment import EnumFragment
from .config_fragment import IntegerFragment
from .config_fragment import ListOf
from .config_fragment import NumThreadsFragment
from .config_fragment import NumWarpsFragment
from .config_fragment import PermutationFragment
from .config_fragment import PowerOfTwoFragment
from .config_fragment import assert_integer_power_of_two
import helion

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from .._compiler.backend import Backend
    from ..runtime.config import IndexingLiteral
    from ..runtime.config import PidTypeLiteral
    from .config_generation import ConfigGeneration

log = logging.getLogger(__name__)


class TensorNumelConstraint(NamedTuple):
    """Tensor element count must stay within Triton's max numel limit."""

    check_fn: Callable[..., bool]
    block_indices: tuple[int, ...]
    expr_str: str


class Tcgen05ClusterM2SearchConstraints(NamedTuple):
    """Search-only envelope where ``tcgen05_cluster_m=2`` is validated."""

    static_k: int
    max_k_tiles: int


def shrink_block_sizes_for_numel_constraints(
    constraints: list[TensorNumelConstraint],
    block_sizes: list[int],
    min_sizes: list[int],
) -> None:
    """Shrink *block_sizes* in-place so every *constraint* is satisfied.

    Halves the largest involved block size first for balanced tiles.
    Fixed-point loop handles cross-constraint interactions.
    """
    prev = list(block_sizes)
    while True:
        for constraint in constraints:
            while not constraint.check_fn(
                *(block_sizes[i] for i in constraint.block_indices)
            ):
                best_idx: int | None = None
                best_val = -1
                for i in constraint.block_indices:
                    can_halve = block_sizes[i] // 2 >= min_sizes[i]
                    if can_halve and block_sizes[i] > best_val:
                        best_val = block_sizes[i]
                        best_idx = i
                if best_idx is None:
                    log.warning(
                        "tensor numel constraint unsatisfiable at minimum "
                        "block sizes: %s",
                        constraint.expr_str,
                    )
                    break
                block_sizes[best_idx] //= 2
        if block_sizes == prev:
            break
        prev = list(block_sizes)


DEFAULT_NUM_WARPS = 4
DEFAULT_NUM_STAGES = 1

# Base backend tunable keys (public)
_BASE_BACKEND_TUNABLE_KEYS: frozenset[str] = frozenset(
    {
        "waves_per_eu",
        "matrix_instr_nonkdim",
        "num_ctas",
        "occupancy",
        "pallas_loop_type",
        "pallas_pre_broadcast",
        "tcgen05_cluster_m",
        "tcgen05_ab_stages",
        "tcgen05_acc_stages",
        "tcgen05_c_stages",
        TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY,
        TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY,
        TCGEN05_C_STORE_MODE_CONFIG_KEY,
        "tcgen05_num_epi_warps",
    }
)
_BACKEND_DIAGNOSTIC_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        TCGEN05_AB_CONSUMER_PHASE_MODE_CONFIG_KEY,
        TCGEN05_AB_CONSUMER_WAIT_MODE_CONFIG_KEY,
        TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODE_CONFIG_KEY,
        TCGEN05_AB_PRODUCER_ACQUIRE_MODE_CONFIG_KEY,
        TCGEN05_AB_PRODUCER_ADVANCE_MODE_CONFIG_KEY,
        TCGEN05_ACC_PRODUCER_ADVANCE_MODE_CONFIG_KEY,
        TCGEN05_ACC_PRODUCER_MODE_CONFIG_KEY,
        TCGEN05_CLUSTER_M2_ONE_CTA_ROLE_LOCAL_CONFIG_KEY,
        TCGEN05_CUBIN_LINEINFO_CONFIG_KEY,
        TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY,
        TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY,
        TCGEN05_LARGE_BN_PROOF_CONFIG_KEY,
    }
)


def _get_backend_tunable_keys() -> frozenset[str]:
    """Get all backend tunable keys, including FB-private ones if available."""
    try:
        from ..fb.mtia_tunables import MTIA_TUNABLES  # pyrefly: ignore [missing-import]

        return _BASE_BACKEND_TUNABLE_KEYS | frozenset(MTIA_TUNABLES)
    except ImportError:
        return _BASE_BACKEND_TUNABLE_KEYS


BACKEND_TUNABLE_KEYS: frozenset[str] = _get_backend_tunable_keys()
# All config keys whose support depends on the backend.  The base Backend
# class rejects these by default; each backend subclass opts in selectively.
BACKEND_SPECIFIC_KEYS: frozenset[str] = (
    BACKEND_TUNABLE_KEYS
    | _BACKEND_DIAGNOSTIC_CONFIG_KEYS
    | {
        "num_threads",
        "pallas_loop_type",
        "pallas_pre_broadcast",
    }
)
VALID_KEYS: frozenset[str] = frozenset(
    [
        "block_sizes",
        "num_threads",
        "loop_orders",
        "l2_groupings",
        "reduction_loops",
        "flatten_loops",
        "range_unroll_factors",
        "range_warp_specializes",
        "range_num_stages",
        "range_multi_buffers",
        "range_flattens",
        "static_ranges",
        "num_warps",
        "num_stages",
        "pid_type",
        "num_sm_multiplier",
        "maxnreg",
        "indexing",
        "atomic_indexing",
        "load_eviction_policies",
        "pallas_loop_type",
        "pallas_pre_broadcast",
        *BACKEND_TUNABLE_KEYS,
        "advanced_controls_file",
        "epilogue_subtile",
        *_BACKEND_DIAGNOSTIC_CONFIG_KEYS,
    ]
)
VALID_PALLAS_LOOP_TYPES = ("emit_pipeline", "unroll", "fori_loop")
VALID_PID_TYPES = (
    "flat",
    "xyz",
    "persistent_blocked",
    "persistent_interleaved",
)
MIN_NUM_SM_MULTIPLIER = 1
MAX_NUM_SM_MULTIPLIER = 128
DEFAULT_NUM_SM_MULTIPLIER = 1
EPILOGUE_SUBTILE_EXTENDED_CHOICES = (None, 2, 4)
EPILOGUE_SUBTILE_DEFAULT_CHOICES = (None, 2)
EPILOGUE_SUBTILE_MIN_K_HINT = 1024
EPILOGUE_SUBTILE_MIN_K_HINT_EXTENDED = 16384
# maxnreg values: None means no limit, otherwise limit to this many registers per thread
# Lower values allow higher occupancy but may hurt performance for register-heavy kernels
VALID_MAXNREG = (None, 32, 64, 128, 256)
DEFAULT_MAXNREG = None
CUTE_TCGEN05_TUNABLE_KEYS: tuple[str, ...] = (
    "tcgen05_cluster_m",
    "tcgen05_ab_stages",
    "tcgen05_acc_stages",
    "tcgen05_c_stages",
    TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY,
    TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY,
    TCGEN05_C_STORE_MODE_CONFIG_KEY,
    "tcgen05_num_epi_warps",
)
_CUTE_IMPLICIT_DEFAULT_KEYS: frozenset[str] = frozenset(
    {
        "loop_orders",
        "flatten_loops",
        "l2_groupings",
        "range_unroll_factors",
        "range_warp_specializes",
        "range_num_stages",
        "range_multi_buffers",
        "range_flattens",
        "static_ranges",
        "load_eviction_policies",
        "indexing",
        "atomic_indexing",
        "num_warps",
        "num_stages",
        "pid_type",
        "num_sm_multiplier",
        "maxnreg",
    }
)


def _epilogue_subtile_autotune_arch() -> tuple[int, int] | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability(torch.cuda.current_device())


# For tileir backend or AMD ROCM, eviction policies are not supported.
# Keep this uncached: some tests patch the AMD capability helper, and caching
# only on backend name can poison later Triton ConfigSpec construction inside
# the same worker process.
def get_valid_eviction_policies(backend_name: str) -> tuple[str, ...]:
    if backend_name == "triton" and not supports_amd_cdna_tunables():
        return ("", "first", "last")
    return ("",)


class ConfigSpec:
    def __init__(
        self,
        *,
        backend: Backend,
        user_defined_tunables: Mapping[str, ConfigSpecFragment] | None = None,
    ) -> None:
        self.backend = backend
        self.backend_name = backend.name
        self.max_reduction_threads = backend.max_reduction_threads()
        self.user_defined_tunables = (
            {} if user_defined_tunables is None else dict(user_defined_tunables)
        )

        self.block_sizes: BlockIdSequence[BlockSizeSpec] = BlockIdSequence()
        self.num_threads: BlockIdSequence[NumThreadsSpec] = BlockIdSequence()
        self.loop_orders: BlockIdSequence[LoopOrderSpec] = BlockIdSequence()
        self.l2_groupings: BlockIdSequence[L2GroupingSpec] = BlockIdSequence()
        self.flatten_loops: BlockIdSequence[FlattenLoopSpec] = BlockIdSequence()
        self.reduction_loops: BlockIdSequence[ReductionLoopSpec] = BlockIdSequence()
        self.range_unroll_factors: BlockIdSequence[RangeUnrollFactorSpec] = (
            BlockIdSequence()
        )
        self.range_warp_specialize: BlockIdSequence[RangeWarpSpecializeSpec] = (
            BlockIdSequence()
        )
        self.range_num_stages: BlockIdSequence[RangeNumStagesSpec] = BlockIdSequence()
        self.range_multi_buffers: BlockIdSequence[RangeMultiBufferSpec] = (
            BlockIdSequence()
        )
        self.range_flattens: BlockIdSequence[RangeFlattenSpec] = BlockIdSequence()
        self.static_ranges: BlockIdSequence[StaticRangeSpec] = BlockIdSequence()

        self.allowed_pid_types: tuple[PidTypeLiteral, ...] = tuple(VALID_PID_TYPES)
        self.max_num_sm_multiplier: int = MAX_NUM_SM_MULTIPLIER
        self.grid_block_ids: list[int] = []
        self.tensor_numel_constraints: list[TensorNumelConstraint] = []
        self.load_eviction_policies = ListOf(
            EnumFragment(choices=get_valid_eviction_policies(self.backend_name)),
            length=0,
        )
        self.indexing = ListOf(
            EnumFragment(choices=self.valid_indexing_types()),
            length=0,
        )
        self.atomic_indexing = ListOf(
            EnumFragment(choices=self.valid_atomic_indexing_types()),
            length=0,
        )
        self.epilogue_subtile_candidate_enabled: bool = False
        self.epilogue_subtile_autotune_choices: tuple[int | None, ...] | None = None
        self.epilogue_subtile_k_hint: int = 0
        self.has_pallas_inner_loops: bool = False
        self.has_symbolic_or_data_dependent_bounds: bool = False
        self.cute_tcgen05_search_enabled: bool = False
        # Allowed values of tcgen05_cluster_m the autotuner is allowed to
        # *search* over. None means "use the default set defined by
        # _tcgen05_optional_fragments". This is consulted by _flat_fields()
        # when constructing the search space and is independent of which
        # values are *legal* in a user-supplied helion.Config (the latter
        # always accepts the full set of legal values).
        self._tcgen05_cluster_m_search_choices: tuple[int, ...] | None = None
        self._tcgen05_cluster_m2_search_constraints: (
            Tcgen05ClusterM2SearchConstraints | None
        ) = None
        # Allowed values of tcgen05_num_epi_warps the autotuner is allowed
        # to *search* over. ``None`` means "use the default IntegerFragment
        # range defined by _tcgen05_optional_fragments". This is consulted by
        # _flat_fields() when constructing the search space.
        self._tcgen05_num_epi_warps_search_choices: tuple[int, ...] | None = None
        # Allowed values of tcgen05_num_epi_warps a *user-supplied*
        # ``helion.Config`` is permitted to specify. ``None`` means "use
        # the default IntegerFragment range" (i.e. accept anything in
        # [1, 4]). Unlike the cluster_m narrowing, the matmul path
        # tightens validation here too because num_epi_warps != 4
        # currently produces *silent wrong output* on B200 — there is
        # no loud crash to alert a user who bypasses autotune via an
        # explicit config, so normalize() must reject the unsafe values.
        self._tcgen05_num_epi_warps_validation_choices: tuple[int, ...] | None = None
        self.store_indices: list[int] = []
        self.backend_tunable_fragments = self.backend.tunable_fragments()
        unknown_tunables = set(self.backend_tunable_fragments) - BACKEND_TUNABLE_KEYS
        if unknown_tunables:
            raise RuntimeError(
                f"Backend {self.backend_name!r} returned unknown tunables: {sorted(unknown_tunables)!r}"
            )

    def _should_keep_epilogue_subtile_for_autotune(self) -> bool:
        if self.epilogue_subtile_autotune_choices is None:
            return False
        return supports_tensor_descriptor()

    def fix_epilogue_subtile_store_indexing(self, config: dict[str, object]) -> None:
        """Force subtiled store indexing to tensor_descriptor for correctness."""
        if (
            not self.epilogue_subtile_candidate_enabled
            or "epilogue_subtile" not in config
        ):
            return
        indexing = config.get("indexing")
        if isinstance(indexing, list):
            for i in self.store_indices:
                indexing[i] = "tensor_descriptor"

    @staticmethod
    def _infer_epilogue_subtile_k_hint(args: Sequence[object]) -> int:
        def _as_concrete_dim(dim: object) -> int | None:
            return dim if type(dim) is int else None

        tensor_args = [
            arg for arg in args if isinstance(arg, torch.Tensor) and arg.ndim >= 2
        ]
        best = 0
        for lhs, rhs in itertools.combinations(tensor_args, 2):
            candidates: list[int] = []
            lhs_last = _as_concrete_dim(lhs.shape[-1])
            lhs_prev = _as_concrete_dim(lhs.shape[-2])
            rhs_last = _as_concrete_dim(rhs.shape[-1])
            rhs_prev = _as_concrete_dim(rhs.shape[-2])
            if lhs_last is not None and rhs_prev is not None and lhs_last == rhs_prev:
                candidates.append(lhs_last)
            if lhs_prev is not None and rhs_last is not None and lhs_prev == rhs_last:
                candidates.append(lhs_prev)
            if candidates:
                best = max(best, *candidates)
        return best

    def configure_epilogue_subtile_autotune(self, args: Sequence[object]) -> None:
        self.epilogue_subtile_k_hint = self._infer_epilogue_subtile_k_hint(args)
        arch = _epilogue_subtile_autotune_arch()
        if arch is None:
            self.epilogue_subtile_autotune_choices = None
            return

        if arch >= (10, 0):
            arch_enabled = (
                self.epilogue_subtile_candidate_enabled and supports_tensor_descriptor()
            )
        else:
            arch_enabled = False

        enabled = (
            arch_enabled and self.epilogue_subtile_k_hint >= EPILOGUE_SUBTILE_MIN_K_HINT
        )
        if not enabled:
            self.epilogue_subtile_autotune_choices = None
        elif (
            arch >= (10, 0)
            and self.epilogue_subtile_k_hint >= EPILOGUE_SUBTILE_MIN_K_HINT_EXTENDED
        ):
            self.epilogue_subtile_autotune_choices = EPILOGUE_SUBTILE_EXTENDED_CHOICES
        else:
            self.epilogue_subtile_autotune_choices = EPILOGUE_SUBTILE_DEFAULT_CHOICES

    def valid_indexing_types(self) -> tuple[IndexingLiteral, ...]:
        if supports_tensor_descriptor():
            return ("pointer", "tensor_descriptor")
        if not self.backend.supports_block_ptr_indexing():
            return ("pointer",)
        return ("pointer", "block_ptr")

    def valid_atomic_indexing_types(self) -> tuple[IndexingLiteral, ...]:
        """Atomic ops only support pointer and tensor_descriptor (no block_ptr)."""
        if supports_tensor_descriptor():
            return ("pointer", "tensor_descriptor")
        return ("pointer",)

    def _remove_duplicates(self) -> None:
        self.num_threads._remove_duplicates()
        self.loop_orders._remove_duplicates()
        self.l2_groupings._remove_duplicates()
        self.flatten_loops._remove_duplicates()
        self.range_unroll_factors._remove_duplicates()
        self.range_warp_specialize._remove_duplicates()
        self.range_num_stages._remove_duplicates()
        self.range_multi_buffers._remove_duplicates()
        self.range_flattens._remove_duplicates()
        self.static_ranges._remove_duplicates()

    def disallow_pid_type(self, pid_type: PidTypeLiteral) -> None:
        """Disallow a pid_type from being used in the config."""

        self.allowed_pid_types = tuple(
            [x for x in self.allowed_pid_types if x != pid_type]
        )
        assert self.allowed_pid_types

    def restrict_tcgen05_cluster_m_search(self, choices: tuple[int, ...]) -> None:
        """Narrow the autotune search over tcgen05_cluster_m to *choices*.

        Does not affect what values are legal in user-supplied configs;
        normalize() validates against the full set of legal values returned by
        _tcgen05_optional_fragments(for_search=False).
        """
        assert choices, "tcgen05_cluster_m search must allow at least one value"
        self._tcgen05_cluster_m_search_choices = choices
        if 2 not in choices:
            self._tcgen05_cluster_m2_search_constraints = None

    def allow_tcgen05_cluster_m2_search(
        self,
        *,
        static_k: int,
        max_k_tiles: int = TCGEN05_TWO_CTA_MAX_K_TILES,
    ) -> None:
        """Allow validated CtaGroup.TWO candidates in the autotune search.

        The search space cannot express cross-field constraints directly.
        This helper exposes ``tcgen05_cluster_m=2`` but records enough context
        for search-time normalization to project ``cluster_m=2`` products onto
        the validated seed pid order + ``256x256`` CtaGroup.TWO block
        tile within the K-tile cap. Problems outside this envelope keep
        ``tcgen05_cluster_m`` narrowed to ``(1,)``.
        """
        assert static_k > 0, "static_k is required for cluster_m=2 K-cap checks"
        assert max_k_tiles > 0, "cluster_m=2 max K tiles must be positive"
        self._tcgen05_cluster_m2_search_constraints = Tcgen05ClusterM2SearchConstraints(
            static_k=static_k,
            max_k_tiles=max_k_tiles,
        )
        # ``restrict_tcgen05_cluster_m_search`` intentionally clears the
        # constraints whenever 2 is absent; set them immediately before
        # reopening the search choices.
        self.restrict_tcgen05_cluster_m_search((1, 2))

    def _tcgen05_cluster_m2_seed_config(self) -> helion.Config | None:
        constraints = self._tcgen05_cluster_m2_search_constraints
        if (
            constraints is None
            or TCGEN05_TWO_CTA_SEED_PID_TYPE not in self.allowed_pid_types
        ):
            return None
        if len(self.block_sizes) != 3:
            return None

        bm_fragment = cast("BlockSizeFragment", self.block_sizes[0]._fragment(self))
        bn_fragment = cast("BlockSizeFragment", self.block_sizes[1]._fragment(self))
        bk_fragment = cast("BlockSizeFragment", self.block_sizes[2]._fragment(self))
        if not (
            bm_fragment.low <= TCGEN05_TWO_CTA_BLOCK_M <= bm_fragment.high
            and bn_fragment.low <= TCGEN05_TWO_CTA_BLOCK_N <= bn_fragment.high
        ):
            return None

        bk = bk_fragment.high
        while bk >= bk_fragment.low:
            if self._tcgen05_cluster_m2_bk_is_valid(bk, constraints):
                seed_config: dict[str, Any] = {
                    "block_sizes": [
                        TCGEN05_TWO_CTA_BLOCK_M,
                        TCGEN05_TWO_CTA_BLOCK_N,
                        bk,
                    ],
                    "l2_groupings": [TCGEN05_TWO_CTA_SEED_L2_GROUPING],
                    "pid_type": TCGEN05_TWO_CTA_SEED_PID_TYPE,
                    "tcgen05_cluster_m": 2,
                    # Matches the validated tcgen05 search restriction.
                    "tcgen05_num_epi_warps": 4,
                }
                # Pure matmul has exactly the A/B/C indexing slots. Fused
                # epilogues add more memory ops, so leave those seeds to the
                # spec default rather than constructing a partial list.
                if self.indexing.length == 3:
                    seed_config["indexing"] = [
                        "tensor_descriptor",
                        "tensor_descriptor",
                        "tensor_descriptor",
                    ]
                return helion.Config(**seed_config)
            bk //= 2
        return None

    @staticmethod
    def _tcgen05_cluster_m2_bk_is_valid(
        bk: int, constraints: Tcgen05ClusterM2SearchConstraints
    ) -> bool:
        return constraints.static_k % bk == 0 and (
            constraints.static_k // bk <= constraints.max_k_tiles
        )

    def autotune_seed_configs(self) -> list[helion.Config]:
        """Return validated extra configs that should be benchmarked early."""
        cluster_m2_seed = self._tcgen05_cluster_m2_seed_config()
        if cluster_m2_seed is None:
            return []
        return [cluster_m2_seed]

    def _fix_tcgen05_cluster_m2_search_config(self, config: dict[str, object]) -> None:
        """Canonicalize unvalidated search-only ``cluster_m=2`` products."""
        if not (
            self.cute_tcgen05_search_enabled and config.get("tcgen05_cluster_m") == 2
        ):
            return
        constraints = self._tcgen05_cluster_m2_search_constraints
        if (
            constraints is None
            or TCGEN05_TWO_CTA_SEED_PID_TYPE not in self.allowed_pid_types
        ):
            config["tcgen05_cluster_m"] = 1
            return
        block_sizes = config.get("block_sizes")
        if not isinstance(block_sizes, list) or len(block_sizes) < 3:
            config["tcgen05_cluster_m"] = 1
            return
        bk = block_sizes[2]
        if not isinstance(bk, int) or isinstance(bk, bool):
            config["tcgen05_cluster_m"] = 1
            return
        if not self._tcgen05_cluster_m2_bk_is_valid(bk, constraints):
            config["tcgen05_cluster_m"] = 1
            return
        config["pid_type"] = TCGEN05_TWO_CTA_SEED_PID_TYPE
        block_sizes[0] = TCGEN05_TWO_CTA_BLOCK_M
        block_sizes[1] = TCGEN05_TWO_CTA_BLOCK_N

    def _fix_tcgen05_cluster_m1_persistent_search_config(
        self, config: dict[str, object]
    ) -> None:
        """Keep search-only CtaGroup.ONE persistent configs on tcgen05 tiles."""
        if not (
            self.cute_tcgen05_search_enabled
            and config.get("tcgen05_cluster_m", 1) == 1
            and config.get("pid_type")
            in {"persistent_blocked", "persistent_interleaved"}
        ):
            return
        block_sizes = config.get("block_sizes")
        if not isinstance(block_sizes, list) or not block_sizes:
            return
        bm = block_sizes[0]
        if isinstance(bm, int) and not isinstance(bm, bool):
            block_sizes[0] = min(bm, TCGEN05_ONE_CTA_MAX_BLOCK_M)

    def restrict_tcgen05_num_epi_warps_search(self, choices: tuple[int, ...]) -> None:
        """Narrow the autotune search over tcgen05_num_epi_warps to *choices*.

        Does not affect what values are legal in user-supplied configs.
        See ``restrict_tcgen05_num_epi_warps_validation`` for the
        complementary helper that tightens normalize().
        """
        assert choices, "tcgen05_num_epi_warps search must allow at least one value"
        self._tcgen05_num_epi_warps_search_choices = choices

    def restrict_tcgen05_num_epi_warps_validation(
        self, choices: tuple[int, ...]
    ) -> None:
        """Restrict the *legal* values of tcgen05_num_epi_warps in
        user-supplied configs to *choices*.

        Unlike the search-only narrowing helpers, this also tightens
        what ``normalize()`` accepts: a user-supplied
        ``helion.Config(tcgen05_num_epi_warps=N)`` with N not in
        *choices* will raise ``InvalidConfig``. Used by the BF16/FP16
        matmul path because num_epi_warps != 4 currently produces
        silent wrong output (no loud crash to alert a user who bypasses
        autotune via an explicit config).
        """
        assert choices, "tcgen05_num_epi_warps validation must allow at least one value"
        self._tcgen05_num_epi_warps_validation_choices = choices

    def narrow_tcgen05_autotune_to_validated_configs(
        self,
        *,
        allow_persistent_pid_types: bool = False,
        allow_cluster_m2_search: bool = False,
        cluster_m2_static_k: int | None = None,
    ) -> None:
        """Narrow the tcgen05 autotune search to combinations validated on B200.

        Three structural limitations bound the safe autotune space today.
        Each narrowing is documented inline so the rationale stays close
        to the call. The cluster_m and pid_type restrictions are
        *search-only* — user-supplied ``helion.Config(...)`` values are
        still validated against the full set of legal options (see
        ``_tcgen05_optional_fragments(for_search=False)``). The
        num_epi_warps restriction additionally tightens validation
        because the underlying failure mode is silent wrong output.

        * **Persistent autotune gating.** Static full-tile single-root
          tcgen05 persistent kernels now use role-local loops and have
          multi-tile runtime coverage for ``cluster_m=1``. Callers may pass
          ``allow_persistent_pid_types=True`` only after proving every
          candidate M/N/K block size in the search space divides the static
          problem extent. Otherwise, drop ``persistent_blocked`` and
          ``persistent_interleaved`` from the autotune pid_type search so
          partial-tile, multi-root, or otherwise unvalidated persistent
          configs cannot be sampled. Explicit unsafe user configs still go
          through the host-side guard
          (``Tcgen05PersistentProgramIDs._emit_host_multi_tile_guard``),
          which raises a loud ``RuntimeError``.

        * **2-CTA cluster.** ``cluster_m=2`` is searched only when the
          caller proves the static problem can form validated role-local
          CtaGroup.TWO candidates. The search view may expose ``(1, 2)``,
          but search-time normalization projects ``cluster_m=2`` products
          onto the validated seed pid order + ``256x256`` tile shape
          with ``K / bk`` at or below the validated K-tile cap. CUDA launch
          failures are loud, so user-supplied ``cluster_m=2`` is still legal
          and handled by the runtime guard when outside the validated
          envelope.

        * **Multi-warp epilogue.** Only ``tcgen05_num_epi_warps=4`` is
          validated correct on B200 today. Direct verification at
          ``num_epi_warps ∈ {1, 2}`` shows the tcgen05 SIMT-store
          epilogue miscompiles (~75% of elements diverge from the
          torch reference at M=N=K=256 bf16); ``num_epi_warps=3`` is
          not separately validated and is treated as unsafe by
          extension. Narrow both the autotune search *and* the
          validation accept-set to ``(4,)`` so the autotuner cannot
          converge on a wrong-output config and a user-supplied
          ``helion.Config(tcgen05_num_epi_warps=N!=4)`` raises
          ``InvalidConfig`` rather than silently miscomputing. The
          underlying correctness bug is what blocks
          ``num_epi_warps != 4`` from being usable; fixing it is the
          actual unblocker for item 2 (multi-warp epilogue with
          c_pipeline SMEM ring + TMA bulk store).
        """
        if not allow_persistent_pid_types:
            self.disallow_pid_type("persistent_blocked")
            self.disallow_pid_type("persistent_interleaved")
        if allow_cluster_m2_search:
            assert allow_persistent_pid_types, (
                "cluster_m=2 search requires persistent pid types"
            )
            assert cluster_m2_static_k is not None, (
                "cluster_m=2 search requires a static K extent"
            )
            self.allow_tcgen05_cluster_m2_search(static_k=cluster_m2_static_k)
        else:
            self.restrict_tcgen05_cluster_m_search((1,))
        self.restrict_tcgen05_num_epi_warps_search((4,))
        self.restrict_tcgen05_num_epi_warps_validation((4,))

    def supports_config_key(self, key: str) -> bool:
        return self.backend.supports_config_key(key)

    def supported_config_keys(self) -> frozenset[str]:
        return frozenset(key for key in VALID_KEYS if self.supports_config_key(key))

    def _default_num_stages(self) -> int:
        return DEFAULT_NUM_STAGES

    def _num_stages_fragment(self) -> ConfigSpecFragment:
        if self.backend_name == "tileir":
            return EnumFragment(choices=tuple(range(1, 11)))
        if supports_amd_cdna_tunables():
            return IntegerFragment(1, 4, self._default_num_stages())
        if self.backend_name == "metal":
            return IntegerFragment(1, 1, 1)
        return IntegerFragment(1, 8, self._default_num_stages())

    def _tcgen05_optional_fragments(
        self, *, for_search: bool = False
    ) -> dict[str, ConfigSpecFragment]:
        # The "validation" view (default) defines what user-supplied
        # configs are allowed to specify. By default it keeps the full
        # set of legal values so e.g. ``helion.Config(...,
        # tcgen05_cluster_m=2)`` round-trips through normalize()
        # unchanged. The matmul path additionally calls
        # ``restrict_tcgen05_num_epi_warps_validation`` to *tighten*
        # this view for ``tcgen05_num_epi_warps`` because non-4 values
        # silently miscompute on B200 (see
        # ``narrow_tcgen05_autotune_to_validated_configs``); without
        # that tightening, an explicit user config bypassing autotune
        # would produce wrong output with no diagnostic.
        #
        # The "search" view (for_search=True) is consulted by _flat_fields()
        # to build the autotune search space, and may be narrower because
        # some legal values are known to regress perf or crash at runtime.
        # ``matmul_ops.enforce_dot_requirements`` calls
        # ``narrow_tcgen05_autotune_to_validated_configs`` for the BF16/FP16
        # matmul path so the autotuner does not pick a regressing /
        # crashing config and tear down the GPU context for the whole
        # tuning run. See ``narrow_tcgen05_autotune_to_validated_configs``
        # for the per-knob rationale.
        if for_search and self._tcgen05_cluster_m_search_choices is not None:
            cluster_m_choices = self._tcgen05_cluster_m_search_choices
        else:
            cluster_m_choices = (1, 2)
        if for_search and self._tcgen05_num_epi_warps_search_choices is not None:
            num_epi_warps_fragment: ConfigSpecFragment = EnumFragment(
                self._tcgen05_num_epi_warps_search_choices
            )
        elif (
            not for_search
            and self._tcgen05_num_epi_warps_validation_choices is not None
        ):
            num_epi_warps_fragment = EnumFragment(
                self._tcgen05_num_epi_warps_validation_choices
            )
        else:
            num_epi_warps_fragment = IntegerFragment(1, 4, 4)
        return {
            "tcgen05_cluster_m": EnumFragment(cluster_m_choices),
            "tcgen05_ab_stages": IntegerFragment(1, 2, 2),
            "tcgen05_acc_stages": IntegerFragment(1, 2, 2),
            "tcgen05_c_stages": EnumFragment((2, 4)),
            "tcgen05_num_epi_warps": num_epi_warps_fragment,
        }

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

    def unsupported_config_keys(self, config: Mapping[str, object]) -> list[str]:
        return sorted(
            key
            for key in config
            if key in VALID_KEYS and not self.supports_config_key(key)
        )

    def is_supported_config(self, config: Mapping[str, object]) -> bool:
        return not self.unsupported_config_keys(config)

    def normalize(
        self, config: helion.Config | dict[str, object], *, _fix_invalid: bool = False
    ) -> None:
        """Normalize the config to match the block_sizes and validate the config.

        Args:
            config: The config to normalize (modified in place).
            _fix_invalid: If True, silently fix invalid combinations instead of raising
                errors. Used internally during autotuning config generation.
        """
        if isinstance(config, helion.Config):
            self.normalize(config.config, _fix_invalid=_fix_invalid)
            return

        for name in (
            "block_size",
            "loop_order",
            "reduction_loop",
            "l2_grouping",
            "flatten_loop",
            "range_unroll_factor",
            "range_warp_specialize",
            "range_num_stage",
            "range_multi_buffer",
            "range_flatten",
            "static_range",
        ):
            if name in config:
                names = f"{name}s"
                if names in config:
                    raise InvalidConfig(f"Cannot specify both {name} and {names}")
                value = config.pop(name)
                if name == "reduction_loop" and len(self.reduction_loops) > 1:
                    # Apply the same reduction_loop setting to every
                    # reduction dimension so a single scalar value works
                    # when multiple dims can be rolled.
                    config[names] = [value for _ in range(len(self.reduction_loops))]
                else:
                    config[names] = [value]

        if unsupported := self.unsupported_config_keys(config):
            # Separate backend-specific keys (e.g. AMD tunables, TileIR tunables)
            # from common keys (e.g. num_warps, num_stages, indexing).
            # Backend-specific keys should raise errors; common keys are
            # silently stripped so configs are portable across backends.
            backend_specific = [k for k in unsupported if k in BACKEND_SPECIFIC_KEYS]
            common = [k for k in unsupported if k not in BACKEND_SPECIFIC_KEYS]
            for key in common:
                config.pop(key, None)
            if backend_specific:
                if _fix_invalid:
                    for key in backend_specific:
                        config.pop(key, None)
                else:
                    raise InvalidConfig(
                        f"Unsupported config keys for backend {self.backend_name!r}: {backend_specific}"
                    )
        provided_keys = set(config)

        for name, mapping, flatten in [
            ("block_sizes", self.block_sizes, True),
            ("num_threads", self.num_threads, True),
            ("flatten_loops", self.flatten_loops, True),
            ("l2_groupings", self.l2_groupings, True),
            ("loop_orders", self.loop_orders, False),
            ("reduction_loops", self.reduction_loops, True),
            ("range_unroll_factors", self.range_unroll_factors, True),
            ("range_warp_specializes", self.range_warp_specialize, True),
            ("range_num_stages", self.range_num_stages, True),
            ("range_multi_buffers", self.range_multi_buffers, True),
            ("range_flattens", self.range_flattens, True),
            ("static_ranges", self.static_ranges, True),
        ]:
            if not self.supports_config_key(name):
                if name in config:
                    raise InvalidConfig(
                        f"{name} is not supported on backend {self.backend_name!r}"
                    )
                config.pop(name, None)
                continue
            config[name] = mapping._normalize(
                name, config.get(name, ()), flatten=flatten
            )
        if self.supports_config_key("num_threads"):
            num_threads = cast("list[int]", config.get("num_threads", []))
            if all(value == 0 for value in num_threads):
                config.pop("num_threads", None)
        else:
            config.pop("num_threads", None)

        # Cap reduction loops at the backend's max reduction thread count
        if self.max_reduction_threads is not None and self.reduction_loops:
            max_threads = self.max_reduction_threads
            reduction_loops = config.get("reduction_loops", [])
            if isinstance(reduction_loops, list):
                new_loops = list(reduction_loops)
                changed = False
                for i, spec in enumerate(self.reduction_loops):
                    if i >= len(new_loops):
                        break
                    if (new_loops[i] is None and spec.size_hint > max_threads) or (
                        new_loops[i] is not None and new_loops[i] > max_threads
                    ):
                        new_loops[i] = max_threads
                        changed = True
                if changed:
                    config["reduction_loops"] = new_loops

        # Disable range_* configs for static ranges
        static_range_block_ids = [
            block_id
            for block_id in self.static_ranges.valid_block_ids()
            if self.static_ranges.config_get(
                cast("list[bool]", config.get("static_ranges", [])),
                block_id,
            )
        ]
        if static_range_block_ids:
            for name, mapping in (
                ("range_unroll_factors", self.range_unroll_factors),
                ("range_warp_specializes", self.range_warp_specialize),
                ("range_num_stages", self.range_num_stages),
                ("range_multi_buffers", self.range_multi_buffers),
                ("range_flattens", self.range_flattens),
            ):
                config[name] = mapping._reset_config_to_default(
                    name, config.get(name, ()), block_ids=static_range_block_ids
                )

        for name in (
            "loop_orders",
            "l2_groupings",
            "flatten_loops",
            "reduction_loops",
            "range_unroll_factors",
            "range_warp_specializes",
            "range_num_stages",
            "range_multi_buffers",
            "range_flattens",
            "static_ranges",
            "load_eviction_policies",
            "indexing",
            "atomic_indexing",
        ):
            if not config.get(name):
                config.pop(name, None)

        # Remove unsupported keys before setting defaults
        for name in (
            "num_warps",
            "num_stages",
            "load_eviction_policies",
            "indexing",
            "atomic_indexing",
            "pid_type",
            "num_sm_multiplier",
            "maxnreg",
        ):
            if not self.supports_config_key(name):
                config.pop(name, None)

        if self.supports_config_key("num_warps"):
            config.setdefault("num_warps", DEFAULT_NUM_WARPS)
        if self.supports_config_key("num_stages"):
            config.setdefault("num_stages", self._default_num_stages())
        if self.supports_config_key("load_eviction_policies"):
            config.setdefault(
                "load_eviction_policies", self.load_eviction_policies.default()
            )
        if self.supports_config_key("indexing"):
            config.setdefault("indexing", self.indexing.default())
        if self.supports_config_key("atomic_indexing"):
            config.setdefault("atomic_indexing", self.atomic_indexing.default())
        for key, fragment in self.backend_tunable_fragments.items():
            config.setdefault(key, fragment.default())
        tcgen05_optional_fragments = self._tcgen05_optional_fragments()
        tcgen05_optional_search_fragments = self._tcgen05_optional_fragments(
            for_search=True
        )
        if self.cute_tcgen05_search_enabled:
            for key, fragment in tcgen05_optional_fragments.items():
                if key in config:
                    config[key] = self._validate_optional_fragment_value(
                        key, fragment, config[key]
                    )
                else:
                    # Fill missing keys from the search-view default so a
                    # minimized config (Config.minimize strips values that
                    # match default_config(), which is built from the
                    # search view via _flat_fields) round-trips back to
                    # the same effective config when re-normalized. The
                    # validation view above is still used to validate
                    # user-supplied values against the full legal range.
                    config[key] = tcgen05_optional_search_fragments[key].default()
        else:
            for key in tcgen05_optional_fragments:
                if key not in config:
                    continue
                if _fix_invalid:
                    config.pop(key, None)
                else:
                    raise InvalidConfig(
                        f"{key} is only supported for tcgen05-enabled CuTe matmul kernels"
                    )
        if TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY in config:
            if not self.cute_tcgen05_search_enabled:
                if _fix_invalid:
                    config.pop(TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY, None)
                else:
                    raise InvalidConfig(
                        f"{TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY} is only "
                        "supported for tcgen05-enabled CuTe matmul kernels"
                    )
            elif (
                config[TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY]
                not in TCGEN05_C_ACQUIRE_PLACEMENTS
            ):
                if _fix_invalid:
                    config.pop(TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY, None)
                else:
                    raise InvalidConfig(
                        f"{TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY} must be one "
                        f"of {TCGEN05_C_ACQUIRE_PLACEMENTS!r}, got "
                        f"{config[TCGEN05_C_ACQUIRE_PLACEMENT_CONFIG_KEY]!r}"
                    )
        if TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY in config:
            if not self.cute_tcgen05_search_enabled:
                if _fix_invalid:
                    config.pop(TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY, None)
                else:
                    raise InvalidConfig(
                        f"{TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY} is only "
                        "supported for tcgen05-enabled CuTe matmul kernels"
                    )
            elif (
                config[TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY]
                not in TCGEN05_ACC_WAIT_PLACEMENTS
            ):
                if _fix_invalid:
                    config.pop(TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY, None)
                else:
                    raise InvalidConfig(
                        f"{TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY} must be one "
                        f"of {TCGEN05_ACC_WAIT_PLACEMENTS!r}, got "
                        f"{config[TCGEN05_ACC_WAIT_PLACEMENT_CONFIG_KEY]!r}"
                    )
        if TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY in config:
            if not self.cute_tcgen05_search_enabled:
                if _fix_invalid:
                    config.pop(TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY, None)
                else:
                    raise InvalidConfig(
                        f"{TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY} is only "
                        "supported for tcgen05-enabled CuTe matmul kernels"
                    )
            elif not isinstance(
                config[TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY],
                bool,
            ):
                if _fix_invalid:
                    config.pop(TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY, None)
                else:
                    raise InvalidConfig(
                        f"{TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY} must be "
                        "a boolean"
                    )

        def validate_tcgen05_diagnostic_mode(
            key: str,
            modes: tuple[str, ...],
            normal_mode: str,
        ) -> None:
            if key not in config:
                return
            if not self.cute_tcgen05_search_enabled:
                if _fix_invalid:
                    config.pop(key, None)
                else:
                    raise InvalidConfig(
                        f"{key} is only supported for tcgen05-enabled CuTe "
                        "matmul kernels"
                    )
            elif config[key] not in modes:
                if _fix_invalid:
                    config.pop(key, None)
                else:
                    raise InvalidConfig(
                        f"{key} must be one of {modes!r}, got {config[key]!r}"
                    )
            elif (
                config[key] != normal_mode
                and config.get(TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY) is not True
            ):
                if _fix_invalid:
                    config.pop(key, None)
                else:
                    raise InvalidConfig(
                        f"{key}={config[key]!r} changes output correctness; set "
                        f"{TCGEN05_DIAGNOSTIC_INVALID_OUTPUT_CONFIG_KEY}=True "
                        "only for diagnostic invalid-output runs"
                    )

        validate_tcgen05_diagnostic_mode(
            TCGEN05_C_STORE_MODE_CONFIG_KEY,
            TCGEN05_C_STORE_MODES,
            TCGEN05_C_STORE_MODE_NORMAL,
        )
        validate_tcgen05_diagnostic_mode(
            TCGEN05_ACC_PRODUCER_MODE_CONFIG_KEY,
            TCGEN05_ACC_PRODUCER_MODES,
            TCGEN05_ACC_PRODUCER_MODE_NORMAL,
        )
        validate_tcgen05_diagnostic_mode(
            TCGEN05_ACC_PRODUCER_ADVANCE_MODE_CONFIG_KEY,
            TCGEN05_ACC_PRODUCER_ADVANCE_MODES,
            TCGEN05_ACC_PRODUCER_ADVANCE_MODE_NORMAL,
        )
        validate_tcgen05_diagnostic_mode(
            TCGEN05_AB_PRODUCER_ACQUIRE_MODE_CONFIG_KEY,
            TCGEN05_AB_PRODUCER_ACQUIRE_MODES,
            TCGEN05_AB_PRODUCER_ACQUIRE_MODE_NORMAL,
        )
        validate_tcgen05_diagnostic_mode(
            TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODE_CONFIG_KEY,
            TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODES,
            TCGEN05_AB_INITIAL_PRODUCER_ACQUIRE_MODE_NORMAL,
        )
        validate_tcgen05_diagnostic_mode(
            TCGEN05_AB_PRODUCER_ADVANCE_MODE_CONFIG_KEY,
            TCGEN05_AB_PRODUCER_ADVANCE_MODES,
            TCGEN05_AB_PRODUCER_ADVANCE_MODE_NORMAL,
        )
        validate_tcgen05_diagnostic_mode(
            TCGEN05_AB_CONSUMER_WAIT_MODE_CONFIG_KEY,
            TCGEN05_AB_CONSUMER_WAIT_MODES,
            TCGEN05_AB_CONSUMER_WAIT_MODE_NORMAL,
        )
        validate_tcgen05_diagnostic_mode(
            TCGEN05_AB_CONSUMER_PHASE_MODE_CONFIG_KEY,
            TCGEN05_AB_CONSUMER_PHASE_MODES,
            TCGEN05_AB_CONSUMER_PHASE_MODE_NORMAL,
        )
        if TCGEN05_CUBIN_LINEINFO_CONFIG_KEY in config:
            if not self.cute_tcgen05_search_enabled:
                if _fix_invalid:
                    config.pop(TCGEN05_CUBIN_LINEINFO_CONFIG_KEY, None)
                else:
                    raise InvalidConfig(
                        f"{TCGEN05_CUBIN_LINEINFO_CONFIG_KEY} is only "
                        "supported for tcgen05-enabled CuTe matmul kernels"
                    )
            elif not isinstance(config[TCGEN05_CUBIN_LINEINFO_CONFIG_KEY], bool):
                if _fix_invalid:
                    config.pop(TCGEN05_CUBIN_LINEINFO_CONFIG_KEY, None)
                else:
                    raise InvalidConfig(
                        f"{TCGEN05_CUBIN_LINEINFO_CONFIG_KEY} must be a boolean"
                    )
        if TCGEN05_LARGE_BN_PROOF_CONFIG_KEY in config:
            if not self.cute_tcgen05_search_enabled:
                if _fix_invalid:
                    config.pop(TCGEN05_LARGE_BN_PROOF_CONFIG_KEY, None)
                else:
                    raise InvalidConfig(
                        f"{TCGEN05_LARGE_BN_PROOF_CONFIG_KEY} is only "
                        "supported for tcgen05-enabled CuTe matmul kernels"
                    )
            elif not isinstance(config[TCGEN05_LARGE_BN_PROOF_CONFIG_KEY], bool):
                if _fix_invalid:
                    config.pop(TCGEN05_LARGE_BN_PROOF_CONFIG_KEY, None)
                else:
                    raise InvalidConfig(
                        f"{TCGEN05_LARGE_BN_PROOF_CONFIG_KEY} must be a boolean"
                    )
            elif config[TCGEN05_LARGE_BN_PROOF_CONFIG_KEY] is True:
                proof_envelope_matches = (
                    tuple(cast("list[int]", config.get("block_sizes", [])))
                    == TCGEN05_LARGE_BN_PROOF_BLOCK_SIZES
                    and config.get("tcgen05_cluster_m", 1)
                    == TCGEN05_LARGE_BN_PROOF_CLUSTER_M
                    and config.get("pid_type", "flat")
                    == TCGEN05_LARGE_BN_PROOF_PID_TYPE
                    and all(
                        config.get(key) == expected
                        for key, expected in TCGEN05_LARGE_BN_PROOF_STAGE_CONFIGS
                    )
                )
                if not proof_envelope_matches:
                    if _fix_invalid:
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
        if TCGEN05_CLUSTER_M2_ONE_CTA_ROLE_LOCAL_CONFIG_KEY in config:
            if not self.cute_tcgen05_search_enabled:
                if _fix_invalid:
                    config.pop(
                        TCGEN05_CLUSTER_M2_ONE_CTA_ROLE_LOCAL_CONFIG_KEY,
                        None,
                    )
                else:
                    raise InvalidConfig(
                        f"{TCGEN05_CLUSTER_M2_ONE_CTA_ROLE_LOCAL_CONFIG_KEY} is "
                        "only supported for tcgen05-enabled CuTe matmul kernels"
                    )
            elif not isinstance(
                config[TCGEN05_CLUSTER_M2_ONE_CTA_ROLE_LOCAL_CONFIG_KEY],
                bool,
            ):
                if _fix_invalid:
                    config.pop(
                        TCGEN05_CLUSTER_M2_ONE_CTA_ROLE_LOCAL_CONFIG_KEY,
                        None,
                    )
                else:
                    raise InvalidConfig(
                        f"{TCGEN05_CLUSTER_M2_ONE_CTA_ROLE_LOCAL_CONFIG_KEY} "
                        "must be a boolean"
                    )
        if TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY in config:
            if not self.cute_tcgen05_search_enabled:
                if _fix_invalid:
                    config.pop(TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY, None)
                else:
                    raise InvalidConfig(
                        f"{TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY} is only "
                        "supported for tcgen05-enabled CuTe matmul kernels"
                    )
            elif (
                config[TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY]
                not in TCGEN05_EPILOGUE_LAYOUTS
            ):
                if _fix_invalid:
                    config.pop(TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY, None)
                else:
                    raise InvalidConfig(
                        f"{TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY} must be one "
                        f"of {TCGEN05_EPILOGUE_LAYOUTS!r}, got "
                        f"{config[TCGEN05_EPILOGUE_LAYOUT_CONFIG_KEY]!r}"
                    )
        if self.has_pallas_inner_loops:
            if self.has_symbolic_or_data_dependent_bounds:
                # "unroll" uses Python range() which can't handle traced bounds.
                # Between the remaining options, prefer "fori_loop": it handles
                # both DMA-aligned and unaligned inner blocks, while
                # "emit_pipeline" fails on unaligned dims.
                config.setdefault("pallas_loop_type", "fori_loop")
            else:
                config.setdefault("pallas_loop_type", VALID_PALLAS_LOOP_TYPES[0])

        if self.supports_config_key("pid_type"):
            if "pid_type" in config:
                if config["pid_type"] not in VALID_PID_TYPES:
                    raise InvalidConfig(
                        f"Invalid value for 'pid_type': {config['pid_type']!r} must be one of {list(VALID_PID_TYPES)!r}"
                    )
            else:
                config["pid_type"] = VALID_PID_TYPES[0]

        if _fix_invalid:
            self._fix_tcgen05_cluster_m2_search_config(config)
            self._fix_tcgen05_cluster_m1_persistent_search_config(config)

        if self.supports_config_key("num_sm_multiplier"):
            # Validate num_sm_multiplier is a power of two in range
            if "num_sm_multiplier" in config:
                val = config["num_sm_multiplier"]
                if (
                    not isinstance(val, int)
                    or val < MIN_NUM_SM_MULTIPLIER
                    or val > MAX_NUM_SM_MULTIPLIER
                    or (val & (val - 1)) != 0  # not a power of two
                ):
                    raise InvalidConfig(
                        f"Invalid value for 'num_sm_multiplier': {val!r} must be a power of two between {MIN_NUM_SM_MULTIPLIER} and {MAX_NUM_SM_MULTIPLIER}"
                    )
            else:
                config["num_sm_multiplier"] = DEFAULT_NUM_SM_MULTIPLIER

        # Only validate maxnreg on CUDA devices (not supported on AMD and Intel GPU)
        if self.supports_config_key("maxnreg") and supports_maxnreg():
            if "maxnreg" in config:
                if config["maxnreg"] not in VALID_MAXNREG:
                    raise InvalidConfig(
                        f"Invalid value for 'maxnreg': {config['maxnreg']!r} must be one of {list(VALID_MAXNREG)!r}"
                    )
            else:
                config["maxnreg"] = VALID_MAXNREG[0]

            # Cap maxnreg so that maxnreg * threads_per_block doesn't exceed
            # the register file.  On sm100+ ptxas honours .maxnreg over
            # .reqntid, so an uncapped value causes "out of resource: threads"
            # at load.
            maxnreg = cast("int | None", config.get("maxnreg"))
            num_warps = config.get("num_warps", DEFAULT_NUM_WARPS)
            if maxnreg is not None and isinstance(num_warps, int):
                limit = _regs_per_block() // warps_to_threads(num_warps)
                if maxnreg > limit:
                    if _fix_invalid:
                        valid = [
                            v for v in VALID_MAXNREG if v is not None and v <= limit
                        ]
                        if valid:
                            config["maxnreg"] = max(valid)
                        else:
                            config.pop("maxnreg", None)
                    else:
                        raise InvalidConfig(
                            f"maxnreg={maxnreg} exceeds register budget for "
                            f"num_warps={num_warps} (max {limit})"
                        )
        else:
            # Remove maxnreg if not supported
            config.pop("maxnreg", None)

        # Handle num_sm_multiplier and maxnreg for non-persistent pid_types
        # These options only make sense for persistent kernels
        pid_type = config.get("pid_type")
        if pid_type in ("flat", "xyz"):
            # Handle num_sm_multiplier
            num_sm_multiplier = config.get(
                "num_sm_multiplier", DEFAULT_NUM_SM_MULTIPLIER
            )
            if num_sm_multiplier != DEFAULT_NUM_SM_MULTIPLIER:
                if _fix_invalid:
                    # Silently fix during autotuning config generation
                    config.pop("num_sm_multiplier", None)
                else:
                    # Raise error for user-specified invalid combinations
                    raise InvalidConfig(
                        f"num_sm_multiplier={num_sm_multiplier} can only be used with persistent "
                        f"pid_type ('persistent_blocked' or 'persistent_interleaved'), "
                        f"got pid_type={pid_type!r}"
                    )
            else:
                # Remove default value from config
                config.pop("num_sm_multiplier", None)

            # Handle maxnreg - only makes sense for persistent kernels (and only on non-AMD and non-Intel GPU)
            if supports_maxnreg():
                maxnreg = config.get("maxnreg", DEFAULT_MAXNREG)
                if maxnreg != DEFAULT_MAXNREG:
                    if _fix_invalid:
                        # Silently fix during autotuning config generation
                        config.pop("maxnreg", None)
                    else:
                        # Raise error for user-specified invalid combinations
                        raise InvalidConfig(
                            f"maxnreg={maxnreg} can only be used with persistent "
                            f"pid_type ('persistent_blocked' or 'persistent_interleaved'), "
                            f"got pid_type={pid_type!r}"
                        )
                else:
                    # Remove default value from config
                    config.pop("maxnreg", None)

        if "advanced_controls_file" in config:
            value = config.get("advanced_controls_file") or ""
            if not isinstance(value, str):
                raise InvalidConfig(
                    f"advanced_controls_file must be a string path, got {value!r}"
                )
            config["advanced_controls_file"] = value

        if "epilogue_subtile" in config:
            val = config["epilogue_subtile"]
            # Normalize bool to int for backward compat
            if val is True:
                config["epilogue_subtile"] = 2
            elif not val:
                config.pop("epilogue_subtile", None)
            elif val not in EPILOGUE_SUBTILE_EXTENDED_CHOICES:
                raise InvalidConfig(
                    f"epilogue_subtile must be one of {EPILOGUE_SUBTILE_EXTENDED_CHOICES!r}, got {val!r}"
                )
            elif _fix_invalid and not self._should_keep_epilogue_subtile_for_autotune():
                config.pop("epilogue_subtile", None)
            # Epilogue subtiling is incompatible with flatten_loops because
            # FlattenedTileStrategy does not support offset_var needed by
            # the epilogue store codegen path.
            flatten_loops = config.get("flatten_loops")
            if (
                "epilogue_subtile" in config
                and isinstance(flatten_loops, list)
                and any(flatten_loops)
            ):
                if _fix_invalid:
                    config.pop("epilogue_subtile", None)
                else:
                    raise InvalidConfig(
                        "epilogue_subtile is incompatible with flatten_loops=True"
                    )

        # Set default values for grid indices when pid_type is not persistent
        if pid_type in ("flat", "xyz") and self.grid_block_ids:
            for name, mapping in (
                ("range_unroll_factors", self.range_unroll_factors),
                ("range_warp_specializes", self.range_warp_specialize),
                ("range_num_stages", self.range_num_stages),
                ("range_multi_buffers", self.range_multi_buffers),
                ("range_flattens", self.range_flattens),
            ):
                config[name] = mapping._reset_config_to_default(
                    name, config.get(name, ()), block_ids=self.grid_block_ids
                )

        range_warp_specializes = cast(
            "list[bool | None]", config.get("range_warp_specializes", [])
        )

        if range_warp_specializes and any(range_warp_specializes):
            # Only one range_warp_specializes is allowed, take the first one
            # Prefer warp specialize on outermost loop
            first_idx = range_warp_specializes.index(True)
            for i in range(first_idx + 1, len(range_warp_specializes)):
                range_warp_specializes[i] = None

            range_unroll_factors = cast(
                "list[int]", config.get("range_unroll_factors", [])
            )
            if range_unroll_factors and range_unroll_factors[first_idx] > 1:
                if range_unroll_factors[first_idx]:
                    range_unroll_factors[first_idx] = 0

                config["range_unroll_factors"] = range_unroll_factors

        if self.supports_config_key("range_warp_specializes"):
            config["range_warp_specializes"] = range_warp_specializes

        if self.backend_name == "cute":
            for key in _CUTE_IMPLICIT_DEFAULT_KEYS - provided_keys:
                config.pop(key, None)

        # Allow tunable parameter keys in addition to backend-supported keys.
        allowed_keys = self.supported_config_keys() | {
            *self.user_defined_tunables.keys()
        }
        if invalid_keys := ({*config} - allowed_keys):
            raise InvalidConfig(f"Invalid config keys {sorted(invalid_keys)!r}")

    def raise_grid_block_minimums(self) -> None:
        """Raise min_size for grid block dimensions based on problem size.

        Very small block sizes produce enormous grids that the autotuner
        wastes time exploring.  This heuristic sets a floor so the total
        number of blocks per dimension stays within a reasonable range
        derived from ``num_compute_units``.

        The raised minimum never exceeds the default block size that
        ``_fragment`` would compute, so memory and shared-memory
        constraints from non-tiled dimensions are respected.
        """
        if not self.grid_block_ids:
            return

        n_cus = num_compute_units()
        n_dims = len(self.grid_block_ids)
        max_blocks_per_dim = math.ceil((n_cus * 64) ** (1.0 / n_dims))

        for grid_bid in self.grid_block_ids:
            try:
                spec = self.block_sizes.block_id_lookup(grid_bid)
            except KeyError:
                continue
            if spec.size_hint <= 0:
                continue
            default = spec._fragment(self).default_val
            min_block = spec.size_hint // max_blocks_per_dim
            min_block = min(min_block, default)
            if min_block >= 2:
                min_block = 1 << (min_block.bit_length() - 1)
                min_block = min(min_block, spec.max_size)
                spec.autotuner_min = assert_integer_power_of_two(
                    max(min_block, spec.autotuner_min)
                )

    def create_config_generation(
        self,
        *,
        overrides: Mapping[str, object] | None = None,
        advanced_controls_files: list[str] | None = None,
        process_group_name: str | None = None,
    ) -> ConfigGeneration:
        from .config_generation import ConfigGeneration

        return ConfigGeneration(
            self,
            overrides=overrides,
            advanced_controls_files=advanced_controls_files,
            process_group_name=process_group_name,
        )

    def default_config(self) -> helion.Config:
        config = self.flat_config(lambda x: x.default())
        self._shrink_for_numel_constraints(config)
        return config

    def _shrink_for_numel_constraints(self, config: helion.Config) -> None:
        """Shrink block_sizes in *config* in-place so every tensor numel
        constraint is satisfied.
        """
        block_sizes = config.config.get("block_sizes")
        if (
            not isinstance(block_sizes, list)
            or not block_sizes
            or not self.tensor_numel_constraints
        ):
            return
        min_sizes = [
            max(self.block_sizes[i].min_size, 1) for i in range(len(block_sizes))
        ]
        shrink_block_sizes_for_numel_constraints(
            self.tensor_numel_constraints, block_sizes, min_sizes
        )

    def _flat_fields(
        self,
    ) -> dict[str, BlockIdSequence[Any] | ConfigSpecFragment]:
        """Return {key: field} for all tunable fields in flat_config() order.

        This is the single source of truth for field ordering.
        """
        fields: dict[str, BlockIdSequence[Any] | ConfigSpecFragment] = {
            "block_sizes": self.block_sizes,
        }
        if self.backend_name == "cute":
            if self.cute_tcgen05_search_enabled:
                fields["l2_groupings"] = self.l2_groupings
                fields.update(self._tcgen05_optional_fragments(for_search=True))
                if self.supports_config_key("pid_type"):
                    fields["pid_type"] = EnumFragment(self.allowed_pid_types)
                if self.supports_config_key("indexing") and self.indexing.length > 0:
                    fields["indexing"] = self.indexing
            elif self.supports_config_key("num_threads"):
                fields["num_threads"] = self.num_threads
            if self.epilogue_subtile_autotune_choices is not None:
                fields["epilogue_subtile"] = EnumFragment(
                    choices=self.epilogue_subtile_autotune_choices
                )
            fields.update(self.user_defined_tunables)
            return fields

        # Only add sequence keys that the backend supports
        fields.update(
            {
                name: seq
                for name, seq in [
                    ("loop_orders", self.loop_orders),
                    ("flatten_loops", self.flatten_loops),
                    ("l2_groupings", self.l2_groupings),
                    ("reduction_loops", self.reduction_loops),
                    ("range_unroll_factors", self.range_unroll_factors),
                    ("range_warp_specializes", self.range_warp_specialize),
                    ("range_num_stages", self.range_num_stages),
                    ("range_multi_buffers", self.range_multi_buffers),
                    ("range_flattens", self.range_flattens),
                    ("static_ranges", self.static_ranges),
                ]
                if self.supports_config_key(name)
            }
        )

        # Scalar fields (ConfigSpecFragment)
        is_tileir = self.backend_name == "tileir"
        if is_tileir:
            # TileIR: num_warps is unused (fixed at 4), num_stages has wider range
            num_warps_fragment: ConfigSpecFragment = NumWarpsFragment(4, 4)
        elif supports_amd_cdna_tunables():
            num_warps_fragment = NumWarpsFragment(1, 16, DEFAULT_NUM_WARPS)
        else:
            num_warps_fragment = NumWarpsFragment(1, 32, DEFAULT_NUM_WARPS)
        num_stages_fragment = self._num_stages_fragment()

        if self.supports_config_key("num_warps"):
            fields["num_warps"] = num_warps_fragment
        if self.supports_config_key("num_stages"):
            fields["num_stages"] = num_stages_fragment
        if self.supports_config_key("indexing"):
            fields["indexing"] = self.indexing
        if self.supports_config_key("atomic_indexing"):
            fields["atomic_indexing"] = self.atomic_indexing
        if self.supports_config_key("pid_type"):
            fields["pid_type"] = EnumFragment(self.allowed_pid_types)
        if self.supports_config_key("num_sm_multiplier"):
            fields["num_sm_multiplier"] = PowerOfTwoFragment(
                MIN_NUM_SM_MULTIPLIER,
                self.max_num_sm_multiplier,
                DEFAULT_NUM_SM_MULTIPLIER,
            )
        if self.supports_config_key("load_eviction_policies"):
            fields["load_eviction_policies"] = self.load_eviction_policies
        if self.cute_tcgen05_search_enabled:
            fields.update(self._tcgen05_optional_fragments(for_search=True))
        if self.supports_config_key("num_threads"):
            fields["num_threads"] = self.num_threads
        if is_tileir:
            fields["num_ctas"] = self.backend_tunable_fragments["num_ctas"]
            fields["occupancy"] = self.backend_tunable_fragments["occupancy"]
        else:
            fields.update(self.backend_tunable_fragments)
        if self.has_pallas_inner_loops:
            choices = VALID_PALLAS_LOOP_TYPES
            if self.has_symbolic_or_data_dependent_bounds:
                # Exclude "unroll" (uses Python range(), can't handle traced
                # bounds) and put "fori_loop" first: it handles both DMA-aligned
                # and unaligned inner blocks, while "emit_pipeline" fails on
                # unaligned dims.
                # TODO(thcmbs): Also exclude "emit_pipeline" when has_pallas_dma_unaligned
                # is set, to avoid wasted autotuning effort. See PR #1969 review discussion.
                choices = ("fori_loop", "emit_pipeline")
            fields["pallas_loop_type"] = EnumFragment(choices=choices)
            if self.supports_config_key("pallas_pre_broadcast"):
                fields["pallas_pre_broadcast"] = BooleanFragment()
        # Only include maxnreg on CUDA devices (not supported on AMD and Intel GPU)
        if self.supports_config_key("maxnreg") and supports_maxnreg():
            fields["maxnreg"] = EnumFragment(VALID_MAXNREG)
        if self.epilogue_subtile_autotune_choices is not None:
            fields["epilogue_subtile"] = EnumFragment(
                choices=self.epilogue_subtile_autotune_choices
            )
        # Add tunable parameters
        fields.update(self.user_defined_tunables)
        return fields

    def structural_fingerprint(
        self, *, advanced_controls_files: list[str] | None = None
    ) -> tuple[tuple[str | int, ...], ...]:
        """Return a hashable structural description of this ConfigSpec's search space.

        Captures field names, sequence lengths, per-item block_ids lengths
        (for PermutationFragment), ListOf inner lengths, and optional ACF slot
        presence.  Two ConfigSpecs with the same fingerprint can safely exchange
        FlatConfig values.
        """
        result: list[tuple[str | int, ...]] = [
            (key, *field.fingerprint()) for key, field in self._flat_fields().items()
        ]
        acf_fragment = self._advanced_controls_file_fragment(advanced_controls_files)
        if acf_fragment is not None:
            result.append(
                (
                    "advanced_controls_file",
                    *cast("tuple[str, ...]", acf_fragment.choices),
                )
            )
        return tuple(result)

    def structural_fingerprint_hash(
        self, *, advanced_controls_files: list[str] | None = None
    ) -> str:
        """Return a hex-digest SHA-256 hash of the structural fingerprint."""
        return hashlib.sha256(
            repr(
                self.structural_fingerprint(
                    advanced_controls_files=advanced_controls_files
                )
            ).encode("utf-8")
        ).hexdigest()

    def _advanced_controls_file_fragment(
        self, advanced_controls_files: list[str] | None
    ) -> EnumFragment | None:
        # Empty list means no autotuning with ACFs.
        if not advanced_controls_files:
            return None
        files = advanced_controls_files
        # When non-empty list is provided then ensure default -O3 is considered.
        if "" not in files:
            files = [*files, ""]
        return EnumFragment(tuple(files))

    def flat_key_layout(
        self, *, advanced_controls_files: list[str] | None = None
    ) -> list[tuple[str, int, bool]]:
        """Return (key_name, num_flat_entries, is_sequence) for each field.

        is_sequence is True for BlockIdSequence keys whose list values
        are spread across individual flat slots.
        """
        result = [
            (key, *field._flat_key_info()) for key, field in self._flat_fields().items()
        ]
        if self._advanced_controls_file_fragment(advanced_controls_files) is not None:
            result.append(("advanced_controls_file", 1, False))
        return result

    def flat_config(
        self,
        fn: Callable[[ConfigSpecFragment], object],
        *,
        advanced_controls_files: list[str] | None = None,
    ) -> helion.Config:
        """Map a flattened version of the config using the given function."""
        config: dict[str, Any] = {}
        for key, field in self._flat_fields().items():
            config[key] = field._flat_config(self, fn)

        for name in (
            "loop_orders",
            "num_threads",
            "flatten_loops",
            "reduction_loops",
            "l2_groupings",
            "range_unroll_factors",
            "range_warp_specializes",
            "range_num_stages",
            "range_multi_buffers",
            "range_flattens",
            "static_ranges",
            "load_eviction_policies",
            "indexing",
            "atomic_indexing",
        ):
            if not config.get(name):
                config.pop(name, None)
        acf_fragment = self._advanced_controls_file_fragment(advanced_controls_files)
        if acf_fragment is not None:
            config["advanced_controls_file"] = fn(acf_fragment)
        self.normalize(config, _fix_invalid=True)
        return helion.Config(**config)


class LoopOrderSpec(_BlockIdItem):
    def _fragment(self, base: ConfigSpec) -> PermutationFragment:
        return PermutationFragment(len(self.block_ids))

    def _normalize(self, name: str, value: object) -> list[int]:
        if type(value) is not list:
            if not isinstance(value, tuple):
                raise InvalidConfig(f"{name} must be a list, got {value!r}")
            value = [*value]
        length = len(self.block_ids)
        if len(value) != length:
            raise InvalidConfig(f"{name} must be length {length}, got {len(value)}")
        if {*value} != {*range(length)}:
            raise InvalidConfig(f"{name} must be permutation, got {value!r}")
        return value

    def _fill_missing(self) -> list[int]:
        """Provide a value when not provided by the user."""
        return [*range(len(self.block_ids))]


class L2GroupingSpec(_PowerOfTwoBlockIdItem):
    def _fragment(self, base: ConfigSpec) -> PowerOfTwoFragment:
        return PowerOfTwoFragment(1, 64, 1)

    def _fill_missing(self) -> int:
        return 1


class BlockSizeSpec(_PowerOfTwoBlockIdItem):
    def __init__(
        self,
        *,
        block_id: int,
        size_hint: int,
        min_size: int = 1,
        max_size: int | None = None,
    ) -> None:
        super().__init__([block_id])
        self.size_hint = size_hint

        # TODO(shunting): it's a bit conservative since not every block is split
        # for different ranks.
        bounded_hint = size_hint
        if dist.is_initialized():
            world_size = dist.get_world_size()
            bounded_hint = bounded_hint // world_size

        bounded_hint = max(bounded_hint, 1)
        self.min_size: int = min_size
        self.autotuner_min: int = min_size
        self.max_size: int = (
            next_power_of_2(bounded_hint) if max_size is None else max_size
        )
        if self.max_size < self.min_size:
            self.max_size = self.min_size
        assert self.min_size <= self.max_size

    def __repr__(self) -> str:
        fields: list[str] = []
        for field, default in (
            ("block_id", None),
            ("size_hint", None),
            ("min_size", 1),
            ("max_size", next_power_of_2(self.size_hint)),
        ):
            value = getattr(self, field)
            if value != default:
                fields.append(f"{field}={value!r}")
        return f"BlockSizeSpec({', '.join(fields)})"

    def _normalize(self, name: str, value: object) -> int | None:
        result = super()._normalize(name, value)
        if isinstance(result, int) and result < self.min_size:
            result = self.min_size
        return result

    def update_min(self, value: int) -> None:
        self.min_size = assert_integer_power_of_two(max(value, self.min_size))
        if self.max_size < self.min_size:
            self.max_size = self.min_size

    def update_max(self, value: int) -> None:
        clamped = max(value, 1)
        self.max_size = assert_integer_power_of_two(min(clamped, self.max_size))

    def update_hint(self, value: int) -> None:
        self.size_hint = value
        self.update_max(next_power_of_2(max(value, 1)))

    def _fragment(self, base: ConfigSpec) -> BlockSizeFragment:
        total_ndim = len(base.block_sizes)
        reduction_numel = _product(
            [next_power_of_2(spec.size_hint) for spec in base.reduction_loops]
        )
        if total_ndim <= 2 and reduction_numel <= 128:
            default = 32
        elif total_ndim >= 3 and reduction_numel > 1:
            # With 3+ tiled dimensions and a non-trivial reduction/full-slice
            # dimension, the total tensor numel (default^total_ndim *
            # reduction_numel) grows quickly and can cause Triton JIT
            # compilation to hang or exceed shared memory limits.
            # Compute a default that keeps total numel <= 32768 (safe for
            # 64KB shared memory with 2-byte elements like bf16).
            target = 32768
            per_dim = int((target / reduction_numel) ** (1.0 / total_ndim))
            default = max(1, 1 << (per_dim.bit_length() - 1)) if per_dim >= 1 else 1
        elif reduction_numel <= 256:
            default = 16
        else:
            default = 1
        low = min(max(self.min_size, self.autotuner_min), self.max_size)
        return BlockSizeFragment(
            low,
            self.max_size,
            default,
        )


class NumThreadsSpec(_PowerOfTwoBlockIdItem):
    def __init__(self, *, block_id: int, size_hint: int) -> None:
        super().__init__([block_id])
        self.size_hint = size_hint

    def _normalize(self, name: str, value: object) -> int | None:
        # 0 is a valid sentinel meaning "use block_size as thread count"
        if value == 0:
            return 0
        return super()._normalize(name, value)

    def _fragment(self, base: ConfigSpec) -> NumThreadsFragment:
        max_threads = min(max(self.size_hint, 1), 1024)
        default = next_power_of_2(max_threads)
        return NumThreadsFragment(default)

    def _fill_missing(self) -> int:
        return 0


class FlattenLoopSpec(_BlockIdItem):
    def _fragment(self, base: ConfigSpec) -> BooleanFragment:
        return BooleanFragment()

    def _normalize(self, name: str, value: object) -> bool:
        if not isinstance(value, bool):
            raise InvalidConfig(f"{name} must be a boolean, got {value!r}") from None
        return value

    def _fill_missing(self) -> bool:
        return False


class ReductionLoopSpec(_PowerOfTwoBlockIdItem):
    def __init__(
        self,
        *,
        block_id: int,
        size_hint: int,
    ) -> None:
        super().__init__([block_id])
        self.size_hint = size_hint

    def _flat_fragment(self, base: ConfigSpec) -> BlockSizeFragment:
        # Shared by both directions:
        # - unflatten: flat integer -> Config value via _flat_config()
        # - flatten: Config value -> flat integer via _encode_flat_value()
        low = 8  # TODO(jansel): is smaller needed?
        high = next_power_of_2(max(low, self.size_hint))
        default = min(high, 4096)
        # Cap default at the backend's max reduction threads so that
        # large reductions default to looped rather than persistent.
        if base.max_reduction_threads is not None:
            if self.size_hint > base.max_reduction_threads:
                default = min(default, base.max_reduction_threads)
        return BlockSizeFragment(low, high, default)

    def _flat_config(
        self, base: ConfigSpec, fn: Callable[[ConfigSpecFragment], object]
    ) -> int | None:
        fragment = self._flat_fragment(base)
        low = fragment.low
        high = fragment.high
        value = fn(fragment)
        assert isinstance(value, int)
        if not (low <= value <= high):
            raise InvalidConfig(
                f"Invalid value for reduction loop {low} <= {value} <= {high}"
            )
        if value >= self.size_hint:
            return None  # max size becomes persistent reduction
        return value

    def _encode_flat_value(self, base: ConfigSpec, value: object) -> object:
        # None means "persistent reduction" in the normalized Config. In the
        # flat search space that same choice is represented by an integer
        # sentinel, typically the fragment default such as 1024 for a 1024-wide
        # reduction. This is the one non-identity Config <-> FlatConfig
        # mapping today.
        if value is None:
            return self._flat_fragment(base).default()
        return value

    def _normalize(self, name: str, value: object) -> int | None:
        if value is None:
            return None
        return super()._normalize(name, value)

    def _fill_missing(self) -> None:
        return None


class _OptionalIntSpec(_BlockIdItem):
    def _normalize(self, name: str, value: object) -> int:
        if not isinstance(value, int):
            raise InvalidConfig(f"{name} must be an integer, got {value!r}")
        return value

    def _fill_missing(self) -> int:
        """Provide a value when not provided by the user."""
        return 0


class _OptionalBoolSpec(_BlockIdItem):
    def _fragment(self, base: ConfigSpec) -> EnumFragment:
        return EnumFragment((None, False, True))

    def _normalize(self, name: str, value: object) -> bool | None:
        if value is not None and not isinstance(value, bool):
            raise InvalidConfig(f"{name} must be a boolean or None, got {value!r}")
        return value

    def _fill_missing(self) -> None:
        """Provide a value when not provided by the user."""
        return None


class RangeUnrollFactorSpec(_OptionalIntSpec):
    def _fragment(self, base: ConfigSpec) -> IntegerFragment:
        return IntegerFragment(0, 4, 0)


class RangeWarpSpecializeSpec(_OptionalBoolSpec):
    pass


class RangeNumStagesSpec(_OptionalIntSpec):
    def _fragment(self, base: ConfigSpec) -> IntegerFragment:
        return IntegerFragment(0, 4, 0)


class RangeMultiBufferSpec(_OptionalBoolSpec):
    pass


class RangeFlattenSpec(_OptionalBoolSpec):
    pass


class StaticRangeSpec(_BlockIdItem):
    def _fragment(self, base: ConfigSpec) -> BooleanFragment:
        return BooleanFragment()

    def _normalize(self, name: str, value: object) -> bool:
        if not isinstance(value, bool):
            raise InvalidConfig(f"{name} must be a boolean, got {value!r}")
        return value

    def _fill_missing(self) -> bool:
        """Provide a value when not provided by the user."""
        return False


def _product(seq: Sequence[int]) -> int:
    """Return the product of the elements in the sequence."""
    return functools.reduce(operator.mul, seq, 1)
