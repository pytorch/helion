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
from .._compat import target_device_capability as get_target_device_capability
from .._compat import warps_to_threads
from .._compiler.cute.tcgen05_config import CUTE_TCGEN05_DIAGNOSTIC_CONFIG_KEYS
from .._compiler.cute.tcgen05_config import CUTE_TCGEN05_STRATEGY_CONFIG_KEYS
from .._compiler.cute.tcgen05_config import CUTE_TCGEN05_TUNABLE_KEYS
from .._compiler.cute.tcgen05_config import CuteTcgen05Config
from .._compiler.cute.tcgen05_config import Tcgen05AbStagesThreeSearchConstraints
from .._compiler.cute.tcgen05_config import Tcgen05ClusterM2SearchConstraints
from .._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_MAX_K_TILES
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
    from .._compiler.memory_op_slots import MemOpSlotMap
    from ..runtime.config import IndexingLiteral
    from ..runtime.config import PidTypeLiteral
    from .config_generation import ConfigGeneration

log = logging.getLogger(__name__)

_TARGET_DEVICE_CAPABILITY_UNSET = object()


class TensorNumelConstraint(NamedTuple):
    """Tensor element count must stay within Triton's max numel limit."""

    check_fn: Callable[..., bool]
    block_indices: tuple[int, ...]
    expr_str: str


class MatmulFact(NamedTuple):
    """Shape facts recorded when matmul requirements are applied."""

    lhs_ndim: int
    rhs_ndim: int
    m_block_id: int | None
    n_block_id: int | None
    k_block_id: int | None
    static_m: int | None
    static_n: int | None
    static_k: int | None
    lhs_dtype: torch.dtype
    rhs_dtype: torch.dtype


class ReductionFact(NamedTuple):
    """Workload facts for one inner reduction dim, recorded at compile time (analogous
    to ``MatmulFact``) so the seed heuristic branches on workload properties, not kernel
    identity. Exactly one per seeded kernel; built in device_ir's
    ``register_rollable_reductions`` (standard) or ``register_user_tiled_reductions``
    (user-tiled).

    - ``block_id`` / ``size_hint``: the reduction axis and its extent (rnumel).
    - ``m_block_ids``: the non-reduction (kept) tile block_ids.
    - ``static_rnumel``: the extent if statically known, else None.
    - ``itemsize``: bytes/element of the reduced tensor; byte caps key on
      ``size_hint * itemsize``.
    - ``num_load``: device loads over this rdim (the ``== 1`` stream-eviction gate).
    - ``num_carried_2d_tiles``: 2-D [M_BLOCK, R_BLOCK] tiles carried across the inner loop (the
      Band-B signal: jsd=2 / kl_div=1 / welford=0). Derived from ``AccumulatorFact``.
    - ``non_reduction_loop_block_ids``: non-grid loop tiles over the extent that are NOT the rdim
      (an apply/normalize pass); ``len(...) >= 1`` is the reduce-then-apply (Band-C) signal.
    - ``row_reread``: True iff the reduction-input row is live across the loop boundary (risks
      spilling). Gates the persist byte cap + re-read eviction. From ``MemoryOpFact``.
    - ``reread_eviction_index``: ``load_eviction_policies`` slot of the re-read load (``None``
      unless ``row_reread``), read from that load's ``MemoryOpFact.eviction_index`` — no re-walk.
    - ``full_width_output``: True iff a store writes the result back over the reduction axis
      ([M, N], e.g. layer_norm/softmax), False for a per-row scalar ([M], e.g. cross_entropy/sum) —
      full-width is store/occupancy-bound, scalar-output reduction-tree-bound (opposite num_warps).
    - ``input_load_itemsize``: element size of the HBM input row load (2 bf16/fp16, 4 fp32) — the
      dtype-faithful per-byte signal, distinct from ``itemsize`` (fp32-promoted = 4 at both). 0 when
      no single reduction-fed row load exists (kl_div/jsd).

    ``grid_rows`` is NOT stored — a pure function of ``m_block_ids`` + env, computed on demand by
    its one consumer (the narrow-row ``num_warps`` lever).
    """

    block_id: int
    size_hint: int
    m_block_ids: tuple[int, ...]
    static_rnumel: int | None
    itemsize: int
    num_load: int
    num_carried_2d_tiles: int = 0
    non_reduction_loop_block_ids: tuple[int, ...] = ()
    row_reread: bool = False
    reread_eviction_index: int | None = None
    full_width_output: bool = True
    input_load_itemsize: int = 0


class MemoryOpFact(NamedTuple):
    """Metadata linking one ``Config.indexing`` slot to its graph memory op, one entry per
    load/store in graph-traversal order (so ``memory_op_facts[i]`` describes ``config.indexing[i]``).
    Lets heuristics reason about *which* load/store a slot is, not a bare positional index.

    The reduction-fact builders consume the enrichment fields below (all reduction-AGNOSTIC — no
    notion of which axis is "the" reduction; the builders index them by the reduction's ``block_id``).
    """

    indexing_index: (
        int  # Config.indexing slot (the map slot; -1 if the op reads no indexing slot,
    )
    # i.e. a tile.index / stack load or a stack store)
    kind: str  # "load" | "store"
    eviction_index: int | None  # slot in Config.load_eviction_policies, else None
    tensor_name: str | None  # host buffer name being accessed, e.g. "x", "weight"
    dtype: torch.dtype | None  # element dtype of the accessed tensor
    ndim: int  # rank of the accessed tensor
    num_reuses: int  # downstream FX consumers of the load (0 for stores)
    matmul_operand: str | None  # matmul/dot operand: "lhs" | "rhs" | None
    # --- reduction-fact enrichment (reduction-agnostic; () / False / None for stores) ---
    # device_ir.graphs index this op lives in (scopes per-graph counts).
    graph_id: int = -1
    # per-axis count of reductions this load FEEDS: ((reduction_axis_block_id, count), ...).
    reductions_fed: tuple[tuple[int, int], ...] = ()
    # stores the load's value reaches WITHOUT passing through a reduction, each keyed by the
    # store's full subscript-axis tuple (store[tile_m, tile_n] -> (id_m, id_n)); the forward
    # walk cuts at both reductions and stores. Empty == the old bypasses_to_store was False.
    stores_fed: tuple[tuple[int | None, ...], ...] = ()
    # block-id per non-bare-int subscript, from the accessed tensor's SHAPE dims (``None`` where
    # unresolvable). Shape-resolved fallback for the gates below (standard
    # ``out[tile_m, :]`` plain slice).
    indexed_block_ids: tuple[int | None, ...] = ()
    # inner-dim extent for a rank>=2 op (legacy reduction-width signal; gates use subscript_block_ids).
    inner_extent: int | None = None
    # AXIS the op's INDEX subscripts address (block-id per non-bare-int position, from the tile/offset
    # subscript so it is reduction-AGNOSTIC; ``None`` for a plain slice). The faithful axis key for
    # the full_width_output / input_load_itemsize gates.
    subscript_block_ids: tuple[int | None, ...] = ()
    # Number of times this source op is emitted in the ORIGINAL (pre-rolling) graphs — its static
    # tuple-unroll factor (1 otherwise). On the triton path the facts are collapsed to one per distinct
    # ``mem_op_id`` (rolled/split/unroll copies share one fact), so this count recovers the instruction
    # tally ``num_load`` consumes (the ``== 1`` stream-eviction gate) without re-walking the graphs.
    # Config-INDEPENDENT (only static unroll affects it; rolled copies live in distinct graph_ids and
    # never reach num_load). 1 on the non-triton (uncollapsed) path.
    static_emission_count: int = 1


class AccumulatorFact(NamedTuple):
    """One loop-carried tensor accumulator in a reduction loop, recorded at compile time.
    Reduction-AGNOSTIC (like ``MemoryOpFact``): ``dim_block_ids`` is the per-dim block-id
    provenance (``None`` for a static dim), ``itemsize`` the element size.
    ``ReductionFact.num_carried_2d_tiles`` counts accumulators whose last dim is the rdim
    (jsd=2 / kl_div=1 / welford=0; standard's 1-D [M_BLOCK] scalar -> 0).
    """

    dim_block_ids: tuple[int | None, ...]
    itemsize: int


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
        *CUTE_TCGEN05_TUNABLE_KEYS,
    }
)
_BACKEND_DIAGNOSTIC_CONFIG_KEYS = CUTE_TCGEN05_DIAGNOSTIC_CONFIG_KEYS


def _get_backend_tunable_keys() -> frozenset[str]:
    """Get all backend tunable keys, including FB-private ones if available."""
    try:
        from ..fb.mtia_tunables import MTIA_TUNABLES  # pyrefly: ignore [missing-import]

        return _BASE_BACKEND_TUNABLE_KEYS | frozenset(MTIA_TUNABLES)
    except ImportError:
        return _BASE_BACKEND_TUNABLE_KEYS


BACKEND_TUNABLE_KEYS: frozenset[str] = _get_backend_tunable_keys()
_BACKEND_STRATEGY_CONFIG_KEYS = CUTE_TCGEN05_STRATEGY_CONFIG_KEYS
# All config keys whose support depends on the backend.  The base Backend
# class rejects these by default; each backend subclass opts in selectively.
BACKEND_SPECIFIC_KEYS: frozenset[str] = (
    BACKEND_TUNABLE_KEYS
    | _BACKEND_DIAGNOSTIC_CONFIG_KEYS
    | _BACKEND_STRATEGY_CONFIG_KEYS
    | {
        "num_threads",
        "cute_vector_widths",
        "load_cache_modifiers",
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
        "load_cache_modifiers",
        "pallas_loop_type",
        "pallas_pre_broadcast",
        "cute_vector_widths",
        *BACKEND_TUNABLE_KEYS,
        "advanced_controls_file",
        "epilogue_subtile",
        *_BACKEND_DIAGNOSTIC_CONFIG_KEYS,
        *_BACKEND_STRATEGY_CONFIG_KEYS,
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


# For tileir backend or AMD ROCM, eviction policies are not supported.
# Keep this uncached: some tests patch the AMD capability helper, and caching
# only on backend name can poison later Triton ConfigSpec construction inside
# the same worker process.
def get_valid_eviction_policies(backend_name: str) -> tuple[str, ...]:
    if backend_name == "triton" and not supports_amd_cdna_tunables():
        return ("", "first", "last")
    return ("",)


def get_valid_load_cache_modifiers(backend_name: str) -> tuple[str, ...]:
    if backend_name == "triton" and supports_amd_cdna_tunables():
        return ("", ".cg")
    return ("",)


class ConfigSpec:
    def __init__(
        self,
        *,
        backend: Backend,
        user_defined_tunables: Mapping[str, ConfigSpecFragment] | None = None,
        target_device_capability: tuple[int, int]
        | object
        | None = _TARGET_DEVICE_CAPABILITY_UNSET,
    ) -> None:
        self.backend = backend
        self.backend_name = backend.name
        self.max_reduction_threads = backend.max_reduction_threads()
        self.max_reduction_loop = backend.max_reduction_loop()
        self.reduction_loop_force_threshold = self.max_reduction_threads
        self.cute_indexed_reduction_block_ids: set[int] = set()
        self.user_defined_tunables = (
            {} if user_defined_tunables is None else dict(user_defined_tunables)
        )
        # Bound kernels pass an explicit target capability. Direct CuTe specs
        # use the current CUDA device so validation still enforces arch gates.
        if target_device_capability is _TARGET_DEVICE_CAPABILITY_UNSET:
            self.target_device_capability: tuple[int, int] | None = (
                get_target_device_capability() if self.backend_name == "cute" else None
            )
        else:
            self.target_device_capability = cast(
                "tuple[int, int] | None",
                target_device_capability,
            )

        self.block_sizes: BlockIdSequence[BlockSizeSpec] = BlockIdSequence()
        self.num_threads: BlockIdSequence[NumThreadsSpec] = BlockIdSequence()
        self.loop_orders: BlockIdSequence[LoopOrderSpec] = BlockIdSequence()
        self.l2_groupings: BlockIdSequence[L2GroupingSpec] = BlockIdSequence()
        self.flatten_loops: BlockIdSequence[FlattenLoopSpec] = BlockIdSequence()
        self.reduction_loops: BlockIdSequence[ReductionLoopSpec] = BlockIdSequence()
        self.cute_vector_widths: BlockIdSequence[CuteVectorWidthSpec] = (
            BlockIdSequence()
        )
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
        self.load_cache_modifiers = ListOf(
            EnumFragment(choices=get_valid_load_cache_modifiers(self.backend_name)),
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
        self._cute_tcgen05_config = CuteTcgen05Config(self)
        self.compiler_default_config: helion.Config | None = None
        self.compiler_seed_configs: list[helion.Config] = []
        self.autotuner_heuristics: list[str] = []
        self.matmul_facts: list[MatmulFact] = []
        self.reduction_facts: list[ReductionFact] = []
        self.accumulator_facts: list[AccumulatorFact] = []
        self.store_indices: list[int] = []
        self.memory_op_facts: list[MemoryOpFact] = []
        # Stable mem_op_id -> tunable slot maps (indexing/eviction/cache/atomic), built at bind from
        # node.meta['mem_op_id']; codegen resolves each emitted op's slot through this instead of a
        # per-config emission counter (transform-invariant). None until lower_to_device_ir runs.
        self.mem_op_slot_map: MemOpSlotMap | None = None
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
        arch = self.target_device_capability
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

    @property
    def cute_tcgen05_search_enabled(self) -> bool:
        return self._cute_tcgen05_config.search_enabled

    @cute_tcgen05_search_enabled.setter
    def cute_tcgen05_search_enabled(self, value: bool) -> None:
        self._cute_tcgen05_config.search_enabled = value

    @property
    def cute_tcgen05_aux_kernel_detected(self) -> bool:
        return self._cute_tcgen05_config.aux_kernel_detected

    @cute_tcgen05_aux_kernel_detected.setter
    def cute_tcgen05_aux_kernel_detected(self, value: bool) -> None:
        self._cute_tcgen05_config.aux_kernel_detected = value

    @property
    def cute_tcgen05_exact_shape_aux_kernel_detected(self) -> bool:
        return self._cute_tcgen05_config.exact_shape_aux_kernel_detected

    @cute_tcgen05_exact_shape_aux_kernel_detected.setter
    def cute_tcgen05_exact_shape_aux_kernel_detected(self, value: bool) -> None:
        self._cute_tcgen05_config.exact_shape_aux_kernel_detected = value

    @property
    def cute_tcgen05_matmul_has_non_tcgen05_operand(self) -> bool:
        return self._cute_tcgen05_config.matmul_has_non_tcgen05_operand

    @cute_tcgen05_matmul_has_non_tcgen05_operand.setter
    def cute_tcgen05_matmul_has_non_tcgen05_operand(self, value: bool) -> None:
        self._cute_tcgen05_config.matmul_has_non_tcgen05_operand = value

    @property
    def _tcgen05_cluster_m_search_choices(self) -> tuple[int, ...] | None:
        return self._cute_tcgen05_config.cluster_m_search_choices

    @_tcgen05_cluster_m_search_choices.setter
    def _tcgen05_cluster_m_search_choices(self, value: tuple[int, ...] | None) -> None:
        self._cute_tcgen05_config.cluster_m_search_choices = value

    @property
    def _tcgen05_cluster_m2_search_constraints(
        self,
    ) -> Tcgen05ClusterM2SearchConstraints | None:
        return self._cute_tcgen05_config.cluster_m2_search_constraints

    @_tcgen05_cluster_m2_search_constraints.setter
    def _tcgen05_cluster_m2_search_constraints(
        self, value: Tcgen05ClusterM2SearchConstraints | None
    ) -> None:
        self._cute_tcgen05_config.cluster_m2_search_constraints = value

    @property
    def _tcgen05_ab_stages_three_search_constraints(
        self,
    ) -> Tcgen05AbStagesThreeSearchConstraints | None:
        return self._cute_tcgen05_config.ab_stages_three_search_constraints

    @_tcgen05_ab_stages_three_search_constraints.setter
    def _tcgen05_ab_stages_three_search_constraints(
        self, value: Tcgen05AbStagesThreeSearchConstraints | None
    ) -> None:
        self._cute_tcgen05_config.ab_stages_three_search_constraints = value

    @property
    def _tcgen05_num_epi_warps_search_choices(self) -> tuple[int, ...] | None:
        return self._cute_tcgen05_config.num_epi_warps_search_choices

    @_tcgen05_num_epi_warps_search_choices.setter
    def _tcgen05_num_epi_warps_search_choices(
        self, value: tuple[int, ...] | None
    ) -> None:
        self._cute_tcgen05_config.num_epi_warps_search_choices = value

    @property
    def _tcgen05_num_epi_warps_validation_choices(self) -> tuple[int, ...] | None:
        return self._cute_tcgen05_config.num_epi_warps_validation_choices

    @_tcgen05_num_epi_warps_validation_choices.setter
    def _tcgen05_num_epi_warps_validation_choices(
        self, value: tuple[int, ...] | None
    ) -> None:
        self._cute_tcgen05_config.num_epi_warps_validation_choices = value

    def _tcgen05_full_tile_direct_entry_seed_eligible(self) -> bool:
        return self._cute_tcgen05_config.full_tile_direct_entry_seed_eligible()

    def _tcgen05_full_tile_direct_entry_seed_bk(self) -> int | None:
        return self._cute_tcgen05_config.full_tile_direct_entry_seed_bk()

    def _tcgen05_full_tile_direct_entry_seed_config(self) -> helion.Config | None:
        return self._cute_tcgen05_config.full_tile_direct_entry_seed_config()

    def restrict_tcgen05_cluster_m_search(self, choices: tuple[int, ...]) -> None:
        self._cute_tcgen05_config.restrict_cluster_m_search(choices)

    def allow_tcgen05_cluster_m2_search(
        self,
        *,
        static_k: int,
        max_k_tiles: int = TCGEN05_TWO_CTA_MAX_K_TILES,
        allow_edge_k_tail_family: bool = False,
    ) -> None:
        self._cute_tcgen05_config.allow_cluster_m2_search(
            static_k=static_k,
            max_k_tiles=max_k_tiles,
            allow_edge_k_tail_family=allow_edge_k_tail_family,
        )

    @staticmethod
    def _tcgen05_cluster_m2_bk_is_valid(
        bk: int, constraints: Tcgen05ClusterM2SearchConstraints
    ) -> bool:
        return CuteTcgen05Config.cluster_m2_bk_is_valid(bk, constraints)

    def _tcgen05_c_input_seed_config(self) -> helion.Config | None:
        return self._cute_tcgen05_config._c_input_seed_config()

    def autotune_seed_configs(self) -> list[helion.Config]:
        return self._cute_tcgen05_config.autotune_seed_configs()

    def _fix_tcgen05_cluster_m2_search_config(self, config: dict[str, object]) -> None:
        self._cute_tcgen05_config._fix_cluster_m2_search_config(config)

    def allow_tcgen05_ab_stages_three_search(
        self,
        *,
        dtype_bytes: int,
        device: torch.device,
    ) -> None:
        self._cute_tcgen05_config.allow_ab_stages_three_search(
            dtype_bytes=dtype_bytes,
            device=device,
        )

    @staticmethod
    def _cute_per_cta_ab_smem_budget_bytes(device: torch.device) -> int:
        return CuteTcgen05Config.per_cta_ab_smem_budget_bytes(device)

    def _tcgen05_ab_stages_three_fits(
        self,
        *,
        bm: int,
        bn: int,
        bk: int,
        cluster_m: int,
    ) -> bool:
        return self._cute_tcgen05_config.ab_stages_three_fits(
            bm=bm,
            bn=bn,
            bk=bk,
            cluster_m=cluster_m,
        )

    def _fix_tcgen05_ab_stages_three_search_config(
        self, config: dict[str, object]
    ) -> None:
        self._cute_tcgen05_config._fix_ab_stages_three_search_config(config)

    def _fix_tcgen05_with_scheduler_search_config(
        self, config: dict[str, object]
    ) -> None:
        self._cute_tcgen05_config._fix_with_scheduler_search_config(config)

    def _fix_tcgen05_cluster_m1_persistent_search_config(
        self, config: dict[str, object]
    ) -> None:
        self._cute_tcgen05_config._fix_cluster_m1_persistent_search_config(config)

    def restrict_tcgen05_num_epi_warps_search(self, choices: tuple[int, ...]) -> None:
        self._cute_tcgen05_config.restrict_num_epi_warps_search(choices)

    def restrict_tcgen05_num_epi_warps_validation(
        self, choices: tuple[int, ...]
    ) -> None:
        self._cute_tcgen05_config.restrict_num_epi_warps_validation(choices)

    def narrow_tcgen05_autotune_to_validated_configs(
        self,
        *,
        allow_persistent_pid_types: bool = False,
        allow_cluster_m2_search: bool = False,
        cluster_m2_static_k: int | None = None,
        allow_cluster_m2_edge_k_tail_family: bool = False,
        ab_stages_three_dtype_bytes: int | None = None,
        ab_stages_three_device: torch.device | None = None,
    ) -> None:
        self._cute_tcgen05_config.narrow_autotune_to_validated_configs(
            allow_persistent_pid_types=allow_persistent_pid_types,
            allow_cluster_m2_search=allow_cluster_m2_search,
            cluster_m2_static_k=cluster_m2_static_k,
            allow_cluster_m2_edge_k_tail_family=allow_cluster_m2_edge_k_tail_family,
            ab_stages_three_dtype_bytes=ab_stages_three_dtype_bytes,
            ab_stages_three_device=ab_stages_three_device,
        )

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
        return self._cute_tcgen05_config.optional_fragments(for_search=for_search)

    def _tcgen05_strategy_autotune_fragments(
        self,
    ) -> dict[str, ConfigSpecFragment]:
        return self._cute_tcgen05_config.strategy_autotune_fragments()

    def _tcgen05_strategy_validation_fragments(
        self,
    ) -> dict[str, ConfigSpecFragment]:
        return self._cute_tcgen05_config.strategy_validation_fragments()

    @staticmethod
    def _tcgen05_strategy_field_default(key: str, *, pid_type: object = None) -> object:
        return CuteTcgen05Config.strategy_field_default(key, pid_type=pid_type)

    def _validate_tcgen05_strategy_invariants_in_normalize(
        self,
        config: dict[str, object],
        *,
        _fix_invalid: bool,
    ) -> None:
        self._cute_tcgen05_config.validate_strategy_invariants(
            config,
            fix_invalid=_fix_invalid,
        )

    @staticmethod
    def _validate_optional_fragment_value(
        name: str, fragment: ConfigSpecFragment, value: object
    ) -> object:
        return CuteTcgen05Config._validate_optional_fragment_value(
            name,
            fragment,
            value,
        )

    def _clamp_tcgen05_l2_swizzle_size_to_shape(
        self, config: dict[str, object]
    ) -> None:
        self._cute_tcgen05_config._clamp_l2_swizzle_size_to_shape(config)

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
            ("cute_vector_widths", self.cute_vector_widths, True),
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

        # Clamp inner block sizes that are bounded by an outer block
        # (e.g. ``hl.tile(outer.begin, outer.end)``): at this point the
        # outer's concrete block size for this config is known, and the
        # inner extent can never exceed it.
        block_sizes_list = config.get("block_sizes")
        if isinstance(block_sizes_list, list):
            changed = False
            new_block_sizes = list(block_sizes_list)
            for i, spec in enumerate(self.block_sizes):
                bb = spec.bounded_by_block_id
                if (
                    bb is None
                    or i >= len(new_block_sizes)
                    or new_block_sizes[i] is None
                ):
                    continue
                try:
                    outer_index = self.block_sizes.block_id_to_index(bb)
                except KeyError:
                    continue
                outer_val = (
                    new_block_sizes[outer_index]
                    if outer_index < len(new_block_sizes)
                    else None
                )
                if (
                    isinstance(outer_val, int)
                    and isinstance(new_block_sizes[i], int)
                    and new_block_sizes[i] > outer_val
                ):
                    new_block_sizes[i] = outer_val
                    changed = True
            if changed:
                config["block_sizes"] = new_block_sizes
                num_threads = config.get("num_threads")
                if isinstance(num_threads, list):
                    new_num_threads = list(num_threads)
                    for i, (block_size, num_thread) in enumerate(
                        zip(new_block_sizes, new_num_threads, strict=False)
                    ):
                        if (
                            type(block_size) is not int
                            or type(num_thread) is not int
                            or num_thread <= 0
                        ):
                            continue
                        if num_thread > block_size:
                            num_thread = 1 << (max(block_size, 1).bit_length() - 1)
                        while num_thread > 1 and block_size % num_thread != 0:
                            num_thread //= 2
                        new_num_threads[i] = max(num_thread, 1)
                    config["num_threads"] = new_num_threads

        if self.supports_config_key("num_threads"):
            num_threads = cast("list[int]", config.get("num_threads", []))
            if all(value == 0 for value in num_threads):
                config.pop("num_threads", None)
        else:
            config.pop("num_threads", None)

        # Cap reduction loops at the backend's max loop chunk, while using the
        # live reduction thread threshold to decide when a persistent reduction
        # must be rolled.
        if self.max_reduction_threads is not None and self.reduction_loops:
            force_threshold = self.reduction_loop_force_threshold
            max_loop = self.max_reduction_loop
            reduction_loops = config.get("reduction_loops", [])
            if force_threshold is not None and isinstance(reduction_loops, list):
                new_loops = list(reduction_loops)
                changed = False
                for i, spec in enumerate(self.reduction_loops):
                    if i >= len(new_loops):
                        break
                    # Indexed reductions (argmin/argmax) on CuTe must keep
                    # the persistent thread count or rolled chunk within a
                    # single warp, since cute.arch.warp_reduction only
                    # supports threads_in_group<=32.
                    block_threshold = force_threshold
                    if (
                        self.backend_name == "cute"
                        and spec.block_id in self.cute_indexed_reduction_block_ids
                    ):
                        block_threshold = min(block_threshold, 32)
                    if new_loops[i] is None and spec.size_hint > block_threshold:
                        new_loops[i] = min(spec.size_hint, block_threshold)
                        changed = True
                    elif (
                        new_loops[i] is not None
                        and max_loop is not None
                        and (
                            new_loops[i] > max_loop
                            or (
                                self.backend_name == "cute"
                                and spec.block_id
                                in self.cute_indexed_reduction_block_ids
                                and new_loops[i] > 32
                            )
                        )
                    ):
                        new_loops[i] = min(
                            new_loops[i] if max_loop is None else max_loop,
                            block_threshold,
                        )
                        changed = True
                if changed:
                    config["reduction_loops"] = new_loops

        # CuTe-specific: persistent reduction whose thread count is shrunk
        # below the reduction extent by adjust_reduction_thread_count would
        # wrap the kernel body in a synthetic lane loop. The lane loop
        # carries the body-level reduction's accumulator across iterations,
        # so the reduction result would only reflect the last lane iter.
        # Force a looped reduction whenever the available reduction threads
        # (max_reduction_threads // product_of_non_reduction_thread_axes)
        # cannot cover the full reduction extent.
        if (
            self.backend_name == "cute"
            and self.max_reduction_threads is not None
            and self.reduction_loops
        ):
            nt_list = cast("list[int]", config.get("num_threads", []) or [])
            bs_list = cast("list[int]", config.get("block_sizes", []) or [])
            other_threads = 1
            for i, _ in enumerate(self.num_threads):
                nt = nt_list[i] if i < len(nt_list) else 0
                if not isinstance(nt, int) or nt <= 0:
                    bs = bs_list[i] if i < len(bs_list) else 1
                    nt = bs if isinstance(bs, int) and bs > 1 else 1
                if nt > 1:
                    other_threads *= nt
            available = max(1, self.max_reduction_threads // other_threads)
            reduction_loops = config.get("reduction_loops", [])
            if isinstance(reduction_loops, list):
                new_loops = list(reduction_loops)
                changed = False
                for i, spec in enumerate(self.reduction_loops):
                    if i >= len(new_loops):
                        break
                    if new_loops[i] is None and spec.size_hint > available:
                        # When other_threads consumes the entire CuTe 1024 thread
                        # budget there is no thread budget left for the reduction
                        # axis. A chunk of 1 is invalid (LoopedReductionStrategy
                        # requires block_size > 1) and a persistent reduction
                        # would hit the synthetic-lane-loop bug described above.
                        # Reject the config so the autotuner skips it.
                        if available < 2:
                            raise InvalidConfig(
                                f"cute backend: reduction axis {i} has no thread "
                                f"budget left (non-reduction axes use "
                                f"{other_threads} of {self.max_reduction_threads} "
                                f"threads)."
                            )
                        chunk = min(spec.size_hint, available)
                        if self.max_reduction_loop is not None:
                            chunk = min(chunk, self.max_reduction_loop)
                        new_loops[i] = chunk
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
            "cute_vector_widths",
            "range_unroll_factors",
            "range_warp_specializes",
            "range_num_stages",
            "range_multi_buffers",
            "range_flattens",
            "static_ranges",
            "load_eviction_policies",
            "load_cache_modifiers",
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
            "load_cache_modifiers",
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
        if (
            self.supports_config_key("load_cache_modifiers")
            and self.load_cache_modifiers.length > 0
        ):
            config.setdefault(
                "load_cache_modifiers", self.load_cache_modifiers.default()
            )
        if self.supports_config_key("indexing"):
            config.setdefault("indexing", self.indexing.default())
        if self.supports_config_key("atomic_indexing"):
            config.setdefault("atomic_indexing", self.atomic_indexing.default())
        for key, fragment in self.backend_tunable_fragments.items():
            config.setdefault(key, fragment.default())
        if self.backend_name == "cute":
            self._cute_tcgen05_config.normalize_pre_pid_type(
                config,
                fix_invalid=_fix_invalid,
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

        if _fix_invalid and self.backend_name == "cute":
            self._cute_tcgen05_config.fix_search_config(config)

        if self.backend_name == "cute":
            self._cute_tcgen05_config.normalize_strategy(
                config,
                fix_invalid=_fix_invalid,
            )

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
            preserve_keys = self._cute_tcgen05_config.implicit_default_keys_to_preserve(
                config
            )
            for key in _CUTE_IMPLICIT_DEFAULT_KEYS - provided_keys - preserve_keys:
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

    def flatten_missing_field_default(
        self,
        key: str,
        config: dict[str, object],
    ) -> tuple[bool, object]:
        if self.backend_name == "cute":
            return self._cute_tcgen05_config.flatten_missing_field_default(key, config)
        return False, None

    def prepare_override_normalization(
        self,
        config: dict[str, object],
        overrides: Mapping[str, object],
    ) -> None:
        if self.backend_name == "cute":
            self._cute_tcgen05_config.prepare_override_normalization(
                config,
                overrides,
            )

    def _base_default_config(self) -> helion.Config:
        config = self.flat_config(lambda x: x.default())
        self._shrink_for_numel_constraints(config)
        return config

    def default_config(self) -> helion.Config:
        if self.compiler_default_config is None:
            return self._base_default_config()
        config = helion.Config.from_dict(self.compiler_default_config.config)
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
                fields.update(self._cute_tcgen05_config.flat_fields())
            elif self.supports_config_key("num_threads"):
                fields["num_threads"] = self.num_threads
                # Universal pid emission honors ``loop_orders`` (the
                # launch grid swaps which tile axis is outer), and the
                # better order is shape-dependent. Expose only on the
                # non-tcgen05 branch — the tcgen05 persistent
                # scheduler relies on a fixed
                # ``pid_info[0]=M, pid_info[1]=N`` mapping for
                # ``cluster_m`` / virtual-PID logic, so sampling
                # ``loop_orders=[[1, 0]]`` there would steer cluster
                # logic onto the wrong axis. Measured evidence for
                # the non-tcgen05 widening lives in ``cute_plan.md``
                # §7.0 "Recent landed work".
                if (
                    self.supports_config_key("loop_orders")
                    and len(self.loop_orders) > 0
                ):
                    fields["loop_orders"] = self.loop_orders
                # Expose ``cute_vector_widths`` per-block so the
                # autotuner can vary V in {1, 2, 4, 8} for lane-loop
                # vec loads (and for ``LoopedReductionStrategy`` rolled
                # reductions).  Without this entry, ``flatten`` strips V
                # back to the default of 1, defeating the seed
                # heuristics that try to bias toward LDG.128 lattices.
                if (
                    self.supports_config_key("cute_vector_widths")
                    and len(self.cute_vector_widths) > 0
                ):
                    fields["cute_vector_widths"] = self.cute_vector_widths
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
        if (
            self.supports_config_key("load_cache_modifiers")
            and self.load_cache_modifiers.length > 0
        ):
            fields["load_cache_modifiers"] = self.load_cache_modifiers
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
    ) -> tuple[tuple[object, ...], ...]:
        """Return a hashable structural description of this ConfigSpec's search space.

        Captures field names, sequence lengths, per-item block_ids lengths
        (for PermutationFragment), ListOf inner lengths + per-slot mem-op id signatures, and
        optional ACF slot presence.  Two ConfigSpecs with the same fingerprint can safely exchange
        FlatConfig values.
        """
        result: list[tuple[object, ...]] = [
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
            "load_cache_modifiers",
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
        bounded_by_block_id: int | None = None,
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
        # Outer block_id whose tile extent caps this block's size in normalize().
        self.bounded_by_block_id: int | None = bounded_by_block_id
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
            ("bounded_by_block_id", None),
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
        # Cap default at the backend's max reduction loop so that
        # large reductions default to looped rather than persistent.
        if base.max_reduction_loop is not None:
            force_threshold = base.reduction_loop_force_threshold
            if force_threshold is not None and self.size_hint > force_threshold:
                default = min(default, base.max_reduction_loop)
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
        # Encode None ("persistent reduction") so the inverse ``_flat_config``
        # decodes it back to None. ``_flat_config`` returns None for any value
        # >= size_hint, so the encoding must also be >= size_hint: use the
        # fragment's ``high`` (always >= size_hint). The fragment *default* is
        # capped at max_reduction_loop and can fall below size_hint, which would
        # round-trip None into a slow looped config (e.g. size_hint=32000 ->
        # default 4096 -> reduction_loops=[4096]).
        if value is None:
            return self._flat_fragment(base).high
        return value

    def _normalize(self, name: str, value: object) -> int | None:
        if value is None:
            return None
        normalized = super()._normalize(name, value)
        # A looped reduction whose chunk equals or exceeds the reduction
        # extent has only one iteration — it is semantically identical to a
        # persistent reduction, but the looped codegen path occasionally
        # produces subtly different results on the CuTe backend (e.g. when a
        # multi-pass kernel like layer_norm reuses the loaded inputs across
        # two reductions).  Collapsing to ``None`` here matches the
        # ``_flat_config`` behaviour and keeps the persistent/loop choice in
        # sync regardless of how the value was generated.
        if isinstance(normalized, int) and normalized >= self.size_hint:
            return None
        return normalized

    def _fill_missing(self) -> None:
        return None


_CUTE_VECTOR_WIDTH_CHOICES: tuple[int, ...] = (1, 2, 4, 8)


class CuteVectorWidthSpec(_BlockIdItem):
    """Per-reduction-block vector load width for the CuTe backend.

    V=1 disables vectorization (scalar loads). V=2/4/8 emits
    ``cute.arch.load(..., ir.VectorType.get([V], elem_dtype.mlir_type))``
    for the inner reduction load, lowering to LDG.64/LDG.128.
    """

    def __init__(
        self,
        *,
        block_id: int,
        size_hint: int,
    ) -> None:
        super().__init__([block_id])
        self.size_hint = size_hint

    def _fragment(self, base: ConfigSpec) -> EnumFragment:
        return EnumFragment(choices=_CUTE_VECTOR_WIDTH_CHOICES)

    def _normalize(self, name: str, value: object) -> int:
        if not isinstance(value, int):
            raise InvalidConfig(f"{name} must be an integer, got {value!r}")
        if value not in _CUTE_VECTOR_WIDTH_CHOICES:
            raise InvalidConfig(
                f"{name} must be one of {_CUTE_VECTOR_WIDTH_CHOICES}, got {value!r}"
            )
        return value

    def _fill_missing(self) -> int:
        return 1


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
